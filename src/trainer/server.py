"""Options trainer server. Same skeleton as the corrugator shift investigator:

  * loopback by default; with PUBLIC_DEMO_ORIGIN set, a hosted demo behind a TLS proxy
    that allows only listed browser origins to POST and never accepts visitor keys
  * the engine endpoints are free and unmetered (pure arithmetic, no model)
  * /api/ask (the agent) exists only once agent.py is written; until then it answers 501

Endpoints
  GET  /api/health                      {"ok": true, "corpus": <chunk count>}
  POST /api/price   {S,K,T,r,sigma,kind,q?}            -> {"price": ...}
  POST /api/greeks  {S,K,T,r,sigma,kind,q?}            -> {"delta":..,"gamma":..,"vega":..,"theta":..,"rho":..}
  POST /api/curve   {S_min,S_max,n,K,T,r,sigma,kind}   -> value + payoff + selected greek across spot (for the chart)
  POST /api/iv      {price,S,K,T,r,kind}               -> {"sigma": ...}
  POST /api/search  {query, book?, k?}                 -> passages with citations (corpus must be present)
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

from . import engine
from .retrieval import Corpus, run_search

WEB = Path(__file__).with_name("web")


@dataclass(frozen=True)
class PublicDemo:
    origins: tuple
    embed_origins: tuple

    @staticmethod
    def _origins(raw):
        out = []
        for item in (raw or "").split(","):
            item = item.strip().rstrip("/")
            if not item:
                continue
            local = item.startswith(("http://127.0.0.1", "http://localhost"))
            _, _, hostport = item.partition("://")
            if not (item.startswith("https://") or local) or not hostport or any(c in hostport for c in " /?#@"):
                raise ValueError(f"Origin must be an https scheme+host (or a loopback http origin): {item!r}")
            out.append(item)
        return tuple(out)

    @classmethod
    def from_env(cls, environ=None):
        environ = os.environ if environ is None else environ
        origins = cls._origins(environ.get("PUBLIC_DEMO_ORIGIN"))
        if not origins:
            return None
        return cls(origins=origins, embed_origins=cls._origins(environ.get("PUBLIC_DEMO_EMBED_ORIGINS")))

    @property
    def frame_ancestors(self):
        return " ".join(self.embed_origins) if self.embed_origins else "'none'"


class App:
    def __init__(self, public=None):
        self.public = public
        self.requests = deque()
        self.corpus = Corpus()

    # ---- request handlers (pure functions of a validated payload) ----
    @staticmethod
    def _num(payload, key, lo=None, hi=None, default=None):
        v = payload.get(key, default)
        if v is None:
            raise ValueError(f"missing {key}")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"{key} must be a number")
        if lo is not None and v < lo or hi is not None and v > hi:
            raise ValueError(f"{key} out of range")
        return float(v)

    def _common(self, p):
        kind = p.get("kind")
        if kind not in ("call", "put"):
            raise ValueError("kind must be 'call' or 'put'")
        return dict(S=self._num(p, "S", 1e-6, 1e7), K=self._num(p, "K", 1e-6, 1e7), T=self._num(p, "T", 0, 50),
                    r=self._num(p, "r", -0.1, 1.0), sigma=self._num(p, "sigma", 1e-6, 5.0), kind=kind,
                    q=self._num(p, "q", -0.1, 1.0, 0.0))

    def price(self, p):
        a = self._common(p)
        return {"price": engine.bs_price(**a)}

    def greeks(self, p):
        a = self._common(p)
        g = engine.greeks(**a)
        return {"delta": g.delta, "gamma": g.gamma, "vega": g.vega, "theta": g.theta, "rho": g.rho,
                "theta_per_day": g.theta / 365.0, "vega_per_point": g.vega / 100.0}

    def curve(self, p):
        a = self._common(p)
        n = int(self._num(p, "n", 2, 400, 121))
        s_min = self._num(p, "S_min", 1e-6, 1e7, a["S"] * 0.5)
        s_max = self._num(p, "S_max", 1e-6, 1e7, a["S"] * 1.5)
        greek = p.get("greek", "delta")
        if greek not in ("delta", "gamma", "vega", "theta", "rho"):
            raise ValueError("greek must be one of delta, gamma, vega, theta, rho")
        xs = [s_min + (s_max - s_min) * i / (n - 1) for i in range(n)]
        rows = []
        for s in xs:
            args = dict(a, S=s)
            value = engine.bs_price(**args)
            payoff = max(s - a["K"], 0.0) if a["kind"] == "call" else max(a["K"] - s, 0.0)
            gval = getattr(engine.greeks(**args), greek) if a["T"] > 0 else None
            rows.append({"S": s, "value": value, "payoff": payoff, greek: gval})
        return {"greek": greek, "rows": rows}

    def iv(self, p):
        kind = p.get("kind")
        if kind not in ("call", "put"):
            raise ValueError("kind must be 'call' or 'put'")
        sigma = engine.implied_vol(self._num(p, "price", 0, 1e7), self._num(p, "S", 1e-6, 1e7), self._num(p, "K", 1e-6, 1e7),
                                   self._num(p, "T", 1e-9, 50), self._num(p, "r", -0.1, 1.0), kind, self._num(p, "q", -0.1, 1.0, 0.0))
        return {"sigma": sigma}

    def search(self, p):
        q = p.get("query")
        if not isinstance(q, str) or not q.strip() or len(q) > 500:
            raise ValueError("query must be 1-500 characters")
        book = p.get("book")
        if book is not None and book not in {"hull-9e", "graham-security-analysis-6e", "lynch-one-up", "damodaran-investment-valuation"}:
            raise ValueError("unknown book")
        k = int(self._num(p, "k", 1, 8, 4))
        return run_search(self.corpus, q, book, k)

    ROUTES = {"/api/price": "price", "/api/greeks": "greeks", "/api/curve": "curve", "/api/iv": "iv", "/api/search": "search"}


class Handler(BaseHTTPRequestHandler):
    server_version = "options-trainer/0.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        pub = self.server.app.public
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                         "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
                         f"frame-ancestors {pub.frame_ancestors if pub else chr(39)+'none'+chr(39)}")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/health":
            return self._send(200, {"ok": True, "corpus": self.server.app.corpus.n})
        path = "/index.html" if self.path in ("/", "") else self.path.split("?", 1)[0]
        f = (WEB / path.lstrip("/")).resolve()
        if WEB.resolve() not in f.parents or not f.is_file():
            return self._send(404, {"error": "not found"})
        ctype = {"html": "text/html; charset=utf-8", "js": "application/javascript", "css": "text/css", "json": "application/json"}.get(f.suffix[1:], "application/octet-stream")
        return self._send(200, f.read_bytes(), ctype)

    def do_POST(self):
        app = self.server.app
        if self.path not in App.ROUTES:
            return self._send(404, {"error": "not found"})
        if app.public is not None:
            origin = (self.headers.get("Origin") or "").rstrip("/")
            if origin not in app.public.origins:
                return self._send(403, {"error": "Origin not allowed"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self._send(415, {"error": "JSON required"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 < length <= 8000:
            return self._send(413, {"error": "Request too large or empty"})
        now = time.monotonic()
        while app.requests and app.requests[0] < now - 60:
            app.requests.popleft()
        if len(app.requests) >= 240:
            return self._send(429, {"error": "Request limit reached; retry in a minute"})
        app.requests.append(now)
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON object required")
            body = getattr(app, App.ROUTES[self.path])(payload)
        except (ValueError, engine.EngineError) as e:
            return self._send(400, {"error": str(e)})
        except NotImplementedError as e:
            return self._send(501, {"error": f"engine not implemented yet: {e}"})
        return self._send(200, body)


def main():
    public = PublicDemo.from_env()
    app = App(public)
    host = "0.0.0.0" if public else "127.0.0.1"
    port = int(os.environ.get("PORT", "7861"))
    server = (ThreadingHTTPServer if public else HTTPServer)((host, port), Handler)
    server.app = app
    print(f"options-trainer on http://{host}:{port} ({'public demo' if public else 'loopback'}; corpus chunks: {app.corpus.n})")
    server.serve_forever()


if __name__ == "__main__":
    main()
