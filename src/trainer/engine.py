"""The pricing engine. Deterministic, dependency-free, validated against Hull (9th ed.).

Every number the trainer shows comes from here. The model never computes; it calls.

Conventions (Hull's): S spot, K strike, T years to expiry, r continuously compounded
risk-free rate, sigma annual volatility, q continuous dividend yield. Dividends paid as
discrete cash amounts are handled by `pv_dividends` + the escrowed-dividend adjustment
(Hull 15.12). American options go through the binomial tree (Hull ch. 13, 21).

Status: STUBBED. Each function raises NotImplementedError until you write it; the
tests in tests/test_engine.py are the specification and Hull's worked examples are
the answers. Fill them in order: norm_cdf, d1/d2, bs_price, greeks, implied_vol,
binomial. That order is also the order the Greeks make sense in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class EngineError(ValueError):
    """Raised for inputs the engine refuses: negative time, zero vol, unknown type."""


# ----------------------------------------------------------------------------- normals

def norm_cdf(x: float) -> float:
    """Standard normal CDF N(x). Use math.erf; no scipy here on purpose."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ----------------------------------------------------------------------------- inputs

def _check(S: float, K: float, T: float, r: float, sigma: float, kind: str) -> None:
    if kind not in ("call", "put"):
        raise EngineError(f"option type must be 'call' or 'put', got {kind!r}")
    if S <= 0 or K <= 0:
        raise EngineError("spot and strike must be positive")
    if T < 0:
        raise EngineError("time to expiry cannot be negative")
    if sigma <= 0:
        raise EngineError("volatility must be positive; the model has no price without one")


# ----------------------------------------------------------------------------- Black-Scholes-Merton

def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    """Hull (15.20)-(15.21) with a dividend yield q (Hull 17.4). T must be > 0."""
    raise NotImplementedError("d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T); d2 = d1 - sigma sqrt T")


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str, q: float = 0.0) -> float:
    """European price. Hull (15.20)/(15.21); with q, Hull (17.4)/(17.5).
    At T == 0 return intrinsic value (no NotImplemented path for expiry)."""
    _check(S, K, T, r, sigma, kind)
    raise NotImplementedError("call = S e^{-qT} N(d1) - K e^{-rT} N(d2); put = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)")


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float      # per 1.00 of vol (multiply by 0.01 for 'per vol point')
    theta: float     # per YEAR (divide by 365 for 'per calendar day', Hull's convention in ch. 19)
    rho: float       # per 1.00 of rate


def greeks(S: float, K: float, T: float, r: float, sigma: float, kind: str, q: float = 0.0) -> Greeks:
    """Analytic Greeks, Hull ch. 19 (Table 19.6 collects them, with q)."""
    _check(S, K, T, r, sigma, kind)
    raise NotImplementedError("delta, gamma, vega, theta, rho from Hull Table 19.6")


def pv_dividends(dividends: list[tuple[float, float]], r: float) -> float:
    """Present value of discrete cash dividends [(t_years, amount), ...]. Hull Example 15.9."""
    return sum(amount * math.exp(-r * t) for t, amount in dividends)


def bs_price_with_dividends(S: float, K: float, T: float, r: float, sigma: float, kind: str,
                            dividends: list[tuple[float, float]]) -> float:
    """European option on a stock paying known cash dividends before T: price on S - PV(div). Hull 15.12."""
    return bs_price(S - pv_dividends(dividends, r), K, T, r, sigma, kind)


# ----------------------------------------------------------------------------- implied vol

def implied_vol(price: float, S: float, K: float, T: float, r: float, kind: str, q: float = 0.0,
                tol: float = 1e-8, max_iter: int = 100) -> float:
    """Invert bs_price for sigma. Newton on vega with a bisection fallback (Hull 15.11).
    Raise EngineError if price is outside the no-arbitrage bounds (Hull ch. 11)."""
    raise NotImplementedError("Newton: sigma_{n+1} = sigma_n - (bs(sigma_n) - price) / vega(sigma_n); bracket [1e-4, 5]")


# ----------------------------------------------------------------------------- binomial

def binomial_price(S: float, K: float, T: float, r: float, sigma: float, kind: str,
                   steps: int = 500, american: bool = False, q: float = 0.0) -> float:
    """Cox-Ross-Rubinstein tree, Hull ch. 13 and 21.1: u = e^{sigma sqrt dt}, d = 1/u,
    p = (e^{(r-q) dt} - d) / (u - d). American: max(continuation, intrinsic) at each node.
    Must converge to bs_price for European as steps grows (test checks within 1 cent at 500)."""
    _check(S, K, T, r, sigma, kind)
    raise NotImplementedError("backward induction over a recombining tree")


# ----------------------------------------------------------------------------- parity

def put_call_parity_gap(call: float, put: float, S: float, K: float, T: float, r: float, q: float = 0.0) -> float:
    """c + K e^{-rT} - (p + S e^{-qT}); zero for European options with no arbitrage. Hull (11.6)/(17.3)."""
    return call + K * math.exp(-r * T) - (put + S * math.exp(-q * T))
