"""The `search` tool: retrieval over the private corpus index, with citations.

Retrieval returns TEXT for the model to read and cite. It never returns numbers for
the model to compute with; the engine does arithmetic. That boundary is the whole
design, and the eval harness checks it (numeric faithfulness: every number in an
answer must trace to an engine result, not to a passage).

Scoring is BM25 over the chunk text with a light boost when the query names the
chapter or section. No embeddings yet: BM25 over book-section chunks is strong for
textbook retrieval, has no dependencies, and is easy to reason about in an eval.
Swap in embeddings later only if the retrieval eval says you need them.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

INDEX = Path(__file__).resolve().parents[2] / "corpus" / "index"
TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-]*")
STOP = set("the a an of to in and or is are be for on with as by at from that this it its which what how why when where does do".split())


@dataclass(frozen=True)
class Passage:
    id: str
    book: str
    chapter: str
    section: str
    pages: tuple
    score: float
    text: str
    cite_pdf_page: bool = False

    def citation(self) -> str:
        where = self.section or self.chapter
        lo, hi = self.pages
        unit = "PDF p." if self.cite_pdf_page else "p."
        pp = f"{unit} {lo}" if lo == hi else f"{unit.rstrip('.')}p. {lo}-{hi}" if not self.cite_pdf_page else f"PDF pp. {lo}-{hi}"
        return f"{self.book}, {where}, {pp}"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class Corpus:
    def __init__(self, index_dir: Path = INDEX, books: list[str] | None = None):
        self.chunks: list[dict] = []
        files = sorted(index_dir.glob("*.jsonl")) if index_dir.exists() else []
        for f in files:
            if books and f.stem not in books:
                continue
            with f.open() as fh:
                self.chunks.extend(json.loads(line) for line in fh)
        self._build()

    def _build(self) -> None:
        self.tokens = [tokenize(c["text"]) for c in self.chunks]
        self.df: Counter = Counter()
        for toks in self.tokens:
            self.df.update(set(toks))
        self.n = len(self.chunks)
        self.avgdl = (sum(len(t) for t in self.tokens) / self.n) if self.n else 0.0

    @property
    def available(self) -> bool:
        return self.n > 0

    def search(self, query: str, k: int = 5, book: str | None = None) -> list[Passage]:
        """BM25 (k1=1.5, b=0.75) with a section/chapter title boost."""
        q = tokenize(query)
        if not q or not self.n:
            return []
        k1, b = 1.5, 0.75
        scored = []
        for i, (chunk, toks) in enumerate(zip(self.chunks, self.tokens)):
            if book and chunk["book"] != book:
                continue
            tf = Counter(toks)
            dl = len(toks) or 1
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * dl / self.avgdl))
            title = f"{chunk['chapter']} {chunk['section']}".lower()
            s += 0.5 * sum(1 for term in q if term in title)
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        out = []
        for s, i in scored[:k]:
            c = self.chunks[i]
            out.append(Passage(c["id"], c["book"], c["chapter"], c["section"], tuple(c["pages"]), round(s, 3), c["text"], bool(c.get("cite_pdf_page"))))
        return out


# ----------------------------------------------------------------------------- tool contract

SEARCH_TOOL = {
    "name": "search",
    "description": ("Search the reference books (Hull 9e on derivatives; Graham, Security Analysis 6e; "
                    "Lynch, One Up on Wall Street; Damodaran, Investment Valuation 2e) and return passages with citations. Use it to explain "
                    "or justify a claim. Do NOT read numbers out of passages to do arithmetic; call the "
                    "engine tools for every number."),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "what to look up, in the book's own vocabulary"},
            "book": {"type": "string", "enum": ["hull-9e", "graham-security-analysis-6e", "lynch-one-up", "damodaran-investment-valuation"],
                     "description": "restrict to one book (optional)"},
            "k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
        },
        "required": ["query"],
    },
}


def run_search(corpus: Corpus, query: str, book: str | None = None, k: int = 4) -> dict:
    if not corpus.available:
        return {"passages": [], "note": "corpus index not present on this host; build it with tools/build_corpus.py"}
    hits = corpus.search(query, k=k, book=book)
    return {"passages": [dict(asdict(p), citation=p.citation(), text=p.text[:1400]) for p in hits]}
