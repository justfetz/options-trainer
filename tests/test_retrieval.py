"""Retrieval eval, run as a test when the private index exists; skipped otherwise.

hit@3 against evals/retrieval.json. The threshold is deliberately below 1.0 because the
BM25 baseline has a known miss (the gamma question); raising the bar is the exercise.
"""
import json
from pathlib import Path

import pytest

from trainer.retrieval import Corpus, INDEX

CASES = json.loads((Path(__file__).parent.parent / "evals" / "retrieval.json").read_text())["cases"]


@pytest.fixture(scope="module")
def corpus():
    c = Corpus()
    if not c.available:
        pytest.skip("corpus index not built (private); run tools/build_corpus.py")
    return c


def test_retrieval_hit_at_3(corpus):
    scored = [c for c in CASES if c["expect_section_prefix"]]
    hits = 0
    misses = []
    for c in scored:
        res = corpus.search(c["query"], k=3, book=c["book"])
        ok = any((p.section or p.chapter).startswith(c["expect_section_prefix"]) for p in res)
        hits += ok
        if not ok:
            misses.append((c["query"], [p.section or p.chapter for p in res]))
    rate = hits / len(scored)
    print(f"\nretrieval hit@3 = {hits}/{len(scored)} = {rate:.0%}")
    for q, got in misses:
        print(f"  miss: {q!r} -> {got}")
    assert rate >= 0.6, f"retrieval hit@3 fell to {rate:.0%}"
