#!/usr/bin/env python3
"""Index the private corpus: chunk each PDF by its own table of contents, keep page numbers.

Input:  corpus/*.pdf                (gitignored; the books you own)
Output: corpus/index/<book>.jsonl    (gitignored; one chunk per line)
        corpus/index/manifest.json   (book -> chapters, pages, chunk counts)

A chunk is one TOC section (e.g. Hull "15.8 The Black-Scholes-Merton pricing formulas"),
split further at ~1,800 characters on paragraph boundaries so a retrieved passage is
self-contained. Every chunk carries book, chapter, section, printed-page range and
PDF-page range, so the agent can cite "Hull 9e, §15.8, p. 338" and you can open the page.

Math comes out of the extractor garbled (ligatures, '¼' for '='). That is fine for
retrieval and citation and useless for numbers, which is why the golden set is typed
by hand from the page. See evals/hull_examples.json.

Usage:  python3 tools/build_corpus.py            # all books in corpus/
        python3 tools/build_corpus.py --book hull-9e
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:  # pragma: no cover
    print("pip install pymupdf", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
INDEX = CORPUS / "index"
CHUNK_CHARS = 1800

# Printed page = PDF page - offset. Set per book by reading one page; Hull's ch.15
# opener is PDF 344 and printed 321, so offset 23. Update when you add a book.
PAGE_OFFSET = {"hull-9e": 23, "graham-security-analysis-6e": 52, "lynch-one-up": 0, "damodaran-investment-valuation": 0}
# Books whose page numbers restart every chapter (Damodaran's manuscript PDF): cite by chapter + PDF page.
CITE_BY_PDF_PAGE = {"lynch-one-up", "damodaran-investment-valuation"}

SKIP_TITLES = re.compile(r"^(cover|title page|copyright|acknowledg|contents|list of|index|preface|dedication|description|untitled|about the author)", re.I)


def clean(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("­", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sections(doc, book: str) -> list[dict]:
    """Flatten the TOC into [(chapter, section, start_pdf_page, end_pdf_page)], deepest level that exists.
    A PDF with no TOC may carry a sidecar `<book>.chapters.json` ([{chapter,title,pdf_start}]) built by hand
    or by tools/map_chapters.py; otherwise fall back to 8-page windows."""
    toc = doc.get_toc()
    sidecar = CORPUS / f"{book}.chapters.json"
    if not toc and sidecar.exists():
        chapters = json.loads(sidecar.read_text())
        out = []
        for i, ch in enumerate(chapters):
            start = ch["pdf_start"]
            end = (chapters[i + 1]["pdf_start"] - 1) if i + 1 < len(chapters) else len(doc)
            title = f"Chapter {ch['chapter']}: {ch['title']}"
            # split long chapters into ~12-page sections so a citation lands within a few pages
            for s0 in range(start, end + 1, 12):
                out.append({"chapter": title, "section": "", "start": s0, "end": min(s0 + 11, end)})
        return out
    if not toc:
        # No TOC: one section per 8 pages
        return [{"chapter": f"pages {p+1}-{min(p+8, len(doc))}", "section": "", "start": p + 1, "end": min(p + 8, len(doc))}
                for p in range(0, len(doc), 8)]
    out, chapter = [], ""
    for i, (lvl, title, page) in enumerate(toc):
        title = title.strip()
        if lvl == 1:
            chapter = title
        if SKIP_TITLES.match(title):
            continue
        nxt = next((p for l, t, p in toc[i + 1:] if p > page), len(doc) + 1)
        out.append({"chapter": chapter, "section": title if lvl > 1 else "", "start": page, "end": max(page, nxt - 1)})
    return out


def split_paragraphs(text: str, limit: int) -> list[str]:
    paras, chunks, cur = text.split("\n\n"), [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > limit and cur:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def build(book: str) -> dict:
    pdf = CORPUS / f"{book}.pdf"
    doc = pymupdf.open(pdf)
    offset = PAGE_OFFSET.get(book, 0)
    INDEX.mkdir(parents=True, exist_ok=True)
    out_path = INDEX / f"{book}.jsonl"
    n_chunks, chapters = 0, {}
    with out_path.open("w") as out:
        for sec in sections(doc, book):
            text = clean("\n".join(doc[p - 1].get_text() for p in range(sec["start"], min(sec["end"], len(doc)) + 1)))
            if len(text) < 200:
                continue
            for j, chunk in enumerate(split_paragraphs(text, CHUNK_CHARS)):
                rec = {
                    "id": f"{book}:{sec['start']}:{j}",
                    "book": book,
                    "chapter": sec["chapter"],
                    "section": sec["section"],
                    "pdf_pages": [sec["start"], sec["end"]],
                    "pages": [sec["start"] - offset, sec["end"] - offset],
                    "cite_pdf_page": book in CITE_BY_PDF_PAGE,
                    "text": chunk,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1
            chapters[sec["chapter"]] = chapters.get(sec["chapter"], 0) + 1
    return {"book": book, "pdf_pages": len(doc), "chunks": n_chunks, "sections": sum(chapters.values()), "chapters": len(chapters)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", help="one book stem (file name without .pdf)")
    args = ap.parse_args(argv)
    books = [args.book] if args.book else sorted(p.stem for p in CORPUS.glob("*.pdf"))
    if not books:
        print("no PDFs in corpus/; drop the books you own there (it is gitignored)", file=sys.stderr)
        return 1
    manifest = {}
    for book in books:
        info = build(book)
        manifest[book] = info
        print(f"{book}: {info['chunks']} chunks from {info['sections']} sections in {info['chapters']} chapters ({info['pdf_pages']} pages)")
    (INDEX / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
