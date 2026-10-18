# corpus/ (private)

Drop the PDFs you own here, then `python3 tools/build_corpus.py`. The PDFs and the
generated `index/` are gitignored and never leave your machine (on Cloud Run the
index is read from a private bucket, not the image).

Expected file stems (used in citations and the `search` tool's `book` enum):

- `hull-9e.pdf` — Hull, *Options, Futures, and Other Derivatives*, 9th ed.
- `graham-security-analysis-6e.pdf` — Graham & Dodd, *Security Analysis*, 6th ed.
- `lynch-one-up.pdf` — Lynch, *One Up on Wall Street*
- `damodaran-investment-valuation.pdf` — Damodaran, *Investment Valuation*, 2nd ed. (manuscript PDF, no TOC;
  `damodaran-investment-valuation.chapters.json` maps its 35 chapters to PDF pages and is committed)

`PAGE_OFFSET` in `tools/build_corpus.py` maps PDF page to printed page per book.
