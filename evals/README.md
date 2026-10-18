# evals/

Two files of cases and (soon) a runner.

- `hull_examples.json` — worked examples transcribed by hand from Hull 9e, with chapter, example number and printed page. Each is also a pytest in `tests/test_engine.py`. **Numbers are typed from the page, never scraped**: the PDF extractor garbles math symbols, and typing them is how you learn them.
- `retrieval.json` — queries with the section that *should* come back. The first case is one the current BM25 gets wrong (gamma near expiry should retrieve Hull §19.6, and it ranks 10.4 and 19.7 above it). That is the point: an eval you can fail.
- `run.py` (step 4) — executes the agent against every case and prints one table: tool choice, numeric faithfulness, refusal correctness, citation grounding, cost, latency.

## What an eval is

A test where the system under test is non-deterministic, so you measure a rate instead of asserting equality. Everything else is the same discipline as pytest: golden cases, a runner, a score per axis, run on every change, delta in the PR.

- Engine → ordinary tests (deterministic): the book is ground truth.
- Retrieval → precision@k against `retrieval.json`: the section you'd open is ground truth.
- Agent → numeric faithfulness (every number in the answer traces to an engine call), tool choice, refusals on the adversarial set, and an LLM judge for grounding that you calibrate on 30 hand-graded cases before trusting.
