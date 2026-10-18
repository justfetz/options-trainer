# options-trainer

A teaching tool for options math, and a course in building evaluated agents, built on the
books I own. Lives at `options.felixaugust.com` once deployed; part of [Felix/August](https://www.felixaugust.com).

The design rule is the same one as my plant-floor agents: **the model proposes, the engine
decides.** Every number on screen comes from a tested pricing engine. Every explanation is
retrieved from the book and cited to a page. The model routes, explains, and drills; it never
does arithmetic.

## The layers, in the order they get built

| step | what | ground truth for its tests | status |
|---|---|---|---|
| 1 | **Engine** — Black-Scholes-Merton, Greeks, implied vol, CRR binomial, parity | Hull's worked examples, typed from the page | stubbed; 17 tests failing on purpose |
| 2 | **Workbench** — sliders, value vs payoff, one Greek across strikes, a time scrubber | the engine | not started |
| 3 | **Retrieval** — `search` over Hull, Graham, Lynch with chapter + page citations | `evals/retrieval.json`, hit@3 | BM25 baseline, 5/8 |
| 4 | **Agent + harness** — bounded tool loop; golden set; runner; one score table | engine (numbers), judge (grounding), adversarial set (refusals) | not started |
| 5 | **Drills** — generate a problem, grade against the engine, track what you miss | the engine | not started |
| 6 | **The smile** — a real chain, implied vols backed out, the model's flat line vs the market | a snapshotted chain | not started |

## Run

```bash
pip install -e ".[dev]"
pytest -q                     # step 1: watch the engine tests fail, then make them pass
python3 tools/build_corpus.py # after dropping your PDFs in corpus/ (private, gitignored)
python3 -m trainer.server     # http://127.0.0.1:7861 ; /api/health, /api/price, /api/search ...
```

## Step 1 is homework, by design

`src/trainer/engine.py` has the function signatures, the docstrings name the Hull equations,
and `tests/test_engine.py` is the specification. Fill them in this order, which is also the
order the ideas depend on each other:

1. `d1_d2` → `bs_price` (Hull 15.8; Example 15.6 is the test: call 4.76, put 0.81)
2. `greeks` (Hull ch. 19, Table 19.6). The tests include the one people get wrong:
   a deep in-the-money European put has **positive** theta (Hull 19.5).
3. `implied_vol` (Hull 15.11): Newton on vega with a bisection bracket; refuse prices outside the
   no-arbitrage bounds of ch. 11.
4. `binomial_price` (Hull ch. 13, 21.1): CRR, European must converge to BSM; American put ≥ European;
   American call on a non-dividend stock = European (Hull 11.5).

When you can't explain a line you wrote, stop and read the section it cites. That is the course.

## The corpus is private

`corpus/` holds the PDFs you own and the index built from them. Both are gitignored. On Cloud
Run the index is mounted from a private bucket, not baked into the image. The repo publishes the
engine, the tests, the citations, and your notes; never the text.

## What an eval is (short version)

A test whose subject is non-deterministic, so you measure a rate instead of asserting equality.
See `evals/README.md`. The engine gets ordinary tests because it is deterministic; retrieval and
the agent get rates. The first retrieval eval already fails three of eight cases, which is what a
useful eval looks like on day one.

## Deploy

`.github/workflows/deploy-cloudrun.yml` is the shift investigator's workflow with the service
renamed: builds from source, one instance, scale to zero, origin allow-list, health smoke test.
Set the `GCP_PROJECT_ID` variable and `GCP_SA_KEY` secret to turn it on; `CORPUS_BUCKET` mounts the
index. Nothing deploys until step 2 exists; there's nothing to look at yet.
