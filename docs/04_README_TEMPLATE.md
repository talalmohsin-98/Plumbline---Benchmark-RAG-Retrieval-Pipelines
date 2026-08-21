# Plumbline

**Upload your documents. Race six retrieval pipelines. Get real numbers.**

Live: [craftwyre.com](https://craftwyre.com) · Dataset: [HF] · Fine-tuned reranker: [HF]

> Everything below is measured on a **human-adjudicated gold set** — 115 questions, every one of them decided by the author — and is reproducible from this repository. The questions were drafted and screened by LLMs; the author then ruled on all 115, disagreeing with the screener on 21% of them. [The evaluation set](#the-evaluation-set) says exactly what that means and what it does not. Commands are in [Reproducing these numbers](#reproducing-these-numbers).

---

## Why this exists

"Add a reranker" and "use hybrid search" are repeated constantly and quantified rarely. Existing evaluation tools score one pipeline after you've built it. Plumbline compares six candidate pipelines against each other on your own corpus, before you commit to one.

## Results

Demo corpus: `{N}` documents, `{N}` chunks. Evaluation set: `{N}` LLM-screened questions audited to `{N}%` human agreement, held-out test split `n={N}`.

| Lane | recall@5 | recall@10 | MRR | Grounded | p95 latency | Cost/query |
|---|---|---|---|---|---|---|
| BM25 only | | | | | | $0.000 |
| Dense only | | | | | | $0.000 |
| Hybrid + RRF | | | | | | $0.000 |
| Hybrid + RRF + reranker | | | | | | $0.000 |
| Hybrid + RRF + reranker + HyDE | | | | | | |
| **Hybrid + RRF + fine-tuned reranker** | | | | | | $0.000 |

**Headline finding:** _{one sentence — e.g. "Fine-tuning the cross-encoder on 420 in-domain pairs lifted MRR from X to Y while leaving recall@10 unchanged, i.e. it improved ranking rather than retrieval."}_

**What didn't work:** _{state at least one thing honestly — a lane that underperformed, a parameter that didn't matter}_

## What the metrics mean

- **recall@k** — fraction of questions where a correct chunk appeared in the top k. The primary metric: if the right chunk isn't retrieved, generation cannot recover.
- **MRR** — mean of `1/(rank of first correct chunk)`. Measures ranking rather than retrieval.
- **Groundedness rate** — fraction of generated answers where every sentence is supported by the retrieved context, scored by an LLM judge. **Judge agreement with human labels on a 30-sample audit: `{N}%`.**
- **p95 latency** — 95th percentile server-side wall clock.
- **Cost per query** — from actual token counts. Retrieval-only lanes cost nothing; only HyDE makes an LLM call.

`nDCG` is deliberately excluded — with binary relevance labels it largely restates MRR.

## The evaluation set

**LLM-screened, human-audited.** Not hand-verified row by row — the distinction matters, so here is exactly what was done.

`{N}` questions were drafted by `openai/gpt-oss-120b`, one per randomly sampled chunk. Each draft was then scored against four drop rules, **one call per rule** so that a rejection says *which* rule fired rather than just "bad":

| Rule | Verdict | Scored by | Fired |
|---|---|---|---|
| Answerable just as well by 3+ other chunks | drop | model | `{N}` |
| Reuses more than four consecutive words from the chunk | fix — rephrased, then re-checked | exact string match | `{N}` |
| The chunk does not state the answer | drop | model | `{N}` |
| Answering requires combining two chunks | drop | model | `{N}` |

`{N}%` of drafts were dropped. The first `{N}` survivors became the gold set.

Rule 2 is checked by a longest-common-run scan rather than by the model: "more than four consecutive words" is an exact property of two strings, and asking an LLM to count words would make it the one number here that a rerun could not reproduce.

Duplicated answers are **multi-labelled, not dropped**. Documentation repeats itself — five FastAPI chunks each say to install `python-multipart`, and the LangSmith deployment sentence is byte-identical across two pages — so a single label would score a lane wrong for finding an equally correct chunk. 21 of the `{N}` questions carry more than one gold chunk, adjudicated by hand and recorded with reasons in `data/multilabel.json`.

### The adjudication — and its limits

A screener nobody checks is circular: it would measure how well retrieval finds chunks that a language model thought were findable. So every draft was screened **twice, independently**, and then **the author personally decided every row in the gold set** — 137 decisions over 139 screened rows, verdicts hidden until after the keypress. The 73 rows the two screeners had agreed to keep were judged in a **fully blind** pass: no screener verdict shown at any point, and no running agreement rate on screen.

**The screener agreed with the author 78% of the time overall — and only 58% on the rows judged with its verdict revealed, which are the disputes.** It is not usable unsupervised.

| | agreement with the author |
|---|---|
| Screener (`gpt-oss-20b`), all 137 rows | **78%** (107/137) |
| — on 64 rows judged with the verdict revealed | 58% (37/64) |
| — on 73 agreed keeps (blind) | 96% (70/73) |
| — on rows the screener **dropped**, first pass | **0%** (0/5) |

The 0% on drops is the important one. Every row the screener discarded was a question the author would have kept — so an unsupervised pipeline would have quietly binned good questions, and nothing in the finished gold set could have revealed it. The audit found this because it deliberately sampled drops, not just keeps.

**Do not read the 96%-vs-58% split as an anchoring effect.** Mode is confounded with which rows got it: the revealed rows are the *disputes*, selected because the screeners contradicted each other, and the blind rows are the agreed majority. Different populations, different base rates. The comparison that does hold is same-population: agreed keeps judged with the verdict revealed (12 controls, 11/12) against agreed keeps judged blind (73 rows, 70/73) — **no detectable difference** (Fisher exact, p = 0.46), though n=12 can only rule out a large effect, not a modest one.

**Measured rejection rate: 13.7%** — 19 of 139 screened rows judged not good enough, counted directly rather than estimated. A further 5 were excluded without a quality judgement (3 removed as irreproducible, 2 screened only by a stand-in model). An earlier version of this README quoted **15.5%**, a stratum-weighted estimate from a 51-row sample; full coverage replaces it with a census.

**What this is weaker than.** Verifying each label from scratch, rather than adjudicating a screener's verdict against four rules. Specifically: the screening layer is still two *models*, not two people; the blind pass is one reader, so an error that reads as reasonable to both models and to that reader survives; and two rows screened only by a stand-in model remain excluded without a verdict. Raw data for anyone who wants to check: `data/goldset_screened.jsonl` (every rule score and reason), `data/claude_screen.jsonl` (the second opinion), `data/audit_decisions.jsonl` (every adjudicated row, stamped with whether it was decided blind), `data/audit_results.json` (the arithmetic above).

Split with seed 42: `{N}` train (reranker fine-tuning only) / `{N}` test (**every number above**). The test split was never seen during training.

## The fine-tuned reranker

Base `cross-encoder/ms-marco-MiniLM-L-6-v2`, fine-tuned on `{N}` pairs — one positive plus four hard negatives per training question, mined from the top-20 retrieved chunks with gold removed. 3 epochs, lr 2e-5, batch 16, seed 42, ~2 min on a Colab T4.

Model card, hyperparameters, and measured delta: [HF link]

## Architecture

_(diagram)_

Six lanes fan out in parallel inside a LangGraph `StateGraph`, join through a reducer, then pass through generation and a groundedness judge. A lane that errors records the error and is excluded from the join rather than failing the run.

| Layer | Choice |
|---|---|
| Orchestration | LangGraph StateGraph (parallel fan-out/join) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Sparse | rank_bm25 (BM25Okapi) |
| Vector store | PostgreSQL + pgvector |
| Rerankers | ms-marco-MiniLM-L-6-v2, stock and fine-tuned |
| Generation & judging | Groq llama-3.1-8b-instant |
| Backend | FastAPI on Hugging Face Spaces |
| Frontend | React 19 + Vite on Vercel |

## Reproducing these numbers

```bash
python -m backend.ingest --corpus data/demo_corpus
python -m backend.goldset.generate --n 350     # draft    -> data/goldset_draft.jsonl
python -m backend.goldset.screen               # 4 rules  -> data/goldset_screened.jsonl
python -m backend.goldset.assemble             # first 120 -> data/goldset.jsonl
python -m backend.goldset.audit                # interactive -> data/audit_results.json
python -m backend.goldset.split                # 70/30, seed 42
python -m training.mine_negatives
# training/train_reranker.ipynb on Colab → push to HF Hub
python -m backend.evaluate --split test --out data/results.json
```

`python -m backend.goldset.verify` is the fully-manual alternative to screen + assemble, kept because it is the ground truth the screener is measured against.

## Known limitations & honest tradeoffs

- **The evaluation set is `{N}` questions.** Enough to show direction, too small for tight confidence intervals. Differences under ~0.05 should not be read as meaningful.
- **Every gold row was read by the author, but it is adjudication, not verification.** The author judged each row against the four rules; nobody re-derived the labels from scratch. `status` stays `screened`, and the set is never described as human-verified.
- **The blind pass is one reader.** 73 agreed keeps were judged with no screener verdict visible and 3 were moved off `keep`, so the two-screener rule is validated at n=73 rather than on 10 controls. An error that reads as reasonable to both models *and* to that reader still survives — both screeners are LLMs, so correlated error remains plausible.
- **Two rows screened only by a stand-in model stay excluded without a verdict.** `q115` and `q117` were screened by `llama-3.1-8b-instant` after the real screener's quota ran out; adjudicating them would import rows the screener never judged. Coverage is 137 of 139 screened rows, and 115 of 115 gold rows.
- **The screener agreed with the author on only 58% of the rows judged with its verdict revealed**, and 0% on the rows it chose to drop in the first pass. The screening step is not trustworthy unsupervised; the adjudication is what makes the set usable.
- **Every question is answerable by a single chunk.** Multi-hop retrieval is a real problem this benchmark does not measure.
- **MiniLM-L-6 over bge-reranker-base** — 22M vs 278M parameters. The larger model would likely score better; it does not fit the free-tier CPU budget. Documented tradeoff, not an oversight.
- **bge-small over bge-large** — 384 vs 1024 dimensions, roughly 3× faster on CPU at some cost to retrieval quality.
- **The groundedness judge is an LLM**, with its own error rate. Measured at `{N}%` agreement with human labels on a 30-sample audit rather than assumed to be correct.
- **Uploaded corpora are in-memory only** — lost on restart, evicted after 2 hours, single-process. Persisting them needs auth and quotas, which are out of scope.
- **Rerank depth is capped at 20.** Deeper reranking would likely improve results and would blow the latency budget.

### Added on Day 2, when the lanes first produced numbers

- **The hybrid lane ties BM25 on recall@10 rather than beating it.** On the 35-row test split both find a gold chunk for 30 questions. They are not the *same* 30 — fusion gets `q032`, BM25 gets `q123` — so the tie is a coincidence of totals at small n, not a ceiling. On the 80-row train split fusion wins 68/80 vs 62/80, and over all 115 rows 98/115 vs 92/115. Fusion also wins the test split on recall@5 (0.83 vs 0.69) and MRR (0.60 vs 0.54). Reproduce with `python -m backend.sweep_rrf`. **The tie was not resolved by tuning `rrf_k`**: k was swept on the train split only, and no cell in k ∈ {1,5,10,20,30,60,100} × depth ∈ {10,20,30,50} turns the test tie into a win. Picking one that did would be fitting the benchmark to its own answer key.
- **`rrf_k` stays at the paper's 60 even though the train sweep's argmax is 20.** The two differ by 0.005 MRR on n=80 — well under one question — and recall@10 is flat across k=10..60. Re-fitting a cited default to that is fitting noise.
- **p95 latency at n=35 is a tail estimate from a handful of samples.** The 95th percentile interpolates between the 3rd- and 2nd-slowest queries. It is not a stable SLA figure and should not be read as one.
- **Latency is measured against a hosted Supabase over the public internet.** The dense lanes' ~450 ms p95 is dominated by that network round-trip, not by pgvector. `ARCHITECTURE §8` budgets 120 ms for a top-50 on a co-located database. The retrieval-quality numbers are unaffected; the latency column is not comparable to a co-located deployment.
- **The reranker's effect is the opposite of what `EVALUATION_SPEC §2` predicts.** The spec expects a reranker to barely move recall@10 while noticeably lifting MRR. Measured: recall@10 +0.0571, MRR +0.0078. Fusion had already put a gold chunk in the top 10 for 30 of 35 questions, so there was little ranking left to win, and the gains came instead from promoting chunks out of the 11–20 band — possible because `rerank_depth` (20) exceeds the scoring depth (10). Recorded, not reconciled.
- **Cost per query is measured token counts × a vendor list price.** The token counts come from each API response. The rates do not — they are Groq's published prices, checked on the date stamped in `results.json.pricing`. If the rate goes stale the cost column is wrong by exactly that factor.
- **The cross-encoder truncates (query, passage) pairs at 512 tokens.** A chunk whose answer sits in its final sentence can be truncated away from the reranker while remaining perfectly retrievable by the lanes that do not rerank.
- **Lane 5 is the one lane whose numbers are not reproducible — and the drift reaches the recall column, not just the third decimal place.** Everything else here is deterministic given the corpus and the seed. HyDE calls a hosted LLM, and batched GPU inference is not bit-reproducible however the sampler is configured. Three runs over the identical test split:

  | run | recall@5 | recall@10 | MRR | $/query |
  |---|---|---|---|---|
  | 1 | 0.8571 | 0.9429 | 0.6286 | $5.043e-05 |
  | 2 | 0.8571 | 0.9429 | 0.6310 | $5.025e-05 |
  | 3 | 0.8286 | 0.9143 | 0.6214 | $4.854e-05 |

  After runs 1 and 2 this entry said the recall figures held and only MRR drifted. Run 3 moved recall@5 and recall@10 by a whole question each, so that claim was an understatement and is corrected here rather than quietly. The generated passage can differ enough to change which chunks the dense arm returns, not merely how they are ordered. **Lane 5's published row is one sample from a distribution about a question wide.** Lanes 1–4 reproduced exactly across all three runs.
- **HyDE's generator is a reasoning model and bills its thinking.** `openai/gpt-oss-20b` emits reasoning tokens before any content, and they count toward both cost and latency. Run at `reasoning_effort="low"` (82 completion tokens, ~370 ms); at `"medium"` it is 267 tokens and ~765 ms, which alone exceeds the 700 ms HyDE budget.
- **The model that screened the gold set no longer exists.** `llama-3.1-8b-instant` was decommissioned by Groq on or before 2026-08-20 and now returns 404. Rows stamped `screen_model: llama-3.1-8b-instant` (the two excluded stand-in rows) cannot be re-screened by that model, and the judge and HyDE generator have moved to `openai/gpt-oss-20b`. The gold set itself was screened by `openai/gpt-oss-120b`, which is still available.

## License

MIT. The evaluation set is released under [license] — please cite if you use it.

---

Built by **Muhammad Talal Mohsin** — [GitHub](https://github.com/talalmohsin-98) · [LinkedIn](https://linkedin.com/in/talal-mohsin-kaleem)
