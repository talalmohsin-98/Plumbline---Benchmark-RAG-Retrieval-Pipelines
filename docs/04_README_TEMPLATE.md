# Plumbline

**Upload your documents. Race six retrieval pipelines. Get real numbers.**

Live: [craftwyre.com](https://craftwyre.com) · Dataset: [HF] · Fine-tuned reranker: [HF]

> Everything below is measured on a human-verified evaluation set and reproducible from this repository. Commands are in [Reproducing these numbers](#reproducing-these-numbers).

---

## Why this exists

"Add a reranker" and "use hybrid search" are repeated constantly and quantified rarely. Existing evaluation tools score one pipeline after you've built it. Plumbline compares six candidate pipelines against each other on your own corpus, before you commit to one.

## Results

Demo corpus: `{N}` documents, `{N}` chunks. Evaluation set: `{N}` human-verified questions, held-out test split `n={N}`.

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

`{N}` questions, each drafted by an LLM from a single chunk and then **read and verified by hand**. Roughly `{N}%` of drafts were rejected as too vague, verbatim-quoting, or answerable by many chunks equally.

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
python -m backend.goldset.generate --n 150
python -m backend.goldset.verify
python -m training.mine_negatives
# training/train_reranker.ipynb on Colab → push to HF Hub
python -m backend.evaluate --split test --out data/results.json
```

## Known limitations & honest tradeoffs

- **The evaluation set is `{N}` questions.** Enough to show direction, too small for tight confidence intervals. Differences under ~0.05 should not be read as meaningful.
- **Every question is answerable by a single chunk.** Multi-hop retrieval is a real problem this benchmark does not measure.
- **MiniLM-L-6 over bge-reranker-base** — 22M vs 278M parameters. The larger model would likely score better; it does not fit the free-tier CPU budget. Documented tradeoff, not an oversight.
- **bge-small over bge-large** — 384 vs 1024 dimensions, roughly 3× faster on CPU at some cost to retrieval quality.
- **The groundedness judge is an LLM**, with its own error rate. Measured at `{N}%` agreement with human labels on a 30-sample audit rather than assumed to be correct.
- **Uploaded corpora are in-memory only** — lost on restart, evicted after 2 hours, single-process. Persisting them needs auth and quotas, which are out of scope.
- **Rerank depth is capped at 20.** Deeper reranking would likely improve results and would blow the latency budget.

## License

MIT. The evaluation set is released under [license] — please cite if you use it.

---

Built by **Muhammad Talal Mohsin** — [GitHub](https://github.com/talalmohsin-98) · [LinkedIn](https://linkedin.com/in/talal-mohsin-kaleem)
