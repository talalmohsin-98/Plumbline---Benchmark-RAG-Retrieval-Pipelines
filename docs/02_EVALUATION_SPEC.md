# Plumbline — Evaluation & Fine-Tuning Specification

This is the most important document in the project. Everything else is plumbing around the numbers this file defines. **If a number cannot be reproduced by following this document, it does not go on the site.**

---

## 1. The gold set

### What it is

A JSONL file where each line is a question paired with the chunk ID that actually answers it.

```json
{"qid": "q001", "question": "What is the fee for an urgent CNIC renewal?", "gold_chunk_ids": ["nadra_ch_042"], "source_doc": "NADRA.txt", "query_type": "factual"}
```

`gold_chunk_ids` is a list because some questions are legitimately answered by more than one chunk. A retrieval is correct if it surfaces **any** of them.

### How it is built

1. **Draft (automated).** For each of ~150 randomly sampled chunks, ask Groq `llama-3.1-8b-instant` at `temperature=0.3`: *"Write one specific question that this passage — and only this passage — answers. Do not use the words 'this passage' or 'the document'."* Record the source chunk as the provisional gold label.

2. **Verify (manual, non-negotiable).** For each drafted pair, the author reads it and marks:
   - **keep** — the question is specific and the chunk genuinely answers it
   - **fix** — the question is right but needs rewording
   - **drop** — vague, unanswerable, or answerable by many chunks equally

3. **Stop at 120 keeps.** Do not chase 150.

### Verification rules

Drop a pair if any of these is true:

- The question could be answered just as well by three or more other chunks. *(It measures nothing — every lane will find something.)*
- The question quotes the chunk nearly verbatim. *(It hands BM25 a free win and makes lanes look identical.)*
- The question needs information the corpus does not contain.
- The answer requires combining two chunks. *(Valid retrieval problem, wrong problem for this benchmark.)*

Roughly 20–30% get dropped. Budget for it.

### Why this cannot be skipped or automated

If the labels come from an LLM and are never checked, the benchmark measures how well retrieval finds chunks that an LLM thought were findable. That is circular and worthless. **Human verification is the only thing separating this project from every unverifiable RAG demo on GitHub.** It is also the least interesting day of the build. Do it anyway, on Day 1, first.

---

## 2. Metric definitions

All metrics are computed by `metrics.py` over the gold set. Formulas are stated precisely because they go in the public README.

### recall@k

**Plain English:** of all the questions, what fraction had a correct chunk somewhere in the top *k* results.

```
recall@k = (number of questions where top-k contains ≥1 gold chunk) / (total questions)
```

`recall@10 = 0.79` means: for 79% of questions, the right chunk made the top 10. Range 0–1, higher is better, and it is **the primary metric** — if the right chunk is not retrieved, no amount of clever generation can recover.

Report both `recall@5` and `recall@10`. The gap between them tells you whether a lane is ranking well or merely retrieving broadly.

### MRR (Mean Reciprocal Rank)

**Plain English:** on average, how high up the list was the first correct chunk.

```
For each question: RR = 1 / (rank of first gold chunk)     [0 if not found in top k]
MRR = mean(RR across all questions)
```

Gold at rank 1 → 1.0. Rank 2 → 0.5. Rank 5 → 0.2. `MRR = 0.62` roughly means "the first correct chunk typically sits around position 1.6".

recall@k asks *did we find it*; MRR asks *did we rank it first*. A reranker should barely move recall@10 while noticeably lifting MRR — **that is exactly the effect this benchmark exists to demonstrate.**

### Groundedness rate

**Plain English:** of the answers generated from each lane's retrieved chunks, what fraction are fully supported by those chunks.

Every sentence of the generated answer is scored by Groq `llama-3.1-8b-instant` at `temperature=0` against the retrieved context. An answer is grounded only if every sentence is supported.

```
groundedness_rate = (fully grounded answers) / (total answers)
```

**Required calibration step:** hand-label 30 of the judge's verdicts yourself and report the agreement percentage in the README. This one paragraph — acknowledging that the measuring instrument has its own error rate — is a stronger competence signal than any metric on the page. Almost nobody does it.

### p95 latency

The 95th percentile of per-query wall-clock time per lane, in milliseconds. p95 rather than mean because tail latency is what users feel. Measured server-side, excluding network.

### Cost per query

```
cost = (prompt_tokens × input_rate + completion_tokens × output_rate)
```

Measured from actual token counts returned by the API, never estimated. Lanes 1–4 and 6 make no LLM call during retrieval and cost **$0.000** — that zero is a finding, not a gap. It is the entire argument against HyDE if HyDE fails to earn its price.

### Not measured

**nDCG@10** is deliberately excluded. It weights by graded relevance and rank position, but this gold set has binary labels, so nDCG would collapse toward a noisier restatement of MRR. Excluding it is a defensible choice — say so if asked, rather than pretending it was an oversight.

---

## 3. Fine-tuning protocol (Lane 6)

The part of the project that closes the transformer gap. **Run on Google Colab free T4.** CPU training will consume an entire build day.

### Training data

Built from the same gold set, split before anything else happens:

| Split | Questions | Use |
|---|---|---|
| Train | 84 (70%) | Reranker fine-tuning |
| Test | 36 (30%) | **All reported metrics** |

**The test split is never seen during training.** Every number on the leaderboard comes from the test split only. Reporting train-split numbers would make the fine-tuned lane look excellent and be completely meaningless — and an interviewer will ask about this split first.

### Hard negative mining

Random negatives teach the model nothing; it learns to separate obviously-unrelated text, which it already does. Hard negatives are chunks that *look* right and aren't.

For each training question:
1. Run BM25 + dense retrieval, take top 20.
2. Remove any gold chunks.
3. Sample 4 from the remainder as negatives — these are chunks that scored well but are wrong.

Yields roughly 84 × 5 = **420 training pairs** (1 positive + 4 negatives per question).

### Training configuration

```python
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

epochs            = 3          # 420 pairs — more will overfit
batch_size        = 16
learning_rate     = 2e-5
warmup_steps      = 10%
loss              = BinaryCrossEntropy
max_length        = 512
eval_strategy     = per epoch, on a held-out slice of train
seed              = 42         # recorded for reproducibility
```

Expect ~2 minutes on a T4. If it takes an hour, something is wrong with the data loader, not the model.

### Publishing

Push to `talalmohsin-98/plumbline-reranker-v1` on HF Hub with a model card stating: base model, training set size, hard-negative strategy, hyperparameters, and the measured delta versus the stock reranker. **The model card is a portfolio artifact in its own right** — a recruiter can open it without cloning anything.

### Reporting the result

Report the lane 4 → lane 6 delta on the test split, **in whichever direction it goes**:

> Fine-tuned reranker: recall@10 0.61 → 0.79 (+0.18), MRR 0.44 → 0.66 (+0.22), test split n=36.

If it loses, the README says so plainly and offers a hypothesis (420 pairs is small; the stock model was already trained on MS MARCO which may cover this domain). **A published negative result is more credible than an unverifiable positive one**, and the ability to say so is precisely the seniority signal this project is buying.

---

## 4. Results schema

`data/results.json` — committed, regenerated by CI, served directly by `/leaderboard`.

```json
{
  "generated_at": "2026-08-14T18:00:00Z",
  "corpus": {"name": "demo", "docs": 5, "chunks": 1240},
  "goldset": {"total": 120, "train": 84, "test": 36, "verified_by": "human"},
  "models": {
    "embedding": "BAAI/bge-small-en-v1.5",
    "reranker_base": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "reranker_tuned": "talalmohsin-98/plumbline-reranker-v1"
  },
  "judge_agreement": {"sampled": 30, "agreement": 0.87},
  "lanes": [
    {
      "id": "hybrid_rerank_tuned",
      "label": "Hybrid + RRF + fine-tuned reranker",
      "recall_at_5": 0.72, "recall_at_10": 0.79,
      "mrr": 0.66, "groundedness": 0.91,
      "p95_latency_ms": 840, "cost_per_query_usd": 0.0
    }
  ]
}
```

Every field on the site traces to this file. Nothing is typed into the frontend by hand.

---

## 5. Reproducibility

The README must let a stranger regenerate every published number:

```bash
python -m backend.ingest --corpus data/demo_corpus
python -m backend.goldset.generate --n 150
python -m backend.goldset.verify              # interactive
python -m training.mine_negatives
# training/train_reranker.ipynb on Colab → push to HF
python -m backend.evaluate --split test --out data/results.json
```

Fixed seeds throughout. If a rerun produces materially different numbers, that is a bug to fix before Friday, not a footnote.
