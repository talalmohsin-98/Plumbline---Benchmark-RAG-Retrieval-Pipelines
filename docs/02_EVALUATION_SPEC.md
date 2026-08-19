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

### Multi-labelling duplicated answers

Documentation repeats itself, and a single label would score a lane wrong for finding the other copy. Five FastAPI chunks each tell the reader to install `python-multipart`; the LangSmith Cloud deployment sentence is byte-identical in the LangChain and LangGraph pages; the Agent Chat UI page is duplicated wholesale. A lane that returns any of them has answered the question.

So every kept question is checked against the corpus and **all equally-correct chunks are added to `gold_chunk_ids`**. Candidates are generated mechanically — every other chunk sharing a ≥8-word sentence with the gold chunk — and then adjudicated by hand, because the mechanical pass is far too generous: it flagged 77 of 98 kept rows, and almost all of those shared only boilerplate (`Before you begin, ensure you have the following`), import blocks, or the 64-token overlap between adjacent chunks of one document. None of that carries an answer. 21 rows survived adjudication, carrying 27 extra chunks between them, and every addition was verified by checking that the added chunk literally contains the answer token. Two candidates failed that check and were dropped.

The map lives in `data/multilabel.json` with a stated reason per entry, so the judgement is reviewable rather than buried in code.

### How it is built

The gold set is **LLM-screened and human-audited**. It is not hand-verified row
by row, and nothing in this repository may describe it as such.

1. **Draft (automated).** For each of 350 randomly sampled eligible chunks, ask Groq `openai/gpt-oss-120b` at `temperature=0.3` to write one specific question that this passage — and only this passage — answers. A second call at `temperature=0` asks which of the chunk and its two document neighbours actually contains the answer, and the gold label follows that answer rather than the chunk the question was drafted from. See `backend/goldset/generate.py`.

2. **Screen (automated).** Every draft is scored against the four rules below, **one call per rule**, at `temperature=0`. A combined "is this good?" call collapses four distinct failure modes into one bit and cannot be debugged; four scores say which rule fired, and the per-rule counts say whether the drafting prompt or the screener needs the work. Output: `data/goldset_screened.jsonl`, carrying every rule's score, the verdict, and a one-line reason. See `backend/goldset/screen.py`.

   The screening model defaults to `GOLDSET_MODEL` and is overridable with `--model`. Whichever is used is stamped on every row as `screen_model`, because two runs screened by different models are not comparable evidence and the row is the only place that record can live. **If the gold set was screened by more than one model, the README says which rows got which.**

3. **Assemble.** Take every row verdicted keep or fix, preferring drafts whose gold label the step-1 check confirmed. Fewer than `MINIMUM` (100, the §5 bar) survivors is a shortfall to report, never a gap to pad with rejects. The set is uncapped: 120 was the Day 1 aspiration, and truncating a closed, fully adjudicated set to hit it would discard good rows. Output: `data/goldset.jsonl`, every row `status: "screened"`. See `backend/goldset/assemble.py`.

4. **Adjudicate (manual, non-negotiable).** Every draft is screened a second time, independently, and the author then decides **every row that enters the gold set** — first the disputed rows and a control sample, then a full-coverage pass over every row the two screeners had agreed to keep. The author's decision is authoritative everywhere. See `backend/goldset/audit.py`, `backend/goldset/assemble.py`, and *What the adjudication measured* below.

**The gold set closed at 115.** The §5 bar is 100; 120 was an aspiration, not a requirement, and no further rows were drafted to reach it.

### How the gold set is described

**LLM-drafted, LLM-screened, and human-adjudicated.** The precise claim:

- Two independent LLM screeners scored every draft against the four rules.
- **The author personally decided every one of the 115 rows in the gold set — 100% coverage.** 137 decisions in total, over 139 screened rows.
- 73 of those decisions were taken in a **fully blind** pass: no screener verdict shown before or after, and no running agreement rate on screen. The other 62 were taken with the screeners' verdicts revealed after each keypress.
- The second opinion was itself produced by an AI (Claude) reading each question and chunk, not by a second person.

Two limits on that claim, and both belong wherever it is quoted:

- **"Two screeners agreed" is two models agreeing, not two people.** The screening step is still entirely LLM work; what is human is the adjudication on top of it.
- **Coverage now includes the rejection stratum.** `q017` and `q042` were the only rows rejected on both screeners agreeing with no human reading either — the invisible-false-drop class. Both were adjudicated and both were kept, which is itself the finding: the screener's drops were wrong again, as they were 0/5 in the first pass. **These two were decided with the screener's verdict and reasoning visible**, so they are stamped `blind: false` and count in the revealed stratum, not the blind one.
- **Two rows remain excluded without a human verdict.** `q115` and `q117` were screened only by a stand-in model (`llama-3.1-8b-instant`) after the screener's quota ran out. They are excluded on process grounds rather than on quality, and no verdict was sought: adjudicating them would import rows the real screener never judged. Coverage is therefore **137 of 139 screened rows**, and 115 of 115 gold rows.

The rows still carry `status: "screened"`, and **"human-verified" must not appear anywhere describing this gold set.** Adjudicating a screener's verdict against four rules is not the same act as verifying a label from scratch, and `verify.py` — which does the latter — is what writes `status: "verified"`.

### The four rules

Each is scored independently. A question can break more than one.

| # | Rule | Verdict | Scored by | Why |
|---|---|---|---|---|
| 1 | The question could be answered just as well by 3+ other chunks | **drop** | model | It measures nothing — every lane will find something |
| 2 | The question reuses more than four consecutive words from the chunk | **fix** | exact match | It hands BM25 a free win and makes lanes look identical |
| 3 | The chunk does not state the answer | **drop** | model | The label is wrong, so the pair measures nothing |
| 4 | Answering requires combining this chunk with another | **drop** | model | Valid retrieval problem, wrong problem for this benchmark |

**A drop rule beats the fix rule.** A question that is both vague and verbatim is not worth rephrasing.

**Rule 2 is the fixable one.** The label is right and only the wording gives the answer away, so the screener rewrites the question — told the exact phrase to avoid — and re-measures. Two attempts, then the pair is dropped. A rephrasing is not re-scored against rules 1, 3 and 4; the audit samples fixes precisely so that assumption is measured rather than trusted.

**Rule 2 is not sent to the model.** "More than four consecutive words" is an exact property of two strings, and a longest-common-run scan answers it identically on every run. An LLM asked to count consecutive words is guessing, and its guess would be the one number in this pipeline that could not be reproduced. Threshold checked against a 30-draft sample: run lengths were 0–4 for 29 of them and 8 for the one genuine offender, so the cut sits in a clear gap rather than mid-distribution.

### Why the audit is the load-bearing step

If the labels come from an LLM, are screened by an LLM, and are never checked by a person, the benchmark measures how well retrieval finds chunks that a language model thought were findable. That is circular and worthless. Screening replaces the *labour* of reading 350 drafts; it does not replace the human, and the audit is where the human stays in the loop.

Two properties make the audit a measurement rather than a formality, and both are enforced in code and covered by tests:

- **The screener's verdict is hidden until after the keypress.** Shown first, it anchors: the eye finds reasons for a verdict it has already read. What gets measured then is compliance, not agreement.
- **The sample is stratified across all three verdicts, not just the keeps.** Auditing keeps alone measures false positives — bad questions that got through. It cannot see a *false drop*, because a wrongly discarded question never appears in the gold set. Those are the expensive errors: invisible in the finished file, and enough of them quietly bias the set toward whatever the screener finds easy.

Fixed rows are shown as the screener first saw them — the draft wording, not the rephrasing — otherwise the auditor judges a question whose verbatim overlap has already been removed and agrees by construction.

`data/audit_results.json` reports the arithmetic. `by_verdict` is where the diagnosis lives: a high keep-agreement with a poor drop-agreement means the screener is discarding good questions, which no amount of inspecting the finished gold set would ever reveal. `agreement_overall` is now a census over every decision rather than a sample rate.

**`agreement_weighted` has been retired.** It re-weighted an evenly-drawn sample back onto the real class sizes, which was the correct arithmetic while most rows were unread. Full coverage leaves no unsampled stratum to project onto, and keeping an estimator beside a census only puts two numbers for one quantity into circulation.

### What the adjudication measured

137 decisions over 139 screened rows, in two passes: 64 rows with the screeners' verdicts revealed after each keypress (50 disputed + 12 controls + the 2 rejection-stratum rows), then a full-coverage blind pass over 73 agreed keeps.

| | agreement with the author |
|---|---|
| `openai/gpt-oss-20b` (screener), all 137 decisions | **78%** (107/137) |
| — on the 64 revealed rows (disputes, controls, rejections) | 58% (37/64) |
| — on the 73 blind rows (agreed keeps) | 96% (70/73) |

**Neither screener is trustworthy alone, and that is still the finding.** On the disputed rows the screener agreed with the author 60% of the time — roughly two in five verdicts would have been decided wrongly by it acting unsupervised. In the first pass the screener's drops fared worst of all: 0/5, every discarded row being one the author would have kept. A false drop is invisible in the finished gold set, which is why the draw went looking for them.

#### What the blind pass guaranteed, and what it did not

The renderer never reads a screener field. `render_question` touches the
question, the source document, the chunk id and the fixed rule text, and
nothing else; in blind mode `run` returns before `render_reveal` and the
progress line withholds the running agreement rate. Four tests pin this,
including one that drives the full loop with `screen_verdict`, `screen_rule`,
`screen_reason`, `screen_scores` and `screen_model` all populated with marker
strings and asserts none reaches the terminal. That last test was added after
the fact: the earlier "identical bytes for all three verdicts" test could not
have caught a `screen_scores` leak, because its fixtures never set the key.

**The queue file is not redacted.** `build_coverage_queue` copies the whole
screened row, so the queue it writes contains every verdict the pass was blind
to. Blindness therefore held structurally at the terminal and procedurally at
the file. (Queue files are outputs and are no longer tracked;
`data/audit_decisions.jsonl` is the durable record.) For this pass the author states as fact that the file
was never opened before or during the run -- the only thing executed against it
was a boolean pattern check printing `True`/`False` and no content.

**Known limitation, deferred:** writing a redacted queue in blind mode would
make this structural rather than procedural. It conflicts with `--queue-in`,
which needs `screen_verdict` to compute `agreed`, so it needs a real design
rather than a patch at the end of a working day.

#### The by_mode table is confounded. Do not read it as an anchoring result.

`by_mode` reports 96% blind against 58% revealed, and that gap is **population, not method**. Mode was never randomised across rows:

- the **revealed** stratum is the disputed rows plus controls — rows selected *because* the two screeners contradicted each other, so the screener is wrong on a large fraction of them by construction;
- the **blind** stratum is the rows both screeners agreed to keep — the easy majority, where the screener is right almost all the time.

Two populations with very different base rates of screener error will produce very different agreement rates under *any* protocol. Reading the 38-point gap as evidence that revealing the verdict anchors the auditor would be a straightforward confound, and the number is reported split only so that the two passes are not silently averaged.

**The one comparison that holds** is same-population, different-mode: agreed keeps judged with the verdict revealed (the 12 controls) against agreed keeps judged blind (the 73).

| agreed keeps | mode | agreement |
|---|---|---|
| controls, n=12 | revealed | 11/12 (92%) |
| coverage pass, n=73 | blind | 70/73 (96%) |

**No detectable anchoring effect** (Fisher exact, two-tailed, p = 0.46). The caveat is the sample size: at n=12 the control arm cannot rule out a modest effect, only a large one. This is a null result on a small sample, not a demonstration that revealing the verdict is harmless.

**Measured rejection rate: 13.7%** — 19 of 139 screened rows judged not good enough, every one of them by the author. A further 5 rows were excluded without being judged on quality at all: 3 removed by hand as irreproducible (`data/exclusions.json`) and 2 screened only by a stand-in model. 115 rows were accepted, 82.7% of those screened.

This is a **direct census**, printed by `backend.goldset.assemble`, and it supersedes the **15.5%** figure this document previously carried. That earlier number was a stratum-weighted *estimate* projected from a 51-row sample onto 116 screened rows — correct arithmetic for its time, but an estimate. Full coverage means the quantity can simply be counted, and the two figures agreeing to within half a point is a coincidence worth neither weight nor comment. **13.7% is the number to publish**, and it sits below the 20–30% this spec originally budgeted for.

### Budget: the constraint that shapes this step

Groq's free tier meters **tokens per day, per model** — 200k/day for `gpt-oss-120b`. Measured on this corpus: drafting costs **~2,190 tokens per row** (a question plus a three-passage label check) and screening **~2,520** (three rule calls). A 350-draft pass plus screening is therefore roughly **1.6M tokens — about eight model-days** of free-tier budget, and quota pools do not combine across models.

Two consequences are built into the code rather than left to discipline:

- `generate` and `screen` both **resume**. `generate` reads its output before truncating it; a rerun re-emits earlier drafts without an API call and fills only the gaps.
- Both **stop** when the daily budget is gone instead of retrying. The per-minute and per-day limits arrive as the same 429, but the daily bucket refills at `limit/1440` tokens a minute, so retrying is futile rather than merely slow. The first run of this pipeline could not tell them apart: it exhausted the budget at chunk 92 and ground through the remaining 259, writing 256 error rows that looked like content failures.

### What this costs, stated plainly

This is a weaker instrument than reading every draft by hand, and the README says so in those words. Measured on Day 1, what is lost:

- **The two-screener rule is now validated rather than assumed.** Every agreed-keep row was read blind by the author: 73 rows, of which 3 were moved off `keep` (1 dropped, 2 sent back as fixes). Screener agreement on that stratum is 70/73. This replaces the earlier claim that 56% of rows rested on two models agreeing with a 10-row control sample behind them — that is no longer true of any row in the gold set, and the warrant is n=73 rather than n=10.
- **A bias both screeners share is still invisible to the screening step.** The screeners' disagreement is what surfaces problems, and where they err in the same direction nothing flags it. The blind pass is the mitigation — a human read every accepted row without seeing either verdict — but it is one reader, and a question that reads as reasonable to both models and to that reader still gets through.
- **The rejections are less well covered than the acceptances.** 2 rows were dropped on screener agreement alone with no human reading them. That asymmetry favours the gold set's contents over its exclusions.
- **The second opinion is not a second person.** "Two screeners agreed" means two models agreed. Nothing about full coverage changes this: the screening layer is still entirely LLM work.

What is gained: the four rules are applied uniformly instead of drifting after the fiftieth question, every verdict carries a written reason that can be argued with, and the disputed-queue design spent the author's first hour on the rows that were genuinely undecided before the coverage pass swept the rest.

The honest summary is *LLM-drafted, LLM-screened, human-adjudicated at 100% of gold rows, screener agreement 78% overall* — never *human-verified*.

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
  "goldset": {
    "total": 120, "train": 84, "test": 36,
    "built_by": "llm-screened, adjudicated by the author with AI assistance",
    "screened": 116,
    "audit": {
      "adjudicated": 51, "disputed": 41, "controls": 10, "controls_held": 9,
      "agreement_screener": 0.59, "agreement_second_opinion": 0.55,
      "agreement_on_drops": 0.0, "rejection_rate": 0.155
    }
  },
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
python -m backend.goldset.generate --n 350     # drafts   -> goldset_draft.jsonl
python -m backend.goldset.screen               # 4 rules  -> goldset_screened.jsonl
python -m backend.goldset.assemble             # first 120 -> goldset.jsonl
python -m backend.goldset.audit                # interactive -> audit_results.json
python -m backend.goldset.split                # 70/30, seed 42
python -m training.mine_negatives
# training/train_reranker.ipynb on Colab → push to HF
python -m backend.evaluate --split test --out data/results.json
```

`backend.goldset.verify` is the fully-manual alternative to screen + assemble: it walks every draft and takes a `k`/`f`/`d` decision on each. It is retained because it is the ground truth the screener is measured against, and it writes `status: "verified"` rather than `status: "screened"` so the two can never be confused downstream.

Fixed seeds throughout — 42 for sampling, splitting, and the audit draw; `temperature=0` for every classification call. `generate` and `screen` are resumable and append as they go, so an interrupted run costs minutes rather than the whole pass. If a rerun produces materially different numbers, that is a bug to fix before Friday, not a footnote.
