# Plumbline — Product Requirements

**Owner:** Muhammad Talal Mohsin
**Status:** Approved for build
**Build window:** 5 days
**Repo:** public · **Live:** craftwyre.com

---

## 1. Problem

Teams building RAG systems pick a retrieval pipeline by reading blog posts, not by measuring. "Add a reranker" and "use hybrid search" are folklore — widely repeated, rarely quantified on the asker's own corpus. The result is pipelines that are more expensive and slower than they need to be, with no evidence they retrieve better.

The tooling that does exist (RAGAS, TruLens, DeepEval, Phoenix) evaluates *one* pipeline after you've already built it. Nothing lets you compare *several candidate pipelines against each other* on your own documents before you commit.

## 2. What Plumbline is

A web app and open-source library that ingests any document set, auto-builds a labelled evaluation set from it, then runs the same questions through six retrieval pipelines and publishes a scoreboard of retrieval quality, groundedness, latency, and cost.

One sentence: **upload your documents, race six retrieval pipelines, get real numbers.**

## 3. Users

| User | Need | What they do here |
|---|---|---|
| **Primary — RAG engineer** | "Is a reranker worth the latency on *my* corpus?" | Uploads their docs, reads the scoreboard, picks a lane |
| **Secondary — hiring manager / reviewer** | "Can this person do retrieval engineering?" | Opens the demo corpus, reads the leaderboard and the repo |
| **Tertiary — learner** | "What do these pipeline stages actually do?" | Runs one query, inspects stage-by-stage output |

The secondary user is not an accident. This project exists partly as public evidence of the author's work; that user's needs (instant load, no signup, visible methodology) are first-class requirements, not decoration.

## 4. The six lanes

| # | Lane | What it isolates |
|---|---|---|
| 1 | BM25 only | Lexical baseline |
| 2 | Dense only | Semantic baseline |
| 3 | Hybrid + RRF | Does fusion beat either alone? |
| 4 | Hybrid + RRF + off-the-shelf cross-encoder | Is a stock reranker worth its cost? |
| 5 | Hybrid + RRF + rerank + HyDE | Does query expansion still help after reranking? |
| 6 | Hybrid + RRF + **fine-tuned** cross-encoder | Does domain fine-tuning beat the stock reranker? |

Lane 6 is the differentiator and is not optional. Every other lane is a control that makes lane 6's result interpretable.

## 5. Scope — in

- Ingest PDF, TXT, MD (≤ 20 files, ≤ 20 MB total per session)
- Automatic gold-set generation with a manual verification workflow
- All six lanes, run in parallel, with per-lane traces
- Metrics: recall@5, recall@10, MRR, groundedness rate, p95 latency, cost per query
- Live single-query race view with per-lane streaming results
- Static leaderboard for the built-in demo corpus, precomputed and instant
- Published artifacts: gold-set dataset and fine-tuned reranker on Hugging Face Hub
- Free public hosting, no signup

## 6. Scope — out (explicit non-goals)

- **No user accounts, no persistence of user uploads beyond a session.** Auth is a day of work that demonstrates nothing new.
- **No chat interface.** This is a measurement tool. A chat box would make it look like every other RAG demo.
- **No nDCG.** recall@k and MRR carry the ranking story; nDCG adds interpretation cost without adding signal at this scale.
- **No multi-turn conversation, no agents that take actions.** Retrieval quality is the subject.
- **No paid tier, no billing, no waitlist.** Monetization is a later decision and must not shape this build.
- **No unaudited LLM-as-judge for retrieval correctness.** Gold labels are drafted and screened by an LLM against four explicit drop rules, and a human adjudicates every row that enters the gold set; the agreement rate between screener and human is published next to every metric. *(Revised on Day 1. The original rule read "gold labels are human-verified" and assumed reading 350 drafts by hand was affordable. What is non-negotiable is that a human measures the labelling process, not that a human performs every labelling act — see `02_EVALUATION_SPEC.md` §1, which also states plainly what the weaker instrument costs.)* The groundedness judge is separate and likewise audited, at n=30.

## 7. Success criteria

The build is a success if all of the following are true on Friday:

1. A stranger can open craftwyre.com and see a populated leaderboard **in under 2 seconds**, with no cold-start wait and no empty state.
2. The scoreboard reports all six metrics for all six lanes on an **LLM-screened, human-audited** gold set of ≥ 100 questions, with the screener–human agreement rate published alongside.
3. Lane 6 uses a reranker fine-tuned by the author, published on HF Hub, and the delta versus lane 4 is reported **whether it is positive or negative**.
4. The repo README states the methodology precisely enough that a reader could reproduce the numbers.
5. A live query can be run against the demo corpus and returns all six lanes' results.
6. `pytest` passes and runs in CI on every push.

## 8. Explicitly acceptable outcomes

- **The fine-tuned reranker loses to the stock one.** This is a publishable result and is reported as such. A negative result honestly reported is stronger evidence of competence than a positive one that nobody can check.
- **HyDE hurts recall.** Also publishable.
- Lanes disagree about which is best on different query types. Interesting; report it.

The one unacceptable outcome is a number that cannot be reproduced from the repo.

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gold-set verification eats too much of Day 1 | High | Hard cap at 120 questions; generate 150, keep the first 120 that verify clean |
| Fine-tune produces no improvement | Medium | Acceptable outcome per §8; report the delta either way |
| Free-tier cold start makes the site look broken | High | Precomputed `results.json` served statically; live query loads behind it |
| CPU inference too slow for six lanes | Medium | bge-small embeddings, MiniLM-L-6 reranker, top-20 rerank depth only |
| Scope creep into a chat product | Medium | §6 is binding |

## 10. Honesty requirements

The README must carry a "Known limitations" section in the style of the author's Rahbar AI project — stating what is simulated, what is small-scale, and what would break in production. This is a requirement, not a nicety. Every claim on the site must be traceable to code in the repo.
