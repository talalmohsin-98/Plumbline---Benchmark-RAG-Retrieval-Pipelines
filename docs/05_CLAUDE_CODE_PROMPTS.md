# Plumbline — Claude Code build prompts

Copy-paste prompts, in order. Written for `D:\Agnetic AI\Plumbline\`.

---

## How to use these

**One session per day.** Run `/clear` between days. Long sessions drift — by hour four Claude Code has forgotten the constraints in `00_PRD.md` and is happily adding auth.

**Never chain two prompts without reading the output.** Every prompt below ends with a stop instruction. Honour it. The whole value of Friday depends on you being able to defend this code on Monday, and you cannot defend code you scrolled past.

**Run the ownership prompt (§7) at the end of every day.** Non-negotiable. It is the difference between a portfolio piece and a liability.

**Path note:** your folder has a space in it. Always quote it: `cd "D:\Agnetic AI\Plumbline"`.

---

## Prompt 0 — Rename and scaffold

> The docs in `docs/` were written under a working title. Do two things.
>
> **1. Rename.** Across every file in `docs/` and `CLAUDE.md`, replace the project name "RAG Arena" with "Plumbline" and the identifier `ragarena` with `plumbline`. This includes HF paths: `talalmohsin-98/ragarena-reranker-v1` becomes `talalmohsin-98/plumbline-reranker-v1`, and the dataset becomes `talalmohsin-98/plumbline-goldset-v1`. Do not change anything else — no rewording, no restructuring, no "improvements" to the specs.
>
> **2. Scaffold.** Read `docs/01_ARCHITECTURE.md §3` and create the exact directory structure described, with empty `__init__.py` files and placeholder modules containing only a docstring stating that module's responsibility. Also create: `.gitignore` (Python + Node + `.env`), `requirements.txt` with pinned versions for the stack in §1, `pyproject.toml` configuring ruff, and `.env.example` listing every variable from §10 with dummy values.
>
> Do not implement any logic yet. When done, show me the tree and the diff summary of the rename, then stop.

---

## Day 1 — Corpus and gold set

### Prompt 1.1 — Config and ingestion

> Read `docs/00_PRD.md`, `docs/01_ARCHITECTURE.md`, and `docs/02_EVALUATION_SPEC.md §1`. These are binding — especially the non-goals in PRD §6.
>
> Implement:
>
> - `backend/config.py` — pydantic-settings, every variable from ARCHITECTURE §10, no defaults for secrets
> - `backend/retrieval/dense_store.py` — Supabase pgvector: table creation with an ivfflat index, upsert chunks, top-k cosine query
> - `backend/ingest.py` — load PDF/TXT/MD from a directory, chunk at 512 tokens with 64 overlap, embed with `BAAI/bge-small-en-v1.5`, store with a **stable deterministic `chunk_id`** derived from source filename + chunk index (so re-ingesting produces identical ids — this is an acceptance criterion)
>
> Load the embedding model once at module level, never inside a function called per-request.
>
> Add a `--dry-run` flag that chunks and reports counts without touching the database, so I can sanity-check chunking before paying for embeddings.
>
> Write tests for chunking and `chunk_id` stability. Stub the embedding model and database — `pytest` must pass with no network and no keys.
>
> Stop when done and show me the chunking logic specifically.

### Prompt 1.2 — Gold set generation

> Read `docs/02_EVALUATION_SPEC.md §1` carefully. Implement `backend/goldset/generate.py`.
>
> Sample N random chunks from the ingested corpus. For each, call Groq `llama-3.1-8b-instant` at `temperature=0.3` with the prompt described in the spec. Write drafts to `data/goldset_draft.jsonl` in the schema shown in the spec, with a `status: "draft"` field.
>
> Requirements: `--n` flag defaulting to 150; skip chunks under 200 characters; retry once on API failure then record the failure and continue — one bad chunk must not kill the run; print a running counter.
>
> Do not filter or judge quality automatically. Verification is a human step and the spec says so.
>
> Stop when done and show me the generation prompt you used.

### Prompt 1.3 — Verification CLI

> Implement `backend/goldset/verify.py` — the interactive terminal loop from `docs/02_EVALUATION_SPEC.md §1`.
>
> For each draft: clear the screen, print the question prominently, print the source chunk below it, print progress (`[34/150] · 22 kept · 8 dropped`), and the four rejection rules from the spec as a persistent reminder. Accept a single keypress: `k` keep, `f` fix (opens `$EDITOR` or prompts for a rewritten question), `d` drop, `q` save and quit.
>
> **Append to `data/goldset.jsonl` after every single decision**, not at the end. If I close the terminal at question 80 I must not lose 80 decisions.
>
> Support resuming: on start, skip drafts already decided.
>
> When 120 keeps are reached, print a clear "target reached" message and offer to stop.
>
> Then implement `backend/goldset/split.py`: 70/30 train/test split, seed 42, written to `data/train.jsonl` and `data/test.jsonl`. Print both counts.
>
> Stop when done. I am going to run this myself for the next two hours.

---

## Day 2 — Lanes and metrics

### Prompt 2.1 — Lane interface and lanes 1–3

> `/clear` first. Read `docs/01_ARCHITECTURE.md §4` and `docs/02_EVALUATION_SPEC.md §2`.
>
> Implement `backend/lanes/base.py` — the `Lane` ABC and the `LaneResult` and `StageTrace` dataclasses exactly as specified. Then:
>
> - `lanes/bm25.py` (lane 1) with `retrieval/bm25_index.py` — BM25Okapi, disk-cached index, tokenizer that preserves hyphenated terms
> - `lanes/dense.py` (lane 2)
> - `lanes/hybrid.py` (lane 3) with `retrieval/fusion.py` — Reciprocal Rank Fusion
>
> In `fusion.py`, add a comment above the `k` parameter stating what k controls, why 60 is the default, and that this project can measure it rather than assume it. I will be asked about this in an interview.
>
> Every lane populates `stages` with what it did — this drives the UI inspector later.
>
> Add a lane registry in `lanes/__init__.py` such that adding a lane requires only a new file plus one registry entry.
>
> Unit-test RRF against a fixture I can verify by hand: two ranked lists, expected fused order computed manually.
>
> Stop when done and walk me through the RRF implementation line by line.

### Prompt 2.2 — Reranking and lanes 4–5

> Implement `retrieval/rerank.py` — cross-encoder scoring with a configurable checkpoint, so the same class serves both the stock and fine-tuned rerankers.
>
> Then `lanes/reranked.py` (lane 4: hybrid → rerank top-20) and `lanes/hyde.py` (lane 5: HyDE query expansion → hybrid → rerank).
>
> Rerank depth comes from config (`RERANK_DEPTH=20`), never a literal. Add a comment explaining why depth is capped at 20 — cross-encoder cost is linear in candidates and this is the largest CPU item in the system.
>
> Lane 5 must record its Groq token usage into `LaneResult.tokens_used`. It is the only lane with a nonzero cost and that fact is a finding.
>
> Stop when done.

### Prompt 2.3 — Metrics and the evaluation runner

> Read `docs/02_EVALUATION_SPEC.md §2 and §4`. Implement `backend/metrics.py`.
>
> Functions: `recall_at_k`, `mean_reciprocal_rank`, `p95_latency`, `cost_per_query`. **Pure functions — no I/O, no network, no globals.** Docstrings state the formula in plain English exactly as the spec words it, because that wording goes in the public README.
>
> Then `backend/evaluate.py`: run every registered lane over a split, compute all metrics, write `data/results.json` in the exact schema from spec §4. Flags: `--split`, `--out`, `--lanes`.
>
> **Tests must assert against fixtures I compute by hand**, not against the implementation's own output. Include an MRR case where gold is absent from top-k (must contribute 0, not crash) and a recall case with multiple gold chunks (any one hit counts).
>
> Run `python -m backend.evaluate --split test` and show me the output.
>
> **Acceptance check: lane 3 must beat lanes 1 and 2 on recall@10. If it doesn't, stop and debug fusion before we go further — that is a bug, not a finding.**

---

## Day 3 — The fine-tune

### Prompt 3.1 — Hard negative mining

> `/clear` first. Read `docs/02_EVALUATION_SPEC.md §3`.
>
> Implement `training/mine_negatives.py`. For each question **in the train split only**, retrieve top-20 via BM25 + dense, remove all gold chunks, sample 4 from the remainder as hard negatives. Output `data/train_pairs.jsonl` with `{query, passage, label}` where label is 1 for gold and 0 for negatives.
>
> **Add an explicit assertion that no test-split question appears in the output, and fail loudly if one does.** This is the single most likely way this project quietly becomes worthless.
>
> Print the pair count and the positive/negative ratio.
>
> Stop when done and show me the assertion.

### Prompt 3.2 — Training notebook

> Create `training/train_reranker.ipynb` for Google Colab free T4, plus `training/README.md` with exact reproduction steps.
>
> The notebook: install deps, mount or upload `train_pairs.jsonl`, fine-tune `cross-encoder/ms-marco-MiniLM-L-6-v2` using the exact configuration in `docs/02_EVALUATION_SPEC.md §3` (3 epochs, lr 2e-5, batch 16, warmup 10%, BCE loss, max_length 512, seed 42), evaluate on a held-out slice of train, push to `talalmohsin-98/plumbline-reranker-v1`.
>
> Generate the model card too: base model, training set size, hard-negative strategy, every hyperparameter, and a placeholder for the measured delta versus the stock reranker.
>
> Markdown cells explain **why** each hyperparameter is what it is — 3 epochs because 420 pairs will overfit beyond that, and so on. I need to defend these choices.
>
> Stop when done.

### Prompt 3.3 — Lane 6, judge, and calibration

> Wire lane 6: same `reranked.py` class, `RERANKER_TUNED` checkpoint. Register it.
>
> Implement `backend/judge.py` — groundedness scoring per `docs/02_EVALUATION_SPEC.md §2`. Sentence-split the generated answer, score each against retrieved context with Groq `llama-3.1-8b-instant` at `temperature=0`, mark the answer grounded only if every sentence is supported.
>
> Then `backend/calibrate_judge.py`: sample 30 judge verdicts, present each for my manual agree/disagree, compute agreement, write it into `results.json` under `judge_agreement`.
>
> Re-run `evaluate.py` with all six lanes and show me the full table.
>
> **Sanity check: if lane 6 scores above 0.95 on recall@10, stop — that almost certainly means test questions leaked into training. Verify the split before we report anything.**

---

## Day 4 — Graph and interface

### Prompt 4.1 — The LangGraph runner

> `/clear` first. Read `docs/01_ARCHITECTURE.md §5`.
>
> Implement `backend/graph.py`: a LangGraph `StateGraph` with the `ArenaState` TypedDict as specified. Six lane nodes fan out **concurrently** from START, join via the `merge_lane_results` reducer, then `generate` → `judge` → END.
>
> A lane that raises must write to `state["errors"]` and be excluded from the join — **one broken lane must never blank the scoreboard.** Bounded retry: max 1, failed lanes only.
>
> Comment the reducer explaining why concurrent writes to a shared dict need one and what would happen without it. This is the most interview-relevant code in the project.
>
> Test: mock three lanes to succeed and one to raise; assert the graph returns three results plus one error and does not throw.
>
> Stop when done and explain the fan-out/join to me.

### Prompt 4.2 — API

> Implement `backend/main.py` with every endpoint in `docs/01_ARCHITECTURE.md §6`.
>
> `GET /leaderboard` reads `data/results.json` from disk and returns it — **no model loading, no database call, no computation.** It must respond on a cold start in under 500 ms.
>
> `POST /race` streams SSE, emitting one event per lane as it completes, so lanes visibly finish at different times. Buffering this into a single response defeats its entire purpose.
>
> Models load at import into module-level singletons. CORS for the Vercel origin. `/health` cheap enough for a 30-minute cron ping.
>
> Stop when done.

### Prompt 4.3 — Frontend

> Read `docs/01_ARCHITECTURE.md §9` on cold start. Build the React 19 + Vite + Tailwind frontend in `frontend/`.
>
> Pages: **Leaderboard** (renders `/leaderboard` immediately on mount — this is what a visitor sees first and it must never show a spinner), **Race** (six columns, SSE-streamed, each lane showing its stage trace as it works), **Methodology** (metric definitions and model versions from `/methodology`), **Upload**.
>
> Requirements: six seeded example questions — never show an empty input box. Real error states with specific messages for rate-limited, sleeping backend, unsupported file, no results. **No text anywhere reading "demo", "prototype", "WIP", or "coming soon".** One footer line only.
>
> **Every displayed number comes from an API response.** Do not hardcode a single metric, not even temporarily.
>
> Design: clean and instrument-like, not playful. This is a measurement tool.
>
> Stop when done and show me the leaderboard render path.

---

## Day 5 — Deploy and finish

### Prompt 5.1 — Deployment

> `/clear` first. Create:
>
> - `Dockerfile` for Hugging Face Spaces — Python 3.11 slim, models pre-downloaded at **build** time not runtime, uvicorn on port 7860, non-root user
> - `README_SPACE.md` with the HF Spaces frontmatter block
> - `vercel.json` for the frontend, with the API base URL as an env var
> - `.github/workflows/ci.yml` — `pytest` plus `npm run build` on push
> - `.github/workflows/keepalive.yml` — cron ping `/health` every 30 minutes
>
> **Railway is not available (expired trial). Do not generate any Railway configuration.**
>
> Stop when done and give me the exact deploy commands for both.

### Prompt 5.2 — README

> Use `docs/04_README_TEMPLATE.md` as the structure. Fill every `{N}` placeholder from the real values in `data/results.json` — do not invent, estimate, or round anything.
>
> Write the "Headline finding" from what the numbers actually show. Write the "What didn't work" section honestly; if the fine-tuned reranker underperformed, say so and give a hypothesis.
>
> Keep the "Known limitations" section complete and do not soften any item.
>
> Then update the HF model card with the measured delta, and publish `goldset.jsonl` as `talalmohsin-98/plumbline-goldset-v1` with a dataset card describing the verification protocol and rejection rate.
>
> Stop when done and show me the README.

---

## Ownership prompt — run at the end of every day

> Do not write any code. For everything we built today:
>
> 1. List every file created or changed and state each one's responsibility in one sentence.
> 2. List every parameter, threshold, or model choice you made that I did not explicitly specify, and the reasoning behind each.
> 3. Name the three places most likely to break in production and why.
> 4. Quiz me: ask me five questions about today's code that a senior engineer would ask in an interview. Wait for my answers, then tell me which were wrong and why.
>
> Be blunt about the ones I get wrong. Passing me on a weak answer today means failing an interview later.

---

## Guardrail prompt — if it starts drifting

> Stop. Re-read `docs/00_PRD.md §6` — the non-goals. Tell me whether anything you have added in this session falls into that list, and remove it if so. Then re-read `docs/03_BUILD_PLAN.md` and tell me which day we are on and whether its acceptance criteria are met.

---

## End-of-day close

> Update the "Next task" checkbox at the bottom of `CLAUDE.md` with tomorrow's first task. Then stage and commit today's work in **separate logical commits** with imperative messages — not one commit for the whole day. Show me the log before pushing.
