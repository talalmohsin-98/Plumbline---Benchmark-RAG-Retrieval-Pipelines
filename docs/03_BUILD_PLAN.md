# Plumbline — 5-Day Build Plan

**Ship date: Friday.** Resume rewrite: Saturday.

A day is "done" only when its acceptance criteria pass. Do not carry an unmet criterion forward — cut scope from a later day instead.

---

## Day 1 — Corpus and gold set

The least enjoyable day and the one everything depends on. **Do not reorder it.** A benchmark with a sloppy gold set is worse than no benchmark, because it produces confident numbers that are wrong.

**Build**
- Project skeleton, `config.py`, Supabase pgvector table + index
- `ingest.py`: load PDF/TXT/MD → chunk (512 tokens, 64 overlap) → embed with bge-small → store with stable `chunk_id`
- Demo corpus: 4–6 documents, ~1000–1500 chunks. **Any domain that isn't legal** — public-sector service docs, an open textbook, API documentation. Generic corpus, generic tool.
- `goldset/generate.py` — draft 150 pairs
- `goldset/verify.py` — a terminal loop: print question + chunk, take `k`/`f`/`d`, append to `goldset.jsonl`
- **Sit down and verify until 120 keeps.** Expect 2–3 hours. This is the job.

**Done when**
- `goldset.jsonl` has ≥ 120 human-verified rows
- Re-running ingest produces identical `chunk_id`s
- Train/test split written to disk with seed 42

---

## Day 2 — Lanes and metrics

**Build**
- `lanes/base.py` — the `Lane` ABC and `LaneResult`
- Lanes 1–5 (lane 6 loads the same class with a different checkpoint on Day 3)
- `retrieval/fusion.py` — RRF, k=60
- `metrics.py` — recall@5, recall@10, MRR, p95 latency, cost
- `evaluate.py` — run every lane over the test split → `results.json`
- Tests for fusion and every metric, using hand-computed fixtures

**Done when**
- `python -m backend.evaluate --split test` emits valid `results.json` for five lanes
- Metric tests pass against fixtures you computed by hand — **not against the code's own output**
- Lane 3 beats lanes 1 and 2 on recall@10. *If it doesn't, fusion is broken; stop and fix it before continuing.*

---

## Day 3 — The fine-tune

The day that closes the transformer gap. Machine time is short; setup time is not.

**Build**
- `training/mine_negatives.py` → ~420 pairs from the **train split only**
- `training/train_reranker.ipynb` on Colab T4 — config per the evaluation spec
- Push to `talalmohsin-98/plumbline-reranker-v1` with a full model card
- Wire lane 6, re-run evaluation
- Groundedness judge + the 30-sample hand-calibration

**Done when**
- The model is public on HF Hub with a complete card
- `results.json` carries all six lanes on the test split
- The lane 4 → lane 6 delta is recorded **in whatever direction it went**
- Judge agreement percentage is recorded

**Guard against the classic mistake:** if lane 6 scores above ~0.95, you have almost certainly leaked test questions into training. Check the split before celebrating.

---

## Day 4 — Graph and interface

**Build**
- `graph.py` — parallel StateGraph with the `merge_lane_results` reducer, per-lane error capture, bounded retry
- `POST /race` streaming SSE, one event per lane completion
- Frontend: Leaderboard (from static JSON), Race view (six live columns), Methodology page, Upload flow
- Six seeded example questions — **never show a recruiter an empty input box**

**Done when**
- Leaderboard paints in under 2 seconds on a hard refresh
- A race streams six lanes finishing at visibly different times
- Killing one lane deliberately still returns the other five

---

## Day 5 — Deploy and finish

**Build**
- Dockerfile → HF Spaces; models loaded once at import
- Frontend → Vercel → craftwyre.com
- Gold set published as an HF dataset
- README from the template: results table, methodology, **Known limitations**, repro steps
- GitHub Actions: `pytest` + `npm run build`; cron ping `/health` every 30 min
- Screenshots in the README
- **Read every file you are about to make public and make sure you can defend it**

**Done when**
- All six PRD success criteria pass
- A cold load of craftwyre.com shows numbers in under 2 seconds
- Repo, dataset, and model are all public and cross-linked

---

## Cut list

If you fall behind, cut in this order and no other:

1. Upload-your-own-corpus flow (ship demo corpus only)
2. HyDE lane (five lanes still tells the story)
3. Race animation (static comparison still works)
4. CI workflow

**Never cut:** gold-set verification, the fine-tuned reranker, the train/test split, the Known Limitations section.

---

## Daily discipline

- Commit at least three times a day with real messages. A repo whose entire history is one "initial commit" on Friday tells its own story.
- End each day by writing tomorrow's first task in `CLAUDE.md`.
- If a day runs 3+ hours over, cut from the list above rather than borrowing from tomorrow. Day 5 has no slack.
