# CLAUDE.md — working agreement

Engineering conventions for this repository, written for AI coding assistants and human contributors alike. This file is committed deliberately: it documents how the project is built and what standards code must meet before it lands.

---

## What this project is

Plumbline benchmarks six retrieval pipelines against each other on an LLM-screened, human-audited gold set and publishes retrieval quality, groundedness, latency, and cost for each.

Full specifications: `docs/00_PRD.md`, `docs/01_ARCHITECTURE.md`, `docs/02_EVALUATION_SPEC.md`, `docs/03_BUILD_PLAN.md`. **Read `02_EVALUATION_SPEC.md` before touching anything that produces a number.**

## The rule that overrides everything

**Every number published by this project must be reproducible from this repository.** No hardcoded metrics, no placeholder values that ship, no figures typed into the frontend. If a number appears on the site, a command in the README regenerates it.

## Conventions

**Python**
- 3.11, type hints on every function signature, `ruff` clean
- Pydantic models for all API boundaries; `pydantic-settings` for config
- No configuration literals in application code — everything through `config.py`
- Module-level model singletons; never load a model inside a request handler

**Structure**
- A new retrieval lane must require only a new file in `lanes/` plus a registry entry. Needing to edit `graph.py` or `metrics.py` means the abstraction broke — fix the abstraction, don't special-case.
- `metrics.py` is pure: data in, numbers out. No I/O, no network, no globals.

**Frontend**
- React function components with hooks; Tailwind utilities, no separate CSS files
- All displayed values come from API responses or `results.json`. Never hardcode a metric for "the demo".

**Tests**
- Metric tests assert against **hand-computed fixtures**, not against the implementation's own output. A metric test that just records current behaviour catches nothing.
- External services (Groq, Supabase, HF) are stubbed. `pytest` must pass with no network and no API keys.

**Commits**
- Imperative, scoped: `feat(lanes): add RRF fusion lane`, `fix(metrics): correct MRR when gold absent from top-k`
- Small and frequent. One commit for the whole project is its own signal.

## Explicit non-goals

Do not add, even if it seems helpful: user accounts or auth, a chat interface, persistent storage of uploads, nDCG, multi-turn conversation, paid tiers, waitlists, or any dependency without a permanently free tier. These are argued in `docs/00_PRD.md §6` — reopening them requires editing that document first.

**Railway is unavailable** (expired trial). Backend deploys to Hugging Face Spaces, frontend to Vercel.

## Honesty requirements

- Report measured results in whichever direction they go. A fine-tuned model that underperforms is a finding to publish, not a bug to hide.
- Keep the README's "Known limitations" section accurate and current. Understating a limitation is a defect.
- Never present LLM-drafted gold labels as verified. The gold set is **LLM-screened and human-adjudicated**: every one of its 115 rows carries the author's verdict, but adjudicating a screener's call against four rules is not verifying a label from scratch. Rows carry `status: "screened"`, and "human-verified" must not appear anywhere describing it. Wherever the gold set is characterised, the screener's agreement rate goes with it.

## Author's standard for generated code

Code in this repository is written with AI assistance, and the author reviews and owns every file before it is committed. Concretely, that means:

- Prefer an explicit, readable implementation over a clever one. This code is read by reviewers, not just executed.
- Where a parameter is chosen (RRF k, rerank depth, chunk size, learning rate), add a comment stating **why that value** and what the alternative was. Those comments are the ones that get asked about.
- Do not introduce a library to solve something a dozen lines of standard library handles.
- If an implementation cannot be explained in two sentences, simplify it or document it — do not merge it as-is.

## Next task

_Update at the end of each working day._

- [x] Day 1: ingest pipeline + gold set. **Closed at 115 rows, 100% author-adjudicated**, split 80/35 with no gold chunk on both sides.
- [ ] Day 2: the six retrieval lanes and `metrics.py`. Read `docs/02_EVALUATION_SPEC.md` §2 first.

Deferred from Day 1, deliberately:
- The blind audit queue is not redacted -- `data/audit_queue_*.jsonl` carries the
  screener verdicts the pass was blind to. Blindness held at the terminal
  (tested) and by procedure at the file. Making it structural conflicts with
  `--queue-in`, which needs `screen_verdict` to compute `agreed`.
