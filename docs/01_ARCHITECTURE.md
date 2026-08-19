# Plumbline — Technical Architecture

---

## 1. Stack

| Layer | Choice | Why this and not the obvious alternative |
|---|---|---|
| Runtime | Python 3.11 | 3.13 wheels are still patchy for `sentence-transformers` on slim images |
| Backend | FastAPI + Uvicorn | SSE streaming for the race view; already known to the author |
| Orchestration | LangGraph `StateGraph` | Needed for parallel fan-out/join, not decoration — see §4 |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) | bge-large is 1024-dim and ~3× slower on CPU; free tier has no GPU |
| Sparse | `rank_bm25` (BM25Okapi), disk-cached | In-process, no service to run |
| Vector store | Supabase PostgreSQL + `pgvector` (free tier) | Author has shipped pgvector before; Qdrant Cloud is the drop-in alternative if row limits bite |
| Reranker (stock) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M) | bge-reranker-base is 278M — too slow for CPU inference across six lanes |
| Reranker (tuned) | Same base, fine-tuned — see `02_EVALUATION_SPEC.md` | Same architecture keeps lanes 4 vs 6 a clean comparison |
| Generation | Groq `llama-3.1-8b-instant` | Free tier, fast; used only for gold-set drafting and groundedness judging |
| Frontend | React 19 + Vite + Tailwind | Author's existing stack |
| Backend host | Hugging Face Spaces (Docker) | Free, permanent, no card. **Railway is unavailable — trial expired** |
| Frontend host | Vercel → craftwyre.com | Free, custom domain |
| Training | Google Colab free T4 | CPU fine-tuning would consume a full build day |
| CI | GitHub Actions | `pytest` + `npm run build` on push |

**Binding constraint: no paid service and no trial-based service anywhere in the stack.** Every dependency must have a permanently free tier.

## 2. System shape

```
┌─────────────────────────────────────────────────────────┐
│  React SPA (Vercel → craftwyre.com)                     │
│  Leaderboard · Race view · Corpus upload · Methodology  │
└────────────┬────────────────────────────────────────────┘
             │  static results.json (instant)  +  SSE /race
┌────────────▼────────────────────────────────────────────┐
│  FastAPI (HF Spaces, Docker)                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  LangGraph StateGraph — parallel lane runner      │  │
│  └───────────────────────────────────────────────────┘  │
│  ingest · goldset · lanes/ · metrics · judge            │
└──────┬──────────────────────┬───────────────────────────┘
       │                      │
┌──────▼────────┐    ┌────────▼─────────┐    ┌──────────────┐
│ Supabase      │    │ Local models     │    │ Groq API     │
│ pgvector      │    │ bge-small        │    │ llama-3.1-8b │
│ (demo corpus) │    │ MiniLM rerankers │    │              │
└───────────────┘    └──────────────────┘    └──────────────┘
```

## 3. Module layout

```
plumbline/
├── backend/
│   ├── main.py              FastAPI app, routes, SSE
│   ├── graph.py             LangGraph StateGraph — fan-out/join
│   ├── ingest.py            load → chunk → embed → store
│   ├── goldset/
│   │   ├── generate.py      LLM drafts question→chunk pairs
│   │   └── verify.py        CLI verification workflow
│   ├── lanes/
│   │   ├── base.py          Lane ABC — the single interface
│   │   ├── bm25.py          Lane 1
│   │   ├── dense.py         Lane 2
│   │   ├── hybrid.py        Lane 3 (RRF fusion)
│   │   ├── reranked.py      Lanes 4 & 6 (base vs tuned checkpoint)
│   │   └── hyde.py          Lane 5
│   ├── retrieval/
│   │   ├── dense_store.py   pgvector queries
│   │   ├── bm25_index.py    BM25 index build + cache
│   │   ├── fusion.py        Reciprocal Rank Fusion
│   │   └── rerank.py        Cross-encoder scoring
│   ├── metrics.py           recall@k, MRR, latency, cost
│   ├── judge.py             Groundedness scoring
│   ├── config.py            Pydantic Settings — all env
│   └── tests/
├── training/
│   ├── mine_negatives.py    Hard negative mining
│   ├── train_reranker.py    HF Trainer — run on Colab
│   └── README.md            Exact repro steps
├── frontend/src/
│   ├── pages/               Leaderboard · Race · Upload · Methodology
│   ├── components/          LaneCard · MetricTable · StageTrace
│   └── api.js
├── data/
│   ├── demo_corpus/
│   ├── goldset_screened.jsonl  Every draft, every rule score, kept and dropped
│   ├── goldset.jsonl        LLM-screened, human-audited — the ground truth
│   ├── audit_results.json   Screener–human agreement, overall and per verdict
│   └── results.json         Precomputed leaderboard (committed)
├── CLAUDE.md
└── docs/
```

## 4. The lane interface

Every lane implements one method. This is the architectural spine — it is what makes six pipelines comparable rather than six separate programs.

```python
class Lane(ABC):
    id: str
    label: str
    cost_per_query_usd: float   # measured, not estimated

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> LaneResult: ...

@dataclass
class LaneResult:
    chunks: list[Chunk]          # ranked, best first
    latency_ms: float
    stages: list[StageTrace]     # what each stage did — powers the inspector
    tokens_used: int
```

Adding a seventh lane must require **only** a new file in `lanes/` and one registry entry. If it requires touching `graph.py` or `metrics.py`, the abstraction is wrong.

## 5. LangGraph state

The graph exists for genuine parallelism: six lanes fan out concurrently, join, then a single groundedness pass runs over all six answers. Conditional edges handle per-lane failure without killing the run.

```python
class ArenaState(TypedDict):
    query: str
    corpus_id: str
    lane_results: Annotated[dict[str, LaneResult], merge_lane_results]
    answers: dict[str, str]
    groundedness: dict[str, float]
    errors: dict[str, str]
    retries: dict[str, int]
```

```
              ┌→ lane_bm25 ──────┐
              ├→ lane_dense ─────┤
   START ──── ├→ lane_hybrid ────┼──→ join ──→ generate ──→ judge ──→ END
              ├→ lane_reranked ──┤              │              │
              ├→ lane_hyde ──────┤              └──── retry ───┘
              └→ lane_tuned ─────┘                 (max 1, failed lanes only)
```

`merge_lane_results` is a reducer, so lanes write to the same dict concurrently without clobbering each other. A lane that raises writes to `errors` and is excluded from the join rather than failing the run — **one broken lane must never blank the scoreboard.**

## 6. API contract

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/leaderboard` | Precomputed demo-corpus results (served from disk, no compute) |
| `GET` | `/lanes` | Lane metadata: id, label, stage description, cost |
| `POST` | `/race` | `{query, corpus_id}` → **SSE stream**, one event per lane as it finishes |
| `POST` | `/corpus/upload` | Multipart files → `{corpus_id, chunk_count}` |
| `POST` | `/corpus/{id}/goldset` | Draft a gold set for an uploaded corpus |
| `POST` | `/corpus/{id}/evaluate` | Run all lanes over the gold set → full metrics |
| `GET` | `/methodology` | Metric definitions and model versions, served to the UI |

`/race` streams because the visible point of the race view is that lanes finish at *different* times. Buffering it into one response destroys the entire effect.

## 7. Storage model

| Data | Where | Lifetime |
|---|---|---|
| Demo corpus chunks + embeddings | Supabase pgvector | Permanent |
| Demo gold set | `data/goldset.jsonl`, committed | Permanent |
| Demo results | `data/results.json`, committed | Permanent, regenerated by CI |
| Uploaded corpus | In-process memory, keyed by `corpus_id` | 2h TTL, evicted on restart |
| Fine-tuned reranker | HF Hub, downloaded at boot | Cached in the Space |

Uploads are deliberately in-memory. Persisting them means auth, quotas, and abuse handling — a day of work that proves nothing about retrieval engineering. The limitation gets documented in the README rather than hidden.

## 8. Performance budget

Free-tier CPU is the binding constraint. Targets per single-query race:

| Stage | Budget |
|---|---|
| Embedding (bge-small, 1 query) | < 40 ms |
| pgvector top-50 | < 120 ms |
| BM25 top-50 (cached index) | < 30 ms |
| RRF fusion | < 5 ms |
| Cross-encoder rerank, **top-20 only** | < 600 ms |
| HyDE (1 Groq call) | < 700 ms |
| **Full six-lane race, wall clock** | **< 3 s** |

Rerank depth is capped at 20 — not 50 — because cross-encoder cost is linear in depth and this is the single largest CPU item in the system.

## 9. Cold start

The #1 thing that makes a free-tier project read as a toy. Three defences:

1. `/leaderboard` serves committed JSON from disk. No model load, no DB call. **Paints in < 500 ms even on a cold Space.**
2. Models load at import time into module-level singletons, never per request.
3. A GitHub Actions cron pings `/health` every 30 minutes to keep the Space warm.

The frontend renders the leaderboard first and lazily enables the race view once `/health` returns green. A visitor never watches a spinner.

## 10. Configuration

All config via `pydantic-settings`, no literals in code:

```
DATABASE_URL, GROQ_API_KEY, HF_TOKEN
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_BASE=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TUNED=talalmohsin-98/plumbline-reranker-v1
RRF_K=60
RERANK_DEPTH=20
RETRIEVE_DEPTH=50
```

## 11. Decisions worth defending in an interview

These are the questions a reviewer will actually ask. Each answer must be understood, not recited.

**Why RRF with k=60?** RRF scores a document as `Σ 1/(k + rank)` across ranked lists. k controls how sharply top ranks dominate: small k lets rank 1 overwhelm everything; large k flattens the lists toward a plain vote. 60 is the value from the original Cormack et al. paper and is a reasonable default — *and it is a parameter this project can measure rather than assume.* If time permits, sweep k ∈ {10, 30, 60, 100} and report it.

**Why MiniLM-L-6 over bge-reranker-base?** 22M vs 278M parameters. On free-tier CPU across six lanes, bge-reranker-base blows the 3-second budget. It would likely score better; that is a documented tradeoff, not an oversight.

**Why rerank depth 20?** Cross-encoder cost is linear in candidates. Rerank@20 captures most of the achievable gain when recall@50 of the fused list is already high — verify this on your own data and report the number.

**Why bge-small over bge-large?** 384 vs 1024 dimensions; roughly 3× faster on CPU, smaller index. Costs some retrieval quality. Same class of tradeoff, same honest documentation.

**Why is the gold set audited rather than simply LLM-judged?** Because LLM-generated labels evaluated by an LLM measures agreement between two language models, not retrieval quality. The screener does the labour — 350 drafts scored against four drop rules, one call per rule — but a screener nobody checks is exactly that circular setup. So a human judges a 40-row sample **blind** (the screener's verdict is hidden until after the keypress, or what gets measured is anchoring) and **stratified across all three verdicts** (auditing only the keeps is structurally blind to good questions wrongly dropped, since those never appear in the finished file). The published agreement rate is what makes the numbers mean anything, and `02_EVALUATION_SPEC.md` §1 states what this buys and what it costs versus reading every draft by hand.
