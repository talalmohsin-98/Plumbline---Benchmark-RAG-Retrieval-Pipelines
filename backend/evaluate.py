"""Run every lane over a split and write `data/results.json`.

The command that produces every number on the site. CLAUDE.md's overriding
rule is that a published number must be reproducible from this repository, so
this file has one job: read the gold split, run the lanes, compute the metrics
in `metrics.py`, and write the result out with enough provenance stamped on it
that a stranger can tell what produced it.

    python -m backend.evaluate --split test

Nothing in here names a lane. Lanes come from `backend.lanes.REGISTRY`, which
is the property that makes "a new lane is one new file plus one registry entry"
true rather than aspirational.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.config import get_settings
from backend.lanes import REGISTRY, build_lanes
from backend.lanes.base import Lane
from backend.metrics import (
    cost_per_query_usd,
    mean_reciprocal_rank,
    p95_latency_ms,
    recall_at_k,
)
from backend.retrieval.corpus import Corpus

DEFAULT_SPLITS = {"test": Path("data/test.jsonl"), "train": Path("data/train.jsonl")}
DEFAULT_OUT = Path("data/results.json")
DEFAULT_AUDIT = Path("data/audit_results.json")

# Retrieval depth for scoring. Every lane returns this many chunks and every
# metric is computed within it. It must be at least the largest k reported
# (10), and it is exactly 10 rather than 50 so that recall@10 and MRR@10 are
# measured on the list a caller would actually receive.
SCORE_DEPTH = 10


# --------------------------------------------------------------------------
# Gold split
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldRow:
    qid: str
    question: str
    gold_chunk_ids: list[str]


def load_split(path: Path) -> list[GoldRow]:
    """Read a split file into gold rows, in qid order."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m backend.goldset.split` first."
        )
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    missing = [r.get("qid") for r in rows if not r.get("gold_chunk_ids")]
    if missing:
        raise ValueError(f"{len(missing)} row(s) in {path} have no gold_chunk_ids: {missing[:5]}")
    return sorted(
        (
            GoldRow(
                qid=row["qid"],
                question=row["question"],
                gold_chunk_ids=list(row["gold_chunk_ids"]),
            )
            for row in rows
        ),
        key=lambda r: r.qid,
    )


# --------------------------------------------------------------------------
# Running one lane
# --------------------------------------------------------------------------


@dataclass
class LaneRun:
    """Everything one lane produced over one split, before aggregation."""

    lane_id: str
    label: str
    retrieved: list[list[str]] = field(default_factory=list)
    gold: list[list[str]] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)

    def metrics(self, input_rate: float, output_rate: float) -> dict[str, Any]:
        """Aggregate into the published fields.

        Latency percentiles are computed over successful queries only -- a
        query that raised has no meaningful wall-clock time. Recall and MRR
        are computed over *every* question in the split, with a failed query
        contributing an empty result list and therefore a miss. That is the
        honest treatment: a lane that could not answer did not answer, and
        scoring it over only the questions it survived would flatter exactly
        the lane most likely to fail.
        """
        return {
            "id": self.lane_id,
            "label": self.label,
            "recall_at_5": round(recall_at_k(self.retrieved, self.gold, 5), 4),
            "recall_at_10": round(recall_at_k(self.retrieved, self.gold, 10), 4),
            "mrr": round(mean_reciprocal_rank(self.retrieved, self.gold, SCORE_DEPTH), 4),
            "p95_latency_ms": (
                round(p95_latency_ms(self.latencies_ms), 1) if self.latencies_ms else None
            ),
            "cost_per_query_usd": round(
                cost_per_query_usd(
                    self.prompt_tokens, self.completion_tokens, input_rate, output_rate
                ),
                8,
            ),
            "questions": len(self.retrieved),
            "queries_failed": len(self.failures),
            "mean_latency_ms": (
                round(sum(self.latencies_ms) / len(self.latencies_ms), 1)
                if self.latencies_ms
                else None
            ),
        }


def run_lane(lane: Lane, rows: list[GoldRow], *, verbose: bool = True) -> LaneRun:
    """Run one lane over every question in the split.

    The lane is warmed first and the warmup is not timed. Without it the first
    query pays for a BM25 index build or a cross-encoder load, and at n=35 that
    single outlier would *be* the reported p95.
    """
    lane.warm()
    run = LaneRun(lane_id=lane.id, label=lane.label)
    for index, row in enumerate(rows, start=1):
        run.gold.append(row.gold_chunk_ids)
        try:
            result = lane.retrieve(row.question, k=SCORE_DEPTH)
        except Exception as exc:  # one bad query must not lose the other 34
            run.retrieved.append([])
            run.failures.append((row.qid, f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"    {row.qid} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        run.retrieved.append(result.chunk_ids)
        run.latencies_ms.append(result.latency_ms)
        run.prompt_tokens.append(result.prompt_tokens)
        run.completion_tokens.append(result.completion_tokens)
        if verbose and index % 10 == 0:
            print(f"    {index}/{len(rows)}", file=sys.stderr)
    return run


# --------------------------------------------------------------------------
# The lane-3 gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """Whether fusion did what fusion is supposed to do."""

    checked: bool
    status: str  # "pass" | "tie" | "fail" | "skipped"
    detail: str

    @property
    def blocking(self) -> bool:
        """A genuine regression stops the build. A tie is recorded, not hidden."""
        return self.status == "fail"

    def as_dict(self) -> dict[str, Any]:
        return {"checked": self.checked, "status": self.status, "detail": self.detail}


def check_fusion_gate(lane_metrics: list[dict[str, Any]]) -> GateResult:
    """Lane 3 must not lose to lanes 1 and 2 on recall@10.

    From `docs/03_BUILD_PLAN.md`: *if it doesn't, fusion is broken; stop and
    fix it before continuing.* The check lives here rather than in a test
    because it is a claim about the data, not about the code -- the fusion
    unit tests pass happily on a fused list that is correct and useless.

    Three states, not two, and the third one was earned. On the 35-row test
    split hybrid_rrf ties BM25 at exactly 30/35 while beating it on recall@5
    (0.83 vs 0.69) and MRR (0.60 vs 0.54). That tie was investigated before it
    was accepted, and the investigation is reproducible with
    `python -m backend.sweep_rrf`:

    - train (n=80):  fused 68/80 vs bm25 62/80, dense 57/80 -- fusion wins by 6
    - all   (n=115): fused 98/115 vs bm25 92/115, dense 86/115 -- wins by 6
    - test  (n=35):  30/35 each, and they are *different* questions. BM25 gets
      q123, fusion gets q032. One question either way.

    So a tie at this n is a coincidence of totals, not a ceiling and not a
    broken fusion, and it is recorded as `tie` rather than laundered into a
    pass. A `fail` -- hybrid strictly below either arm -- still stops the run,
    because that is the shape a real fusion bug takes.

    Deliberately NOT resolved by tuning `rrf_k` on the test split. k was swept
    on train only; no cell in {1,5,10,20,30,60,100} x depth {10,20,30,50}
    turns the test tie into a win, and picking one that did would be fitting
    the benchmark to its own answer key.
    """
    by_id = {m["id"]: m for m in lane_metrics}
    required = ("bm25", "dense", "hybrid_rrf")
    if not all(lane_id in by_id for lane_id in required):
        present = sorted(by_id)
        return GateResult(
            checked=False,
            status="skipped",
            detail=f"not checked: needs {list(required)}, this run had {present}",
        )
    hybrid = by_id["hybrid_rrf"]["recall_at_10"]
    bm25 = by_id["bm25"]["recall_at_10"]
    dense = by_id["dense"]["recall_at_10"]
    best_arm = max(bm25, dense)
    if hybrid > best_arm:
        status = "pass"
    elif hybrid == best_arm:
        status = "tie"
    else:
        status = "fail"
    return GateResult(
        checked=True,
        status=status,
        detail=(
            f"{status.upper()}: recall@10 hybrid_rrf {hybrid:.4f} vs "
            f"bm25 {bm25:.4f}, dense {dense:.4f}"
        ),
    )


# --------------------------------------------------------------------------
# Assembling the results file
# --------------------------------------------------------------------------


def read_audit(path: Path) -> dict[str, Any]:
    """Fold the audit summary into results.json rather than retyping it.

    The gold set's provenance is published next to the numbers it produced,
    and there is exactly one copy of those figures on disk.
    """
    if not path.exists():
        return {}
    audit = json.loads(path.read_text(encoding="utf-8"))
    return {
        "adjudicated": audit.get("sampled"),
        "agreement_screener": audit.get("agreement_overall"),
        "agreement_second_opinion": audit.get("agreement_with_second"),
        "agreement_on_drops": audit.get("by_verdict", {}).get("drop", {}).get("agreement"),
        "blind": audit.get("by_mode", {}).get("blind", {}).get("audited"),
        "revealed": audit.get("by_mode", {}).get("revealed", {}).get("audited"),
    }


def build_results(
    runs: list[LaneRun],
    *,
    split_name: str,
    rows: list[GoldRow],
    corpus: Corpus,
    gate: GateResult,
    ablation: dict[str, Any] | None = None,
    audit_path: Path = DEFAULT_AUDIT,
) -> dict[str, Any]:
    """Assemble the published results document.

    Deliberately does not invent the Day 3 fields. `groundedness` and
    `judge_agreement` are absent rather than null-or-zero, because a zero in a
    published table reads as a measurement and this is an absence.
    """
    settings = get_settings()
    train_path = DEFAULT_SPLITS["train"]
    test_path = DEFAULT_SPLITS["test"]
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": split_name,
        "score_depth": SCORE_DEPTH,
        "corpus": {
            "name": corpus.corpus_id,
            "docs": len({c.source_doc for c in corpus.chunks.values()}),
            "chunks": len(corpus.chunks),
        },
        "goldset": {
            "total": _count_lines(train_path) + _count_lines(test_path),
            "train": _count_lines(train_path),
            "test": _count_lines(test_path),
            "scored_here": len(rows),
            "built_by": "llm-screened, adjudicated by the author with AI assistance",
            "status": "screened",
            "audit": read_audit(audit_path),
        },
        "models": {
            "embedding": settings.embedding_model,
            "reranker_base": settings.reranker_base,
            "hyde_generator": settings.groq_model,
        },
        "parameters": {
            "rrf_k": settings.rrf_k,
            "retrieve_depth": settings.retrieve_depth,
            "rerank_depth": settings.rerank_depth,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "bge_query_prefix_enabled": True,
        },
        "pricing": {
            "groq_input_usd_per_million": settings.groq_input_usd_per_million,
            "groq_output_usd_per_million": settings.groq_output_usd_per_million,
            "checked_on": settings.groq_rates_checked_on,
            "note": (
                "Vendor list price, not a measurement. The token counts it "
                "multiplies are measured from each API response."
            ),
        },
        "fusion_gate": gate.as_dict(),
        "methodology_notes": {"bge_query_prefix": ablation} if ablation else {},
        "lanes": [
            run.metrics(
                settings.groq_input_usd_per_million,
                settings.groq_output_usd_per_million,
            )
            for run in runs
        ],
        "failures": {
            run.lane_id: [{"qid": qid, "error": error} for qid, error in run.failures]
            for run in runs
            if run.failures
        },
    }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# --------------------------------------------------------------------------
# The bge query-prefix ablation
# --------------------------------------------------------------------------


def run_prefix_ablation(corpus: Corpus, rows: list[GoldRow]) -> dict[str, Any]:
    """Measure what bge's query instruction is worth on this corpus.

    A methodology note, deliberately not a sixth lane. It answers one question
    -- does the documented asymmetric prefix help here -- and adding a lane
    for it would put a configuration variant on a leaderboard of architectures
    and imply they are the same kind of comparison.
    """
    from backend.lanes.dense import DenseLane

    with_prefix = run_lane(DenseLane(corpus, use_query_prefix=True), rows, verbose=False)
    without_prefix = run_lane(DenseLane(corpus, use_query_prefix=False), rows, verbose=False)

    def summarise(run: LaneRun) -> dict[str, float]:
        return {
            "recall_at_5": round(recall_at_k(run.retrieved, run.gold, 5), 4),
            "recall_at_10": round(recall_at_k(run.retrieved, run.gold, 10), 4),
            "mrr": round(mean_reciprocal_rank(run.retrieved, run.gold, SCORE_DEPTH), 4),
        }

    on = summarise(with_prefix)
    off = summarise(without_prefix)
    return {
        "lane": "dense",
        "with_prefix": on,
        "without_prefix": off,
        "delta": {key: round(on[key] - off[key], 4) for key in on},
        "prefix": get_settings().bge_query_prefix,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def format_table(lane_metrics: list[dict[str, Any]]) -> str:
    """Render the leaderboard for the terminal."""
    header = (
        f"{'lane':<34} {'recall@5':>9} {'recall@10':>10} {'MRR':>7} "
        f"{'p95 ms':>9} {'$/query':>10} {'fail':>5}"
    )
    lines = [header, "-" * len(header)]
    for metric in lane_metrics:
        p95 = f"{metric['p95_latency_ms']:.1f}" if metric["p95_latency_ms"] is not None else "-"
        lines.append(
            f"{metric['label']:<34} {metric['recall_at_5']:>9.4f} "
            f"{metric['recall_at_10']:>10.4f} {metric['mrr']:>7.4f} "
            f"{p95:>9} {metric['cost_per_query_usd']:>10.6f} "
            f"{metric['queries_failed']:>5}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.evaluate",
        description="Run every lane over a gold split and write results.json.",
    )
    parser.add_argument("--split", choices=sorted(DEFAULT_SPLITS), default="test")
    parser.add_argument("--split-file", type=Path, default=None, help="override the split path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpus-id", default="demo")
    parser.add_argument(
        "--lanes",
        default=None,
        help=f"comma-separated subset of {sorted(REGISTRY)} (default: all)",
    )
    parser.add_argument(
        "--skip-remote",
        action="store_true",
        help="skip lanes that call an external API (HyDE), for when the Groq budget is gone",
    )
    parser.add_argument(
        "--prefix-ablation",
        action="store_true",
        help="also measure the dense lane with and without the bge query prefix",
    )
    parser.add_argument("--limit", type=int, default=None, help="first N questions only, for smoke")
    args = parser.parse_args(argv)

    if args.split == "train":
        print(
            "WARNING: scoring the TRAIN split. No number from this run is publishable -- "
            "the leaderboard comes from the test split only.",
            file=sys.stderr,
        )

    split_path = args.split_file or DEFAULT_SPLITS[args.split]
    rows = load_split(split_path)
    if args.limit:
        rows = rows[: args.limit]
    print(f"split      {args.split}  ({len(rows)} questions from {split_path})", file=sys.stderr)

    lane_ids = args.lanes.split(",") if args.lanes else None
    if args.skip_remote:
        from backend.lanes import REMOTE_LANES

        lane_ids = [i for i in (lane_ids or list(REGISTRY)) if i not in REMOTE_LANES]

    corpus = Corpus(args.corpus_id)
    try:
        lanes = build_lanes(corpus, lane_ids)
        print(f"corpus     {corpus.corpus_id}  ({len(corpus.chunks)} chunks)", file=sys.stderr)

        runs: list[LaneRun] = []
        for lane in lanes:
            print(f"\n  {lane.id}  ({lane.label})", file=sys.stderr)
            runs.append(run_lane(lane, rows))

        ablation = run_prefix_ablation(corpus, rows) if args.prefix_ablation else None
    finally:
        corpus.close()

    settings = get_settings()
    lane_metrics = [
        run.metrics(settings.groq_input_usd_per_million, settings.groq_output_usd_per_million)
        for run in runs
    ]
    gate = check_fusion_gate(lane_metrics)
    results = build_results(
        runs,
        split_name=args.split,
        rows=rows,
        corpus=corpus,
        gate=gate,
        ablation=ablation,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print()
    print(format_table(lane_metrics))
    print()
    if ablation:
        delta = ablation["delta"]
        print(
            "bge query prefix  recall@5 {:+.4f}  recall@10 {:+.4f}  MRR {:+.4f}".format(
                delta["recall_at_5"], delta["recall_at_10"], delta["mrr"]
            )
        )
    print(f"fusion gate  {gate.detail}")
    print(f"written      {args.out}")

    if gate.status == "fail":
        print(
            "\nFUSION GATE FAILED. Lane 3 scored BELOW an arm it fuses on recall@10. "
            "Per docs/03_BUILD_PLAN.md this is not a finding to write up -- fusion "
            "is broken and nothing downstream of it is meaningful. Stop and fix it.",
            file=sys.stderr,
        )
        return 1
    if gate.status == "tie":
        print(
            "\nFUSION GATE TIED. Lane 3 matched its best arm on recall@10 rather than "
            "beating it. At n=35 that is one question wide; see `python -m backend.sweep_rrf` "
            "for the train (68/80 vs 62/80) and full-set (98/115 vs 92/115) comparison, "
            "and the Known Limitations entry that carries it.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
