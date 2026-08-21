"""Apply the pre-registered paired tests to two lanes' per-question outcomes.

    python -m backend.significance

The decision rule this implements was written into `docs/02_EVALUATION_SPEC.md`
§3 and committed **before** hard-negative mining began, let alone before any
lane-6 number existed. Nothing here chooses a test after seeing the data:

- **recall@10 and recall@5** -- exact McNemar on the discordant pairs, alpha
  0.05, two-sided. Reported with the ceiling: how many pairs the challenger
  could win at most, and the best p-value that ceiling allows. On this split
  recall@10 cannot reach alpha at all, and the pre-registration says so in
  advance rather than discovering it afterwards.
- **MRR@10** -- the primary inferential test. 10,000-resample paired percentile
  bootstrap on the per-question differences in reciprocal rank, seed 42.
  Real if the interval excludes 0, noise if it does not.

The verdict comes from the MRR interval alone, because that is what was
pre-registered. A recall result that happens to look good is reported and is
not promoted to the verdict.

Reads `data/per_question.json`, written by `backend.evaluate`. It does not
retrieve anything itself: rerunning a comparison must not be able to produce a
different set of retrievals from the one that produced the leaderboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.metrics import (
    discordant_pairs,
    max_detectable_wins,
    mcnemar_exact,
    mean,
    paired_bootstrap_ci,
)

DEFAULT_PER_QUESTION = Path("data/per_question.json")
DEFAULT_OUT = Path("data/significance.json")

# The pre-registered comparison. Both are overridable on the command line so
# any two lanes can be compared descriptively, but these are the two the
# criterion in EVALUATION_SPEC §3 names.
CHALLENGER = "hybrid_rerank_tuned"
BASELINE = "hybrid_rerank"

ALPHA = 0.05
RESAMPLES = 10_000
CONFIDENCE = 0.95
SEED = 42

# EVALUATION_SPEC §3, "Leakage tripwire". Above this, the run stops and the
# pair file is audited against the test split before any number is reported.
# 0.95 of 35 questions is 33.25, so the tripwire fires at 34/35.
TRIPWIRE_RECALL_AT_10 = 0.95


class MissingLaneError(KeyError):
    """A lane named for comparison is not in the per-question file."""


@dataclass(frozen=True)
class McNemarResult:
    """One paired hit/miss comparison, with the ceiling it was measured against."""

    metric: str
    challenger_hits: int
    baseline_hits: int
    questions: int
    challenger_wins: int
    baseline_wins: int
    p_value: float
    max_possible_wins: int
    best_possible_p: float

    @property
    def significant(self) -> bool:
        return self.p_value < ALPHA

    @property
    def can_reach_alpha(self) -> bool:
        """Whether *any* outcome of this comparison could have cleared alpha.

        The question that makes the difference between "no effect" and "no
        power". A challenger can only win a question the baseline missed, so
        when the baseline misses three of 35 the best available p-value is
        0.25 and the test was decided before it ran.
        """
        return self.best_possible_p < ALPHA

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "challenger_hits": self.challenger_hits,
            "baseline_hits": self.baseline_hits,
            "questions": self.questions,
            "delta": round(
                self.challenger_hits / self.questions - self.baseline_hits / self.questions, 4
            ),
            "challenger_wins": self.challenger_wins,
            "baseline_wins": self.baseline_wins,
            "discordant_pairs": self.challenger_wins + self.baseline_wins,
            "p_value": round(self.p_value, 6),
            "significant_at_0.05": self.significant,
            "max_possible_challenger_wins": self.max_possible_wins,
            "best_possible_p_value": round(self.best_possible_p, 6),
            "could_ever_reach_alpha": self.can_reach_alpha,
        }


@dataclass(frozen=True)
class BootstrapResult:
    """The paired bootstrap on MRR@10, the pre-registered primary test."""

    challenger_mrr: float
    baseline_mrr: float
    mean_difference: float
    ci_low: float
    ci_high: float
    questions: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": "mrr_at_10",
            "challenger_mrr": round(self.challenger_mrr, 4),
            "baseline_mrr": round(self.baseline_mrr, 4),
            "mean_difference": round(self.mean_difference, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "ci_width": round(self.ci_high - self.ci_low, 4),
            "confidence": CONFIDENCE,
            "resamples": RESAMPLES,
            "seed": SEED,
            "method": "paired percentile bootstrap over questions",
            "excludes_zero": self.excludes_zero,
            "questions": self.questions,
        }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_per_question(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m backend.evaluate --split test` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def aligned_outcomes(
    document: dict[str, Any], challenger: str, baseline: str
) -> tuple[list[str], list[dict], list[dict]]:
    """Both lanes' per-question records, in one shared qid order.

    Ordering by the questions list rather than by either lane's dict keeps the
    pairing explicit. Two dicts that happen to iterate in the same order are
    not a guarantee, and a silently mispaired comparison would still produce a
    plausible p-value -- the worst failure this module could have.
    """
    lanes = document["lanes"]
    for lane_id in (challenger, baseline):
        if lane_id not in lanes:
            raise MissingLaneError(
                f"lane {lane_id!r} is not in the per-question file. It has "
                f"{sorted(lanes)}. Re-run `backend.evaluate` including it."
            )
    qids = [q["qid"] for q in document["questions"]]
    missing = [q for q in qids if q not in lanes[challenger] or q not in lanes[baseline]]
    if missing:
        raise MissingLaneError(f"{len(missing)} question(s) missing from a lane: {missing[:5]}")
    return qids, [lanes[challenger][q] for q in qids], [lanes[baseline][q] for q in qids]


# --------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------


def run_mcnemar(metric: str, challenger: list[dict], baseline: list[dict]) -> McNemarResult:
    """Exact McNemar on one hit/miss metric, with its power ceiling attached."""
    challenger_hits = [bool(row[metric]) for row in challenger]
    baseline_hits = [bool(row[metric]) for row in baseline]
    wins, losses = discordant_pairs(challenger_hits, baseline_hits)
    ceiling = max_detectable_wins(baseline_hits)
    return McNemarResult(
        metric=metric,
        challenger_hits=sum(challenger_hits),
        baseline_hits=sum(baseline_hits),
        questions=len(challenger_hits),
        challenger_wins=wins,
        baseline_wins=losses,
        p_value=mcnemar_exact(wins, losses),
        max_possible_wins=ceiling,
        # The best case is a clean sweep: the challenger takes every question
        # the baseline missed and loses none.
        best_possible_p=mcnemar_exact(ceiling, 0),
    )


def run_bootstrap(challenger: list[dict], baseline: list[dict]) -> BootstrapResult:
    """Paired percentile bootstrap on the per-question reciprocal ranks."""
    challenger_rr = [float(row["reciprocal_rank"]) for row in challenger]
    baseline_rr = [float(row["reciprocal_rank"]) for row in baseline]
    differences = [c - b for c, b in zip(challenger_rr, baseline_rr, strict=True)]
    low, high = paired_bootstrap_ci(
        differences, resamples=RESAMPLES, confidence=CONFIDENCE, seed=SEED
    )
    return BootstrapResult(
        challenger_mrr=mean(challenger_rr),
        baseline_mrr=mean(baseline_rr),
        mean_difference=mean(differences),
        ci_low=low,
        ci_high=high,
        questions=len(differences),
    )


def check_tripwire(recall_at_10: float, challenger: str) -> dict[str, Any]:
    """EVALUATION_SPEC §3: above 0.95 recall@10, stop and audit before reporting.

    Not a proof of leakage -- the band is one question wide, since lane 5
    already reaches 0.9429 on this split -- but the audit is mandatory and its
    result is reported before the metric is.
    """
    fired = recall_at_10 > TRIPWIRE_RECALL_AT_10
    return {
        "threshold": TRIPWIRE_RECALL_AT_10,
        "observed_recall_at_10": round(recall_at_10, 4),
        "fired": fired,
        "lane": challenger,
        "action": (
            "STOP. Audit data/train_pairs.jsonl against data/test.jsonl for shared "
            "qids, shared question text and gold-chunk overlap, and report the audit "
            "before this metric."
            if fired
            else "not fired"
        ),
    }


def decide(bootstrap: BootstrapResult, tripwire: dict[str, Any]) -> dict[str, str]:
    """The pre-registered verdict rule. One test decides it: the MRR interval.

    Deliberately does not look at the McNemar results. They are reported, and
    on this split recall@10 cannot clear alpha whatever happens, so letting a
    recall outcome into the verdict would be choosing the test that agreed with
    us after the fact.
    """
    if tripwire["fired"]:
        return {
            "verdict": "withheld",
            "statement": (
                "Leakage tripwire fired. No verdict is issued until the training "
                "pairs have been audited against the test split."
            ),
        }
    if not bootstrap.excludes_zero:
        return {
            "verdict": "no detectable difference",
            "statement": (
                f"No detectable difference. MRR@10 {bootstrap.baseline_mrr:.4f} -> "
                f"{bootstrap.challenger_mrr:.4f} ({bootstrap.mean_difference:+.4f}), "
                f"95% CI [{bootstrap.ci_low:+.4f}, {bootstrap.ci_high:+.4f}] includes 0. "
                f"The point estimates are reported; the difference is not."
            ),
        }
    direction = "improvement" if bootstrap.mean_difference > 0 else "regression"
    return {
        "verdict": direction,
        "statement": (
            f"Detectable {direction}. MRR@10 {bootstrap.baseline_mrr:.4f} -> "
            f"{bootstrap.challenger_mrr:.4f} ({bootstrap.mean_difference:+.4f}), "
            f"95% CI [{bootstrap.ci_low:+.4f}, {bootstrap.ci_high:+.4f}] excludes 0."
        ),
    }


def compare(
    document: dict[str, Any], challenger: str, baseline: str
) -> dict[str, Any]:
    """Run every pre-registered test on one pair of lanes."""
    qids, challenger_rows, baseline_rows = aligned_outcomes(document, challenger, baseline)
    recall_10 = run_mcnemar("hit_at_10", challenger_rows, baseline_rows)
    recall_5 = run_mcnemar("hit_at_5", challenger_rows, baseline_rows)
    bootstrap = run_bootstrap(challenger_rows, baseline_rows)
    tripwire = check_tripwire(recall_10.challenger_hits / recall_10.questions, challenger)

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": document.get("split"),
        "questions": len(qids),
        "challenger": challenger,
        "baseline": baseline,
        "protocol": "docs/02_EVALUATION_SPEC.md §3, pre-registered 2026-08-21",
        "alpha": ALPHA,
        "primary_test": "mrr_at_10 paired bootstrap",
        "leakage_tripwire": tripwire,
        "mrr_bootstrap": bootstrap.as_dict(),
        "mcnemar": {"recall_at_10": recall_10.as_dict(), "recall_at_5": recall_5.as_dict()},
        **decide(bootstrap, tripwire),
        "questions_won": [
            q
            for q, c, b in zip(qids, challenger_rows, baseline_rows, strict=True)
            if c["hit_at_10"] and not b["hit_at_10"]
        ],
        "questions_lost": [
            q
            for q, c, b in zip(qids, challenger_rows, baseline_rows, strict=True)
            if b["hit_at_10"] and not c["hit_at_10"]
        ],
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def format_comparison(result: dict[str, Any]) -> str:
    lines: list[str] = []
    tripwire = result["leakage_tripwire"]
    if tripwire["fired"]:
        lines += [
            "!! LEAKAGE TRIPWIRE FIRED !!",
            f"   {result['challenger']} scored recall@10 "
            f"{tripwire['observed_recall_at_10']:.4f}, above the {tripwire['threshold']} "
            "threshold pre-registered in EVALUATION_SPEC §3.",
            f"   {tripwire['action']}",
            "",
        ]

    boot = result["mrr_bootstrap"]
    lines += [
        f"{result['challenger']} vs {result['baseline']}   "
        f"({result['questions']} questions, split {result['split']})",
        "",
        "PRIMARY  MRR@10, paired percentile bootstrap "
        f"({boot['resamples']:,} resamples, seed {boot['seed']})",
        f"         {boot['baseline_mrr']:.4f} -> {boot['challenger_mrr']:.4f}  "
        f"({boot['mean_difference']:+.4f})",
        f"         95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]  "
        f"width {boot['ci_width']:.4f}  "
        f"{'excludes' if boot['excludes_zero'] else 'includes'} 0",
        "",
    ]

    for key, mc in result["mcnemar"].items():
        power = (
            ""
            if mc["could_ever_reach_alpha"]
            else f"  -- CANNOT reach alpha={result['alpha']} at this n"
        )
        lines += [
            f"         {key}, exact McNemar{power}",
            f"         {mc['baseline_hits']}/{mc['questions']} -> "
            f"{mc['challenger_hits']}/{mc['questions']}  ({mc['delta']:+.4f})",
            f"         discordant {mc['challenger_wins']} won / {mc['baseline_wins']} lost, "
            f"p = {mc['p_value']:.4f}",
            f"         ceiling: at most {mc['max_possible_challenger_wins']} wins available, "
            f"best possible p = {mc['best_possible_p_value']:.4f}",
            "",
        ]

    lines += [f"VERDICT  {result['verdict'].upper()}", f"         {result['statement']}"]
    if result["questions_won"] or result["questions_lost"]:
        lines += [
            f"         won  {result['questions_won']}",
            f"         lost {result['questions_lost']}",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.significance",
        description="Paired McNemar and bootstrap tests between two lanes.",
    )
    parser.add_argument("--per-question", type=Path, default=DEFAULT_PER_QUESTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--challenger", default=CHALLENGER)
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args(argv)

    document = load_per_question(args.per_question)
    if document.get("split") != "test":
        print(
            f"WARNING: these outcomes are from the {document.get('split')!r} split. "
            "The pre-registered comparison is on the test split; nothing from this "
            "run is publishable.",
            file=sys.stderr,
        )

    result = compare(document, args.challenger, args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(format_comparison(result))
    print(f"\nwritten  {args.out}")

    # A fired tripwire is a stop, not a finding, so it exits non-zero: a CI run
    # must not go on to publish a leaderboard behind it.
    return 1 if result["leakage_tripwire"]["fired"] else 0


if __name__ == "__main__":
    sys.exit(main())
