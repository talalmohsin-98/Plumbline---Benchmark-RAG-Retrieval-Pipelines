"""Assemble the gold set from the screener's keeps and fixes.

Selection is deliberately dull: sort, take everything that survived, stop. The
one judgement call is the sort key -- drafts whose gold label the
generation-time check *confirmed* come before drafts whose label it moved,
because a relabelled row has already been wrong once about which chunk answers
its question.

If fewer than `MINIMUM` rows survive screening this writes nothing and says so.
Padding the shortfall with rows the screener rejected would produce a gold set
of exactly 120 whose last few entries are known-bad, and every metric computed
from it would be quietly wrong in the direction that flatters the project.

`--allow-partial` overrides the stop, for exercising the downstream pipeline
before the gold set is finished. It never pads: it writes only the rows that
did survive, and stamps `provisional: true` on every one of them. That stamp is
the point. A short gold set is a legitimate work-in-progress; a short gold set
that nothing downstream can tell apart from a finished one is a trap, so
`split.py` refuses to stay quiet about it and no number computed from a
provisional set may be published.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from backend.goldset.console import load_jsonl, write_jsonl

DEFAULT_SCREENED = Path("data/goldset_screened.jsonl")
DEFAULT_GOLDSET = Path("data/goldset.jsonl")
DEFAULT_AUDIT = Path("data/audit_decisions.jsonl")
DEFAULT_SECOND = Path("data/claude_screen.jsonl")
DEFAULT_MULTILABEL = Path("data/multilabel.json")
DEFAULT_EXCLUSIONS = Path("data/exclusions.json")

# The model whose verdicts count as "the screener". Rows screened by anything
# else were a stopgap when its daily quota ran out and are not comparable, so
# they are only usable where the audit reached them.
SCREENER_MODEL = "openai/gpt-oss-20b"

# docs/00_PRD.md §5 sets the bar at 100 gold rows. 120 was the Day 1 aspiration
# and 100 is the requirement, and only one of those may decide whether a run
# fails: a set of 115 adjudicated rows is finished, not short.
MINIMUM = 100

# The verdicts that earn a place in the gold set.
ACCEPTED: tuple[str, ...] = ("keep", "fix")

# What `status` a row in goldset.jsonl may carry. "screened" is written here;
# "verified" is what the fully-manual `verify.py` writes. Both are gold; a
# draft is not, and `split.py` checks this set rather than trusting the file.
GOLD_STATUSES: tuple[str, ...] = ("screened", "verified")

_LABEL_ORDER = {"confirmed": 0, "relabelled": 1}


def sort_key(row: dict) -> tuple[int, str]:
    """Confirmed labels first, then relabelled, each in qid order."""
    return (_LABEL_ORDER.get(row.get("label_check"), 2), row["qid"])


class Adjudication(NamedTuple):
    """One row's final verdict and where the authority for it came from."""

    verdict: str
    source: str  # "hand-excluded" | "audit" | "both-agree" | "screener-only"
    #             | "unvalidated" | "unaudited-dispute"


def adjudicate(
    row: dict,
    audit: dict[str, dict],
    second: dict[str, dict],
    screener_model: str = SCREENER_MODEL,
    exclusions: dict[str, dict] | None = None,
) -> Adjudication:
    """Settle one row against the hand exclusions, the audit, then the screeners.

    Precedence, highest first:

    0. **`data/exclusions.json`.** Removed by hand for a stated reason, and the
       reasons there are not about question quality: they are rows a clean re-run
       could not produce, because the chunk they were drafted from is no longer
       in the eligible pool. That outranks the audit rather than losing to it --
       a human judging the question good does not make the row reproducible.
    1. **The audit.** A human read the question, the chunk and the four rules,
       and decided. That overrides both screeners by construction.
    2. **Both screeners agreeing.** A fallback, and after the full-coverage
       pass it decides almost nothing: every row entering the gold set now
       carries a human verdict, so this branch is reached only by rows the
       author has not yet read. Its warrant is no longer a 10-row control
       sample but the pass itself -- 73 agreed keeps read blind, 3 of which
       the author moved off keep.
    3. **Neither.** An unaudited row the screeners split on is undecided, and a
       row screened by a stand-in model was never validated at all. Both are
       excluded rather than guessed, and both are counted in the report.
    """
    qid = row["qid"]
    if exclusions and qid in exclusions:
        return Adjudication("excluded", "hand-excluded")
    if qid in audit:
        return Adjudication(audit[qid]["human_verdict"], "audit")
    if not second:
        # No second opinion configured at all: the screener is all there is,
        # which is the pre-audit behaviour and still the right answer for a
        # corpus that has only been screened once.
        return Adjudication(row["screen_verdict"], "screener-only")
    if screener_model is not None and row.get("screen_model") != screener_model:
        return Adjudication("excluded", "unvalidated")
    other = second.get(qid)
    if other is not None and other.get("screen_verdict") == row.get("screen_verdict"):
        return Adjudication(row["screen_verdict"], "both-agree")
    return Adjudication("excluded", "unaudited-dispute")


def apply_multilabel(row: dict, multilabel: dict[str, dict]) -> dict:
    """Add every equally-correct chunk to `gold_chunk_ids`.

    Recall counts a hit on any listed chunk, so a lane that returns a different
    chunk stating the same answer scores correct -- which it should, because it
    was right. Duplicated boilerplate is common in documentation and single
    labels would have quietly penalised lanes for finding the other copy.
    """
    entry = multilabel.get(row["qid"])
    if not entry:
        return row
    extra = [c for c in entry["also"] if c not in row["gold_chunk_ids"]]
    if not extra:
        return row
    merged = dict(row)
    merged["gold_chunk_ids"] = [*row["gold_chunk_ids"], *extra]
    merged["multilabel_reason"] = entry["why"]
    return merged


def load_hand_map(path: Path) -> dict[str, dict]:
    """Read a hand-adjudicated qid -> entry map, ignoring `_`-prefixed keys.

    Shared by `data/multilabel.json` and `data/exclusions.json`. Both are the
    same shape on purpose: a qid, a stated reason, and a `_README` explaining
    the judgement, so a decision made by hand is reviewable in the file rather
    than buried in a commit message.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# Exclusions that are not judgements about the question. A row removed because
# a clean re-run could not produce it, or because a stand-in model screened it,
# says nothing about question quality and must not inflate a rejection rate.
NOT_A_JUDGEMENT: tuple[str, ...] = ("hand-excluded", "unvalidated")


def census(settled: list[dict]) -> dict[str, int]:
    """Accept/reject counts over every screened row, by direct count.

    A census, not an estimate. The earlier 15.5% was a stratum-weighted
    projection from a 51-row sample onto 116 screened rows, which was the right
    arithmetic while most rows were unread. Once every row entering the gold
    set carries a human verdict there is no unsampled stratum left to project
    from, so the number is simply counted.

    `rejected_on_quality` is the figure that answers "how many drafts were not
    good enough": rows a human or both screeners judged `drop`. `excluded` is
    counted apart because those rows were never judged on their merits.
    """
    counts = {
        "screened": len(settled),
        "accepted": 0,
        "rejected_on_quality": 0,
        "excluded": 0,
        "rejected_unreviewed": 0,
    }
    for row in settled:
        verdict, source = row["screen_verdict"], row.get("verdict_source")
        if verdict in ACCEPTED:
            counts["accepted"] += 1
        elif source in NOT_A_JUDGEMENT:
            counts["excluded"] += 1
        else:
            counts["rejected_on_quality"] += 1
            # A drop nobody read. Invisible in the finished gold set by
            # construction, so it is counted where it can be seen.
            if source != "audit":
                counts["rejected_unreviewed"] += 1
    return counts


def select(rows: list[dict], cap: int | None = None) -> list[dict]:
    """Every screened row verdicted keep or fix, in preference order.

    No cap by default. An earlier version took the first 120 and stopped, which
    was right while drafting continued and more rows were expected than needed.
    The gold set is closed, so truncating now would throw away adjudicated rows
    to hit a number nobody is aiming at any more. `cap` is kept for the case
    where a deliberately smaller set is wanted.
    """
    accepted = [r for r in rows if r.get("screen_verdict") in ACCEPTED]
    ordered = sorted(accepted, key=sort_key)
    return ordered[:cap] if cap is not None else ordered


def gold_row(row: dict, provisional: bool = False) -> dict:
    """One gold-set row, in the schema from docs/02_EVALUATION_SPEC.md §1.

    `status` is "screened", never "verified": no human has read this row. The
    audit measures the screener over a sample, and the distinction between
    "audited to N% agreement" and "checked by hand" has to survive into the
    file itself, not just the README.
    """
    gold = {
        "qid": row["qid"],
        "question": row["question"],
        "gold_chunk_ids": row["gold_chunk_ids"],
        "source_doc": row["source_doc"],
        "query_type": row.get("query_type", "factual"),
        "status": "screened",
        "decision": row["screen_verdict"],
        "screen_rule": row.get("screen_rule"),
        "screen_reason": row.get("screen_reason"),
        "label_check": row.get("label_check"),
        "verdict_source": row.get("verdict_source"),
        "assembled_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if "original_question" in row:
        gold["original_question"] = row["original_question"]
    if "multilabel_reason" in row:
        gold["multilabel_reason"] = row["multilabel_reason"]
    if provisional:
        # Only ever set, never cleared. A later full run rewrites the file from
        # scratch, so the flag cannot survive into a finished gold set.
        gold["provisional"] = True
    return gold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.goldset.assemble",
        description="Take the screener's keeps and fixes and write the gold set.",
    )
    parser.add_argument("--screened", type=Path, default=DEFAULT_SCREENED)
    parser.add_argument("--out", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument(
        "--minimum",
        type=int,
        default=MINIMUM,
        help="fewest gold rows that counts as a finished set (docs/00_PRD.md §5)",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="truncate to this many rows; off by default, the set is not capped",
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--second-opinion", type=Path, default=DEFAULT_SECOND)
    parser.add_argument("--multilabel", type=Path, default=DEFAULT_MULTILABEL)
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=DEFAULT_EXCLUSIONS,
        help=(
            "rows removed by hand, with a stated reason each. Excluded ahead of "
            "the audit: these are rows a clean re-run could not produce."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "write a short gold set instead of stopping, stamped provisional. "
            "For exercising the pipeline only -- nothing computed from it is "
            "publishable."
        ),
    )
    args = parser.parse_args(argv)

    rows = load_jsonl(args.screened)
    if not rows:
        print(f"No rows in {args.screened}. Run backend.goldset.screen first.")
        return 1

    audit = {d["qid"]: d for d in load_jsonl(args.audit)}
    second = {r["qid"]: r for r in load_jsonl(args.second_opinion)}
    multilabel = load_hand_map(args.multilabel)
    exclusions = load_hand_map(args.exclusions)

    # Overwrite each row's verdict with the adjudicated one before selecting,
    # so everything downstream sees one settled verdict per row.
    settled: list[dict] = []
    sources: Counter[str] = Counter()
    for row in rows:
        decision = adjudicate(row, audit, second, SCREENER_MODEL, exclusions)
        sources[f"{decision.source}:{decision.verdict}"] += 1
        settled.append(
            {**row, "screen_verdict": decision.verdict, "verdict_source": decision.source}
        )

    verdicts: Counter[str] = Counter(r["screen_verdict"] for r in settled)
    tally = census(settled)
    chosen = [apply_multilabel(r, multilabel) for r in select(settled, args.cap)]

    print(f"screened     {len(rows)} rows from {args.screened}")
    print(f"audit        {len(audit)} adjudicated verdicts from {args.audit}")
    print(f"second       {len(second)} verdicts from {args.second_opinion}")
    print(f"exclusions   {len(exclusions)} removed by hand from {args.exclusions}")
    for qid, entry in sorted(exclusions.items()):
        print(f"  {qid}       {entry['why']}")
    print("\nadjudicated")
    for source, count in sorted(sources.items()):
        print(f"  {source:<26} {count:>4}")
    print("\nfinal verdicts")
    for verdict, count in verdicts.most_common():
        print(f"  {verdict!s:<10} {count:>4}")

    reviewed = sum(1 for r in settled if r.get("verdict_source") == "audit")
    print()
    print("census       every screened row, counted directly")
    print(f"  screened   {tally['screened']}")
    print(f"  accepted   {tally['accepted']}  {tally['accepted'] / tally['screened']:.1%}")
    print(f"  rejected   {tally['rejected_on_quality']}  "
          f"{tally['rejected_on_quality'] / tally['screened']:.1%}  (judged not good enough)")
    print(f"  excluded   {tally['excluded']}  not judged on quality")
    print(f"  adjudicated {reviewed} of {tally['screened']} screened rows carry a human verdict")
    if tally["rejected_unreviewed"]:
        print(f"  UNREVIEWED {tally['rejected_unreviewed']} rejection(s) no human read")

    labelled = sum(1 for r in chosen if len(r["gold_chunk_ids"]) > 1)
    extra = sum(len(r["gold_chunk_ids"]) - 1 for r in chosen)
    print(f"\nmulti-label  {labelled} rows carry {extra} extra equally-correct chunk(s)")

    available = verdicts["keep"] + verdicts["fix"]
    short = available < args.minimum
    if short and not args.allow_partial:
        print(
            f"\nSHORTFALL: {available} rows survived screening, {args.minimum} needed."
            f"\nNothing written. Draft and screen more chunks -- do not pad with drops."
            f"\nTo write the {available} anyway for pipeline testing: --allow-partial"
        )
        return 1

    write_jsonl(args.out, [gold_row(r, provisional=short) for r in chosen])

    taken: Counter[str] = Counter(r["screen_verdict"] for r in chosen)
    labels: Counter[str] = Counter(r.get("label_check") for r in chosen)
    print(f"\ngold set     {len(chosen)} rows -> {args.out}")
    print(f"  keep       {taken['keep']}")
    print(f"  fix        {taken['fix']}  (rephrased by the screener)")
    print(f"  confirmed  {labels['confirmed']} label-checked, {labels['relabelled']} relabelled")
    print(f"  spare      {available - len(chosen)} keeps/fixes not needed")
    if short:
        print(
            f"\nPROVISIONAL: {len(chosen)} rows, {args.minimum - len(chosen)} short of "
            f"{args.minimum}. Every row is stamped provisional: true.\n"
            "Nothing computed from this set is publishable. Draft and screen more "
            "chunks, then re-run without --allow-partial."
        )
    print("\nScreened, not hand-verified. Audit a sample with backend.goldset.audit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
