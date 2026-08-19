"""Gold-set verification by hand: read every draft, decide every row.

**This is no longer the default path.** docs/02_EVALUATION_SPEC.md §1 now
builds the gold set with `screen.py` (an LLM scores every draft against the
four drop rules) plus `audit.py` (a human checks a stratified sample and the
agreement rate is published). This module is the fully-manual alternative,
kept because it is the ground truth the screener is measured against and
because a corpus small enough to read end to end deserves it.

Nothing here judges a question; it presents one and records what a human
decided. Two files are written, both appended after every single decision:

  data/goldset.jsonl            the gold set -- kept and fixed rows only
  data/goldset_decisions.jsonl  every decision, including drops

The second exists for two reasons. It is what makes resuming possible, and it
is the only record of the rejection rate, which the README has to report
honestly. Keeping drops out of goldset.jsonl means nothing downstream can
mistake a rejected draft for a label.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from backend.goldset.console import append_jsonl, clear_screen, load_jsonl, read_key
from backend.goldset.rules import RULES
from backend.retrieval import dense_store

DEFAULT_DRAFTS = Path("data/goldset_draft.jsonl")
DEFAULT_GOLDSET = Path("data/goldset.jsonl")
DEFAULT_DECISIONS = Path("data/goldset_decisions.jsonl")

# docs/00_PRD.md §5 and the evaluation spec both stop here. Do not chase 150.
TARGET_KEEPS = 120

# The same four rules the screener and the audit use, from rules.py. On screen
# for every decision, because the fiftieth question is where the standard
# slips -- and identical wording across all three is what makes the audit's
# agreement number mean anything.
DROP_RULES = tuple(rule.text for rule in RULES)

KEEP, FIX, DROP, QUIT = "k", "f", "d", "q"


def sort_drafts(drafts: list[dict]) -> list[dict]:
    """Confirmed labels first, then relabelled, each in qid order.

    The clean ones come first so that stopping early still leaves a usable
    gold set rather than a random half of one.
    """
    order = {"confirmed": 0, "relabelled": 1}
    return sorted(drafts, key=lambda r: (order.get(r.get("label_check"), 2), r["qid"]))


@dataclass
class Progress:
    """Running counts, restored from the decision log on resume."""

    total: int
    kept: int = 0
    fixed: int = 0
    dropped: int = 0
    decided: set[str] = field(default_factory=set)

    @property
    def seen(self) -> int:
        return len(self.decided)

    @property
    def keeps(self) -> int:
        """Rows that entered the gold set: kept plus fixed."""
        return self.kept + self.fixed

    def line(self) -> str:
        return (
            f"[{self.seen}/{self.total}] · {self.kept} kept · "
            f"{self.dropped} dropped · {self.fixed} fixed"
        )


def restore_progress(decisions: list[dict], total: int) -> Progress:
    progress = Progress(total=total)
    for row in decisions:
        progress.decided.add(row["qid"])
        if row["decision"] == "keep":
            progress.kept += 1
        elif row["decision"] == "fix":
            progress.fixed += 1
        elif row["decision"] == "drop":
            progress.dropped += 1
    return progress


def gold_row(draft: dict, question: str, decision: str) -> dict:
    """A verified row, in the schema from docs/02_EVALUATION_SPEC.md §1."""
    return {
        "qid": draft["qid"],
        "question": question,
        "gold_chunk_ids": draft["gold_chunk_ids"],
        "source_doc": draft["source_doc"],
        "query_type": draft.get("query_type", "factual"),
        "status": "verified",
        "decision": decision,
        "label_check": draft.get("label_check"),
        "verified_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def decision_row(draft: dict, question: str, decision: str) -> dict:
    return {
        "qid": draft["qid"],
        "decision": decision,
        "question": question,
        "gold_chunk_ids": draft["gold_chunk_ids"],
        "label_check": draft.get("label_check"),
        "decided_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def render(draft: dict, chunk_text: str, progress: Progress) -> str:
    """The full screen for one draft."""
    rule = "=" * 78
    label = draft.get("label_check", "?")
    lines = [
        rule,
        f"  {progress.line()}    label: {label}",
        rule,
        "",
        f"  QUESTION:  {draft['question']}",
        "",
        f"  source: {draft['source_doc']}   chunk: {draft['gold_chunk_ids'][0]}",
    ]
    if label == "relabelled":
        lines.append(f"  (relabelled from {draft.get('drafted_from')})")
    lines += [
        "-" * 78,
        chunk_text,
        "-" * 78,
        "",
        "  DROP RULES",
    ]
    lines += [f"    - {r}" for r in DROP_RULES]
    lines += [
        "",
        "  [k] keep    [f] fix    [d] drop    [q] save and quit",
        "",
    ]
    return "\n".join(lines)


def run(
    drafts: list[dict],
    chunks: dict[str, str],
    goldset_path: Path,
    decisions_path: Path,
    target: int = TARGET_KEEPS,
) -> Progress:
    """The verification loop. Every decision hits disk before the next screen."""
    decisions = load_jsonl(decisions_path)
    progress = restore_progress(decisions, total=len(drafts))
    announced_target = progress.keeps >= target

    for draft in drafts:
        if draft["qid"] in progress.decided:
            continue

        chunk_text = chunks.get(draft["gold_chunk_ids"][0], "[chunk not found in the store]")
        clear_screen()
        print(render(draft, chunk_text, progress))

        while True:
            key = read_key()
            if key in {KEEP, FIX, DROP, QUIT}:
                break

        if key == QUIT:
            print("\nSaved. Re-run to pick up where you left off.")
            return progress

        question = draft["question"]
        if key == FIX:
            print("  Rewritten question (blank keeps the original):")
            replacement = input("  > ").strip()
            if replacement:
                question = replacement
            decision = "fix"
            progress.fixed += 1
        elif key == KEEP:
            decision = "keep"
            progress.kept += 1
        else:
            decision = "drop"
            progress.dropped += 1

        # Gold set first, then the log: if the process dies between the two,
        # the row is in the gold set and will simply be offered again.
        if decision in {"keep", "fix"}:
            append_jsonl(goldset_path, gold_row(draft, question, decision))
        append_jsonl(decisions_path, decision_row(draft, question, decision))
        progress.decided.add(draft["qid"])

        if progress.keeps >= target and not announced_target:
            announced_target = True
            clear_screen()
            print("=" * 78)
            print(f"  TARGET REACHED: {progress.keeps} verified rows (target {target}).")
            print(f"  {progress.line()}")
            print("=" * 78)
            print(f"\n  {goldset_path} is ready. The spec says stop at {target}; do not chase 150.")
            print("\n  [q] stop here    any other key to keep going")
            if read_key() == QUIT:
                return progress

    return progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.goldset.verify",
        description="Verify gold-set drafts by hand. Drafts become labels only here.",
    )
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--corpus-id", default="demo")
    parser.add_argument("--target", type=int, default=TARGET_KEEPS)
    args = parser.parse_args(argv)

    rows = load_jsonl(args.drafts)
    if not rows:
        print(f"No drafts in {args.drafts}. Run backend.goldset.generate first.")
        return 1

    drafts = sort_drafts([r for r in rows if r.get("status") == "draft"])
    if not drafts:
        print(f"{args.drafts} has no rows with status 'draft'.")
        return 1

    with dense_store.connect() as conn:
        chunks = {c.chunk_id: c.text for c in dense_store.fetch_chunks(conn, args.corpus_id)}

    progress = run(drafts, chunks, args.out, args.decisions, target=args.target)

    print(f"\n{progress.line()}")
    print(f"gold set    {args.out}  ({progress.keeps} rows)")
    print(f"decisions   {args.decisions}  ({progress.seen} recorded)")
    if progress.seen:
        print(f"drop rate   {progress.dropped / progress.seen:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
