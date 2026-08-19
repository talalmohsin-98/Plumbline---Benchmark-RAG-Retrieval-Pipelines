"""Train/test split of the gold set.

The single most consequential file in the project. Every published number comes
from the test split, and the reranker is fine-tuned on the train split; if one
question leaks across, lane 6 looks excellent and means nothing. The assertion
at the bottom of `split` is the guard, and it is not decorative.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from backend.goldset.assemble import GOLD_STATUSES

DEFAULT_GOLDSET = Path("data/goldset.jsonl")
DEFAULT_TRAIN = Path("data/train.jsonl")
DEFAULT_TEST = Path("data/test.jsonl")

# 70/30 per docs/02_EVALUATION_SPEC.md §3. Seed 42 throughout this project so a
# rerun reproduces the same split and therefore the same published numbers.
TRAIN_FRACTION = 0.7
SEED = 42


class LeakageError(AssertionError):
    """Signal shared across the split. Nothing measured after this is valid."""


@dataclass(frozen=True)
class Split:
    train: list[dict]
    test: list[dict]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.test)


def load_goldset(path: Path) -> list[dict]:
    """Read gold rows only: screened by `assemble.py` or hand-verified by `verify.py`.

    `goldset.jsonl` holds only kept and fixed rows, but this filters on status
    anyway: a file that ever gains a draft row must not silently train on it.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Screen and assemble some drafts first.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r.get("status") in GOLD_STATUSES]


def group_by_shared_chunk(rows: list[dict]) -> list[list[dict]]:
    """Group rows into components that share a gold chunk, directly or transitively.

    Multi-labelling makes a plain row-wise shuffle unsafe: `q023` and `q056`
    both answer "which package must be installed for forms", and four chunks
    are gold for both. Split them apart and those chunks are on both sides.

    Transitivity matters and is why this is a component walk rather than a
    pairwise check. If A and B share a chunk and B and C share a different one,
    all three have to travel together -- putting A with C but not B would
    reintroduce exactly the overlap being avoided.

    Deterministic: rows are visited in qid order and each component is returned
    in qid order, so the grouping never depends on file order.
    """
    by_chunk: dict[str, list[str]] = {}
    ordered = sorted(rows, key=lambda r: r["qid"])
    by_qid = {r["qid"]: r for r in ordered}
    for row in ordered:
        for chunk_id in row["gold_chunk_ids"]:
            by_chunk.setdefault(chunk_id, []).append(row["qid"])

    seen: set[str] = set()
    groups: list[list[dict]] = []
    for row in ordered:
        if row["qid"] in seen:
            continue
        stack, component = [row["qid"]], []
        seen.add(row["qid"])
        while stack:
            qid = stack.pop()
            component.append(qid)
            for chunk_id in by_qid[qid]["gold_chunk_ids"]:
                for neighbour in by_chunk[chunk_id]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
        groups.append([by_qid[q] for q in sorted(component)])
    return groups


def split_rows(rows: list[dict], train_fraction: float = TRAIN_FRACTION, seed: int = SEED) -> Split:
    """Shuffle whole groups and cut. Deterministic for a given seed.

    Groups, not rows: every question sharing a gold chunk with another must land
    on the same side, or that chunk is both a training positive and a test
    answer. Sorted before shuffling so the result depends on the seed alone and
    not on the order decisions happened to be recorded in.

    Groups are placed largest-first into whichever side is furthest below its
    quota. Filling in shuffled order instead lets one late 5-row group overshoot
    the cut and land the whole thing in test, which moved the realised split as
    far as 63/37 on this gold set. The seeded shuffle still decides ties and
    order within a size, so the split stays reproducible.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    groups = group_by_shared_chunk(rows)
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)

    target = round(len(rows) * train_fraction)
    train: list[dict] = []
    test: list[dict] = []
    for group in groups:
        train_deficit = target - len(train)
        test_deficit = (len(rows) - target) - len(test)
        if train_deficit >= test_deficit:
            train.extend(group)
        else:
            test.extend(group)

    split = Split(train=sorted(train, key=lambda r: r["qid"]),
                  test=sorted(test, key=lambda r: r["qid"]))
    assert_no_leakage(split)
    return split


def assert_no_leakage(split: Split) -> None:
    """Fail loudly if a qid, or a gold chunk, is on both sides of the split."""
    train_ids = {r["qid"] for r in split.train}
    test_ids = {r["qid"] for r in split.test}
    overlap = train_ids & test_ids
    if overlap:
        raise LeakageError(
            f"{len(overlap)} qid(s) appear in both splits: {sorted(overlap)[:5]}. "
            "Every reported number would be invalid."
        )
    if len(train_ids) != len(split.train) or len(test_ids) != len(split.test):
        raise LeakageError("duplicate qids within a split")
    assert_no_shared_chunks(split)


def assert_no_shared_chunks(split: Split) -> None:
    """Fail if any gold chunk is labelled on both sides of the split.

    Distinct from the qid check, and not implied by it. Multi-labelling means
    two different questions can share a gold chunk -- five FastAPI chunks each
    answer "which package must be installed", and `q123`'s Annotated note is
    verbatim in three. If one of those questions lands in train and another in
    test, the chunk is a positive the reranker trained on and a gold answer it
    is then scored against.

    Day 3 makes it worse: hard negatives are mined from the top-20 retrieved
    chunks with gold removed. "Gold" is per-question, so a chunk that is gold
    for a test question but not for the train question being mined can be
    sampled as a *negative* -- training the reranker to push down a chunk the
    test set expects it to return. That is a leak in the direction that quietly
    lowers the score, which is harder to notice than one that flatters it.
    """
    train_chunks = {c: r["qid"] for r in split.train for c in r["gold_chunk_ids"]}
    shared: list[str] = []
    for row in split.test:
        for chunk_id in row["gold_chunk_ids"]:
            if chunk_id in train_chunks:
                shared.append(f"{chunk_id} (train {train_chunks[chunk_id]}, test {row['qid']})")
    if shared:
        raise LeakageError(
            f"{len(shared)} gold chunk(s) are labelled on both sides of the split: "
            f"{shared[:5]}. Hard-negative mining would train against the test set."
        )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.goldset.split",
        description="Split the gold set 70/30 into train and test.",
    )
    parser.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    rows = load_goldset(args.goldset)
    if not rows:
        print(f"{args.goldset} has no gold rows.")
        return 1

    split = split_rows(rows, args.fraction, args.seed)
    write_jsonl(args.train, split.train)
    write_jsonl(args.test, split.test)

    print(f"gold set   {split.total} gold rows (seed {args.seed})")
    print(f"train      {len(split.train)}  -> {args.train}")
    print(f"test       {len(split.test)}  -> {args.test}")

    # A provisional gold set splits perfectly well and means nothing yet. Said
    # here as well as in `assemble.py` because this is the command whose output
    # gets pasted into a commit message a week later.
    provisional = sum(1 for r in rows if r.get("provisional"))
    if provisional:
        print(
            f"\nPROVISIONAL: {provisional} of {split.total} rows are marked provisional. "
            "This split exercises the pipeline; no number from it is publishable."
        )
    print("\nThe test split is never seen during training. Every reported number comes from it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
