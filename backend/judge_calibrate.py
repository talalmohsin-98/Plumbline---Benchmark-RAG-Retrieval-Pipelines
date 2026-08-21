"""Blind hand-labelling of 30 judge verdicts, and the agreement rate it produces.

EVALUATION_SPEC §2 requires this step: the groundedness rate is an LLM's opinion
until somebody has checked the LLM. Three commands, in order:

    python -m backend.judge_calibrate --build     # 30-sample blind queue
    python -m backend.judge_calibrate             # label them, one at a time
    python -m backend.judge_calibrate --score     # agreement, kappa, confusion

**How the blindness is enforced, and why it is enforced this way.**

Day 1's audit pass checked blindness by asserting that no `second_*` key reached
the screen. That check was too narrow: it named the fields that existed when it
was written, so `screen_scores` -- added later -- sailed straight past it. The
lesson was not "add screen_scores to the list".

So a queue row here is **constructed from a whitelist**, never copied and
stripped. `blind_row` names the five fields a labeller may see and builds a new
dict from them. A field added to the verdict file tomorrow cannot leak, because
nothing copies the record. The sentences shown are re-derived from the answer
text by `split_sentences` rather than read off the judge's verdict list, which
is the same principle applied one level down: the verdict list carries
`supported` and `reason` on every element.

And the test suite pins it with `TELLTALE_` markers on every judge-side field,
asserting they appear in neither the serialised row nor the rendered screen --
plus the stronger form, that the bytes are identical whichever way the judge
voted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.judge import split_sentences
from backend.metrics import cohens_kappa
from backend.retrieval.corpus import Corpus

DEFAULT_VERDICTS = Path("data/judge_verdicts.jsonl")
DEFAULT_QUEUE = Path("data/judge_calibration_queue.jsonl")
DEFAULT_KEY = Path("data/judge_calibration_key.jsonl")
DEFAULT_LABELS = Path("data/judge_calibration_labels.jsonl")
DEFAULT_REPORT = Path("data/judge_calibration.json")

SAMPLE_SIZE = 30  # EVALUATION_SPEC §2
SEED = 42

# Exactly the fields a labeller may see, and the only keys `blind_row` emits.
# A test asserts the two are the same set, so adding a field to the queue is a
# deliberate act with a test to answer to -- while adding a field to the verdict
# file, which is the thing that actually happens, cannot leak at all.
BLIND_FIELDS = (
    "item_id",
    "qid",
    "lane_id",
    "question",
    "answer",
    "sentences",
    "context",
)


class CalibrationError(RuntimeError):
    """The calibration cannot proceed, and proceeding anyway would be worse."""


# --------------------------------------------------------------------------
# Building the blind queue
# --------------------------------------------------------------------------


def item_id(record: dict) -> str:
    """Stable id for one (lane, question) answer. Deterministic across runs."""
    return f"{record['lane_id']}:{record['qid']}"


def blind_row(record: dict, texts: dict[str, str]) -> dict[str, Any]:
    """One queue row, built from the whitelist. Never a copy of the record.

    Note what is *derived* rather than copied: `sentences` comes from
    re-splitting the answer, not from `record["verdicts"]`, because every
    element of that list carries the judge's `supported` flag and its reason.
    Copying the list and popping two keys would work today and break the first
    time the judge records a third field.

    `context` deliberately carries the chunk text, because a labeller cannot
    judge groundedness without reading what the answer was supposed to be
    grounded in. Chunk ids and their text say nothing about the verdict.

    `texts` is passed in rather than looked up, so this function stays pure and
    the blindness tests need no corpus, no database and no network.
    """
    return {
        "item_id": item_id(record),
        "qid": record["qid"],
        "lane_id": record["lane_id"],
        "question": record["question"],
        "answer": record["answer"],
        "sentences": split_sentences(record["answer"]),
        "context": [
            {"chunk_id": chunk_id, "text": texts.get(chunk_id, "")}
            for chunk_id in record["chunk_ids"]
        ],
    }


def key_row(record: dict, *, stratified: bool = False) -> dict[str, Any]:
    """The judge's verdict, kept in a *separate* file the labelling never opens.

    Separate file rather than a hidden field, because a field is one careless
    `print(row)` away from the screen and a file that is never read cannot be
    rendered by accident.
    """
    return {
        "item_id": item_id(record),
        "judge_grounded": bool(record["grounded"]),
        "judge_verdicts": record.get("verdicts", []),
        # Recorded here rather than on the queue row. It is a fact about the
        # judge's verdicts, so putting it on the queue would be one more thing
        # to keep off the labeller's screen; the key file is never rendered.
        "stratified": stratified,
    }


def build_queue(
    records: list[dict],
    texts: dict[str, str],
    *,
    size: int = SAMPLE_SIZE,
    seed: int = SEED,
    stratify: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Sample `size` records and return (blind queue, answer key).

    Uniform by default. Uniform makes the agreement rate an unbiased estimate
    of agreement over the population of answers, which is the number
    EVALUATION_SPEC §2 asks to publish.

    `stratify` balances the draw across the judge's two verdicts instead. It is
    available because a judge that says "grounded" to 95% of answers teaches you
    almost nothing about its behaviour on the other 5% from a uniform sample of
    30 -- but a stratified rate is NOT a population rate, and the report stamps
    `population_representative: false` so it can never be quoted as one.

    The queue is shuffled after sampling, so its order gives nothing away.
    """
    if len(records) < size:
        raise CalibrationError(
            f"only {len(records)} judged answers available, need {size}. "
            "Run `python -m backend.judge` over more questions first."
        )
    rng = random.Random(seed)
    ordered = sorted(records, key=item_id)

    if stratify:
        grounded = [r for r in ordered if r["grounded"]]
        ungrounded = [r for r in ordered if not r["grounded"]]
        half = size // 2
        # Take as close to half of each as the pool allows, topping up from
        # whichever side is larger rather than returning a short queue.
        take_grounded = min(half, len(grounded))
        take_ungrounded = min(size - take_grounded, len(ungrounded))
        take_grounded = min(len(grounded), size - take_ungrounded)
        chosen = rng.sample(grounded, take_grounded) + rng.sample(ungrounded, take_ungrounded)
    else:
        chosen = rng.sample(ordered, size)

    rng.shuffle(chosen)
    return (
        [blind_row(r, texts) for r in chosen],
        [key_row(r, stratified=stratify) for r in chosen],
    )


# --------------------------------------------------------------------------
# The labelling console
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Progress:
    """Position in the run. Never the agreement rate.

    The agreement rate is the screener's verdict in aggregate: watch it move
    after each decision and you can infer what the judge said about the one you
    just labelled, and anchor on it for the next. Withheld for the same reason
    Day 1's blind audit pass withholds it.
    """

    total: int
    done: int

    def line(self) -> str:
        return f"[{self.done}/{self.total}]"


def render_item(row: dict, progress: Progress) -> str:
    """The screen a labeller sees. Reads only whitelisted fields.

    Every field it touches is named literally below. It never iterates the row,
    never falls back to "print anything else you find", and never sees the key
    file. That is what makes the blindness structural rather than a promise.
    """
    lines = [
        "=" * 78,
        f"{progress.line()}  {row['qid']}",
        "",
        f"QUESTION   {row['question']}",
        "",
        "ANSWER",
    ]
    lines += [f"  {n}. {sentence}" for n, sentence in enumerate(row["sentences"], start=1)]
    lines += ["", "CONTEXT PASSAGES (what the answer had to be grounded in)"]
    for index, passage in enumerate(row["context"], start=1):
        body = " ".join(passage["text"].split())
        lines.append(f"  [{index}] {body[:600]}{'...' if len(body) > 600 else ''}")
    lines += [
        "",
        "Is EVERY sentence of the answer supported by these passages?",
        "  g = grounded (all sentences supported)",
        "  u = ungrounded (at least one sentence is not)",
        "  s = skip (genuinely cannot tell)",
        "  q = quit and save",
    ]
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_row(item: dict, choice: str) -> dict[str, Any]:
    """One human decision. Carries no judge field, by construction."""
    return {
        "item_id": item["item_id"],
        "qid": item["qid"],
        "lane_id": item["lane_id"],
        "human_grounded": choice == "g",
        "skipped": choice == "s",
        "labelled_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "blind": True,
    }


def run_console(queue: list[dict], labels_path: Path) -> int:
    """Walk the queue, appending decisions. Resumable: already-labelled rows skip."""
    done = {row["item_id"] for row in read_jsonl(labels_path)}
    remaining = [row for row in queue if row["item_id"] not in done]
    if not remaining:
        print(f"All {len(queue)} items already labelled. Run --score.")
        return 0

    print(
        f"{len(remaining)} of {len(queue)} items left. You are labelling BLIND: "
        "the judge's verdict is in a separate file this command never opens.\n"
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(remaining, start=1):
            print(render_item(row, Progress(total=len(queue), done=len(done) + index - 1)))
            while True:
                choice = input("  [g/u/s/q] ").strip().lower()
                if choice in {"g", "u", "s", "q"}:
                    break
                print("  g, u, s or q.")
            if choice == "q":
                print(f"\nStopped. {len(done)} labelled so far -> {labels_path}")
                return 0
            handle.write(json.dumps(label_row(row, choice), ensure_ascii=False) + "\n")
            handle.flush()
            done.add(row["item_id"])
    print(f"\nDone. {len(done)} labelled -> {labels_path}")
    return 0


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score(
    labels: list[dict], key: list[dict], *, stratified: bool
) -> dict[str, Any]:
    """Agreement between the human labels and the judge's verdicts.

    Reports Cohen's kappa alongside raw agreement, because raw agreement is
    close to meaningless when one class dominates: a judge that calls every
    answer grounded scores 90% agreement against a population that is 90%
    grounded while carrying no information at all. Kappa is the number that
    notices.
    """
    verdicts = {row["item_id"]: row["judge_grounded"] for row in key}
    scored = [row for row in labels if not row["skipped"] and row["item_id"] in verdicts]
    skipped = [row for row in labels if row["skipped"]]

    human = [bool(row["human_grounded"]) for row in scored]
    judge = [verdicts[row["item_id"]] for row in scored]

    both_grounded = sum(1 for h, j in zip(human, judge, strict=True) if h and j)
    both_ungrounded = sum(1 for h, j in zip(human, judge, strict=True) if not h and not j)
    judge_only = sum(1 for h, j in zip(human, judge, strict=True) if j and not h)
    human_only = sum(1 for h, j in zip(human, judge, strict=True) if h and not j)
    agreed = both_grounded + both_ungrounded

    def class_agreement(flag: bool) -> dict[str, Any]:
        rows = [(h, j) for h, j in zip(human, judge, strict=True) if j is flag]
        return {
            "judge_said": "grounded" if flag else "ungrounded",
            "n": len(rows),
            "agreed": sum(1 for h, j in rows if h == j),
            "agreement": round(sum(1 for h, j in rows if h == j) / len(rows), 4) if rows else None,
        }

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sampled": len(labels),
        "scored": len(scored),
        "skipped": len(skipped),
        "agreement": round(agreed / len(scored), 4) if scored else None,
        "cohens_kappa": round(cohens_kappa(human, judge), 4) if scored else None,
        "confusion": {
            "both_grounded": both_grounded,
            "both_ungrounded": both_ungrounded,
            "judge_grounded_human_not": judge_only,
            "human_grounded_judge_not": human_only,
        },
        "by_judge_verdict": [class_agreement(True), class_agreement(False)],
        "judge_grounded_rate_in_sample": (
            round(sum(judge) / len(judge), 4) if judge else None
        ),
        "human_grounded_rate_in_sample": (
            round(sum(human) / len(human), 4) if human else None
        ),
        "blind": all(row.get("blind") for row in labels),
        "stratified": stratified,
        "population_representative": not stratified,
        "note": (
            "Stratified sample: agreement is per-class and must NOT be quoted as a "
            "population agreement rate."
            if stratified
            else "Uniform sample of the judged answers, so this agreement rate estimates "
            "agreement over the population."
        ),
    }


def format_report(report: dict[str, Any]) -> str:
    confusion = report["confusion"]
    lines = [
        f"scored          {report['scored']} of {report['sampled']} "
        f"({report['skipped']} skipped)",
        f"agreement       {report['agreement']}",
        f"Cohen's kappa   {report['cohens_kappa']}",
        "",
        f"  judge grounded, human grounded    {confusion['both_grounded']}",
        f"  judge ungrounded, human ungrounded {confusion['both_ungrounded']}",
        f"  judge grounded, human NOT          {confusion['judge_grounded_human_not']}",
        f"  human grounded, judge NOT          {confusion['human_grounded_judge_not']}",
        "",
        f"  judge grounded rate in sample     {report['judge_grounded_rate_in_sample']}",
        f"  human grounded rate in sample     {report['human_grounded_rate_in_sample']}",
        "",
    ]
    for row in report["by_judge_verdict"]:
        lines.append(
            f"  when the judge said {row['judge_said']:<11} "
            f"n={row['n']:<3} agreement {row['agreement']}"
        )
    lines += ["", report["note"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.judge_calibrate",
        description="Build, label and score a blind 30-sample calibration of the judge.",
    )
    parser.add_argument("--build", action="store_true", help="build the blind queue")
    parser.add_argument("--score", action="store_true", help="score the labels against the key")
    parser.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--corpus-id", default="demo")
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="balance the draw across the judge's verdicts (NOT a population rate)",
    )
    args = parser.parse_args(argv)

    if args.build:
        records = read_jsonl(args.verdicts)
        if not records:
            print(
                f"{args.verdicts} is empty. Run `python -m backend.judge` first.",
                file=sys.stderr,
            )
            return 1
        # The chunk text is not in the verdict file -- it would multiply its size
        # by the context depth -- so it is read back from the corpus here. The
        # labeller cannot judge grounding without it.
        corpus = Corpus(args.corpus_id)
        try:
            texts = dict(corpus.texts)
        finally:
            corpus.close()
        try:
            queue, key = build_queue(
                records, texts, size=args.size, seed=args.seed, stratify=args.stratify
            )
        except CalibrationError as exc:
            # A short queue is worse than no queue: EVALUATION_SPEC §2 asks for
            # 30, and a calibration over 12 would be published as though it were
            # the required one. Refuse, and say what to run instead.
            print(f"{exc}", file=sys.stderr)
            print(
                "\nThe judge sweep is resumable -- re-running it appends to "
                "the verdict file rather than starting over. On the free tier "
                "one judged answer costs about 8,500 prompt tokens against a "
                "200,000/day ceiling, so this may need another day's budget.",
                file=sys.stderr,
            )
            return 1
        write_jsonl(args.queue, queue)
        write_jsonl(args.key, key)
        print(f"queue   {len(queue)} items -> {args.queue}   (carries no judge verdict)")
        print(f"key     {len(key)} verdicts -> {args.key}   (do not open until --score)")
        if args.stratify:
            print(
                "\nSTRATIFIED. The resulting agreement rate is per-class and is not a "
                "population rate. The report will say so."
            )
        print("\nNext: `python -m backend.judge_calibrate` to label them.")
        return 0

    if args.score:
        labels = read_jsonl(args.labels)
        key = read_jsonl(args.key)
        if not labels:
            print(f"{args.labels} is empty. Label the queue first.", file=sys.stderr)
            return 1
        stratified = any(row.get("stratified") for row in key)
        report = score(labels, key, stratified=stratified)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(format_report(report))
        print(f"\nwritten  {args.report}")
        return 0

    queue = read_jsonl(args.queue)
    if not queue:
        print(f"{args.queue} is empty. Run --build first.", file=sys.stderr)
        return 1
    return run_console(queue, args.labels)


if __name__ == "__main__":
    sys.exit(main())
