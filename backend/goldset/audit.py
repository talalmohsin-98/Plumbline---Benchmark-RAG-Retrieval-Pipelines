"""Human audit of the LLM screener: the number that makes the gold set credible.

The screener replaced reading 350 drafts by hand. What stops that from being
the circular setup docs/02_EVALUATION_SPEC.md §1 warns about is this file: a
human reads a sample, judges it blind, and the disagreement rate is published
next to every metric the gold set produces.

Two design decisions carry the whole measurement.

**The screener's verdict is hidden until after the keypress.** Shown first, it
is an anchor: the eye finds reasons for a verdict it has already read, and the
resulting number measures compliance rather than agreement. Hiding it costs
nothing and is the difference between an audit and a rubber stamp.

**The sample is stratified across all three verdicts, not just the keeps.**
Auditing keeps alone measures false positives -- questions the screener let
through that it should not have. It cannot see a false drop, because a wrongly
dropped question never appears. Those are the expensive errors: they are
invisible in the finished gold set, and enough of them quietly bias it toward
whatever the screener finds easy.

Fixed rows are shown as the screener first saw them -- the draft wording, not
the rephrasing -- because otherwise the auditor is judging a question whose
verbatim overlap has already been removed and would agree by construction. The
rephrasing is revealed afterwards, alongside the verdict.

**Disputed mode** (`--second-opinion`) replaces the stratified draw when a
second screener has judged the same drafts. The queue is then every row the two
screeners disagree about, plus a sample of rows they both kept as a control.
This is a far better use of a human hour: a row two independent screeners agree
on carries little information, while a disputed row is exactly where the gold
set is being decided by a coin flip. The controls exist because a queue of pure
disagreements would tell you nothing about the agreed majority -- if the human
also splits from both screeners on rows they agreed about, the whole screening
step is in question, not just the disputed slice.

Controls are shuffled in with the disputed rows and are indistinguishable on
screen. Marking them would restore the anchoring this file exists to prevent.

Writes `data/audit_results.json`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from backend.goldset.assemble import (
    ACCEPTED,
    DEFAULT_EXCLUSIONS,
    NOT_A_JUDGEMENT,
    adjudicate,
    load_hand_map,
)
from backend.goldset.console import (
    append_jsonl,
    clear_screen,
    load_jsonl,
    read_key,
    write_jsonl,
)
from backend.goldset.rules import RULES, VERDICTS
from backend.retrieval import dense_store

DEFAULT_SCREENED = Path("data/goldset_screened.jsonl")
DEFAULT_RESULTS = Path("data/audit_results.json")
DEFAULT_DECISIONS = Path("data/audit_decisions.jsonl")
DEFAULT_QUEUE = Path("data/audit_queue.jsonl")

# Rows both screeners kept, mixed into the disputed queue as a control. Ten is
# enough to notice a human who disagrees with the agreed majority (at a true
# rate of 20% the chance of seeing none is under 11%), and small enough that
# the queue stays mostly disputed rows, which is the point of the mode.
CONTROL_ROWS = 10

# 40 is the sample size, ~11% of 350. Large enough that a per-class rate is
# worth quoting (13 rows per class puts the 95% interval at roughly ±20 points,
# which distinguishes a working screener from a broken one), small enough to
# sit through in one pass. The groundedness judge gets the same treatment at
# n=30 -- see §2 of the evaluation spec.
SAMPLE_SIZE = 40

# 42 throughout this project, so a rerun audits the same rows.
SEED = 42

KEEP, FIX, DROP, QUIT = "k", "f", "d", "q"
_KEYS = {KEEP: "keep", FIX: "fix", DROP: "drop"}


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def allocate(total: int, sizes: dict[str, int]) -> dict[str, int]:
    """Split `total` draws as evenly as possible across classes, capped by supply.

    Even rather than proportional. The audit exists for the per-class
    agreement rates, and a proportional draw from a class holding 8% of the
    rows would return three rows and a rate no one should quote. Evenness costs
    the headline figure its population weighting; `summarise` used to put that
    back by re-weighting, and no longer does, because a full-coverage pass
    leaves nothing unsampled to re-weight onto. This sampler is for the case
    where a census is not affordable.
    """
    picked = dict.fromkeys(sizes, 0)
    remaining = total
    # Smallest class first: it hits its cap earliest, and whatever it cannot
    # supply is then still available to the classes that can.
    open_classes = sorted((n for n in sizes if sizes[n] > 0), key=lambda n: sizes[n])
    for position, name in enumerate(open_classes):
        share = -(-remaining // (len(open_classes) - position))  # ceiling division
        picked[name] = min(share, sizes[name], remaining)
        remaining -= picked[name]
    return picked


def stratified_sample(
    rows: list[dict], total: int = SAMPLE_SIZE, seed: int = SEED
) -> list[dict]:
    """Draw `total` rows spread across the screener's three verdicts.

    Sorted by qid before sampling so the draw depends on the seed alone and not
    on the order screening happened to write the rows in.
    """
    by_verdict: dict[str, list[dict]] = {v: [] for v in VERDICTS}
    for row in sorted(rows, key=lambda r: r["qid"]):
        if row.get("screen_verdict") in by_verdict:
            by_verdict[row["screen_verdict"]].append(row)

    quota = allocate(total, {v: len(rs) for v, rs in by_verdict.items()})
    rng = random.Random(seed)
    drawn: list[dict] = []
    for verdict in VERDICTS:
        drawn += rng.sample(by_verdict[verdict], quota[verdict])

    # Shuffled so the auditor does not see thirteen keeps, then thirteen fixes:
    # a run of one class is itself a hint about what the screener said.
    rng.shuffle(drawn)
    return drawn


def build_queue(
    screened: list[dict],
    second: list[dict],
    controls: int = CONTROL_ROWS,
    seed: int = SEED,
    control_pool: set[str] | None = None,
    control_seed: int | None = None,
) -> list[dict]:
    """Every row the two screeners disagree about, plus `controls` agreed keeps.

    Rows the second screener never saw are skipped rather than counted as
    agreement: no verdict is not the same as the same verdict.

    The returned rows carry the second opinion under `second_*` keys and a
    `queue_kind` of "disputed" or "control". Nothing that renders the blind
    screen may read either.

    `control_pool` narrows which agreed keeps may be drawn as controls, and
    `control_seed` draws them with a seed of their own. Both exist because a
    control is only worth auditing while nobody knows it is one: a control that
    has been named, or that can be recomputed from a published seed, measures
    nothing. They also allow the controls to be aimed at one slice of the
    agreed majority rather than spread across all of it, which is the right
    draw when "both screeners agreed" means something different in that slice
    than it does elsewhere.
    """
    by_qid = {row["qid"]: row for row in second}
    disputed: list[dict] = []
    agreed_keeps: list[dict] = []

    for row in sorted(screened, key=lambda r: r["qid"]):
        other = by_qid.get(row["qid"])
        if other is None or row.get("screen_verdict") not in VERDICTS:
            continue
        merged = dict(row)
        merged["second_verdict"] = other["screen_verdict"]
        merged["second_rule"] = other.get("screen_rule")
        merged["second_reason"] = other.get("screen_reason")
        merged["second_model"] = other.get("screen_model")
        if other["screen_verdict"] != row["screen_verdict"]:
            merged["queue_kind"] = "disputed"
            disputed.append(merged)
        elif row["screen_verdict"] == "keep":
            merged["queue_kind"] = "control"
            agreed_keeps.append(merged)

    candidates = (
        agreed_keeps
        if control_pool is None
        else [row for row in agreed_keeps if row["qid"] in control_pool]
    )
    # A separate generator, so the control draw can be re-seeded without also
    # reordering the queue -- and so the queue order gives nothing away about
    # which seed drew the controls.
    control_rng = random.Random(seed if control_seed is None else control_seed)
    drawn = control_rng.sample(candidates, min(controls, len(candidates)))

    queue = disputed + drawn
    # Shuffled so a control cannot be spotted by its position in the run.
    random.Random(seed).shuffle(queue)
    return queue


def build_coverage_queue(
    screened: list[dict],
    audit: dict[str, dict],
    second: dict[str, dict],
    exclusions: dict[str, dict] | None = None,
    seed: int = SEED,
    include_rejections: bool = False,
) -> list[dict]:
    """Every row whose fate rests on screener agreement rather than a human.

    The disputed queue spends a human hour where the screeners disagree and
    lets agreement stand in everywhere else. This is the queue for when that
    substitution is no longer wanted: the rows the gold set is currently
    accepting on the strength of two models agreeing.

    `include_rejections` extends it to the other side of the same substitution
    -- rows *dropped* because both screeners agreed to drop them, which no
    human has read either. Those are the expensive errors. A wrongly kept row
    can be found by reading the gold set; a wrongly dropped one is gone, and
    no inspection of the finished file can reveal it. Covering the keeps but
    not the drops leaves the claim true in only one direction.

    Membership is decided by `assemble.adjudicate`, not by a second copy of
    its rules here. That function is what actually builds the gold set, so
    asking it directly is the only way this queue cannot drift out of step
    with the file it is meant to cover. Rows excluded for reasons that are not
    judgements about the question -- removed by hand, or screened only by a
    stand-in model -- stay out either way: adjudicating them would not change
    what happens to them.

    The rows carry no `second_*` keys, so nothing downstream can render a
    verdict the auditor is not meant to see.
    """
    queue: list[dict] = []
    for row in sorted(screened, key=lambda r: r["qid"]):
        decision = adjudicate(row, audit, second, exclusions=exclusions)
        if decision.source == "audit" or decision.source in NOT_A_JUDGEMENT:
            continue
        if decision.verdict not in ACCEPTED and not include_rejections:
            continue
        queue.append({**row, "queue_kind": "coverage"})

    # Shuffled so the pass is not ordered by qid, which would run batch 1
    # before batch 2 and group each document's chunks together.
    random.Random(seed).shuffle(queue)
    return queue


def load_qids(path: Path) -> set[str]:
    """Read a qid-per-line file, ignoring blank lines and `#` comments."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def audited_question(row: dict) -> str:
    """The question as the screener saw it, before any rephrasing."""
    return row.get("original_question") or row["question"]


# --------------------------------------------------------------------------
# The screens
# --------------------------------------------------------------------------


@dataclass
class Progress:
    """Running counts, restored from the decision log on resume."""

    total: int
    agreed: int = 0
    decided: set[str] = field(default_factory=set)
    blind: bool = False

    @property
    def seen(self) -> int:
        return len(self.decided)

    def line(self) -> str:
        """Position in the run, and -- outside a blind pass -- agreement so far.

        The agreement rate is withheld when blind. It is the screener's verdict
        in aggregate: watching it climb tells you what the screener has been
        saying, which is the anchor the blind screen exists to remove.
        """
        if self.blind:
            return f"[{self.seen}/{self.total}]"
        rate = f"{self.agreed / self.seen:.0%}" if self.seen else "--"
        return f"[{self.seen}/{self.total}] · {self.agreed} agreed · {rate}"


def render_question(row: dict, chunk_text: str, progress: Progress) -> str:
    """The blind screen: everything except what the screener decided.

    Nothing here may reveal the verdict, including indirectly. The rule text is
    shown in full and in fixed order for every row, so its presence carries no
    signal about which rule the screener thought fired.
    """
    rule = "=" * 78
    lines = [
        rule,
        f"  AUDIT  {progress.line()}",
        rule,
        "",
        f"  QUESTION:  {audited_question(row)}",
        "",
        f"  source: {row['source_doc']}   chunk: {row['gold_chunk_ids'][0]}",
        "-" * 78,
        chunk_text,
        "-" * 78,
        "",
        "  RULES",
    ]
    lines += [f"    - {r.text}" for r in RULES]
    lines += [
        "",
        "  Your verdict:  [k] keep    [f] fix    [d] drop    [q] save and quit",
        "",
    ]
    return "\n".join(lines)


def render_reveal(row: dict, human: str) -> str:
    """The reveal screen: what each screener said, and whether you agreed."""
    screener = row["screen_verdict"]
    lines = [
        "",
        "=" * 78,
        f"  YOU: {human.upper():<6}    SCREENER: {screener.upper():<6}    "
        f"{'AGREE' if screener == human else '>>> DISAGREE <<<'}",
        "=" * 78,
        f"  screener  {row.get('screen_model', '?')}",
        f"    rule:   {row.get('screen_rule') or '-'}",
        f"    reason: {row.get('screen_reason', '')}",
    ]
    if "second_verdict" in row:
        second = row["second_verdict"]
        lines += [
            "",
            f"  SECOND OPINION: {second.upper():<6}  "
            f"{'agrees with you' if second == human else 'disagrees with you'}",
            f"    {row.get('second_model', '?')}",
            f"    rule:   {row.get('second_rule') or '-'}",
            f"    reason: {row.get('second_reason', '')}",
        ]
    if "original_question" in row:
        lines += ["", f"  rephrased to: {row['question']}"]
    lines += ["", "  any key to continue    [q] save and quit", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


def summarise(decisions: list[dict], population: Counter[str]) -> dict:
    """Agreement overall, per verdict class, per queue kind, and per mode.

    `agreement_overall` is the rate across every decision on file. Under full
    coverage it is a census rather than an estimate, so the re-weighting this
    function used to do has been retired: the old `agreement_weighted` existed
    to project an evenly-drawn sample back onto the real class sizes, and
    there is no longer an unsampled stratum to project onto.

    **`by_mode` is confounded and must not be read as an anchoring result.**
    Mode is not randomised: the revealed stratum is the disputed rows plus
    controls, the blind stratum is the rows both screeners agreed to keep.
    Those populations have very different base rates of being wrong, so the
    gap between the two figures is population, not method. The comparison that
    does hold is same-population, different-mode -- revealed controls against
    blind agreed keeps -- and it lives in the spec, not here, because it needs
    the caveat about its sample size beside it.

    `by_verdict` is where the diagnosis lives. A high keep rate with a poor
    drop rate means the screener is discarding good questions, which no amount
    of looking at the finished gold set would ever show.
    """
    by_verdict: dict[str, dict] = {}
    for verdict in VERDICTS:
        rows = [d for d in decisions if d["screen_verdict"] == verdict]
        if not rows:
            continue
        agreed = sum(1 for d in rows if d["agreed"])
        by_verdict[verdict] = {
            "population": population[verdict],
            "sampled": len(rows),
            "agreed": agreed,
            "agreement": round(agreed / len(rows), 4),
            "human_said": dict(Counter(d["human_verdict"] for d in rows)),
        }

    agreed_total = sum(1 for d in decisions if d["agreed"])

    # Disputed mode: who the human sided with is the whole question, so it is
    # reported per queue kind. A human who splits from both screeners on the
    # control rows indicts the screening step itself, not just the disputed slice.
    disputed_stats: dict[str, dict] = {}
    scored = [d for d in decisions if "second_verdict" in d]
    if scored:
        for kind in ("disputed", "control"):
            rows = [d for d in scored if d.get("queue_kind") == kind]
            if not rows:
                continue
            disputed_stats[kind] = {
                "audited": len(rows),
                "sided_with_screener": sum(1 for d in rows if d["agreed"]),
                "sided_with_second": sum(1 for d in rows if d["agreed_with_second"]),
                "sided_with_neither": sum(
                    1 for d in rows if not d["agreed"] and not d["agreed_with_second"]
                ),
            }

    # Split by how the decision was reached. A blind decision is stronger
    # evidence than one taken in a pass that reveals the screener between
    # rows, and averaging the two would hide which kind the gold set rests on.
    by_mode: dict[str, dict] = {}
    for mode, wanted in (("blind", True), ("revealed", False)):
        rows = [d for d in decisions if bool(d.get("blind")) is wanted]
        if rows:
            agreed = sum(1 for d in rows if d["agreed"])
            by_mode[mode] = {
                "audited": len(rows),
                "agreed": agreed,
                "agreement": round(agreed / len(rows), 4),
            }

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sampled": len(decisions),
        "by_mode": by_mode,
        "target_sample": SAMPLE_SIZE,
        "seed": SEED,
        "screened_population": dict(population),
        "agreement_overall": round(agreed_total / len(decisions), 4) if decisions else 0.0,
        "by_verdict": by_verdict,
        "by_queue_kind": disputed_stats,
        "agreement_with_second": (
            round(sum(1 for d in scored if d["agreed_with_second"]) / len(scored), 4)
            if scored
            else None
        ),
        "sampling": (
            "by_mode is confounded: the revealed rows are disputes plus "
            "controls, the blind rows are agreed keeps. The gap between those "
            "two rates is population, not anchoring. Compare like with like -- "
            "revealed controls against blind agreed keeps."
        ),
    }


def write_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def restore_progress(
    decisions: list[dict], total: int, queue: set[str] | None = None
) -> Progress:
    """Counts restored from the decision log, limited to `queue` when given.

    The log accumulates across every queue ever audited, so counting all of it
    against the queue in hand puts a number on screen that describes neither.
    That is the confusion that once printed "audited 62 of 60".
    """
    progress = Progress(total=total)
    for row in decisions:
        if queue is not None and row["qid"] not in queue:
            continue
        progress.decided.add(row["qid"])
        progress.agreed += bool(row["agreed"])
    return progress


def decision_row(row: dict, human: str, blind: bool = False) -> dict:
    """One decision, plus how it was reached.

    `blind` records that the row was decided with no screener verdict shown at
    any point, not even afterwards. Two rows decided under the two modes are
    not the same evidence -- a verdict reached after seeing the screener's on
    the previous row is a weaker thing than one reached cold -- so the mode is
    stamped here rather than inferred later from timestamps.
    """
    decision = {
        "qid": row["qid"],
        "human_verdict": human,
        "screen_verdict": row["screen_verdict"],
        "screen_rule": row.get("screen_rule"),
        "agreed": human == row["screen_verdict"],
        "question": audited_question(row),
        "blind": blind,
        "decided_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if "queue_kind" in row:
        decision["queue_kind"] = row["queue_kind"]
    if "second_verdict" in row:
        decision["second_verdict"] = row["second_verdict"]
        decision["agreed_with_second"] = human == row["second_verdict"]
    return decision


def run(
    sample: list[dict],
    chunks: dict[str, str],
    decisions_path: Path,
    blind: bool = False,
) -> list[dict]:
    """Audit each sampled row blind, then reveal. Every decision hits disk.

    `blind` suppresses the reveal as well. The reveal is harmless for the row
    it follows -- that verdict is already on disk -- but across a full pass it
    is a running commentary on how often you and the screener agree, and a
    pass whose purpose is to stand alone as the authority must not be able to
    drift toward or away from the thing it is replacing.
    """
    decisions = load_jsonl(decisions_path)
    progress = restore_progress(
        decisions, total=len(sample), queue={row["qid"] for row in sample}
    )
    progress.blind = blind

    for row in sample:
        if row["qid"] in progress.decided:
            continue

        chunk_text = chunks.get(row["gold_chunk_ids"][0], "[chunk not found in the store]")
        clear_screen()
        print(render_question(row, chunk_text, progress))

        while True:
            key = read_key()
            if key in {KEEP, FIX, DROP, QUIT}:
                break
        if key == QUIT:
            print("\nSaved. Re-run to pick up where you left off.")
            return decisions

        human = _KEYS[key]
        decision = decision_row(row, human, blind=blind)
        append_jsonl(decisions_path, decision)
        decisions.append(decision)
        progress.decided.add(row["qid"])
        progress.agreed += bool(decision["agreed"])

        if blind:
            continue

        print(render_reveal(row, human))
        if read_key() == QUIT:
            print("\nSaved. Re-run to pick up where you left off.")
            return decisions

    return decisions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.goldset.audit",
        description="Audit a stratified sample of the screener's verdicts, blind.",
    )
    parser.add_argument("--screened", type=Path, default=DEFAULT_SCREENED)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--corpus-id", default="demo")
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--second-opinion",
        type=Path,
        default=None,
        help=(
            "a second screener's verdicts. Switches the queue from a stratified "
            "sample to every disputed row plus --controls agreed keeps."
        ),
    )
    parser.add_argument("--controls", type=int, default=CONTROL_ROWS)
    parser.add_argument(
        "--control-pool",
        type=Path,
        default=None,
        help=(
            "a file of qids, one per line, limiting which agreed keeps may be "
            "drawn as controls. Use it to aim the controls at the slice of the "
            "agreed majority whose agreement is actually in question."
        ),
    )
    parser.add_argument(
        "--control-seed",
        type=int,
        default=None,
        help=(
            "seed for the control draw alone; defaults to --seed. A control is "
            "only a control while nobody can recompute which rows it picked, so "
            "a re-blinded draw needs a seed that is not the published one."
        ),
    )
    parser.add_argument(
        "--queue-in",
        type=Path,
        default=None,
        help=(
            "audit a queue prepared earlier instead of drawing one. The point is "
            "separation: whoever sits the audit never handles the control draw, "
            "and so cannot know which rows are controls."
        ),
    )
    parser.add_argument(
        "--full-coverage",
        action="store_true",
        help=(
            "queue every row entering the gold set that you have not decided, "
            "instead of sampling. Turns the gold set from screener-adjudicated "
            "into human-adjudicated; implies --blind."
        ),
    )
    parser.add_argument(
        "--cover-rejections",
        action="store_true",
        help=(
            "with --full-coverage, also queue rows dropped on screener "
            "agreement alone. A wrongly dropped row is invisible in the "
            "finished gold set, so covering only the keeps leaves the "
            "coverage claim true in one direction."
        ),
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help=(
            "never show a screener verdict, not even after the keypress, and "
            "withhold the running agreement rate. Stamps blind: true on every "
            "decision so the two modes can be reported apart."
        ),
    )
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument(
        "--queue-out",
        type=Path,
        default=DEFAULT_QUEUE,
        help="where to record the queue that was actually audited",
    )
    parser.add_argument(
        "--queue-only",
        action="store_true",
        help=(
            "write the queue and stop, without starting the interactive pass. "
            "The queue is built by the same seeded draw either way, so the rows "
            "listed here are exactly the rows a later run will present."
        ),
    )
    args = parser.parse_args(argv)

    rows = load_jsonl(args.screened)
    if not rows:
        print(f"No rows in {args.screened}. Run backend.goldset.screen first.")
        return 1

    population: Counter[str] = Counter(
        r["screen_verdict"] for r in rows if r.get("screen_verdict") in VERDICTS
    )

    # A coverage pass replaces the screener rather than measuring it, so
    # showing a screener verdict at any point would defeat the exercise.
    blind = args.blind or args.full_coverage

    if args.full_coverage:
        if not args.second_opinion:
            print("--full-coverage needs --second-opinion to know which rows agreed.")
            return 1
        second_rows = load_jsonl(args.second_opinion)
        if not second_rows:
            print(f"No rows in {args.second_opinion}.")
            return 1
        decided = {d["qid"]: d for d in load_jsonl(args.decisions)}
        sample = build_coverage_queue(
            rows,
            decided,
            {r["qid"]: r for r in second_rows},
            exclusions=load_hand_map(args.exclusions),
            seed=args.seed,
            include_rejections=args.cover_rejections,
        )
        print(f"screened     {sum(population.values())} rows from {args.screened}")
        print(f"decided      {len(decided)} rows already carry your verdict")
        print(f"queue        {len(sample)} rows (seed {args.seed}), shuffled")
        print("mode         blind: no screener verdict shown, before or after")
    elif args.queue_in:
        sample = load_jsonl(args.queue_in)
        if not sample:
            print(f"No rows in {args.queue_in}.")
            return 1
        # The length and nothing else. This mode exists so that whoever sits
        # the audit does not know the queue's composition, and a breakdown by
        # kind printed here would hand it straight back.
        print(f"screened     {sum(population.values())} rows from {args.screened}")
        print(f"queue        {len(sample)} rows from {args.queue_in}")
    elif args.second_opinion:
        second = load_jsonl(args.second_opinion)
        if not second:
            print(f"No rows in {args.second_opinion}.")
            return 1
        sample = build_queue(
            rows,
            second,
            args.controls,
            args.seed,
            control_pool=load_qids(args.control_pool) if args.control_pool else None,
            control_seed=args.control_seed,
        )
        kinds: Counter[str] = Counter(r["queue_kind"] for r in sample)
        judged = {r["qid"] for r in second} & {r["qid"] for r in rows}
        print(f"screened     {sum(population.values())} rows from {args.screened}")
        print(f"second       {args.second_opinion} ({len(judged)} rows in common)")
        print(f"queue        {len(sample)} rows (seed {args.seed}), shuffled:")
        print(f"  disputed   {kinds['disputed']} of {len(judged)} "
              f"({kinds['disputed'] / len(judged):.0%} disagreement)")
        print(f"  control    {kinds['control']} rows both screeners kept")
    else:
        sample = stratified_sample(rows, args.n, args.seed)
        drawn: Counter[str] = Counter(r["screen_verdict"] for r in sample)
        print(f"screened     {sum(population.values())} rows from {args.screened}")
        print(f"sampling     {len(sample)} (seed {args.seed}), stratified:")
        for verdict in VERDICTS:
            print(f"  {verdict:<10} {drawn[verdict]:>3} of {population[verdict]}")

    if not sample:
        print("Nothing to audit.")
        return 1

    with dense_store.connect() as conn:
        chunks = {c.chunk_id: c.text for c in dense_store.fetch_chunks(conn, args.corpus_id)}

    # Not rewritten when the queue was read from disk: it is already the record.
    if args.queue_out and not args.queue_in:
        write_jsonl(args.queue_out, sample)
        print(f"queue saved  {args.queue_out}")

    if args.queue_only:
        decided = {d["qid"] for d in load_jsonl(args.decisions)}
        pending = [r for r in sample if r["qid"] not in decided]
        print(f"\nqueued       {len(sample)} rows, {len(pending)} still undecided")
        print("Nothing audited. Re-run without --queue-only to work through them.")
        return 0

    if blind:
        print("\nNo screener verdict is shown at any point, and no agreement rate.")
    else:
        print("\nEvery verdict is hidden until after your keypress.")
        if args.second_opinion:
            print("Controls are shuffled in and look identical to disputed rows.")
    print("Press any key to begin.")
    read_key()

    decisions = run(sample, chunks, args.decisions, blind=blind)
    if not decisions:
        print("\nNothing audited.")
        return 1

    results = summarise(decisions, population)
    write_results(args.results, results)

    # Counted against this queue, not against the whole decision log. The log
    # accumulates across queues, so comparing its length to the queue's once
    # printed "audited 62 of 60".
    target = len(sample)
    decided_qids = {d["qid"] for d in decisions}
    audited = sum(1 for r in sample if r["qid"] in decided_qids)
    carried = results["sampled"] - audited
    print(f"\naudited      {audited} of {target} in this queue")
    if carried:
        print(f"decisions    {results['sampled']} on file, {carried} from an earlier queue")
    print(f"agreement    {results['agreement_overall']:.0%} with the screener")
    if results["agreement_with_second"] is not None:
        print(f"             {results['agreement_with_second']:.0%} with the second opinion")
    for verdict in VERDICTS:
        stats = results["by_verdict"].get(verdict)
        if stats:
            print(
                f"  {verdict:<10} {stats['agreement']:.0%}  "
                f"({stats['agreed']}/{stats['sampled']} of {stats['population']} screened)"
            )
    for kind, stats in results["by_queue_kind"].items():
        print(
            f"\n{kind:<12} {stats['audited']} audited: "
            f"{stats['sided_with_screener']} sided with the screener, "
            f"{stats['sided_with_second']} with the second opinion, "
            f"{stats['sided_with_neither']} with neither"
        )
    for mode, stats in results["by_mode"].items():
        print(
            f"\n{mode:<12} {stats['audited']} decided, "
            f"{stats['agreed']} of them agreeing with the screener"
        )
    print(f"\nresults      {args.results}")
    print(f"decisions    {args.decisions}")
    if audited < target:
        print(
            f"\nINCOMPLETE: {target - audited} rows left. "
            "Re-run to finish; the numbers above are from a partial sample."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
