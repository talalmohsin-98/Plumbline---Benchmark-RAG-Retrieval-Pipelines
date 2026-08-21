"""Hard negative mining from the train split only, producing reranker training pairs.

Protocol is pre-registered in `docs/02_EVALUATION_SPEC.md` §3, "Hard negative
mining", written and committed before this file existed. The five decisions
that move the result -- source list, split restriction, gold exclusion,
sampling rule, and the deliberate *non*-filtering of test-gold chunks -- are
argued there and implemented here without deviation.

    python -m training.mine_negatives

Writes `data/train_pairs.jsonl` (the Colab upload) and
`data/mining_report.json` (the diagnostics that go in the README).

The load-bearing property of this file is negative: **no test question is ever
embedded, retrieved for, or scored here.** That is enforced structurally by
`TrainOnlyCorpus`, which raises at the moment a forbidden string reaches the
retriever, and re-checked against the written file before it lands. A comment
promising it would not be worth anything.
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
from typing import Any

from backend.config import get_settings
from backend.lanes.hybrid import HybridLane
from backend.retrieval.corpus import Corpus

DEFAULT_TRAIN = Path("data/train.jsonl")
DEFAULT_TEST = Path("data/test.jsonl")
DEFAULT_PAIRS = Path("data/train_pairs.jsonl")
DEFAULT_REPORT = Path("data/mining_report.json")

# 4 negatives per positive, from EVALUATION_SPEC §3. The alternative was 8:
# more negatives per question is more signal, but with 80 questions it also
# means the loss is 89% negatives and a BinaryCrossEntropy model can score
# well by learning to say "no". 4 keeps the ratio at 1:4, which is the
# published MS MARCO cross-encoder recipe's neighbourhood.
NEGATIVES_PER_QUESTION = 4

# Seed 42 throughout this project. Mixed with the qid rather than used
# directly, so that adding or removing a row changes only that row's draw --
# with one global RNG consumed in row order, inserting a question at q005
# reshuffles the negatives of every question after it and the pair file stops
# being comparable to the previous one.
SEED = 42


class LeakGuardError(AssertionError):
    """A test question reached the retriever. Nothing mined after this is valid."""


def normalise(question: str) -> str:
    """Casefolded, whitespace-collapsed form, for comparing question strings.

    Exact equality alone is too weak a check: a question that differs from a
    test question by a trailing space or a capital letter is the same question
    for leakage purposes and would slip past `==`.
    """
    return " ".join(question.split()).casefold()


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


class TrainOnlyCorpus:
    """A `Corpus` that refuses to retrieve for, or embed, a non-train question.

    A proxy rather than a flag on `Corpus`, because the restriction belongs to
    this one job and the serving path must not carry a mode that can be set
    wrong. Everything the miner does not intercept passes straight through, so
    chunk text and warm-up behave exactly as they do everywhere else.

    Both entry points that see a question string are intercepted:
    `bm25_search` (retrieved for) and `embed_query` (embedded). Dense search
    takes an already-computed vector and never sees text, so guarding
    `embed_query` guards it too -- the only way to a vector is through here.
    """

    def __init__(self, corpus: Corpus, allowed: list[str], forbidden: list[str]) -> None:
        self._corpus = corpus
        self._allowed = {normalise(q) for q in allowed}
        self._forbidden = {normalise(q) for q in forbidden}
        self.queries_seen: list[str] = []

    def __getattr__(self, name: str) -> Any:
        # Reached only for attributes this proxy does not define -- chunks,
        # texts, sources, text_of, dense_search_vector, warm, close.
        return getattr(self._corpus, name)

    def _check(self, question: str) -> None:
        key = normalise(question)
        if key in self._forbidden:
            raise LeakGuardError(
                f"a TEST question reached the retriever during mining: {question!r}. "
                "Every pair mined in this run is contaminated; delete the output."
            )
        if key not in self._allowed:
            raise LeakGuardError(
                f"a question that is in neither split reached the retriever: {question!r}. "
                "Mining reads train.jsonl and nothing else."
            )
        self.queries_seen.append(question)

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        self._check(query)
        return self._corpus.bm25_search(query, k)

    def embed_query(self, text: str, *, prefix: bool = True) -> list[float]:
        # The warm-up embeds the literal string "warmup", which is not a
        # question and must not trip the guard. It is allowed by name rather
        # than by a general escape hatch, so nothing else can slip through.
        if text != "warmup":
            self._check(text)
        return self._corpus.embed_query(text, prefix=prefix)


# --------------------------------------------------------------------------
# Rows and pairs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One (question, chunk) training example: label 1.0 positive, 0.0 negative."""

    qid: str
    question: str
    chunk_id: str
    text: str
    source_doc: str
    label: float
    # Where the chunk sat in lane 3's fused list, or None if it was not in the
    # mined depth at all. Only positives can be None: negatives are drawn from
    # the top-`depth` by construction.
    fused_rank: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_doc": self.source_doc,
            "label": self.label,
            "fused_rank": self.fused_rank,
        }


@dataclass
class RowOutcome:
    """What mining produced for one training question, for the report."""

    qid: str
    gold_count: int
    pool_size: int
    negatives: list[str] = field(default_factory=list)
    negative_ranks: list[int] = field(default_factory=list)
    gold_rank: int | None = None

    @property
    def shortfall(self) -> int:
        return max(0, NEGATIVES_PER_QUESTION - len(self.negatives))


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m backend.goldset.split` first."
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    missing = [r.get("qid") for r in rows if not r.get("gold_chunk_ids")]
    if missing:
        raise ValueError(f"{len(missing)} row(s) in {path} have no gold_chunk_ids: {missing[:5]}")
    return sorted(rows, key=lambda r: r["qid"])


def choose_positive(gold_chunk_ids: list[str]) -> str:
    """The one positive for a row: the lowest chunk id among its gold chunks.

    Multi-labelled rows carry up to five gold chunks and they are duplicates of
    each other -- five FastAPI pages each saying to install `python-multipart`
    -- so any of them is a true positive and the only thing that matters is
    which one is picked *deterministically*.

    Sorted-first rather than highest-ranked-in-the-fused-list, which was the
    alternative. Picking the highest-ranked would make the training data a
    function of the retriever's current ranking: change `rrf_k` and the pair
    file silently changes with it, and the fine-tune stops being comparable to
    the one before. One positive per row rather than all of them, so a
    five-label row does not carry five times the weight of a single-label one.
    """
    return sorted(gold_chunk_ids)[0]


def mine_row(
    row: dict,
    fused: list[tuple[str, float]],
    depth: int,
    negatives_per_question: int,
    seed: int,
) -> RowOutcome:
    """Choose one row's negatives from its fused list. Pure: no corpus, no I/O.

    Split out from `mine` so the sampling rule can be tested against a
    hand-written fused list, with no database and no model.
    """
    qid = row["qid"]
    gold = set(row["gold_chunk_ids"])
    ranked = fused[:depth]
    rank_of = {chunk_id: index for index, (chunk_id, _) in enumerate(ranked, start=1)}

    # Decision 3: every gold chunk for THIS row is removed, not just the one
    # chosen as the positive. A multi-labelled row's other four chunks say the
    # same thing as its positive; sampling one as a negative teaches the model
    # that a correct answer is wrong, on a question where the correct answer is
    # sitting in the same batch with label 1.0.
    pool = [chunk_id for chunk_id, _ in ranked if chunk_id not in gold]

    rng = random.Random(f"{seed}:{qid}")
    take = min(negatives_per_question, len(pool))
    # Decision 4: uniform over the whole top-`depth` remainder, not the top few.
    # `sample` rather than `choices`: a chunk must not appear twice as a
    # negative for the same question.
    negatives = sorted(rng.sample(pool, take))

    return RowOutcome(
        qid=qid,
        gold_count=len(gold),
        pool_size=len(pool),
        negatives=negatives,
        negative_ranks=[rank_of[c] for c in negatives],
        gold_rank=min((rank_of[c] for c in gold if c in rank_of), default=None),
    )


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def mine(
    corpus: Corpus,
    train_rows: list[dict],
    test_rows: list[dict],
    *,
    depth: int,
    negatives_per_question: int = NEGATIVES_PER_QUESTION,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[list[Pair], list[RowOutcome]]:
    """Run lane 3 over every train question and mine pairs from its fused list."""
    guarded = TrainOnlyCorpus(
        corpus,
        allowed=[r["question"] for r in train_rows],
        forbidden=[r["question"] for r in test_rows],
    )
    lane = HybridLane(guarded)  # type: ignore[arg-type]  # proxy is Corpus-shaped
    lane.warm()

    pairs: list[Pair] = []
    outcomes: list[RowOutcome] = []
    for index, row in enumerate(train_rows, start=1):
        fused, _ = lane.fuse(row["question"])
        outcome = mine_row(row, fused, depth, negatives_per_question, seed)
        outcomes.append(outcome)

        positive = choose_positive(row["gold_chunk_ids"])
        rank_of = {c: i for i, (c, _) in enumerate(fused[:depth], start=1)}
        pairs.append(
            Pair(
                qid=row["qid"],
                question=row["question"],
                chunk_id=positive,
                text=corpus.text_of(positive),
                source_doc=corpus.sources.get(positive, ""),
                label=1.0,
                fused_rank=rank_of.get(positive),
            )
        )
        for chunk_id in outcome.negatives:
            pairs.append(
                Pair(
                    qid=row["qid"],
                    question=row["question"],
                    chunk_id=chunk_id,
                    text=corpus.text_of(chunk_id),
                    source_doc=corpus.sources.get(chunk_id, ""),
                    label=0.0,
                    fused_rank=rank_of[chunk_id],
                )
            )
        if verbose and index % 20 == 0:
            print(f"    {index}/{len(train_rows)}", file=sys.stderr)

    assert_train_only(guarded, train_rows, test_rows)
    return pairs, outcomes


def assert_train_only(
    guarded: TrainOnlyCorpus, train_rows: list[dict], test_rows: list[dict]
) -> None:
    """Re-check after the fact what the guard enforced during the run.

    The guard raises at the moment a bad string arrives, which is the check
    that matters. This one catches the case the guard cannot: a question that
    was never issued at all, i.e. a row silently skipped. A miner that
    quietly drops rows produces a smaller, weaker training set and no error.
    """
    issued = {normalise(q) for q in guarded.queries_seen}
    expected = {normalise(r["question"]) for r in train_rows}
    if issued != expected:
        raise LeakGuardError(
            f"mining issued {len(issued)} distinct questions but the train split has "
            f"{len(expected)}: {len(expected - issued)} never retrieved, "
            f"{len(issued - expected)} unexpected."
        )
    forbidden = {normalise(r["question"]) for r in test_rows}
    if issued & forbidden:
        raise LeakGuardError(f"{len(issued & forbidden)} test question(s) were retrieved for")


def assert_pairs_are_clean(
    pairs: list[Pair], train_rows: list[dict], test_rows: list[dict]
) -> None:
    """Refuse to write a pair file that contains anything from the test split.

    Third check on the same property, on the artefact itself rather than on the
    process that produced it. This is the one that would still fire if the
    guard were removed or the mining loop rewritten, which is exactly why it
    reads the output rather than trusting the run.

    Note what is deliberately NOT asserted: that no negative is a gold chunk
    for some test question. It is *reported* instead -- see
    `build_report` and EVALUATION_SPEC §3, decision 5. Asserting it would mean
    consulting the test answer key to shape training.
    """
    test_qids = {r["qid"] for r in test_rows}
    train_qids = {r["qid"] for r in train_rows}
    offending = sorted({p.qid for p in pairs} & test_qids)
    if offending:
        raise LeakGuardError(f"pair file contains test qids: {offending[:5]}")
    stray = sorted({p.qid for p in pairs} - train_qids)
    if stray:
        raise LeakGuardError(f"pair file contains qids in neither split: {stray[:5]}")

    test_questions = {normalise(r["question"]) for r in test_rows}
    leaked = sorted({p.question for p in pairs if normalise(p.question) in test_questions})
    if leaked:
        raise LeakGuardError(f"pair file contains test question text: {leaked[:3]}")

    empty = sorted({p.chunk_id for p in pairs if not p.text.strip()})
    if empty:
        raise LeakGuardError(
            f"{len(empty)} pair(s) have empty chunk text: {empty[:5]}. "
            "The chunk id does not resolve in this corpus -- wrong --corpus-id?"
        )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def build_report(
    pairs: list[Pair],
    outcomes: list[RowOutcome],
    test_rows: list[dict],
    *,
    depth: int,
    negatives_per_question: int,
    seed: int,
    corpus_id: str,
    chunks_in_corpus: int,
) -> dict[str, Any]:
    """The mining diagnostics. Every number the README quotes about mining is here."""
    per_row = Counter(len(o.negatives) for o in outcomes)
    short = sorted(o.qid for o in outcomes if o.shortfall)
    negatives = [p for p in pairs if p.label == 0.0]
    positives = [p for p in pairs if p.label == 1.0]

    # Pre-registered decision 5's diagnostic. Chunks that are gold for a test
    # question, sampled as negatives against a train question. `split.py`
    # guarantees these are not gold for the train question -- they are a
    # different question's answer that happens to rank well here -- and they
    # are left in, because removing them would use test labels to shape
    # training. Counted so the cost is visible rather than absorbed.
    test_gold: dict[str, list[str]] = {}
    for row in test_rows:
        for chunk_id in row["gold_chunk_ids"]:
            test_gold.setdefault(chunk_id, []).append(row["qid"])
    collisions = [
        {"train_qid": p.qid, "chunk_id": p.chunk_id, "gold_for_test_qids": test_gold[p.chunk_id]}
        for p in negatives
        if p.chunk_id in test_gold
    ]

    affected_test_qids = sorted({q for c in collisions for q in c["gold_for_test_qids"]})

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": "docs/02_EVALUATION_SPEC.md §3, pre-registered 2026-08-21",
        "source_lane": "hybrid_rrf",
        "corpus": {"id": corpus_id, "chunks": chunks_in_corpus},
        "parameters": {
            "mine_depth": depth,
            "negatives_per_question": negatives_per_question,
            "seed": seed,
            "positive_per_question": 1,
        },
        "questions_mined": len(outcomes),
        "pairs": {
            "total": len(pairs),
            "positives": len(positives),
            "negatives": len(negatives),
            "ratio": round(len(negatives) / len(positives), 3) if positives else 0.0,
        },
        "negatives_per_positive": {
            "mean": round(len(negatives) / len(outcomes), 3) if outcomes else 0.0,
            "distribution": {str(count): rows for count, rows in sorted(per_row.items())},
            "rows_below_target": len(short),
            "rows_below_target_qids": short,
            "rows_with_zero_negatives": sum(1 for o in outcomes if not o.negatives),
        },
        # How often the positive was in the mined band at all. A row whose gold
        # chunk lane 3 never surfaced still contributes its positive, and it is
        # the most valuable row in the file -- but it is also the row where the
        # 20 candidates are all wrong, so the count belongs in the report.
        "positive_in_mined_depth": {
            "rows": sum(1 for o in outcomes if o.gold_rank is not None),
            "of": len(outcomes),
            "median_gold_rank": _median([o.gold_rank for o in outcomes if o.gold_rank is not None]),
        },
        "negative_rank_distribution": {
            str(rank): count
            for rank, count in sorted(
                Counter(r for o in outcomes for r in o.negative_ranks).items()
            )
        },
        "distinct_chunks_used_as_negatives": len({p.chunk_id for p in negatives}),
        "test_gold_chunks_mined_as_negatives": {
            "count": len(collisions),
            "of_negatives": len(negatives),
            "fraction": round(len(collisions) / len(negatives), 4) if negatives else 0.0,
            "distinct_chunks": len({c["chunk_id"] for c in collisions}),
            # The number that actually matters. 24 collisions spread over 16
            # chunks sounds small; those 16 chunks are the gold answers to this
            # many *test* questions, which is the population lane 6 is scored
            # on. Reported separately because the raw collision count
            # understates the exposure by a factor of the multi-labelling.
            "distinct_test_questions_affected": len(affected_test_qids),
            "of_test_questions": len(test_rows),
            "test_qids_affected": affected_test_qids,
            "note": (
                "NOT filtered. These chunks are gold for a test question and not for the "
                "train question they were mined against. Removing them would consult the "
                "test answer key to shape training, which is a leak in the flattering "
                "direction. Left in and reported per EVALUATION_SPEC §3, decision 5. "
                "Any cost to lane 6 is a real property of training on 80 questions."
            ),
            "collisions": collisions,
        },
    }


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def format_report(report: dict[str, Any]) -> str:
    """The terminal summary. What goes in the commit message a week later."""
    npp = report["negatives_per_positive"]
    pos = report["positive_in_mined_depth"]
    collisions = report["test_gold_chunks_mined_as_negatives"]
    lines = [
        f"questions mined     {report['questions_mined']}  (train split only)",
        f"pairs               {report['pairs']['total']}  "
        f"({report['pairs']['positives']} positive, {report['pairs']['negatives']} negative)",
        f"negatives/positive  {npp['mean']:.2f} mean, target "
        f"{report['parameters']['negatives_per_question']}",
        "  distribution      "
        + ", ".join(f"{k} neg x{v}" for k, v in npp["distribution"].items()),
        f"  short rows        {npp['rows_below_target']}"
        + (f"  {npp['rows_below_target_qids']}" if npp["rows_below_target_qids"] else ""),
        f"  zero-negative     {npp['rows_with_zero_negatives']}",
        f"positive in top-{report['parameters']['mine_depth']}    "
        f"{pos['rows']}/{pos['of']}  (median gold rank {pos['median_gold_rank']})",
        f"distinct negatives  {report['distinct_chunks_used_as_negatives']} chunks",
        f"test-gold negatives {collisions['count']}/{collisions['of_negatives']} "
        f"({collisions['fraction']:.2%})  -- reported, NOT filtered",
        f"  test qs exposed   {collisions['distinct_test_questions_affected']}"
        f"/{collisions['of_test_questions']} test questions have a gold chunk "
        f"trained as a negative",
    ]
    return "\n".join(lines)


def write_pairs(path: Path, pairs: list[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.as_dict(), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.mine_negatives",
        description="Mine hard negatives from the train split for the lane 6 fine-tune.",
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--corpus-id", default="demo")
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="mine from the top N of lane 3's fused list (default: rerank_depth)",
    )
    parser.add_argument("--negatives", type=int, default=NEGATIVES_PER_QUESTION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    depth = args.depth if args.depth is not None else get_settings().rerank_depth

    train_rows = load_rows(args.train)
    test_rows = load_rows(args.test)
    overlap = {r["qid"] for r in train_rows} & {r["qid"] for r in test_rows}
    if overlap:
        raise LeakGuardError(
            f"train and test share qids before mining even starts: {sorted(overlap)[:5]}"
        )

    print(f"train      {len(train_rows)} questions from {args.train}", file=sys.stderr)
    print(f"test       {len(test_rows)} questions, forbidden to the retriever", file=sys.stderr)
    print(
        f"mining     lane 3 fused list, top {depth}, "
        f"{args.negatives} negatives, seed {args.seed}",
        file=sys.stderr,
    )

    corpus = Corpus(args.corpus_id)
    try:
        pairs, outcomes = mine(
            corpus,
            train_rows,
            test_rows,
            depth=depth,
            negatives_per_question=args.negatives,
            seed=args.seed,
        )
        assert_pairs_are_clean(pairs, train_rows, test_rows)
        report = build_report(
            pairs,
            outcomes,
            test_rows,
            depth=depth,
            negatives_per_question=args.negatives,
            seed=args.seed,
            corpus_id=corpus.corpus_id,
            chunks_in_corpus=len(corpus.chunks),
        )
    finally:
        corpus.close()

    write_pairs(args.out, pairs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print(format_report(report))
    print()
    print(f"pairs      {args.out}")
    print(f"report     {args.report}")
    print("\nNo test question was embedded, retrieved for, or scored in this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
