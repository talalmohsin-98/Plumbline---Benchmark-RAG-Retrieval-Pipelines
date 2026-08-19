"""Mechanical candidate generation for equally-correct gold chunks.

Documentation repeats itself. Five FastAPI chunks each tell the reader to
install `python-multipart`; the LangSmith deployment sentence is byte-identical
in the LangChain and LangGraph pages. A lane that returns the other copy has
answered the question, and a single label would score it wrong -- so
`gold_chunk_ids` is a list and recall counts a hit on any member.

This module finds *candidates* and nothing more. It flags every other chunk
that shares a sentence of `MIN_SENTENCE_WORDS` or more with a gold chunk, which
is deliberately far too generous: the first pass flagged 77 of 98 rows and only
21 survived. Almost all of the rest shared boilerplate ("Before you begin,
ensure you have the following"), an import block, or the 64-token overlap
between adjacent chunks of one document -- none of which carries an answer.

**Nothing here may write `data/multilabel.json`.** Whether a chunk genuinely
states the answer is a judgement about meaning, and a shared sentence is not
evidence of it. The output is a review queue; a human decides, and the reason
they give is what lands in the map. Auto-accepting these would put chunks in
`gold_chunk_ids` that do not answer the question, which inflates recall for
every lane at once and is invisible in the finished file.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from backend.goldset.assemble import load_hand_map
from backend.goldset.console import load_jsonl, write_jsonl
from backend.retrieval import dense_store

if TYPE_CHECKING:
    from backend.retrieval.dense_store import StoredChunk

DEFAULT_GOLDSET = Path("data/goldset.jsonl")
DEFAULT_MULTILABEL = Path("data/multilabel.json")
DEFAULT_OUT = Path("data/multilabel_candidates.jsonl")

# Eight words. Shorter runs are how boilerplate matches: "Before you begin" and
# "Needs to be installed separately" are five and six, and they appear in dozens
# of chunks without carrying anything. Eight is long enough that an accidental
# collision is rare and short enough to catch a one-line duplicated instruction,
# which is the case this exists for. Measured on the first pass: at 8 words the
# rule flagged 77 of 98 rows; at 5 it flagged essentially all of them, which is
# the same as flagging none.
MIN_SENTENCE_WORDS = 8

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+|\n+")
_WORD = re.compile(r"[a-z0-9]+")


def sentences(text: str) -> set[str]:
    """Normalised sentences of `MIN_SENTENCE_WORDS`+ words.

    Normalised to lowercase word tokens joined by single spaces, so that two
    copies differing only in markdown emphasis, link syntax, or wrapping still
    match. Punctuation is dropped for the same reason: `**Rust**` and `Rust`
    are the same word in two renderings of one sentence.
    """
    found: set[str] = set()
    for raw in _SENTENCE_SPLIT.split(text):
        words = _WORD.findall(raw.lower())
        if len(words) >= MIN_SENTENCE_WORDS:
            found.add(" ".join(words))
    return found


@dataclass(frozen=True)
class Candidate:
    """One other chunk that shares a long sentence with a gold chunk."""

    qid: str
    question: str
    gold_chunk_id: str
    candidate_chunk_id: str
    shared: str

    def as_row(self) -> dict[str, object]:
        return {
            "qid": self.qid,
            "question": self.question,
            "gold_chunk_id": self.gold_chunk_id,
            "candidate_chunk_id": self.candidate_chunk_id,
            "shared_sentence": self.shared,
            # Never pre-filled. A default here would be auto-accept wearing a
            # different hat: whoever reviews must state the verdict themselves.
            "verdict": None,
            "why": None,
        }


def build_index(chunks: list[StoredChunk]) -> dict[str, set[str]]:
    """Map every chunk id to its set of long sentences."""
    return {chunk.chunk_id: sentences(chunk.text) for chunk in chunks}


def candidates_for(
    row: dict,
    index: dict[str, set[str]],
    by_sentence: dict[str, set[str]],
) -> list[Candidate]:
    """Every chunk sharing a long sentence with this row's gold chunks.

    Chunks already listed on the row are skipped -- they are labels, not
    candidates. The row's own multi-labels are included in that check, so a
    re-run does not re-offer what a previous pass already accepted.
    """
    already = set(row["gold_chunk_ids"])
    found: dict[str, Candidate] = {}
    for gold_id in row["gold_chunk_ids"]:
        for sentence in index.get(gold_id, set()):
            for other in by_sentence.get(sentence, set()):
                if other in already or other in found:
                    continue
                found[other] = Candidate(
                    qid=row["qid"],
                    question=row["question"],
                    gold_chunk_id=gold_id,
                    candidate_chunk_id=other,
                    shared=sentence,
                )
    return list(found.values())


def generate(rows: list[dict], chunks: list[StoredChunk]) -> list[Candidate]:
    """Candidates for every row, in qid order.

    Run over the whole gold set rather than only the new rows. The cross-batch
    case needs both directions: a new row's answer may sit in a chunk already
    labelled for an older question, and an older row's answer may sit in a
    chunk that only became a label when the new rows landed. Generating for
    every row and filtering afterwards covers both without a second code path.
    """
    index = build_index(chunks)
    by_sentence: dict[str, set[str]] = defaultdict(set)
    for chunk_id, sents in index.items():
        for sentence in sents:
            by_sentence[sentence].add(chunk_id)

    found: list[Candidate] = []
    for row in sorted(rows, key=lambda r: r["qid"]):
        found.extend(candidates_for(row, index, by_sentence))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.goldset.multilabel",
        description="Flag chunks that may answer a gold question as well as its label does.",
    )
    parser.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--multilabel", type=Path, default=DEFAULT_MULTILABEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpus-id", default="demo")
    parser.add_argument(
        "--only-new",
        action="store_true",
        help=(
            "skip rows already adjudicated in the multilabel map. Cross-batch "
            "candidates still surface, because an already-adjudicated row is "
            "only skipped once its own candidates were judged."
        ),
    )
    args = parser.parse_args(argv)

    rows = load_jsonl(args.goldset)
    if not rows:
        print(f"No rows in {args.goldset}. Run backend.goldset.assemble first.")
        return 1

    adjudicated = load_hand_map(args.multilabel)
    with dense_store.connect() as conn:
        chunks = dense_store.fetch_chunks(conn, args.corpus_id)

    found = generate(rows, chunks)
    if args.only_new:
        found = [c for c in found if c.qid not in adjudicated]

    flagged = {c.qid for c in found}
    write_jsonl(args.out, [c.as_row() for c in found])

    print(f"gold set     {len(rows)} rows from {args.goldset}")
    print(f"corpus       {len(chunks)} chunks")
    print(f"rule         every other chunk sharing a {MIN_SENTENCE_WORDS}+ word sentence")
    print(f"adjudicated  {len(adjudicated)} rows already carry a decided multi-label")
    print(f"\ncandidates   {len(found)} across {len(flagged)} of {len(rows)} rows")
    print(f"written      {args.out}")
    print(
        "\nThese are candidates, not labels. The rule is deliberately over-broad: "
        "\nmost shared sentences are boilerplate or the overlap between adjacent "
        "\nchunks. Verify each by hand and record the survivors, with a reason, in "
        f"\n{args.multilabel}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
