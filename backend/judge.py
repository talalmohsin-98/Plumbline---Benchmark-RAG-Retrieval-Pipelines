"""Groundedness scoring of generated answers against their retrieved context.

Two steps, both against the same lane's retrieved chunks:

1. **Generate.** Write an answer to the question using only those chunks.
2. **Judge.** Split the answer into sentences and decide, per sentence, whether
   the chunks support it. The answer is grounded only if every sentence is.

    python -m backend.judge --split test --lanes hybrid_rerank --limit 10

The measuring instrument has its own error rate, so it is measured:
`backend.judge_calibrate` builds a blind 30-sample queue, the author labels it
with no judge verdict visible, and the agreement rate is published next to the
groundedness numbers. EVALUATION_SPEC §2 requires this and it is the part of
this file worth reading.

**The reasoning-token trap.** `openai/gpt-oss-20b` is a reasoning model: it
emits reasoning tokens before any content, and they count against `max_tokens`.
Lane 5 hit this on Day 2 at `max_tokens=160` -- empty `content`, `finish_reason
= "length"`, every query failed. This judge asks for strictly more thinking than
HyDE did (several sentences, each against 10 chunks), so the budget is set from
a measurement rather than a guess and an empty completion raises by name instead
of being silently scored as ungrounded. Scoring it ungrounded would be the worst
available failure: a broken judge would look like a badly-grounded lane.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.config import get_settings
from backend.lanes import REGISTRY, build_lanes
from backend.retrieval.corpus import Corpus

if TYPE_CHECKING:
    from groq import Groq

DEFAULT_SPLITS = {"test": Path("data/test.jsonl"), "train": Path("data/train.jsonl")}
DEFAULT_OUT = Path("data/groundedness.json")
DEFAULT_VERDICTS = Path("data/judge_verdicts.jsonl")

# Chunks handed to the generator and to the judge. The same k the metrics score
# at, so "grounded in what the lane retrieved" means grounded in the list a
# caller would actually have received -- not in a wider context the lane never
# returned.
CONTEXT_DEPTH = 10

TEMPERATURE = 0.0

# Two separate budgets, because the two calls do different amounts of thinking
# and one number covering both would be wrong for one of them.
#
# Measured over 12 test questions at reasoning_effort="low", reproducible with
# `python -m backend.judge --limit 12 --measure`. Completion tokens INCLUDE the
# reasoning tokens, which is the figure that matters here:
#
#   call        n   min   median   max   budget   headroom
#   generate   12    25       35    86      800       9.3x
#   judge      12    71       89   126     1600      12.7x
#
# An earlier draft of this comment guessed ~250 and ~400 before measuring. It
# was wrong by 3x, which is the argument for the `--measure` flag existing.
#
# The headroom looks extravagant and is deliberate. `max_tokens` is a cap, not
# a reservation -- Groq bills the tokens actually emitted -- so the cost of
# setting it high is nothing, and the failure it guards against is severe and
# silent: if the reasoning phase spends the whole budget, `content` comes back
# empty with finish_reason="length", which does not look like truncation until
# you know the model thinks before it writes. Lane 5 lost every query to this
# on Day 2. The asymmetry between the two errors is the whole justification.
GENERATE_MAX_TOKENS = 800
JUDGE_MAX_TOKENS = 1600

# The same setting lane 5 landed on, for the same reason: "medium" roughly
# triples the completion tokens and the latency. Deciding whether a sentence
# appears in a passage is a lookup, not a deliberation.
REASONING_EFFORT = "low"

GENERATE_SYSTEM = (
    "You answer questions using only the numbered context passages you are given. "
    "Write two to four short sentences. Do not use any knowledge that is not in "
    "the passages. Do not cite passage numbers, do not hedge, and do not mention "
    "the passages or that you were given context. If the passages do not answer "
    "the question, say so in one sentence and stop."
)

GENERATE_USER = """Context passages:
{context}

Question: {question}

Answer using only the passages above."""

JUDGE_SYSTEM = (
    "You check whether each sentence of an answer is supported by the given "
    "context passages. A sentence is supported only if the passages state it or "
    "directly entail it. Plausible, widely-known, or merely-not-contradicted is "
    "NOT supported. A sentence that says the passages do not contain the answer "
    "is supported if that is true of the passages. "
    'Reply with JSON only: {"verdicts": [{"n": 1, "supported": true, '
    '"reason": "..."}]} with one entry per numbered sentence, in order. '
    "Keep each reason under fifteen words."
)

JUDGE_USER = """Context passages:
{context}

Question: {question}

Answer sentences:
{sentences}

For each numbered sentence, is it supported by the context passages?"""


class JudgeError(RuntimeError):
    """The judge could not produce a verdict. Never scored as 'ungrounded'."""


class DailyBudgetExhausted(RuntimeError):
    """The account's tokens-per-day budget is gone. Retrying cannot help.

    Distinct from `JudgeError` on purpose. A judge error is a fact about one
    answer and is recorded against it; this is a fact about the account, and
    every remaining call in the run will fail the same way. Raised out of the
    loop so the run stops and can be resumed when the bucket refills.
    """


class EmptyCompletionError(JudgeError):
    """The model returned no content. Usually the reasoning-token trap."""


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

# Split on a sentence-ending mark followed by whitespace and a capital or a
# digit. Deliberately not nltk or spacy: CLAUDE.md says not to add a library
# for something a dozen lines of standard library handles, and the input here
# is two to four sentences of generated technical prose, not arbitrary text.
#
# The abbreviation problem is real but bounded -- "e.g." and "i.e." are the
# ones that appear in this corpus's register -- so they are protected
# explicitly rather than by a general model of English abbreviation.
_PROTECTED = (("e.g.", "\x00EG\x00"), ("i.e.", "\x00IE\x00"), ("etc.", "\x00ETC\x00"))
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(])")


def split_sentences(text: str) -> list[str]:
    """Split generated prose into sentences, dropping empties.

    Not a general-purpose sentence splitter and not trying to be. What it must
    do is be *deterministic*, because the judge's verdict is per sentence and a
    split that varied between runs would change the groundedness of an answer
    without the answer changing.
    """
    working = text.strip()
    if not working:
        return []
    for literal, token in _PROTECTED:
        working = working.replace(literal, token)
    parts = _BOUNDARY.split(working)
    restored = []
    for part in parts:
        for literal, token in _PROTECTED:
            part = part.replace(token, literal)
        cleaned = " ".join(part.split())
        if cleaned:
            restored.append(cleaned)
    return restored


def format_context(chunks: list[tuple[str, str]]) -> str:
    """Number the passages so the judge and the human read the same thing."""
    return "\n\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(chunks, start=1))


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceVerdict:
    n: int
    sentence: str
    supported: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "sentence": self.sentence,
            "supported": self.supported,
            "reason": self.reason,
        }


@dataclass
class AnswerRecord:
    """One lane's answer to one question, and the judge's verdict on it."""

    qid: str
    lane_id: str
    question: str
    answer: str
    chunk_ids: list[str]
    verdicts: list[SentenceVerdict] = field(default_factory=list)
    # Kept per call rather than summed, because the two calls have different
    # budgets and only a per-call number can justify either of them. `--measure`
    # reads exactly these.
    generate_prompt_tokens: int = 0
    generate_completion_tokens: int = 0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.generate_prompt_tokens + self.judge_prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.generate_completion_tokens + self.judge_completion_tokens

    @property
    def grounded(self) -> bool:
        """Every sentence supported. An answer with no sentences is not grounded.

        The empty case is deliberate rather than vacuously true: a lane whose
        generator returned nothing has not produced a grounded answer, and
        `all([])` would quietly score it 1.0.
        """
        return bool(self.verdicts) and all(v.supported for v in self.verdicts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "lane_id": self.lane_id,
            "question": self.question,
            "answer": self.answer,
            "chunk_ids": self.chunk_ids,
            "grounded": self.grounded,
            "verdicts": [v.as_dict() for v in self.verdicts],
            "generate_prompt_tokens": self.generate_prompt_tokens,
            "generate_completion_tokens": self.generate_completion_tokens,
            "judge_prompt_tokens": self.judge_prompt_tokens,
            "judge_completion_tokens": self.judge_completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


# --------------------------------------------------------------------------
# Calling the model
# --------------------------------------------------------------------------


def get_client() -> Groq:
    from backend.goldset.generate import get_client as _client

    return _client()


def complete(
    client: Groq,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    """One chat completion, returning (content, prompt_tokens, completion_tokens).

    Retries once on any failure, matching lane 5: Groq's per-minute limit is
    transient and one blip should not cost a sample. It does not retry twice --
    past that the failure is the daily budget or the service.

    An empty `content` raises rather than returning "". Everything downstream
    of here turns text into a verdict, and "" would become a defensible-looking
    "unsupported".
    """
    from backend.goldset.generate import is_daily_quota_error

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                reasoning_effort=REASONING_EFFORT,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            choice = response.choices[0]
            content = (choice.message.content or "").strip()
            if not content:
                raise EmptyCompletionError(
                    f"empty content, finish_reason={choice.finish_reason!r}"
                    + (
                        f" -- max_tokens={max_tokens} was consumed by reasoning tokens "
                        f"before any content was emitted. Raise the budget or lower "
                        f"reasoning_effort; do NOT score this as ungrounded."
                        if choice.finish_reason == "length"
                        else ""
                    )
                )
            usage = response.usage
            return (
                content,
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
            )
        except Exception as exc:
            # Groq signals per-minute and per-day limits through the same 429.
            # The first clears on a retry; the second refills at limit/1440 a
            # minute, so retrying it burns the rest of the run against a wall.
            # `goldset.generate` learned this the expensive way on Day 1 and
            # owns the discrimination; this reuses it rather than re-deriving it.
            if is_daily_quota_error(exc):
                raise DailyBudgetExhausted(str(exc)) from exc
            last_error = exc
            if attempt == 0:
                time.sleep(2.0)
    raise JudgeError(f"{model} did not return a usable completion: {last_error}") from last_error


def generate_answer(
    client: Groq,
    model: str,
    question: str,
    chunks: list[tuple[str, str]],
) -> tuple[str, int, int]:
    """Write an answer to `question` from `chunks` alone."""
    content, prompt_tokens, completion_tokens = complete(
        client,
        model,
        GENERATE_SYSTEM,
        GENERATE_USER.format(context=format_context(chunks), question=question),
        GENERATE_MAX_TOKENS,
    )
    return content, prompt_tokens, completion_tokens


def parse_verdicts(content: str, sentences: list[str]) -> list[SentenceVerdict]:
    """Read the judge's JSON into one verdict per sentence, in order.

    Tolerant about the envelope and strict about the contents. Models wrap JSON
    in prose or a fence often enough that failing on it would throw away good
    verdicts, so the first `{` to the last `}` is extracted. But a response
    with the wrong number of verdicts raises: silently padding or truncating
    would misalign every verdict after the gap and still look like a result.
    """
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        raise JudgeError(f"no JSON object in judge response: {content[:200]!r}")
    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge response is not valid JSON: {content[:200]!r}") from exc

    rows = payload.get("verdicts")
    if not isinstance(rows, list):
        raise JudgeError(f"judge response has no 'verdicts' list: {content[:200]!r}")
    if len(rows) != len(sentences):
        raise JudgeError(
            f"judge returned {len(rows)} verdicts for {len(sentences)} sentences. "
            "Refusing to align them by guessing."
        )
    return [
        SentenceVerdict(
            n=index,
            sentence=sentence,
            supported=bool(row.get("supported")),
            reason=str(row.get("reason", "")).strip(),
        )
        for index, (sentence, row) in enumerate(zip(sentences, rows, strict=True), start=1)
    ]


def judge_answer(
    client: Groq,
    model: str,
    question: str,
    answer: str,
    chunks: list[tuple[str, str]],
) -> tuple[list[SentenceVerdict], int, int]:
    """Score every sentence of `answer` against `chunks`.

    One call for the whole answer rather than one per sentence. Per-sentence
    calls would isolate each judgement, which sounds stricter and is worse
    here: generated sentences are not independent, and a sentence like "It
    must be installed separately" is unjudgeable without its predecessor. It
    would be scored unsupported for a reason that is about the splitting, not
    the grounding. Costs 1 call per answer instead of 3-4, which also keeps the
    six-lane sweep inside a daily budget.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return [], 0, 0
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    content, prompt_tokens, completion_tokens = complete(
        client,
        model,
        JUDGE_SYSTEM,
        JUDGE_USER.format(
            context=format_context(chunks), question=question, sentences=numbered
        ),
        JUDGE_MAX_TOKENS,
    )
    return parse_verdicts(content, sentences), prompt_tokens, completion_tokens


def score_one(
    client: Groq,
    model: str,
    qid: str,
    lane_id: str,
    question: str,
    chunks: list[tuple[str, str]],
) -> AnswerRecord:
    """Generate then judge, returning the full record."""
    answer, gen_prompt, gen_completion = generate_answer(client, model, question, chunks)
    verdicts, judge_prompt, judge_completion = judge_answer(
        client, model, question, answer, chunks
    )
    return AnswerRecord(
        qid=qid,
        lane_id=lane_id,
        question=question,
        answer=answer,
        chunk_ids=[chunk_id for chunk_id, _ in chunks],
        verdicts=verdicts,
        generate_prompt_tokens=gen_prompt,
        generate_completion_tokens=gen_completion,
        judge_prompt_tokens=judge_prompt,
        judge_completion_tokens=judge_completion,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def load_split(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return sorted(rows, key=lambda r: r["qid"])


def read_records(path: Path) -> list[dict]:
    """Answers already judged, from any previous run."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def already_done(path: Path) -> set[tuple[str, str]]:
    """(lane_id, qid) pairs that do not need judging again.

    Resumability is not a nicety here. One answer costs about 8,500 prompt
    tokens across its two calls, and the free tier's ceiling is 200,000 tokens
    a day -- roughly 23 answers. A 35-question sweep does not fit in one day,
    so a run that could not be resumed could never finish at all.

    `generate` and `screen` append as they go for the same reason, and
    EVALUATION_SPEC §5 states the property: an interrupted run costs minutes,
    not the whole pass.
    """
    return {(row["lane_id"], row["qid"]) for row in read_records(path)}


def append_record(handle: Any, record: AnswerRecord) -> None:
    """Write one record and flush. Flushing matters: the process is expected to
    be killed by a quota, and a buffered record is a call paid for and lost."""
    handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
    handle.flush()


def summarise(
    records: list[AnswerRecord] | list[dict],
    failures: list[dict],
    *,
    questions_in_split: int | None = None,
) -> dict[str, Any]:
    """Groundedness per lane, plus what failed and why.

    Failures are reported separately and are NOT counted as ungrounded. A judge
    that could not answer has not found an answer unsupported, and folding the
    two together would make an outage look like a quality problem.
    """
    settings = get_settings()
    # Accepts dicts as well as records, so a resumed run can summarise the whole
    # verdicts file rather than only the answers this process happened to judge.
    rows: list[dict] = [r if isinstance(r, dict) else r.as_dict() for r in records]
    by_lane: dict[str, list[dict]] = {}
    for row in rows:
        by_lane.setdefault(row["lane_id"], []).append(row)

    from backend.metrics import groundedness_rate

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "judge_model": settings.groq_model,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": TEMPERATURE,
        "context_depth": CONTEXT_DEPTH,
        "max_tokens": {"generate": GENERATE_MAX_TOKENS, "judge": JUDGE_MAX_TOKENS},
        "max_completion_tokens_observed": {
            "generate": max((r["generate_completion_tokens"] for r in rows), default=0),
            "judge": max((r["judge_completion_tokens"] for r in rows), default=0),
        },
        "lanes": {
            lane_id: {
                "answers_scored": len(lane_rows),
                "grounded": sum(1 for r in lane_rows if r["grounded"]),
                "groundedness_rate": round(
                    groundedness_rate([r["grounded"] for r in lane_rows]), 4
                ),
                "sentences": sum(len(r["verdicts"]) for r in lane_rows),
                "sentences_unsupported": sum(
                    1 for r in lane_rows for v in r["verdicts"] if not v["supported"]
                ),
                "prompt_tokens": sum(r["prompt_tokens"] for r in lane_rows),
                "completion_tokens": sum(r["completion_tokens"] for r in lane_rows),
                "mean_prompt_tokens_per_answer": (
                    round(sum(r["prompt_tokens"] for r in lane_rows) / len(lane_rows))
                    if lane_rows
                    else 0
                ),
                # The load-bearing field. A groundedness rate over 5 of 35
                # questions is not this lane's groundedness rate, and the only
                # thing standing between that number and a published table is
                # something downstream being able to tell. `evaluate.py` reads
                # exactly this flag and omits the metric when it is false.
                "complete": (
                    questions_in_split is not None and len(lane_rows) >= questions_in_split
                ),
                "of_questions_in_split": questions_in_split,
            }
            for lane_id, lane_rows in sorted(by_lane.items())
        },
        "failures": failures,
        "questions_in_split": questions_in_split,
        "note": (
            "Judge failures are reported here and excluded from the rate. A judge "
            "that could not answer has not found an answer unsupported. A lane "
            "whose `complete` is false has been swept only partially -- its rate "
            "describes the answers judged so far and nothing else, and "
            "evaluate.py will not publish it."
        ),
    }


def report_measurements(records: list[AnswerRecord]) -> str:
    """What the two `max_tokens` budgets are actually set against.

    The comment at the top of this file states measured completion-token
    figures. This is the command that produces them, so the comment is a claim
    the repository can regenerate rather than a number someone remembers.

    Completion tokens INCLUDE the reasoning tokens, which is the whole point:
    the trap is a budget that the reasoning phase eats before any content is
    emitted, and only the total tells you how close you are to it.
    """
    lines = ["call        n   min   median   max   budget   headroom at max"]
    for name, budget, values in (
        (
            "generate",
            GENERATE_MAX_TOKENS,
            sorted(r.generate_completion_tokens for r in records),
        ),
        ("judge", JUDGE_MAX_TOKENS, sorted(r.judge_completion_tokens for r in records)),
    ):
        if not values:
            continue
        median = values[len(values) // 2]
        lines.append(
            f"{name:<10} {len(values):>3} {values[0]:>5} {median:>8} {values[-1]:>5} "
            f"{budget:>8} {budget / max(values[-1], 1):>15.1f}x"
        )
    lines.append("")
    lines.append(
        "Completion tokens include reasoning tokens. The failure being guarded "
        "is an empty completion, not a truncated one."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.judge",
        description="Generate answers from each lane's chunks and score their groundedness.",
    )
    parser.add_argument("--split", choices=sorted(DEFAULT_SPLITS), default="test")
    parser.add_argument("--lanes", default="hybrid_rerank", help=f"one of {sorted(REGISTRY)}")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS)
    parser.add_argument("--corpus-id", default="demo")
    parser.add_argument(
        "--measure",
        action="store_true",
        help="also report the completion-token distribution the max_tokens budgets are set against",
    )
    args = parser.parse_args(argv)

    rows = load_split(DEFAULT_SPLITS[args.split])
    if args.limit:
        rows = rows[: args.limit]
    settings = get_settings()
    client = get_client()

    done = already_done(args.verdicts)
    if done:
        print(f"resuming   {len(done)} answer(s) already judged in {args.verdicts}",
              file=sys.stderr)

    corpus = Corpus(args.corpus_id)
    scored_here: list[AnswerRecord] = []
    failures: list[dict] = []
    exhausted: str | None = None
    args.verdicts.parent.mkdir(parents=True, exist_ok=True)
    try:
        lanes = build_lanes(corpus, args.lanes.split(","))
        with args.verdicts.open("a", encoding="utf-8") as handle:
            for lane in lanes:
                if exhausted:
                    break
                lane.warm()
                todo = [r for r in rows if (lane.id, r["qid"]) not in done]
                print(
                    f"\n  {lane.id}  ({len(todo)} to judge, {len(rows) - len(todo)} done)",
                    file=sys.stderr,
                )
                for index, row in enumerate(todo, start=1):
                    result = lane.retrieve(row["question"], k=CONTEXT_DEPTH)
                    chunks = [(c.chunk_id, c.text) for c in result.chunks]
                    try:
                        record = score_one(
                            client,
                            settings.groq_model,
                            row["qid"],
                            lane.id,
                            row["question"],
                            chunks,
                        )
                    except DailyBudgetExhausted as exc:
                        exhausted = str(exc)
                        break
                    except JudgeError as exc:
                        failures.append(
                            {"qid": row["qid"], "lane_id": lane.id, "error": str(exc)}
                        )
                        print(f"    {row['qid']} JUDGE FAILED: {exc}", file=sys.stderr)
                        continue
                    append_record(handle, record)
                    scored_here.append(record)
                    if index % 5 == 0:
                        print(f"    {index}/{len(todo)}", file=sys.stderr)
    finally:
        corpus.close()

    records = scored_here
    summary = summarise(
        read_records(args.verdicts), failures, questions_in_split=len(rows)
    )
    summary["budget"] = {
        "judged_this_run": len(scored_here),
        "judged_total": len(read_records(args.verdicts)),
        "daily_budget_exhausted": exhausted is not None,
        "detail": exhausted,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    if args.measure and records:
        print(report_measurements(records))
        print()
    for lane_id, stats in summary["lanes"].items():
        flag = "" if stats["complete"] else "   PARTIAL -- not publishable"
        print(
            f"{lane_id:24} groundedness {stats['groundedness_rate']:.4f}  "
            f"({stats['grounded']}/{stats['answers_scored']})  "
            f"{stats['sentences_unsupported']}/{stats['sentences']} sentences "
            f"unsupported{flag}"
        )
    if failures:
        print(f"\n{len(failures)} judge failure(s), excluded from the rate, listed in {args.out}")
    print(f"\nverdicts   {args.verdicts}  ({summary['budget']['judged_total']} total)")
    print(f"summary    {args.out}")
    print(
        "\nThis rate is an LLM's opinion until it is calibrated. Run "
        "`python -m backend.judge_calibrate --build` next."
    )
    if exhausted:
        print(
            "\nDAILY TOKEN BUDGET EXHAUSTED. Judged "
            f"{summary['budget']['judged_this_run']} answer(s) this run, "
            f"{summary['budget']['judged_total']} in total. Nothing is lost: every "
            "answer was appended as it completed, and re-running this command "
            "resumes from where it stopped.\n"
            f"  {exhausted}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
