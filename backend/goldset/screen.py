"""LLM screening: every draft scored against the four drop rules, one at a time.

This replaces reading all 350 drafts by hand. It does not replace human
judgement -- `audit.py` samples the screener's output and measures how often a
human agrees, and that agreement number is published alongside every metric.
An unaudited screener is exactly the circular setup docs/02_EVALUATION_SPEC.md
§1 warns about.

**Each rule is scored by its own call.** One call asking "is this question any
good?" collapses four distinct failure modes into a single yes/no, which is
precisely what made the drafting prompt so hard to debug: when a draft came
back bad there was no way to tell which constraint the model had dropped. Four
independent scores cost four times as much and tell you which rule fired, and
the per-rule counts in the run summary are the thing that says whether the
*drafting* prompt needs work or the screener does.

**Rule 2 is not sent to the model.** "Reuses more than four consecutive words"
is an exact, checkable property of two strings, and a longest-common-run scan
answers it in a dozen lines, identically on every run. An LLM asked to count
consecutive words is guessing, and its guess would be the one number in this
pipeline that could not be reproduced. The other three rules genuinely need
judgement, so they go to the model.

Writes `data/goldset_screened.jsonl`: one row per draft, carrying the verdict,
the rule that forced it, a one-line reason, and every rule's individual score.
Rows are appended as they land and already-screened qids are skipped, so an
interrupted run resumes rather than restarts.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from backend.config import get_settings
from backend.goldset import rules as drop_rules
from backend.goldset.console import append_jsonl, load_jsonl
from backend.goldset.generate import DailyQuotaExhausted, get_client, is_daily_quota_error
from backend.goldset.rules import GENERIC, MULTI_CHUNK, UNANSWERABLE, VERBATIM
from backend.retrieval import dense_store

if TYPE_CHECKING:
    from groq import Groq

DEFAULT_DRAFTS = Path("data/goldset_draft.jsonl")
DEFAULT_OUT = Path("data/goldset_screened.jsonl")

# Bumped whenever any prompt below changes, and recorded on every row. Two
# screening runs made by different prompts are not comparable, and after the
# fact the row is the only place that record can live.
SCREEN_PROMPT_VERSION = "v1"

# Zero, not the drafting temperature of 0.3: this is a classification, and it
# has to give the same verdict for the same inputs or the gold set is not
# reproducible.
TEMPERATURE = 0.0

# docs/02_EVALUATION_SPEC.md §1: "more than four consecutive words". Four
# because ordinary English hands you three shared words by accident ("the
# number of", "in order to") while five is nearly always a lifted phrase. The
# threshold was checked against a 30-draft sample of the v3 drafting prompt:
# run lengths were 0-4 for 29 of them and 8 for the one genuine offender, so
# the cut sits inside a clear gap rather than in the middle of a distribution.
# Measured over the whole passage, not just the sentence holding the answer --
# a question that string-matches anywhere in the chunk hands BM25 the same
# free win.
VERBATIM_RUN = 4

# Two tries at a rephrase, then the pair is dropped under the verbatim rule.
# The second try is told the exact phrase the first one failed to shake, which
# is what makes it a different attempt rather than a re-roll of the same one.
# Two rather than one because a single failure is usually the model reaching
# for the passage's noun phrase again; two rather than five because each
# attempt costs a call and the rows are cheap to replace -- there are more
# drafts than the gold set needs.
REPHRASE_ATTEMPTS = 2


# --------------------------------------------------------------------------
# Rule 2, mechanically
# --------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    """Lowercase word tokens, punctuation discarded.

    Case and punctuation are stripped so that "Set the X-Accel-Buffering
    header" and "set the x accel buffering header" count as the same run. A
    rephrase that only changes capitalisation has not rephrased anything.
    """
    return _WORD.findall(text.lower())


@dataclass(frozen=True)
class Overlap:
    """The longest run of consecutive words the question shares with the chunk."""

    length: int
    phrase: str


def longest_overlap(question: str, passage: str) -> Overlap:
    """Find the longest word run common to both strings.

    A textbook longest-common-substring scan over word lists rather than
    characters, so "config" and "configuration" are different words. Two rows
    of the table are enough because a run ending at (i, j) only ever extends a
    run ending at (i-1, j-1).
    """
    q, p = _words(question), _words(passage)
    if not q or not p:
        return Overlap(0, "")

    best_length = 0
    best_end = 0
    previous = [0] * (len(p) + 1)
    for qi in range(1, len(q) + 1):
        current = [0] * (len(p) + 1)
        for pi in range(1, len(p) + 1):
            if q[qi - 1] == p[pi - 1]:
                current[pi] = previous[pi - 1] + 1
                if current[pi] > best_length:
                    best_length = current[pi]
                    best_end = qi
        previous = current

    return Overlap(best_length, " ".join(q[best_end - best_length : best_end]))


# --------------------------------------------------------------------------
# Rules 1, 3 and 4, by model
# --------------------------------------------------------------------------

SCREEN_SYSTEM = (
    "You audit draft questions for a document-retrieval benchmark. You are "
    "asked one narrow yes/no question about one draft at a time. Reply with "
    "YES or NO on the first line and one short sentence of justification on "
    "the second. Nothing else."
)

# Every prompt is phrased so that YES means "this rule fires, the draft has
# this problem". Keeping the polarity uniform across all three matters more
# than any individual wording: a prompt where YES meant "fine" would silently
# invert one rule's counts and nothing downstream would notice.
GENERIC_PROMPT = """\
A benchmark question is worthless if many passages answer it equally well:
every retrieval system finds something, and the measurement says nothing.

Think about a large corpus of software documentation - web frameworks, agent
libraries, and similar. Ignore how the passage below happens to be worded.

Would three or more OTHER passages, from anywhere in such a corpus, answer this
question just as well as this one does?

YES if the question is generic: what something is for, why it matters, what a
widely-used tool does, or anything a dozen tutorials would answer identically.
NO if the answer is one concrete thing stated in few places - an identifier, a
header, a package, a parameter, a default value, a version, a setting.

Question: {question}

Passage:
\"\"\"
{chunk_text}
\"\"\"
"""

UNANSWERABLE_PROMPT = """\
A benchmark question must be answerable from the single passage it is labelled
with. If it is not, the label is wrong and the pair measures nothing.

Read the passage, then the question.

Is the answer ABSENT from the passage?

YES if the passage does not state the answer - it is missing, merely hinted at,
or lives somewhere the reader cannot see: inside a code example, behind an
include directive such as {{* ../../docs_src/example.py *}}, or in a file the
passage only names.
NO if the passage states the answer plainly enough that a reader holding this
passage alone could answer correctly.

Question: {question}

Passage:
\"\"\"
{chunk_text}
\"\"\"
"""

MULTI_CHUNK_PROMPT = """\
This benchmark measures whether a retriever finds the ONE passage that answers
a question. A question needing two passages is a valid retrieval problem and
the wrong problem for this benchmark.

Read the passage, then the question.

Is a second passage needed as well as this one?

YES if answering requires something this passage does not supply: it compares
this passage's subject with another, asks about a procedure only partly shown
here, or leans on a definition given elsewhere.
NO if this passage alone is enough.

Question: {question}

Passage:
\"\"\"
{chunk_text}
\"\"\"
"""

PROMPTS: dict[str, str] = {
    GENERIC.id: GENERIC_PROMPT,
    UNANSWERABLE.id: UNANSWERABLE_PROMPT,
    MULTI_CHUNK.id: MULTI_CHUNK_PROMPT,
}

# The rules that cost an API call, in the order they are asked. VERBATIM is
# absent by design -- see the module docstring.
MODEL_SCORED: tuple[str, ...] = (GENERIC.id, UNANSWERABLE.id, MULTI_CHUNK.id)


@dataclass(frozen=True)
class Judgement:
    """One rule's score. `fired` is None when the reply could not be read."""

    fired: bool | None
    reason: str


_VERDICT = re.compile(r"^(YES|NO)\b[\s:.,\-]*", re.IGNORECASE)
_MAX_REASON_CHARS = 200


def parse_judgement(reply: str) -> Judgement:
    """Read a YES/NO verdict and its one-line reason out of the model's reply.

    An unreadable reply becomes `fired=None` rather than a default. Defaulting
    it either way would quietly turn a failed call into a judgement, which is
    the one thing this file must never do.
    """
    lines = [line.strip() for line in (reply or "").splitlines() if line.strip()]
    if not lines:
        return Judgement(None, "empty reply")

    match = _VERDICT.match(lines[0])
    if match is None:
        return Judgement(None, f"unparseable reply: {' '.join(lines)[:_MAX_REASON_CHARS]}")

    # The model is asked for the reason on line two, but often puts it on line
    # one after the verdict. Accept both rather than lose the reason.
    tail = lines[0][match.end() :].strip()
    reason = tail or " ".join(lines[1:]).strip()
    return Judgement(match.group(1).upper() == "YES", reason[:_MAX_REASON_CHARS])


def judge_rule(
    client: Groq,
    model: str,
    rule_id: str,
    question: str,
    chunk_text: str,
) -> Judgement:
    """Score one rule with one call."""
    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SCREEN_SYSTEM},
            {
                "role": "user",
                "content": PROMPTS[rule_id].format(question=question, chunk_text=chunk_text),
            },
        ],
    )
    return parse_judgement(response.choices[0].message.content or "")


# --------------------------------------------------------------------------
# The fix path
# --------------------------------------------------------------------------

REPHRASE_SYSTEM = (
    "You rewrite evaluation questions. You reply with the rewritten question "
    "only: no preamble, no numbering, no quotation marks."
)

REPHRASE_PROMPT = """\
This question has the right answer and the right passage, but it lifts the
passage's own wording. A retrieval system can then find the answer by string
matching instead of understanding it, so the question measures nothing.

Rewrite it so that:

1. The answer is exactly the same concrete thing.
2. It stays just as specific - do not make it broader or vaguer.
3. No run of more than four consecutive words is shared with the passage.
4. It uses different vocabulary from the sentence containing the answer.
5. It ends in a question mark.

The phrase you must not reuse is: "{overlap}"

Question: {question}

Passage:
\"\"\"
{chunk_text}
\"\"\"

Rewritten question:"""


def rephrase_question(
    client: Groq,
    model: str,
    question: str,
    chunk_text: str,
    overlap: Overlap,
) -> str:
    """One rewrite attempt. Returns the question, stripped of stray quoting."""
    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": REPHRASE_SYSTEM},
            {
                "role": "user",
                "content": REPHRASE_PROMPT.format(
                    question=question, chunk_text=chunk_text, overlap=overlap.phrase
                ),
            },
        ],
    )
    rewritten = (response.choices[0].message.content or "").strip()
    if len(rewritten) >= 2 and rewritten[0] == rewritten[-1] and rewritten[0] in "\"'":
        rewritten = rewritten[1:-1].strip()
    return rewritten


# --------------------------------------------------------------------------
# Screening one draft
# --------------------------------------------------------------------------


def _retry(call: Callable[[], object], attempts: int = 3, delay: float = 2.0):  # noqa: ANN202
    """Run `call`, retrying on any failure. Returns (result, error_message).

    Backoff is linear rather than fixed because the per-minute limit is tokens,
    not calls: when a 429 arrives the only useful lever is to wait longer, and
    waiting a little longer each time clears a full bucket without stalling the
    run for a transient network error.

    The daily budget is the exception and raises instead. Retrying that one is
    not slow, it is futile -- the bucket refills at limit/1440 tokens a minute,
    so every call for the next several hours fails identically.
    """
    error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call(), None
        except Exception as exc:  # any API failure is the same to us here
            if is_daily_quota_error(exc):
                raise DailyQuotaExhausted(str(exc)) from exc
            error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(delay * attempt)
    return None, error


@dataclass
class Screening:
    """Everything one draft's screening produced."""

    verdict: str  # "keep" | "fix" | "drop" | "unscored"
    rule: str | None
    reason: str
    question: str
    scores: dict[str, dict[str, object]]
    overlap: Overlap
    rephrase_attempts: int = 0


def screen_draft(
    client: Groq,
    model: str,
    question: str,
    chunk_text: str,
    rephrase_attempts: int = REPHRASE_ATTEMPTS,
) -> Screening:
    """Score all four rules, then resolve them into one verdict.

    Every rule is scored even once a drop is certain. Short-circuiting would be
    cheaper and would hollow out the per-rule counts, which are the only
    evidence for whether the drafting prompt or the screener needs the work.
    """
    scores: dict[str, dict[str, object]] = {}
    fired: dict[str, bool] = {}

    overlap = longest_overlap(question, chunk_text)
    verbatim_fired = overlap.length > VERBATIM_RUN
    fired[VERBATIM.id] = verbatim_fired
    scores[VERBATIM.id] = {
        "fired": verbatim_fired,
        "reason": (
            f'shares a {overlap.length}-word run with the chunk: "{overlap.phrase}"'
            if verbatim_fired
            else f"longest shared run is {overlap.length} words"
        ),
        "scored_by": "exact",
    }

    unscored: list[str] = []
    for rule_id in MODEL_SCORED:
        judgement, error = _retry(
            lambda r=rule_id: judge_rule(client, model, r, question, chunk_text)
        )
        if judgement is None:
            judgement = Judgement(None, f"call failed: {error}")
        if judgement.fired is None:
            unscored.append(rule_id)
        else:
            fired[rule_id] = judgement.fired
        scores[rule_id] = {
            "fired": judgement.fired,
            "reason": judgement.reason,
            "scored_by": "model",
        }

    if unscored:
        return Screening(
            verdict="unscored",
            rule=None,
            reason=f"could not score: {', '.join(unscored)}",
            question=question,
            scores=scores,
            overlap=overlap,
        )

    verdict, rule = drop_rules.verdict_for(fired)
    if verdict != "fix":
        reason = str(scores[rule]["reason"]) if rule else "no rule fired"
        return Screening(verdict, rule, reason, question, scores, overlap)

    return _rephrase(client, model, question, chunk_text, scores, overlap, rephrase_attempts)


def _rephrase(
    client: Groq,
    model: str,
    question: str,
    chunk_text: str,
    scores: dict[str, dict[str, object]],
    overlap: Overlap,
    attempts: int,
) -> Screening:
    """Rewrite until the overlap is gone, or drop the pair for trying.

    The rewrite is not re-scored against rules 1, 3 and 4. It is instructed to
    keep the same answer and the same specificity, and the human audit samples
    fixes precisely so that this assumption is measured rather than trusted.
    """
    current, current_overlap = question, overlap
    for attempt in range(1, attempts + 1):
        rewritten, error = _retry(
            lambda q=current, o=current_overlap: rephrase_question(
                client, model, q, chunk_text, o
            )
        )
        if not rewritten:
            return Screening(
                verdict="unscored",
                rule=VERBATIM.id,
                reason=f"rephrase call failed: {error}",
                question=question,
                scores=scores,
                overlap=overlap,
                rephrase_attempts=attempt,
            )

        current = str(rewritten)
        current_overlap = longest_overlap(current, chunk_text)
        if current_overlap.length <= VERBATIM_RUN:
            return Screening(
                verdict="fix",
                rule=VERBATIM.id,
                reason=(
                    f'rephrased on attempt {attempt}; {overlap.length}-word overlap '
                    f'"{overlap.phrase}" reduced to {current_overlap.length} words'
                ),
                question=current,
                scores=scores,
                overlap=overlap,
                rephrase_attempts=attempt,
            )

    return Screening(
        verdict="drop",
        rule=VERBATIM.id,
        reason=(
            f"{attempts} rephrasings left a {current_overlap.length}-word overlap "
            f'("{current_overlap.phrase}")'
        ),
        question=question,
        scores=scores,
        overlap=overlap,
        rephrase_attempts=attempts,
    )


# --------------------------------------------------------------------------
# Rows and the run
# --------------------------------------------------------------------------


def screened_row(draft: dict, result: Screening, model: str) -> dict:
    """The draft, plus what screening decided about it.

    The draft's own fields are carried through unchanged so that this file is
    a superset of the drafts and nothing downstream has to join two files.
    `question` holds the final wording; `original_question` appears only when
    a rephrasing actually changed it.
    """
    row = dict(draft)
    row["question"] = result.question
    if result.question != draft.get("question"):
        row["original_question"] = draft.get("question")
    row.update(
        {
            "screen_verdict": result.verdict,
            "screen_rule": result.rule,
            "screen_reason": result.reason,
            "screen_scores": result.scores,
            "shared_run": result.overlap.length,
            "rephrase_attempts": result.rephrase_attempts,
            "screen_model": model,
            "screen_prompt_version": SCREEN_PROMPT_VERSION,
            "screened_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    return row


def run(
    drafts: list[dict],
    chunks: dict[str, str],
    out_path: Path,
    client: Groq,
    model: str,
    rephrase_attempts: int = REPHRASE_ATTEMPTS,
) -> Counter[str]:
    """Screen every draft not already in `out_path`, appending as each lands.

    Resumable by qid: this run costs four API calls per draft over several
    hundred drafts, and losing an hour of it to a dropped connection is not an
    acceptable failure mode.
    """
    done = load_jsonl(out_path)
    already = {row["qid"] for row in done}
    tally: Counter[str] = Counter(row["screen_verdict"] for row in done)

    pending = [d for d in drafts if d["qid"] not in already]
    if already:
        print(f"resuming: {len(already)} already screened, {len(pending)} to go\n")

    for index, draft in enumerate(pending, start=1):
        chunk_id = draft["gold_chunk_ids"][0]
        chunk_text = chunks.get(chunk_id)
        if chunk_text is None:
            # A draft whose labelled chunk is no longer in the store cannot be
            # screened and must not become a keep by default.
            result = Screening(
                verdict="unscored",
                rule=None,
                reason=f"chunk {chunk_id} not found in the store",
                question=draft["question"],
                scores={},
                overlap=Overlap(0, ""),
            )
        else:
            try:
                result = screen_draft(
                    client, model, draft["question"], chunk_text, rephrase_attempts
                )
            except DailyQuotaExhausted as exc:
                # Stop rather than write "unscored" across every remaining
                # draft. Those rows would be indistinguishable from genuine
                # screening failures, and the file is resumed by qid.
                print(f"\n  DAILY TOKEN BUDGET EXHAUSTED at {draft['qid']}.\n  {exc}")
                print(f"  {index - 1} of {len(pending)} screened this pass. Re-run to continue.\n")
                tally["quota_exhausted"] = 1
                return tally

        append_jsonl(out_path, screened_row(draft, result, model))
        tally[result.verdict] += 1
        rule = f" [{result.rule}]" if result.rule else ""
        print(
            f"  [{index:>3}/{len(pending)}] {draft['qid']}  "
            f"{result.verdict.upper():<9}{rule} {result.reason[:70]}"
        )

    return tally


def rule_counts(rows: list[dict]) -> Counter[str]:
    """How many times each rule fired across the screened rows."""
    counts: Counter[str] = Counter()
    for row in rows:
        for rule_id, score in (row.get("screen_scores") or {}).items():
            if score.get("fired"):
                counts[rule_id] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.goldset.screen",
        description="Score every draft against the four drop rules, one call per rule.",
    )
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpus-id", default="demo")
    parser.add_argument(
        "--limit", type=int, default=None, help="screen only the first N drafts"
    )
    parser.add_argument("--rephrase-attempts", type=int, default=REPHRASE_ATTEMPTS)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "override GOLDSET_MODEL for screening. Recorded on every row: two "
            "runs screened by different models are not comparable evidence."
        ),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    model = args.model or settings.goldset_model
    # Fail before opening a database connection or writing a file.
    client = get_client()

    rows = load_jsonl(args.drafts)
    if not rows:
        print(f"No drafts in {args.drafts}. Run backend.goldset.generate first.")
        return 1
    drafts = sorted(
        (r for r in rows if r.get("status") == "draft"), key=lambda r: r["qid"]
    )
    if not drafts:
        print(f"{args.drafts} has no rows with status 'draft'.")
        return 1
    if args.limit is not None:
        drafts = drafts[: args.limit]

    with dense_store.connect() as conn:
        chunks = {c.chunk_id: c.text for c in dense_store.fetch_chunks(conn, args.corpus_id)}

    print(f"drafts       {len(drafts)} from {args.drafts}")
    print(f"model        {model} @ temperature {TEMPERATURE}")
    print(f"prompt       {SCREEN_PROMPT_VERSION}")
    print(f"rules        {len(MODEL_SCORED)} by model, verbatim by exact match")
    print(f"writing      {args.out}\n")

    tally = run(drafts, chunks, args.out, client, model, args.rephrase_attempts)

    # Counted from the file rather than the tally: the file is the artefact,
    # and it stays correct across a resumed run and a quota-shortened one.
    screened = load_jsonl(args.out)
    verdicts: Counter[str] = Counter(r["screen_verdict"] for r in screened)
    total = len(screened)
    print(f"\nscreened     {total} of {len(drafts)}")
    for verdict in (*drop_rules.VERDICTS, "unscored"):
        count = verdicts[verdict]
        share = f"{count / total:.0%}" if total else "0%"
        print(f"  {verdict:<10} {count:>4}  {share}")

    print("\nrule fired")
    counts = rule_counts(screened)
    for rule in drop_rules.RULES:
        print(f"  {rule.id:<14} {counts[rule.id]:>4}  -> {rule.verdict}")

    usable = verdicts["keep"] + verdicts["fix"]
    print(f"\nkeep + fix   {usable}")
    if tally["quota_exhausted"]:
        print(
            f"\nSTOPPED EARLY: the daily token budget ran out with "
            f"{len(drafts) - total} drafts unscreened. Re-run to continue."
        )
        return 1
    print("\nThese are screened, not verified. Audit a sample with backend.goldset.audit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
