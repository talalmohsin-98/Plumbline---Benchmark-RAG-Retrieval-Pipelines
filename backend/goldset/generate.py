"""Gold-set drafting: LLM-generated question/chunk pairs awaiting screening.

Nothing written here is a label. Every row lands with `status: "draft"` and
becomes a gold label only after `screen.py` has scored it against the four
drop rules — see `docs/02_EVALUATION_SPEC.md` §1. This module therefore does
no quality filtering of its own: judging a draft is the screener's job, and a
filter here would quietly bias what the screener and the audit ever see.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from backend.config import get_settings, require_groq_api_key
from backend.retrieval import dense_store

if TYPE_CHECKING:
    from groq import Groq

    from backend.retrieval.dense_store import StoredChunk

DEFAULT_OUT = Path("data/goldset_draft.jsonl")

# Bumped whenever the wording below changes, and recorded on every row. Two
# drafts written by different prompts are not comparable evidence, and after
# the fact the row is the only place that record can live.
PROMPT_VERSION = "v3"

# docs/02_EVALUATION_SPEC.md §1 fixes this at 0.3: enough variation to avoid
# 150 identically-shaped questions, not so much that the model drifts off the
# passage. The judge runs at 0 — different job, different setting.
TEMPERATURE = 0.3

# Below this a chunk is a heading, a stub, or a fragment of a table; there is
# nothing specific enough to ask about.
MIN_CHUNK_CHARS = 200

# A chunk more than half code, serialised output, or include directives has
# too little prose to carry a question. See `non_prose_fraction`.
MAX_NON_PROSE_FRACTION = 0.5

# A chunk more than half YAML frontmatter and MDX import statements is page
# scaffolding, not content. Added after the Day 1 audit: of the 14 rows a human
# rejected, 7 asked what a page's `title:` or `sidebarTitle:` was and 2 asked
# for the name of an imported MDX snippet. Every chunk in the corpus has
# frontmatter, so those questions are answerable by any chunk and measure
# nothing -- and the fix belongs here, before an API call is spent drafting
# from a chunk that cannot carry a question. See `scaffolding_fraction`.
MAX_SCAFFOLDING_FRACTION = 0.5

SYSTEM_PROMPT = (
    "You write evaluation questions for a document-retrieval benchmark. "
    "You reply with the question only: no preamble, no numbering, no quotation marks."
)

# The passage is delimited so the model can tell instructions from content.
# Every rule and every example below exists because an earlier version produced
# a question that had to be dropped by hand. The worked examples carry more
# weight than the rules did: v2 followed the rules literally and satisfied rule
# 4 by lifting the sentence that contained the answer, which is exactly the
# verbatim-quoting failure rule 5 now forbids. Do not prune either without
# re-running the validation sample.
USER_PROMPT = """\
Write one specific question that this passage - and only this passage - answers.

RULES

1. Do not use the words "this passage" or "the document".

2. The passage is an extract. It may begin or end part-way through a code block,
   a JSON fragment, or serialised output. Ignore any such fragment entirely.
   Never ask about a value, key, field, or number that appears inside one.

3. The passage may reference code with an include directive such as
   {{* ../../docs_src/example.py *}}, or mention a code block that is not shown.
   That code is NOT available to whoever answers your question. Ask only about
   what the prose itself states. Never ask what a code example does, what a
   function in it is called, or what any referenced file contains.

4. The ANSWER must be one concrete thing the prose states: an identifier, a
   header, a package, a parameter, a method, a version number, a setting, or a
   value. Ask for that answer. Do not write "What is the purpose of ...", "What
   does ... do according to the documentation", or "Why is ... important" -
   those are answered equally well by dozens of other passages, which makes
   them worthless for measuring retrieval.

5. Do not reuse more than four consecutive words from the passage. Rephrase,
   using different vocabulary from the sentence that contains the answer. If a
   reader could find the answer by string-matching your question against the
   text, the question measures nothing.

6. If the passage shows the same idea in more than one programming language (for
   example :::python and :::js tabs), ask about the shared concept. Do not ask
   about one language's variant: someone searching with your question cannot see
   which tab you meant.

7. Reply with a single question ending in a question mark. No preamble, no
   numbering, no quotation marks.

EXAMPLES

Passage: "Set a special header X-Accel-Buffering: no to prevent buffering in
some proxies like Nginx"
BAD:  "What special header is set to prevent buffering in some proxies like
       Nginx?"  (breaks rule 5 - reuses the passage's wording)
GOOD: "Which response header stops reverse proxies from holding back a streamed
       response?"  (same answer, different words)

Passage: "You can use Pydantic models to declare form fields in FastAPI. To use
forms, first install python-multipart."
BAD:  "What is the name of the framework being discussed?"  (breaks rule 4 -
       far too general, hundreds of other passages answer it just as well)
GOOD: "Which extra package must be added to a project before form fields can be
       declared?"  (one concrete answer, stated only here)

Passage:
\"\"\"
{chunk_text}
\"\"\"

Question:"""


# --------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*(```|~~~)")
_INCLUDE_DIRECTIVE = re.compile(r"\{\*.*?\*\}")


def _is_symbol_dense(line: str) -> bool:
    """True for a line that reads as serialised output rather than a sentence.

    Chunk boundaries cut through code blocks, so a chunk can open in the middle
    of one with no fence in sight. Those orphaned fragments are dense in
    quotes, braces, and equals signs in a way prose is not.
    """
    if len(line) < 8:
        return False
    prose_chars = sum(1 for c in line if c.isalpha() or c.isspace())
    return prose_chars / len(line) < 0.75


def _opens_inside_fence(lines: list[str]) -> bool:
    """True if the first fence marker is a closing one.

    A bare ``` closes a block; a tagged ```python opens one. So a chunk whose
    first marker is bare began inside a code block that started in the previous
    chunk.
    """
    for line in lines:
        if _FENCE.match(line):
            return line.strip() in {"```", "~~~"}
    return False


def non_prose_fraction(text: str) -> float:
    """Fraction of characters that are code, serialised output, or directives."""
    if not text:
        return 1.0

    lines = text.splitlines(keepends=True)
    in_fence = _opens_inside_fence(lines)

    non_prose = 0
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
            non_prose += len(line)
        elif in_fence or _INCLUDE_DIRECTIVE.search(line) or _is_symbol_dense(line.strip()):
            non_prose += len(line)
    return non_prose / len(text)


_IMPORT = re.compile(r"^\s*import\s+\w+\s+from\s+['\"]")
_FRONTMATTER_KEY = re.compile(r"^[a-zA-Z][\w-]*:\s")


def scaffolding_fraction(text: str) -> float:
    """Fraction of characters that are frontmatter or MDX import statements.

    Frontmatter is the leading `---`-delimited block only; a `---` later in the
    page is a horizontal rule, not a second front matter. Imports are counted
    anywhere, because a chunk that opens part-way through a doc's import list
    has no leading delimiter to find.
    """
    if not text:
        return 1.0

    lines = text.splitlines(keepends=True)
    scaffolding = 0
    in_frontmatter = False
    # True until the first line that is not frontmatter-shaped. A chunk can
    # begin part-way through the frontmatter block, i.e. at `title: ...` with
    # no opening delimiter above it, so the leading run has to be recognised
    # by shape rather than by a delimiter that was cut off.
    leading = True
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index == 0 and stripped == "---":
            in_frontmatter = True
            scaffolding += len(line)
        elif in_frontmatter:
            scaffolding += len(line)
            if stripped == "---":
                in_frontmatter = False
                leading = False
        elif _IMPORT.match(line) or (leading and _FRONTMATTER_KEY.match(stripped)):
            scaffolding += len(line)
        elif stripped:
            leading = False
    return scaffolding / len(text)


def is_scaffolding(text: str) -> bool:
    """True if the chunk is mostly page scaffolding rather than prose."""
    return scaffolding_fraction(text) > MAX_SCAFFOLDING_FRACTION


def opens_inside_code(text: str) -> bool:
    """True if the chunk starts part-way through a code block or output dump."""
    lines = text.splitlines()
    if _opens_inside_fence(lines):
        return True
    first = next((line for line in lines if line.strip()), "")
    return _is_symbol_dense(first.strip())


def closes_inside_code(text: str) -> bool:
    """True if the chunk ends part-way through a code block or output dump."""
    lines = text.splitlines()
    in_fence = _opens_inside_fence(lines)
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
    if in_fence:
        return True
    last = next((line for line in reversed(lines) if line.strip()), "")
    return _is_symbol_dense(last.strip())


def has_code_edges(text: str) -> bool:
    """True if the chunk begins or ends part-way through code.

    Structural only. An earlier version also asked whether the first or last
    200 characters were predominantly non-prose, which rejected 378 chunks and
    was demonstrably over-broad: it removed passages that produced good
    questions while catching nothing the structural test misses. The v1 failure
    this rule exists for (`q002`, a question about values in an orphaned JSON
    fragment) is caught here because that chunk's first fence marker is a bare
    closing ```.
    """
    return opens_inside_code(text) or closes_inside_code(text)


@dataclass(frozen=True)
class Eligibility:
    """Why the eligible pool is the size it is, for the run summary."""

    eligible: list[StoredChunk]
    too_short: int
    too_much_code: int
    code_edges: int
    scaffolding: int = 0

    @property
    def considered(self) -> int:
        return (
            len(self.eligible)
            + self.too_short
            + self.too_much_code
            + self.code_edges
            + self.scaffolding
        )


def select_eligible(chunks: list[StoredChunk]) -> Eligibility:
    """Drop chunks that cannot support a question, and say how many went."""
    eligible: list[StoredChunk] = []
    too_short = too_much_code = code_edges = scaffolding = 0
    for chunk in chunks:
        if len(chunk.text) < MIN_CHUNK_CHARS:
            too_short += 1
        elif non_prose_fraction(chunk.text) > MAX_NON_PROSE_FRACTION:
            too_much_code += 1
        elif is_scaffolding(chunk.text):
            scaffolding += 1
        elif has_code_edges(chunk.text):
            code_edges += 1
        else:
            eligible.append(chunk)
    return Eligibility(
        eligible=eligible,
        too_short=too_short,
        too_much_code=too_much_code,
        code_edges=code_edges,
        scaffolding=scaffolding,
    )


_HEADING = re.compile(r"^#{1,6}\s")


def starts_at_heading(chunk: StoredChunk) -> bool:
    """True if the chunk opens a section rather than continuing one.

    Chunk 0 counts: the start of a document is self-contained by definition.
    This is the strongest available signal for a self-contained passage. Chunks
    that begin mid-section are what produced mislabelled gold, because the
    model writes a question from context that resolves in the following chunk.
    """
    if chunk.chunk_index == 0:
        return True
    first = next((line for line in chunk.text.splitlines() if line.strip()), "")
    return bool(_HEADING.match(first))


def labelled_chunk_ids(path: Path) -> set[str]:
    """Every chunk id already carrying a gold label, primary or multi-label.

    Read from the assembled gold set rather than from the drafts: a chunk whose
    draft was screened out carries no label and is a legitimate second draw,
    while a chunk that is already someone's answer would produce a duplicate
    row measuring the same retrieval twice.
    """
    if not path.exists():
        return set()
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {chunk_id for row in rows for chunk_id in row.get("gold_chunk_ids", [])}


def sample_chunks(chunks: list[StoredChunk], n: int, seed: int) -> list[StoredChunk]:
    """Draw n chunks reproducibly, preferring section starts.

    Sorted before sampling so the draw depends on the seed alone, not on the
    order rows happened to come back in. Section starts are exhausted before
    any mid-section chunk is taken.
    """
    ordered = sorted(chunks, key=lambda c: c.chunk_id)
    rng = random.Random(seed)
    preferred = [c for c in ordered if starts_at_heading(c)]
    fallback = [c for c in ordered if not starts_at_heading(c)]

    if n <= len(preferred):
        return rng.sample(preferred, n)
    return rng.sample(preferred, len(preferred)) + rng.sample(
        fallback, min(n - len(preferred), len(fallback))
    )


# --------------------------------------------------------------------------
# Groq
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_client() -> Groq:
    """The process-wide Groq client. Fails naming GROQ_API_KEY if unset."""
    from groq import Groq

    return Groq(api_key=require_groq_api_key())


class DailyQuotaExhausted(RuntimeError):
    """The account's tokens-per-day budget is gone. Retrying cannot help."""


# Groq enforces two different rate limits through the same 429. Per-minute is
# transient and a retry clears it. Per-day is not: the bucket refills at
# limit/1440 tokens a minute, so every subsequent call for hours will fail the
# same way. The first run of this module could not tell them apart, retried
# through 259 chunks after the daily budget ran out, and wrote 256 error rows
# that looked like content failures. Whatever is left when this fires is worth
# keeping; the next few hundred calls are not.
_DAILY_QUOTA = re.compile(r"tokens per day|\bTPD\b", re.IGNORECASE)


def is_daily_quota_error(exc: Exception) -> bool:
    """True if this failure is the daily token budget rather than a blip."""
    return type(exc).__name__ == "RateLimitError" and bool(_DAILY_QUOTA.search(str(exc)))


def build_prompt(chunk_text: str) -> str:
    return USER_PROMPT.format(chunk_text=chunk_text)


# Deliberately NOT a retrieval check. Verifying a label by embedding the
# question and looking at its rank would keep only questions the dense
# retriever already answers, inflate every lane's score, and turn the six-lane
# comparison into a measurement of the thing used to build the gold set. The
# check has to sit outside the system under test, so it reads the passages.
LABEL_CHECK_SYSTEM = (
    "You check data labels. You reply with a single passage number, or NONE. "
    "No explanation, no punctuation, no other words."
)

LABEL_CHECK_PROMPT = """\
Below are consecutive passages from one document, numbered in the order they
appear, followed by a question.

Which single passage contains the answer to the question? Reply with just its
number. If none of them contains the answer, reply NONE.

Question: {question}

{passages}
Answer:"""


def build_label_check_prompt(question: str, passages: list[str]) -> str:
    """Render the label-check prompt for passages given in document order."""
    rendered = "\n\n".join(
        f'Passage {i}:\n"""\n{text}\n"""' for i, text in enumerate(passages, start=1)
    )
    return LABEL_CHECK_PROMPT.format(question=question, passages=rendered)


def parse_label_choice(reply: str, count: int) -> int | None:
    """Read the model's answer as a 1-based passage index, or None."""
    text = reply.strip().upper()
    if text.startswith("NONE") or "NONE" in text[:12]:
        return None
    match = re.search(r"\b([1-9])\b", text)
    if match is None:
        return None
    choice = int(match.group(1))
    return choice if 1 <= choice <= count else None


@dataclass(frozen=True)
class LabelCheck:
    """The outcome of checking one draft's gold label."""

    verdict: str  # "confirmed" | "relabelled" | "unverified"
    chunk: StoredChunk | None


def verify_label(
    client: Groq,
    model: str,
    question: str,
    target: StoredChunk,
    neighbours: list[StoredChunk],
) -> LabelCheck:
    """Ask which of the target and its siblings actually answers the question.

    `neighbours` is the target plus its immediate previous and next siblings,
    in document order. Order is documentary rather than shuffled so the run is
    reproducible; the target is not always in the same position, because the
    first chunk of a document has no predecessor.
    """
    passages = [c.text for c in neighbours]
    response = client.chat.completions.create(
        model=model,
        # Zero: this is a classification, and it has to give the same answer
        # for the same inputs or the gold set is not reproducible.
        temperature=0,
        messages=[
            {"role": "system", "content": LABEL_CHECK_SYSTEM},
            {"role": "user", "content": build_label_check_prompt(question, passages)},
        ],
    )
    choice = parse_label_choice(response.choices[0].message.content or "", len(passages))
    if choice is None:
        return LabelCheck("unverified", None)
    picked = neighbours[choice - 1]
    verdict = "confirmed" if picked.chunk_id == target.chunk_id else "relabelled"
    return LabelCheck(verdict, picked)


def draft_question(client: Groq, model: str, chunk_text: str) -> str:
    """One Groq call. Returns the question text, stripped of stray quoting."""
    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(chunk_text)},
        ],
    )
    question = (response.choices[0].message.content or "").strip()
    # Models wrap answers in quotes often enough to be worth undoing, and a
    # stray quote would otherwise survive into the published gold set.
    if len(question) >= 2 and question[0] == question[-1] and question[0] in "\"'":
        question = question[1:-1].strip()
    return question


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


def draft_row(
    qid: str,
    chunk: StoredChunk,
    question: str,
    model: str,
    check: LabelCheck,
) -> dict[str, object]:
    """One draft in the schema from docs/02_EVALUATION_SPEC.md §1.

    `gold_chunk_ids` carries the chunk the label check settled on, which is not
    always the one the question was drafted from.
    """
    gold = check.chunk or chunk
    return {
        "qid": qid,
        "question": question,
        "gold_chunk_ids": [gold.chunk_id],
        "source_doc": gold.source_doc,
        "label_check": check.verdict,
        "drafted_from": chunk.chunk_id,
        # Drafts default to "factual". The verifier corrects it; nothing
        # downstream reads it before a human has seen the row.
        "query_type": "factual",
        # "unverified" rows are recorded but are not drafts: the check could
        # not find the answer in the labelled chunk or either neighbour, so
        # there is no labelled pair left to screen.
        "status": "draft" if check.verdict != "unverified" else "unverified",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "temperature": TEMPERATURE,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def error_row(qid: str, chunk: StoredChunk, message: str, model: str) -> dict[str, object]:
    """A failed call, kept alongside the drafts so the run is auditable.

    `status` is "error", not "draft", so the verification CLI skips it. A run
    that silently produced 143 of 150 rows would look like a clean run.
    """
    return {
        "qid": qid,
        "question": None,
        "gold_chunk_ids": [chunk.chunk_id],
        "source_doc": chunk.source_doc,
        "query_type": None,
        "status": "error",
        "error": message,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "temperature": TEMPERATURE,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def neighbourhood(
    chunk: StoredChunk,
    by_position: dict[tuple[str, int], StoredChunk],
) -> list[StoredChunk]:
    """The chunk plus its immediate siblings, in document order."""
    return [
        by_position[key]
        for key in (
            (chunk.source_doc, chunk.chunk_index - 1),
            (chunk.source_doc, chunk.chunk_index),
            (chunk.source_doc, chunk.chunk_index + 1),
        )
        if key in by_position
    ]


def _retry_once(call, retry_delay: float):  # noqa: ANN001, ANN202
    """Run `call`, retrying once. Returns (result, error_message).

    Raises `DailyQuotaExhausted` rather than retrying when the daily token
    budget is gone -- see `is_daily_quota_error`.
    """
    error: str | None = None
    for attempt in (1, 2):
        try:
            return call(), None
        except Exception as exc:  # any API failure is the same to us here
            if is_daily_quota_error(exc):
                raise DailyQuotaExhausted(str(exc)) from exc
            error = f"{type(exc).__name__}: {exc}"
            if attempt == 1:
                time.sleep(retry_delay)
    return None, error


def existing_drafts(path: Path) -> dict[str, dict]:
    """Rows from an earlier run that are worth keeping, keyed by qid.

    Error rows are deliberately not kept: a failed call is work to redo, not a
    result. Everything else is, because each row cost API budget that the free
    tier meters by the day.
    """
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["qid"]: r for r in rows if r.get("status") in {"draft", "unverified"}}


def _draft_one(
    qid: str,
    chunk: StoredChunk,
    model: str,
    client: Groq,
    by_position: dict[tuple[str, int], StoredChunk],
    retry_delay: float,
) -> tuple[dict, str, str]:
    """Draft and label-check one chunk. Returns (row, note, tally key)."""
    question, error = _retry_once(
        lambda c=chunk: draft_question(client, model, c.text), retry_delay
    )
    if not question:
        row = error_row(qid, chunk, error or "empty response", model)
        return row, f"FAILED ({error})", "failed"

    neighbours = neighbourhood(chunk, by_position) or [chunk]
    check, check_error = _retry_once(
        lambda c=chunk, q=question, n=neighbours: verify_label(client, model, q, c, n),
        retry_delay,
    )
    # A failed check is not a confirmed label. Treat it as one and the whole
    # point of this step is lost.
    if check is None:
        check = LabelCheck("unverified", None)
        note = f"unverified (check failed: {check_error})"
    elif check.verdict == "relabelled":
        note = f"relabelled -> {check.chunk.chunk_id}"  # type: ignore[union-attr]
    else:
        note = check.verdict
    return draft_row(qid, chunk, question, model, check), note, check.verdict


def generate(
    chunks: list[StoredChunk],
    out_path: Path,
    model: str,
    client: Groq,
    by_position: dict[tuple[str, int], StoredChunk] | None = None,
    retry_delay: float = 2.0,
    qid_start: int = 1,
) -> Counter[str]:
    """Draft a question per chunk, check its label, and write each row as it lands.

    Rows are flushed individually so an interrupted run keeps what it earned,
    and one failed chunk never ends the run — a single bad passage must not
    cost hundreds of API calls.

    Resumable, and it has to be: the free tier meters tokens by the day, so a
    full pass can span more than one. Drafts from an earlier run are re-emitted
    without an API call, and only the gaps are drafted. Running out of daily
    budget stops the calls and keeps every row already earned, rather than
    retrying its way through the remaining chunks writing error rows.

    `qid` is `qid_start` plus the position in the sample, so resuming is only
    coherent with the same `--n`, `--seed` and `--qid-start`. All three are
    recorded on every row.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_position = by_position or {}
    tally: Counter[str] = Counter()
    total = len(chunks)
    # Read before truncating, or the rerun destroys what it meant to resume.
    done = existing_drafts(out_path)
    if done:
        print(f"resuming: {len(done)} drafts kept from an earlier run\n")
    quota_gone = False

    with out_path.open("w", encoding="utf-8") as handle:
        for index, chunk in enumerate(chunks, start=1):
            qid = f"q{qid_start + index - 1:03d}"

            if qid in done:
                row = done[qid]
                tally[row.get("label_check", "unverified")] += 1
                note = "kept from an earlier run"
            elif quota_gone:
                continue
            else:
                try:
                    row, note, key = _draft_one(
                        qid, chunk, model, client, by_position, retry_delay
                    )
                except DailyQuotaExhausted as exc:
                    quota_gone = True
                    print(f"\n  DAILY TOKEN BUDGET EXHAUSTED at {qid}.\n  {exc}")
                    print("  Keeping every row earned so far. Re-run to continue.\n")
                    continue
                tally[key] += 1

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"  [{index:>3}/{total}] {chunk.chunk_id:<52} {note}")

    if quota_gone:
        tally["quota_exhausted"] = 1
    return tally


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.goldset.generate",
        description="Draft gold-set questions from ingested chunks. Drafts are not labels.",
    )
    parser.add_argument("--n", type=int, default=150, help="how many chunks to sample")
    parser.add_argument("--corpus-id", default="demo")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="sampling seed; 42 throughout this project so runs are comparable",
    )
    parser.add_argument(
        "--exclude-labelled",
        type=Path,
        default=None,
        help=(
            "a gold-set JSONL. Every chunk it labels is removed from the pool "
            "before sampling, so a top-up run cannot redraft a chunk that "
            "already has a question."
        ),
    )
    parser.add_argument(
        "--qid-start",
        type=int,
        default=1,
        help=(
            "first qid number. A top-up run must not reuse the qids of the run "
            "it is topping up: two different questions sharing a qid would "
            "silently collide when the drafts are screened into one file."
        ),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    # Fail before opening a database connection or writing a file.
    client = get_client()

    with dense_store.connect() as conn:
        stored = dense_store.fetch_chunks(conn, args.corpus_id)
    if not stored:
        print(f"No chunks found for corpus {args.corpus_id!r}. Run backend.ingest first.")
        return 1

    pool = select_eligible(stored)
    labelled = labelled_chunk_ids(args.exclude_labelled) if args.exclude_labelled else set()
    available = [c for c in pool.eligible if c.chunk_id not in labelled]
    sample = sample_chunks(available, args.n, args.seed)
    by_position = {(c.source_doc, c.chunk_index): c for c in stored}
    section_starts = sum(1 for c in sample if starts_at_heading(c))

    print(f"corpus       {args.corpus_id}: {pool.considered} chunks")
    print(f"  skipped    {pool.too_short} under {MIN_CHUNK_CHARS} chars")
    print(f"  skipped    {pool.too_much_code} over {MAX_NON_PROSE_FRACTION:.0%} non-prose")
    print(f"  skipped    {pool.scaffolding} over "
          f"{MAX_SCAFFOLDING_FRACTION:.0%} frontmatter/imports")
    print(f"  skipped    {pool.code_edges} opening or closing inside code")
    print(f"  eligible   {len(pool.eligible)}")
    if labelled:
        print(f"  skipped    {len(pool.eligible) - len(available)} already gold-labelled "
              f"({args.exclude_labelled})")
        print(f"  available  {len(available)}")
    if len(available) < args.n:
        print(f"\nWARNING: only {len(available)} chunks available for a request of {args.n}.")
    print(f"sampling     {len(sample)} (requested {args.n}, seed {args.seed})")
    print(f"  of which   {section_starts} start at a heading, "
          f"{len(sample) - section_starts} mid-section")
    print(f"model        {settings.goldset_model} @ temperature {TEMPERATURE}")
    print(f"prompt       {PROMPT_VERSION}")
    print(f"qids         q{args.qid_start:03d}-q{args.qid_start + len(sample) - 1:03d}")
    print(f"writing      {args.out}\n")

    tally = generate(
        sample, args.out, settings.goldset_model, client, by_position, qid_start=args.qid_start
    )

    usable = tally["confirmed"] + tally["relabelled"]
    print(f"\nconfirmed    {tally['confirmed']}")
    print(f"relabelled   {tally['relabelled']}")
    print(f"unverified   {tally['unverified']}  (recorded, not offered for screening)")
    print(f"failed       {tally['failed']}")
    print(f"drafts       {usable}")
    print("\nThese are drafts, not labels. Screen them with backend.goldset.screen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
