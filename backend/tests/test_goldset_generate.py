"""Gold-set drafting tests. The Groq client and the store are both stubbed."""

import json

import pytest

from backend.goldset.generate import (
    MAX_NON_PROSE_FRACTION,
    MIN_CHUNK_CHARS,
    PROMPT_VERSION,
    TEMPERATURE,
    build_prompt,
    closes_inside_code,
    generate,
    has_code_edges,
    neighbourhood,
    non_prose_fraction,
    opens_inside_code,
    parse_label_choice,
    sample_chunks,
    select_eligible,
    starts_at_heading,
)
from backend.retrieval.dense_store import StoredChunk

PROSE = (
    "You can define background tasks to be run after returning a response. "
    "This is useful for operations that need to happen after a request, but "
    "that the client does not have to wait for before receiving the response. "
    "Email notifications are the usual example, because connecting to a mail "
    "server tends to be slow enough to be worth deferring."
)


def chunk(
    chunk_id: str,
    text: str,
    source_doc: str = "fastapi/index.md",
    chunk_index: int = 0,
) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        corpus_id="demo",
        source_doc=source_doc,
        chunk_index=chunk_index,
        text=text,
    )


# --- non-prose detection ------------------------------------------------


def test_plain_prose_is_mostly_prose():
    assert non_prose_fraction(PROSE) == 0.0


def test_include_directives_count_as_non_prose():
    text = "\n".join(["{* ../../docs_src/background_tasks/tutorial001.py hl[1,13] *}"] * 4)
    assert non_prose_fraction(text) == 1.0


def test_fenced_code_counts_as_non_prose():
    # 60 chars total: a 34-char prose line, then 26 chars of fence and code.
    text = "Some prose line here about tasks.\n```python\nx = 1\ny = 2\n```\n"
    assert len(text) == 60
    assert non_prose_fraction(text) == pytest.approx(26 / 60)


def test_chunk_opening_inside_a_code_block_is_detected():
    """A bare closing fence means the block opened in the previous chunk."""
    orphan = "    values={'messages': [HumanMessage(content='hi', id='abc')]},\n    next=(),\n```\n"
    assert non_prose_fraction(orphan) > MAX_NON_PROSE_FRACTION


def test_serialised_output_without_any_fence_is_detected():
    dump = (
        "'thread_id': '1'}, created_at='2025-05-05T16:01:23.862295+00:00', "
        "parent_config={...} tasks=(), interrupts=() ), StateSnapshot(\n"
    )
    assert non_prose_fraction(dump) > MAX_NON_PROSE_FRACTION


def test_empty_text_is_all_non_prose():
    assert non_prose_fraction("") == 1.0


# --- eligibility --------------------------------------------------------


def test_short_chunks_are_skipped():
    short = chunk("a_ch_000", "x" * (MIN_CHUNK_CHARS - 1))
    long = chunk("b_ch_000", PROSE)
    result = select_eligible([short, long])

    assert result.too_short == 1
    assert [c.chunk_id for c in result.eligible] == ["b_ch_000"]
    assert result.considered == 2


def test_code_heavy_chunks_are_skipped():
    code = chunk("a_ch_000", "```python\n" + "value = compute(x, y)\n" * 20 + "```\n")
    result = select_eligible([code, chunk("b_ch_000", PROSE)])

    assert result.too_much_code == 1
    assert [c.chunk_id for c in result.eligible] == ["b_ch_000"]


def test_a_chunk_is_counted_once_even_if_it_fails_both_rules():
    tiny_code = chunk("a_ch_000", "```\nx=1\n```")
    result = select_eligible([tiny_code])
    assert result.too_short + result.too_much_code + result.code_edges == 1
    assert result.considered == 1


# --- code edges (the v1 q002 regression) --------------------------------

# The real chunk that produced the v1 failure: it opens with an orphaned JSON
# fragment, and the model asked "What data types are specified for the 'tax'
# and 42.0 values in the given JSON object?".
Q002_CHUNK = (
    '42.0,\n    "tax": 3.2\n}\n```\n\n## Recap { #recap }\n\n'
    "You can add multiple body parameters to your path operation function, even "
    "though a request can only have a single body.\n\n"
    "But FastAPI will handle it, give you the correct data in your function, and "
    "validate and document the correct schema in the path operation.\n\n"
    "You can also declare singular values to be received as part of the body.\n"
)


def test_the_q002_chunk_reads_as_prose_overall():
    """Why the overall fraction was not enough: this chunk looks fine by it."""
    assert non_prose_fraction(Q002_CHUNK) < MAX_NON_PROSE_FRACTION


def test_the_q002_chunk_is_caught_by_the_edge_rule():
    assert has_code_edges(Q002_CHUNK) is True


def test_the_q002_chunk_is_now_ineligible():
    result = select_eligible([chunk("fastapi_tutorial_body_multiple_params_ch_003", Q002_CHUNK)])
    assert result.eligible == []
    assert result.code_edges == 1


def test_chunk_opening_after_a_tagged_fence_is_not_treated_as_orphaned():
    """```python opens a block, so the chunk did not start inside one."""
    text = "Some prose about tasks here.\n```python\nx = 1\n```\n" + PROSE
    assert opens_inside_code(text) is False


def test_chunk_whose_first_fence_is_bare_opened_inside_a_block():
    text = "    result = compute(x)\n```\n\n" + PROSE
    assert opens_inside_code(text) is True


def test_chunk_ending_inside_an_open_fence_is_detected():
    text = PROSE + "\n```python\nfrom langgraph.graph import StateGraph\nbuilder = St"
    assert closes_inside_code(text) is True


def test_clean_prose_has_no_code_edges():
    assert has_code_edges(PROSE) is False


def test_prose_with_a_closed_code_block_in_the_middle_is_eligible():
    text = PROSE + "\n\n```python\nx = 1\n```\n\n" + PROSE
    assert has_code_edges(text) is False
    assert select_eligible([chunk("a_ch_000", text)]).eligible != []


# --- sampling -----------------------------------------------------------


def test_sampling_is_reproducible_for_a_given_seed():
    pool = [chunk(f"c_ch_{i:03d}", PROSE) for i in range(50)]
    first = [c.chunk_id for c in sample_chunks(pool, 10, seed=42)]
    second = [c.chunk_id for c in sample_chunks(pool, 10, seed=42)]
    assert first == second


def test_sampling_does_not_depend_on_input_order():
    pool = [chunk(f"c_ch_{i:03d}", PROSE) for i in range(50)]
    forward = [c.chunk_id for c in sample_chunks(pool, 10, seed=42)]
    backward = [c.chunk_id for c in sample_chunks(list(reversed(pool)), 10, seed=42)]
    assert forward == backward


def test_a_different_seed_draws_a_different_sample():
    pool = [chunk(f"c_ch_{i:03d}", PROSE) for i in range(50)]
    assert [c.chunk_id for c in sample_chunks(pool, 10, seed=42)] != [
        c.chunk_id for c in sample_chunks(pool, 10, seed=7)
    ]


def test_requesting_more_than_the_pool_returns_the_whole_pool():
    pool = [chunk(f"c_ch_{i:03d}", PROSE) for i in range(3)]
    assert len(sample_chunks(pool, 150, seed=42)) == 3


# --- prompt -------------------------------------------------------------


def test_prompt_carries_the_spec_wording_and_the_corpus_constraint():
    prompt = build_prompt(PROSE)
    assert "only this passage" in prompt
    assert '"this passage"' in prompt
    assert '"the document"' in prompt
    # Corpus-specific: referenced code is not in the ingested text.
    assert "{* ../../docs_src/example.py *}" in prompt
    assert "NOT available" in prompt
    assert PROSE in prompt


def test_prompt_carries_each_fix_for_an_earlier_failure():
    prompt = build_prompt(PROSE)
    # orphaned fragments -- the v1 q002 failure
    assert "Ignore any such fragment entirely" in prompt
    # vague phrasings -- the v1 q001/q006/q007 failure
    assert "What is the purpose of" in prompt
    assert "Why is ... important" in prompt
    # parallel language tabs -- the v1 q009 failure
    assert ":::python" in prompt
    assert "shared concept" in prompt
    # verbatim quoting -- the v2 q003/q004/q005 failure
    assert "more than four consecutive words" in prompt
    # v2 emitted questions ending in a full stop
    assert "ending in a question mark" in prompt


def test_prompt_carries_both_worked_example_pairs():
    prompt = build_prompt(PROSE)
    # The rule-5 pair, verbatim.
    assert "X-Accel-Buffering: no" in prompt
    assert "Which response header stops reverse proxies" in prompt
    # The rule-4 pair, drawn from the v2 q002 failure.
    assert "What is the name of the framework being discussed?" in prompt
    assert "Which extra package must be added" in prompt
    # Each bad example names the rule it breaks.
    assert "breaks rule 5" in prompt
    assert "breaks rule 4" in prompt


# --- the run ------------------------------------------------------------


class StubClient:
    """Canned question, then a canned label-check answer.

    A label check is any call whose prompt asks for a passage number, which is
    how the real two-call sequence is distinguished here.
    """

    def __init__(self, fail_for=(), fail_times=99, label_answer="2"):
        self.fail_for = set(fail_for)
        self.fail_times = fail_times
        self.label_answer = label_answer
        self.calls: list[str] = []
        self.label_calls: list[str] = []
        self.completions = self
        self.chat = self

    def create(self, model, temperature, messages):
        body = messages[1]["content"]
        is_label_check = "Which single passage contains the answer" in body
        if is_label_check:
            self.label_calls.append(body)
            return _Response(self.label_answer)

        self.calls.append(body)
        for marker in self.fail_for:
            if marker in body and len(self.calls) <= self.fail_times:
                raise RuntimeError("upstream 503")
        return _Response('"What are background tasks for?"')


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Message:
    def __init__(self, content):
        self.content = content


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_rows_match_the_spec_schema(tmp_path):
    out = tmp_path / "draft.jsonl"
    tally = generate(
        [chunk("nadra_ch_042", PROSE)],
        out,
        "llama-3.1-8b-instant",
        StubClient(label_answer="1"),
    )

    assert tally["confirmed"] == 1
    (row,) = read_rows(out)
    assert row["qid"] == "q001"
    assert row["gold_chunk_ids"] == ["nadra_ch_042"]
    assert row["source_doc"] == "fastapi/index.md"
    assert row["status"] == "draft"
    assert row["label_check"] == "confirmed"
    assert row["query_type"] == "factual"
    assert row["model"] == "llama-3.1-8b-instant"
    assert row["prompt_version"] == PROMPT_VERSION
    assert row["temperature"] == TEMPERATURE


def test_surrounding_quotes_are_stripped_from_the_question(tmp_path):
    out = tmp_path / "draft.jsonl"
    generate([chunk("a_ch_000", PROSE)], out, "m", StubClient(label_answer="1"))
    assert read_rows(out)[0]["question"] == "What are background tasks for?"


def test_qids_are_sequential_and_zero_padded(tmp_path):
    out = tmp_path / "draft.jsonl"
    generate(
        [chunk(f"a_ch_{i:03d}", PROSE) for i in range(3)],
        out,
        "m",
        StubClient(label_answer="1"),
    )
    assert [r["qid"] for r in read_rows(out)] == ["q001", "q002", "q003"]


def test_qid_start_offsets_a_top_up_run(tmp_path):
    """A top-up must not reuse the qids of the run it tops up."""
    out = tmp_path / "draft.jsonl"
    generate(
        [chunk(f"a_ch_{i:03d}", PROSE) for i in range(3)],
        out,
        "m",
        StubClient(label_answer="1"),
        qid_start=120,
    )
    assert [r["qid"] for r in read_rows(out)] == ["q120", "q121", "q122"]


def test_one_failure_does_not_end_the_run(tmp_path):
    out = tmp_path / "draft.jsonl"
    chunks = [
        chunk("a_ch_000", PROSE),
        chunk("b_ch_000", "BADCHUNK " + PROSE),
        chunk("c_ch_000", PROSE),
    ]
    tally = generate(
        chunks, out, "m", StubClient(fail_for=["BADCHUNK"], label_answer="1"), retry_delay=0
    )

    rows = read_rows(out)
    assert tally["confirmed"] == 2
    assert tally["failed"] == 1
    assert [r["status"] for r in rows] == ["draft", "error", "draft"]
    assert rows[1]["question"] is None
    assert "upstream 503" in rows[1]["error"]
    # The failed chunk keeps its provenance so the run can be audited.
    assert rows[1]["gold_chunk_ids"] == ["b_ch_000"]


def test_the_call_is_retried_exactly_once(tmp_path):
    out = tmp_path / "draft.jsonl"
    client = StubClient(fail_for=["BADCHUNK"], fail_times=1, label_answer="1")
    tally = generate(
        [chunk("b_ch_000", "BADCHUNK " + PROSE)], out, "m", client, retry_delay=0
    )

    assert tally["confirmed"] == 1  # first call failed, retry succeeded
    assert len(client.calls) == 2


def test_no_quality_filtering_happens(tmp_path):
    """Verification is a human step; a vague draft must still reach the file."""
    out = tmp_path / "draft.jsonl"

    class VagueClient(StubClient):
        def create(self, model, temperature, messages):
            if "Which single passage contains the answer" in messages[1]["content"]:
                return _Response("1")
            return _Response("What is it?")

    generate([chunk("a_ch_000", PROSE)], out, "m", VagueClient())
    assert read_rows(out)[0]["question"] == "What is it?"
    assert read_rows(out)[0]["status"] == "draft"


# --- label verification -------------------------------------------------


def test_neighbourhood_is_the_chunk_and_its_two_siblings():
    doc = "fastapi/index.md"
    chunks = [chunk(f"a_ch_{i:03d}", PROSE, doc, chunk_index=i) for i in range(4)]
    by_position = {(c.source_doc, c.chunk_index): c for c in chunks}

    middle = neighbourhood(chunks[2], by_position)
    assert [c.chunk_id for c in middle] == ["a_ch_001", "a_ch_002", "a_ch_003"]
    # The first chunk of a document has no predecessor.
    first = neighbourhood(chunks[0], by_position)
    assert [c.chunk_id for c in first] == ["a_ch_000", "a_ch_001"]
    # ...and the last has no successor.
    last = neighbourhood(chunks[3], by_position)
    assert [c.chunk_id for c in last] == ["a_ch_002", "a_ch_003"]


def test_neighbours_never_cross_a_document_boundary():
    by_position = {
        ("a.md", 0): chunk("a_ch_000", PROSE, "a.md", 0),
        ("b.md", 0): chunk("b_ch_000", PROSE, "b.md", 0),
    }
    assert [c.chunk_id for c in neighbourhood(by_position[("a.md", 0)], by_position)] == [
        "a_ch_000"
    ]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("1", 1), ("2", 2), (" 3 ", 3), ("NONE", None), ("none", None), ("Passage 2", 2), ("", None)],
)
def test_label_choice_parsing(reply, expected):
    assert parse_label_choice(reply, 3) == expected


def test_a_choice_outside_the_offered_passages_is_none():
    assert parse_label_choice("7", 3) is None


def test_confirmed_label_keeps_the_drafted_chunk(tmp_path):
    out = tmp_path / "draft.jsonl"
    doc = "fastapi/index.md"
    chunks = [chunk(f"a_ch_{i:03d}", PROSE, doc, chunk_index=i) for i in range(3)]
    by_position = {(c.source_doc, c.chunk_index): c for c in chunks}

    # Passage 2 of [prev, target, next] is the target.
    generate([chunks[1]], out, "m", StubClient(label_answer="2"), by_position)
    row = read_rows(out)[0]
    assert row["label_check"] == "confirmed"
    assert row["gold_chunk_ids"] == ["a_ch_001"]
    assert row["drafted_from"] == "a_ch_001"
    assert row["status"] == "draft"


def test_a_neighbour_wins_and_the_row_is_relabelled(tmp_path):
    """The failure this whole step exists for: the answer is in the next chunk."""
    out = tmp_path / "draft.jsonl"
    doc = "fastapi/index.md"
    chunks = [chunk(f"a_ch_{i:03d}", PROSE, doc, chunk_index=i) for i in range(3)]
    by_position = {(c.source_doc, c.chunk_index): c for c in chunks}

    generate([chunks[1]], out, "m", StubClient(label_answer="3"), by_position)
    row = read_rows(out)[0]
    assert row["label_check"] == "relabelled"
    assert row["gold_chunk_ids"] == ["a_ch_002"]  # moved to the neighbour
    assert row["drafted_from"] == "a_ch_001"  # provenance kept
    assert row["status"] == "draft"


def test_none_of_the_passages_marks_the_row_unverified(tmp_path):
    out = tmp_path / "draft.jsonl"
    generate([chunk("a_ch_000", PROSE)], out, "m", StubClient(label_answer="NONE"))
    row = read_rows(out)[0]
    assert row["label_check"] == "unverified"
    # Not a draft: there is nothing for a human to verify.
    assert row["status"] == "unverified"


def test_a_failed_label_check_is_not_treated_as_confirmed(tmp_path):
    """A check that errored proves nothing; defaulting to confirmed would
    silently reintroduce exactly the mislabelling this step prevents."""
    out = tmp_path / "draft.jsonl"

    class CheckFailsClient(StubClient):
        def create(self, model, temperature, messages):
            if "Which single passage contains the answer" in messages[1]["content"]:
                raise RuntimeError("upstream 503")
            return _Response("What are background tasks for?")

    tally = generate(
        [chunk("a_ch_000", PROSE)], out, "m", CheckFailsClient(), retry_delay=0
    )
    assert tally["unverified"] == 1
    assert read_rows(out)[0]["label_check"] == "unverified"


def test_the_label_check_never_sees_a_retrieval_score(tmp_path):
    """The check must stay outside the system under test.

    If the gold set were filtered by retrieval rank it would keep only the
    questions the dense lane already answers, and every lane's score would be
    inflated by construction.
    """
    out = tmp_path / "draft.jsonl"
    client = StubClient(label_answer="1")
    generate([chunk("a_ch_000", PROSE)], out, "m", client)

    (check_prompt,) = client.label_calls
    assert "Passage 1" in check_prompt
    for forbidden in ("similarity", "distance", "rank", "embedding", "score"):
        assert forbidden not in check_prompt.lower()


# --- heading-preferred sampling -----------------------------------------


def test_chunk_zero_counts_as_a_section_start():
    assert starts_at_heading(chunk("a_ch_000", PROSE, chunk_index=0)) is True


def test_a_chunk_opening_with_a_heading_is_a_section_start():
    assert starts_at_heading(chunk("a_ch_005", "## Recap { #recap }\n\n" + PROSE, chunk_index=5))


def test_a_chunk_opening_mid_sentence_is_not_a_section_start():
    assert starts_at_heading(chunk("a_ch_005", PROSE, chunk_index=5)) is False


def test_sampling_exhausts_section_starts_before_mid_section_chunks():
    heads = [
        chunk(f"h_ch_{i:03d}", "# Title\n\n" + PROSE, "h.md", chunk_index=i) for i in range(1, 6)
    ]
    mids = [chunk(f"m_ch_{i:03d}", PROSE, "m.md", chunk_index=i) for i in range(1, 21)]

    drawn = sample_chunks(heads + mids, 5, seed=42)
    assert {c.chunk_id for c in drawn} == {c.chunk_id for c in heads}


def test_sampling_tops_up_from_mid_section_chunks_when_it_must():
    heads = [
        chunk(f"h_ch_{i:03d}", "# Title\n\n" + PROSE, "h.md", chunk_index=i) for i in range(1, 4)
    ]
    mids = [chunk(f"m_ch_{i:03d}", PROSE, "m.md", chunk_index=i) for i in range(1, 21)]

    drawn = sample_chunks(heads + mids, 8, seed=42)
    assert len(drawn) == 8
    assert {c.chunk_id for c in heads} <= {c.chunk_id for c in drawn}


@pytest.mark.parametrize("missing", ["GROQ_API_KEY"])
def test_missing_groq_key_names_the_variable(monkeypatch, missing):
    from backend.config import MissingSecretError, get_settings, require_groq_api_key

    monkeypatch.delenv(missing, raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    get_settings.cache_clear()
    try:
        with pytest.raises(MissingSecretError) as excinfo:
            require_groq_api_key()
    finally:
        get_settings.cache_clear()
    assert missing in str(excinfo.value)


# --- daily quota and resuming -------------------------------------------
#
# The first 350-chunk run exhausted the account's tokens-per-day budget at
# chunk 92, then retried its way through the remaining 259 and wrote 256 error
# rows. These tests pin down both halves of the fix: stop when the budget is
# gone, and never destroy drafts that budget already paid for.


class RateLimitError(Exception):
    """Named to match the class the Groq SDK raises; detection is by name."""


DAILY = (
    "Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` in "
    "organization `org_x` service tier `on_demand` on tokens per day (TPD): "
    "Limit 200000, Used 199265, Requested 1670."
)
PER_MINUTE = (
    "Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` on "
    "tokens per minute (TPM): Limit 8000, Used 7900, Requested 1670."
)


def test_a_daily_quota_error_is_told_apart_from_a_per_minute_one():
    from backend.goldset.generate import is_daily_quota_error

    assert is_daily_quota_error(RateLimitError(DAILY))
    assert not is_daily_quota_error(RateLimitError(PER_MINUTE))
    assert not is_daily_quota_error(RuntimeError(DAILY))  # a 503 is not a quota


class QuotaClient(StubClient):
    """Drafts normally until `after` questions, then reports the daily budget gone."""

    def __init__(self, after, **kwargs):
        super().__init__(**kwargs)
        self.after = after

    def create(self, model, temperature, messages):
        body = messages[1]["content"]
        if "Which single passage contains the answer" not in body and len(self.calls) >= self.after:
            raise RateLimitError(DAILY)
        return super().create(model, temperature, messages)


def test_running_out_of_daily_budget_stops_rather_than_writing_error_rows(tmp_path):
    out = tmp_path / "drafts.jsonl"
    chunks = [chunk(f"a_ch_{i:03d}", PROSE, chunk_index=i) for i in range(6)]

    tally = generate(chunks, out, "m", QuotaClient(after=2, label_answer="1"), retry_delay=0)

    rows = read_rows(out)
    assert [r["qid"] for r in rows] == ["q001", "q002"]  # nothing after the wall
    assert not any(r["status"] == "error" for r in rows)
    assert tally["quota_exhausted"] == 1


def test_a_rerun_keeps_drafts_that_already_cost_budget(tmp_path):
    """`generate` opens the file with "w". Reading before truncating is the
    only thing standing between a rerun and the loss of a day's tokens."""
    out = tmp_path / "drafts.jsonl"
    chunks = [chunk(f"a_ch_{i:03d}", PROSE, chunk_index=i) for i in range(4)]

    generate(chunks, out, "m", QuotaClient(after=2, label_answer="1"), retry_delay=0)
    first_pass = read_rows(out)
    generate(chunks, out, "m", StubClient(label_answer="1"), retry_delay=0)

    rows = read_rows(out)
    assert [r["qid"] for r in rows] == ["q001", "q002", "q003", "q004"]
    # The two originals are byte-identical: kept, not redrafted.
    assert rows[:2] == first_pass


def test_a_resumed_draft_costs_no_api_call(tmp_path):
    out = tmp_path / "drafts.jsonl"
    chunks = [chunk(f"a_ch_{i:03d}", PROSE, chunk_index=i) for i in range(2)]

    generate(chunks, out, "m", StubClient(label_answer="1"), retry_delay=0)
    client = StubClient(label_answer="1")
    generate(chunks, out, "m", client, retry_delay=0)

    assert client.calls == []  # every row came from the file


def test_error_rows_are_retried_on_a_rerun_rather_than_kept(tmp_path):
    """A failed call is work to redo, not a result."""
    out = tmp_path / "drafts.jsonl"
    chunks = [chunk("a_ch_000", "BADCHUNK " + PROSE)]  # the stub matches on prompt text

    generate(chunks, out, "m", StubClient(fail_for=["BADCHUNK"], label_answer="1"), retry_delay=0)
    assert read_rows(out)[0]["status"] == "error"

    generate(chunks, out, "m", StubClient(label_answer="1"), retry_delay=0)
    assert read_rows(out)[0]["status"] == "draft"


def test_the_tally_counts_resumed_rows_alongside_new_ones(tmp_path):
    out = tmp_path / "drafts.jsonl"
    chunks = [chunk(f"a_ch_{i:03d}", PROSE, chunk_index=i) for i in range(4)]

    generate(chunks[:2], out, "m", StubClient(label_answer="1"), retry_delay=0)
    tally = generate(chunks, out, "m", StubClient(label_answer="1"), retry_delay=0)

    assert tally["confirmed"] == 4


# --- scaffolding ---------------------------------------------------------
#
# Added after the Day 1 audit rejected 7 questions asking what a page's
# `title:` was and 2 asking for an imported MDX snippet's name. Every chunk has
# frontmatter, so such questions are answerable by any chunk and measure nothing.


FRONTMATTER = """\
---
title: Build a custom SQL agent
sidebarTitle: Custom SQL agent
---

import ChatModelTabsPy from '/snippets/chat-model-tabs.mdx';
import SqlAgentToolsPy from '/snippets/code-samples/sql-agent-tools-py.mdx';
import SqlAgentRunAgentPy from '/snippets/code-samples/sql-agent-run-agent-py.mdx';
"""


def test_a_chunk_that_is_all_frontmatter_and_imports_is_scaffolding():
    from backend.goldset.generate import is_scaffolding, scaffolding_fraction

    # Not exactly 1.0: the blank line separating the block from the imports is
    # neither frontmatter nor an import, and counting it either way is noise.
    assert scaffolding_fraction(FRONTMATTER) > 0.99
    assert is_scaffolding(FRONTMATTER)


def test_plain_prose_is_not_scaffolding():
    from backend.goldset.generate import is_scaffolding, scaffolding_fraction

    assert scaffolding_fraction(PROSE) == 0.0
    assert not is_scaffolding(PROSE)


def test_frontmatter_followed_by_enough_prose_survives():
    """The filter removes chunks that are *mostly* scaffolding, not any that have it."""
    from backend.goldset.generate import is_scaffolding

    assert not is_scaffolding("---\ntitle: Tools\n---\n\n" + PROSE)


def test_a_horizontal_rule_is_not_a_second_frontmatter_block():
    """`---` mid-page is a rule. Treating it as frontmatter would swallow the rest."""
    from backend.goldset.generate import scaffolding_fraction

    assert scaffolding_fraction(PROSE + "\n---\n\n" + PROSE) == 0.0


def test_a_chunk_opening_mid_frontmatter_is_still_detected():
    """Chunk boundaries cut through the frontmatter block, leaving no delimiter."""
    from backend.goldset.generate import is_scaffolding

    assert is_scaffolding("title: Agents\nsidebarTitle: Agents\ndescription: Build agents\n")


def test_scaffolding_chunks_are_excluded_from_the_eligible_pool():
    pool = select_eligible(
        [
            chunk("a_ch_000", FRONTMATTER),
            chunk("a_ch_001", PROSE),
        ]
    )
    assert [c.chunk_id for c in pool.eligible] == ["a_ch_001"]
    assert pool.scaffolding == 1
    assert pool.considered == 2


# --- excluding chunks that already carry a label -------------------------


def test_labelled_chunk_ids_reads_primary_and_multilabel_ids(tmp_path):
    """A multi-labelled row's extra chunks are answers too, so they are used up."""
    from backend.goldset.generate import labelled_chunk_ids

    path = tmp_path / "goldset.jsonl"
    path.write_text(
        json.dumps({"qid": "q001", "gold_chunk_ids": ["a_ch_000"]})
        + "\n"
        + json.dumps({"qid": "q002", "gold_chunk_ids": ["b_ch_000", "c_ch_000"]})
        + "\n",
        encoding="utf-8",
    )
    assert labelled_chunk_ids(path) == {"a_ch_000", "b_ch_000", "c_ch_000"}


def test_labelled_chunk_ids_is_empty_when_there_is_no_gold_set_yet(tmp_path):
    from backend.goldset.generate import labelled_chunk_ids

    assert labelled_chunk_ids(tmp_path / "absent.jsonl") == set()
