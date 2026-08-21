"""Judge tests: sentence splitting, verdict parsing, and the reasoning-token trap.

No network. The Groq client is a stub whose responses are written by hand, so
these tests assert what `judge.py` does with a response rather than how good
gpt-oss-20b is at judging. That second question is what the blind calibration in
`test_judge_calibrate.py` exists to answer, and it cannot be answered by a test.
"""

import pytest

from backend.judge import (
    GENERATE_MAX_TOKENS,
    JUDGE_MAX_TOKENS,
    REASONING_EFFORT,
    TEMPERATURE,
    AnswerRecord,
    DailyBudgetExhausted,
    JudgeError,
    SentenceVerdict,
    complete,
    format_context,
    generate_answer,
    judge_answer,
    parse_verdicts,
    split_sentences,
    summarise,
)


class StubMessage:
    def __init__(self, content):
        self.content = content


class StubChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = StubMessage(content)
        self.finish_reason = finish_reason


class StubUsage:
    def __init__(self, prompt_tokens=100, completion_tokens=50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class StubResponse:
    def __init__(self, content, finish_reason="stop", usage=(100, 50)):
        self.choices = [StubChoice(content, finish_reason)]
        self.usage = StubUsage(*usage)


class StubGroq:
    """Returns canned completions in order and records the kwargs it was called with."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# --- sentence splitting ---------------------------------------------------


def test_splitting_finds_the_boundaries_a_generated_answer_actually_has():
    text = "Install python-multipart. It is required for form data. FastAPI raises otherwise."

    assert split_sentences(text) == [
        "Install python-multipart.",
        "It is required for form data.",
        "FastAPI raises otherwise.",
    ]


def test_splitting_does_not_break_on_the_abbreviations_this_corpus_uses():
    text = "Use a form library, e.g. python-multipart. That is the only supported option."

    assert split_sentences(text) == [
        "Use a form library, e.g. python-multipart.",
        "That is the only supported option.",
    ]


def test_splitting_keeps_a_version_number_in_one_sentence():
    # "0.115." would be a boundary under a naive split-on-period.
    text = "Available from version 0.115.0 onward. Earlier releases raise."

    assert split_sentences(text) == [
        "Available from version 0.115.0 onward.",
        "Earlier releases raise.",
    ]


def test_splitting_normalises_whitespace_so_the_same_answer_splits_the_same_way():
    assert split_sentences("One.   Two.") == split_sentences("One.\n\nTwo.") == ["One.", "Two."]


def test_splitting_an_empty_answer_gives_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_splitting_handles_a_sentence_that_opens_with_a_backtick():
    text = "The field is optional. `version` holds the identifier."

    assert split_sentences(text) == ["The field is optional.", "`version` holds the identifier."]


# --- the reasoning-token trap --------------------------------------------


def test_an_empty_completion_raises_and_names_the_reasoning_budget():
    """The Day 2 failure, reproduced. It must never become a verdict."""
    client = StubGroq(
        StubResponse("", finish_reason="length"),
        StubResponse("", finish_reason="length"),
    )

    with pytest.raises(JudgeError) as caught:
        complete(client, "model", "sys", "user", max_tokens=160)

    message = str(caught.value)
    assert "reasoning tokens" in message
    assert "max_tokens=160" in message
    assert "do NOT score this as ungrounded" in message


def test_an_empty_completion_that_was_not_truncated_still_raises():
    client = StubGroq(StubResponse("", finish_reason="stop"), StubResponse("", "stop"))

    with pytest.raises(JudgeError):
        complete(client, "model", "sys", "user", max_tokens=800)


def test_every_call_sets_the_reasoning_effort_and_a_deliberate_budget():
    """A call that forgets either one walks into the Day 2 trap."""
    client = StubGroq(StubResponse("an answer"))

    generate_answer(client, "model", "why?", [("c1", "body")])

    kwargs = client.calls[0]
    assert kwargs["reasoning_effort"] == REASONING_EFFORT == "low"
    assert kwargs["max_tokens"] == GENERATE_MAX_TOKENS
    assert kwargs["temperature"] == TEMPERATURE == 0.0


def test_the_judge_call_gets_the_larger_budget():
    client = StubGroq(StubResponse('{"verdicts":[{"n":1,"supported":true,"reason":"ok"}]}'))

    judge_answer(client, "model", "why?", "One sentence.", [("c1", "body")])

    assert client.calls[0]["max_tokens"] == JUDGE_MAX_TOKENS
    assert JUDGE_MAX_TOKENS > GENERATE_MAX_TOKENS


def test_a_transient_failure_is_retried_once_and_then_gives_up():
    client = StubGroq(RuntimeError("rate limited"), StubResponse("recovered"))

    content, _, _ = complete(client, "model", "sys", "user", max_tokens=100)

    assert content == "recovered"
    assert len(client.calls) == 2


def test_two_failures_raise_rather_than_retrying_forever():
    client = StubGroq(RuntimeError("down"), RuntimeError("still down"))

    with pytest.raises(JudgeError, match="did not return a usable completion"):
        complete(client, "model", "sys", "user", max_tokens=100)


def test_measured_token_usage_stays_well_inside_the_budgets():
    """Pins the headroom claim in the module comment: max 86 and 126 observed."""
    assert GENERATE_MAX_TOKENS >= 86 * 5
    assert JUDGE_MAX_TOKENS >= 126 * 5


# --- verdict parsing ------------------------------------------------------


def test_verdicts_are_aligned_to_the_sentences_they_scored():
    content = (
        '{"verdicts": [{"n": 1, "supported": true, "reason": "stated"}, '
        '{"n": 2, "supported": false, "reason": "invented"}]}'
    )

    verdicts = parse_verdicts(content, ["First.", "Second."])

    assert [v.sentence for v in verdicts] == ["First.", "Second."]
    assert [v.supported for v in verdicts] == [True, False]
    assert verdicts[1].reason == "invented"


def test_json_wrapped_in_prose_or_a_fence_is_still_read():
    content = (
        "Sure! Here you go:\n```json\n"
        '{"verdicts":[{"n":1,"supported":true,"reason":"x"}]}'
        "\n```"
    )

    assert parse_verdicts(content, ["One."])[0].supported is True


def test_a_verdict_count_mismatch_refuses_to_guess():
    """Padding or truncating would misalign every verdict after the gap and
    still return something that looks like a result."""
    content = '{"verdicts":[{"n":1,"supported":true,"reason":"x"}]}'

    with pytest.raises(JudgeError, match="Refusing to align them by guessing"):
        parse_verdicts(content, ["One.", "Two."])


def test_a_response_with_no_json_raises():
    with pytest.raises(JudgeError, match="no JSON object"):
        parse_verdicts("I think it is fine.", ["One."])


def test_malformed_json_raises_rather_than_defaulting_to_unsupported():
    with pytest.raises(JudgeError, match="not valid JSON"):
        parse_verdicts('{"verdicts": [oops]}', ["One."])


def test_a_missing_supported_key_reads_as_unsupported_not_as_an_error():
    # A verdict that omits the flag has not asserted support.
    verdicts = parse_verdicts('{"verdicts":[{"n":1,"reason":"unsure"}]}', ["One."])

    assert verdicts[0].supported is False


def test_judging_an_empty_answer_makes_no_call_at_all():
    client = StubGroq()

    verdicts, prompt, completion = judge_answer(client, "model", "q", "   ", [("c", "body")])

    assert verdicts == []
    assert (prompt, completion) == (0, 0)
    assert client.calls == []


# --- the grounded rule ----------------------------------------------------


def record(*supported: bool) -> AnswerRecord:
    return AnswerRecord(
        qid="q001",
        lane_id="hybrid_rerank",
        question="why?",
        answer="an answer",
        chunk_ids=["c1"],
        verdicts=[
            SentenceVerdict(n=i, sentence=f"s{i}", supported=flag, reason="")
            for i, flag in enumerate(supported, start=1)
        ],
    )


def test_an_answer_is_grounded_only_if_every_sentence_is():
    assert record(True, True, True).grounded
    assert not record(True, True, False).grounded
    assert not record(False).grounded


def test_an_answer_with_no_sentences_is_not_grounded():
    """`all([])` is True, which would score a generator that returned nothing 1.0."""
    assert not record().grounded


def test_token_counts_are_the_sum_of_the_two_calls():
    one = AnswerRecord(
        qid="q",
        lane_id="l",
        question="q",
        answer="a",
        chunk_ids=[],
        generate_prompt_tokens=10,
        generate_completion_tokens=3,
        judge_prompt_tokens=20,
        judge_completion_tokens=7,
    )

    assert one.prompt_tokens == 30
    assert one.completion_tokens == 10


# --- reporting ------------------------------------------------------------


def test_judge_failures_are_reported_separately_and_never_counted_as_ungrounded():
    """Folding an outage into the rate makes a broken judge look like a bad lane."""
    summary = summarise(
        [record(True), record(True), record(False)],
        failures=[{"qid": "q004", "lane_id": "hybrid_rerank", "error": "empty content"}],
    )
    lane = summary["lanes"]["hybrid_rerank"]

    assert lane["answers_scored"] == 3
    assert lane["grounded"] == 2
    assert lane["groundedness_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert len(summary["failures"]) == 1


def test_the_summary_records_the_settings_that_produced_it():
    summary = summarise([record(True)], failures=[])

    assert summary["reasoning_effort"] == "low"
    assert summary["temperature"] == 0.0
    assert summary["max_tokens"] == {
        "generate": GENERATE_MAX_TOKENS,
        "judge": JUDGE_MAX_TOKENS,
    }


def test_context_passages_are_numbered_for_the_model_and_the_human_alike():
    rendered = format_context([("c1", "first body"), ("c2", "second body")])

    assert rendered == "[1] first body\n\n[2] second body"


# --- the daily budget -----------------------------------------------------


class StubRateLimitError(Exception):
    """Shaped like groq's RateLimitError, which is matched by class name."""


StubRateLimitError.__name__ = "RateLimitError"


def test_the_daily_token_budget_stops_the_run_instead_of_being_retried():
    """Per-minute and per-day arrive as the same 429. Retrying the second one
    burns the rest of the run against a wall that refills at limit/1440 a minute."""
    client = StubGroq(
        StubRateLimitError("Rate limit reached ... on tokens per day (TPD): Limit 200000")
    )

    with pytest.raises(DailyBudgetExhausted):
        complete(client, "model", "sys", "user", max_tokens=100)

    assert len(client.calls) == 1  # not retried


def test_a_per_minute_limit_is_still_retried():
    client = StubGroq(
        StubRateLimitError("Rate limit reached ... requests per minute"),
        StubResponse("recovered"),
    )

    content, _, _ = complete(client, "model", "sys", "user", max_tokens=100)

    assert content == "recovered"
    assert len(client.calls) == 2


def test_the_daily_budget_error_is_not_a_judge_error():
    """A JudgeError is a fact about one answer and gets recorded against it. This
    is a fact about the account, and recording it per-answer would file 32
    identical rows describing a quota as if it were a grounding problem."""
    assert not issubclass(DailyBudgetExhausted, JudgeError)


# --- resumability ---------------------------------------------------------


def test_already_done_reads_back_lane_and_question_pairs(tmp_path):
    from backend.judge import already_done

    path = tmp_path / "verdicts.jsonl"
    path.write_text(
        '{"lane_id":"hybrid_rerank","qid":"q001"}\n{"lane_id":"hyde","qid":"q001"}\n',
        encoding="utf-8",
    )

    assert already_done(path) == {("hybrid_rerank", "q001"), ("hyde", "q001")}


def test_already_done_on_a_missing_file_is_empty_not_an_error(tmp_path):
    from backend.judge import already_done

    assert already_done(tmp_path / "absent.jsonl") == set()


def test_the_summary_can_be_built_from_the_whole_file_not_just_this_run():
    """A resumed run has to report over everything judged so far, or the
    groundedness rate would describe only the last few minutes of work."""
    rows = [record(True).as_dict(), record(False).as_dict()]

    summary = summarise(rows, failures=[])

    assert summary["lanes"]["hybrid_rerank"]["answers_scored"] == 2
    assert summary["lanes"]["hybrid_rerank"]["groundedness_rate"] == 0.5
