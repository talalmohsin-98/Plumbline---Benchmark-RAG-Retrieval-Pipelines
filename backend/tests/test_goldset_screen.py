"""Screening tests. The Groq client is stubbed; no call leaves the process."""

import json
from pathlib import Path

import pytest

from backend.goldset.rules import GENERIC, MULTI_CHUNK, UNANSWERABLE, VERBATIM, verdict_for
from backend.goldset.screen import (
    MODEL_SCORED,
    REPHRASE_ATTEMPTS,
    VERBATIM_RUN,
    Judgement,
    longest_overlap,
    parse_judgement,
    rule_counts,
    run,
    screen_draft,
    screened_row,
)

PASSAGE = (
    "Set a special header X-Accel-Buffering with the value no to prevent "
    "buffering in some proxies like Nginx. To use forms, first install the "
    "python-multipart package."
)


# --- rule 2, computed by hand -------------------------------------------


def test_no_shared_words_is_a_run_of_zero():
    assert longest_overlap("Which port does it bind?", "totally unrelated prose").length == 0


def test_the_longest_run_is_counted_not_the_first():
    # "install the" is 2; "prevent buffering in some proxies" is 5.
    question = "How do you install the thing that prevent buffering in some proxies?"
    overlap = longest_overlap(question, PASSAGE)
    assert overlap.length == 5
    assert overlap.phrase == "prevent buffering in some proxies"


def test_case_and_punctuation_do_not_break_a_run():
    # Five words, differing only in case and a hyphen the tokenizer drops.
    assert longest_overlap("PREVENT BUFFERING, IN SOME PROXIES!", PASSAGE).length == 5


def test_a_run_must_be_consecutive():
    # Every word is present in the passage; none of them adjacently.
    assert longest_overlap("header install proxies value", PASSAGE).length == 1


def test_word_boundaries_are_respected():
    """"buffer" is not "buffering": the scan is over words, not characters."""
    assert longest_overlap("prevent buffer", PASSAGE).length == 1


def test_an_empty_question_has_no_overlap():
    assert longest_overlap("", PASSAGE).length == 0


def test_four_consecutive_words_is_within_the_rule():
    """The rule fires above four, so exactly four must not fire."""
    overlap = longest_overlap("What does buffering in some proxies mean?", PASSAGE)
    assert overlap.length == 4
    assert not overlap.length > VERBATIM_RUN


# --- reading the model's reply ------------------------------------------


def test_a_verdict_and_a_reason_on_separate_lines():
    judgement = parse_judgement("YES\nThe passage never states the value.")
    assert judgement == Judgement(True, "The passage never states the value.")


def test_a_verdict_and_a_reason_on_one_line():
    assert parse_judgement("NO - the answer is stated plainly.") == Judgement(
        False, "the answer is stated plainly."
    )


def test_lowercase_and_surrounding_blank_lines_are_tolerated():
    assert parse_judgement("\n\nyes\n\nToo generic.\n").fired is True


def test_an_unreadable_reply_scores_nothing_rather_than_defaulting():
    """A reply that cannot be read must never become a judgement."""
    assert parse_judgement("I think it depends on context.").fired is None
    assert parse_judgement("").fired is None


def test_a_verdict_word_inside_the_reason_is_not_mistaken_for_the_verdict():
    assert parse_judgement("NO, yes it is answerable here.").fired is False


# --- resolving four scores into one verdict -----------------------------


def test_no_rule_fired_is_a_keep():
    assert verdict_for({r.id: False for r in (GENERIC, VERBATIM, UNANSWERABLE, MULTI_CHUNK)}) == (
        "keep",
        None,
    )


def test_verbatim_alone_is_a_fix():
    assert verdict_for({VERBATIM.id: True}) == ("fix", VERBATIM.id)


def test_a_drop_rule_beats_the_fix_rule():
    """Vague and verbatim is not worth rephrasing."""
    assert verdict_for({VERBATIM.id: True, GENERIC.id: True}) == ("drop", GENERIC.id)


def test_the_recorded_rule_is_deterministic_when_two_drop_rules_fire():
    fired = {UNANSWERABLE.id: True, MULTI_CHUNK.id: True, GENERIC.id: True}
    assert verdict_for(fired) == ("drop", GENERIC.id)  # RULES order


# --- screening one draft, with a scripted client ------------------------


class ScriptedClient:
    """A Groq-shaped client that replays queued replies and records prompts."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts = []
        self.chat = self  # client.chat.completions.create
        self.completions = self

    def create(self, *, model, temperature, messages):
        self.prompts.append(messages[-1]["content"])
        if not self._replies:
            raise AssertionError("the client was called more times than the test scripted")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Response(reply)


class _Response:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


def clean(*, generic="NO\nSpecific.", unanswerable="NO\nStated.", multi="NO\nSelf-contained."):
    """Replies for the three model-scored rules, in the order they are asked."""
    return {GENERIC.id: generic, UNANSWERABLE.id: unanswerable, MULTI_CHUNK.id: multi}


def replies_for(scores):
    return [scores[rule_id] for rule_id in MODEL_SCORED]


def test_a_clean_draft_is_kept_and_costs_one_call_per_model_rule():
    client = ScriptedClient(replies_for(clean()))
    result = screen_draft(client, "m", "Which package handles form parsing?", PASSAGE)

    assert result.verdict == "keep"
    assert result.rule is None
    assert len(client.prompts) == len(MODEL_SCORED)


def test_every_rule_is_scored_even_once_a_drop_is_certain():
    """Short-circuiting would hollow out the per-rule counts in the summary."""
    client = ScriptedClient(replies_for(clean(generic="YES\nAny tutorial answers it.")))
    result = screen_draft(client, "m", "What is a web framework?", PASSAGE)

    assert result.verdict == "drop"
    assert result.rule == GENERIC.id
    assert len(client.prompts) == len(MODEL_SCORED)
    assert set(result.scores) == {r.id for r in (GENERIC, VERBATIM, UNANSWERABLE, MULTI_CHUNK)}


def test_the_verbatim_rule_costs_no_call():
    client = ScriptedClient(replies_for(clean()))
    screen_draft(client, "m", "Which package handles form parsing?", PASSAGE)

    joined = " ".join(client.prompts)
    assert "consecutive" not in joined  # rule 2 is never described to the model
    assert client.prompts[0].startswith("A benchmark question is worthless")


def test_an_unscorable_rule_makes_the_row_unscored_not_a_keep():
    client = ScriptedClient(replies_for(clean(multi="maybe, it is hard to say")))
    result = screen_draft(client, "m", "Which package handles form parsing?", PASSAGE)

    assert result.verdict == "unscored"
    assert MULTI_CHUNK.id in result.reason


def test_a_failed_call_makes_the_row_unscored(monkeypatch):
    monkeypatch.setattr("backend.goldset.screen.time.sleep", lambda _s: None)
    # Three attempts per rule; the first rule exhausts all of them.
    client = ScriptedClient([RuntimeError("429")] * 3 + replies_for(clean())[1:])
    result = screen_draft(client, "m", "Which package handles form parsing?", PASSAGE)

    assert result.verdict == "unscored"
    assert "call failed" in result.scores[GENERIC.id]["reason"]


# --- the fix path -------------------------------------------------------

VERBATIM_QUESTION = "What prevents buffering in some proxies like Nginx?"


def test_a_verbatim_question_is_rephrased_and_kept_as_a_fix():
    client = ScriptedClient(
        [*replies_for(clean()), "Which response header stops a reverse proxy holding a stream?"]
    )
    result = screen_draft(client, "m", VERBATIM_QUESTION, PASSAGE)

    assert result.verdict == "fix"
    assert result.rule == VERBATIM.id
    assert result.question != VERBATIM_QUESTION
    assert longest_overlap(result.question, PASSAGE).length <= VERBATIM_RUN
    assert result.rephrase_attempts == 1


def test_the_offending_phrase_is_handed_to_the_rephraser():
    client = ScriptedClient([*replies_for(clean()), "A properly reworded question?"])
    screen_draft(client, "m", VERBATIM_QUESTION, PASSAGE)

    assert 'must not reuse is: "buffering in some proxies like' in client.prompts[-1]


def test_a_second_attempt_is_told_the_phrase_the_first_one_failed_to_shake():
    # The first rewrite escapes the original phrase and lands on a different
    # 5-word run ("install the python multipart package"), so it fires again.
    client = ScriptedClient(
        [
            *replies_for(clean()),
            "Which name do you use to install the python multipart package?",
            "A properly reworded question?",
        ]
    )
    result = screen_draft(client, "m", VERBATIM_QUESTION, PASSAGE)

    assert result.verdict == "fix"
    assert result.rephrase_attempts == 2
    assert 'must not reuse is: "install the python multipart package"' in client.prompts[-1]


def test_a_rephrasing_that_never_lands_is_dropped_not_shipped():
    """A known-verbatim question must not reach the gold set."""
    stubborn = "Which header prevents buffering in some proxies like Nginx?"
    client = ScriptedClient(replies_for(clean()) + [stubborn] * REPHRASE_ATTEMPTS)
    result = screen_draft(client, "m", VERBATIM_QUESTION, PASSAGE)

    assert result.verdict == "drop"
    assert result.rule == VERBATIM.id
    assert result.question == VERBATIM_QUESTION  # the original, unmodified


# --- rows ---------------------------------------------------------------


def draft(qid="q001", question="Which package handles form parsing?", chunk_id="c1"):
    return {
        "qid": qid,
        "question": question,
        "gold_chunk_ids": [chunk_id],
        "source_doc": "fastapi/index.md",
        "label_check": "confirmed",
        "query_type": "factual",
        "status": "draft",
    }


def test_a_kept_row_carries_the_draft_through_untouched():
    client = ScriptedClient(replies_for(clean()))
    row = screened_row(draft(), screen_draft(client, "m", draft()["question"], PASSAGE), "m")

    assert row["qid"] == "q001"
    assert row["source_doc"] == "fastapi/index.md"
    assert row["label_check"] == "confirmed"
    assert row["screen_verdict"] == "keep"
    assert "original_question" not in row


def test_a_fixed_row_carries_the_rephrasing_and_keeps_the_original():
    rewritten = "Which response header stops a reverse proxy holding a stream?"
    client = ScriptedClient([*replies_for(clean()), rewritten])
    result = screen_draft(client, "m", VERBATIM_QUESTION, PASSAGE)
    row = screened_row(draft(question=VERBATIM_QUESTION), result, "m")

    assert row["question"] == rewritten
    assert row["original_question"] == VERBATIM_QUESTION


def test_rule_counts_tallies_each_rule_across_rows():
    rows = [
        {"screen_scores": {GENERIC.id: {"fired": True}, VERBATIM.id: {"fired": False}}},
        {"screen_scores": {GENERIC.id: {"fired": True}, VERBATIM.id: {"fired": True}}},
        {"screen_scores": {GENERIC.id: {"fired": None}}},  # unscored counts as not fired
    ]
    counts = rule_counts(rows)
    assert counts[GENERIC.id] == 2
    assert counts[VERBATIM.id] == 1


# --- the run ------------------------------------------------------------


def read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_each_row_is_written_before_the_next_draft_is_screened(tmp_path):
    drafts = [draft(f"q{i:03d}") for i in (1, 2)]
    client = ScriptedClient(replies_for(clean()) * 2)
    out = tmp_path / "screened.jsonl"

    run(drafts, {"c1": PASSAGE}, out, client, "m")

    assert [r["qid"] for r in read(out)] == ["q001", "q002"]


def test_already_screened_drafts_are_skipped_on_resume(tmp_path):
    drafts = [draft(f"q{i:03d}") for i in (1, 2, 3)]
    out = tmp_path / "screened.jsonl"

    run(drafts[:1], {"c1": PASSAGE}, out, ScriptedClient(replies_for(clean())), "m")
    # Exactly two drafts' worth of replies: a third call would raise.
    tally = run(drafts, {"c1": PASSAGE}, out, ScriptedClient(replies_for(clean()) * 2), "m")

    assert [r["qid"] for r in read(out)] == ["q001", "q002", "q003"]
    assert tally["keep"] == 3  # restored from the file, not just this pass


def test_a_missing_chunk_is_unscored_rather_than_a_default_keep(tmp_path):
    out = tmp_path / "screened.jsonl"
    tally = run([draft()], {}, out, ScriptedClient([]), "m")

    assert tally["unscored"] == 1
    assert read(out)[0]["screen_verdict"] == "unscored"


@pytest.mark.parametrize("rule_id", MODEL_SCORED)
def test_every_model_scored_rule_has_a_prompt(rule_id):
    from backend.goldset.screen import PROMPTS

    assert "{question}" not in PROMPTS[rule_id].format(question="q", chunk_text="c")
