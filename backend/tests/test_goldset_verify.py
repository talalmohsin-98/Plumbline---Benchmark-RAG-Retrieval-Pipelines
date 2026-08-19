"""Verification CLI tests. The keypress reader and the store are stubbed."""

import json

import pytest

from backend.goldset import verify
from backend.goldset.verify import (
    DROP_RULES,
    Progress,
    gold_row,
    render,
    restore_progress,
    run,
    sort_drafts,
)


def draft(qid, label_check="confirmed", chunk_id=None, question=None):
    return {
        "qid": qid,
        "question": question or f"Question for {qid}?",
        "gold_chunk_ids": [chunk_id or f"chunk_{qid}"],
        "source_doc": "fastapi/index.md",
        "query_type": "factual",
        "status": "draft",
        "label_check": label_check,
        "drafted_from": chunk_id or f"chunk_{qid}",
    }


def chunks_for(drafts):
    return {d["gold_chunk_ids"][0]: f"body of {d['qid']}" for d in drafts}


def read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def keys(monkeypatch):
    """Feed a scripted sequence of keypresses to the loop."""
    pressed = []

    def press(sequence):
        pressed.extend(sequence)

    def fake_read_key():
        return pressed.pop(0)

    monkeypatch.setattr(verify, "read_key", fake_read_key)
    monkeypatch.setattr(verify, "clear_screen", lambda: None)
    return press


# --- ordering -----------------------------------------------------------


def test_confirmed_labels_are_offered_before_relabelled_ones():
    drafts = [
        draft("q003", "relabelled"),
        draft("q001", "confirmed"),
        draft("q002", "relabelled"),
        draft("q004", "confirmed"),
    ]
    assert [d["qid"] for d in sort_drafts(drafts)] == ["q001", "q004", "q002", "q003"]


def test_rows_with_an_unknown_label_check_sort_last():
    drafts = [draft("q002", None), draft("q001", "confirmed")]
    assert [d["qid"] for d in sort_drafts(drafts)] == ["q001", "q002"]


# --- decisions hit disk immediately -------------------------------------


def test_each_decision_is_written_before_the_next_question(tmp_path, keys):
    """The requirement: closing the terminal at question 80 loses nothing."""
    drafts = [draft(f"q{i:03d}") for i in range(1, 4)]
    gold, decisions = tmp_path / "gold.jsonl", tmp_path / "dec.jsonl"
    keys(["k", "d", "k"])

    run(drafts, chunks_for(drafts), gold, decisions)

    assert [r["qid"] for r in read(gold)] == ["q001", "q003"]
    assert [r["decision"] for r in read(decisions)] == ["keep", "drop", "keep"]


def test_a_dropped_draft_never_enters_the_gold_set(tmp_path, keys):
    drafts = [draft("q001")]
    gold, decisions = tmp_path / "gold.jsonl", tmp_path / "dec.jsonl"
    keys(["d"])

    run(drafts, chunks_for(drafts), gold, decisions)

    assert read(gold) == []
    assert read(decisions)[0]["decision"] == "drop"


def test_quitting_stops_without_recording_a_decision(tmp_path, keys):
    drafts = [draft("q001"), draft("q002")]
    gold, decisions = tmp_path / "gold.jsonl", tmp_path / "dec.jsonl"
    keys(["k", "q"])

    progress = run(drafts, chunks_for(drafts), gold, decisions)

    assert [r["qid"] for r in read(decisions)] == ["q001"]
    assert progress.seen == 1


def test_unrecognised_keys_are_ignored(tmp_path, keys):
    drafts = [draft("q001")]
    gold, decisions = tmp_path / "gold.jsonl", tmp_path / "dec.jsonl"
    keys(["x", "?", "\r", "k"])

    run(drafts, chunks_for(drafts), gold, decisions)
    assert read(decisions)[0]["decision"] == "keep"


# --- fix ----------------------------------------------------------------


def test_fix_records_the_rewritten_question(tmp_path, keys, monkeypatch):
    drafts = [draft("q001", question="Vague question?")]
    gold, decisions = tmp_path / "gold.jsonl", tmp_path / "dec.jsonl"
    keys(["f"])
    monkeypatch.setattr("builtins.input", lambda _="": "  A sharper question?  ")

    run(drafts, chunks_for(drafts), gold, decisions)

    row = read(gold)[0]
    assert row["question"] == "A sharper question?"
    assert row["decision"] == "fix"


def test_an_empty_rewrite_keeps_the_original_question(tmp_path, keys, monkeypatch):
    drafts = [draft("q001", question="Original?")]
    gold, decisions = tmp_path / "gold.jsonl", tmp_path / "dec.jsonl"
    keys(["f"])
    monkeypatch.setattr("builtins.input", lambda _="": "")

    run(drafts, chunks_for(drafts), gold, decisions)
    assert read(gold)[0]["question"] == "Original?"


# --- resume -------------------------------------------------------------


def test_already_decided_drafts_are_skipped_on_resume(tmp_path, keys):
    drafts = [draft(f"q{i:03d}") for i in range(1, 4)]
    gold, decisions = tmp_path / "gold.jsonl", tmp_path / "dec.jsonl"

    keys(["k", "q"])
    run(drafts, chunks_for(drafts), gold, decisions)

    keys(["d", "k"])  # only q002 and q003 remain
    progress = run(drafts, chunks_for(drafts), gold, decisions)

    assert [r["qid"] for r in read(decisions)] == ["q001", "q002", "q003"]
    assert progress.seen == 3
    assert progress.kept == 2
    assert progress.dropped == 1


def test_counts_are_restored_from_the_decision_log():
    decisions = [
        {"qid": "q001", "decision": "keep"},
        {"qid": "q002", "decision": "drop"},
        {"qid": "q003", "decision": "fix"},
    ]
    progress = restore_progress(decisions, total=10)
    assert (progress.kept, progress.dropped, progress.fixed) == (1, 1, 1)
    assert progress.seen == 3
    assert progress.keeps == 2  # kept + fixed both enter the gold set


# --- target -------------------------------------------------------------


def test_target_reached_offers_to_stop_and_stops(tmp_path, keys):
    drafts = [draft(f"q{i:03d}") for i in range(1, 6)]
    gold, decisions = tmp_path / "gold.jsonl", tmp_path / "dec.jsonl"
    # keep, keep, then 'q' at the target prompt.
    keys(["k", "k", "q"])

    progress = run(drafts, chunks_for(drafts), gold, decisions, target=2)

    assert progress.keeps == 2
    assert progress.seen == 2  # stopped rather than continuing to q003


def test_declining_to_stop_continues_verifying(tmp_path, keys):
    drafts = [draft(f"q{i:03d}") for i in range(1, 4)]
    gold, decisions = tmp_path / "gold.jsonl", tmp_path / "dec.jsonl"
    keys(["k", "k", "n", "d"])  # 'n' declines the offer to stop

    progress = run(drafts, chunks_for(drafts), gold, decisions, target=2)

    assert progress.seen == 3
    assert progress.dropped == 1


def test_the_target_offer_is_made_only_once(tmp_path, keys):
    drafts = [draft(f"q{i:03d}") for i in range(1, 5)]
    gold, decisions = tmp_path / "gold.jsonl", tmp_path / "dec.jsonl"
    keys(["k", "k", "n", "k", "k"])  # no second offer after q003's keep

    progress = run(drafts, chunks_for(drafts), gold, decisions, target=2)
    assert progress.seen == 4


# --- the screen ---------------------------------------------------------


def test_every_drop_rule_is_on_screen():
    drafts = [draft("q001")]
    screen = render(drafts[0], "the chunk body", Progress(total=1))
    for rule in DROP_RULES:
        assert rule in screen


def test_the_screen_shows_the_question_chunk_and_progress():
    progress = Progress(total=350, kept=22, fixed=4, dropped=8)
    progress.decided = {f"q{i}" for i in range(34)}
    screen = render(draft("q034"), "the chunk body", progress)

    assert "Question for q034?" in screen
    assert "the chunk body" in screen
    assert "[34/350] · 22 kept · 8 dropped · 4 fixed" in screen


def test_a_relabelled_draft_says_so_on_screen():
    row = draft("q001", "relabelled")
    row["drafted_from"] = "chunk_previous"
    assert "relabelled from chunk_previous" in render(row, "body", Progress(total=1))


def test_a_missing_chunk_is_reported_rather_than_crashing(tmp_path, keys):
    drafts = [draft("q001")]
    gold, decisions = tmp_path / "gold.jsonl", tmp_path / "dec.jsonl"
    keys(["d"])
    run(drafts, {}, gold, decisions)  # empty chunk store
    assert read(decisions)[0]["decision"] == "drop"


# --- the written row ----------------------------------------------------


def test_gold_rows_carry_the_spec_schema():
    row = gold_row(draft("q001", chunk_id="nadra_ch_042"), "A question?", "keep")
    assert row["qid"] == "q001"
    assert row["question"] == "A question?"
    assert row["gold_chunk_ids"] == ["nadra_ch_042"]
    assert row["source_doc"] == "fastapi/index.md"
    assert row["query_type"] == "factual"
    assert row["status"] == "verified"
    assert row["decision"] == "keep"
