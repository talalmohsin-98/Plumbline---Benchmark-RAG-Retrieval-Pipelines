"""Assembly tests: selection order, the shortfall guard, and the row schema."""

import json

import pytest

from backend.goldset import assemble
from backend.goldset.assemble import GOLD_STATUSES, gold_row, main, select


@pytest.fixture(autouse=True)
def isolate_inputs(tmp_path, monkeypatch):
    """Keep the repository's real audit and second-opinion files out of these tests.

    `main` defaults to reading them, and they exist in the working tree, so
    without this every fixture row here would be adjudicated against real
    verdicts for qids it does not share and excluded.
    """
    absent = tmp_path / "absent"
    monkeypatch.setattr(assemble, "DEFAULT_AUDIT", absent / "audit.jsonl")
    monkeypatch.setattr(assemble, "DEFAULT_SECOND", absent / "second.jsonl")
    monkeypatch.setattr(assemble, "DEFAULT_MULTILABEL", absent / "multilabel.json")
    monkeypatch.setattr(assemble, "DEFAULT_EXCLUSIONS", absent / "exclusions.json")


def screened(qid, verdict="keep", label_check="confirmed", **extra):
    row = {
        "qid": qid,
        "question": f"Question for {qid}?",
        "gold_chunk_ids": [f"chunk_{qid}"],
        "source_doc": "fastapi/index.md",
        "query_type": "factual",
        "label_check": label_check,
        "screen_verdict": verdict,
        "screen_rule": None,
        "screen_reason": "no rule fired",
    }
    row.update(extra)
    return row


def write(path, rows):
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --- selection ----------------------------------------------------------


def test_only_keeps_and_fixes_are_selected():
    rows = [
        screened("q001", "keep"),
        screened("q002", "drop"),
        screened("q003", "fix"),
        screened("q004", "unscored"),
    ]
    assert [r["qid"] for r in select(rows)] == ["q001", "q003"]


def test_confirmed_labels_come_before_relabelled_ones():
    rows = [
        screened("q001", label_check="relabelled"),
        screened("q002", label_check="confirmed"),
        screened("q003", label_check="relabelled"),
        screened("q004", label_check="confirmed"),
    ]
    assert [r["qid"] for r in select(rows)] == ["q002", "q004", "q001", "q003"]


def test_an_unknown_label_check_sorts_last():
    rows = [screened("q001", label_check=None), screened("q002", label_check="relabelled")]
    assert [r["qid"] for r in select(rows)] == ["q002", "q001"]


def test_selection_is_uncapped_by_default():
    """The gold set is closed; truncating would discard adjudicated rows."""
    rows = [screened(f"q{i:03d}") for i in range(1, 11)]
    assert len(select(rows)) == 10


def test_an_explicit_cap_still_truncates():
    rows = [screened(f"q{i:03d}") for i in range(1, 11)]
    assert len(select(rows, cap=4)) == 4


def test_a_cap_is_taken_from_the_preferred_end():
    rows = [screened(f"q{i:03d}", label_check="relabelled") for i in range(1, 5)]
    rows += [screened("q009", label_check="confirmed")]
    assert [r["qid"] for r in select(rows, cap=2)] == ["q009", "q001"]


# --- the shortfall guard ------------------------------------------------


def test_a_shortfall_writes_nothing_and_fails(tmp_path, capsys):
    src, out = tmp_path / "screened.jsonl", tmp_path / "gold.jsonl"
    write(src, [screened("q001"), screened("q002"), screened("q003", "drop")])

    code = main(["--screened", str(src), "--out", str(out), "--minimum", "5"])

    assert code == 1
    assert not out.exists()  # no partial gold set left behind
    assert "SHORTFALL" in capsys.readouterr().out


def test_drops_never_pad_a_shortfall(tmp_path):
    """The whole point of the guard: 120 rows where 3 are known-bad is worse
    than 117 rows that are not."""
    src, out = tmp_path / "screened.jsonl", tmp_path / "gold.jsonl"
    write(src, [screened("q001")] + [screened(f"q{i:03d}", "drop") for i in range(2, 20)])

    assert main(["--screened", str(src), "--out", str(out), "--minimum", "5"]) == 1
    assert not out.exists()


def test_exactly_enough_rows_succeeds(tmp_path):
    src, out = tmp_path / "screened.jsonl", tmp_path / "gold.jsonl"
    write(src, [screened(f"q{i:03d}") for i in range(1, 6)])

    assert main(["--screened", str(src), "--out", str(out), "--minimum", "5"]) == 0
    assert len(read(out)) == 5


# --- the written row ----------------------------------------------------


def test_a_gold_row_is_never_marked_verified():
    """No human read this row. The file must say so, not just the README."""
    row = gold_row(screened("q001"))
    assert row["status"] == "screened"
    assert row["status"] in GOLD_STATUSES
    assert "verified" not in json.dumps(row)


def test_a_gold_row_carries_the_spec_schema():
    row = gold_row(screened("q001"))
    assert row["qid"] == "q001"
    assert row["question"] == "Question for q001?"
    assert row["gold_chunk_ids"] == ["chunk_q001"]
    assert row["source_doc"] == "fastapi/index.md"
    assert row["query_type"] == "factual"


def test_a_fixed_row_ships_the_rephrasing_and_records_the_original():
    row = gold_row(
        screened(
            "q001",
            "fix",
            question="Which header stops a proxy buffering a stream?",
            original_question="What prevents buffering in some proxies like Nginx?",
        )
    )
    assert row["question"] == "Which header stops a proxy buffering a stream?"
    assert row["original_question"] == "What prevents buffering in some proxies like Nginx?"
    assert row["decision"] == "fix"


def test_the_screener_reasoning_survives_into_the_gold_set(tmp_path):
    src, out = tmp_path / "screened.jsonl", tmp_path / "gold.jsonl"
    write(src, [screened("q001", screen_rule=None, screen_reason="longest shared run is 2 words")])

    main(["--screened", str(src), "--out", str(out), "--minimum", "1"])
    assert read(out)[0]["screen_reason"] == "longest shared run is 2 words"


# --- the provisional escape hatch ---------------------------------------


def test_allow_partial_writes_what_survived_without_padding(tmp_path):
    src, out = tmp_path / "screened.jsonl", tmp_path / "gold.jsonl"
    kept = [screened("q001"), screened("q002")]
    write(src, kept + [screened(f"q{i:03d}", "drop") for i in (3, 4)])

    code = main(["--screened", str(src), "--out", str(out), "--minimum", "5", "--allow-partial"])

    rows = read(out)
    assert code == 0
    assert [r["qid"] for r in rows] == ["q001", "q002"]  # the drops stayed out
    assert all(r["provisional"] is True for r in rows)


def test_a_full_gold_set_is_never_stamped_provisional(tmp_path):
    src, out = tmp_path / "screened.jsonl", tmp_path / "gold.jsonl"
    write(src, [screened(f"q{i:03d}") for i in range(1, 6)])

    main(["--screened", str(src), "--out", str(out), "--minimum", "5", "--allow-partial"])
    assert all("provisional" not in r for r in read(out))


def test_the_provisional_stamp_does_not_survive_a_later_full_run(tmp_path):
    """The file is rewritten from scratch, so the flag cannot linger."""
    src, out = tmp_path / "screened.jsonl", tmp_path / "gold.jsonl"

    write(src, [screened(f"q{i:03d}") for i in range(1, 4)])
    main(["--screened", str(src), "--out", str(out), "--minimum", "5", "--allow-partial"])
    assert all(r.get("provisional") for r in read(out))

    write(src, [screened(f"q{i:03d}") for i in range(1, 6)])
    main(["--screened", str(src), "--out", str(out), "--minimum", "5"])
    assert all("provisional" not in r for r in read(out))


def test_the_shortfall_still_stops_by_default(tmp_path):
    src, out = tmp_path / "screened.jsonl", tmp_path / "gold.jsonl"
    write(src, [screened("q001")])

    assert main(["--screened", str(src), "--out", str(out), "--minimum", "5"]) == 1
    assert not out.exists()


# --- adjudication: the audit overrides both screeners --------------------


def audit_row(qid, verdict):
    return {"qid": qid, "human_verdict": verdict}


def opinion(qid, verdict):
    return {"qid": qid, "screen_verdict": verdict}


def screened_by(qid, verdict, model="openai/gpt-oss-20b"):
    return screened(qid, verdict, screen_model=model)


def test_the_audit_overrides_the_screener():
    from backend.goldset.assemble import adjudicate

    row = screened_by("q001", "keep")
    result = adjudicate(row, {"q001": audit_row("q001", "drop")}, {"q001": opinion("q001", "keep")})
    assert result == ("drop", "audit")


def test_the_audit_overrides_both_screeners_agreeing():
    """Two screeners agreeing is evidence; a human reading the row is authority."""
    from backend.goldset.assemble import adjudicate

    row = screened_by("q001", "drop")
    result = adjudicate(row, {"q001": audit_row("q001", "keep")}, {"q001": opinion("q001", "drop")})
    assert result == ("keep", "audit")


def test_an_unaudited_row_is_kept_when_both_screeners_agree():
    from backend.goldset.assemble import adjudicate

    row = screened_by("q001", "keep")
    assert adjudicate(row, {}, {"q001": opinion("q001", "keep")}) == ("keep", "both-agree")


def test_an_unaudited_row_the_screeners_split_on_is_excluded():
    """Undecided is not the same as keep, and guessing it would be inventing data."""
    from backend.goldset.assemble import adjudicate

    row = screened_by("q001", "keep")
    result = adjudicate(row, {}, {"q001": opinion("q001", "drop")})
    assert result == ("excluded", "unaudited-dispute")


def test_a_row_screened_by_a_stand_in_model_is_unvalidated():
    from backend.goldset.assemble import adjudicate

    row = screened_by("q001", "keep", model="llama-3.1-8b-instant")
    result = adjudicate(row, {}, {"q001": opinion("q001", "keep")})
    assert result == ("excluded", "unvalidated")


def test_a_stand_in_row_the_audit_reached_is_still_usable():
    """The audit read it, so which model screened it no longer matters."""
    from backend.goldset.assemble import adjudicate

    row = screened_by("q001", "drop", model="llama-3.1-8b-instant")
    result = adjudicate(row, {"q001": audit_row("q001", "keep")}, {"q001": opinion("q001", "drop")})
    assert result == ("keep", "audit")


def test_with_no_second_opinion_the_screener_stands():
    from backend.goldset.assemble import adjudicate

    assert adjudicate(screened_by("q001", "keep"), {}, {}) == ("keep", "screener-only")


# --- multi-labelling -----------------------------------------------------


MULTI = {"q001": {"also": ["chunk_b", "chunk_c"], "why": "same answer sentence"}}


def test_equally_correct_chunks_are_added_to_the_gold_ids():
    from backend.goldset.assemble import apply_multilabel

    row = apply_multilabel(screened("q001"), MULTI)
    assert row["gold_chunk_ids"] == ["chunk_q001", "chunk_b", "chunk_c"]
    assert row["multilabel_reason"] == "same answer sentence"


def test_the_original_gold_chunk_stays_first():
    """The chunk the question was written from remains the primary label."""
    from backend.goldset.assemble import apply_multilabel

    assert apply_multilabel(screened("q001"), MULTI)["gold_chunk_ids"][0] == "chunk_q001"


def test_a_row_with_no_duplicates_is_untouched():
    from backend.goldset.assemble import apply_multilabel

    row = screened("q999")
    assert apply_multilabel(row, MULTI) == row


def test_a_chunk_already_listed_is_not_duplicated():
    from backend.goldset.assemble import apply_multilabel

    row = screened("q001")
    row["gold_chunk_ids"] = ["chunk_q001", "chunk_b"]
    assert apply_multilabel(row, MULTI)["gold_chunk_ids"] == ["chunk_q001", "chunk_b", "chunk_c"]


def test_the_readme_key_is_not_read_as_a_row(tmp_path):
    from backend.goldset.assemble import load_hand_map

    path = tmp_path / "multilabel.json"
    path.write_text(json.dumps({"_README": ["notes"], "q001": MULTI["q001"]}), encoding="utf-8")
    assert list(load_hand_map(path)) == ["q001"]


# --- hand exclusions -----------------------------------------------------


EXCLUDED = {"q001": {"why": "drafted from a chunk the scaffolding filter now rejects"}}


def test_a_hand_excluded_row_is_excluded():
    from backend.goldset.assemble import adjudicate

    row = screened_by("q001", "keep")
    result = adjudicate(row, {}, {"q001": opinion("q001", "keep")}, exclusions=EXCLUDED)
    assert result == ("excluded", "hand-excluded")


def test_a_hand_exclusion_outranks_the_audit():
    """The audit judged the question. This says the row cannot be produced again."""
    from backend.goldset.assemble import adjudicate

    row = screened_by("q001", "keep")
    result = adjudicate(
        row, {"q001": audit_row("q001", "keep")}, {"q001": opinion("q001", "keep")},
        exclusions=EXCLUDED,
    )
    assert result == ("excluded", "hand-excluded")


def test_rows_not_listed_are_unaffected_by_the_exclusion_map():
    from backend.goldset.assemble import adjudicate

    row = screened_by("q002", "keep")
    result = adjudicate(row, {}, {"q002": opinion("q002", "keep")}, exclusions=EXCLUDED)
    assert result == ("keep", "both-agree")


def test_an_excluded_row_does_not_reach_the_gold_set(tmp_path):
    """End-to-end: the file assemble writes must not contain the excluded qid."""
    src, out = tmp_path / "screened.jsonl", tmp_path / "gold.jsonl"
    excl = tmp_path / "exclusions.json"
    write(src, [screened_by("q001", "keep"), screened_by("q002", "keep")])
    excl.write_text(json.dumps({"_README": ["notes"], **EXCLUDED}), encoding="utf-8")

    code = main(
        ["--screened", str(src), "--out", str(out), "--minimum", "2",
         "--exclusions", str(excl), "--allow-partial"]
    )

    assert code == 0
    assert [r["qid"] for r in read(out)] == ["q002"]


# --- the rejection census ------------------------------------------------


def settled(qid, verdict, source):
    return {"qid": qid, "screen_verdict": verdict, "verdict_source": source}


def test_the_census_counts_every_screened_row_once():
    from backend.goldset.assemble import census

    rows = [
        settled("q001", "keep", "audit"),
        settled("q002", "fix", "audit"),
        settled("q003", "drop", "audit"),
        settled("q004", "excluded", "hand-excluded"),
        settled("q005", "excluded", "unvalidated"),
    ]
    counts = census(rows)

    assert counts["screened"] == 5
    assert counts["accepted"] == 2
    assert counts["rejected_on_quality"] == 1
    assert counts["excluded"] == 2
    assert counts["accepted"] + counts["rejected_on_quality"] + counts["excluded"] == 5


def test_a_row_removed_for_irreproducibility_is_not_a_quality_rejection():
    """Counting it as one would inflate the published rejection rate."""
    from backend.goldset.assemble import census

    counts = census([settled("q001", "excluded", "hand-excluded")])
    assert counts["rejected_on_quality"] == 0
    assert counts["excluded"] == 1


def test_a_rejection_no_human_read_is_counted_where_it_can_be_seen():
    """A false drop is invisible in the finished gold set by construction."""
    from backend.goldset.assemble import census

    rows = [settled("q001", "drop", "both-agree"), settled("q002", "drop", "audit")]
    counts = census(rows)

    assert counts["rejected_on_quality"] == 2
    assert counts["rejected_unreviewed"] == 1
