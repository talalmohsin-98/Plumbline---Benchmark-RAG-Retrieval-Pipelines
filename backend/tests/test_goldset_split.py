"""Split tests. Leakage between train and test invalidates every number."""

import json

import pytest

from backend.goldset.split import (
    LeakageError,
    Split,
    assert_no_leakage,
    load_goldset,
    split_rows,
)


def rows(n, start=1):
    return [
        {
            "qid": f"q{i:03d}",
            "question": f"Question {i}?",
            "gold_chunk_ids": [f"chunk_{i}"],
            "source_doc": "fastapi/index.md",
            "status": "verified",
        }
        for i in range(start, start + n)
    ]


def test_split_is_seventy_thirty():
    split = split_rows(rows(120))
    assert (len(split.train), len(split.test)) == (84, 36)
    assert split.total == 120


def test_split_is_deterministic_for_a_given_seed():
    first = split_rows(rows(120), seed=42)
    second = split_rows(rows(120), seed=42)
    assert [r["qid"] for r in first.train] == [r["qid"] for r in second.train]
    assert [r["qid"] for r in first.test] == [r["qid"] for r in second.test]


def test_split_does_not_depend_on_input_order():
    forward = split_rows(rows(120), seed=42)
    backward = split_rows(list(reversed(rows(120))), seed=42)
    assert [r["qid"] for r in forward.test] == [r["qid"] for r in backward.test]


def test_a_different_seed_gives_a_different_split():
    assert [r["qid"] for r in split_rows(rows(120), seed=42).test] != [
        r["qid"] for r in split_rows(rows(120), seed=7).test
    ]


def test_no_qid_is_in_both_splits():
    split = split_rows(rows(120))
    assert {r["qid"] for r in split.train} & {r["qid"] for r in split.test} == set()


def test_every_row_lands_in_exactly_one_split():
    source = rows(137)
    split = split_rows(source)
    assert {r["qid"] for r in split.train} | {r["qid"] for r in split.test} == {
        r["qid"] for r in source
    }
    assert split.total == 137


def test_leakage_is_detected_and_named():
    leaked = rows(3)
    with pytest.raises(LeakageError, match="q001"):
        assert_no_leakage(Split(train=leaked, test=leaked[:1]))


def test_a_duplicate_within_a_split_is_leakage():
    duplicated = rows(2) + rows(1)
    with pytest.raises(LeakageError, match="duplicate"):
        assert_no_leakage(Split(train=duplicated, test=[]))


@pytest.mark.parametrize("fraction", [0, 1, -0.5, 1.5])
def test_an_impossible_fraction_raises(fraction):
    with pytest.raises(ValueError):
        split_rows(rows(10), train_fraction=fraction)


def test_only_verified_rows_are_loaded(tmp_path):
    """A draft row must never become training data."""
    path = tmp_path / "goldset.jsonl"
    mixed = [*rows(2), {"qid": "q900", "status": "draft", "question": "not verified"}]
    path.write_text(
        "\n".join(json.dumps(r) for r in mixed) + "\n",
        encoding="utf-8",
    )
    loaded = load_goldset(path)
    assert [r["qid"] for r in loaded] == ["q001", "q002"]


def test_a_missing_goldset_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_goldset(tmp_path / "nope.jsonl")


def test_a_provisional_gold_set_is_announced_by_the_splitter(tmp_path, capsys):
    """It splits perfectly well and means nothing yet. Saying so is the point."""
    from backend.goldset.split import main

    gold = tmp_path / "gold.jsonl"
    rows = [
        {"qid": f"q{i:03d}", "question": "q?", "gold_chunk_ids": ["c"], "status": "screened",
         "provisional": True}
        for i in range(1, 11)
    ]
    gold.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    code = main([
        "--goldset", str(gold),
        "--train", str(tmp_path / "train.jsonl"),
        "--test", str(tmp_path / "test.jsonl"),
    ])

    assert code == 0
    assert "PROVISIONAL: 10 of 10" in capsys.readouterr().out


def test_a_finished_gold_set_says_nothing_about_provisional(tmp_path, capsys):
    from backend.goldset.split import main

    gold = tmp_path / "gold.jsonl"
    rows = [
        {"qid": f"q{i:03d}", "question": "q?", "gold_chunk_ids": ["c"], "status": "screened"}
        for i in range(1, 11)
    ]
    gold.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    main([
        "--goldset", str(gold),
        "--train", str(tmp_path / "train.jsonl"),
        "--test", str(tmp_path / "test.jsonl"),
    ])
    assert "PROVISIONAL" not in capsys.readouterr().out


# --- grouped split: no gold chunk on both sides -------------------------


def gold(qid, *chunk_ids, status="screened"):
    return {
        "qid": qid,
        "question": f"Question for {qid}?",
        "gold_chunk_ids": list(chunk_ids) or [f"chunk_{qid}"],
        "source_doc": "fastapi/index.md",
        "status": status,
    }


def test_rows_sharing_a_gold_chunk_land_on_the_same_side():
    """q023 and q056 both answer 'which package'; four chunks are gold for both."""
    from backend.goldset.split import split_rows

    rows = [gold(f"q{i:03d}") for i in range(1, 21)]
    rows += [gold("q090", "shared_chunk"), gold("q091", "shared_chunk")]

    split = split_rows(rows, seed=42)
    sides = {r["qid"]: "train" for r in split.train}
    sides.update({r["qid"]: "test" for r in split.test})
    assert sides["q090"] == sides["q091"]


def test_a_transitive_group_travels_together():
    """A-B share one chunk, B-C share another: all three must move as one."""
    from backend.goldset.split import group_by_shared_chunk

    rows = [gold("q001", "a"), gold("q002", "a", "b"), gold("q003", "b"), gold("q004", "z")]
    groups = {tuple(r["qid"] for r in g) for g in group_by_shared_chunk(rows)}
    assert groups == {("q001", "q002", "q003"), ("q004",)}


def test_the_shared_chunk_check_fires_when_it_should():
    from backend.goldset.split import LeakageError, Split, assert_no_shared_chunks

    split = Split(train=[gold("q001", "shared")], test=[gold("q002", "shared")])
    with pytest.raises(LeakageError, match="both sides"):
        assert_no_shared_chunks(split)


def test_a_shared_chunk_is_not_caught_by_the_qid_check_alone():
    """The two checks are independent; one does not imply the other."""
    from backend.goldset.split import LeakageError, Split, assert_no_leakage

    split = Split(train=[gold("q001", "shared")], test=[gold("q002", "shared")])
    assert {r["qid"] for r in split.train}.isdisjoint({r["qid"] for r in split.test})
    with pytest.raises(LeakageError):
        assert_no_leakage(split)


def test_the_grouped_split_stays_near_the_requested_fraction():
    rows = [gold(f"q{i:03d}") for i in range(1, 101)]
    split = split_rows(rows, train_fraction=0.7, seed=42)
    assert 65 <= len(split.train) <= 75


def test_the_grouped_split_is_reproducible():
    rows = [gold(f"q{i:03d}") for i in range(1, 41)]
    first = split_rows(rows, seed=42)
    again = split_rows(rows, seed=42)
    assert [r["qid"] for r in first.train] == [r["qid"] for r in again.train]
