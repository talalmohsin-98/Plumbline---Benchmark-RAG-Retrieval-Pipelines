"""Fine-tune driver tests: the parts that run without a GPU.

`train` and `evaluate_slice` need torch and a downloaded checkpoint, so they are
exercised by actually running the script rather than here. What is tested here is
everything that decides *what the model is trained on* and *what the model card
claims* — the two places a mistake would be invisible in a loss curve.
"""

import json

import pytest

from training.train_reranker import (
    Example,
    build_model_card,
    load_pairs,
    split_by_question,
)


def examples(questions: int = 10, per_question: int = 5) -> list[Example]:
    """`questions` questions, each with 1 positive and `per_question - 1` negatives."""
    out = []
    for q in range(questions):
        qid = f"q{q:03d}"
        out.append(Example(qid=qid, question=f"question {q}", text="gold body", label=1.0))
        for n in range(per_question - 1):
            out.append(Example(qid=qid, question=f"question {q}", text=f"neg {n}", label=0.0))
    return out


# --- the holdout ----------------------------------------------------------


def test_the_holdout_splits_whole_questions_never_pairs():
    """The failure this prevents: a question's positive in train, its negatives in eval.

    The eval slice would then be scoring a question the model had already been
    shown the answer to, and the monitor would read high for the wrong reason.
    """
    train, held = split_by_question(examples(), holdout_fraction=0.2, seed=42)

    assert {e.qid for e in train}.isdisjoint({e.qid for e in held})
    assert len(train) + len(held) == 50
    # every question keeps all five of its pairs on one side
    for qid in {e.qid for e in held}:
        assert sum(1 for e in held if e.qid == qid) == 5


def test_the_holdout_is_the_requested_fraction_of_questions():
    _, held = split_by_question(examples(80), holdout_fraction=0.1, seed=42)

    assert len({e.qid for e in held}) == 8  # 10% of 80


def test_the_holdout_is_deterministic_for_a_seed():
    _, first = split_by_question(examples(), seed=42)
    _, second = split_by_question(examples(), seed=42)

    assert [e.qid for e in first] == [e.qid for e in second]


def test_a_different_seed_holds_out_different_questions():
    _, a = split_by_question(examples(80), seed=42)
    _, b = split_by_question(examples(80), seed=7)

    assert {e.qid for e in a} != {e.qid for e in b}


def test_at_least_one_question_is_always_held_out():
    """A tiny pair file must still produce a monitor rather than an empty slice."""
    _, held = split_by_question(examples(3), holdout_fraction=0.1, seed=42)

    assert len({e.qid for e in held}) == 1


def test_holdout_order_does_not_depend_on_input_order():
    """qids are sorted before shuffling, so file order cannot change the split."""
    forward = examples(20)
    _, a = split_by_question(forward, seed=42)
    _, b = split_by_question(list(reversed(forward)), seed=42)

    assert {e.qid for e in a} == {e.qid for e in b}


# --- loading --------------------------------------------------------------


def test_load_pairs_reads_labels_as_floats(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"qid": "q1", "question": "q", "text": "t", "label": 1, "chunk_id": "c"},
                {"qid": "q1", "question": "q", "text": "u", "label": 0, "chunk_id": "d"},
            )
        ),
        encoding="utf-8",
    )

    loaded = load_pairs(path)

    assert [e.label for e in loaded] == [1.0, 0.0]
    assert all(isinstance(e.label, float) for e in loaded)


def test_load_pairs_refuses_an_empty_file(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_pairs(path)


def test_load_pairs_names_the_command_that_makes_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="mine_negatives"):
        load_pairs(tmp_path / "absent.jsonl")


# --- the model card -------------------------------------------------------


def metrics_fixture() -> dict:
    return {
        "generated_at": "2026-08-21T10:00:00Z",
        "base_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "device": "cuda",
        "hyperparameters": {
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 2e-05,
            "warmup_fraction": 0.1,
            "warmup_steps": 7,
            "max_length": 512,
            "loss": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "scheduler": "linear with warmup",
            "seed": 42,
        },
        "data": {
            "pairs_total": 400,
            "pairs_train": 360,
            "pairs_holdout": 40,
            "questions_train": 72,
            "questions_holdout": 8,
            "holdout_fraction": 0.1,
            "holdout_qids": [],
        },
        "steps": {"per_epoch": 23, "total": 69},
        "per_epoch": [
            {"epoch": 0, "bce_loss": 0.7378, "positive_ranked_first": 0.625},
            {"epoch": 1, "train_loss": 0.61, "bce_loss": 0.69, "positive_ranked_first": 0.75},
        ],
    }


def mining_fixture() -> dict:
    return {
        "questions_mined": 80,
        "parameters": {"mine_depth": 20, "seed": 42},
        "pairs": {"total": 400, "positives": 80, "negatives": 320},
        "negatives_per_positive": {"mean": 4.0, "rows_below_target": 0},
        "positive_in_mined_depth": {"rows": 71, "of": 80, "median_gold_rank": 2.0},
        "test_gold_chunks_mined_as_negatives": {
            "count": 24,
            "of_negatives": 320,
            "fraction": 0.075,
            "distinct_chunks": 16,
            "distinct_test_questions_affected": 15,
            "of_test_questions": 35,
        },
    }


def test_the_card_carries_the_measured_numbers_not_placeholders():
    card = build_model_card(metrics_fixture(), mining_fixture())

    assert "80 training questions → 400 pairs" in card
    assert "71/80 rows (median gold rank 2.0)" in card
    assert "| Epochs | 3 |" in card
    assert "| Warmup | 10% (7 of 69 steps) |" in card
    assert "| 0 | — (stock) | 0.7378 | 0.625 |" in card
    assert "| 1 | 0.61 | 0.69 | 0.75 |" in card
    assert "$" not in card  # every template placeholder was substituted


def test_the_card_states_the_contamination_rather_than_hiding_it():
    """The number that would be easiest to quietly drop, so it is pinned."""
    card = build_model_card(metrics_fixture(), mining_fixture())

    assert "24 of 320 negatives (7.5%)" in card
    assert "15 of 35 test questions" in card
    assert "43% of the questions it is later scored on" in card
    assert "reported rather than filtered" in card.lower()


def test_the_card_says_a_loss_would_be_published():
    card = build_model_card(metrics_fixture(), mining_fixture())

    assert "If this model loses to the stock one, that is the published result." in card
    assert "pre-registered" in card.lower()


def test_the_card_survives_a_missing_mining_report_and_says_so():
    """Degraded, but never silently: a card with no contamination line must say why."""
    card = build_model_card(metrics_fixture(), None)

    assert "mining report was not supplied" in card
    assert "$" not in card
    assert "| Epochs | 3 |" in card


def test_the_card_reports_the_epoch_zero_baseline():
    """Without the stock row, a run that made the model worse looks like one that did not."""
    card = build_model_card(metrics_fixture(), mining_fixture())

    assert "(stock)" in card
