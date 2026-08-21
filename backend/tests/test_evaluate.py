"""Evaluation-runner tests: the gate, and how a failed query is accounted for.

No lanes and no corpus here -- `LaneRun` is a plain accumulator and
`check_fusion_gate` reads a list of dicts, so both are testable as pure data.
"""

import json

import pytest

from backend.evaluate import GoldRow, LaneRun, check_fusion_gate, load_split


def lane_metric(lane_id, recall_at_10):
    return {"id": lane_id, "recall_at_10": recall_at_10}


def three_lanes(bm25, dense, hybrid):
    return [
        lane_metric("bm25", bm25),
        lane_metric("dense", dense),
        lane_metric("hybrid_rrf", hybrid),
    ]


# --------------------------------------------------------------------------
# The fusion gate
# --------------------------------------------------------------------------


def test_gate_passes_when_fusion_beats_both_arms():
    gate = check_fusion_gate(three_lanes(0.80, 0.75, 0.85))
    assert gate.status == "pass"
    assert gate.blocking is False


def test_gate_ties_when_fusion_only_matches_its_best_arm():
    # The observed Day 2 case: 30/35 each on the test split. Recorded as its
    # own state rather than laundered into a pass, because a reader of
    # results.json has to be able to tell the two apart.
    gate = check_fusion_gate(three_lanes(0.8571, 0.8286, 0.8571))
    assert gate.status == "tie"
    assert gate.blocking is False
    assert "TIE" in gate.detail


def test_gate_fails_when_fusion_loses_to_an_arm():
    # The shape a real fusion bug takes, and the only one that stops the run.
    gate = check_fusion_gate(three_lanes(0.90, 0.75, 0.85))
    assert gate.status == "fail"
    assert gate.blocking is True


def test_gate_fails_even_when_fusion_beats_the_weaker_arm():
    # Beating dense while losing to BM25 is still a failure. A gate that
    # checked only one arm would pass a fusion that had degenerated into
    # returning the dense list.
    gate = check_fusion_gate(three_lanes(0.90, 0.60, 0.70))
    assert gate.status == "fail"


def test_gate_compares_against_the_stronger_arm_whichever_it_is():
    # Symmetric: it must not assume BM25 is the one to beat.
    assert check_fusion_gate(three_lanes(0.60, 0.90, 0.85)).status == "fail"
    assert check_fusion_gate(three_lanes(0.60, 0.90, 0.95)).status == "pass"


def test_gate_is_skipped_rather_than_passed_when_a_lane_is_missing():
    # `--lanes hyde` must not silently report a passing gate it never checked.
    gate = check_fusion_gate([lane_metric("hyde", 0.9)])
    assert gate.checked is False
    assert gate.status == "skipped"
    assert gate.blocking is False


def test_gate_serialises_its_status_for_results_json():
    gate = check_fusion_gate(three_lanes(0.80, 0.75, 0.85))
    assert gate.as_dict() == {"checked": True, "status": "pass", "detail": gate.detail}


# --------------------------------------------------------------------------
# LaneRun accounting
# --------------------------------------------------------------------------


def test_metrics_over_a_clean_run():
    # Two questions, gold found at rank 1 and rank 2.
    #   recall@5 = recall@10 = 1.0
    #   MRR = (1 + 1/2) / 2 = 0.75
    run = LaneRun(lane_id="x", label="X")
    run.retrieved = [["a", "b"], ["b", "a"]]
    run.gold = [["a"], ["a"]]
    run.latencies_ms = [10.0, 20.0]
    run.prompt_tokens = [0, 0]
    run.completion_tokens = [0, 0]
    metrics = run.metrics(0.075, 0.30)
    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr"] == 0.75
    assert metrics["cost_per_query_usd"] == 0.0
    assert metrics["queries_failed"] == 0
    assert metrics["questions"] == 2


def test_a_failed_query_counts_as_a_miss_not_as_an_absent_question():
    # The honest treatment. Scoring a lane over only the questions it survived
    # would flatter exactly the lane most likely to fail -- HyDE, the one with
    # an external dependency.
    run = LaneRun(lane_id="hyde", label="HyDE")
    run.retrieved = [["a"], []]  # second query raised
    run.gold = [["a"], ["b"]]
    run.latencies_ms = [10.0]  # only the successful query has a latency
    run.failures = [("q002", "HydeGenerationError: ...")]
    metrics = run.metrics(0.075, 0.30)
    assert metrics["recall_at_10"] == 0.5
    assert metrics["questions"] == 2
    assert metrics["queries_failed"] == 1


def test_latency_is_measured_over_successful_queries_only():
    # A query that raised has no meaningful wall-clock time, and folding a 0
    # into the percentile would quietly lower the reported p95.
    run = LaneRun(lane_id="x", label="X")
    run.retrieved = [["a"], []]
    run.gold = [["a"], ["b"]]
    run.latencies_ms = [100.0]
    run.failures = [("q002", "boom")]
    assert run.metrics(0.075, 0.30)["p95_latency_ms"] == 100.0


def test_a_lane_that_failed_every_query_reports_no_latency_rather_than_zero():
    # None, not 0.0. A zero in a latency column reads as "instant".
    run = LaneRun(lane_id="x", label="X")
    run.retrieved = [[]]
    run.gold = [["a"]]
    run.failures = [("q001", "boom")]
    metrics = run.metrics(0.075, 0.30)
    assert metrics["p95_latency_ms"] is None
    assert metrics["mean_latency_ms"] is None
    assert metrics["recall_at_10"] == 0.0


def test_cost_uses_the_measured_token_counts():
    # 157 prompt + 82 completion on one query, at $0.075/$0.30 per million:
    #   157 * 0.075 / 1e6 = 0.000011775
    #    82 * 0.30  / 1e6 = 0.0000246
    #   total             = 0.000036375
    run = LaneRun(lane_id="hyde", label="HyDE")
    run.retrieved = [["a"]]
    run.gold = [["a"]]
    run.latencies_ms = [400.0]
    run.prompt_tokens = [157]
    run.completion_tokens = [82]
    assert run.metrics(0.075, 0.30)["cost_per_query_usd"] == pytest.approx(0.000036375)


# --------------------------------------------------------------------------
# Split loading
# --------------------------------------------------------------------------


def test_load_split_reads_rows_in_qid_order(tmp_path):
    path = tmp_path / "split.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"qid": "q009", "question": "b?", "gold_chunk_ids": ["c2"]},
                {"qid": "q002", "question": "a?", "gold_chunk_ids": ["c1", "c3"]},
            ]
        ),
        encoding="utf-8",
    )
    rows = load_split(path)
    assert [r.qid for r in rows] == ["q002", "q009"]
    assert rows[0].gold_chunk_ids == ["c1", "c3"]
    assert isinstance(rows[0], GoldRow)


def test_load_split_rejects_a_row_with_no_gold_label(tmp_path):
    # A row with no gold chunk is unscoreable and would silently count as a
    # miss for every lane, dragging every number down by 1/n.
    path = tmp_path / "split.jsonl"
    path.write_text(
        json.dumps({"qid": "q001", "question": "a?", "gold_chunk_ids": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no gold_chunk_ids"):
        load_split(path)


def test_load_split_names_the_command_that_creates_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"goldset.split"):
        load_split(tmp_path / "absent.jsonl")


# --- groundedness is published only when the sweep finished ----------------
#
# The free tier makes a partial sweep the normal case: one judged answer costs
# about 8,500 prompt tokens against a 200,000/day ceiling. A rate over 5 of 35
# questions is not a groundedness rate, and the only thing between it and a
# published table is this check.


def _groundedness_file(tmp_path, complete, rate=0.9, of_questions=35):
    import json

    path = tmp_path / "groundedness.json"
    path.write_text(
        json.dumps(
            {
                "lanes": {
                    "hybrid_rerank": {
                        "groundedness_rate": rate,
                        "answers_scored": 35 if complete else 5,
                        "complete": complete,
                        "of_questions_in_split": of_questions,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_partial_groundedness_sweep_is_not_published(tmp_path):
    from backend.evaluate import read_groundedness

    assert read_groundedness(_groundedness_file(tmp_path, complete=False), 35) == {}


def test_a_complete_groundedness_sweep_is_published(tmp_path):
    from backend.evaluate import read_groundedness

    assert read_groundedness(_groundedness_file(tmp_path, complete=True), 35) == {
        "hybrid_rerank": 0.9
    }


def test_a_sweep_over_a_different_split_size_is_not_published(tmp_path):
    """A rate measured over the 80-row train split must not attach to a 35-row
    test run just because the lane ids happen to match."""
    from backend.evaluate import read_groundedness

    path = _groundedness_file(tmp_path, complete=True, of_questions=80)

    assert read_groundedness(path, 35) == {}


def test_a_missing_groundedness_file_is_not_an_error(tmp_path):
    from backend.evaluate import read_groundedness

    assert read_groundedness(tmp_path / "absent.json", 35) == {}


def test_groundedness_is_absent_rather_than_zero_when_unmeasured():
    """A zero in a published table reads as a measurement. This is an absence."""
    from backend.evaluate import _with_groundedness

    metrics = {"id": "bm25", "recall_at_10": 0.85}

    assert _with_groundedness(metrics, {}) == metrics
    assert "groundedness" not in _with_groundedness(metrics, {})
    assert _with_groundedness(metrics, {"bm25": 0.77})["groundedness"] == 0.77
