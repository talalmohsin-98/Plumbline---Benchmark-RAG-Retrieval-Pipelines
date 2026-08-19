"""Metric tests against fixtures computed by hand.

CLAUDE.md is explicit that a metric test which records the implementation's
current output catches nothing. So every expected value here is worked out in
the comment above the assertion, in fractions, and the fixture is small enough
that the arithmetic can be followed without running anything.
"""

import pytest

from backend.metrics import (
    cost_per_query_usd,
    hit_at_k,
    mean_reciprocal_rank,
    p95_latency_ms,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

# A ranked list of ten ids, c1 best. Every fixture below indexes into this.
TOP_TEN = [f"c{i}" for i in range(1, 11)]


# --------------------------------------------------------------------------
# hit_at_k / reciprocal_rank
# --------------------------------------------------------------------------


def test_hit_at_k_respects_the_cutoff_exactly():
    # c5 is at rank 5: inside k=5, inside k=10, outside k=4.
    assert hit_at_k(TOP_TEN, ["c5"], 5) is True
    assert hit_at_k(TOP_TEN, ["c5"], 10) is True
    assert hit_at_k(TOP_TEN, ["c5"], 4) is False


def test_hit_at_k_is_true_when_any_gold_chunk_lands():
    # Multi-labelled row: c99 is never retrieved, c2 is at rank 2. Finding
    # either is answering the question.
    assert hit_at_k(TOP_TEN, ["c99", "c2"], 5) is True
    assert hit_at_k(TOP_TEN, ["c99", "c98"], 10) is False


def test_reciprocal_rank_is_one_over_the_first_gold_position():
    assert reciprocal_rank(TOP_TEN, ["c1"], 10) == pytest.approx(1.0)
    assert reciprocal_rank(TOP_TEN, ["c2"], 10) == pytest.approx(0.5)
    assert reciprocal_rank(TOP_TEN, ["c4"], 10) == pytest.approx(0.25)


def test_reciprocal_rank_takes_the_earliest_gold_not_the_first_listed():
    # gold = [c8, c3] in that order, but c3 is retrieved at rank 3 and c8 at
    # rank 8. RR must be 1/3, not 1/8: returning either copy of a duplicated
    # passage first is not a ranking error.
    assert reciprocal_rank(TOP_TEN, ["c8", "c3"], 10) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_past_the_cutoff():
    # c8 exists in the list but sits outside k=5.
    assert reciprocal_rank(TOP_TEN, ["c8"], 5) == 0.0
    assert reciprocal_rank(TOP_TEN, ["c99"], 10) == 0.0


def test_cutoffs_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        hit_at_k(TOP_TEN, ["c1"], 0)
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank(TOP_TEN, ["c1"], -1)


# --------------------------------------------------------------------------
# recall@k and MRR over a four-question fixture
#
#   Q1 gold c3   -> rank 3    in top-5, in top-10, RR = 1/3
#   Q2 gold c8   -> rank 8    not top-5, in top-10, RR = 1/8
#   Q3 gold c99  -> absent    no hit at any k,      RR = 0
#   Q4 gold c1   -> rank 1    in top-5, in top-10,  RR = 1
#
#   recall@5  = 2/4 = 0.5
#   recall@10 = 3/4 = 0.75
#   MRR@10 = (1/3 + 1/8 + 0 + 1)/4 = (8/24 + 3/24 + 24/24)/4 = (35/24)/4 = 35/96
#   MRR@5  = (1/3 +  0  + 0 + 1)/4 = (4/3)/4 = 1/3
# --------------------------------------------------------------------------

FOUR_RETRIEVED = [TOP_TEN, TOP_TEN, TOP_TEN, TOP_TEN]
FOUR_GOLD = [["c3"], ["c8"], ["c99"], ["c1"]]


def test_recall_at_five_is_one_half():
    assert recall_at_k(FOUR_RETRIEVED, FOUR_GOLD, 5) == pytest.approx(0.5)


def test_recall_at_ten_is_three_quarters():
    assert recall_at_k(FOUR_RETRIEVED, FOUR_GOLD, 10) == pytest.approx(0.75)


def test_mrr_at_ten_is_thirty_five_ninety_sixths():
    assert mean_reciprocal_rank(FOUR_RETRIEVED, FOUR_GOLD, 10) == pytest.approx(35 / 96)
    # Stated as a decimal too, so a typo in the fraction cannot pass silently.
    assert mean_reciprocal_rank(FOUR_RETRIEVED, FOUR_GOLD, 10) == pytest.approx(
        0.3645833333333333
    )


def test_mrr_at_five_drops_the_question_answered_at_rank_eight():
    assert mean_reciprocal_rank(FOUR_RETRIEVED, FOUR_GOLD, 5) == pytest.approx(1 / 3)


def test_recall_counts_a_question_once_however_many_gold_chunks_it_has():
    # Four gold chunks, three of them retrieved. This is 1.0, not 3/4: the
    # metric asks whether the question was answered, not what fraction of the
    # labels were recovered. EVALUATION_SPEC §2 defines it this way.
    assert recall_at_k([TOP_TEN], [["c1", "c2", "c3", "c99"]], 10) == pytest.approx(1.0)


def test_perfect_and_empty_extremes():
    assert recall_at_k([TOP_TEN], [["c1"]], 10) == 1.0
    assert mean_reciprocal_rank([TOP_TEN], [["c1"]], 10) == 1.0
    assert recall_at_k([[]], [["c1"]], 10) == 0.0
    assert mean_reciprocal_rank([[]], [["c1"]], 10) == 0.0
    assert recall_at_k([], [], 10) == 0.0
    assert mean_reciprocal_rank([], [], 10) == 0.0


def test_misaligned_inputs_raise_rather_than_score_the_wrong_questions():
    with pytest.raises(ValueError, match="not the same questions"):
        recall_at_k([TOP_TEN, TOP_TEN], [["c1"]], 10)
    with pytest.raises(ValueError, match="not the same questions"):
        mean_reciprocal_rank([TOP_TEN], [["c1"], ["c2"]], 10)


def test_recall_at_ten_is_never_below_recall_at_five():
    # A property rather than a fixture: this is the invariant that would break
    # first if a cutoff were ever applied off-by-one.
    at_five = recall_at_k(FOUR_RETRIEVED, FOUR_GOLD, 5)
    at_ten = recall_at_k(FOUR_RETRIEVED, FOUR_GOLD, 10)
    assert at_ten >= at_five


# --------------------------------------------------------------------------
# percentile / p95
#
#   values = [10, 20, 30, 40], n = 4, so positions run 0..3.
#   p95:  0.95 * 3 = 2.85 -> between index 2 (30) and 3 (40), weight 0.85
#         30 * 0.15 + 40 * 0.85 = 4.5 + 34.0 = 38.5
#   p50:  0.50 * 3 = 1.5  -> 20 * 0.5 + 30 * 0.5 = 25.0
# --------------------------------------------------------------------------

FOUR_LATENCIES = [10.0, 20.0, 30.0, 40.0]


def test_p95_interpolates_between_ranks():
    assert p95_latency_ms(FOUR_LATENCIES) == pytest.approx(38.5)


def test_percentile_midpoint_and_endpoints():
    assert percentile(FOUR_LATENCIES, 50.0) == pytest.approx(25.0)
    assert percentile(FOUR_LATENCIES, 0.0) == pytest.approx(10.0)
    assert percentile(FOUR_LATENCIES, 100.0) == pytest.approx(40.0)


def test_percentile_does_not_care_about_input_order():
    assert p95_latency_ms([40.0, 10.0, 30.0, 20.0]) == pytest.approx(38.5)


def test_percentile_of_one_value_is_that_value():
    assert p95_latency_ms([7.5]) == pytest.approx(7.5)


def test_percentile_rejects_empty_input_and_out_of_range_q():
    with pytest.raises(ValueError, match="undefined"):
        percentile([], 95.0)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        percentile(FOUR_LATENCIES, 101.0)


# --------------------------------------------------------------------------
# cost
#
#   prompt = [1000, 2000] -> 3000 tokens at $0.05/M = $0.000150
#   completion = [100, 200] ->  300 tokens at $0.08/M = $0.000024
#   total $0.000174 over 2 queries = $0.000087 per query
# --------------------------------------------------------------------------


def test_cost_per_query_from_measured_tokens():
    cost = cost_per_query_usd([1000, 2000], [100, 200], 0.05, 0.08)
    assert cost == pytest.approx(0.000087)


def test_a_lane_that_makes_no_llm_call_costs_exactly_zero():
    # Not "about zero". The published table prints $0.000 for five of six
    # lanes and that has to be a real zero, not a rounded small number.
    assert cost_per_query_usd([], [], 0.05, 0.08) == 0.0
    assert cost_per_query_usd([0, 0], [0, 0], 0.05, 0.08) == 0.0


def test_cost_rejects_mismatched_token_sequences():
    with pytest.raises(ValueError, match="same length"):
        cost_per_query_usd([1000], [100, 200], 0.05, 0.08)
