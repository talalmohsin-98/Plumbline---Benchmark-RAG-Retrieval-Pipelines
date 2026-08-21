"""Metric tests against fixtures computed by hand.

CLAUDE.md is explicit that a metric test which records the implementation's
current output catches nothing. So every expected value here is worked out in
the comment above the assertion, in fractions, and the fixture is small enough
that the arithmetic can be followed without running anything.
"""

import pytest

from backend.metrics import (
    cost_per_query_usd,
    discordant_pairs,
    hit_at_k,
    max_detectable_wins,
    mcnemar_exact,
    mean,
    mean_reciprocal_rank,
    p95_latency_ms,
    paired_bootstrap_ci,
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


# --------------------------------------------------------------------------
# Paired significance tests
# --------------------------------------------------------------------------
#
# Every p-value below is worked out in the comment above it. These are the
# functions the lane 6 verdict rests on, and a test that merely recorded what
# the implementation returned would be worth nothing at all.


def test_discordant_pairs_counts_only_the_questions_that_disagree():
    #  q0: both hit          -> concordant, carries no information
    #  q1: a hit, b missed   -> a wins
    #  q2: b hit, a missed   -> b wins
    #  q3: both missed       -> concordant
    #  q4: a hit, b missed   -> a wins
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]

    assert discordant_pairs(a, b) == (2, 1)


def test_discordant_pairs_refuses_misaligned_inputs():
    with pytest.raises(ValueError, match="not the same questions"):
        discordant_pairs([True, False], [True])


def test_mcnemar_a_clean_sweep_of_six_is_the_smallest_significant_result():
    # n = 6 discordant pairs, all one way.
    #   P(X >= 6) = C(6,6) / 2^6 = 1/64
    #   two-sided  = 2 * 1/64    = 0.03125 < 0.05
    assert mcnemar_exact(6, 0) == pytest.approx(0.03125)
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)  # symmetric


def test_mcnemar_a_clean_sweep_of_five_is_not_significant():
    # 2 * C(5,5)/2^5 = 2/32 = 0.0625 -- just over alpha. Five is not enough.
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)


def test_mcnemar_three_nil_is_the_ceiling_this_benchmark_actually_faces():
    # Lane 4 misses 3 of 35 questions at k=10, so lane 6 can win at most 3
    # discordant pairs. 2 * C(3,3)/2^3 = 2/8 = 0.25. This number is why
    # EVALUATION_SPEC §3 demotes recall@10 to a descriptive count -- and it is
    # asserted here so the claim in the spec is checkable rather than rhetorical.
    assert mcnemar_exact(3, 0) == pytest.approx(0.25)


def test_mcnemar_with_losses_mixed_in():
    # 8 wins, 1 loss. n = 9, extreme = 8.
    #   P(X >= 8) = (C(9,8) + C(9,9)) / 2^9 = (9 + 1)/512 = 10/512
    #   two-sided = 20/512 = 0.0390625
    assert mcnemar_exact(8, 1) == pytest.approx(0.0390625)

    # 7 wins, 1 loss. n = 8, extreme = 7.
    #   (C(8,7) + C(8,8)) / 2^8 = 9/256; doubled = 18/256 = 0.0703125
    assert mcnemar_exact(7, 1) == pytest.approx(0.0703125)


def test_mcnemar_caps_at_one_when_the_split_is_even():
    # n = 2, extreme = 1. (C(2,1) + C(2,2))/4 = 3/4; doubled = 1.5, capped to 1.
    assert mcnemar_exact(1, 1) == 1.0


def test_mcnemar_with_no_discordant_pairs_is_one():
    # Two lanes that agreed on every question. No evidence either way.
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_exact(-1, 3)


def test_max_detectable_wins_is_the_baselines_miss_count():
    # A challenger can only win a question the baseline got wrong.
    assert max_detectable_wins([True, True, True, False, False]) == 2
    assert max_detectable_wins([True] * 35) == 0  # a perfect baseline cannot be beaten


def test_bootstrap_of_a_constant_difference_collapses_to_that_constant():
    # Every resample averages the same value, so both interval ends are it.
    low, high = paired_bootstrap_ci([0.25] * 20, resamples=200, seed=42)

    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)


def test_bootstrap_of_differences_centred_on_zero_straddles_zero():
    low, high = paired_bootstrap_ci([0.4, -0.4] * 20, resamples=2000, seed=42)

    assert low < 0.0 < high


def test_bootstrap_of_a_clear_positive_effect_excludes_zero():
    low, high = paired_bootstrap_ci([0.3, 0.35, 0.4, 0.32, 0.38] * 8, resamples=2000, seed=42)

    assert low > 0.0


def test_bootstrap_is_deterministic_for_a_seed():
    values = [0.1, -0.2, 0.4, 0.0, 0.3, -0.1, 0.25]

    assert paired_bootstrap_ci(values, resamples=500, seed=42) == paired_bootstrap_ci(
        values, resamples=500, seed=42
    )


def test_bootstrap_with_a_different_seed_gives_a_different_interval():
    values = [0.1, -0.2, 0.4, 0.0, 0.3, -0.1, 0.25]

    assert paired_bootstrap_ci(values, resamples=500, seed=42) != paired_bootstrap_ci(
        values, resamples=500, seed=7
    )


def test_bootstrap_rejects_inputs_it_cannot_resample():
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_ci([])
    with pytest.raises(ValueError, match="resamples must be positive"):
        paired_bootstrap_ci([0.1], resamples=0)
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap_ci([0.1], confidence=1.0)


def test_mean_of_nothing_is_zero_rather_than_an_exception():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0, 6.0]) == pytest.approx(3.0)
