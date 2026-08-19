"""RRF tests. Every expected score below is worked out by hand in the comment
above it, because a fusion test that asserts against the implementation's own
output would pass just as happily on a fusion that adds the wrong reciprocals.
"""

import pytest

from backend.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion


def test_single_list_is_returned_in_order_with_hand_computed_scores():
    # One list [a, b, c], k=60. Scores are 1/(60+rank):
    #   a -> 1/61 = 0.016393442622950820
    #   b -> 1/62 = 0.016129032258064516
    #   c -> 1/63 = 0.015873015873015872
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1 / 61)
    assert fused[1][1] == pytest.approx(1 / 62)
    assert fused[2][1] == pytest.approx(1 / 63)


def test_appearing_in_both_lists_beats_a_better_rank_in_one():
    # This is the whole reason the hybrid lane exists, so it is asserted
    # directly rather than inferred from a bigger fixture.
    #
    #   A = [a, b, c]      B = [b, d, a]      k = 60
    #
    #   a = 1/61 + 1/63 = 0.016393442623 + 0.015873015873 = 0.032266458496
    #   b = 1/62 + 1/61 = 0.016129032258 + 0.016393442623 = 0.032522474881
    #   d = 1/62        = 0.016129032258
    #   c = 1/63        = 0.015873015873
    #
    # b wins despite ranking 2nd in A, because it is the only id near the top
    # of both lists. a ranks 1st in A and still loses.
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1][1] == pytest.approx(1 / 61 + 1 / 63)
    assert fused[2][1] == pytest.approx(1 / 62)
    assert fused[3][1] == pytest.approx(1 / 63)


def test_k_controls_how_sharply_rank_one_dominates():
    # ARCHITECTURE §11 claims small k lets rank 1 overwhelm the other list and
    # large k flattens toward a vote. This fixture is that claim, checked.
    #
    #   A = [a, b]                 B = [c1..c9, b]   (b is 10th in B)
    #
    # a is 1st in A and absent from B. b is 2nd in A and 10th in B. The
    # question is whether one strong placement beats two mediocre ones, and k
    # is what decides it:
    #
    #   k=1:   a = 1/2         = 0.500000
    #          b = 1/3 + 1/11  = 0.333333 + 0.090909 = 0.424242   -> a above b
    #
    #   k=60:  a = 1/61        = 0.016393
    #          b = 1/62 + 1/70 = 0.016129 + 0.014286 = 0.030415   -> b above a
    #
    # Positions are compared rather than the top of the list, because c1 is
    # rank 1 of B and ties a exactly (1/2 at k=1, 1/61 at k=60). That tie is
    # correct and is not what this test is about.
    short = ["a", "b"]
    long = [f"c{i}" for i in range(1, 10)] + ["b"]

    sharp = reciprocal_rank_fusion([short, long], k=1)
    sharp_order = [chunk_id for chunk_id, _ in sharp]
    assert sharp_order.index("a") < sharp_order.index("b")
    assert dict(sharp)["a"] == pytest.approx(0.5)
    assert dict(sharp)["b"] == pytest.approx(1 / 3 + 1 / 11)

    flat = reciprocal_rank_fusion([short, long], k=60)
    flat_order = [chunk_id for chunk_id, _ in flat]
    assert flat_order.index("b") < flat_order.index("a")
    assert flat_order[0] == "b"
    assert dict(flat)["a"] == pytest.approx(1 / 61)
    assert dict(flat)["b"] == pytest.approx(1 / 62 + 1 / 70)


def test_absence_from_a_list_costs_nothing_beyond_the_missing_term():
    # There is no penalty term: an id missing from B simply scores its A
    # contribution. a = 1/61, and nothing subtracts from it.
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 61)


def test_a_duplicate_within_one_list_is_only_paid_once():
    # [a, a, b]: a must score 1/61, not 1/61 + 1/62. Otherwise a retriever
    # that returned a chunk twice would fuse its way to the top.
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 63)


def test_ties_break_on_chunk_id_so_the_output_is_deterministic():
    # Symmetric input: a is 1st in A and 2nd in B, b is 2nd in A and 1st in B,
    # so both score exactly 1/61 + 1/62. The order must not depend on dict
    # iteration -- recall@10 would then vary between runs on the same data.
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(fused[1][1])

    reversed_input = reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60)
    assert [chunk_id for chunk_id, _ in reversed_input] == ["a", "b"]


def test_empty_input_fuses_to_nothing():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_k_must_be_positive():
    # k=0 would divide by zero at rank 0 only, but a negative k silently
    # produces negative scores and reverses the ranking. Refuse both.
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], k=-60)


def test_default_k_is_sixty():
    # Named because the README quotes it and the config default must agree.
    assert DEFAULT_RRF_K == 60
