"""Significance runner tests: pairing, the tripwire, and the verdict rule.

The arithmetic itself lives in `metrics.py` and is tested there against
hand-computed fixtures. What is tested here is everything around it — that the
two lanes are paired by qid rather than by dict order, that the tripwire fires
where the pre-registration says it does, and that the verdict comes from the
pre-registered test and not from whichever test happened to agree.
"""

import pytest

from backend.significance import (
    ALPHA,
    TRIPWIRE_RECALL_AT_10,
    MissingLaneError,
    aligned_outcomes,
    check_tripwire,
    compare,
    decide,
    run_bootstrap,
    run_mcnemar,
)


def outcome(hit10: bool, hit5: bool | None = None, rr: float | None = None) -> dict:
    hit5 = hit10 if hit5 is None else hit5
    if rr is None:
        rr = 0.5 if hit10 else 0.0
    return {
        "retrieved": ["g"] if hit10 else ["x"],
        "hit_at_5": hit5,
        "hit_at_10": hit10,
        "reciprocal_rank": rr,
        "latency_ms": 1.0,
    }


def document(challenger: list[dict], baseline: list[dict], qids: list[str] | None = None) -> dict:
    qids = qids or [f"q{i:03d}" for i in range(len(challenger))]
    return {
        "split": "test",
        "score_depth": 10,
        "questions": [{"qid": q, "gold_chunk_ids": ["g"]} for q in qids],
        "lanes": {
            "hybrid_rerank_tuned": dict(zip(qids, challenger, strict=True)),
            "hybrid_rerank": dict(zip(qids, baseline, strict=True)),
        },
    }


# --- pairing --------------------------------------------------------------


def test_the_two_lanes_are_paired_by_qid_not_by_dict_order():
    """The failure this prevents produces a plausible p-value from a scrambled
    comparison, which is the worst thing this module could do quietly."""
    qids = ["q001", "q002", "q003"]
    doc = document([outcome(True), outcome(False), outcome(True)], [outcome(False)] * 3, qids)
    # Re-insert the baseline in reversed order; the pairing must not notice.
    doc["lanes"]["hybrid_rerank"] = {
        "q003": outcome(True),
        "q002": outcome(False),
        "q001": outcome(False),
    }

    order, challenger, baseline = aligned_outcomes(doc, "hybrid_rerank_tuned", "hybrid_rerank")

    assert order == qids
    assert [row["hit_at_10"] for row in baseline] == [False, False, True]
    assert [row["hit_at_10"] for row in challenger] == [True, False, True]


def test_a_lane_missing_from_the_file_is_named_not_skipped():
    doc = document([outcome(True)], [outcome(True)])

    with pytest.raises(MissingLaneError, match="hybrid_rerank_tuned"):
        aligned_outcomes(
            {**doc, "lanes": {"hybrid_rerank": doc["lanes"]["hybrid_rerank"]}},
            "hybrid_rerank_tuned",
            "hybrid_rerank",
        )


def test_a_question_missing_from_one_lane_is_refused():
    doc = document([outcome(True), outcome(True)], [outcome(True), outcome(True)])
    del doc["lanes"]["hybrid_rerank_tuned"]["q001"]

    with pytest.raises(MissingLaneError, match="missing from a lane"):
        aligned_outcomes(doc, "hybrid_rerank_tuned", "hybrid_rerank")


# --- the power ceiling ----------------------------------------------------


def test_mcnemar_reports_the_ceiling_alongside_the_result():
    """Reproduces the pre-registration's table from data rather than from prose.

    Baseline hits 32 of 35 at k=10, so at most 3 discordant pairs are available
    and the best attainable p-value is 0.25.
    """
    baseline = [outcome(True)] * 32 + [outcome(False)] * 3
    challenger = [outcome(True)] * 33 + [outcome(False)] * 2

    result = run_mcnemar("hit_at_10", challenger, baseline)

    assert result.baseline_hits == 32
    assert result.challenger_hits == 33
    assert result.max_possible_wins == 3
    assert result.best_possible_p == pytest.approx(0.25)
    assert not result.can_reach_alpha
    assert ALPHA == 0.05


def test_a_metric_with_enough_headroom_can_reach_alpha():
    """recall@5: the baseline misses 6, so a clean 6-0 sweep reaches p=0.03125."""
    baseline = [outcome(True)] * 29 + [outcome(False)] * 6
    challenger = [outcome(True)] * 35

    result = run_mcnemar("hit_at_10", challenger, baseline)

    assert result.max_possible_wins == 6
    assert result.best_possible_p == pytest.approx(0.03125)
    assert result.can_reach_alpha
    assert result.significant  # this run actually achieved the sweep


def test_mcnemar_counts_losses_as_well_as_wins():
    baseline = [outcome(True), outcome(False), outcome(False)]
    challenger = [outcome(False), outcome(True), outcome(False)]

    result = run_mcnemar("hit_at_10", challenger, baseline)

    assert (result.challenger_wins, result.baseline_wins) == (1, 1)
    assert result.p_value == 1.0


# --- the tripwire ---------------------------------------------------------


def test_the_tripwire_fires_at_34_of_35_and_not_at_33():
    """EVALUATION_SPEC §3: above 0.95 recall@10. The band is one question wide,
    because lane 5 already reaches 33/35 legitimately."""
    assert TRIPWIRE_RECALL_AT_10 == 0.95
    assert not check_tripwire(33 / 35, "lane")["fired"]  # 0.9429, lane 5's real score
    assert check_tripwire(34 / 35, "lane")["fired"]  # 0.9714
    assert check_tripwire(35 / 35, "lane")["fired"]


def test_a_fired_tripwire_withholds_the_verdict_entirely():
    """Not "significant, but check for leakage". No verdict until the audit."""
    bootstrap = run_bootstrap([outcome(True, rr=1.0)] * 20, [outcome(True, rr=0.1)] * 20)
    fired = check_tripwire(35 / 35, "lane")

    assert bootstrap.excludes_zero  # the effect is real-looking
    assert decide(bootstrap, fired)["verdict"] == "withheld"
    assert "audited" in decide(bootstrap, fired)["statement"]


def test_a_fired_tripwire_makes_the_comparison_exit_non_zero_material():
    challenger = [outcome(True)] * 35
    baseline = [outcome(True)] * 32 + [outcome(False)] * 3

    result = compare(document(challenger, baseline), "hybrid_rerank_tuned", "hybrid_rerank")

    assert result["leakage_tripwire"]["fired"]
    assert result["verdict"] == "withheld"
    assert "STOP" in result["leakage_tripwire"]["action"]


# --- the verdict rule -----------------------------------------------------


def test_an_interval_that_includes_zero_is_no_detectable_difference():
    """Reported in those words. Not "better, but the sample is small"."""
    # Four questions both lanes miss, so the challenger stays under the
    # leakage tripwire and the verdict rule is what is being exercised.
    challenger = [outcome(True, rr=r) for r in (0.5, 0.2, 1.0, 0.33, 0.25) * 4] + [
        outcome(False)
    ] * 4
    baseline = [outcome(True, rr=r) for r in (0.33, 0.5, 0.25, 1.0, 0.2) * 4] + [
        outcome(False)
    ] * 4

    result = compare(document(challenger, baseline), "hybrid_rerank_tuned", "hybrid_rerank")

    assert not result["mrr_bootstrap"]["excludes_zero"]
    assert result["verdict"] == "no detectable difference"
    assert "includes 0" in result["statement"]


def test_a_negative_interval_publishes_as_a_regression_rather_than_nothing():
    """A loss is a result. It gets a verdict of its own, not silence."""
    challenger = [outcome(True, rr=0.2)] * 30 + [outcome(False)] * 4
    baseline = [outcome(True, rr=0.9)] * 30 + [outcome(False)] * 4

    result = compare(document(challenger, baseline), "hybrid_rerank_tuned", "hybrid_rerank")

    assert result["verdict"] == "regression"
    assert result["mrr_bootstrap"]["mean_difference"] < 0
    assert result["mrr_bootstrap"]["excludes_zero"]


def test_the_verdict_ignores_mcnemar_even_when_mcnemar_is_significant():
    """The pre-registered primary test is the MRR interval, and only it decides.

    Here recall clears alpha on a 6-0 sweep while the MRR interval straddles
    zero. Letting the recall result into the verdict would be choosing, after
    the fact, whichever test agreed with us.
    """
    # 6 questions flip from miss to hit; on the rest the RRs are shuffled so the
    # mean difference is near zero with a wide interval.
    baseline = (
        [outcome(False, rr=0.0)] * 6
        + [outcome(True, rr=r) for r in (1.0, 0.2, 0.5, 0.25, 0.33) * 4]
        + [outcome(False)] * 4
    )
    challenger = (
        [outcome(True, rr=0.05)] * 6
        + [outcome(True, rr=r) for r in (0.2, 1.0, 0.25, 0.5, 0.2) * 4]
        + [outcome(False)] * 4
    )

    result = compare(document(challenger, baseline), "hybrid_rerank_tuned", "hybrid_rerank")

    assert result["mcnemar"]["recall_at_10"]["significant_at_0.05"]
    assert not result["mrr_bootstrap"]["excludes_zero"]
    assert result["verdict"] == "no detectable difference"


def test_the_comparison_names_the_questions_won_and_lost():
    """So a reader can check "lane 6 wins q032" against per_question.json."""
    qids = ["q001", "q002", "q003", "q004"]
    challenger = [outcome(True), outcome(False), outcome(True), outcome(True)]
    baseline = [outcome(False), outcome(True), outcome(True), outcome(True)]

    result = compare(document(challenger, baseline, qids), "hybrid_rerank_tuned", "hybrid_rerank")

    assert result["questions_won"] == ["q001"]
    assert result["questions_lost"] == ["q002"]


def test_the_comparison_stamps_the_protocol_it_followed():
    result = compare(
        document([outcome(True)] * 5, [outcome(True)] * 5), "hybrid_rerank_tuned", "hybrid_rerank"
    )

    assert "02_EVALUATION_SPEC.md" in result["protocol"]
    assert result["primary_test"] == "mrr_at_10 paired bootstrap"
    assert result["alpha"] == 0.05
