"""Calibration tests. The blindness ones are the point.

Day 1's audit checked blindness by asserting no `second_*` key reached the
screen. That check named the fields that existed when it was written, so
`screen_scores` — added later — walked straight past it. The lesson was not
"add screen_scores to the list", it was that a deny-list cannot be right about
a field nobody has invented yet.

So the fixture below stamps `TELLTALE_` on **every** judge-side field, including
one that does not exist in the codebase at all, and the tests assert those
markers appear in neither the serialised queue row nor the rendered screen. Plus
the strongest form: the bytes are identical whichever way the judge voted.
"""

import json

import pytest

from backend.judge_calibrate import (
    BLIND_FIELDS,
    CalibrationError,
    Progress,
    blind_row,
    build_queue,
    item_id,
    key_row,
    label_row,
    render_item,
    score,
)

TEXTS = {f"c{i:02d}": f"passage {i} body text" for i in range(1, 11)}


def judged_record(qid="q001", grounded=True, lane_id="hybrid_rerank") -> dict:
    """A record shaped like the ones `judge.py` actually writes — plus more.

    Every judge-side value carries a TELLTALE_ marker, and `TELLTALE_FUTURE` is
    a field that does not exist in `judge.py` today. It stands for the field
    somebody adds next month: a whitelist keeps it off the screen without
    anyone remembering to, which a deny-list cannot.
    """
    return {
        "qid": qid,
        "lane_id": lane_id,
        "question": "Which package is needed for form data?",
        "answer": "Install python-multipart. It is required for forms.",
        "chunk_ids": ["c01", "c02", "c03"],
        "grounded": grounded,
        "verdicts": [
            {
                "n": 1,
                "sentence": "Install python-multipart.",
                "supported": True,
                "reason": "TELLTALE_REASON stated verbatim in passage 1",
            },
            {
                "n": 2,
                "sentence": "It is required for forms.",
                "supported": grounded,
                "reason": "TELLTALE_REASON not stated anywhere",
            },
        ],
        "generate_completion_tokens": 42,
        "judge_completion_tokens": 91,
        "TELLTALE_FUTURE": "a field added to judge.py after this test was written",
    }


# --- blindness ------------------------------------------------------------


def test_no_judge_field_reaches_the_queue_row():
    row = blind_row(judged_record(), TEXTS)

    assert "TELLTALE" not in json.dumps(row)
    assert "grounded" not in json.dumps(row)
    assert "supported" not in json.dumps(row)


def test_no_judge_field_reaches_the_rendered_screen():
    screen = render_item(blind_row(judged_record(), TEXTS), Progress(total=30, done=0))

    assert "TELLTALE" not in screen
    # The word "supported" is in the *instructions* ("g = grounded (all sentences
    # supported)"), so its presence proves nothing either way. What must not
    # appear is anything the judge decided, which the markers cover, and the
    # byte-identity test below covers the rest.
    assert "verbatim in passage" not in screen
    assert "not stated anywhere" not in screen


def test_a_field_added_to_the_verdict_file_later_cannot_leak():
    """The whitelist property, stated directly. This is the test Day 1 lacked."""
    record = judged_record()
    record["some_future_verdict_field"] = "TELLTALE_TOMORROW"
    record["confidence"] = 0.97

    row = blind_row(record, TEXTS)

    assert "TELLTALE_TOMORROW" not in json.dumps(row)
    assert "confidence" not in row
    assert set(row) == set(BLIND_FIELDS)


def test_the_queue_row_is_byte_identical_whichever_way_the_judge_voted():
    """The strongest form: the bytes cannot vary with anything the judge decided."""
    grounded = judged_record(grounded=True)
    ungrounded = judged_record(grounded=False)
    ungrounded["verdicts"][0]["supported"] = False
    ungrounded["verdicts"][0]["reason"] = "TELLTALE_REASON invented"

    assert blind_row(grounded, TEXTS) == blind_row(ungrounded, TEXTS)
    assert render_item(blind_row(grounded, TEXTS), Progress(30, 0)) == render_item(
        blind_row(ungrounded, TEXTS), Progress(30, 0)
    )


def test_the_sentences_shown_are_re_split_not_read_off_the_verdicts():
    """The verdict list carries `supported` on every element. Copying it and
    popping two keys works today and breaks the first time a third is added."""
    record = judged_record()
    # Corrupt the verdict list. The screen must not notice.
    record["verdicts"] = [{"sentence": "TELLTALE_SENTENCE", "supported": False, "reason": "x"}]

    row = blind_row(record, TEXTS)

    assert row["sentences"] == ["Install python-multipart.", "It is required for forms."]
    assert "TELLTALE_SENTENCE" not in json.dumps(row)


def test_the_progress_line_withholds_the_agreement_rate():
    """Watch a running agreement rate move and you can infer the verdict on the
    item you just labelled, then anchor on it for the next."""
    line = Progress(total=30, done=17).line()

    assert line == "[17/30]"
    assert "agree" not in line.lower()


def test_the_human_label_carries_no_judge_field():
    label = label_row(blind_row(judged_record(), TEXTS), "g")

    assert "TELLTALE" not in json.dumps(label)
    assert set(label) == {
        "item_id",
        "qid",
        "lane_id",
        "human_grounded",
        "skipped",
        "labelled_at",
        "blind",
    }
    assert label["human_grounded"] is True
    assert label["blind"] is True


def test_a_skip_is_recorded_as_a_skip_not_as_ungrounded():
    """"Cannot tell" is not "unsupported", and scoring it as one would move the
    agreement rate on items the labeller declined to judge."""
    label = label_row(blind_row(judged_record(), TEXTS), "s")

    assert label["skipped"] is True
    assert label["human_grounded"] is False  # not read: `skipped` gates it


def test_the_labeller_sees_the_context_they_need_to_judge_grounding():
    """Blind to the verdict, not blind to the evidence. A labeller who cannot
    read the passages cannot judge groundedness at all."""
    screen = render_item(blind_row(judged_record(), TEXTS), Progress(30, 0))

    assert "passage 1 body text" in screen
    assert "Which package is needed for form data?" in screen
    assert "Install python-multipart." in screen


# --- the queue ------------------------------------------------------------


def records(n: int, grounded_fraction: float = 1.0) -> list[dict]:
    cut = round(n * grounded_fraction)
    return [judged_record(qid=f"q{i:03d}", grounded=i < cut) for i in range(n)]


def test_the_key_is_a_separate_artefact_from_the_queue():
    """A hidden field is one careless print away from the screen; a file the
    labelling command never opens cannot be rendered by accident."""
    queue, key = build_queue(records(40), TEXTS, size=30)

    assert len(queue) == len(key) == 30
    assert "TELLTALE" not in json.dumps(queue)
    assert "TELLTALE" in json.dumps(key)  # the verdicts really are in the key
    assert {row["item_id"] for row in queue} == {row["item_id"] for row in key}


def test_the_queue_is_deterministic_for_a_seed():
    first, _ = build_queue(records(40), TEXTS, size=30, seed=42)
    second, _ = build_queue(records(40), TEXTS, size=30, seed=42)

    assert [row["item_id"] for row in first] == [row["item_id"] for row in second]


def test_too_few_judged_answers_refuses_rather_than_returning_a_short_queue():
    with pytest.raises(CalibrationError, match="need 30"):
        build_queue(records(12), TEXTS, size=30)


def test_a_stratified_queue_is_stamped_on_the_key_not_on_the_queue_row():
    """The flag is a fact about the judge's verdicts. On a queue row it would be
    one more thing to keep off the screen."""
    _, key = build_queue(records(40, grounded_fraction=0.9), TEXTS, size=30, stratify=True)

    assert all(row["stratified"] for row in key)
    queue, _ = build_queue(records(40, grounded_fraction=0.9), TEXTS, size=30, stratify=True)
    assert "stratified" not in json.dumps(queue)


def test_stratifying_takes_every_minority_item_it_can_and_tops_up_from_the_rest():
    """40 records, 36 grounded and 4 ungrounded, asking for 20.

    Half of 20 is 10, but only 4 ungrounded exist. Rather than returning a short
    queue the draw takes all 4 and fills the remaining 16 from the grounded side.
    That is still a better look at the minority class than a uniform draw, which
    would expect 20 x 0.1 = 2 of them.
    """
    _, key = build_queue(records(40, grounded_fraction=0.9), TEXTS, size=20, stratify=True)
    grounded = sum(1 for row in key if row["judge_grounded"])

    assert (grounded, len(key) - grounded) == (16, 4)


def test_stratifying_splits_evenly_when_both_classes_are_plentiful():
    _, key = build_queue(records(40, grounded_fraction=0.5), TEXTS, size=20, stratify=True)
    grounded = sum(1 for row in key if row["judge_grounded"])

    assert (grounded, len(key) - grounded) == (10, 10)


def test_item_ids_distinguish_the_same_question_across_lanes():
    assert item_id(judged_record("q001", lane_id="hyde")) == "hyde:q001"
    assert item_id(judged_record("q001", lane_id="bm25")) != item_id(
        judged_record("q001", lane_id="hyde")
    )


# --- scoring --------------------------------------------------------------


def key_for(pairs: list[tuple[str, bool]]) -> list[dict]:
    return [
        key_row({**judged_record(qid=item), "grounded": grounded, "lane_id": "l"})
        for item, grounded in pairs
    ]


def labels_for(pairs: list[tuple[str, bool, bool]]) -> list[dict]:
    return [
        {
            "item_id": f"l:{item}",
            "qid": item,
            "lane_id": "l",
            "human_grounded": human,
            "skipped": skipped,
            "blind": True,
        }
        for item, human, skipped in pairs
    ]


def test_agreement_and_kappa_on_a_hand_computed_fixture():
    # judge:  T T T F      human: T T F F
    #   agreement  = 3/4 = 0.75
    #   p_chance   = (2/4 x 3/4) + (2/4 x 1/4) = 0.375 + 0.125 = 0.5
    #   kappa      = (0.75 - 0.5) / 0.5 = 0.5
    key = key_for([("a", True), ("b", True), ("c", True), ("d", False)])
    labels = labels_for(
        [("a", True, False), ("b", True, False), ("c", False, False), ("d", False, False)]
    )

    report = score(labels, key, stratified=False)

    assert report["agreement"] == 0.75
    assert report["cohens_kappa"] == 0.5
    assert report["confusion"] == {
        "both_grounded": 2,
        "both_ungrounded": 1,
        "judge_grounded_human_not": 1,
        "human_grounded_judge_not": 0,
    }


def test_a_judge_that_says_grounded_to_everything_scores_high_agreement_and_zero_kappa():
    """The finding this whole step exists to be able to state. 90% agreement and
    kappa 0.0 is a judge carrying no information, and raw agreement hides it."""
    key = key_for([(f"q{i}", True) for i in range(10)])
    labels = labels_for([(f"q{i}", i != 0, False) for i in range(10)])

    report = score(labels, key, stratified=False)

    assert report["agreement"] == 0.9
    assert report["cohens_kappa"] == 0.0


def test_skipped_items_are_excluded_from_the_rate_not_counted_as_disagreement():
    key = key_for([("a", True), ("b", True)])
    labels = labels_for([("a", True, False), ("b", False, True)])

    report = score(labels, key, stratified=False)

    assert report["scored"] == 1
    assert report["skipped"] == 1
    assert report["agreement"] == 1.0


def test_per_class_agreement_is_reported_separately():
    """Where a judge fails matters more than how often. A judge that is perfect
    on grounded answers and useless on ungrounded ones has one number worth
    knowing and it is not the overall rate."""
    key = key_for([("a", True), ("b", True), ("c", False), ("d", False)])
    labels = labels_for(
        [("a", True, False), ("b", True, False), ("c", True, False), ("d", True, False)]
    )

    report = score(labels, key, stratified=False)
    by_verdict = {row["judge_said"]: row for row in report["by_judge_verdict"]}

    assert by_verdict["grounded"]["agreement"] == 1.0
    assert by_verdict["ungrounded"]["agreement"] == 0.0


def test_a_stratified_report_refuses_to_be_read_as_a_population_rate():
    key = [{**row, "stratified": True} for row in key_for([("a", True), ("b", False)])]
    labels = labels_for([("a", True, False), ("b", False, False)])

    report = score(labels, key, stratified=True)

    assert report["population_representative"] is False
    assert "must NOT be quoted as a population agreement rate" in report["note"]


def test_a_uniform_report_says_it_estimates_the_population():
    key = key_for([("a", True), ("b", False)])
    labels = labels_for([("a", True, False), ("b", False, False)])

    report = score(labels, key, stratified=False)

    assert report["population_representative"] is True
    assert report["blind"] is True
