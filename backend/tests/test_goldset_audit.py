"""Audit tests: stratification, blindness, and the agreement arithmetic.

The blindness tests are the ones that matter. If the screener's verdict leaks
onto the first screen, every agreement number this project publishes is a
measurement of anchoring instead.
"""

import json
from collections import Counter

import pytest

from backend.goldset import audit
from backend.goldset.audit import (
    Progress,
    allocate,
    audited_question,
    render_question,
    render_reveal,
    run,
    stratified_sample,
    summarise,
)
from backend.goldset.rules import RULES


def screened_row(qid, verdict, rule=None, reason="because", **extra):
    row = {
        "qid": qid,
        "question": f"Question for {qid}?",
        "gold_chunk_ids": [f"chunk_{qid}"],
        "source_doc": "fastapi/index.md",
        "screen_verdict": verdict,
        "screen_rule": rule,
        "screen_reason": reason,
    }
    row.update(extra)
    return row


def population(keeps=0, fixes=0, drops=0):
    return (
        [screened_row(f"k{i:03d}", "keep") for i in range(keeps)]
        + [screened_row(f"f{i:03d}", "fix") for i in range(fixes)]
        + [screened_row(f"d{i:03d}", "drop") for i in range(drops)]
    )


# --- allocation, computed by hand ---------------------------------------


def test_an_even_split_when_every_class_is_large_enough():
    assert allocate(40, {"keep": 200, "fix": 80, "drop": 70}) == {
        "keep": 13,
        "fix": 13,
        "drop": 14,
    }


def test_a_small_class_gives_what_it_has_and_the_rest_is_redistributed():
    # fix supplies only 2, so 38 are split between drop and keep.
    assert allocate(40, {"keep": 200, "fix": 2, "drop": 70}) == {
        "keep": 19,
        "fix": 2,
        "drop": 19,
    }


def test_an_empty_class_is_skipped_entirely():
    assert allocate(40, {"keep": 100, "fix": 0, "drop": 100}) == {
        "keep": 20,
        "fix": 0,
        "drop": 20,
    }


def test_a_population_smaller_than_the_sample_returns_all_of_it():
    assert allocate(40, {"keep": 5, "fix": 3, "drop": 2}) == {"keep": 5, "fix": 3, "drop": 2}


def test_allocation_never_exceeds_the_requested_total():
    for total in (1, 7, 40, 97):
        picked = allocate(total, {"keep": 300, "fix": 40, "drop": 60})
        assert sum(picked.values()) == total


# --- stratification -----------------------------------------------------


def test_all_three_verdicts_are_sampled_not_just_the_keeps():
    """Auditing keeps alone cannot see a false drop."""
    drawn = Counter(r["screen_verdict"] for r in stratified_sample(population(200, 80, 70), 40))
    assert drawn["keep"] > 0 and drawn["fix"] > 0 and drawn["drop"] > 0
    assert sum(drawn.values()) == 40


def test_the_sample_is_reproducible_for_a_seed():
    rows = population(200, 80, 70)
    assert [r["qid"] for r in stratified_sample(rows, 40, seed=42)] == [
        r["qid"] for r in stratified_sample(rows, 40, seed=42)
    ]


def test_a_different_seed_draws_differently():
    rows = population(200, 80, 70)
    assert [r["qid"] for r in stratified_sample(rows, 40, seed=42)] != [
        r["qid"] for r in stratified_sample(rows, 40, seed=7)
    ]


def test_the_sample_order_does_not_group_by_verdict():
    """A run of thirteen keeps is itself a hint about the screener's verdict."""
    verdicts = [r["screen_verdict"] for r in stratified_sample(population(200, 80, 70), 40)]
    assert len(set(verdicts[:6])) > 1


def test_unscored_rows_are_never_audited():
    rows = [*population(10, 10, 10), screened_row("u001", "unscored")]
    assert all(r["screen_verdict"] != "unscored" for r in stratified_sample(rows, 30))


def test_the_draw_does_not_depend_on_the_order_rows_were_written():
    rows = population(50, 20, 20)
    shuffled = list(reversed(rows))
    assert [r["qid"] for r in stratified_sample(rows, 12)] == [
        r["qid"] for r in stratified_sample(shuffled, 12)
    ]


# --- blindness ----------------------------------------------------------


def test_the_question_screen_does_not_depend_on_the_verdict_at_all():
    """The strongest form of the requirement: identical bytes for all three.

    Asserting the word "drop" is absent would not do -- it appears in the rule
    text, which is shown to everyone. What must not vary is the screen.
    """
    screens = {
        verdict: render_question(
            screened_row("q001", verdict, rule="generic", reason="a dozen tutorials answer it"),
            "the chunk body",
            Progress(total=40),
        )
        for verdict in ("keep", "fix", "drop")
    }
    assert len(set(screens.values())) == 1


def test_the_screener_reasoning_is_absent_from_the_question_screen():
    row = screened_row("q001", "drop", rule="generic", reason="a dozen tutorials answer it")
    screen = render_question(row, "the chunk body", Progress(total=40))

    assert "a dozen tutorials answer it" not in screen
    assert "screener" not in screen.lower()


def test_a_fixed_row_is_shown_as_the_screener_first_saw_it():
    """Showing the rephrasing would make the auditor agree by construction: the
    overlap the screener removed is already gone."""
    row = screened_row(
        "q001",
        "fix",
        question="Which header stops a proxy buffering a stream?",
        original_question="What prevents buffering in some proxies like Nginx?",
    )
    screen = render_question(row, "body", Progress(total=40))

    assert "What prevents buffering in some proxies like Nginx?" in screen
    assert "Which header stops a proxy buffering a stream?" not in screen


def test_an_unfixed_row_shows_its_own_question():
    assert audited_question(screened_row("q001", "keep")) == "Question for q001?"


def test_every_rule_is_on_screen_for_every_row():
    """Rules shown in full and in fixed order, so their presence leaks nothing."""
    for verdict in ("keep", "fix", "drop"):
        screen = render_question(screened_row("q001", verdict), "body", Progress(total=1))
        for rule in RULES:
            assert rule.text in screen


def test_the_reveal_shows_the_verdict_the_rule_and_the_reason():
    row = screened_row("q001", "drop", rule="generic", reason="a dozen tutorials answer it")
    reveal = render_reveal(row, "keep")

    assert "DISAGREE" in reveal
    assert "generic" in reveal
    assert "a dozen tutorials answer it" in reveal


def test_the_reveal_shows_the_rephrasing_for_a_fixed_row():
    row = screened_row(
        "q001",
        "fix",
        question="Which header stops a proxy buffering a stream?",
        original_question="What prevents buffering in some proxies like Nginx?",
    )
    assert "Which header stops a proxy buffering a stream?" in render_reveal(row, "fix")


def test_agreement_is_announced_when_the_verdicts_match():
    reveal = render_reveal(screened_row("q001", "keep"), "keep")
    assert "AGREE" in reveal and "DISAGREE" not in reveal


# --- the agreement arithmetic, computed by hand -------------------------


def decision(qid, human, screener):
    return {
        "qid": qid,
        "human_verdict": human,
        "screen_verdict": screener,
        "agreed": human == screener,
    }


def test_per_class_agreement_is_computed_within_each_class():
    decisions = (
        [decision(f"k{i}", "keep", "keep") for i in range(3)]
        + [decision("k3", "drop", "keep")]  # 3/4 keeps
        + [decision(f"d{i}", "drop", "drop") for i in range(2)]  # 2/2 drops
    )
    results = summarise(decisions, Counter({"keep": 200, "drop": 100}))

    assert results["by_verdict"]["keep"]["agreement"] == 0.75
    assert results["by_verdict"]["drop"]["agreement"] == 1.0


def test_the_raw_rate_is_the_plain_sample_average():
    decisions = [decision(f"k{i}", "keep", "keep") for i in range(3)] + [
        decision("d0", "keep", "drop")
    ]
    results = summarise(decisions, Counter({"keep": 200, "drop": 100}))
    assert results["agreement_overall"] == 0.75


def test_the_reweighted_rate_is_retired():
    """Full coverage leaves no unsampled stratum, so there is nothing to project onto.

    The figure it produced (15.5%) was correct arithmetic for a 51-row sample
    over 116 screened rows. Keeping it alongside a census would put two numbers
    for the same quantity in circulation, which is how the wrong one gets
    quoted.
    """
    decisions = [decision(f"k{i}", "keep", "keep") for i in range(4)] + [
        decision(f"d{i}", "keep", "drop") for i in range(4)
    ]
    results = summarise(decisions, Counter({"keep": 90, "drop": 10}))

    assert results["agreement_overall"] == 0.5
    assert "agreement_weighted" not in results


def test_what_the_human_said_instead_is_recorded():
    decisions = [decision("d0", "keep", "drop"), decision("d1", "fix", "drop")]
    results = summarise(decisions, Counter({"drop": 70}))
    assert results["by_verdict"]["drop"]["human_said"] == {"keep": 1, "fix": 1}


def test_a_class_that_was_never_sampled_is_absent_rather_than_zero():
    results = summarise([decision("k0", "keep", "keep")], Counter({"keep": 200, "drop": 100}))
    assert "drop" not in results["by_verdict"]


def test_the_population_sizes_are_recorded_alongside_the_rates():
    results = summarise([decision("k0", "keep", "keep")], Counter({"keep": 200, "fix": 30}))
    assert results["screened_population"] == {"keep": 200, "fix": 30}
    assert results["by_verdict"]["keep"]["population"] == 200


# --- the loop -----------------------------------------------------------


@pytest.fixture
def keys(monkeypatch):
    pressed = []

    def press(sequence):
        pressed.extend(sequence)

    monkeypatch.setattr(audit, "read_key", lambda: pressed.pop(0))
    monkeypatch.setattr(audit, "clear_screen", lambda: None)
    return press


def read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_each_decision_is_written_before_the_next_row(tmp_path, keys):
    sample = [screened_row("q001", "keep"), screened_row("q002", "drop")]
    log = tmp_path / "audit.jsonl"
    keys(["k", " ", "k", " "])  # verdict, dismiss reveal, verdict, dismiss

    decisions = run(sample, {"chunk_q001": "a", "chunk_q002": "b"}, log)

    assert [r["qid"] for r in read(log)] == ["q001", "q002"]
    assert [r["agreed"] for r in decisions] == [True, False]


def test_quitting_at_a_question_records_nothing_for_that_row(tmp_path, keys):
    sample = [screened_row("q001", "keep"), screened_row("q002", "keep")]
    log = tmp_path / "audit.jsonl"
    keys(["k", " ", "q"])

    run(sample, {"chunk_q001": "a", "chunk_q002": "b"}, log)
    assert [r["qid"] for r in read(log)] == ["q001"]


def test_quitting_at_the_reveal_keeps_the_decision_just_made(tmp_path, keys):
    log = tmp_path / "audit.jsonl"
    keys(["d", "q"])

    run([screened_row("q001", "drop")], {"chunk_q001": "a"}, log)
    assert [r["human_verdict"] for r in read(log)] == ["drop"]


def test_already_audited_rows_are_skipped_on_resume(tmp_path, keys):
    sample = [screened_row(f"q{i:03d}", "keep") for i in (1, 2, 3)]
    chunks = {f"chunk_q{i:03d}": "body" for i in (1, 2, 3)}
    log = tmp_path / "audit.jsonl"

    keys(["k", " ", "q"])
    run(sample, chunks, log)
    keys(["d", " ", "k", " "])  # only q002 and q003 remain
    decisions = run(sample, chunks, log)

    assert [r["qid"] for r in read(log)] == ["q001", "q002", "q003"]
    assert [r["agreed"] for r in decisions] == [True, False, True]


def test_unrecognised_keys_are_ignored_at_the_question(tmp_path, keys):
    log = tmp_path / "audit.jsonl"
    keys(["x", "\r", "5", "k", " "])

    run([screened_row("q001", "keep")], {"chunk_q001": "a"}, log)
    assert read(log)[0]["human_verdict"] == "keep"


def test_a_missing_chunk_is_reported_rather_than_crashing(tmp_path, keys):
    log = tmp_path / "audit.jsonl"
    keys(["d", " "])

    run([screened_row("q001", "drop")], {}, log)
    assert read(log)[0]["human_verdict"] == "drop"


# --- the disputed queue -------------------------------------------------


def second(qid, verdict, rule=None, reason="second says so"):
    return {
        "qid": qid,
        "screen_verdict": verdict,
        "screen_rule": rule,
        "screen_reason": reason,
        "screen_model": "claude",
    }


def test_the_queue_is_every_disagreement_plus_the_controls():
    from backend.goldset.audit import build_queue

    screened = [screened_row("q001", "keep"), screened_row("q002", "keep")]
    screened += [screened_row(f"q{i:03d}", "keep") for i in range(3, 20)]
    opinions = [second("q001", "drop"), second("q002", "keep")]
    opinions += [second(f"q{i:03d}", "keep") for i in range(3, 20)]

    queue = build_queue(screened, opinions, controls=5)
    kinds = Counter(r["queue_kind"] for r in queue)

    assert kinds["disputed"] == 1
    assert kinds["control"] == 5
    assert next(r for r in queue if r["queue_kind"] == "disputed")["qid"] == "q001"


def test_a_row_the_second_screener_never_saw_is_skipped_not_counted_as_agreement():
    from backend.goldset.audit import build_queue

    screened = [screened_row("q001", "keep"), screened_row("q002", "drop")]
    queue = build_queue(screened, [second("q001", "keep")], controls=10)

    assert [r["qid"] for r in queue] == ["q001"]  # q002 has no second verdict


def test_only_agreed_keeps_become_controls():
    """An agreed drop is not a control: it is not in the gold set to begin with."""
    from backend.goldset.audit import build_queue

    screened = [screened_row("q001", "drop"), screened_row("q002", "keep")]
    opinions = [second("q001", "drop"), second("q002", "keep")]

    queue = build_queue(screened, opinions, controls=10)
    assert [r["qid"] for r in queue] == ["q002"]
    assert queue[0]["queue_kind"] == "control"


def test_the_queue_is_reproducible_for_a_seed():
    from backend.goldset.audit import build_queue

    screened = [screened_row(f"q{i:03d}", "keep") for i in range(30)]
    opinions = [second(f"q{i:03d}", "keep") for i in range(30)]
    first = build_queue(screened, opinions, controls=5, seed=42)
    again = build_queue(screened, opinions, controls=5, seed=42)
    assert [r["qid"] for r in first] == [r["qid"] for r in again]


def test_the_control_pool_restricts_which_rows_can_be_controls():
    """Controls can be aimed at one slice of the agreed majority."""
    from backend.goldset.audit import build_queue

    screened = [screened_row(f"q{i:03d}", "keep") for i in range(30)]
    opinions = [second(f"q{i:03d}", "keep") for i in range(30)]
    pool = {"q020", "q021", "q022", "q023"}

    queue = build_queue(screened, opinions, controls=2, control_pool=pool)
    controls = [r["qid"] for r in queue if r["queue_kind"] == "control"]

    assert len(controls) == 2
    assert set(controls) <= pool


def test_the_control_pool_never_yields_more_than_it_holds():
    from backend.goldset.audit import build_queue

    screened = [screened_row(f"q{i:03d}", "keep") for i in range(30)]
    opinions = [second(f"q{i:03d}", "keep") for i in range(30)]

    queue = build_queue(screened, opinions, controls=10, control_pool={"q005"})
    assert [r["qid"] for r in queue if r["queue_kind"] == "control"] == ["q005"]


def test_a_different_control_seed_redraws_the_controls():
    """A named control is spent, so re-blinding has to be able to redraw it."""
    from backend.goldset.audit import build_queue

    screened = [screened_row(f"q{i:03d}", "keep") for i in range(30)]
    opinions = [second(f"q{i:03d}", "keep") for i in range(30)]

    def controls_for(control_seed):
        queue = build_queue(screened, opinions, controls=4, control_seed=control_seed)
        return {r["qid"] for r in queue if r["queue_kind"] == "control"}

    assert controls_for(7) != controls_for(99)
    assert controls_for(7) == controls_for(7)


def test_the_control_seed_does_not_disturb_the_disputed_rows():
    """Re-seeding the controls must not quietly change what is being audited."""
    from backend.goldset.audit import build_queue

    screened = [screened_row(f"q{i:03d}", "keep") for i in range(30)]
    screened += [screened_row("q900", "keep")]
    opinions = [second(f"q{i:03d}", "keep") for i in range(30)]
    opinions += [second("q900", "drop")]

    def disputed_for(control_seed):
        queue = build_queue(screened, opinions, controls=4, control_seed=control_seed)
        return {r["qid"] for r in queue if r["queue_kind"] == "disputed"}

    assert disputed_for(7) == disputed_for(99) == {"q900"}


def test_controls_are_not_distinguishable_on_the_blind_screen():
    """Marking a control would restore the anchoring this file exists to prevent."""
    from backend.goldset.audit import build_queue

    screened = [screened_row("q001", "keep"), screened_row("q002", "keep")]
    opinions = [second("q001", "drop"), second("q002", "keep")]
    queue = {r["queue_kind"]: r for r in build_queue(screened, opinions, controls=1)}

    disputed = render_question(queue["disputed"], "body", Progress(total=2))
    control = render_question(queue["control"], "body", Progress(total=2))
    # Same shape, and neither mentions its kind or either verdict.
    for screen in (disputed, control):
        assert "disputed" not in screen and "control" not in screen
        assert "second" not in screen.lower()


def test_the_reveal_shows_both_verdicts():
    row = screened_row("q001", "keep")
    row.update(
        second_verdict="drop",
        second_rule="generic",
        second_reason="answerable by many chunks",
        second_model="claude",
    )
    reveal = render_reveal(row, "drop")

    assert "SECOND OPINION: DROP" in reveal
    assert "answerable by many chunks" in reveal
    assert "DISAGREE" in reveal  # the human split from the screener


def test_the_summary_reports_who_the_human_sided_with():
    decisions = [
        # disputed: screener said keep, second said drop
        {"qid": "q1", "human_verdict": "drop", "screen_verdict": "keep", "agreed": False,
         "second_verdict": "drop", "agreed_with_second": True, "queue_kind": "disputed"},
        {"qid": "q2", "human_verdict": "keep", "screen_verdict": "keep", "agreed": True,
         "second_verdict": "drop", "agreed_with_second": False, "queue_kind": "disputed"},
        {"qid": "q3", "human_verdict": "fix", "screen_verdict": "keep", "agreed": False,
         "second_verdict": "drop", "agreed_with_second": False, "queue_kind": "disputed"},
        # control: both kept
        {"qid": "q4", "human_verdict": "keep", "screen_verdict": "keep", "agreed": True,
         "second_verdict": "keep", "agreed_with_second": True, "queue_kind": "control"},
    ]
    results = summarise(decisions, Counter({"keep": 100}))

    assert results["by_queue_kind"]["disputed"] == {
        "audited": 3,
        "sided_with_screener": 1,
        "sided_with_second": 1,
        "sided_with_neither": 1,
    }
    assert results["by_queue_kind"]["control"]["sided_with_screener"] == 1
    assert results["agreement_with_second"] == 0.5


def test_a_stratified_audit_reports_no_queue_kinds():
    results = summarise([decision("k0", "keep", "keep")], Counter({"keep": 10}))
    assert results["by_queue_kind"] == {}
    assert results["agreement_with_second"] is None


# --- full-coverage blind pass -------------------------------------------


def agreed_row(qid, verdict="keep", **extra):
    """A screened row as the real screener writes it, model stamp included."""
    return screened_row(qid, verdict, screen_model="openai/gpt-oss-20b", **extra)


def coverage(screened, decided=None, second=None, exclusions=None, include_rejections=False):
    from backend.goldset.audit import build_coverage_queue

    opinions = second if second is not None else {
        r["qid"]: {"qid": r["qid"], "screen_verdict": r["screen_verdict"]} for r in screened
    }
    return build_coverage_queue(
        screened, decided or {}, opinions, exclusions, include_rejections=include_rejections
    )


def test_coverage_queues_the_rows_the_screeners_agreed_on():
    queue = coverage([agreed_row("q001"), agreed_row("q002")])
    assert {r["qid"] for r in queue} == {"q001", "q002"}
    assert {r["queue_kind"] for r in queue} == {"coverage"}


def test_coverage_skips_rows_that_already_carry_a_human_verdict():
    """The whole point is to cover what the author has not decided."""
    screened = [agreed_row("q001"), agreed_row("q002")]
    queue = coverage(screened, decided={"q001": {"human_verdict": "keep"}})
    assert [r["qid"] for r in queue] == ["q002"]


def test_coverage_skips_rows_that_are_not_entering_the_gold_set():
    """An agreed drop is not in the gold set, so adjudicating it changes nothing."""
    screened = [agreed_row("q001", "drop"), agreed_row("q002")]
    assert [r["qid"] for r in coverage(screened)] == ["q002"]


def test_coverage_skips_a_row_the_screeners_split_on():
    """A dispute is not accepted on agreement, so it is not this queue's job."""
    screened = [agreed_row("q001"), agreed_row("q002")]
    second = {
        "q001": {"qid": "q001", "screen_verdict": "drop"},
        "q002": {"qid": "q002", "screen_verdict": "keep"},
    }
    assert [r["qid"] for r in coverage(screened, second=second)] == ["q002"]


def test_coverage_skips_a_hand_excluded_row():
    """A row removed by hand must not come back through the audit queue."""
    screened = [agreed_row("q001"), agreed_row("q002")]
    queue = coverage(screened, exclusions={"q001": {"why": "not reproducible"}})
    assert [r["qid"] for r in queue] == ["q002"]


def test_coverage_rows_carry_no_second_opinion():
    """Nothing downstream can render a verdict that is not in the row."""
    queue = coverage([agreed_row("q001")])
    assert not any(key.startswith("second") for key in queue[0])


# --- blindness of the coverage pass -------------------------------------


def test_the_blind_progress_line_withholds_the_agreement_rate():
    """Watching the rate climb is the screener's verdict, delivered in aggregate."""
    revealed = Progress(total=10, agreed=7, decided={f"q{i}" for i in range(9)})
    blind = Progress(total=10, agreed=7, decided={f"q{i}" for i in range(9)}, blind=True)

    assert "agreed" in revealed.line()
    assert "agreed" not in blind.line()
    assert "7" not in blind.line()
    assert "[9/10]" in blind.line()


def test_a_blind_decision_is_stamped_as_one():
    from backend.goldset.audit import decision_row

    row = agreed_row("q001")
    assert decision_row(row, "keep", blind=True)["blind"] is True
    assert decision_row(row, "keep")["blind"] is False


def test_a_blind_run_never_prints_the_reveal(tmp_path, monkeypatch, capsys):
    """The reveal after the keypress is still an anchor for every later row."""
    keys = iter(["k", "k"])
    monkeypatch.setattr(audit, "read_key", lambda: next(keys))
    monkeypatch.setattr(audit, "clear_screen", lambda: None)

    sample = [agreed_row("q001"), screened_row("q002", "drop")]
    run(sample, {"chunk_q001": "body", "chunk_q002": "body"},
        tmp_path / "decisions.jsonl", blind=True)

    out = capsys.readouterr().out
    assert "SCREENER" not in out
    assert "AGREE" not in out


def test_agreement_is_reported_separately_for_each_mode():
    decisions = [
        {"qid": "q001", "screen_verdict": "keep", "human_verdict": "keep",
         "agreed": True, "blind": True},
        {"qid": "q002", "screen_verdict": "keep", "human_verdict": "drop",
         "agreed": False, "blind": True},
        {"qid": "q003", "screen_verdict": "keep", "human_verdict": "keep",
         "agreed": True, "blind": False},
    ]
    modes = summarise(decisions, Counter({"keep": 3}))["by_mode"]

    assert modes["blind"] == {"audited": 2, "agreed": 1, "agreement": 0.5}
    assert modes["revealed"] == {"audited": 1, "agreed": 1, "agreement": 1.0}


def test_coverage_skips_a_row_screened_by_a_stand_in_model():
    """Never validated, so it is not in the gold set and not this queue's job."""
    screened = [
        screened_row("q001", "keep", screen_model="llama-3.1-8b-instant"),
        agreed_row("q002"),
    ]
    assert [r["qid"] for r in coverage(screened)] == ["q002"]


def test_progress_counts_only_the_queue_in_hand():
    """The decision log spans queues; the counter on screen must not."""
    from backend.goldset.audit import restore_progress

    decisions = [
        {"qid": "q001", "agreed": True},
        {"qid": "q999", "agreed": True},  # decided under an earlier queue
    ]
    progress = restore_progress(decisions, total=2, queue={"q001", "q002"})

    assert progress.seen == 1
    assert "[1/2]" in progress.line()


def test_coverage_can_reach_the_drops_nobody_read():
    """A wrongly dropped row is invisible in the finished gold set."""
    screened = [agreed_row("q001"), agreed_row("q002", "drop")]
    assert [r["qid"] for r in coverage(screened)] == ["q001"]
    assert sorted(r["qid"] for r in coverage(screened, include_rejections=True)) == [
        "q001",
        "q002",
    ]


def test_covering_rejections_still_skips_non_judgement_exclusions():
    """A stand-in-model row stays excluded whatever verdict it is given."""
    screened = [
        screened_row("q001", "drop", screen_model="llama-3.1-8b-instant"),
        agreed_row("q002", "drop"),
    ]
    queue = coverage(screened, include_rejections=True)
    assert [r["qid"] for r in queue] == ["q002"]


# --- the queue file carries the verdict; the screen must not ------------


def full_screened_row(qid="q001", verdict="keep"):
    """A row shaped like the ones `build_coverage_queue` actually writes.

    The coverage queue copies the whole screened row, so every screener field
    -- verdict, rule, reason, and the per-rule `screen_scores` -- is present in
    the dict handed to the renderer. What keeps the pass blind is that the
    renderer reads none of them, and that is what these tests pin.
    """
    return screened_row(
        qid,
        verdict,
        rule="generic",
        reason="TELLTALE_REASON a dozen tutorials answer it",
        screen_model="openai/gpt-oss-20b",
        screen_prompt_version="v1",
        screen_scores={
            "generic": {"fired": True, "reason": "TELLTALE_SCORE generic fired"},
            "verbatim": {"fired": False, "reason": "TELLTALE_SCORE run is 2 words"},
            "unanswerable": {"fired": False, "reason": "TELLTALE_SCORE answer is stated"},
            "multi_chunk": {"fired": False, "reason": "TELLTALE_SCORE one chunk is enough"},
        },
        shared_run=2,
        queue_kind="coverage",
    )


def test_no_screener_field_reaches_the_blind_screen():
    """screen_scores is the one the earlier tests missed: it was never populated."""
    screen = render_question(full_screened_row(), "the chunk body", Progress(total=73, blind=True))

    assert "TELLTALE_SCORE" not in screen
    assert "TELLTALE_REASON" not in screen
    assert "generic" not in screen
    assert "gpt-oss-20b" not in screen
    assert "coverage" not in screen


def test_the_blind_screen_is_identical_whatever_the_scores_say():
    """The strongest form: the bytes cannot vary with anything the screener wrote."""
    kept = full_screened_row("q001", "keep")
    dropped = full_screened_row("q001", "drop")
    dropped["screen_scores"]["generic"]["fired"] = False
    dropped["screen_scores"]["unanswerable"] = {"fired": True, "reason": "not stated"}
    dropped["shared_run"] = 9

    assert render_question(kept, "body", Progress(total=1)) == render_question(
        dropped, "body", Progress(total=1)
    )


def test_a_blind_run_leaks_no_screener_field_to_the_terminal(tmp_path, monkeypatch, capsys):
    """End to end, through the loop that actually writes decisions."""
    keys = iter(["k", "k"])
    monkeypatch.setattr(audit, "read_key", lambda: next(keys))
    monkeypatch.setattr(audit, "clear_screen", lambda: None)

    sample = [full_screened_row("q001", "keep"), full_screened_row("q002", "drop")]
    run(sample, {"chunk_q001": "body", "chunk_q002": "body"},
        tmp_path / "decisions.jsonl", blind=True)

    out = capsys.readouterr().out
    assert "TELLTALE_SCORE" not in out
    assert "TELLTALE_REASON" not in out
    assert "SCREENER" not in out
    assert "gpt-oss-20b" not in out
