"""Hard-negative mining tests. No database, no model files, no network.

The tests that matter here are the negative ones: that a test question cannot
reach the retriever, that a row's own gold chunks cannot be sampled against it,
and that a pair file carrying anything from the test split is refused before it
is written. Those three properties are the whole reason lane 6's number means
anything, so they are pinned three different ways rather than trusted once.
"""

import json

import pytest

from training.mine_negatives import (
    NEGATIVES_PER_QUESTION,
    LeakGuardError,
    Pair,
    TrainOnlyCorpus,
    assert_pairs_are_clean,
    assert_train_only,
    build_report,
    choose_positive,
    load_rows,
    mine_row,
    normalise,
)


def fused(n: int = 20, prefix: str = "c") -> list[tuple[str, float]]:
    """A fused list of `n` chunks, ranks 1..n, scores descending and hand-countable."""
    return [(f"{prefix}{i:02d}", 1.0 / i) for i in range(1, n + 1)]


def row(qid: str = "q001", gold: list[str] | None = None, question: str = "how do I X?") -> dict:
    return {"qid": qid, "question": question, "gold_chunk_ids": gold or ["c01"]}


# --- gold exclusion -------------------------------------------------------


def test_every_gold_chunk_of_the_row_is_excluded_not_just_the_positive():
    """The multi-label case. Five golds go in, none of the five comes back out.

    This is the failure the whole exclusion step exists to prevent: sampling a
    row's second gold chunk as a negative puts the same passage in the batch
    twice with opposite labels.
    """
    gold = ["c01", "c02", "c03", "c04", "c05"]
    outcome = mine_row(row(gold=gold), fused(), depth=20, negatives_per_question=4, seed=42)

    assert set(outcome.negatives).isdisjoint(gold)
    assert outcome.pool_size == 15  # 20 candidates - 5 gold
    assert outcome.gold_count == 5


def test_gold_outside_the_mined_depth_is_still_excluded():
    """A gold chunk at rank 25 is not in the pool anyway, and must not crash the count."""
    outcome = mine_row(
        row(gold=["c01", "c25"]), fused(30), depth=20, negatives_per_question=4, seed=42
    )

    assert "c25" not in outcome.negatives
    assert outcome.pool_size == 19  # top-20 minus c01; c25 was never a candidate
    assert outcome.gold_rank == 1


def test_a_row_whose_gold_never_surfaced_still_mines_negatives():
    """The hardest rows. 71 of 80 real rows had gold in the top 20; these are the other 9."""
    outcome = mine_row(row(gold=["c99"]), fused(), depth=20, negatives_per_question=4, seed=42)

    assert outcome.gold_rank is None
    assert len(outcome.negatives) == 4
    assert outcome.pool_size == 20


# --- the sampling rule ----------------------------------------------------


def test_exactly_four_negatives_and_no_duplicates():
    outcome = mine_row(row(), fused(), depth=20, negatives_per_question=4, seed=42)

    assert len(outcome.negatives) == NEGATIVES_PER_QUESTION == 4
    assert len(set(outcome.negatives)) == 4


def test_the_draw_is_deterministic_for_a_seed():
    first = mine_row(row(), fused(), depth=20, negatives_per_question=4, seed=42)
    second = mine_row(row(), fused(), depth=20, negatives_per_question=4, seed=42)

    assert first.negatives == second.negatives


def test_each_row_is_seeded_independently_of_every_other_row():
    """Adding a question must not reshuffle the negatives of the ones after it.

    A single RNG consumed in row order would make every row's draw depend on
    how many rows preceded it, so inserting one question at q005 would rewrite
    the whole pair file and make it incomparable with the previous one.
    """
    alone = mine_row(row("q050"), fused(), depth=20, negatives_per_question=4, seed=42)
    for other in ("q001", "q002", "q003"):
        mine_row(row(other), fused(), depth=20, negatives_per_question=4, seed=42)
    after_others = mine_row(row("q050"), fused(), depth=20, negatives_per_question=4, seed=42)

    assert alone.negatives == after_others.negatives


def test_two_different_questions_do_not_get_the_same_draw():
    """Seeded per qid, so the draws differ; a shared seed would be a bug worth catching."""
    a = mine_row(row("q001"), fused(), depth=20, negatives_per_question=4, seed=42)
    b = mine_row(row("q002"), fused(), depth=20, negatives_per_question=4, seed=42)

    assert a.negatives != b.negatives


def test_negatives_are_drawn_from_the_whole_band_not_only_the_top():
    """Pre-registered decision 4: uniform over the top-20 remainder.

    An implementation that took the four highest-ranked non-gold chunks would
    return ranks [2,3,4,5] every time. Across 40 seeds a uniform sampler must
    reach past rank 10; a top-k sampler never can.
    """
    seen = set()
    for seed in range(40):
        outcome = mine_row(row(), fused(), depth=20, negatives_per_question=4, seed=seed)
        seen.update(outcome.negative_ranks)

    assert max(seen) > 10
    assert seen <= set(range(2, 21))  # rank 1 is the gold chunk c01


def test_mining_depth_is_respected():
    outcome = mine_row(row(), fused(50), depth=20, negatives_per_question=4, seed=42)

    assert all(rank <= 20 for rank in outcome.negative_ranks)


# --- short rows -----------------------------------------------------------


def test_a_row_with_too_few_candidates_yields_what_it_has_and_reports_the_shortfall():
    """Three candidates, one of them gold: two negatives available, not four."""
    outcome = mine_row(
        row(gold=["c01"]), fused(3), depth=20, negatives_per_question=4, seed=42
    )

    assert len(outcome.negatives) == 2
    assert outcome.shortfall == 2


def test_a_row_whose_candidates_are_all_gold_yields_no_negatives():
    outcome = mine_row(
        row(gold=["c01", "c02"]), fused(2), depth=20, negatives_per_question=4, seed=42
    )

    assert outcome.negatives == []
    assert outcome.shortfall == 4


# --- the positive ---------------------------------------------------------


def test_the_positive_is_the_lowest_gold_chunk_id_not_the_best_ranked():
    """Deterministic and independent of the retriever's current ranking.

    Picking the highest-ranked gold instead would make the pair file a function
    of `rrf_k`: change the fusion constant and the training data changes with
    it, silently.
    """
    assert choose_positive(["c09", "c02", "c07"]) == "c02"
    assert choose_positive(["c02"]) == "c02"


# --- the guard ------------------------------------------------------------


class StubCorpus:
    corpus_id = "stub"

    def __init__(self):
        self.bm25_calls: list[str] = []
        self.embed_calls: list[str] = []
        self.texts = {"c01": "text of c01"}
        self.sources = {"c01": "doc.md"}

    def bm25_search(self, query, k):
        self.bm25_calls.append(query)
        return [("c01", 1.0)]

    def embed_query(self, text, *, prefix=True):
        self.embed_calls.append(text)
        return [0.1, 0.2]

    def dense_search_vector(self, vector, k):
        return [("c01", 0.9)]

    def text_of(self, chunk_id):
        return self.texts.get(chunk_id, "")


def guarded() -> tuple[TrainOnlyCorpus, StubCorpus]:
    inner = StubCorpus()
    return TrainOnlyCorpus(inner, allowed=["train question"], forbidden=["TEST question"]), inner


def test_a_test_question_cannot_be_retrieved_for():
    proxy, inner = guarded()

    with pytest.raises(LeakGuardError, match="TEST question"):
        proxy.bm25_search("TEST question", 20)

    assert inner.bm25_calls == []  # raised before the retriever saw it


def test_a_test_question_cannot_be_embedded():
    proxy, inner = guarded()

    with pytest.raises(LeakGuardError, match="TEST question"):
        proxy.embed_query("TEST question")

    assert inner.embed_calls == []


def test_the_guard_is_not_fooled_by_case_or_whitespace():
    """`==` on the raw string is too weak: same question, different bytes."""
    proxy, _ = guarded()

    with pytest.raises(LeakGuardError):
        proxy.bm25_search("  test   QUESTION  ", 20)


def test_a_question_in_neither_split_is_refused_too():
    """Not just a test-set check: mining reads train.jsonl and nothing else."""
    proxy, _ = guarded()

    with pytest.raises(LeakGuardError, match="neither split"):
        proxy.bm25_search("a question from nowhere", 20)


def test_a_train_question_passes_through_and_is_recorded():
    proxy, inner = guarded()

    proxy.bm25_search("train question", 20)
    proxy.embed_query("train question")

    assert inner.bm25_calls == ["train question"]
    assert proxy.queries_seen == ["train question", "train question"]


def test_the_warmup_string_is_allowed_by_name():
    """`Corpus.warm` embeds the literal "warmup"; it is not a question."""
    proxy, inner = guarded()

    proxy.embed_query("warmup")

    assert inner.embed_calls == ["warmup"]
    assert proxy.queries_seen == []


def test_unintercepted_attributes_pass_straight_through():
    proxy, inner = guarded()

    assert proxy.text_of("c01") == "text of c01"
    assert proxy.corpus_id == "stub"
    assert proxy.dense_search_vector([0.1], 5) == [("c01", 0.9)]


# --- the after-the-fact checks -------------------------------------------


def test_assert_train_only_catches_a_silently_skipped_row():
    """The one thing the guard cannot see: a row that was never issued at all."""
    proxy, _ = guarded()
    proxy.queries_seen = ["train question"]
    train = [row("q001", question="train question"), row("q002", question="skipped question")]

    with pytest.raises(LeakGuardError, match="never retrieved"):
        assert_train_only(proxy, train, [])


def test_assert_train_only_passes_when_every_row_was_issued():
    proxy, _ = guarded()
    proxy.queries_seen = ["train question"]

    assert_train_only(proxy, [row("q001", question="train question")], [])


def pair(qid="q001", question="train question", chunk_id="c01", label=0.0, text="body") -> Pair:
    return Pair(
        qid=qid,
        question=question,
        chunk_id=chunk_id,
        text=text,
        source_doc="doc.md",
        label=label,
        fused_rank=3,
    )


def test_a_pair_file_carrying_a_test_qid_is_refused():
    train = [row("q001", question="train question")]
    test = [row("q900", question="TEST question")]

    with pytest.raises(LeakGuardError, match="test qids"):
        assert_pairs_are_clean([pair(qid="q900")], train, test)


def test_a_pair_file_carrying_test_question_text_is_refused():
    """Catches the case where the qid was rewritten but the question came along."""
    train = [row("q001", question="train question")]
    test = [row("q900", question="TEST question")]

    with pytest.raises(LeakGuardError, match="test question text"):
        assert_pairs_are_clean([pair(qid="q001", question="test question")], train, test)


def test_a_pair_file_with_unresolved_chunk_text_is_refused():
    """An empty body means the chunk id did not resolve -- usually the wrong corpus."""
    train = [row("q001", question="train question")]

    with pytest.raises(LeakGuardError, match="empty chunk text"):
        assert_pairs_are_clean([pair(text="   ")], train, [])


def test_a_clean_pair_file_passes():
    train = [row("q001", question="train question")]
    test = [row("q900", question="TEST question")]

    assert_pairs_are_clean([pair(), pair(label=1.0)], train, test)


# --- the reported diagnostic ---------------------------------------------


def test_test_gold_negatives_are_counted_and_not_removed():
    """Pre-registered decision 5, in numbers.

    c07 is gold for two test questions and was mined as a negative against one
    train question. It stays in the pairs; the report says so. Filtering it
    would mean reading the test answer key to shape training.
    """
    pairs = [
        pair(qid="q001", chunk_id="c07", label=0.0),
        pair(qid="q001", chunk_id="c08", label=0.0),
        pair(qid="q001", chunk_id="c01", label=1.0),
    ]
    outcome = mine_row(row(gold=["c01"]), fused(), depth=20, negatives_per_question=4, seed=42)
    test = [
        row("q900", gold=["c07"], question="test a"),
        row("q901", gold=["c07", "c99"], question="test b"),
    ]

    report = build_report(
        pairs, [outcome], test, depth=20, negatives_per_question=4, seed=42,
        corpus_id="stub", chunks_in_corpus=20,
    )
    collisions = report["test_gold_chunks_mined_as_negatives"]

    assert collisions["count"] == 1
    assert collisions["distinct_chunks"] == 1
    assert collisions["distinct_test_questions_affected"] == 2
    assert collisions["test_qids_affected"] == ["q900", "q901"]
    assert collisions["fraction"] == 0.5  # 1 of 2 negatives
    # and the offending pair is still in the file
    assert any(p.chunk_id == "c07" and p.label == 0.0 for p in pairs)


def test_the_report_counts_pairs_and_shortfalls_from_the_outcomes():
    outcomes = [
        mine_row(row("q001"), fused(), depth=20, negatives_per_question=4, seed=1),
        mine_row(row("q002", gold=["c01"]), fused(3), depth=20, negatives_per_question=4, seed=1),
    ]
    pairs = [pair(label=1.0), pair(), pair(), pair(), pair(), pair(label=1.0), pair(), pair()]

    report = build_report(
        pairs, outcomes, [], depth=20, negatives_per_question=4, seed=42,
        corpus_id="stub", chunks_in_corpus=20,
    )

    assert report["pairs"] == {"total": 8, "positives": 2, "negatives": 6, "ratio": 3.0}
    assert report["negatives_per_positive"]["rows_below_target"] == 1
    assert report["negatives_per_positive"]["rows_below_target_qids"] == ["q002"]
    assert report["negatives_per_positive"]["distribution"] == {"2": 1, "4": 1}
    assert report["positive_in_mined_depth"] == {"rows": 2, "of": 2, "median_gold_rank": 1.0}


# --- loading --------------------------------------------------------------


def test_load_rows_refuses_a_row_with_no_gold(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"qid": "q001", "question": "q", "gold_chunk_ids": []}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no gold_chunk_ids"):
        load_rows(path)


def test_normalise_collapses_case_and_whitespace():
    assert normalise("  How   do I X? ") == normalise("how do i x?")
