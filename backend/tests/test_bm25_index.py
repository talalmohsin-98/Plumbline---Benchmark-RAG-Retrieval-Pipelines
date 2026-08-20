"""BM25 tokenizer and index tests. No database: `build_index` takes chunks directly."""

import pytest

from backend.retrieval.bm25_index import build_index, tokenize
from backend.retrieval.dense_store import StoredChunk


def chunk(chunk_id, text):
    return StoredChunk(
        chunk_id=chunk_id,
        corpus_id="test",
        source_doc="doc.md",
        chunk_index=0,
        text=text,
    )


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("FastAPI's HTTPException, raised.") == [
        "fastapi",
        "s",
        "httpexception",
        "raised",
    ]


def test_tokenize_strips_punctuation_that_whitespace_splitting_keeps():
    # The actual reason this tokenizer is not `str.split()`. Documentation is
    # dense in trailing punctuation and a whitespace split glues it on, so
    # BM25 scores "raised." and "raised" as unrelated terms.
    text = "FastAPI's HTTPException, raised."
    assert text.lower().split() == ["fastapi's", "httpexception,", "raised."]
    assert tokenize(text) == ["fastapi", "s", "httpexception", "raised"]


def test_tokenize_splits_the_non_breaking_hyphen_like_a_plain_one():
    # U+2011 NON-BREAKING HYPHEN is dash punctuation, not whitespace, so a
    # whitespace split keeps "auto<U+2011>reload" whole while the plain-hyphen
    # spelling used in the questions splits. Both must tokenize alike or the
    # question never matches its own gold chunk.
    #
    # Written as an escape rather than the literal character: an invisible or
    # near-invisible codepoint in source is unreadable, and ruff rejects it as
    # ambiguous for exactly that reason.
    nbhyphen = "auto\u2011reload"
    assert nbhyphen.split() == [nbhyphen]
    assert tokenize(nbhyphen) == tokenize("auto-reload") == ["auto", "reload"]


def test_narrow_no_break_space_is_whitespace_and_needs_no_special_handling():
    # Recorded because this file's first draft claimed the opposite and built
    # the tokenizer's justification on it. U+202F appears in 25 of the 115
    # gold questions; it is Unicode category Zs, so `str.split()` already
    # handles it. `\w+` agrees. Neither is doing anything clever here.
    phrase = "a\u202fyield\u202fstatement"
    assert phrase.split() == ["a", "yield", "statement"]
    assert tokenize(phrase) == ["a", "yield", "statement"]


def test_tokenize_keeps_digits_and_underscores():
    assert tokenize("chunk_id 512 v1.5") == ["chunk_id", "512", "v1", "5"]


def test_tokenize_of_pure_punctuation_is_empty():
    assert tokenize("?!  --") == []


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------


@pytest.fixture
def index():
    return build_index(
        [
            chunk("a_ch_000", "install python-multipart to handle form data"),
            chunk("b_ch_000", "form data requires python-multipart installed"),
            chunk("c_ch_000", "background tasks run after the response is sent"),
        ]
    )


def test_search_finds_the_lexically_matching_chunks_first(index):
    # Both a and b contain every query term; c contains none. The two matches
    # come first and the non-match last, which is the property that matters.
    hits = index.search("python-multipart form", k=3)
    assert set([chunk_id for chunk_id, _ in hits][:2]) == {"a_ch_000", "b_ch_000"}
    assert hits[2][0] == "c_ch_000"


def test_the_shorter_of_two_equally_matching_chunks_ranks_higher(index):
    # b ("form data requires python-multipart installed", 6 words) outranks
    # a ("install python-multipart to handle form data", 7 words) on the same
    # query terms. That is BM25's length normalisation (b=0.75) doing its job,
    # not a tie-break -- asserted because the obvious expectation is corpus
    # order, and getting this backwards once is how the fixture was written.
    hits = index.search("python-multipart form", k=3)
    assert next(chunk_id for chunk_id, _ in hits) == "b_ch_000"
    assert dict(hits)["b_ch_000"] > dict(hits)["a_ch_000"]


def test_a_chunk_sharing_no_query_term_scores_exactly_zero(index):
    scores = dict(index.search("background tasks", k=3))
    assert scores["c_ch_000"] > 0
    assert scores["a_ch_000"] == 0.0
    assert scores["b_ch_000"] == 0.0


def test_zero_scored_ties_break_on_chunk_id_so_the_order_is_stable(index):
    # Hundreds of chunks tie at 0.0 on a real corpus. If their order were not
    # deterministic, the top-50 handed to fusion could differ between runs on
    # identical data, and so could recall@10.
    hits = index.search("background tasks", k=3)
    assert [chunk_id for chunk_id, _ in hits] == ["c_ch_000", "a_ch_000", "b_ch_000"]
    assert index.search("background tasks", k=3) == hits


def test_search_respects_k(index):
    assert len(index.search("form data", k=1)) == 1
    assert len(index.search("form data", k=2)) == 2


def test_a_query_of_pure_punctuation_returns_nothing(index):
    # Not "returns an arbitrary k chunks". Every chunk would score 0.0 and the
    # lane would hand fusion 50 chunks it has no evidence for.
    assert index.search("???", k=10) == []


def test_index_rows_line_up_with_chunk_ids_regardless_of_input_order():
    forward = build_index([chunk("a_ch_000", "alpha text"), chunk("b_ch_000", "beta text")])
    backward = build_index([chunk("b_ch_000", "beta text"), chunk("a_ch_000", "alpha text")])
    assert forward.chunk_ids == backward.chunk_ids == ["a_ch_000", "b_ch_000"]
    assert forward.search("alpha", k=2) == backward.search("alpha", k=2)


def test_building_over_an_empty_corpus_raises(index):
    with pytest.raises(ValueError, match="empty corpus"):
        build_index([])


def test_k_must_be_positive(index):
    with pytest.raises(ValueError, match="positive"):
        index.search("form", k=0)
