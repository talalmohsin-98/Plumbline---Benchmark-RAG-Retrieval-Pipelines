"""Chunking tests.

Every expectation below is computed by hand from the fixture, not captured
from the implementation's output. A test that records current behaviour would
pass just as happily against broken windowing.
"""

from itertools import pairwise

import pytest

from backend.ingest import chunk_document, window_token_spans

# --- window_token_spans -------------------------------------------------
#
# 10 tokens, size 5, overlap 2 -> stride 3.
#   start 0: [0,5)   end != 10, advance to 3
#   start 3: [3,8)   end != 10, advance to 6
#   start 6: [6,10)  end == 10, stop
HAND_COMPUTED = [
    # (token_count, size, overlap, expected_windows)
    (10, 5, 2, [(0, 5), (3, 8), (6, 10)]),
    (5, 5, 2, [(0, 5)]),  # exactly one full window
    (3, 5, 2, [(0, 3)]),  # shorter than one window
    (6, 5, 2, [(0, 5), (3, 6)]),  # tail of 3 new tokens, > overlap
    (10, 5, 0, [(0, 5), (5, 10)]),  # no overlap -> plain partition
    (0, 5, 2, []),
    (1, 5, 2, [(0, 1)]),
]


@pytest.mark.parametrize(("count", "size", "overlap", "expected"), HAND_COMPUTED)
def test_windows_match_hand_computed_fixture(count, size, overlap, expected):
    assert window_token_spans(count, size, overlap) == expected


@pytest.mark.parametrize("count", [1, 2, 7, 10, 63, 64, 65, 447, 448, 449, 512, 513, 960, 1000])
@pytest.mark.parametrize(("size", "overlap"), [(5, 2), (512, 64), (10, 9), (8, 0)])
def test_windows_cover_every_token_without_gaps(count, size, overlap):
    windows = window_token_spans(count, size, overlap)
    assert windows[0][0] == 0
    assert windows[-1][1] == count
    for (_, prev_end), (next_start, _) in pairwise(windows):
        # Consecutive windows must touch or overlap; a gap would drop tokens.
        assert next_start <= prev_end


@pytest.mark.parametrize("count", [1, 6, 7, 65, 512, 513, 576, 960, 961, 1408, 1409])
@pytest.mark.parametrize(("size", "overlap"), [(5, 2), (512, 64), (10, 9)])
def test_no_window_is_pure_overlap(count, size, overlap):
    """The documented invariant: every window adds more than `overlap` tokens.

    If this ever fails, the last chunk of some document is fully contained in
    its predecessor — a duplicate that would pollute retrieval and the gold set.
    """
    windows = window_token_spans(count, size, overlap)
    for start, end in windows:
        assert end - start <= size
    for start, end in windows[1:]:
        assert end - start > overlap


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (5, 5), (5, 6), (5, -1)],
)
def test_invalid_window_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        window_token_spans(10, size, overlap)


# --- chunk_document -----------------------------------------------------
#
# "w0 w1 ... w9": each token is 2 chars, separated by 1 space, so token i
# occupies characters [3i, 3i+2).
TEXT = " ".join(f"w{i}" for i in range(10))


def test_chunk_text_is_sliced_from_the_original_string(tokenizer):
    chunks = chunk_document(TEXT, "NADRA.txt", "demo", tokenizer, size=5, overlap=2)

    assert [c.text for c in chunks] == [
        "w0 w1 w2 w3 w4",  # tokens [0,5)  -> chars [0, 14)
        "w3 w4 w5 w6 w7",  # tokens [3,8)  -> chars [9, 23)
        "w6 w7 w8 w9",  # tokens [6,10) -> chars [18, 29)
    ]
    assert [(c.start_char, c.end_char) for c in chunks] == [(0, 14), (9, 23), (18, 29)]
    # Slices, not reconstructions: each chunk must appear verbatim in the source.
    for chunk in chunks:
        assert TEXT[chunk.start_char : chunk.end_char] == chunk.text


def test_chunk_indices_are_dense_and_zero_based(tokenizer):
    chunks = chunk_document(TEXT, "NADRA.txt", "demo", tokenizer, size=5, overlap=2)
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_overlap_actually_repeats_tokens(tokenizer):
    chunks = chunk_document(TEXT, "NADRA.txt", "demo", tokenizer, size=5, overlap=2)
    assert chunks[0].text.endswith("w3 w4")
    assert chunks[1].text.startswith("w3 w4")


def test_empty_document_produces_no_chunks(tokenizer):
    assert chunk_document("   \n  ", "Empty.txt", "demo", tokenizer, size=5, overlap=2) == []


def test_document_shorter_than_one_window_is_one_chunk(tokenizer):
    chunks = chunk_document("a b c", "Short.txt", "demo", tokenizer, size=5, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].text == "a b c"
