"""Chunk id stability.

`chunk_id` is the join key between the corpus and the gold set. If an id moves
between ingests, every gold label silently points at the wrong text and the
published numbers become fiction. These tests exist to make that failure loud.
"""

import pytest

from backend.ingest import build_chunks, make_chunk_id, slugify_source


def test_id_format_matches_the_evaluation_spec():
    # docs/02_EVALUATION_SPEC.md §1 uses "nadra_ch_042" for NADRA.txt chunk 42.
    assert make_chunk_id("NADRA.txt", 42) == "nadra_ch_042"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("NADRA.txt", "nadra"),
        ("nadra.txt", "nadra"),
        ("Guide Notes.md", "guide_notes"),
        ("guide-notes.md", "guide_notes"),
        ("A  Very__Odd   Name!.pdf", "a_very_odd_name"),
        ("2024_report.pdf", "2024_report"),
        ("trailing-.txt", "trailing"),
    ],
)
def test_slug_is_case_and_punctuation_insensitive(filename, expected):
    assert slugify_source(filename) == expected


@pytest.mark.parametrize("filename", ["...txt", "___.md", "!!!.pdf"])
def test_filename_with_no_alphanumerics_raises(filename):
    with pytest.raises(ValueError, match="alphanumeric"):
        slugify_source(filename)


def test_index_is_zero_padded_to_three_but_not_truncated():
    assert make_chunk_id("doc.txt", 0) == "doc_ch_000"
    assert make_chunk_id("doc.txt", 7) == "doc_ch_007"
    assert make_chunk_id("doc.txt", 999) == "doc_ch_999"
    # Padding is a minimum width, never a cap: a long document must not wrap.
    assert make_chunk_id("doc.txt", 1000) == "doc_ch_1000"


def test_negative_index_raises():
    with pytest.raises(ValueError):
        make_chunk_id("doc.txt", -1)


def test_reingesting_the_same_corpus_produces_identical_ids(corpus_dir, tokenizer):
    first = build_chunks(corpus_dir, "demo", tokenizer, size=5, overlap=2)
    second = build_chunks(corpus_dir, "demo", tokenizer, size=5, overlap=2)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_ids_are_unique_across_the_corpus(corpus_dir, tokenizer):
    chunks = build_chunks(corpus_dir, "demo", tokenizer, size=5, overlap=2)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_ids_do_not_depend_on_corpus_id(corpus_dir, tokenizer):
    """The same file ingested into a different corpus keeps its ids."""
    demo = build_chunks(corpus_dir, "demo", tokenizer, size=5, overlap=2)
    upload = build_chunks(corpus_dir, "upload-abc123", tokenizer, size=5, overlap=2)
    assert [c.chunk_id for c in demo] == [c.chunk_id for c in upload]


def test_adding_a_document_does_not_move_existing_ids(corpus_dir, tokenizer):
    """Ids are per-document, so a new file cannot renumber the old ones.

    A corpus-wide running counter would fail this, and would invalidate the
    whole gold set the first time a document was added.
    """
    before = {c.chunk_id: c.text for c in build_chunks(corpus_dir, "demo", tokenizer, 5, 2)}

    # "AAA.txt" sorts first, so a corpus-wide counter would shift everything.
    (corpus_dir / "AAA.txt").write_text("x y z", encoding="utf-8")
    after = {c.chunk_id: c.text for c in build_chunks(corpus_dir, "demo", tokenizer, 5, 2)}

    assert before.items() <= after.items()
    assert "aaa_ch_000" in after


def test_chunk_ids_are_grouped_by_source_document(corpus_dir, tokenizer):
    chunks = build_chunks(corpus_dir, "demo", tokenizer, size=5, overlap=2)
    by_doc: dict[str, list[str]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.source_doc, []).append(chunk.chunk_id)

    # NADRA.txt: 10 tokens -> 3 windows. Guide Notes.md: 7 tokens -> 2 windows.
    assert by_doc["NADRA.txt"] == ["nadra_ch_000", "nadra_ch_001", "nadra_ch_002"]
    assert by_doc["Guide Notes.md"] == ["guide_notes_ch_000", "guide_notes_ch_001"]
    assert "ignored.csv" not in by_doc  # unsupported suffixes are skipped


def test_slug_collision_raises_instead_of_silently_overwriting(tmp_path, tokenizer):
    (tmp_path / "My Doc.txt").write_text("a b c", encoding="utf-8")
    (tmp_path / "my-doc.md").write_text("d e f", encoding="utf-8")

    with pytest.raises(ValueError, match="collision"):
        build_chunks(tmp_path, "demo", tokenizer, size=5, overlap=2)


def test_crlf_and_lf_copies_of_a_document_chunk_identically(tmp_path, tokenizer):
    """Line endings must not change chunk text, or ids would lie across platforms."""
    lf = tmp_path / "lf"
    crlf = tmp_path / "crlf"
    lf.mkdir()
    crlf.mkdir()
    body = "alpha beta\ngamma delta\n\n\n\nepsilon"
    (lf / "doc.txt").write_text(body, encoding="utf-8", newline="")
    (crlf / "doc.txt").write_text(body.replace("\n", "\r\n"), encoding="utf-8", newline="")

    lf_chunks = build_chunks(lf, "demo", tokenizer, size=5, overlap=2)
    crlf_chunks = build_chunks(crlf, "demo", tokenizer, size=5, overlap=2)

    assert [(c.chunk_id, c.text) for c in lf_chunks] == [
        (c.chunk_id, c.text) for c in crlf_chunks
    ]
