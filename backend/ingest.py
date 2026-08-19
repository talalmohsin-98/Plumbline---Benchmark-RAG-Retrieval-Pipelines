"""Corpus ingestion: load documents, chunk, embed, and store with stable chunk ids."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

from backend.config import Settings, get_settings
from backend.retrieval.dense_store import ChunkRecord

if TYPE_CHECKING:  # heavy imports stay out of the module import path
    from sentence_transformers import SentenceTransformer

# .mdx is markdown with JSX components; the LangChain and LangGraph docs are
# written in it. The components are left in the text rather than stripped —
# silently rewriting corpus content would break the manifest's content hash.
SUPPORTED_SUFFIXES = frozenset({".pdf", ".txt", ".md", ".mdx"})

# A chunk id is <source-slug>_ch_<index>, e.g. NADRA.txt chunk 42 -> nadra_ch_042.
# The gold set keys off these ids, so the format is a contract, not a detail.
CHUNK_ID_INDEX_WIDTH = 3
_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


# --------------------------------------------------------------------------
# Chunk identity
# --------------------------------------------------------------------------


def slugify_source(source_doc: str) -> str:
    """Reduce a corpus-relative source path to the stable slug used in its ids.

    The whole path participates, not just the filename: a real corpus has
    `fastapi/tutorial/index.md` and `langgraph/index.md`, and a stem-only slug
    would collide on every `index` in the tree. For a flat corpus this is
    unchanged — `NADRA.txt` still slugs to `nadra`.
    """
    relative = PurePosixPath(source_doc.replace("\\", "/"))
    slug = _NON_SLUG_CHARS.sub("_", relative.with_suffix("").as_posix().lower()).strip("_")
    if not slug:
        raise ValueError(f"source path {source_doc!r} has no alphanumeric characters")
    return slug


def make_chunk_id(source_doc: str, chunk_index: int) -> str:
    """Build the deterministic id for one chunk of one document.

    Depends only on the filename and the position of the chunk within that
    file — never on ingestion order, wall-clock time, corpus membership, or
    embedding output. Re-ingesting the same bytes must produce the same ids,
    because every gold label points at one.
    """
    if chunk_index < 0:
        raise ValueError("chunk_index cannot be negative")
    return f"{slugify_source(source_doc)}_ch_{chunk_index:0{CHUNK_ID_INDEX_WIDTH}d}"


@dataclass(frozen=True)
class Chunk:
    """A chunk before embedding: identity, provenance, and the original text."""

    chunk_id: str
    corpus_id: str
    source_doc: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int

    @property
    def char_length(self) -> int:
        return self.end_char - self.start_char


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------


class SpanTokenizer(Protocol):
    """Anything that can report the character span of each token in a string."""

    def token_spans(self, text: str) -> list[tuple[int, int]]: ...


class HFSpanTokenizer:
    """Character spans from the embedding model's own fast tokenizer.

    Chunking uses the *model's* tokenizer so that "512 tokens" means 512 tokens
    as bge-small will count them. Measuring in words or characters would let
    chunks silently exceed the model's context and be truncated at embed time.
    """

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self._tokenizer.is_fast:
            raise RuntimeError(
                f"{model_name} has no fast tokenizer; offset mapping is required "
                "to slice chunks out of the original text"
            )

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            # The whole document is tokenized at once purely to get offsets;
            # nothing this long is ever fed to the model. Without verbose=False
            # transformers warns about exceeding the 512-token limit on every
            # document, which is noise that hides real problems.
            verbose=False,
        )
        # Zero-width spans are special/control tokens; they carry no source text.
        return [(start, end) for start, end in encoded["offset_mapping"] if end > start]


@lru_cache(maxsize=1)
def get_tokenizer(model_name: str) -> HFSpanTokenizer:
    """Return the process-wide tokenizer, loaded once.

    Cheap enough to load on demand and cached forever, which is what --dry-run
    needs: chunk boundaries without touching the embedding model.
    """
    return HFSpanTokenizer(model_name)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def window_token_spans(
    token_count: int,
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Slice `token_count` tokens into overlapping [start, end) windows.

    Pure and index-only, so it can be verified by hand against a fixture.

    The loop stops as soon as a window reaches the end of the document. That
    guarantees every window contributes more than `overlap` new tokens: a
    window is only ever opened at `start` when the previous one ended before
    the document did, i.e. `start - stride + size < token_count`, which
    rearranges to `token_count - start > overlap`. So there is no degenerate
    tail chunk whose content is already fully covered by its predecessor, and
    no need to special-case one away.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    if token_count <= 0:
        return []

    stride = size - overlap
    windows: list[tuple[int, int]] = []
    start = 0
    while start < token_count:
        end = min(start + size, token_count)
        windows.append((start, end))
        if end == token_count:
            break
        start += stride
    return windows


def chunk_document(
    text: str,
    source_doc: str,
    corpus_id: str,
    tokenizer: SpanTokenizer,
    size: int,
    overlap: int,
) -> list[Chunk]:
    """Split one document into overlapping chunks of `size` tokens.

    Chunk text is sliced out of the *original* string using token offsets
    rather than decoded back from token ids. Decoding an uncased WordPiece
    vocabulary would lower-case the text, normalise its whitespace, and turn
    anything out-of-vocabulary into [UNK] — which would corrupt BM25, the
    chunks shown in the inspector, and the passages the reranker is trained on.
    """
    spans = tokenizer.token_spans(text)
    windows = window_token_spans(len(spans), size, overlap)

    chunks: list[Chunk] = []
    for index, (first, last) in enumerate(windows):
        start_char = spans[first][0]
        end_char = spans[last - 1][1]
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(source_doc, index),
                corpus_id=corpus_id,
                source_doc=source_doc,
                chunk_index=index,
                text=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
            )
        )
    return chunks


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def normalise_text(text: str) -> str:
    """Normalise line endings and collapse runs of blank lines.

    Determinism, not tidiness: token offsets are character offsets, so the same
    document checked out with CRLF endings would otherwise chunk differently
    from the same document with LF endings, and produce different chunk text
    under identical chunk ids.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _EXCESS_BLANK_LINES.sub("\n\n", text).strip()


def load_document(path: Path) -> str:
    """Read one supported document into normalised plain text."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported file type: {path.name}")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        raw = "\n\n".join(pages)
    elif suffix in {".txt", ".md", ".mdx"}:
        raw = path.read_text(encoding="utf-8")
    else:  # pragma: no cover - guarded by the suffix check above
        raise ValueError(f"unsupported file type: {path.name}")
    return normalise_text(raw)


def discover_documents(corpus_dir: Path) -> list[Path]:
    """List the supported files under a corpus directory, in a stable order.

    Recursive: the demo corpus is grouped one directory per upstream source
    (`fastapi/`, `langchain/`, `langgraph/`) and mirrors each project's own
    nesting. Sorted on the corpus-relative posix path so the order does not
    depend on the platform's directory-listing order.
    """
    if not corpus_dir.is_dir():
        raise NotADirectoryError(f"corpus directory not found: {corpus_dir}")
    return sorted(
        (
            p
            for p in corpus_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda p: p.relative_to(corpus_dir).as_posix(),
    )


def build_chunks(
    corpus_dir: Path,
    corpus_id: str,
    tokenizer: SpanTokenizer,
    size: int,
    overlap: int,
) -> list[Chunk]:
    """Load and chunk every supported document in a directory.

    Raises on a slug collision rather than letting two documents write chunks
    under the same ids. A silent collision would corrupt the gold set in a way
    that is invisible until the numbers are already published.
    """
    claimed: dict[str, str] = {}
    chunks: list[Chunk] = []
    for path in discover_documents(corpus_dir):
        source_doc = path.relative_to(corpus_dir).as_posix()
        slug = slugify_source(source_doc)
        if slug in claimed and claimed[slug] != source_doc:
            raise ValueError(
                f"chunk id collision: {source_doc!r} and {claimed[slug]!r} both "
                f"reduce to the slug {slug!r}; rename one of them"
            )
        claimed[slug] = source_doc
        chunks.extend(
            chunk_document(
                text=load_document(path),
                source_doc=source_doc,
                corpus_id=corpus_id,
                tokenizer=tokenizer,
                size=size,
                overlap=overlap,
            )
        )
    return chunks


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> SentenceTransformer:
    """Return the process-wide embedding model, loaded exactly once.

    Cached at module scope so it is never loaded inside a request handler. It
    is loaded on first use rather than at import so that importing this module
    — which the test suite and --dry-run both do — needs neither the model
    files nor a network. `backend.main` warms this at import time so the
    serving path never pays the load.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_chunks(chunks: list[Chunk], embedder: SentenceTransformer) -> list[list[float]]:
    """Embed chunk text, L2-normalised so cosine distance is a plain dot product."""
    if not chunks:
        return []
    vectors = embedder.encode(
        [c.text for c in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


@dataclass
class IngestReport:
    """What one ingest run did, for the CLI and for the tests."""

    corpus_id: str
    corpus_dir: Path
    dry_run: bool
    chunks_per_doc: dict[str, int]
    chunks: list[Chunk]
    stored: int = 0
    pruned: int = 0
    ivfflat_lists: int | None = None

    @property
    def document_count(self) -> int:
        return len(self.chunks_per_doc)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def ingest(
    corpus_dir: Path,
    corpus_id: str,
    *,
    dry_run: bool = False,
) -> IngestReport:
    """Chunk a corpus directory and, unless `dry_run`, embed and store it."""
    settings = get_settings() if not dry_run else _settings_for_dry_run()
    chunks = build_chunks(
        corpus_dir=corpus_dir,
        corpus_id=corpus_id,
        tokenizer=get_tokenizer(settings.embedding_model),
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    per_doc: dict[str, int] = {}
    for chunk in chunks:
        per_doc[chunk.source_doc] = per_doc.get(chunk.source_doc, 0) + 1

    report = IngestReport(
        corpus_id=corpus_id,
        corpus_dir=corpus_dir,
        dry_run=dry_run,
        chunks_per_doc=per_doc,
        chunks=chunks,
    )
    if dry_run:
        return report

    embedder = get_embedder(settings.embedding_model)
    vectors = embed_chunks(chunks, embedder)
    records = [
        ChunkRecord(
            chunk_id=chunk.chunk_id,
            corpus_id=chunk.corpus_id,
            source_doc=chunk.source_doc,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            embedding=vector,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    from backend.retrieval import dense_store

    conn = dense_store.connect()
    try:
        dense_store.ensure_schema(conn, dim=len(vectors[0]) if vectors else 0)
        report.stored = dense_store.upsert_chunks(conn, records)
        # Then remove anything this corpus no longer contains, so the store is
        # a mirror of data/demo_corpus rather than the union of every ingest.
        report.pruned = dense_store.prune_chunks(conn, corpus_id, [c.chunk_id for c in chunks])
        # After the writes: ivfflat clusters whatever exists when it is built.
        report.ivfflat_lists = dense_store.create_embedding_index(conn)
    finally:
        conn.close()
    return report


def _settings_for_dry_run() -> Settings:
    """Settings for a run that touches neither the database nor an API.

    --dry-run must work before `.env` exists, so DATABASE_URL is filled with a
    placeholder that is never read. Everything that shapes chunk boundaries
    still comes from the real environment if it is set.
    """
    return Settings(
        database_url=os.environ.get("DATABASE_URL", "unused-in-dry-run"),  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def format_report(report: IngestReport, preview: int = 0) -> str:
    """Render an ingest report for the terminal."""
    lines: list[str] = []
    mode = "DRY RUN — nothing embedded, nothing written" if report.dry_run else "INGEST"
    lines.append(f"{mode}")
    lines.append(f"corpus     {report.corpus_id}  ({report.corpus_dir})")
    lines.append(f"documents  {report.document_count}")
    lines.append(f"chunks     {report.chunk_count}")
    if report.chunks:
        lengths = sorted(c.char_length for c in report.chunks)
        lines.append(
            f"chars/chunk  min {lengths[0]}  median {lengths[len(lengths) // 2]}  "
            f"max {lengths[-1]}"
        )
    lines.append("")
    for doc, count in sorted(report.chunks_per_doc.items()):
        lines.append(f"  {count:>5}  {doc}")
    if not report.dry_run:
        lines.append("")
        lines.append(f"stored     {report.stored} rows")
        lines.append(f"pruned     {report.pruned} rows no longer in the corpus")
        lines.append(f"ivfflat    lists = {report.ivfflat_lists}")

    if preview and report.chunks:
        lines.append("")
        lines.append(f"first {min(preview, len(report.chunks))} chunk boundaries:")
        for chunk in report.chunks[:preview]:
            head = chunk.text[:70].replace("\n", "\\n")
            tail = chunk.text[-70:].replace("\n", "\\n")
            lines.append(f"  {chunk.chunk_id}  chars [{chunk.start_char}:{chunk.end_char}]")
            lines.append(f"      head: {head!r}")
            lines.append(f"      tail: {tail!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.ingest",
        description="Load, chunk, embed, and store a corpus directory.",
    )
    parser.add_argument("--corpus", type=Path, required=True, help="directory of PDF/TXT/MD files")
    parser.add_argument("--corpus-id", default="demo", help="corpus identifier (default: demo)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="chunk and report only: no embedding, no database",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="print the head and tail of the first N chunks",
    )
    args = parser.parse_args(argv)

    report = ingest(args.corpus, args.corpus_id, dry_run=args.dry_run)
    print(format_report(report, preview=args.preview))
    return 0


if __name__ == "__main__":
    sys.exit(main())
