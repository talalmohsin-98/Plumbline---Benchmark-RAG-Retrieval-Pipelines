"""Pipeline wiring: --dry-run, embedding, and storage, all against stubs."""

import pytest

from backend import ingest as ingest_module
from backend.ingest import Chunk, embed_chunks, ingest, normalise_text
from backend.retrieval import dense_store


def _explode(*args, **kwargs):
    raise AssertionError("dry run must not reach this")


@pytest.fixture
def small_chunks(monkeypatch, tokenizer):
    """Wire the module to the stub tokenizer and a hand-sized window."""
    monkeypatch.setattr(ingest_module, "get_tokenizer", lambda _name: tokenizer)
    monkeypatch.setenv("CHUNK_SIZE", "5")
    monkeypatch.setenv("CHUNK_OVERLAP", "2")


def test_dry_run_neither_embeds_nor_connects(monkeypatch, corpus_dir, small_chunks):
    monkeypatch.setattr(ingest_module, "get_embedder", _explode)
    monkeypatch.setattr(dense_store, "connect", _explode)

    report = ingest(corpus_dir, "demo", dry_run=True)

    assert report.dry_run is True
    assert report.document_count == 2
    assert report.chunk_count == 5  # NADRA.txt -> 3, Guide Notes.md -> 2
    assert report.chunks_per_doc == {"NADRA.txt": 3, "Guide Notes.md": 2}
    assert report.stored == 0
    assert report.ivfflat_lists is None


def test_dry_run_needs_no_secrets(monkeypatch, corpus_dir, small_chunks):
    """--dry-run has to work before .env exists."""
    for key in ("DATABASE_URL", "GROQ_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(ingest_module, "get_embedder", _explode)

    assert ingest(corpus_dir, "demo", dry_run=True).chunk_count == 5


def test_full_run_embeds_and_stores_every_chunk(monkeypatch, corpus_dir, small_chunks):
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    monkeypatch.setenv("GROQ_API_KEY", "stub")
    monkeypatch.setenv("HF_TOKEN", "stub")
    ingest_module.get_settings.cache_clear()

    seen: dict[str, object] = {}
    connection = FakeConnection()

    def fake_embed(chunks, embedder):
        seen["texts"] = [c.text for c in chunks]
        return [[0.1, 0.2, 0.3] for _ in chunks]

    monkeypatch.setattr(ingest_module, "get_embedder", lambda _name: "stub-model")
    monkeypatch.setattr(ingest_module, "embed_chunks", fake_embed)
    monkeypatch.setattr(dense_store, "connect", lambda: connection)

    report = ingest(corpus_dir, "demo", dry_run=False)
    ingest_module.get_settings.cache_clear()

    assert report.stored == 5
    assert len(seen["texts"]) == 5
    assert connection.closed is True
    # Dimension is taken from the vectors, never hardcoded.
    assert "vector(3)" in " ".join(connection.statements)
    # The index is built after the rows land, not before.
    joined = connection.statements
    upsert_at = next(i for i, s in enumerate(joined) if "INSERT INTO chunks" in s)
    index_at = next(i for i, s in enumerate(joined) if "ivfflat" in s and "CREATE INDEX" in s)
    assert upsert_at < index_at


class FakeConnection:
    """Minimal psycopg-shaped connection; see conftest.RecordingConnection."""

    def __init__(self):
        self.executed: list[tuple[str, object]] = []
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        pass

    def close(self):
        self.closed = True

    @property
    def statements(self) -> list[str]:
        return [sql for sql, _ in self.executed]


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._conn.executed.append((" ".join(str(sql).split()), params))

    def executemany(self, sql, rows):
        self._conn.executed.append((" ".join(str(sql).split()), list(rows)))

    def fetchone(self):
        return (5,)

    def fetchall(self):
        return []


def test_embed_chunks_normalises_and_preserves_order():
    """The embedder is stubbed; what is asserted is the contract we depend on."""
    calls: dict[str, object] = {}

    class StubEmbedder:
        def encode(self, texts, **kwargs):
            calls.update(kwargs)
            calls["texts"] = list(texts)
            return [_Vector([float(i)]) for i, _ in enumerate(texts)]

    chunks = [
        Chunk("d_ch_000", "demo", "d.txt", 0, "first", 0, 5),
        Chunk("d_ch_001", "demo", "d.txt", 1, "second", 6, 12),
    ]
    vectors = embed_chunks(chunks, StubEmbedder())

    assert calls["texts"] == ["first", "second"]
    # bge vectors must be L2-normalised: dense_store treats cosine as a dot product.
    assert calls["normalize_embeddings"] is True
    assert vectors == [[0.0], [1.0]]


def test_embed_chunks_on_empty_input_does_not_call_the_model():
    assert embed_chunks([], _explode) == []


class _Vector:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\r\nb", "a\nb"),
        ("a\rb", "a\nb"),
        ("a\n\n\n\n\nb", "a\n\nb"),
        ("  padded  ", "padded"),
        ("a\n\nb", "a\n\nb"),
    ],
)
def test_normalise_text(raw, expected):
    assert normalise_text(raw) == expected
