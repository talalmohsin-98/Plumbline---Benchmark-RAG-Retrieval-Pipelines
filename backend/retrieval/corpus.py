"""One corpus, ready to retrieve from: chunk text, the BM25 index, and embeddings.

Every lane needs the same three things and none of them should be built per
query. This holds them together so a lane's constructor takes one object
rather than a connection, an index, an embedder and a text map -- which is
what keeps "a new lane is one new file" true.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from backend.config import get_settings
from backend.retrieval import bm25_index, dense_store

if TYPE_CHECKING:
    import psycopg

    from backend.retrieval.dense_store import StoredChunk


class Corpus:
    """Retrieval access to one corpus id, with everything loaded at most once.

    Not a dataclass: the expensive members (index, embedder, connection) are
    built on first use, so that a lexical-only run never loads a 133 MB
    embedding model and a test that only needs chunk text never opens a socket.
    """

    def __init__(self, corpus_id: str = "demo") -> None:
        self.corpus_id = corpus_id
        self._chunks: dict[str, StoredChunk] | None = None
        self._local = threading.local()

    # ----------------------------------------------------------------
    # Connections
    # ----------------------------------------------------------------

    def connection(self) -> psycopg.Connection[Any]:
        """A connection owned by the calling thread.

        Thread-local rather than a single shared connection because Day 4 fans
        the lanes out concurrently, and a psycopg connection is not safe to
        use from two threads at once -- two lanes interleaving on one
        connection corrupt each other's result sets. Thread-local rather than
        per-call because opening a connection to a hosted Postgres costs tens
        of milliseconds, which on a lane whose entire budget is 120 ms would
        be most of the measured latency.

        Not a pool: psycopg_pool is a dependency this does not need at six
        lanes and one process. If lane concurrency ever exceeds a handful of
        threads, a pool is the right upgrade.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn = dense_store.connect()
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close this thread's connection. Safe to call more than once."""
        conn = getattr(self._local, "conn", None)
        if conn is not None and not conn.closed:
            conn.close()
        self._local.conn = None

    # ----------------------------------------------------------------
    # Chunk text
    # ----------------------------------------------------------------

    @property
    def chunks(self) -> dict[str, StoredChunk]:
        """Every chunk of this corpus by id, read once and held.

        1,480 chunks of documentation is a few megabytes; holding them costs
        less than re-reading a chunk's text out of Postgres for every one of
        the 20 candidates a reranking lane scores, on every query.
        """
        if self._chunks is None:
            rows = dense_store.fetch_chunks(self.connection(), self.corpus_id)
            if not rows:
                raise ValueError(
                    f"corpus {self.corpus_id!r} has no chunks in the store. "
                    f"Run `python -m backend.ingest --corpus data/demo_corpus` first."
                )
            self._chunks = {row.chunk_id: row for row in rows}
        return self._chunks

    @property
    def texts(self) -> dict[str, str]:
        return {chunk_id: row.text for chunk_id, row in self.chunks.items()}

    @property
    def sources(self) -> dict[str, str]:
        return {chunk_id: row.source_doc for chunk_id, row in self.chunks.items()}

    def text_of(self, chunk_id: str) -> str:
        chunk = self.chunks.get(chunk_id)
        return chunk.text if chunk else ""

    # ----------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-`k` (chunk_id, bm25_score), best first."""
        return bm25_index.get_index(self.corpus_id).search(query, k)

    def embed_query(self, text: str, *, prefix: bool = True) -> list[float]:
        """Embed one query string, L2-normalised to match the stored vectors.

        `prefix` prepends bge's query instruction. bge-*-en-v1.5 is trained
        asymmetrically: the query side carries "Represent this sentence for
        searching relevant passages: " and the passage side is embedded bare.
        Passages in the store were embedded bare by `ingest.py`, so this is the
        only place the asymmetry has to be honoured -- and the flag exists so
        the with/without delta can be measured rather than asserted.
        """
        from backend.ingest import get_embedder

        settings = get_settings()
        embedder = get_embedder(settings.embedding_model)
        payload = f"{settings.bge_query_prefix}{text}" if prefix else text
        vector = embedder.encode(
            payload,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def dense_search(
        self,
        query: str,
        k: int,
        *,
        prefix: bool = True,
    ) -> list[tuple[str, float]]:
        """Top-`k` (chunk_id, cosine_similarity), best first."""
        return self.dense_search_vector(self.embed_query(query, prefix=prefix), k)

    def dense_search_vector(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Top-`k` by cosine against an already-computed query vector.

        Split out from `dense_search` for HyDE, which searches with the
        embedding of a generated passage rather than of the query text.
        """
        hits = dense_store.top_k(self.connection(), vector, self.corpus_id, k)
        return [(hit.chunk_id, hit.similarity) for hit in hits]

    # ----------------------------------------------------------------
    # Warmup
    # ----------------------------------------------------------------

    def warm(self, *, embedder: bool = True) -> None:
        """Build the index, load the model and open the connection up front.

        Called before any timed query. With n=35 questions a single first-query
        outlier that includes a model load *is* the p95, so this is the
        difference between a latency number and a fiction.
        """
        _ = self.chunks
        bm25_index.get_index(self.corpus_id)
        if embedder:
            self.embed_query("warmup")
