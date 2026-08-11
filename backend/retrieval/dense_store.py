"""pgvector storage: table and index management, upserts, and top-k cosine queries."""

from dataclasses import dataclass
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from backend.config import get_settings

TABLE = "chunks"


@dataclass(frozen=True)
class ChunkRecord:
    """One chunk as it is stored: identity, provenance, text, and its vector."""

    chunk_id: str
    corpus_id: str
    source_doc: str
    chunk_index: int
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class ScoredChunk:
    """One retrieved chunk and its cosine similarity to the query."""

    chunk_id: str
    corpus_id: str
    source_doc: str
    chunk_index: int
    text: str
    similarity: float


def connect(dsn: str | None = None) -> psycopg.Connection[Any]:
    """Open a connection with the pgvector type adapters registered."""
    if dsn is None:
        dsn = get_settings().database_url.get_secret_value()
    conn = psycopg.connect(dsn)
    register_vector(conn)
    return conn


def schema_ddl(dim: int) -> list[str]:
    """Return the DDL statements for the chunk table, in execution order.

    `dim` is interpolated rather than bound because a type modifier cannot be a
    query parameter. It is validated as a positive int first, so nothing
    attacker-controlled can reach the string.
    """
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise ValueError(f"embedding dimension must be a positive int, got {dim!r}")
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            chunk_id    text PRIMARY KEY,
            corpus_id   text NOT NULL,
            source_doc  text NOT NULL,
            chunk_index integer NOT NULL,
            text        text NOT NULL,
            embedding   vector({dim}) NOT NULL
        )
        """,
        f"CREATE INDEX IF NOT EXISTS {TABLE}_corpus_id_idx ON {TABLE} (corpus_id)",
    ]


def ensure_schema(conn: psycopg.Connection[Any], dim: int) -> None:
    """Create the extension, table, and corpus index if they do not exist."""
    with conn.cursor() as cur:
        for statement in schema_ddl(dim):
            cur.execute(statement)
    conn.commit()


def ivfflat_lists(row_count: int) -> int:
    """Choose the ivfflat `lists` parameter for a table of `row_count` rows.

    pgvector's guidance is rows/1000 for tables up to 1M rows. The demo corpus
    is ~1.2k chunks, which lands on 1 list — i.e. effectively an exact scan.
    That is the right answer at this size: splitting 1.2k vectors across many
    lists would put ~12 rows in each, and a single probe would then miss most
    true neighbours. Recall matters more here than a scan we cannot measure.
    """
    if row_count < 0:
        raise ValueError("row_count cannot be negative")
    return max(1, row_count // 1000)


def index_ddl(lists: int) -> str:
    """Return the DDL for the ivfflat cosine index."""
    if not isinstance(lists, int) or isinstance(lists, bool) or lists <= 0:
        raise ValueError(f"lists must be a positive int, got {lists!r}")
    return (
        f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding_ivfflat "
        f"ON {TABLE} USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
    )


def count_rows(conn: psycopg.Connection[Any], corpus_id: str | None = None) -> int:
    """Count stored chunks, optionally within one corpus."""
    with conn.cursor() as cur:
        if corpus_id is None:
            cur.execute(f"SELECT count(*) FROM {TABLE}")
        else:
            cur.execute(f"SELECT count(*) FROM {TABLE} WHERE corpus_id = %s", (corpus_id,))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def create_embedding_index(conn: psycopg.Connection[Any], lists: int | None = None) -> int:
    """Build the ivfflat index, sizing `lists` from the current row count.

    Called after the upsert, never before: ivfflat clusters the vectors that
    exist when the index is built, so indexing an empty table produces useless
    centroids.
    """
    if lists is None:
        lists = ivfflat_lists(count_rows(conn))
    with conn.cursor() as cur:
        cur.execute(index_ddl(lists))
    conn.commit()
    return lists


def upsert_chunks(conn: psycopg.Connection[Any], records: list[ChunkRecord]) -> int:
    """Insert or replace chunks by `chunk_id`.

    Upsert rather than insert because `chunk_id` is deterministic: re-ingesting
    the same corpus must overwrite in place, not duplicate or fail.
    """
    if not records:
        return 0
    rows = [
        (r.chunk_id, r.corpus_id, r.source_doc, r.chunk_index, r.text, r.embedding)
        for r in records
    ]
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {TABLE}
                (chunk_id, corpus_id, source_doc, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                corpus_id   = EXCLUDED.corpus_id,
                source_doc  = EXCLUDED.source_doc,
                chunk_index = EXCLUDED.chunk_index,
                text        = EXCLUDED.text,
                embedding   = EXCLUDED.embedding
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def top_k(
    conn: psycopg.Connection[Any],
    query_embedding: list[float],
    corpus_id: str,
    k: int,
    probes: int = 1,
) -> list[ScoredChunk]:
    """Return the k nearest chunks in `corpus_id` by cosine distance.

    Embeddings are stored L2-normalised (see `backend.ingest`), so cosine
    distance `<=>` and inner product rank identically; cosine is used because
    it reads as a similarity in [0, 1] for the UI.

    Note the `corpus_id` filter: Postgres may fall back to a sequential scan
    when a filter is selective enough, which at demo-corpus size is fine and
    exact. If corpora grow, the fix is a partial index per corpus, not a
    bigger `lists`.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    with conn.cursor() as cur:
        # SET LOCAL keeps the probe count scoped to this transaction.
        cur.execute("SET LOCAL ivfflat.probes = %s", (probes,))
        cur.execute(
            f"""
            SELECT chunk_id, corpus_id, source_doc, chunk_index, text,
                   1 - (embedding <=> %(q)s) AS similarity
            FROM {TABLE}
            WHERE corpus_id = %(corpus_id)s
            ORDER BY embedding <=> %(q)s
            LIMIT %(k)s
            """,
            {"q": query_embedding, "corpus_id": corpus_id, "k": k},
        )
        rows = cur.fetchall()
    return [
        ScoredChunk(
            chunk_id=row[0],
            corpus_id=row[1],
            source_doc=row[2],
            chunk_index=row[3],
            text=row[4],
            similarity=float(row[5]),
        )
        for row in rows
    ]
