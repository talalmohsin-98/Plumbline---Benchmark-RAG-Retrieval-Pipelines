"""dense_store tests: DDL shape and index sizing, against a recording connection."""

import pytest

from backend.retrieval.dense_store import (
    ChunkRecord,
    ensure_schema,
    index_ddl,
    ivfflat_lists,
    schema_ddl,
    top_k,
    upsert_chunks,
)
from backend.tests.conftest import RecordingConnection


@pytest.mark.parametrize(
    ("rows", "expected"),
    [(0, 1), (1, 1), (999, 1), (1_000, 1), (1_999, 1), (2_000, 2), (1_200_000, 1_200)],
)
def test_ivfflat_lists_follows_the_rows_per_1000_rule(rows, expected):
    assert ivfflat_lists(rows) == expected


def test_demo_corpus_size_yields_a_single_list():
    """~1.2k chunks -> 1 list -> exact search. Deliberate; see the docstring."""
    assert ivfflat_lists(1_240) == 1


def test_negative_row_count_raises():
    with pytest.raises(ValueError):
        ivfflat_lists(-1)


def test_schema_uses_the_dimension_it_is_given():
    statements = schema_ddl(384)
    assert "vector(384)" in statements[1]
    assert "CREATE EXTENSION IF NOT EXISTS vector" in statements[0]
    assert "chunk_id    text PRIMARY KEY" in statements[1]


@pytest.mark.parametrize("bad", [0, -1, "384", 3.0, True, None])
def test_schema_rejects_a_non_positive_int_dimension(bad):
    with pytest.raises(ValueError):
        schema_ddl(bad)


@pytest.mark.parametrize("bad", [0, -1, "10", True])
def test_index_ddl_rejects_bad_lists(bad):
    with pytest.raises(ValueError):
        index_ddl(bad)


def test_index_is_cosine_not_l2():
    """RRF and the UI both read these scores as cosine similarity."""
    assert "vector_cosine_ops" in index_ddl(1)
    assert "USING ivfflat" in index_ddl(1)


def test_ensure_schema_is_idempotent_ddl():
    conn = RecordingConnection()
    ensure_schema(conn, dim=384)

    assert len(conn.statements) == 3
    assert all("IF NOT EXISTS" in s for s in conn.statements)
    assert conn.commits == 1


def test_upsert_overwrites_by_chunk_id():
    """Re-ingest must overwrite in place: ids are deterministic, so a plain
    INSERT would fail and an INSERT-ignore would leave stale text behind."""
    conn = RecordingConnection()
    records = [
        ChunkRecord("nadra_ch_000", "demo", "NADRA.txt", 0, "text a", [0.1, 0.2]),
        ChunkRecord("nadra_ch_001", "demo", "NADRA.txt", 1, "text b", [0.3, 0.4]),
    ]

    assert upsert_chunks(conn, records) == 2
    sql, rows = conn.executed[0]
    assert "ON CONFLICT (chunk_id) DO UPDATE" in sql
    assert len(rows) == 2
    assert rows[0][0] == "nadra_ch_000"


def test_upsert_of_nothing_issues_no_sql():
    conn = RecordingConnection()
    assert upsert_chunks(conn, []) == 0
    assert conn.executed == []


def test_top_k_orders_by_distance_and_filters_by_corpus():
    conn = RecordingConnection(
        fetchall_result=[("nadra_ch_000", "demo", "NADRA.txt", 0, "text a", 0.91)]
    )
    results = top_k(conn, [0.1, 0.2], corpus_id="demo", k=10)

    probes_sql, probes_params = conn.executed[0]
    assert "ivfflat.probes" in probes_sql
    assert probes_params == (1,)

    query_sql, query_params = conn.executed[1]
    assert "ORDER BY embedding <=> %(q)s" in query_sql
    assert "WHERE corpus_id = %(corpus_id)s" in query_sql
    assert query_params == {"q": [0.1, 0.2], "corpus_id": "demo", "k": 10}

    assert len(results) == 1
    assert results[0].chunk_id == "nadra_ch_000"
    assert results[0].similarity == pytest.approx(0.91)


def test_top_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        top_k(RecordingConnection(), [0.1], corpus_id="demo", k=0)
