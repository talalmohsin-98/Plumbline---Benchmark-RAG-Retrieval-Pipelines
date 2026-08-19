"""Shared stubs. Nothing here touches a network, a model file, or a database."""

import re

import pytest

from backend.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    """Keep the developer's .env out of the test suite.

    pydantic-settings reads `.env` as well as the environment, so without this
    a real DATABASE_URL or GROQ_API_KEY on the machine changes what the tests
    exercise. The suite has to behave the same on a laptop with every secret
    configured and in CI with none.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

_WORD = re.compile(r"\S+")


class WhitespaceSpanTokenizer:
    """A tokenizer where one token is one whitespace-delimited word.

    Deliberately trivial: it lets the chunking fixtures below be counted by
    hand. The real tokenizer's job (how text maps to tokens) is not what these
    tests are checking — the windowing arithmetic is.
    """

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in _WORD.finditer(text)]


class RecordingCursor:
    """A psycopg-shaped cursor that records SQL instead of executing it."""

    rowcount = 0

    def __init__(self, store):
        self._store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._store.executed.append((" ".join(str(sql).split()), params))

    def executemany(self, sql, rows):
        self._store.executed.append((" ".join(str(sql).split()), list(rows)))

    def fetchone(self):
        return self._store.fetchone_result

    def fetchall(self):
        return self._store.fetchall_result


class RecordingConnection:
    """A psycopg-shaped connection that records SQL instead of executing it."""

    def __init__(self, fetchone_result=None, fetchall_result=()):
        self.executed: list[tuple[str, object]] = []
        self.commits = 0
        self.closed = False
        self.fetchone_result = fetchone_result
        self.fetchall_result = list(fetchall_result)

    def cursor(self):
        return RecordingCursor(self)

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True

    @property
    def statements(self) -> list[str]:
        return [sql for sql, _ in self.executed]


@pytest.fixture
def tokenizer() -> WhitespaceSpanTokenizer:
    return WhitespaceSpanTokenizer()


@pytest.fixture
def corpus_dir(tmp_path):
    """A two-document corpus with hand-countable word counts."""
    (tmp_path / "NADRA.txt").write_text(
        " ".join(f"w{i}" for i in range(10)),
        encoding="utf-8",
    )
    (tmp_path / "Guide Notes.md").write_text(
        " ".join(f"g{i}" for i in range(7)),
        encoding="utf-8",
    )
    (tmp_path / "ignored.csv").write_text("a,b,c", encoding="utf-8")
    return tmp_path
