"""BM25 index build + cache.

An in-memory index over the whole corpus. At demo scale (1,480 chunks) that is
a few megabytes and a sub-second build, so the index is held in a process-wide
singleton and rebuilt on process start rather than persisted. Persisting it
would add a cache-invalidation problem -- a stale index silently scoring
against chunks the corpus no longer contains -- to buy back under a second.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

    from backend.retrieval.dense_store import StoredChunk

# Unicode-aware word characters, which is not the same thing as splitting on
# whitespace. What `str.split()` gets wrong on this corpus, measured rather
# than assumed:
#
#   "FastAPI's HTTPException, raised."  ->  ["fastapi's", "httpexception,",
#                                            "raised."]
#
# BM25 then treats "raised." and "raised" as different terms, and a question
# asking about `HTTPException` never matches the chunk that documents it.
# Documentation is unusually dense in exactly this: trailing punctuation,
# `--reload`, `python-multipart`.
#
#   "auto<U+2011>reload"  ->  one token, hyphen and all
#
# U+2011 NON-BREAKING HYPHEN is dash punctuation, not whitespace, so a
# whitespace split keeps it glued while the plain-hyphen spelling in the
# questions splits. `\w+` splits both the same way.
#
# Note for anyone who assumes the opposite, as this file's first draft did:
# U+202F NARROW NO-BREAK SPACE — which 25 of the 115 gold questions contain —
# is category Zs and `str.split()` handles it correctly. It is not the reason
# for this regex.
_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-case Unicode word tokens.

    NFKC first, so compatibility forms fold before matching: U+00A0, full-width
    characters, and ligatures all normalise to the spelling the rest of the
    corpus uses. It is cheap insurance rather than a fix for an observed
    failure -- the observed failures are the punctuation cases above.

    No stemming and no stopword list. Both are defensible additions, and both
    would be an unmeasured change to the lexical lane rather than a free win:
    stemming needs a dependency (NLTK/snowball) that CLAUDE.md's "no library
    for what a dozen lines handles" rule points away from, and dropping
    stopwords changes BM25's length normalisation for every chunk. The lane is
    reported as plain BM25, which is what it is.
    """
    return _WORD.findall(unicodedata.normalize("NFKC", text).lower())


@dataclass
class BM25Index:
    """A built BM25 index and the chunk ids its rows correspond to.

    `chunk_ids[i]` is the id of the document at row `i` of the index, and the
    two are only ever built together in `build_index`. Keeping them in one
    object rather than as parallel module state is what stops them drifting.
    """

    chunk_ids: list[str]
    bm25: BM25Okapi

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the top-`k` (chunk_id, score) pairs for `query`, best first.

        Ties break on chunk_id. BM25 gives a score of exactly 0.0 to every
        chunk sharing no query term, and on a 1,480-chunk corpus a short query
        leaves hundreds of them tied at zero. Without the id tie-break the
        order of that block depends on the sort's view of equal keys and the
        top-50 handed to fusion could differ between runs.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        tokens = tokenize(query)
        if not tokens:
            # A query of pure punctuation. rank_bm25 would return an all-zero
            # score array and we would hand fusion an arbitrary 50 chunks.
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.chunk_ids, (float(s) for s in scores), strict=True),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:k]


def build_index(chunks: list[StoredChunk]) -> BM25Index:
    """Tokenize and index a corpus. Pure: no database, no cache, no globals.

    Separate from `get_index` so the tests can build a three-document index
    over hand-written text without touching Postgres.
    """
    from rank_bm25 import BM25Okapi

    if not chunks:
        raise ValueError("cannot build a BM25 index over an empty corpus")
    # Sorted by chunk_id so row order is a function of the corpus alone. The
    # store already returns them ordered; this makes the guarantee local
    # rather than borrowed from a caller three layers up.
    ordered = sorted(chunks, key=lambda c: c.chunk_id)
    return BM25Index(
        chunk_ids=[c.chunk_id for c in ordered],
        # BM25Okapi defaults: k1=1.5, b=0.75. Left at the defaults deliberately
        # -- tuning them against this gold set would be fitting the lexical
        # baseline to the test split, which is the one thing the split exists
        # to prevent.
        bm25=BM25Okapi([tokenize(c.text) for c in ordered]),
    )


@lru_cache(maxsize=2)
def get_index(corpus_id: str) -> BM25Index:
    """The process-wide index for one corpus, built once on first use.

    Module-level singleton per CLAUDE.md: never built inside a request handler.
    maxsize=2 rather than 1 so a race over the demo corpus does not evict and
    rebuild when a second corpus is queried; the demo corpus is the only one
    that matters for the published numbers.
    """
    from backend.retrieval import dense_store

    conn = dense_store.connect()
    try:
        chunks = dense_store.fetch_chunks(conn, corpus_id)
    finally:
        conn.close()
    if not chunks:
        raise ValueError(
            f"corpus {corpus_id!r} has no chunks in the store. "
            f"Run `python -m backend.ingest --corpus data/demo_corpus` first."
        )
    return build_index(chunks)
