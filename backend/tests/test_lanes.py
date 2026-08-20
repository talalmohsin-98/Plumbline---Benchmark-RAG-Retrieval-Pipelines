"""Lane interface tests. No database, no model files, no network.

Every lane is exercised against a stub corpus whose retrieval results are
written by hand, so these tests assert what the *lane* does -- what it fuses,
what it reranks, what it puts in its trace -- rather than how good BM25 is.
Retrieval quality is `evaluate.py`'s job and is measured against the gold set.
"""

import pytest

from backend.lanes import REGISTRY, REMOTE_LANES, build_lanes
from backend.lanes.base import Lane, LaneResult, RetrievedChunk, StageTrace, rank_chunks, stopwatch
from backend.lanes.bm25 import BM25Lane
from backend.lanes.dense import DenseLane
from backend.lanes.hybrid import HybridLane
from backend.lanes.hyde import HydeGenerationError, HydeLane
from backend.lanes.reranked import RerankedLane


class StubChunk:
    def __init__(self, chunk_id, text):
        self.chunk_id = chunk_id
        self.text = text
        self.source_doc = "doc.md"


class StubCorpus:
    """A Corpus-shaped object whose two retrievers return fixed lists.

    The lists are deliberately disjoint at the top so that fusion has
    something to do and a lane that quietly returns one arm is visible.
    """

    corpus_id = "stub"

    def __init__(self, bm25=None, dense=None):
        self._bm25 = bm25 if bm25 is not None else [("a", 9.0), ("b", 4.0), ("c", 1.0)]
        self._dense = dense if dense is not None else [("c", 0.9), ("d", 0.8), ("a", 0.7)]
        ids = {cid for cid, _ in self._bm25} | {cid for cid, _ in self._dense}
        self.chunks = {cid: StubChunk(cid, f"text of {cid}") for cid in sorted(ids)}
        self.embedded = []
        self.warmed = 0

    @property
    def texts(self):
        return {cid: c.text for cid, c in self.chunks.items()}

    @property
    def sources(self):
        return {cid: c.source_doc for cid, c in self.chunks.items()}

    def text_of(self, chunk_id):
        chunk = self.chunks.get(chunk_id)
        return chunk.text if chunk else ""

    def bm25_search(self, query, k):
        return self._bm25[:k]

    def embed_query(self, text, *, prefix=True):
        self.embedded.append((text, prefix))
        return [0.1, 0.2, 0.3]

    def dense_search_vector(self, vector, k):
        return self._dense[:k]

    def dense_search(self, query, k, *, prefix=True):
        return self.dense_search_vector(self.embed_query(query, prefix=prefix), k)

    def warm(self, *, embedder=True):
        self.warmed += 1


class StubCrossEncoder:
    """Scores by a fixed table, so the reranked order is known in advance."""

    def __init__(self, table):
        self.table = table
        self.seen_pairs = None

    def predict(self, pairs, show_progress_bar=False):
        self.seen_pairs = list(pairs)
        return [self.table.get(text, 0.0) for _, text in pairs]


# --------------------------------------------------------------------------
# The registry contract
# --------------------------------------------------------------------------


def test_every_registry_entry_builds_a_lane_whose_id_matches_its_key():
    # The registry key is what `--lanes` and results.json use. A lane whose
    # `id` disagrees with its key would be selectable under one name and
    # reported under another.
    corpus = StubCorpus()
    for key, factory in REGISTRY.items():
        lane = factory(corpus)
        assert isinstance(lane, Lane)
        assert lane.id == key
        assert lane.label


def test_all_five_day_two_lanes_are_registered():
    assert list(REGISTRY) == ["bm25", "dense", "hybrid_rrf", "hybrid_rerank", "hyde"]


def test_lane_labels_are_unique():
    # Two lanes sharing a label makes the leaderboard unreadable and the
    # results file ambiguous.
    labels = [factory(StubCorpus()).label for factory in REGISTRY.values()]
    assert len(set(labels)) == len(labels)


def test_build_lanes_returns_registry_order_not_caller_order():
    # So that `--lanes dense,bm25` and `--lanes bm25,dense` produce the same
    # results file rather than two files that differ only in row order.
    corpus = StubCorpus()
    forward = [lane.id for lane in build_lanes(corpus, ["dense", "bm25"])]
    backward = [lane.id for lane in build_lanes(corpus, ["bm25", "dense"])]
    assert forward == backward == ["bm25", "dense"]


def test_build_lanes_with_no_selection_builds_everything():
    assert len(build_lanes(StubCorpus())) == len(REGISTRY)


def test_build_lanes_rejects_an_unknown_lane_by_name():
    with pytest.raises(KeyError, match="nonexistent"):
        build_lanes(StubCorpus(), ["bm25", "nonexistent"])


def test_hyde_is_the_only_lane_that_calls_an_external_api():
    # If this changes, `--skip-remote` silently stops protecting the run from
    # a spent Groq budget.
    assert set(REMOTE_LANES) == {"hyde"}


# --------------------------------------------------------------------------
# base.py helpers
# --------------------------------------------------------------------------


def test_rank_chunks_sorts_by_score_and_numbers_ranks_from_one():
    ranked = rank_chunks(
        [("a", 0.1), ("b", 0.9), ("c", 0.5)],
        {"a": "ta", "b": "tb", "c": "tc"},
        {"a": "da", "b": "db", "c": "dc"},
        k=3,
    )
    assert [c.chunk_id for c in ranked] == ["b", "c", "a"]
    assert [c.rank for c in ranked] == [1, 2, 3]
    assert ranked[0].text == "tb"
    assert ranked[0].source_doc == "db"


def test_rank_chunks_breaks_score_ties_on_chunk_id():
    # BM25 ties whole blocks of chunks at 0.0; without this the top-k could
    # differ between runs on identical data.
    ranked = rank_chunks([("z", 1.0), ("a", 1.0), ("m", 1.0)], {}, {}, k=3)
    assert [c.chunk_id for c in ranked] == ["a", "m", "z"]


def test_rank_chunks_truncates_to_k():
    assert len(rank_chunks([(c, 1.0) for c in "abcdef"], {}, {}, k=2)) == 2


def test_rank_chunks_tolerates_a_chunk_id_with_no_text():
    # Defensive: a fused id whose chunk left the corpus must not crash the
    # lane mid-race. It comes back with empty text rather than a KeyError.
    ranked = rank_chunks([("ghost", 1.0)], {}, {}, k=1)
    assert ranked[0].text == ""
    assert ranked[0].source_doc == ""


def test_lane_result_exposes_chunk_ids_in_rank_order():
    result = LaneResult(
        chunks=[
            RetrievedChunk("a", "d", "t", 1.0, 1),
            RetrievedChunk("b", "d", "t", 0.5, 2),
        ],
        latency_ms=1.0,
    )
    assert result.chunk_ids == ["a", "b"]


def test_lane_result_defaults_to_zero_tokens():
    # Five of six lanes make no LLM call and must report a real zero.
    result = LaneResult(chunks=[], latency_ms=0.0)
    assert result.tokens_used == 0
    assert (result.prompt_tokens, result.completion_tokens) == (0, 0)


def test_lane_result_tokens_used_is_the_sum():
    result = LaneResult(chunks=[], latency_ms=0.0, prompt_tokens=157, completion_tokens=82)
    assert result.tokens_used == 239


def test_stopwatch_measures_a_block_and_never_returns_negative():
    with stopwatch() as watch:
        sum(range(10000))
    assert watch.ms >= 0.0


# --------------------------------------------------------------------------
# Lane 1: BM25
# --------------------------------------------------------------------------


def test_bm25_lane_returns_the_lexical_ranking():
    lane = BM25Lane(StubCorpus())
    result = lane.retrieve("q", k=3)
    assert result.chunk_ids == ["a", "b", "c"]
    assert [c.rank for c in result.chunks] == [1, 2, 3]


def test_bm25_lane_never_embeds_anything():
    # It must stay runnable with no model files present at all.
    corpus = StubCorpus()
    BM25Lane(corpus).retrieve("q", k=3)
    assert corpus.embedded == []


def test_bm25_lane_warms_without_the_embedder():
    corpus = StubCorpus()

    def warm(*, embedder=True):
        assert embedder is False
        corpus.warmed += 1

    corpus.warm = warm
    BM25Lane(corpus).warm()
    assert corpus.warmed == 1


def test_bm25_lane_costs_nothing():
    assert BM25Lane(StubCorpus()).retrieve("q").tokens_used == 0


def test_bm25_lane_traces_one_stage():
    trace = BM25Lane(StubCorpus()).retrieve("q", k=3).stages
    assert [s.name for s in trace] == ["bm25"]
    assert trace[0].candidates_out == 3


# --------------------------------------------------------------------------
# Lane 2: dense
# --------------------------------------------------------------------------


def test_dense_lane_returns_the_vector_ranking():
    result = DenseLane(StubCorpus()).retrieve("q", k=3)
    assert result.chunk_ids == ["c", "d", "a"]


def test_dense_lane_applies_the_bge_query_prefix_by_default():
    corpus = StubCorpus()
    DenseLane(corpus).retrieve("what is a dependency?")
    assert corpus.embedded == [("what is a dependency?", True)]


def test_dense_lane_can_be_built_without_the_prefix_for_the_ablation():
    corpus = StubCorpus()
    DenseLane(corpus, use_query_prefix=False).retrieve("what is a dependency?")
    assert corpus.embedded == [("what is a dependency?", False)]


def test_dense_lane_traces_embed_then_search():
    trace = DenseLane(StubCorpus()).retrieve("q", k=3).stages
    assert [s.name for s in trace] == ["embed", "pgvector"]


def test_dense_lane_records_the_prefix_setting_in_its_trace():
    # The trace is what the methodology note is read off. If it does not say
    # which variant ran, the ablation is unattributable.
    on = DenseLane(StubCorpus()).retrieve("q").stages[0].detail
    off = DenseLane(StubCorpus(), use_query_prefix=False).retrieve("q").stages[0].detail
    assert "with bge query prefix" in on
    assert "no query prefix" in off


# --------------------------------------------------------------------------
# Lane 3: hybrid + RRF
# --------------------------------------------------------------------------


def test_hybrid_lane_fuses_both_arms_rather_than_returning_either():
    # bm25 = [a, b, c], dense = [c, d, a], k=60.
    #   a = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    #   c = 1/63 + 1/61 = 0.015873 + 0.016393 = 0.032266
    #   b = 1/62                              = 0.016129
    #   d = 1/62                              = 0.016129
    # a and c tie exactly and break on id; b and d likewise.
    result = HybridLane(StubCorpus()).retrieve("q", k=4)
    assert result.chunk_ids == ["a", "c", "b", "d"]
    # d exists only in the dense arm and a only tops the lexical one, so
    # neither arm alone could have produced this list.
    assert "d" in result.chunk_ids


def test_hybrid_lane_surfaces_a_chunk_both_arms_rank_middling():
    # b is 2nd in one list and 2nd in the other; x is 1st in one and absent
    # from the other. Agreement beats a single strong placement at k=60.
    corpus = StubCorpus(
        bm25=[("x", 9.0), ("b", 5.0)],
        dense=[("y", 0.9), ("b", 0.8)],
    )
    assert HybridLane(corpus).retrieve("q", k=3).chunk_ids[0] == "b"


def test_hybrid_lane_traces_all_four_stages():
    trace = HybridLane(StubCorpus()).retrieve("q", k=3).stages
    assert [s.name for s in trace] == ["bm25", "embed", "pgvector", "rrf"]


def test_hybrid_lane_costs_nothing():
    assert HybridLane(StubCorpus()).retrieve("q").tokens_used == 0


def test_hybrid_fuse_returns_the_full_list_not_a_truncated_one():
    # The reranking lanes slice this to rerank_depth. If `fuse` truncated to
    # k, lane 4 would be limited by a cutoff chosen in lane 3.
    fused, _ = HybridLane(StubCorpus()).fuse("q")
    assert len(fused) == 4


# --------------------------------------------------------------------------
# Lane 4: reranked
# --------------------------------------------------------------------------


@pytest.fixture
def patched_cross_encoder(monkeypatch):
    """Install a scoring table in place of the real cross-encoder."""

    def install(table):
        encoder = StubCrossEncoder(table)
        monkeypatch.setattr(
            "backend.lanes.reranked.get_cross_encoder", lambda name: encoder
        )
        monkeypatch.setattr("backend.lanes.hyde.get_cross_encoder", lambda name: encoder)
        return encoder

    return install


def test_reranked_lane_reorders_the_fused_list(patched_cross_encoder):
    # Fusion order is [a, c, b, d]. The cross-encoder disagrees and puts d
    # first; the lane must follow the reranker, or it is not a reranking lane.
    patched_cross_encoder({"text of d": 5.0, "text of a": 1.0, "text of c": 0.5})
    result = RerankedLane(StubCorpus()).retrieve("q", k=4)
    assert result.chunk_ids[0] == "d"


def test_reranked_lane_scores_against_the_original_query(patched_cross_encoder):
    encoder = patched_cross_encoder({})
    RerankedLane(StubCorpus()).retrieve("how do dependencies work?", k=2)
    assert {query for query, _ in encoder.seen_pairs} == {"how do dependencies work?"}


def test_reranked_lane_only_reranks_down_to_rerank_depth(patched_cross_encoder, monkeypatch):
    # Cross-encoder cost is linear in candidates and is the largest CPU item
    # in the system. A regression that reranked the whole fused list would
    # still return correct results, just slowly -- so it needs its own test.
    from backend.config import get_settings

    monkeypatch.setenv("RERANK_DEPTH", "2")
    get_settings.cache_clear()
    encoder = patched_cross_encoder({})
    RerankedLane(StubCorpus()).retrieve("q", k=4)
    assert len(encoder.seen_pairs) == 2


def test_reranked_lane_appends_a_rerank_stage_after_the_hybrid_stages(patched_cross_encoder):
    patched_cross_encoder({})
    trace = RerankedLane(StubCorpus()).retrieve("q", k=3).stages
    assert [s.name for s in trace] == ["bm25", "embed", "pgvector", "rrf", "rerank"]


def test_lane_four_and_lane_six_are_the_same_class_with_different_checkpoints():
    # This is what makes their comparison attributable to the model rather
    # than to the pipeline, and it is the headline claim of Day 3.
    from backend.lanes import lane_six_factory

    corpus = StubCorpus()
    stock = REGISTRY["hybrid_rerank"](corpus)
    tuned = lane_six_factory()(corpus)
    assert type(stock) is type(tuned) is RerankedLane
    assert stock.model_name != tuned.model_name
    assert stock.id != tuned.id


def test_lane_six_is_not_in_the_registry_until_the_checkpoint_exists():
    # An entry here would make every evaluation run 404 on a model that has
    # not been trained yet.
    assert "hybrid_rerank_tuned" not in REGISTRY


# --------------------------------------------------------------------------
# Lane 5: HyDE
# --------------------------------------------------------------------------


class StubGroq:
    """Returns canned completions and counts how many calls it took."""

    def __init__(self, passage="A generated passage.", fail_times=0, usage=(157, 82)):
        self.passage = passage
        self.fail_times = fail_times
        self.usage = usage
        self.calls = 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.calls <= self.fail_times:
            raise RuntimeError("rate limited")
        return _Response(self.passage, self.usage)


class _Response:
    def __init__(self, content, usage):
        self.choices = [_Choice(content)]
        self.usage = _Usage(*usage)


class _Choice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()
        self.finish_reason = "stop"


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def hyde_with(monkeypatch, corpus, client):
    lane = HydeLane(corpus)
    monkeypatch.setattr(lane, "_client", lambda: client)
    return lane


def test_hyde_embeds_the_generated_passage_without_the_query_prefix(monkeypatch):
    # Decision 1 of the lane. The generated text is a pseudo-*passage*, and
    # bge's prefix marks a query; prefixing it undoes the symmetry HyDE buys.
    corpus = StubCorpus()
    client = StubGroq(passage="Dependencies with yield create a context manager.")
    lane = hyde_with(monkeypatch, corpus, client)
    monkeypatch.setattr("backend.lanes.hyde.get_cross_encoder", lambda n: StubCrossEncoder({}))
    lane.retrieve("how do dependencies work?", k=2)
    assert corpus.embedded == [("Dependencies with yield create a context manager.", False)]


def test_hyde_searches_bm25_with_the_original_question(monkeypatch):
    # Decision 2. A hallucinated identifier would be matched literally by
    # BM25 and drag the lexical arm somewhere the corpus never went.
    corpus = StubCorpus()
    seen = []
    corpus.bm25_search = lambda q, k: seen.append(q) or [("a", 1.0)]
    lane = hyde_with(monkeypatch, corpus, StubGroq())
    monkeypatch.setattr("backend.lanes.hyde.get_cross_encoder", lambda n: StubCrossEncoder({}))
    lane.retrieve("the original question", k=2)
    assert seen == ["the original question"]


def test_hyde_reranks_against_the_original_question(monkeypatch):
    # Decision 3. The cross-encoder judges whether a real passage answers the
    # user's question, not whether it resembles a guess.
    encoder = StubCrossEncoder({})
    lane = hyde_with(monkeypatch, StubCorpus(), StubGroq(passage="invented text"))
    monkeypatch.setattr("backend.lanes.hyde.get_cross_encoder", lambda n: encoder)
    lane.retrieve("the original question", k=2)
    assert {query for query, _ in encoder.seen_pairs} == {"the original question"}


def test_hyde_records_the_measured_token_counts(monkeypatch):
    lane = hyde_with(monkeypatch, StubCorpus(), StubGroq(usage=(157, 82)))
    monkeypatch.setattr("backend.lanes.hyde.get_cross_encoder", lambda n: StubCrossEncoder({}))
    result = lane.retrieve("q", k=2)
    assert (result.prompt_tokens, result.completion_tokens) == (157, 82)
    assert result.tokens_used == 239


def test_hyde_retries_once_on_a_transient_failure(monkeypatch):
    monkeypatch.setattr("backend.lanes.hyde.time.sleep", lambda s: None)
    client = StubGroq(fail_times=1)
    lane = hyde_with(monkeypatch, StubCorpus(), client)
    monkeypatch.setattr("backend.lanes.hyde.get_cross_encoder", lambda n: StubCrossEncoder({}))
    lane.retrieve("q", k=2)
    assert client.calls == 2


def test_hyde_gives_up_after_the_second_failure(monkeypatch):
    monkeypatch.setattr("backend.lanes.hyde.time.sleep", lambda s: None)
    client = StubGroq(fail_times=5)
    lane = hyde_with(monkeypatch, StubCorpus(), client)
    with pytest.raises(HydeGenerationError):
        lane.retrieve("q", k=2)
    assert client.calls == 2


def test_hyde_never_silently_falls_back_to_the_plain_question(monkeypatch):
    # The failure that would be worst: lane 5 quietly becoming lane 4 and
    # being published as HyDE. It must raise instead.
    monkeypatch.setattr("backend.lanes.hyde.time.sleep", lambda s: None)
    lane = hyde_with(monkeypatch, StubCorpus(), StubGroq(fail_times=9))
    with pytest.raises(HydeGenerationError, match="could not generate"):
        lane.retrieve("q", k=2)


def test_hyde_treats_an_empty_passage_as_a_failure(monkeypatch):
    monkeypatch.setattr("backend.lanes.hyde.time.sleep", lambda s: None)
    lane = hyde_with(monkeypatch, StubCorpus(), StubGroq(passage="   "))
    with pytest.raises(HydeGenerationError, match="empty passage"):
        lane.retrieve("q", k=2)


def test_hyde_requests_low_reasoning_effort(monkeypatch):
    # gpt-oss bills reasoning tokens and the default effort blows the 700 ms
    # budget. Asserted because it is a one-word change that silently triples
    # the lane's cost and latency.
    client = StubGroq()
    lane = hyde_with(monkeypatch, StubCorpus(), client)
    monkeypatch.setattr("backend.lanes.hyde.get_cross_encoder", lambda n: StubCrossEncoder({}))
    lane.retrieve("q", k=2)
    assert client.kwargs["reasoning_effort"] == "low"
    assert client.kwargs["temperature"] == 0.0


def test_hyde_traces_generation_first(monkeypatch):
    lane = hyde_with(monkeypatch, StubCorpus(), StubGroq())
    monkeypatch.setattr("backend.lanes.hyde.get_cross_encoder", lambda n: StubCrossEncoder({}))
    trace = lane.retrieve("q", k=2).stages
    assert [s.name for s in trace] == ["hyde", "bm25", "embed", "pgvector", "rrf", "rerank"]


# --------------------------------------------------------------------------
# Cross-lane invariants
# --------------------------------------------------------------------------


def test_every_offline_lane_reports_zero_cost_and_a_positive_latency(patched_cross_encoder):
    patched_cross_encoder({})
    for lane_id, factory in REGISTRY.items():
        if lane_id in REMOTE_LANES:
            continue
        result = factory(StubCorpus()).retrieve("q", k=3)
        assert result.tokens_used == 0, lane_id
        assert result.latency_ms >= 0.0, lane_id


def test_every_offline_lane_respects_k(patched_cross_encoder):
    patched_cross_encoder({})
    for lane_id, factory in REGISTRY.items():
        if lane_id in REMOTE_LANES:
            continue
        assert len(factory(StubCorpus()).retrieve("q", k=2).chunks) <= 2, lane_id


def test_every_stage_trace_is_well_formed(patched_cross_encoder):
    patched_cross_encoder({})
    for lane_id, factory in REGISTRY.items():
        if lane_id in REMOTE_LANES:
            continue
        for stage in factory(StubCorpus()).retrieve("q", k=3).stages:
            assert isinstance(stage, StageTrace)
            assert stage.name
            assert stage.latency_ms >= 0.0
            assert stage.candidates_out >= 0
