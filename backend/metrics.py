"""Pure metric functions: recall@k, MRR, p95 latency, and cost per query.

Pure in the sense CLAUDE.md means it: data in, numbers out. No I/O, no
network, no globals, no config reads. Every function here takes ranked chunk
ids and gold chunk ids and returns a float, which is what makes the metric
tests able to assert against fixtures computed by hand.

The formula definitions are `docs/02_EVALUATION_SPEC.md` §2 and the docstrings
below must not drift from it -- these numbers go in the public README.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# --------------------------------------------------------------------------
# Per-question primitives
# --------------------------------------------------------------------------


def hit_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> bool:
    """True if any gold chunk appears in the first `k` retrieved ids.

    `gold` is a list because a question can legitimately be answered by more
    than one chunk -- five FastAPI chunks each say to install
    `python-multipart` -- and a lane that returns any of them has answered it.
    See EVALUATION_SPEC §1, "Multi-labelling duplicated answers".
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return bool(set(retrieved[:k]) & set(gold))


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1 / (rank of the first gold chunk within the top `k`), else 0.0.

    "First gold chunk" means the earliest position holding *any* gold id, not
    the position of a designated primary one. With multi-labelled rows the
    alternative would punish a lane for returning the second copy of an
    identical passage first, which is not a ranking error.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    gold_set = set(gold)
    for position, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in gold_set:
            return 1.0 / position
    return 0.0


# --------------------------------------------------------------------------
# Aggregates over a split
# --------------------------------------------------------------------------


def recall_at_k(
    retrieved_per_question: Sequence[Sequence[str]],
    gold_per_question: Sequence[Sequence[str]],
    k: int,
) -> float:
    """Fraction of questions whose top-`k` contains at least one gold chunk.

    The primary metric. If the right chunk is not retrieved, no amount of
    clever generation downstream can recover it.

    Note what this is *not*: it is not the fraction of gold chunks found. A
    question with four gold chunks scores 1.0 for finding one of them, because
    the question is answered. Calling that "recall" is the convention in the
    retrieval literature and EVALUATION_SPEC §2 states it in exactly these
    terms, so the README and the code agree.
    """
    _check_pairing(retrieved_per_question, gold_per_question)
    if not retrieved_per_question:
        return 0.0
    hits = sum(
        hit_at_k(retrieved, gold, k)
        for retrieved, gold in zip(retrieved_per_question, gold_per_question, strict=True)
    )
    return hits / len(retrieved_per_question)


def mean_reciprocal_rank(
    retrieved_per_question: Sequence[Sequence[str]],
    gold_per_question: Sequence[Sequence[str]],
    k: int = 10,
) -> float:
    """Mean of 1/(rank of first gold chunk), counting a miss as 0.

    recall@k asks *did we find it*; MRR asks *did we rank it first*. A reranker
    should barely move recall@10 while noticeably lifting MRR, and
    demonstrating that separation is why both are reported.

    `k` is explicit and defaults to 10 because MRR is only well defined
    relative to a cutoff: the same lane scored over top-10 and over top-50
    gives different numbers, and a README that says "MRR 0.62" without saying
    at what depth has not said anything. Every lane here is scored at one k.
    """
    _check_pairing(retrieved_per_question, gold_per_question)
    if not retrieved_per_question:
        return 0.0
    total = sum(
        reciprocal_rank(retrieved, gold, k)
        for retrieved, gold in zip(retrieved_per_question, gold_per_question, strict=True)
    )
    return total / len(retrieved_per_question)


# --------------------------------------------------------------------------
# Operational metrics
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """The `q`th percentile (0-100) by linear interpolation between ranks.

    This is the "linear" / R-7 definition, the same one `numpy.percentile` uses
    by default, written out here so this module keeps no dependency it does not
    need and so the fixture can be checked with a pocket calculator.

    The honest caveat that belongs with any p95 this returns: with n=35
    questions the 95th percentile sits at index 0.95 * 34 = 32.3, i.e. it
    interpolates between the 3rd- and 2nd-slowest queries. It is a tail
    estimate from a handful of samples, not a stable SLA figure.
    """
    if not values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (q / 100.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def p95_latency_ms(latencies_ms: Sequence[float]) -> float:
    """95th-percentile per-query latency. p95 rather than mean: tail is felt."""
    return percentile(latencies_ms, 95.0)


def cost_per_query_usd(
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    input_rate_per_million: float,
    output_rate_per_million: float,
) -> float:
    """Mean USD per query from *measured* token counts, never an estimate.

    Rates are per million tokens, which is how every provider publishes them;
    converting at the call site instead invites a factor-of-1000 error that
    would be invisible in a table of numbers this small.

    A lane that makes no LLM call passes empty or all-zero sequences and gets
    exactly 0.0 back. That zero is the entire argument against HyDE if HyDE
    does not earn its price.
    """
    if len(prompt_tokens) != len(completion_tokens):
        raise ValueError(
            f"token sequences must be the same length: "
            f"{len(prompt_tokens)} prompt vs {len(completion_tokens)} completion"
        )
    if not prompt_tokens:
        return 0.0
    input_cost = sum(prompt_tokens) * input_rate_per_million / 1_000_000
    output_cost = sum(completion_tokens) * output_rate_per_million / 1_000_000
    return (input_cost + output_cost) / len(prompt_tokens)


# --------------------------------------------------------------------------
# Internal
# --------------------------------------------------------------------------


def _check_pairing(
    retrieved_per_question: Sequence[Sequence[str]],
    gold_per_question: Sequence[Sequence[str]],
) -> None:
    """Refuse to score misaligned inputs.

    A silent zip-to-shortest here would score a lane against the wrong
    questions and still return a plausible-looking number, which is the worst
    failure this file could have.
    """
    if len(retrieved_per_question) != len(gold_per_question):
        raise ValueError(
            f"{len(retrieved_per_question)} retrieved lists but "
            f"{len(gold_per_question)} gold lists: these are not the same questions"
        )
