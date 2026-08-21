"""Pure metric functions: recall@k, MRR, p95 latency, and cost per query.

Pure in the sense CLAUDE.md means it: data in, numbers out. No I/O, no
network, no globals, no config reads. Every function here takes ranked chunk
ids and gold chunk ids and returns a float, which is what makes the metric
tests able to assert against fixtures computed by hand.

The formula definitions are `docs/02_EVALUATION_SPEC.md` §2 and the docstrings
below must not drift from it -- these numbers go in the public README.

The paired significance tests at the bottom are pre-registered in §3 and belong
here for the same reason: they take two lists of per-question outcomes and
return a number, and their fixtures can be worked out on paper.
"""

from __future__ import annotations

import math
import random
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


def groundedness_rate(grounded: Sequence[bool]) -> float:
    """Fraction of answers where every sentence was supported by the context.

    Answer-level rather than sentence-level, per EVALUATION_SPEC §2: one
    unsupported sentence makes the whole answer unsafe to show, so averaging
    over sentences would let a four-sentence answer with one invention score
    0.75 and read as mostly fine.

    Takes the per-answer verdicts already computed by `judge.py`, so this
    module stays pure -- it never sees a model or a prompt.
    """
    if not grounded:
        return 0.0
    return sum(1 for flag in grounded if flag) / len(grounded)


def cohens_kappa(rater_a: Sequence[bool], rater_b: Sequence[bool]) -> float:
    """Chance-corrected agreement between two binary raters.

    Reported next to raw agreement because raw agreement is close to
    meaningless on a skewed class balance, which is exactly what a groundedness
    audit has. A judge that calls every answer grounded scores 90% agreement
    against a population that is 90% grounded, while carrying no information at
    all -- and kappa is the number that says so: it is 0.0 for that rater.

        kappa = (p_observed - p_chance) / (1 - p_chance)

    where p_chance is the probability the two raters coincide if each keeps its
    own marginal rate but decides independently.

    Worked, and this is the fixture the test asserts:
        a = [T, T, T, F], b = [T, T, F, F]
        p_observed = 3/4 = 0.75
        p_chance   = (3/4 x 2/4) + (1/4 x 2/4) = 0.375 + 0.125 = 0.5
        kappa      = (0.75 - 0.5) / (1 - 0.5) = 0.5

    Returns 0.0 when p_chance is 1 -- both raters used a single class for
    everything, so there is no agreement beyond chance to measure and the
    formula is 0/0. Reporting 1.0 there would flatter a rater that said one
    word thirty times.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(
            f"{len(rater_a)} and {len(rater_b)} judgements: these are not the same items"
        )
    if not rater_a:
        return 0.0
    n = len(rater_a)
    observed = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n
    a_rate = sum(1 for a in rater_a if a) / n
    b_rate = sum(1 for b in rater_b if b) / n
    chance = a_rate * b_rate + (1 - a_rate) * (1 - b_rate)
    if chance >= 1.0:
        return 0.0
    return (observed - chance) / (1 - chance)


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
# Paired significance tests
# --------------------------------------------------------------------------
#
# Pre-registered in EVALUATION_SPEC §3 before lane 6 was trained. These are here
# rather than in the runner because they are the same kind of thing as the rest
# of this module -- data in, numbers out, hand-checkable -- and because the
# fixtures for them can be computed on paper, which is what CLAUDE.md asks of a
# metric test.
#
# Both are *paired*: the same questions, differenced per question. At n=35 an
# unpaired comparison of two point estimates throws away the only information
# that makes the comparison possible.


def discordant_pairs(
    hits_a: Sequence[bool], hits_b: Sequence[bool]
) -> tuple[int, int]:
    """Count (a wins, b wins) over paired hit/miss outcomes.

    A "win" is a question one system got and the other did not. Questions both
    got, or both missed, carry no information about which is better and are
    exactly what McNemar discards.
    """
    if len(hits_a) != len(hits_b):
        raise ValueError(
            f"{len(hits_a)} and {len(hits_b)} outcomes: these are not the same questions"
        )
    a_wins = sum(1 for a, b in zip(hits_a, hits_b, strict=True) if a and not b)
    b_wins = sum(1 for a, b in zip(hits_a, hits_b, strict=True) if b and not a)
    return a_wins, b_wins


def mcnemar_exact(a_wins: int, b_wins: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts.

    Exact rather than the chi-squared approximation, because the approximation
    needs roughly 25 discordant pairs to be trustworthy and this benchmark will
    never have them: lane 4 misses three questions at k=10, so the discordant
    count is bounded above by a single digit. Using chi-squared here would
    report a p-value the sample cannot support.

    Under the null, each discordant pair is a fair coin, so the count of
    a-wins is Binomial(n, 0.5). The p-value is twice the tail at or beyond the
    observed extreme, capped at 1 (the doubling can exceed 1 when the split is
    near even).

    Worked, and these are the fixtures the test asserts:
        6-0  -> 2 * (1/64)              = 0.03125   significant
        3-0  -> 2 * (1/8)               = 0.25      not, and cannot be
        0-0  -> 1.0                     no discordant pairs at all
    """
    if a_wins < 0 or b_wins < 0:
        raise ValueError(f"counts must be non-negative, got {a_wins} and {b_wins}")
    total = a_wins + b_wins
    if total == 0:
        return 1.0
    extreme = max(a_wins, b_wins)
    tail = sum(math.comb(total, i) for i in range(extreme, total + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def max_detectable_wins(baseline_hits: Sequence[bool]) -> int:
    """How many discordant pairs a challenger could win at most.

    It can only win a question the baseline missed, so this is the baseline's
    miss count -- and it is the number that decides in advance whether a metric
    can reach significance at all. Reported alongside every McNemar result so a
    reader can see the ceiling rather than infer it.
    """
    return sum(1 for hit in baseline_hits if not hit)


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired per-question differences.

    Resamples *questions*, not measurements: each draw takes n questions with
    replacement and averages their differences, which is what preserves the
    pairing. The interval is then the empirical quantiles of those means.

    Percentile method rather than BCa: BCa corrects for skew and bias and is
    better, and implementing it correctly needs a jackknife and an inverse
    normal CDF -- more machinery than this module should carry to sharpen an
    interval that is dominated by n=35 either way. The plain percentile
    interval is the honest, legible choice and it is named as such wherever it
    is reported.

    Deterministic given `seed`. `random.Random` rather than numpy so this
    module keeps no dependency it does not need.
    """
    if not differences:
        raise ValueError("bootstrap of an empty sequence is undefined")
    if resamples <= 0:
        raise ValueError(f"resamples must be positive, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = random.Random(seed)
    n = len(differences)
    values = list(differences)
    means = [
        sum(rng.choice(values) for _ in range(n)) / n
        for _ in range(resamples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return (percentile(means, alpha * 100), percentile(means, (1.0 - alpha) * 100))


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence.

    Here rather than `statistics.mean` because that raises on empty input, and
    every aggregate in this module returns 0.0 for a split with no questions
    rather than making each caller handle it.
    """
    return sum(values) / len(values) if values else 0.0


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
