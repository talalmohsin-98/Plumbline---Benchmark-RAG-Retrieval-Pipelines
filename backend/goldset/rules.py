"""The four drop rules from docs/02_EVALUATION_SPEC.md §1, in one place.

Three things judge a draft against these rules: the LLM screener
(`screen.py`), the human audit (`audit.py`), and the fully-manual verification
CLI (`verify.py`). The audit's whole purpose is to measure how often the
screener and the human agree, and that number means nothing unless both were
asked the same four questions. So the rules live here and are imported, never
retyped.

Rule 2 is the odd one out: firing it means the question is *fixable* rather
than worthless, because the label is right and only the wording gives the
answer away. Every other rule means the pair cannot be salvaged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    """One drop rule: its id, what firing it implies, and how it reads."""

    id: str
    # "drop" or "fix" -- the verdict this rule forces when it fires.
    verdict: str
    # Shown verbatim to a human auditor and paraphrased into the screener's
    # prompt. Kept short because it is on screen for every single decision.
    text: str


GENERIC = Rule(
    "generic",
    "drop",
    "Drop if the question could be answered just as well by 3+ other chunks.",
)
VERBATIM = Rule(
    "verbatim",
    "fix",
    "Fix if the question reuses more than four consecutive words from the chunk.",
)
UNANSWERABLE = Rule(
    "unanswerable",
    "drop",
    "Drop if the chunk does not state the answer.",
)
MULTI_CHUNK = Rule(
    "multi_chunk",
    "drop",
    "Drop if answering requires combining this chunk with another.",
)

# Order matters only for display. The verdict does not depend on it: a drop
# rule beats a fix rule wherever both fire (see `verdict_for`).
RULES: tuple[Rule, ...] = (GENERIC, VERBATIM, UNANSWERABLE, MULTI_CHUNK)

BY_ID: dict[str, Rule] = {rule.id: rule for rule in RULES}

# Every verdict a screener or an auditor may reach. "unscored" is not here: it
# is a screener failure state, not a judgement, and nothing may act on it.
VERDICTS: tuple[str, ...] = ("keep", "fix", "drop")


def verdict_for(fired: dict[str, bool]) -> tuple[str, str | None]:
    """Turn a per-rule scorecard into one verdict and the rule that forced it.

    Drop beats fix: a question that is both vague and verbatim is not worth
    rephrasing. Rules are consulted in `RULES` order within each class, so the
    recorded rule is deterministic when two of them fire together.
    """
    for rule in RULES:
        if rule.verdict == "drop" and fired.get(rule.id):
            return "drop", rule.id
    for rule in RULES:
        if rule.verdict == "fix" and fired.get(rule.id):
            return "fix", rule.id
    return "keep", None
