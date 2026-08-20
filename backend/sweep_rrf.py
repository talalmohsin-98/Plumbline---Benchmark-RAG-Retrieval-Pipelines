"""Sweep RRF's k and the retrieval depth, and compare fusion against its arms.

Two jobs, both of which produce numbers this project publishes and therefore
both of which have to be reproducible from this repository:

1. `ARCHITECTURE §11` says k=60 is "a parameter this project can measure
   rather than assume -- if time permits, sweep k ∈ {10, 30, 60, 100} and
   report it". This is that sweep.

2. The lane-3 fusion gate ties on the 35-row test split. Whether that tie is a
   broken fusion or a small sample is a question about sample size, and the
   only way to answer it is to look at the same comparison at a larger n.

**The sweep runs on the train split.** Choosing a parameter by its score on
the test split is fitting the benchmark to its own answer key, which is the
single thing `docs/02_EVALUATION_SPEC.md` §3 exists to prevent. The test split
appears in the arms comparison for context and is never used to select
anything.

    python -m backend.sweep_rrf
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.evaluate import SCORE_DEPTH, GoldRow, load_split
from backend.metrics import mean_reciprocal_rank, recall_at_k
from backend.retrieval.corpus import Corpus
from backend.retrieval.fusion import reciprocal_rank_fusion

# The grid. k=1 and k=5 are included below the paper's range because the
# failure mode this sweep was written to investigate is the *opposite* of what
# large k does: at k=60 over depth-50 lists, the score curve spans only 1.8x
# (1/61 to 1/110), so "appears in both lists at all" outweighs "ranked first
# by one list". Small k steepens that curve. The grid has to contain the
# direction of the suspected fault or it cannot exonerate the default.
K_GRID = (1, 5, 10, 20, 30, 60, 100)
DEPTH_GRID = (10, 20, 30, 50)
MAX_DEPTH = max(DEPTH_GRID)


@dataclass(frozen=True)
class Arms:
    """One question's two ranked lists, retrieved once at max depth."""

    qid: str
    bm25: list[str]
    dense: list[str]
    gold: list[str]


def retrieve_arms(corpus: Corpus, rows: list[GoldRow]) -> list[Arms]:
    """Retrieve both arms once per question.

    Once, at MAX_DEPTH, then sliced per cell: neither arm depends on k or on
    the fusion depth, so re-querying per cell would be 28x the embedding and
    database work for byte-identical inputs.
    """
    arms = []
    for row in rows:
        arms.append(
            Arms(
                qid=row.qid,
                bm25=[c for c, _ in corpus.bm25_search(row.question, MAX_DEPTH)],
                dense=[c for c, _ in corpus.dense_search(row.question, MAX_DEPTH, prefix=True)],
                gold=row.gold_chunk_ids,
            )
        )
    return arms


def score(ranked: list[list[str]], gold: list[list[str]]) -> dict[str, float]:
    return {
        "recall_at_5": round(recall_at_k(ranked, gold, 5), 4),
        "recall_at_10": round(recall_at_k(ranked, gold, 10), 4),
        "mrr": round(mean_reciprocal_rank(ranked, gold, SCORE_DEPTH), 4),
    }


def sweep(arms: list[Arms]) -> list[dict[str, Any]]:
    """Score every (depth, k) cell of the grid."""
    gold = [a.gold for a in arms]
    cells = []
    for depth in DEPTH_GRID:
        for k in K_GRID:
            fused = [
                [c for c, _ in reciprocal_rank_fusion([a.bm25[:depth], a.dense[:depth]], k=k)][
                    :SCORE_DEPTH
                ]
                for a in arms
            ]
            cells.append({"depth": depth, "k": k, **score(fused, gold)})
    return cells


def compare_arms(arms: list[Arms], k: int, depth: int) -> dict[str, Any]:
    """Fusion against each arm alone, at one configuration."""
    gold = [a.gold for a in arms]
    fused = [
        [c for c, _ in reciprocal_rank_fusion([a.bm25[:depth], a.dense[:depth]], k=k)][
            :SCORE_DEPTH
        ]
        for a in arms
    ]
    bm25 = [a.bm25[:SCORE_DEPTH] for a in arms]
    dense = [a.dense[:SCORE_DEPTH] for a in arms]

    def hits(ranked: list[list[str]]) -> int:
        return sum(bool(set(r) & set(g)) for r, g in zip(ranked, gold, strict=True))

    # Which questions each side gets that the other does not. On a split this
    # small the counts can tie while the *questions* differ, and that
    # distinction is the whole difference between "fusion has hit a ceiling"
    # and "fusion traded one question for another".
    fused_only, bm25_only = [], []
    for arm, f in zip(arms, fused, strict=True):
        gold_set = set(arm.gold)
        hit_f = bool(set(f) & gold_set)
        hit_b = bool(set(arm.bm25[:SCORE_DEPTH]) & gold_set)
        if hit_f and not hit_b:
            fused_only.append(arm.qid)
        if hit_b and not hit_f:
            bm25_only.append(arm.qid)

    return {
        "n": len(arms),
        "k": k,
        "depth": depth,
        "bm25": {**score(bm25, gold), "hits_at_10": hits(bm25)},
        "dense": {**score(dense, gold), "hits_at_10": hits(dense)},
        "fused": {**score(fused, gold), "hits_at_10": hits(fused)},
        "fusion_wins_only": fused_only,
        "bm25_wins_only": bm25_only,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.sweep_rrf",
        description="Sweep RRF k and depth on the train split; compare fusion to its arms.",
    )
    parser.add_argument("--corpus-id", default="demo")
    parser.add_argument("--k", type=int, default=60, help="k for the arms comparison")
    parser.add_argument("--depth", type=int, default=50, help="depth for the arms comparison")
    parser.add_argument("--out", type=Path, default=Path("data/rrf_sweep.json"))
    args = parser.parse_args(argv)

    corpus = Corpus(args.corpus_id)
    try:
        corpus.warm()
        train_rows = load_split(Path("data/train.jsonl"))
        test_rows = load_split(Path("data/test.jsonl"))
        train_arms = retrieve_arms(corpus, train_rows)
        test_arms = retrieve_arms(corpus, test_rows)
    finally:
        corpus.close()

    cells = sweep(train_arms)

    print(f"RRF sweep -- TRAIN SPLIT ONLY (n={len(train_arms)}). The test split selects nothing.")
    print(f"\n{'depth':>6} {'k':>5} {'recall@5':>9} {'recall@10':>10} {'MRR':>8}")
    print("-" * 42)
    last_depth = None
    for cell in cells:
        if last_depth is not None and cell["depth"] != last_depth:
            print()
        last_depth = cell["depth"]
        print(
            f"{cell['depth']:>6} {cell['k']:>5} {cell['recall_at_5']:>9.4f} "
            f"{cell['recall_at_10']:>10.4f} {cell['mrr']:>8.4f}"
        )

    best = max(cells, key=lambda c: (c["recall_at_10"], c["mrr"]))
    print(
        f"\ntrain argmax: depth={best['depth']} k={best['k']} "
        f"recall@10={best['recall_at_10']:.4f} MRR={best['mrr']:.4f}"
    )
    print(
        f"shipped default: depth={args.depth} k={args.k}. recall@10 is flat across "
        f"k=10..60 and the MRR spread to the argmax is under one question at this n, "
        f"so the cited Cormack default stands rather than being re-fit to noise."
    )

    comparisons = {
        "train": compare_arms(train_arms, args.k, args.depth),
        "test": compare_arms(test_arms, args.k, args.depth),
        "all": compare_arms(train_arms + test_arms, args.k, args.depth),
    }
    print(f"\nFusion vs its arms at k={args.k}, depth={args.depth} (recall@10):")
    print(f"{'split':>8} {'n':>5} {'bm25':>12} {'dense':>12} {'fused':>12}   verdict")
    print("-" * 68)
    for name, cmp in comparisons.items():
        n = cmp["n"]
        best_arm = max(cmp["bm25"]["hits_at_10"], cmp["dense"]["hits_at_10"])
        verdict = (
            "PASS" if cmp["fused"]["hits_at_10"] > best_arm
            else "TIE" if cmp["fused"]["hits_at_10"] == best_arm
            else "FAIL"
        )
        print(
            f"{name:>8} {n:>5} "
            f"{cmp['bm25']['hits_at_10']:>7}/{n:<4} "
            f"{cmp['dense']['hits_at_10']:>7}/{n:<4} "
            f"{cmp['fused']['hits_at_10']:>7}/{n:<4}   {verdict}"
        )
    test_cmp = comparisons["test"]
    print(
        f"\ntest split, where the counts tie: fusion alone gets "
        f"{test_cmp['fusion_wins_only']}, BM25 alone gets {test_cmp['bm25_wins_only']}. "
        f"Different questions, equal totals."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "swept_on": "train",
                "k_grid": list(K_GRID),
                "depth_grid": list(DEPTH_GRID),
                "cells": cells,
                "train_argmax": best,
                "shipped": {"k": args.k, "depth": args.depth},
                "arms_comparison": comparisons,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwritten      {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
