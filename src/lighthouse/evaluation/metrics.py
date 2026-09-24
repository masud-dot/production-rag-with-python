"""Retrieval metrics.

From *Retrieval-Augmented Generation (RAG) in Production*, Chapter 17.
"""
from __future__ import annotations

import math


def hit_rate(ranked: list[str], gold: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & gold else 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 1.0
    return len(set(ranked[:k]) & gold) / len(gold)


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / k


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for pos, cid in enumerate(ranked, start=1):
        if cid in gold:
            return 1 / pos
    return 0.0


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float:
    def dcg(items: list[int]) -> float:
        return sum((2 ** g - 1) / math.log2(i + 1)
                   for i, g in enumerate(items, start=1))
    actual = dcg([grades.get(c, 0) for c in ranked[:k]])
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0
