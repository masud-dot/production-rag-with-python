"""Reciprocal rank fusion.

From *Retrieval-Augmented Generation (RAG) in Production*, Chapter 12.
"""
from __future__ import annotations

from lighthouse.retrieval.types import Candidate


def rrf(runs: list[list[Candidate]], k: int = 60) -> list[Candidate]:
    points: dict[str, float] = {}
    seen: dict[str, Candidate] = {}
    for run in runs:
        for rank, cand in enumerate(run, start=1):
            cid = cand.meta.chunk_id
            points[cid] = points.get(cid, 0.0) + 1 / (k + rank)
            seen.setdefault(cid, cand)
    order = sorted(points, key=lambda c: points[c], reverse=True)
    return [Candidate(seen[c].meta, seen[c].text, points[c]) for c in order]
