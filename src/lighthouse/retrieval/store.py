"""Flat vector store and BM25 keyword index.

From *Retrieval-Augmented Generation (RAG) in Production*, Chapter 8 and 12.
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from lighthouse.retrieval.keyword import tokenize
from lighthouse.retrieval.types import Candidate


class FlatStore:
    def __init__(self, metas, texts, vectors):
        self.metas, self.texts, self.vectors = metas, texts, vectors

    def search(self, vector, k, allowed=None) -> list[Candidate]:
        scores = self.vectors @ vector
        order = np.argsort(scores)[::-1]
        out = []
        for i in order:
            if allowed is not None and self.metas[i].chunk_id not in allowed:
                continue
            out.append(Candidate(self.metas[i], self.texts[i],
                                 float(scores[i])))
            if len(out) >= k:
                break
        return out


class KeywordIndex:
    def __init__(self, metas, texts):
        self.metas, self.texts = metas, texts
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, question, k, allowed=None) -> list[Candidate]:
        scores = self.bm25.get_scores(tokenize(question))
        order = np.argsort(scores)[::-1]
        out = []
        for i in order:
            if allowed is not None and self.metas[i].chunk_id not in allowed:
                continue
            out.append(Candidate(self.metas[i], self.texts[i],
                                 float(scores[i])))
            if len(out) >= k:
                break
        return out


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
