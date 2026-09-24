"""Candidate: a chunk paired with its score.

From *Retrieval-Augmented Generation (RAG) in Production*, Chapter 8.
"""
from __future__ import annotations

from dataclasses import dataclass

from lighthouse.ingestion.schema import ChunkMeta


@dataclass
class Candidate:
    meta: ChunkMeta
    text: str
    score: float
