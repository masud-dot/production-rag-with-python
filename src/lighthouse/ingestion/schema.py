"""Chunk metadata schema.

From *Retrieval-Augmented Generation (RAG) in Production*, Chapter 5.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkMeta:
    chunk_id: str
    doc_id: str
    collection: str
    position: int
    page: int | None
    clause: str | None
    effective_from: str
    effective_to: str | None
    jurisdiction: str
    access: str
    embed_model: str
