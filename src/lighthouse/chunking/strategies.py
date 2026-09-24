"""Four chunking strategies behind one interface.

From *Retrieval-Augmented Generation (RAG) in Production*, Chapter 6.
"""
from __future__ import annotations

from typing import Protocol

from lighthouse.ingestion.schema import ChunkMeta


class Chunker(Protocol):
    name: str

    def split(self, doc: dict) -> list[str]: ...


class FixedChunker:
    name = "fixed"

    def __init__(self, size: int = 800, overlap: int = 100):
        self.size, self.overlap = size, overlap

    def split(self, doc):
        step = self.size - self.overlap
        t = doc["text"]
        return [t[i:i + self.size] for i in range(0, len(t), step)] or [t]


class RecursiveChunker:
    name = "recursive"

    def __init__(self, size: int = 800, overlap: int = 100):
        self.size, self.overlap = size, overlap

    def split(self, doc):
        out, cur = [], ""
        for para in doc["text"].split("\n"):
            if len(cur) + len(para) + 1 > self.size and cur:
                out.append(cur)
                cur = cur[-self.overlap:] if self.overlap else ""
            cur = f"{cur}\n{para}".strip()
        if cur:
            out.append(cur)
        return out or [doc["text"]]


class StructuralChunker:
    name = "structural"

    def __init__(self, max_size: int = 1200):
        self.max_size = max_size

    def split(self, doc):
        chunks, cur = [], []
        for block in doc["blocks"]:
            opens = block["clause"] is not None
            big = sum(len(x) for x in cur) > self.max_size
            if cur and (opens or big):
                chunks.append("\n".join(cur))
                cur = []
            cur.append(block["text"])
        if cur:
            chunks.append("\n".join(cur))
        return chunks or [doc["text"]]


class ParentChunker:
    """Index child chunks, return the whole document section."""
    name = "parent"

    def __init__(self, size: int = 300):
        self.size = size

    def split(self, doc):
        out = []
        for block in doc["blocks"]:
            t = block["text"]
            out.extend(t[i:i + self.size]
                       for i in range(0, len(t), self.size)) if t else None
        return out or [doc["text"]]


def build_chunks(docs, chunker) -> tuple[list[ChunkMeta], list[str]]:
    metas, texts = [], []
    for doc in docs:
        for i, piece in enumerate(chunker.split(doc)):
            metas.append(ChunkMeta(
                chunk_id=f"{doc['doc_id']}#{i}", doc_id=doc["doc_id"],
                collection=doc["collection"], position=i,
                page=doc["blocks"][0]["page"], clause=None,
                effective_from=doc["effective_from"],
                effective_to=doc["effective_to"],
                jurisdiction=doc["jurisdiction"], access=doc["access"],
                embed_model="tfidf-svd-256"))
            texts.append(piece)
    return metas, texts
