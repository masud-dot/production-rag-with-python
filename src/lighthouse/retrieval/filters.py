"""Role and date scoping.

From *Retrieval-Augmented Generation (RAG) in Production*, Chapter 13.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from lighthouse.ingestion.schema import ChunkMeta


ROLE_ACCESS = {
    "broker_support": ("public", "internal"),
    "claims_handler": ("public", "internal"),
    "underwriter": ("public", "internal", "restricted"),
}


@dataclass(frozen=True)
class UserContext:
    role: str
    jurisdictions: tuple[str, ...] = ("GB",)
    as_of: str | None = None


def allowed_ids(metas: Sequence[ChunkMeta], ctx: UserContext) -> set[str]:
    levels = ROLE_ACCESS.get(ctx.role)
    if levels is None:
        raise PermissionError(f"unknown role: {ctx.role}")
    out: set[str] = set()
    for m in metas:
        if m.access not in levels:
            continue
        if m.jurisdiction not in ctx.jurisdictions:
            continue
        if ctx.as_of is None and m.effective_to is not None:
            continue
        out.add(m.chunk_id)
    return out
