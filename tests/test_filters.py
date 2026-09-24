"""Chapter 13: access control fails closed."""
import pytest

from lighthouse.ingestion.schema import ChunkMeta
from lighthouse.retrieval.filters import UserContext, allowed_ids


def meta(cid: str, access: str, juris: str = "GB",
         eff_to: str | None = None) -> ChunkMeta:
    return ChunkMeta(chunk_id=cid, doc_id=cid.split("#")[0],
                     collection="wordings", position=0, page=1, clause=None,
                     effective_from="2024-01-01", effective_to=eff_to,
                     jurisdiction=juris, access=access, embed_model="test")


METAS = [meta("A#0", "public"), meta("B#0", "internal"),
         meta("C#0", "restricted"), meta("D#0", "internal", juris="IE"),
         meta("E#0", "internal", eff_to="2025-01-01")]


def test_broker_cannot_reach_restricted() -> None:
    assert "C#0" not in allowed_ids(METAS, UserContext(role="broker_support"))


def test_underwriter_can() -> None:
    assert "C#0" in allowed_ids(METAS, UserContext(role="underwriter"))


def test_superseded_excluded_by_default() -> None:
    assert "E#0" not in allowed_ids(METAS, UserContext(role="underwriter"))


def test_jurisdiction_scoping() -> None:
    ctx = UserContext(role="underwriter", jurisdictions=("GB", "IE"))
    assert "D#0" in allowed_ids(METAS, ctx)
    assert "D#0" not in allowed_ids(METAS, UserContext(role="underwriter"))


def test_unknown_role_raises() -> None:
    with pytest.raises(PermissionError):
        allowed_ids(METAS, UserContext(role="typo_in_role_name"))
