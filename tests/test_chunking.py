"""Chapter 6: structure-aware chunking keeps clauses whole."""
from lighthouse.chunking.strategies import (FixedChunker, RecursiveChunker,
                                            StructuralChunker)

DOC = {"text": "\n".join(["7.2 Escape of water is excluded.",
                          "7.3 The exclusion does not apply where occupied."]),
       "blocks": [{"page": 1, "clause": "7.2",
                   "text": "7.2 Escape of water is excluded."},
                  {"page": 1, "clause": "7.3",
                   "text": "7.3 The exclusion does not apply where occupied."}]}


def test_structural_splits_on_clause_boundaries() -> None:
    out = StructuralChunker().split(DOC)
    assert len(out) == 2
    assert out[0].startswith("7.2") and out[1].startswith("7.3")


def test_every_strategy_returns_chunks() -> None:
    for c in (FixedChunker(), RecursiveChunker(), StructuralChunker()):
        assert c.split(DOC), c.name
