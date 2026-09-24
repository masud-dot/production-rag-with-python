"""Chapter 12: identifiers must survive tokenisation."""
from lighthouse.retrieval.keyword import IDENTIFIER, tokenize


def test_identifier_stays_one_token() -> None:
    assert "end-4471-b" in tokenize("flood sub-limit under END-4471-B")


def test_naive_split_would_lose_it() -> None:
    naive = "flood sub-limit under END-4471-B".replace("-", " ").lower().split()
    assert "end-4471-b" not in naive


def test_identifier_pattern_detects() -> None:
    assert IDENTIFIER.search("see END-4471-B clause 4.2")
    assert not IDENTIFIER.search("see clause 4.2")
