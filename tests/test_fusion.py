"""Chapter 12: RRF fuses ranks, not scores."""
from lighthouse.retrieval.fusion import rrf
from lighthouse.retrieval.types import Candidate

from tests.test_filters import meta


def cand(cid: str, score: float) -> Candidate:
    return Candidate(meta(cid, "public"), f"text {cid}", score)


def test_agreement_beats_strength_in_one_list() -> None:
    """A chunk both retrievers accept beats one only dense likes."""
    dense = [cand("A#0", 0.9)] + [cand(f"X{i}#0", 0.5) for i in range(11)] \
        + [cand("T#0", 0.4)]
    lexical = [cand("T#0", 40.0)] + [cand(f"Y{i}#0", 9.0) for i in range(20)] \
        + [cand("A#0", 3.0)]
    out = rrf([dense, lexical])
    assert out[0].meta.chunk_id == "T#0"


def test_rrf_is_symmetric_in_rank_positions() -> None:
    """Rank 1 + rank 12 scores the same as rank 12 + rank 1."""
    a = [cand("P#0", 1.0)] + [cand(f"Z{i}#0", 0.1) for i in range(11)] \
        + [cand("Q#0", 0.1)]
    b = [cand("Q#0", 1.0)] + [cand(f"W{i}#0", 0.1) for i in range(11)] \
        + [cand("P#0", 0.1)]
    out = {c.meta.chunk_id: c.score for c in rrf([a, b])}
    assert abs(out["P#0"] - out["Q#0"]) < 1e-12


def test_unbounded_scores_do_not_dominate() -> None:
    dense = [cand("A#0", 0.99)]
    lexical = [cand("B#0", 987.0)]
    out = rrf([dense, lexical])
    assert abs(out[0].score - out[1].score) < 1e-9
