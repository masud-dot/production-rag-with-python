"""Chapter 17 metric implementations, hand-verified in Phase 2.6."""
import math

from lighthouse.evaluation.metrics import (hit_rate, ndcg_at_k,
                                           precision_at_k, recall_at_k,
                                           reciprocal_rank)

RANKED = ["A", "B", "C", "D", "E"]
GRADES = {"B": 2, "D": 2, "G": 1}      # G sits at rank 8, outside the cut
GOLD = set(GRADES)


def test_hit_rate() -> None:
    assert hit_rate(RANKED, GOLD, 5) == 1.0
    assert hit_rate(["X", "Y"], GOLD, 5) == 0.0


def test_recall_counts_fraction_not_presence() -> None:
    assert recall_at_k(RANKED, GOLD, 5) == 2 / 3


def test_precision() -> None:
    assert precision_at_k(RANKED, GOLD, 5) == 2 / 5


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RANKED, GOLD) == 0.5
    assert reciprocal_rank(["X"], GOLD) == 0.0


def test_ndcg_matches_hand_calculation() -> None:
    dcg = 3 / math.log2(3) + 3 / math.log2(5)
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert abs(ndcg_at_k(RANKED, GRADES, 5) - dcg / ideal) < 1e-9
    assert abs(ndcg_at_k(RANKED, GRADES, 5) - 0.591) < 0.001


def test_reranking_moves_mrr_and_ndcg_not_hit_rate() -> None:
    reordered = ["B", "D", "A", "G", "C"]
    assert hit_rate(reordered, GOLD, 5) == hit_rate(RANKED, GOLD, 5)
    assert reciprocal_rank(reordered, GOLD) > reciprocal_rank(RANKED, GOLD)
    assert ndcg_at_k(reordered, GRADES, 5) > ndcg_at_k(RANKED, GRADES, 5)
