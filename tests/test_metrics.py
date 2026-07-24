"""Tests for metrics (ranking metrics, hand-checked)."""

import math

import pytest

import metrics as M


def test_recall_basic():
    assert M.recall_at_k([1, 2, 3, 4], {1, 3}, k=3) == 1.0
    assert M.recall_at_k([1, 2, 3, 4], {1, 3}, k=2) == 0.5


def test_recall_no_relevant_is_zero():
    assert M.recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_hit_rate():
    assert M.hit_rate_at_k([1, 2, 3], {3}, k=3) == 1.0
    assert M.hit_rate_at_k([1, 2, 3], {3}, k=2) == 0.0


def test_mrr_reciprocal_rank():
    assert M.mrr([9, 8, 7], {7}) == pytest.approx(1 / 3)
    assert M.mrr([7, 8, 9], {7}) == 1.0
    assert M.mrr([1, 2, 3], {99}) == 0.0


def test_mrr_truncated():
    assert M.mrr([9, 8, 7], {7}, k=2) == 0.0


def test_ndcg_perfect_ordering_is_one():
    assert M.ndcg_at_k([1, 2, 3, 4], {1, 2}, k=4) == pytest.approx(1.0)


def test_ndcg_single_relevant_at_rank_two():
    expected = (1 / math.log2(3)) / 1.0
    assert M.ndcg_at_k([9, 5, 8], {5}, k=3) == pytest.approx(expected)


def test_ndcg_no_relevant_is_zero():
    assert M.ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_evaluate_ranking_averages():
    ranked = [[1, 2, 3], [4, 5, 6]]
    relevant = [{1}, {6}]  # rank 1 and rank 3
    out = M.evaluate_ranking(ranked, relevant, ks=(3,), metrics=("recall", "hr"))
    assert out["recall@3"] == pytest.approx(1.0)
    assert out["hr@3"] == pytest.approx(1.0)
    assert out["mrr"] == pytest.approx((1.0 + 1 / 3) / 2)


def test_evaluate_ranking_length_mismatch_raises():
    with pytest.raises(ValueError):
        M.evaluate_ranking([[1]], [{1}, {2}])
