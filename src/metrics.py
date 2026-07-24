"""Ranking metrics for the MovieLens retrieval task.

MovieLens is framed as implicit-feedback retrieval (rating ≥ 4 = positive), so we
report the standard top-k ranking metrics: Recall@K, nDCG@K, MRR, HR@K. Each
function scores *one query* — a ranked list of movie ids against that user's set of
held-out relevant movie ids — and returns a float. `evaluate_ranking` averages them
over a batch of users into the dict you log.

Gains are binary (a movie is relevant or it isn't). Inputs may be Python lists,
numpy arrays, or torch tensors — anything `numpy.asarray` accepts.

(AUC / LogLoss live with the CTR dataset that needs them, not here — this slice is
self-contained on purpose.)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "hit_rate_at_k",
    "evaluate_ranking",
]


def _as_id_array(ranked: Sequence | np.ndarray) -> np.ndarray:
    """Ranked predictions -> 1-D array of item ids, best-first."""
    return np.asarray(ranked).reshape(-1)


def _relevant_set(relevant: Iterable) -> set:
    """Collection of relevant ids -> a set for O(1) membership tests."""
    return set(np.asarray(list(relevant)).reshape(-1).tolist())


def recall_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    """Fraction of the relevant items that appear in the top-k.

    recall@k = |top_k(ranked) ∩ relevant| / |relevant|

    Returns 0.0 when there are no relevant items (nothing was retrievable).
    """
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    top_k = _as_id_array(ranked)[:k]
    hits = sum(1 for i in top_k.tolist() if i in rel)
    return hits / len(rel)


def hit_rate_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    """1.0 if *any* relevant item is in the top-k, else 0.0 (HR@k).

    For single-positive tasks (leave-one-out eval) HR@k and recall@k coincide.
    """
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    top_k = _as_id_array(ranked)[:k]
    return 1.0 if any(i in rel for i in top_k.tolist()) else 0.0


def mrr(ranked: Sequence, relevant: Iterable, k: int | None = None) -> float:
    """Reciprocal rank of the first relevant hit (1-indexed), else 0.0.

    mrr = 1 / rank_of_first_relevant. Optionally truncated to the top-k
    (a hit below rank k contributes 0), which gives MRR@k.
    """
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    top = _as_id_array(ranked)
    if k is not None:
        top = top[:k]
    for rank, item in enumerate(top.tolist(), start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k, with binary gains.

    DCG@k  = Σ_{i=1..k} rel_i / log2(i + 1)   with rel_i ∈ {0, 1}
    IDCG@k = DCG of the ideal ordering (all relevant items first)
    nDCG@k = DCG@k / IDCG@k  ∈ [0, 1]

    Returns 0.0 when there are no relevant items.
    """
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    top_k = _as_id_array(ranked)[:k].tolist()

    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # positions 1..k
    gains = np.array([1.0 if item in rel else 0.0 for item in top_k])
    dcg = float((gains * discounts[: len(gains)]).sum())

    n_ideal = min(len(rel), k)  # best case: relevant items fill the top
    idcg = float(discounts[:n_ideal].sum())
    return dcg / idcg if idcg > 0 else 0.0


_RANKING_FNS = {
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "hr": hit_rate_at_k,
}


def evaluate_ranking(
    ranked_lists: Sequence[Sequence],
    relevant_sets: Sequence[Iterable],
    ks: Sequence[int] = (10, 20),
    metrics: Sequence[str] = ("recall", "ndcg", "hr"),
    include_mrr: bool = True,
) -> dict[str, float]:
    """Average the ranking metrics over a batch of users.

    Args:
        ranked_lists:  one ranked movie-id list per user (best-first).
        relevant_sets: the held-out relevant movie-id set for each user.
        ks:            cutoffs to report each @k metric at.
        metrics:       which @k metrics to compute (`recall`, `ndcg`, `hr`).
        include_mrr:   also report untruncated MRR.

    Returns e.g. {"recall@10": 0.31, "ndcg@10": 0.18, "hr@10": 0.52, "mrr": 0.21}.
    """
    if len(ranked_lists) != len(relevant_sets):
        raise ValueError(
            f"ranked_lists ({len(ranked_lists)}) and relevant_sets "
            f"({len(relevant_sets)}) must be the same length."
        )
    if not ranked_lists:
        raise ValueError("need at least one user to evaluate.")

    out: dict[str, float] = {}
    for name in metrics:
        fn = _RANKING_FNS[name]
        for k in ks:
            scores = [fn(r, rel, k) for r, rel in zip(ranked_lists, relevant_sets)]
            out[f"{name}@{k}"] = float(np.mean(scores))
    if include_mrr:
        out["mrr"] = float(
            np.mean([mrr(r, rel) for r, rel in zip(ranked_lists, relevant_sets)])
        )
    return out
