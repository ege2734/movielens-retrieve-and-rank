"""Shape + consistency checks for the cross-encoder Ranker."""

# Imports below intentionally follow pytest.importorskip("torch").
# pylint: disable=wrong-import-position
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataset import csr_item_bags  # noqa: E402
from dataset import build_content_csr, csr_user_bags  # noqa: E402
from features import HashConfig  # noqa: E402
from rerank import (  # noqa: E402
    Ranker,
    fuse_scores,
    rerank_shortlists,
    score_full_catalog,
)


def _toy():
    hcfg = HashConfig(id_buckets=1000, genre_buckets=64, tag_buckets=64)
    movie_genre = {
        0: np.array([1, 2]),
        1: np.array([3]),
        2: np.array([], dtype=np.int64),
    }
    movie_tag = {0: np.array([5]), 1: np.array([6, 7]), 2: np.array([8])}
    catalog_ids = np.array([0, 1, 2], dtype=np.int64)
    csr = build_content_csr(catalog_ids, (movie_genre, movie_tag), hcfg)
    return hcfg, csr


def test_ranker_shapes_and_shortlist_matches_pairs():
    torch.manual_seed(0)
    hcfg, csr = _toy()
    model = Ranker(hcfg, dim=16, hidden=32, cross_hidden=32)

    contexts = [np.array([0, 1]), np.array([2])]  # 2 users
    u_ids, u_gen, u_tag = csr_user_bags(contexts, csr)
    user_rep = model.user_rep(u_ids, u_gen, u_tag)
    assert user_rep.shape == (2, 16)

    i_ids, i_gen, i_tag = csr_item_bags(np.array([0, 1, 2]), csr)
    item_reps = model.item_rep(i_ids, i_gen, i_tag)  # [3, 16] all catalog
    assert item_reps.shape == (3, 16)

    # Two users, K=3 candidates each (the whole catalog here).
    candidates = torch.tensor([[0, 1, 2], [2, 0, 1]])
    cand_reps = item_reps[candidates]  # [2, 3, 16]
    shortlist_scores = model.score_shortlist(user_rep, cand_reps)  # [2, 3]
    assert shortlist_scores.shape == (2, 3)

    # score_shortlist must agree with score_pairs on the aligned pairs.
    for u in range(2):
        for j in range(3):
            pair = model.score_pairs(user_rep[u : u + 1], cand_reps[u, j : j + 1])
            torch.testing.assert_close(pair.squeeze(), shortlist_scores[u, j])


def test_fuse_scores_scale_invariant_and_beta_zero_is_tower():
    torch.manual_seed(0)
    tower = torch.randn(4, 10)
    cross = torch.randn(4, 10)

    # beta=0 -> fused order is exactly the tower order (cross ignored).
    fused0 = fuse_scores(tower, cross, beta=0.0)
    assert torch.equal(
        fused0.argsort(1, descending=True), tower.argsort(1, descending=True)
    )

    # Per-row z-norm makes fusion invariant to affine rescaling of either input.
    fused = fuse_scores(tower, cross, beta=0.5)
    fused_scaled = fuse_scores(tower * 7.0 + 3.0, cross * 0.1 - 2.0, beta=0.5)
    torch.testing.assert_close(
        fused.argsort(1, descending=True), fused_scaled.argsort(1, descending=True)
    )


def test_rerank_and_full_catalog_are_permutations():
    torch.manual_seed(0)
    hcfg, csr = _toy()
    model = Ranker(hcfg, dim=16, hidden=32, cross_hidden=32)
    i_ids, i_gen, i_tag = csr_item_bags(np.array([0, 1, 2]), csr)
    item_reps = model.item_rep(i_ids, i_gen, i_tag)
    u_ids, u_gen, u_tag = csr_user_bags([np.array([0, 1]), np.array([2])], csr)
    user_reps = model.user_rep(u_ids, u_gen, u_tag)

    candidates = torch.tensor([[0, 1, 2], [2, 0, 1]])
    reranked = rerank_shortlists(model, user_reps, item_reps, candidates)
    assert reranked.shape == candidates.shape
    for row_in, row_out in zip(candidates.tolist(), reranked.tolist()):
        assert sorted(row_in) == sorted(row_out)  # a reordering, no items invented

    full = score_full_catalog(model, user_reps, item_reps, item_chunk=2)
    assert full.shape == (2, 3)
    # Full-catalog score of a user against item j must equal the aligned pair score.
    for u in range(2):
        for j in range(3):
            pair = model.score_pairs(user_reps[u : u + 1], item_reps[j : j + 1])
            torch.testing.assert_close(pair.squeeze(), full[u, j])
