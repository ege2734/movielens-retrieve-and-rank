"""Cross-encoder reranker for the MovieLens slice.

The two-tower scores a (user, item) pair with a single dot product - cheap enough to
rank the whole catalog, but the two vectors never *interact* before that product. A
cross-encoder instead lets the user and item representations mix through an MLP, so it
can model interactions a dot product cannot ("this genre matters for this user only
together with that tag"). The price is that scoring is no longer a matmul, so we only
run it on a shortlist the retriever already narrowed down.

This is a *late-interaction* cross-encoder: each side is pooled independently into a
representation (item reps precompute once, user reps once per user), and the two reps
meet only in a small cross head over ``[p, q, p*q, |p-q|]`` -> score. That factoring
buys two uses from one model:

  - **reranker**: score the K retrieved candidates per user = K head evals per user;
  - **standalone retriever**: precompute all reps and (at eval cost O(users x catalog))
    score every item, to measure the quality *ceiling* of the cross head against the
    two-tower's dot product.

The tower feature plumbing is deliberately identical to `HashedTwoTower` (hashed
id/genre/tag Bags), so the reranker is inductive too and reuses `dataset`'s bag builders
(`csr_*` on CPU for eval, `build_bags_on_gpu` / `item_bags_resident` on device).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from features import HashConfig
from model import Bag


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
    )


class Ranker(nn.Module):
    """Late-interaction cross-encoder over hashed user-history / item features.

    `user_rep` / `item_rep` pool each side into a `dim`-vector (its own tables, so this
    is a standalone model, not a head bolted onto the two-tower). `score_shortlist`
    crosses a user rep with each of its candidate item reps through the cross head.
    Reps are NOT L2-normalized: the cross head, not a cosine, decides the score.
    """

    def __init__(
        self,
        hcfg: HashConfig,
        dim: int = 64,
        hidden: int = 128,
        cross_hidden: int = 128,
        use_dot_feature: bool = True,
    ):
        super().__init__()
        self.hcfg = hcfg
        self.dim = dim
        self.use_dot_feature = use_dot_feature

        # User tower: pooled hashes of history movie ids + their genres + tags.
        self.u_id = nn.EmbeddingBag(hcfg.id_buckets, dim, mode="mean")
        self.u_genre = nn.EmbeddingBag(hcfg.genre_buckets, dim, mode="mean")
        self.u_tag = nn.EmbeddingBag(hcfg.tag_buckets, dim, mode="mean")
        self.u_mlp = _mlp(3 * dim, hidden, dim)

        # Item tower: the candidate movie's own id (single) + its genres + tags.
        self.i_id = nn.Embedding(hcfg.id_buckets, dim)
        self.i_genre = nn.EmbeddingBag(hcfg.genre_buckets, dim, mode="mean")
        self.i_tag = nn.EmbeddingBag(hcfg.tag_buckets, dim, mode="mean")
        self.i_mlp = _mlp(3 * dim, hidden, dim)

        # Cross head: [p, q, p*q, |p-q|] (+ optional dot(p,q)) -> scalar score.
        cross_in = 4 * dim + (1 if use_dot_feature else 0)
        self.cross = nn.Sequential(
            nn.Linear(cross_in, cross_hidden),
            nn.ReLU(),
            nn.Linear(cross_hidden, cross_hidden),
            nn.ReLU(),
            nn.Linear(cross_hidden, 1),
        )

    def user_rep(self, hist_ids: Bag, hist_genres: Bag, hist_tags: Bag) -> Tensor:
        """[B, dim] pooled user representations from history features (un-normalized)."""
        parts = [
            self.u_id(hist_ids.values, hist_ids.offsets),
            self.u_genre(hist_genres.values, hist_genres.offsets),
            self.u_tag(hist_tags.values, hist_tags.offsets),
        ]
        return self.u_mlp(torch.cat(parts, dim=-1))

    def item_rep(self, movie_id: Tensor, genres: Bag, tags: Bag) -> Tensor:
        """[B, dim] pooled item representations. `movie_id` is [B] hashed ids."""
        parts = [
            self.i_id(movie_id),
            self.i_genre(genres.values, genres.offsets),
            self.i_tag(tags.values, tags.offsets),
        ]
        return self.i_mlp(torch.cat(parts, dim=-1))

    def cross_feats(self, p: Tensor, q: Tensor) -> Tensor:
        """Interaction features for aligned reps p, q of shape [..., dim].

        Concatenates ``[p, q, p*q, |p-q|]`` (+ optional ``dot(p, q)``) - the input the
        cross head scores. Public so callers can build all-pairs score matrices
        (in-batch training, full-catalog eval) without duplicating the feature recipe.
        """
        feats = [p, q, p * q, (p - q).abs()]
        if self.use_dot_feature:
            feats.append((p * q).sum(-1, keepdim=True))
        return torch.cat(feats, dim=-1)

    def score_shortlist(self, user_rep: Tensor, cand_reps: Tensor) -> Tensor:
        """Score each user against its own candidate shortlist.

        Args:
            user_rep:  [B, dim] one rep per user.
            cand_reps: [B, K, dim] the K candidate item reps for each user.
        Returns:
            [B, K] scores.
        """
        b, k, d = cand_reps.shape
        p = user_rep.unsqueeze(1).expand(b, k, d)
        return self.cross(self.cross_feats(p, cand_reps)).squeeze(-1)

    def score_pairs(self, user_rep: Tensor, item_rep: Tensor) -> Tensor:
        """Score aligned [N, dim] user/item reps -> [N] (one score per row)."""
        return self.cross(self.cross_feats(user_rep, item_rep)).squeeze(-1)


def fuse_scores(tower_scores: Tensor, cross_scores: Tensor, beta: float) -> Tensor:
    """Blend retriever and cross-encoder scores on a common scale, per row.

    The two-tower's cosine scores and the cross head's logits live on different scales,
    so a raw sum is meaningless. We z-normalize each score vector within a row (a user's
    candidate list) and return ``tower_z + beta * cross_z``. Reordering a shortlist by
    this fused score is what beats either signal alone: the tower supplies the strong
    prior, the cross head a correction (best around ``beta ~ 0.5`` on MovieLens-32M).

    Args:
        tower_scores: [B, K] retriever scores for each user's K candidates.
        cross_scores: [B, K] cross-encoder scores for the same candidates.
        beta:         weight on the cross-encoder correction.
    Returns:
        [B, K] fused scores (higher = better), same layout as the inputs.
    """

    def zrow(x: Tensor) -> Tensor:
        return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)

    return zrow(tower_scores) + beta * zrow(cross_scores)


@torch.no_grad()
def rerank_shortlists(
    model: Ranker,
    user_reps: Tensor,
    item_reps: Tensor,
    candidates: Tensor,
    score_chunk: int = 4096,
) -> Tensor:
    """Reorder each user's candidate list by the cross head, best-first.

    Args:
        user_reps:  [U, dim] one rep per user (from `model.user_rep`).
        item_reps:  [M, dim] all catalog item reps (from `model.item_rep`).
        candidates: [U, K] catalog indices the retriever proposed per user.
        score_chunk: how many users to score at once (bounds the [chunk, K, dim] temp).
    Returns:
        [U, K] catalog indices, reordered best-first by the reranker.
    """
    out = []
    for s in range(0, candidates.size(0), score_chunk):
        cand = candidates[s : s + score_chunk]
        cand_reps = item_reps[cand]  # [chunk, K, dim]
        scores = model.score_shortlist(user_reps[s : s + score_chunk], cand_reps)
        order = scores.argsort(dim=1, descending=True)
        out.append(torch.gather(cand, 1, order))
    return torch.cat(out, dim=0)


@torch.no_grad()
def score_full_catalog(
    model: Ranker,
    user_reps: Tensor,
    item_reps: Tensor,
    item_chunk: int = 4096,
) -> Tensor:
    """Standalone-retriever scoring: cross-head score of every user against every item.

    O(U x M) head evals - only for measuring the cross head's quality ceiling, never for
    serving. Chunks over items to bound the [U_chunk, item_chunk, dim] temporary; the
    caller is expected to pass a modest user block in `user_reps`.
    """
    u, d = user_reps.shape
    scores = user_reps.new_empty((u, item_reps.size(0)))
    for s in range(0, item_reps.size(0), item_chunk):
        q = item_reps[s : s + item_chunk]
        cand_reps = q.unsqueeze(0).expand(u, q.size(0), d)  # [U, chunk, dim]
        scores[:, s : s + item_chunk] = model.score_shortlist(user_reps, cand_reps)
    return scores
