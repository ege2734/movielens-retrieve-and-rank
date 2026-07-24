"""The inductive, hashed two-tower model.

Neither tower has a per-entity id vocabulary — everything is a hashed feature pooled
into a shared embedding space. Retrieval is dot product between L2-normalized vectors.

    user = MLP( mean(hist movie-id emb) ⊕ mean(hist genre emb) ⊕ mean(hist tag emb) )
    item = MLP(      movie-id emb        ⊕ mean(genre emb)      ⊕ mean(tag emb)      )
    score(user, item) = user · item

Because the user is pooled from its *history* (no `user_id` row) and the item leans on
shared genre/tag tables, both generalize to entities unseen in training. The user-side
movie-id table and the item-side movie-id table are **separate** on purpose: "a movie
in my history" and "a movie as a candidate" learn different representations.

Variable-length pooled features (a history, a movie's genres/tags) are passed as
`Bag(values, offsets)` — the flattened-ids + per-row offsets form `nn.EmbeddingBag`
consumes. An empty bag pools to a zero vector.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from features import HashConfig


class Bag(NamedTuple):
    """A batch of variable-length id lists for `nn.EmbeddingBag`."""

    values: Tensor  # 1-D concatenation of every row's ids
    offsets: Tensor  # [B] start index of each row within `values`


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
    )


class HashedTwoTower(nn.Module):
    def __init__(self, hcfg: HashConfig, dim: int = 64, hidden: int = 128):
        super().__init__()
        self.hcfg = hcfg
        self.dim = dim

        # User tower: pooled hashes of history movie ids + their genres + their tags.
        self.u_id = nn.EmbeddingBag(hcfg.id_buckets, dim, mode="mean")
        self.u_genre = nn.EmbeddingBag(hcfg.genre_buckets, dim, mode="mean")
        self.u_tag = nn.EmbeddingBag(hcfg.tag_buckets, dim, mode="mean")
        self.u_mlp = _mlp(3 * dim, hidden, dim)

        # Item tower: the candidate movie's own id (single) + its genres + its tags.
        # Separate id table from the user side — different representation on purpose.
        self.i_id = nn.Embedding(hcfg.id_buckets, dim)
        self.i_genre = nn.EmbeddingBag(hcfg.genre_buckets, dim, mode="mean")
        self.i_tag = nn.EmbeddingBag(hcfg.tag_buckets, dim, mode="mean")
        self.i_mlp = _mlp(3 * dim, hidden, dim)

    def user(self, hist_ids: Bag, hist_genres: Bag, hist_tags: Bag) -> Tensor:
        """[B, dim] L2-normalized user vectors pooled from history features."""
        parts = [
            self.u_id(hist_ids.values, hist_ids.offsets),
            self.u_genre(hist_genres.values, hist_genres.offsets),
            self.u_tag(hist_tags.values, hist_tags.offsets),
        ]
        return F.normalize(self.u_mlp(torch.cat(parts, dim=-1)), dim=-1)

    def item(self, movie_id: Tensor, genres: Bag, tags: Bag) -> Tensor:
        """[B, dim] L2-normalized item vectors. `movie_id` is [B] hashed ids."""
        parts = [
            self.i_id(movie_id),
            self.i_genre(genres.values, genres.offsets),
            self.i_tag(tags.values, tags.offsets),
        ]
        return F.normalize(self.i_mlp(torch.cat(parts, dim=-1)), dim=-1)
