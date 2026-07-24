"""Serving + inductive evaluation for the hashed two-tower.

Retrieval is exact in-memory kNN: encode the whole catalog once (each movie from its
own content), then for each eval user build a query vector from their *history* and
dot-product against the catalog. Users are encoded from history (no id row), so we can
evaluate users the model never trained on.

Movies are dense-indexed (0..M-1), so a user's seen items and the target are just
catalog indices — masking seen items is a direct scatter, no id lookup.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from dataset import ContentCSR, csr_item_bags, csr_user_bags
from metrics import evaluate_ranking
from model import Bag, HashedTwoTower


def _to(bag: Bag, device) -> Bag:
    return Bag(
        bag.values.to(device, non_blocking=True),
        bag.offsets.to(device, non_blocking=True),
    )


@torch.no_grad()
def encode_catalog(model, n_movies: int, csr: ContentCSR, device, batch=8192) -> Tensor:
    """Encode every catalog movie (dense index 0..M-1) -> [M, dim]."""
    model.eval()
    out = []
    for s in range(0, n_movies, batch):
        dense = np.arange(s, min(s + batch, n_movies))
        ids, gen, tag = csr_item_bags(dense, csr)
        out.append(model.item(ids.to(device), _to(gen, device), _to(tag, device)))
    return torch.cat(out)


@torch.no_grad()
def encode_users(model, contexts, csr: ContentCSR, device) -> Tensor:
    """Encode a list of dense-indexed history contexts -> [U, dim]."""
    model.eval()
    uid, ug, ut = csr_user_bags(contexts, csr)
    return model.user(_to(uid, device), _to(ug, device), _to(ut, device))


@torch.no_grad()
def evaluate_inductive(
    model: HashedTwoTower,
    n_movies: int,
    csr: ContentCSR,
    contexts: list[np.ndarray],
    targets: np.ndarray,
    device,
    k: int = 50,
    ks: tuple[int, ...] = (10, 20),
    user_batch: int = 1024,
) -> dict[str, float]:
    """Retrieve top-k catalog movies per held-out user and score ranking metrics.

    The user's own history items are masked out of retrieval (we don't credit
    re-recommending a known movie); the target is their held-out last item.
    """
    model.eval()
    item_emb = encode_catalog(model, n_movies, csr, device)

    ranked_lists, relevant = [], []
    for s in range(0, len(contexts), user_batch):
        chunk = contexts[s : s + user_batch]
        user_emb = encode_users(model, chunk, csr, device)
        scores = user_emb @ item_emb.t()  # [b, M]
        for r, ctx in enumerate(chunk):
            scores[r, ctx] = float("-inf")  # dense indices == catalog columns
        top = scores.topk(k, dim=1).indices.cpu().numpy()
        for r in range(len(chunk)):
            ranked_lists.append(top[r].tolist())
            relevant.append({int(targets[s + r])})

    return evaluate_ranking(ranked_lists, relevant, ks=ks)
