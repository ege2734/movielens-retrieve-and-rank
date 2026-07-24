"""Training losses for the two-tower retriever.

The whole point of a large batch is that every *other* item in the batch is a free
negative. `in_batch_softmax` turns a batch of matched (user, item) pairs into a
[B, B] similarity matrix and asks each user to pick its own item out of the batch —
a B-way classification where the B-1 off-diagonal items are the negatives. Bigger
batch → more negatives → better retrieval.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def in_batch_softmax(
    user_emb: Tensor,
    item_emb: Tensor,
    item_ids: Tensor,
    temperature: float = 0.05,
    log_q: Tensor | None = None,
) -> Tensor:
    """In-batch sampled-softmax (InfoNCE) loss with accidental-hit masking.

    Args:
        user_emb: [B, D] L2-normalized user vectors.
        item_emb: [B, D] L2-normalized item vectors; row i is user i's positive.
        item_ids: [B] item ids for the batch. Required — we always mask **accidental
            hits**: off-diagonal columns whose item equals the row's own positive
            item. Without this, a popular item appearing in two rows makes each row
            treat the other's identical embedding as a negative (a contradiction) that
            collapses training once the batch is large relative to the catalog. When
            all items in the batch are distinct the mask is a harmless no-op, so
            there's never a reason not to pass ids.
        temperature: scales the cosine similarities before softmax (smaller = sharper).
        log_q: optional [B] log sampling probability of each in-batch item
            (≈ log popularity). Subtracted from the logits — the "logQ correction"
            that stops popular items from being unfairly penalized as negatives.

    Returns:
        Scalar cross-entropy loss where the correct class for row i is column i.
    """
    logits = user_emb @ item_emb.t() / temperature  # [B, B] cosine sims
    if log_q is not None:
        logits = logits - log_q.unsqueeze(
            0
        )  # correct for sampling the positive as a negative
    # Mask off-diagonal duplicates of each row's positive item (accidental hits).
    same_item = item_ids.unsqueeze(0) == item_ids.unsqueeze(1)  # [B, B]
    diag = torch.eye(item_ids.size(0), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same_item & ~diag, float("-inf"))
    labels = torch.arange(user_emb.size(0), device=user_emb.device)
    return F.cross_entropy(logits, labels)
