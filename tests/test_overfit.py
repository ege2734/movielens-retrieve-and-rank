"""Single-batch overfit sanity check for the inductive hashed two-tower.

Distinct history contexts and distinct target movies → the in-batch softmax has a
perfect solution, so a healthy model drives the loss to ~0 and retrieves each user's
own target at rank 1. If this fails, the bug is in the model / loss / optimizer wiring.
Also exercises all-empty genre/tag bags (no content given). Needs torch.
"""

# Imports below intentionally follow pytest.importorskip("torch").
# pylint: disable=wrong-import-position
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataset import item_bags, user_bags  # noqa: E402
from features import HashConfig  # noqa: E402
from losses import in_batch_softmax  # noqa: E402
from model import HashedTwoTower  # noqa: E402


def test_overfit_single_batch():
    torch.manual_seed(0)
    batch = 32
    hcfg = HashConfig(id_buckets=1000, genre_buckets=16, tag_buckets=16)
    model = HashedTwoTower(hcfg, dim=16, hidden=32)

    contexts = [
        np.array([i], dtype=np.int64) for i in range(batch)
    ]  # each user watched movie i
    targets = np.arange(1000, 1000 + batch, dtype=np.int64)  # distinct target movies
    content = ({}, {})  # no genres/tags -> all-empty bags

    u_ids, u_gen, u_tag = user_bags(contexts, content, hcfg)
    i_ids, i_gen, i_tag = item_bags(targets, content, hcfg)
    target_raw = torch.from_numpy(targets)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = torch.tensor(float("inf"))
    for _ in range(300):
        user_emb = model.user(u_ids, u_gen, u_tag)
        item_emb = model.item(i_ids, i_gen, i_tag)
        loss = in_batch_softmax(user_emb, item_emb, target_raw, temperature=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model.user(u_ids, u_gen, u_tag) @ model.item(i_ids, i_gen, i_tag).t()
        top1 = (logits.argmax(dim=1) == torch.arange(batch)).float().mean().item()

    assert loss.item() < 0.05, f"loss failed to collapse: {loss.item():.4f}"
    assert top1 == 1.0, f"top-1 retrieval accuracy {top1:.3f} != 1.0"
