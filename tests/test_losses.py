"""Tests for the in-batch softmax loss, focused on accidental-hit masking."""

# Imports below intentionally follow pytest.importorskip("torch").
# pylint: disable=wrong-import-position
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from losses import in_batch_softmax  # noqa: E402


def test_masking_lowers_loss_for_duplicate_positive_item():
    torch.manual_seed(0)
    # Same embeddings, but row 2's item is identical to row 0's. Labeling them with the
    # same id masks the accidental hit; labeling them distinct treats the identical
    # col 2 as a negative for row 0 -> strictly higher loss.
    user = F.normalize(torch.randn(3, 8), dim=-1)
    item = F.normalize(torch.randn(3, 8), dim=-1)
    item[2] = item[0]

    masked = in_batch_softmax(user, item, torch.tensor([5, 7, 5]), temperature=0.05)
    unmasked = in_batch_softmax(user, item, torch.tensor([5, 7, 9]), temperature=0.05)

    assert torch.isfinite(masked)
    assert masked.item() < unmasked.item()


def test_distinct_ids_equals_plain_cross_entropy():
    # With all-distinct ids the mask is a no-op, so the loss must equal a plain
    # softmax-cross-entropy over the cosine-similarity logits.
    torch.manual_seed(1)
    user = F.normalize(torch.randn(4, 8), dim=-1)
    item = F.normalize(torch.randn(4, 8), dim=-1)
    ids = torch.tensor([0, 1, 2, 3])

    logits = (user @ item.t()) / 0.05
    ref = F.cross_entropy(logits, torch.arange(4))
    got = in_batch_softmax(user, item, ids, temperature=0.05)
    assert got.item() == pytest.approx(ref.item(), rel=1e-5)
