"""Batching for the inductive model: causal (context, target) pairs -> pooled Bags.

A training example is (user history *before* a target position, the target movie).
Pooling the context excludes the target, so the model can't cheat by reading the
answer out of the history. Variable-length context/content is collated into the
flattened-values + offsets form `nn.EmbeddingBag` wants (`model.Bag`).

To test the *inductive* claim we split users into disjoint train / eval sets: eval
users never appear in training, yet their vector is built from their history at eval
time. Movies are shared across both (an item is an item regardless of who saw it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from features import HashConfig, hash_ids
from model import Bag

_EMPTY = np.zeros(0, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Vectorized content lookup (CSR) — replaces per-movie Python dict lookups
# --------------------------------------------------------------------------- #
@dataclass
class ContentCSR:
    """Per-movie hashed content in compressed-sparse-row form, keyed by dense index.

    Movie `d`'s genres are `g_values[g_offsets[d]:g_offsets[d+1]]` (same for tags).
    `id_hash[d]` is that movie's hashed id. Built once from the raw dicts; lets the
    collate gather a whole batch's features with numpy fancy-indexing instead of a
    Python loop over every movie in every history.
    """

    id_hash: np.ndarray  # [M]
    g_values: np.ndarray
    g_offsets: np.ndarray  # [M + 1]
    t_values: np.ndarray
    t_offsets: np.ndarray  # [M + 1]


def _csr(dict_: dict, catalog_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arrays = [dict_.get(int(m), _EMPTY) for m in catalog_ids.tolist()]
    lengths = np.fromiter((len(a) for a in arrays), count=len(arrays), dtype=np.int64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    values = np.concatenate(arrays) if lengths.sum() > 0 else _EMPTY
    return values, offsets


def build_content_csr(
    catalog_ids: np.ndarray, content: tuple[dict, dict], hcfg: HashConfig
) -> ContentCSR:
    """Build the CSR tables. `catalog_ids` defines the dense index order (idx = position)."""
    movie_genre, movie_tag = content
    id_hash = catalog_ids.astype(np.int64) % hcfg.id_buckets
    g_values, g_offsets = _csr(movie_genre, catalog_ids)
    t_values, t_offsets = _csr(movie_tag, catalog_ids)
    return ContentCSR(id_hash, g_values, g_offsets, t_values, t_offsets)


def ragged_gather(values: np.ndarray, offsets: np.ndarray, idx: np.ndarray):
    """Concatenate values[offsets[i]:offsets[i+1]] for i in idx, vectorized.

    Returns (gathered_values, per_idx_length). No Python loop over idx.
    """
    idx = np.asarray(idx, dtype=np.int64)
    starts = offsets[idx]
    lengths = offsets[idx + 1] - starts
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=values.dtype), lengths
    out_starts = np.zeros(len(idx), dtype=np.int64)
    np.cumsum(lengths[:-1], out=out_starts[1:])
    gather = np.repeat(starts, lengths) + (
        np.arange(total) - np.repeat(out_starts, lengths)
    )
    return values[gather], lengths


def _offsets_from_lengths(lengths: np.ndarray) -> torch.Tensor:
    off = np.zeros(len(lengths), dtype=np.int64)
    if len(lengths) > 1:
        np.cumsum(lengths[:-1], out=off[1:])
    return torch.from_numpy(off)


def csr_user_bags(contexts: list[np.ndarray], csr: ContentCSR) -> tuple[Bag, Bag, Bag]:
    """Pool each dense-indexed context into id / genre / tag Bags (vectorized)."""
    ctx_len = np.fromiter(
        (len(c) for c in contexts), count=len(contexts), dtype=np.int64
    )
    ctx_flat = (
        np.concatenate(contexts).astype(np.int64) if ctx_len.sum() > 0 else _EMPTY
    )
    id_bag = Bag(
        torch.from_numpy(csr.id_hash[ctx_flat].copy()), _offsets_from_lengths(ctx_len)
    )

    def content_bag(values, offsets):
        vals, per_movie = ragged_gather(values, offsets, ctx_flat)
        # sum genre/tag counts within each example's span of movies
        per_ex = (
            np.add.reduceat(per_movie, _offsets_from_lengths(ctx_len).numpy())
            if len(ctx_flat)
            else np.zeros(len(contexts), np.int64)
        )
        return Bag(torch.from_numpy(vals.copy()), _offsets_from_lengths(per_ex))

    return (
        id_bag,
        content_bag(csr.g_values, csr.g_offsets),
        content_bag(csr.t_values, csr.t_offsets),
    )


def csr_item_bags(movie_dense: np.ndarray, csr: ContentCSR) -> tuple[Tensor, Bag, Bag]:
    """A candidate movie's hashed id (single) + its genre / tag Bags (vectorized)."""
    movie_dense = np.asarray(movie_dense, dtype=np.int64)
    ids = torch.from_numpy(csr.id_hash[movie_dense].copy())

    def content_bag(values, offsets):
        vals, per_row = ragged_gather(values, offsets, movie_dense)
        return Bag(torch.from_numpy(vals.copy()), _offsets_from_lengths(per_row))

    return (
        ids,
        content_bag(csr.g_values, csr.g_offsets),
        content_bag(csr.t_values, csr.t_offsets),
    )


# --------------------------------------------------------------------------- #
# GPU-resident batch build — collate ships only (user, pos) indices
# --------------------------------------------------------------------------- #
# The CSR content tables and per-user histories are FIXED during training; only
# which (user, target-position) pairs a batch selects changes. So we ship those
# tables to the GPU once as resident buffers and, each step, transfer only the two
# tiny index tensors and run the ragged gather ON the device. This removes the big
# per-step host->device tag payload (the profiled bottleneck) and the CPU collate.
# The gather logic mirrors the numpy CSR path above exactly, in torch.
@dataclass
class ResidentTables:
    """Content CSR + flattened histories, all as device-resident int64 tensors.

    Movie `d`'s genres are `g_values[g_offsets[d]:g_offsets[d+1]]` (tags likewise);
    user `u`'s time-sorted dense history is `hist_flat[hist_off[u]:hist_off[u+1]]`.
    Everything a training batch reads lives here, so no feature data crosses the bus
    per step - only the `(user, pos)` selection does.
    """

    id_hash: Tensor  # [M]
    g_values: Tensor
    g_offsets: Tensor  # [M + 1]
    t_values: Tensor
    t_offsets: Tensor  # [M + 1]
    hist_flat: Tensor  # concatenation of every user's dense history
    hist_off: Tensor  # [n_users + 1]


def build_resident_tables(
    csr: ContentCSR, histories: list[np.ndarray], device
) -> ResidentTables:
    """Move the CSR tables and flattened histories onto `device`, once."""
    hist_lens = np.fromiter(
        (len(h) for h in histories), count=len(histories), dtype=np.int64
    )
    hist_flat = (
        np.concatenate(histories).astype(np.int64) if hist_lens.sum() else _EMPTY
    )
    hist_off = np.zeros(len(histories) + 1, dtype=np.int64)
    np.cumsum(hist_lens, out=hist_off[1:])

    def dev(arr: np.ndarray) -> Tensor:
        return torch.as_tensor(arr, dtype=torch.int64, device=device)

    return ResidentTables(
        dev(csr.id_hash),
        dev(csr.g_values),
        dev(csr.g_offsets),
        dev(csr.t_values),
        dev(csr.t_offsets),
        dev(hist_flat),
        dev(hist_off),
    )


def _offsets_torch(lengths: Tensor) -> Tensor:
    """Per-row start offsets from per-row lengths: [0, l0, l0+l1, ...] (torch)."""
    off = torch.zeros_like(lengths)
    if lengths.numel() > 1:
        off[1:] = torch.cumsum(lengths[:-1], 0)
    return off


def _ragged_gather_torch(
    values: Tensor, offsets: Tensor, idx: Tensor
) -> tuple[Tensor, Tensor]:
    """torch twin of `ragged_gather`: concat values[offsets[i]:offsets[i+1]] for i in idx.

    Returns (gathered_values, per_idx_length). Same index arithmetic as the numpy
    version, so results match element-for-element.
    """
    starts = offsets[idx]
    lengths = offsets[idx + 1] - starts
    total = int(lengths.sum().item())
    if total == 0:
        return values.new_zeros(0), lengths
    out_starts = _offsets_torch(lengths)
    ar = torch.arange(total, device=values.device)
    gather = torch.repeat_interleave(starts, lengths) + (
        ar - torch.repeat_interleave(out_starts, lengths)
    )
    return values[gather], lengths


def _contiguous_gather(flat: Tensor, abs_start: Tensor, lengths: Tensor) -> Tensor:
    """Concat the contiguous slices flat[abs_start[j]:abs_start[j]+lengths[j]] (torch)."""
    total = int(lengths.sum().item())
    if total == 0:
        return flat.new_zeros(0)
    out_starts = _offsets_torch(lengths)
    ar = torch.arange(total, device=flat.device)
    gather = torch.repeat_interleave(abs_start, lengths) + (
        ar - torch.repeat_interleave(out_starts, lengths)
    )
    return flat[gather]


def _user_content_bag(
    values: Tensor, offsets: Tensor, ctx_flat: Tensor, ctx_len: Tensor
) -> Bag:
    """Pool a batch of contexts' genre/tag ids into one Bag (values + per-example offsets)."""
    vals, per_movie = _ragged_gather_torch(values, offsets, ctx_flat)
    # Sum each example's per-movie content counts into its own span (segment sum).
    b = ctx_len.numel()
    seg = torch.repeat_interleave(torch.arange(b, device=ctx_len.device), ctx_len)
    per_ex = torch.zeros(b, dtype=per_movie.dtype, device=ctx_len.device)
    per_ex.scatter_add_(0, seg, per_movie)
    return Bag(vals, _offsets_torch(per_ex))


def _item_content_bag(values: Tensor, offsets: Tensor, movie_dense: Tensor) -> Bag:
    """A candidate movie's genre/tag Bag (one movie per row, so no segment sum)."""
    vals, per_row = _ragged_gather_torch(values, offsets, movie_dense)
    return Bag(vals, _offsets_torch(per_row))


def item_bags_resident(
    movie_dense: Tensor, tables: ResidentTables
) -> tuple[Tensor, Bag, Bag]:
    """Hashed id + genre/tag Bags for arbitrary candidate movies, gathered on-device.

    The reranker encodes a *dynamic* shortlist of candidate movies each step, so it
    needs the item-side features for any set of dense movie indices without a CPU
    round-trip. Mirrors `csr_item_bags`, straight off the resident tables.
    """
    return (
        tables.id_hash[movie_dense],
        _item_content_bag(tables.g_values, tables.g_offsets, movie_dense),
        _item_content_bag(tables.t_values, tables.t_offsets, movie_dense),
    )


def build_bags_on_gpu(
    user_idx: Tensor, target_pos: Tensor, tables: ResidentTables, max_len: int
) -> dict:
    """Gather a batch's pooled Bags from resident tables, entirely on `tables`' device.

    `user_idx`/`target_pos` are the only per-batch input (small [B] int64 tensors).
    The causal context for example j is history[start:target_pos], start capped so at
    most `max_len` items are pooled - the same window `HistoryDataset` slices on CPU.
    Output matches `Collate`'s dict so the training loop is unchanged.
    """
    base = tables.hist_off[user_idx]
    start = torch.clamp(target_pos - max_len, min=0)
    ctx_len = target_pos - start
    abs_start = base + start
    ctx_flat = _contiguous_gather(tables.hist_flat, abs_start, ctx_len)
    target_dense = tables.hist_flat[base + target_pos]

    u_ids = Bag(tables.id_hash[ctx_flat], _offsets_torch(ctx_len))
    u_gen = _user_content_bag(tables.g_values, tables.g_offsets, ctx_flat, ctx_len)
    u_tag = _user_content_bag(tables.t_values, tables.t_offsets, ctx_flat, ctx_len)

    return {
        "user": (u_ids, u_gen, u_tag),
        "item_ids": tables.id_hash[target_dense],
        "item": (
            _item_content_bag(tables.g_values, tables.g_offsets, target_dense),
            _item_content_bag(tables.t_values, tables.t_offsets, target_dense),
        ),
        "target_raw": target_dense,  # dense ids (fine for accidental-hit masking)
    }


class ExampleIndexDataset(Dataset):
    """Yields just the (user, target-position) index pair for each training example.

    The heavy ragged gather is deferred to `build_bags_on_gpu` on resident tables, so
    the collate ships only two tiny index tensors per batch instead of pooled features.
    """

    def __init__(self, ex_user, ex_pos):
        self.ex_user = ex_user
        self.ex_pos = ex_pos

    def __len__(self):
        return len(self.ex_user)

    def __getitem__(self, i):
        return int(self.ex_user[i]), int(self.ex_pos[i])


def collate_indices(batch) -> tuple[Tensor, Tensor]:
    """Stack (user, pos) pairs into two [B] int64 tensors - the whole per-batch payload."""
    user = torch.tensor([b[0] for b in batch], dtype=torch.int64)
    pos = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    return user, pos


def split_users(n_users: int, holdout_frac: float) -> np.ndarray:
    """Deterministic boolean mask: True = training user, False = held-out eval user.

    Every k-th user (k = round(1/holdout_frac)) is held out — no RNG, reproducible.
    """
    if holdout_frac <= 0:
        return np.ones(n_users, dtype=bool)
    k = max(2, round(1 / holdout_frac))
    return (np.arange(n_users) % k) != 0


def enumerate_train_examples(
    histories: list[np.ndarray], is_train_user: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """All (user, target-position) training pairs from train users.

    For a train user with history length n, every position t in [1, n) is a target
    whose context is h[:t] (capped later). Eval users contribute nothing here.
    """
    ex_user, ex_pos = [], []
    for u, h in enumerate(histories):
        if not is_train_user[u]:
            continue
        for t in range(1, len(h)):
            ex_user.append(u)
            ex_pos.append(t)
    return np.array(ex_user, dtype=np.int64), np.array(ex_pos, dtype=np.int64)


def eval_examples(
    histories: list[np.ndarray], is_train_user: np.ndarray, max_len: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """Held-out-user eval set: (context = history[:-1] capped, target = last item)."""
    contexts, targets = [], []
    for u, h in enumerate(histories):
        if is_train_user[u] or len(h) < 2:
            continue
        contexts.append(np.asarray(h[:-1])[-max_len:])
        targets.append(int(h[-1]))
    return contexts, np.array(targets, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Bag construction
# --------------------------------------------------------------------------- #
def _bag(arrays: list[np.ndarray]) -> Bag:
    """List of per-row id arrays -> Bag(values, offsets) for nn.EmbeddingBag."""
    lengths = np.fromiter((len(a) for a in arrays), count=len(arrays), dtype=np.int64)
    offsets = np.zeros(len(arrays), dtype=np.int64)
    if len(arrays) > 1:
        offsets[1:] = np.cumsum(lengths[:-1])
    values = np.concatenate(arrays) if lengths.sum() > 0 else _EMPTY
    return Bag(torch.from_numpy(values.copy()), torch.from_numpy(offsets))


def user_bags(
    contexts: list[np.ndarray], content: tuple[dict, dict], hcfg: HashConfig
) -> tuple[Bag, Bag, Bag]:
    """Pool each context's movie ids / genres / tags into three Bags."""
    movie_genre, movie_tag = content

    def genres(ctx):
        arrs = [movie_genre.get(int(m), _EMPTY) for m in ctx]
        return np.concatenate(arrs) if arrs else _EMPTY

    def tags(ctx):
        arrs = [movie_tag.get(int(m), _EMPTY) for m in ctx]
        return np.concatenate(arrs) if arrs else _EMPTY

    ids = _bag([hash_ids(c, hcfg.id_buckets) for c in contexts])
    gen = _bag([genres(c) for c in contexts])
    tag = _bag([tags(c) for c in contexts])
    return ids, gen, tag


def item_bags(
    movie_ids: np.ndarray, content: tuple[dict, dict], hcfg: HashConfig
) -> tuple[Tensor, Bag, Bag]:
    """A candidate movie's hashed id (single) + its genre / tag Bags."""
    movie_genre, movie_tag = content
    ids = torch.from_numpy(hash_ids(movie_ids, hcfg.id_buckets))
    gen = _bag([movie_genre.get(int(m), _EMPTY) for m in movie_ids])
    tag = _bag([movie_tag.get(int(m), _EMPTY) for m in movie_ids])
    return ids, gen, tag


class HistoryDataset(Dataset):
    """Yields (context raw movie ids, target raw movie id) for each training pair."""

    def __init__(self, histories, ex_user, ex_pos, max_len):
        self.histories = histories
        self.ex_user = ex_user
        self.ex_pos = ex_pos
        self.max_len = max_len

    def __len__(self):
        return len(self.ex_user)

    def __getitem__(self, i):
        u, t = int(self.ex_user[i]), int(self.ex_pos[i])
        h = self.histories[u]
        ctx = np.asarray(h[max(0, t - self.max_len) : t])
        return ctx, int(h[t])


class Collate:
    """Collate dense-indexed (context, target) pairs into pooled Bags (vectorized CSR)."""

    def __init__(self, csr: ContentCSR):
        self.csr = csr

    def __call__(self, batch):
        contexts = [b[0] for b in batch]
        targets = np.array([b[1] for b in batch], dtype=np.int64)
        u_ids, u_gen, u_tag = csr_user_bags(contexts, self.csr)
        i_ids, i_gen, i_tag = csr_item_bags(targets, self.csr)
        return {
            "user": (u_ids, u_gen, u_tag),
            "item_ids": i_ids,
            "item": (i_gen, i_tag),
            "target_raw": torch.from_numpy(
                targets
            ),  # dense ids (fine for accidental-hit masking)
        }
