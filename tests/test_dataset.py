"""The vectorized CSR bag-building must match the reference dict-based path exactly."""

# Imports below intentionally follow pytest.importorskip("torch").
# pylint: disable=wrong-import-position
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import dataset as DS  # noqa: E402
from features import HashConfig  # noqa: E402


def _same(a, b):
    torch.testing.assert_close(a.values, b.values)
    torch.testing.assert_close(a.offsets, b.offsets)


def test_csr_matches_dict_path():
    hcfg = HashConfig(id_buckets=1000, genre_buckets=64, tag_buckets=64)
    movie_genre = {
        10: np.array([1, 2]),
        20: np.array([3]),
        30: np.array([], dtype=np.int64),
    }
    movie_tag = {10: np.array([5]), 20: np.array([6, 7]), 30: np.array([8])}
    content = (movie_genre, movie_tag)

    catalog_ids = np.array([10, 20, 30], dtype=np.int64)  # dense idx 0,1,2
    dense = {10: 0, 20: 1, 30: 2}
    csr = DS.build_content_csr(catalog_ids, content, hcfg)

    contexts_raw = [np.array([10, 20]), np.array([30, 10]), np.array([20])]
    contexts_dense = [np.array([dense[int(m)] for m in c]) for c in contexts_raw]

    # user bags
    ref = DS.user_bags(contexts_raw, content, hcfg)
    got = DS.csr_user_bags(contexts_dense, csr)
    for a, b in zip(got, ref):
        _same(a, b)

    # item bags
    targets_raw = np.array([20, 10, 30])
    targets_dense = np.array([dense[int(m)] for m in targets_raw])
    ref_ids, ref_g, ref_t = DS.item_bags(targets_raw, content, hcfg)
    got_ids, got_g, got_t = DS.csr_item_bags(targets_dense, csr)
    torch.testing.assert_close(got_ids, ref_ids)
    _same(got_g, ref_g)
    _same(got_t, ref_t)


def test_build_bags_on_gpu_matches_collate():
    """The device-resident gather must reproduce the CPU Collate output exactly.

    Run on a CPU device: the resident path is device-agnostic torch, so equivalence
    here proves the same numbers on CUDA - only the buffers' location differs.
    """
    hcfg = HashConfig(id_buckets=1000, genre_buckets=64, tag_buckets=64)
    movie_genre = {
        10: np.array([1, 2]),
        20: np.array([3]),
        30: np.array([], dtype=np.int64),
        40: np.array([4, 5, 6]),
    }
    movie_tag = {
        10: np.array([5]),
        20: np.array([6, 7]),
        30: np.array([8]),
        40: np.array([], dtype=np.int64),
    }
    content = (movie_genre, movie_tag)
    catalog_ids = np.array([10, 20, 30, 40], dtype=np.int64)  # dense idx 0,1,2,3
    csr = DS.build_content_csr(catalog_ids, content, hcfg)

    histories = [  # per user, dense movie indices, time-sorted
        np.array([0, 1, 2, 3], dtype=np.int64),
        np.array([3, 2, 1, 0], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
    ]
    max_len = 2
    ex_user = np.array([0, 0, 1, 2], dtype=np.int64)
    ex_pos = np.array([3, 1, 2, 1], dtype=np.int64)

    # CPU reference: HistoryDataset slices the causal window, Collate pools it.
    hist_ds = DS.HistoryDataset(histories, ex_user, ex_pos, max_len)
    ref = DS.Collate(csr)([hist_ds[i] for i in range(len(ex_user))])

    # Resident path on a CPU device.
    tables = DS.build_resident_tables(csr, histories, torch.device("cpu"))
    got = DS.build_bags_on_gpu(
        torch.from_numpy(ex_user), torch.from_numpy(ex_pos), tables, max_len
    )

    for a, b in zip(got["user"], ref["user"]):
        _same(a, b)
    torch.testing.assert_close(got["item_ids"], ref["item_ids"])
    for a, b in zip(got["item"], ref["item"]):
        _same(a, b)
    torch.testing.assert_close(got["target_raw"], ref["target_raw"])


def test_ragged_gather_basic():
    values = np.array([10, 11, 12, 13, 14])
    offsets = np.array([0, 2, 2, 5])  # movie0=[10,11], movie1=[], movie2=[12,13,14]
    vals, lens = DS.ragged_gather(values, offsets, np.array([2, 0]))
    np.testing.assert_array_equal(vals, [12, 13, 14, 10, 11])
    np.testing.assert_array_equal(lens, [3, 2])
