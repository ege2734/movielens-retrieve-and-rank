"""Tests for the hashed feature layer (features)."""

import numpy as np

import features as F


def test_hash_ids_range_and_determinism():
    ids = np.array([1, 200_001, 7, 999_999])
    a = F.hash_ids(ids, buckets=200_000)
    b = F.hash_ids(ids, buckets=200_000)
    assert a.min() >= 0 and a.max() < 200_000
    np.testing.assert_array_equal(a, b)  # deterministic
    assert a[0] == 1 and a[1] == 1  # 1 and 200_001 collide under mod 200_000


def test_hash_strings_stable_across_calls():
    tags = ["sci-fi", "Kevin Kline", "misogyny"]
    a = F.hash_strings(tags, buckets=20_000)
    b = F.hash_strings(tags, buckets=20_000)
    np.testing.assert_array_equal(a, b)  # crc32 is process-stable (unlike hash())
    assert a.min() >= 0 and a.max() < 20_000


def test_build_user_histories_time_sorted_and_grouped():
    # user 1 has 3 interactions (unsorted times), user 2 has 1
    user = np.array([1, 2, 1, 1])
    movie = np.array([10, 99, 20, 30])
    ts = np.array([300, 5, 100, 200])
    hist = F.build_user_histories(user, movie, ts)
    assert len(hist) == 2
    # user 1's movies ordered by ts 100<200<300 -> [20, 30, 10]
    np.testing.assert_array_equal(hist[0], np.array([20, 30, 10]))
    np.testing.assert_array_equal(hist[1], np.array([99]))


def test_build_user_histories_empty():
    assert not F.build_user_histories(np.array([]), np.array([]), np.array([]))
