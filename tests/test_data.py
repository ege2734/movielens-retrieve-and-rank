"""Tests for the pure transforms in data.

The pandas-backed loaders (load_ratings / load_movie_*_strings / load_movie_tags)
need the real dataset, so they're exercised by the end-to-end run. History building,
hashing, and the split live in features.py / dataset.py (tested separately).
"""

import numpy as np

import data as D


def test_filter_positive():
    ratings = np.array([5.0, 3.5, 4.0, 2.0, 4.5])
    np.testing.assert_array_equal(
        D.filter_positive(ratings), np.array([True, False, True, False, True])
    )
