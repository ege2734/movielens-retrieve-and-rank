"""Hashed features + per-user histories for the inductive two-tower model.

The model has **no per-entity id vocabulary**: movie ids, genres, and tags are all
mapped into fixed-size embedding tables by the *hashing trick* — `hash(x) % buckets`.
This deletes the remap/vocab-building step, gives fixed memory, and never crashes on
an id or tag unseen in training (it just lands in some bucket). Collisions are the
price; with enough buckets they're rare.

Both towers are built entirely from these shared-vocabulary features:
* user  = pooled hashes of the movies in their history + those movies' genres/tags
* item  = the candidate movie's own hashed id + genres + tags

so a user or movie never seen in training still gets a meaningful (non-random) vector
from its content. The per-tower id tables are kept **separate** (user-history movie
ids vs. item-tower movie ids) so each learns its own representation.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

import data as D


@dataclass
class HashConfig:
    """Bucket counts for each hashed feature table."""

    id_buckets: int = 200_000  # movie ids (~87k distinct -> few collisions)
    genre_buckets: int = 512  # only ~19 genres; roomy so genres rarely collide
    tag_buckets: int = 20_000  # ~140k distinct tags folded into 20k buckets


def hash_ids(ids: np.ndarray, buckets: int) -> np.ndarray:
    """Hash integer ids into `[0, buckets)`. Integer ids just need a modulo."""
    return (np.asarray(ids, dtype=np.int64) % buckets).astype(np.int64)


def hash_strings(strings: list[str], buckets: int) -> np.ndarray:
    """Hash strings into `[0, buckets)` with a *stable* hash (crc32, not Python hash()).

    Python's built-in `hash()` is salted per process, so it can't be used — the same
    tag would map to different buckets across runs. crc32 is deterministic and fast.
    """
    return np.array(
        [zlib.crc32(s.encode("utf-8")) % buckets for s in strings], dtype=np.int64
    )


def build_user_histories(
    user: np.ndarray, movie: np.ndarray, ts: np.ndarray
) -> list[np.ndarray]:
    """Group positives into per-user, time-sorted movie-id sequences.

    Returns a list (one entry per distinct user) of raw movie ids ordered oldest ->
    newest. Downstream, each user's sequence is sliced into causal (context, target)
    pairs: the last item is the test target, the second-to-last is val, and earlier
    positions are training targets whose context is the items *before* them.
    """
    user = np.asarray(user)
    movie = np.asarray(movie)
    ts = np.asarray(ts)
    if len(user) == 0:
        return []
    order = np.lexsort((ts, user))  # sort by user, then time ascending
    sorted_user, sorted_movie = user[order], movie[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_user)) + 1))
    return np.split(sorted_movie, starts[1:])


def build_movie_content(
    root: str, hcfg: HashConfig
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Precompute per-movie hashed genre ids and hashed tag ids, keyed by raw movie id.

    Returns (movie_genre_hashes, movie_tag_hashes). A movie missing from tags.csv (or
    with no genres) simply gets an empty array — the pooling downstream handles it.
    """
    genre_strings = D.load_movie_genre_strings(root)
    tag_strings = D.load_movie_tags(root)

    movie_genre = {
        m: hash_strings(gs, hcfg.genre_buckets) for m, gs in genre_strings.items()
    }
    movie_tag = {m: hash_strings(ts, hcfg.tag_buckets) for m, ts in tag_strings.items()}
    return movie_genre, movie_tag
