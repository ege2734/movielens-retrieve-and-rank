"""MovieLens-32M data: download, load, and turn ratings into a retrieval task.

Pipeline:
    download(root)                -> unzips ml-32m/ under root
    load_ratings(root)            -> (user, movie, rating, ts) numpy arrays
    filter_positive(rating)       -> keep rating >= 4  (implicit positives)
    load_movie_genre_strings(...) -> {movieId: [genre string]}   (for hashed features)
    load_movie_tags(...)          -> {movieId: [tag string]}     (for hashed features)

Histories, hashing, and the train/eval split live in `features.py` / `dataset.py`.
The load step uses pandas (fast for 32M rows). Task framing: implicit feedback,
rating ≥ 4 is a positive interaction.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
DATASET_DIRNAME = "ml-32m"
POSITIVE_THRESHOLD = 4.0


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download(root: str | Path = "data", force: bool = False) -> Path:
    """Download and unzip MovieLens-32M under `root`. Returns the ml-32m dir.

    Skips the download if the extracted directory already exists (unless `force`).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / DATASET_DIRNAME
    if dest.exists() and not force:
        return dest

    zip_path = root / "ml-32m.zip"
    if not zip_path.exists() or force:
        print(f"Downloading {DATASET_URL} -> {zip_path}")
        urlretrieve(DATASET_URL, zip_path)
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    return dest


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
@dataclass
class Ratings:
    """Raw interaction table as parallel numpy arrays (one row per rating)."""

    user: np.ndarray  # raw userId,  int64
    movie: np.ndarray  # raw movieId, int64
    rating: np.ndarray  # 0.5..5.0,   float32
    ts: np.ndarray  # unix seconds, int64

    def __len__(self) -> int:
        return len(self.user)


def load_ratings(root: str | Path = "data", max_rows: int | None = None) -> Ratings:
    """Read ml-32m/ratings.csv into a `Ratings` table (needs pandas).

    `max_rows` caps how many rows are read (via pandas `nrows`) — the low-memory
    escape hatch for laptops. ratings.csv is sorted by userId, so the first N rows
    are the *complete* histories of the lowest-id users, which is a clean subsample
    for a dev/smoke run rather than a random slice of half-users.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel  # transforms below don't need pandas

    path = Path(root) / DATASET_DIRNAME / "ratings.csv"
    df = pd.read_csv(
        path,
        dtype={
            "userId": np.int64,
            "movieId": np.int64,
            "rating": np.float32,
            "timestamp": np.int64,
        },
        nrows=max_rows,
    )
    return Ratings(
        user=df["userId"].to_numpy(),
        movie=df["movieId"].to_numpy(),
        rating=df["rating"].to_numpy(),
        ts=df["timestamp"].to_numpy(),
    )


# --------------------------------------------------------------------------- #
# Raw content strings (for the hashed feature model — no vocab building)
# --------------------------------------------------------------------------- #
def load_movie_genre_strings(root: str | Path = "data") -> dict[int, list[str]]:
    """Read movies.csv -> {movieId: [genre string, ...]} (empty list if untagged)."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    path = Path(root) / DATASET_DIRNAME / "movies.csv"
    df = pd.read_csv(
        path, usecols=["movieId", "genres"], dtype={"movieId": np.int64, "genres": str}
    )
    out: dict[int, list[str]] = {}
    for movie_id, raw in zip(df["movieId"].tolist(), df["genres"].tolist()):
        out[int(movie_id)] = (
            [] if not raw or raw == "(no genres listed)" else raw.split("|")
        )
    return out


def load_movie_tags(
    root: str | Path = "data", max_per_movie: int = 32
) -> dict[int, list[str]]:
    """Read tags.csv -> {movieId: [distinct tag string, ...]} capped per movie.

    Tags are user-generated free text (~140k distinct); we lowercase them and keep up
    to `max_per_movie` distinct tags per movie so a few heavily-tagged films don't
    dominate. The hashing trick downstream means we never build a tag vocabulary.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    path = Path(root) / DATASET_DIRNAME / "tags.csv"
    df = pd.read_csv(
        path, usecols=["movieId", "tag"], dtype={"movieId": np.int64, "tag": str}
    )
    df = df.dropna(subset=["tag"])
    out: dict[int, list[str]] = {}
    seen: dict[int, set[str]] = {}
    for movie_id, tag in zip(df["movieId"].tolist(), df["tag"].tolist()):
        mid = int(movie_id)
        bucket = out.setdefault(mid, [])
        if len(bucket) >= max_per_movie:
            continue
        tag = tag.lower()
        s = seen.setdefault(mid, set())
        if tag not in s:
            s.add(tag)
            bucket.append(tag)
    return out


# --------------------------------------------------------------------------- #
# Transforms (pure numpy — unit-tested without pandas/torch)
# --------------------------------------------------------------------------- #
def filter_positive(
    rating: np.ndarray, threshold: float = POSITIVE_THRESHOLD
) -> np.ndarray:
    """Boolean mask of positive interactions (rating >= threshold)."""
    return np.asarray(rating) >= threshold
