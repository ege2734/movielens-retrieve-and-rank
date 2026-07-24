# MovieLens Retrieve and Rank

This is a repo for a humble retrieval & ranking solution for the [MovieLens-32M](https://grouplens.org/datasets/movielens/32m/) dataset.

## tl;dr

- Two-Tower retrieval + cross-encoder reranker, all in pytorch.
- Approach is very basic, and training is suboptimal. See [Future Work](#future-work) below.

## Results

Based on sampled, held out users. See [Validation](#validation) for more.

| stage | recall@10 | recall@20 | nDCG@20 |
|---|---|---|---|
| two-tower retrieval | 0.131 | 0.205 | 0.089 |
| **+ cross-encoder fusion rerank** | **0.141** | **0.210** | **0.092** |

## Overview

```
                    ┌──────────────┐        ┌──────────────┐
   user + history ─▶│  user tower  │        │  item tower  │◀── movie + genres + tags
                    │ (bi-encoder) │        │ (bi-encoder) │
                    └──────┬───────┘        └──────┬───────┘
                           │  embedding            │  (catalog precomputed)
                           ▼                        ▼
              stage 1 │  in-memory kNN retrieve top-K  (exact matmul + topk) │  ← recall
                    └───────────────────┬─────────────────┘
                                        ▼
              stage 2 │  cross-encoder reranker  (rescore top-K, fuse)  │  ← precision
                    └───────────────────┬─────────────────┘
                                        ▼
                                 recommendations
```

### Retrieval

Set up as a two-tower model, trained on cosine similarity only. Does full kNN, because dataset is small enough.

Areas for improvement:

- Add memorized sparse item features on item tower, also based on user interactions.
- Add an overarch at the end, and predict user rating.
- Use JaggedKeyedTensors to drastically speed up training. Currently takes ~1.5 hour, realistically should be O(minutes).

### Ranking

Late interaction cross-encoder, just sparse features.

Areas for improvement:

- Try [Wukong](https://arxiv.org/pdf/2403.02545) and similar for better scale/order of interactions. Also just fun to try other architectures.
- Train using true ratings (see [Validation](#validation)).

### Training

- **Task framing.** Implicit feedback: a rating **≥ 4** is a positive `(user, movie)`
  interaction; everything else is dropped. There is deliberately **no minimum-interaction
  filter** on users.
- **Causal, no leakage.** A training example pools a user's history *strictly before* the
  target movie, so the model never sees the answer in its own inputs.
- **In-batch softmax with logQ correction.** Doubled retrieval's recall because we were using popular media disproportionately as negatives in-batch.
- **Two-stage.** The reranker is a late-interaction cross-encoder trained with the *same*
  in-batch softmax against a **frozen** tower, then its score is fused with the tower's
  (`z-normalize each per user, tower + β·cross`, β ≈ 0.5). It reranks the tower's top-100.

### Validation

This validation approach is subpar because:

1. **MovieLens has no official held-out test set or leaderboard.** The train/eval split is
   *self-constructed*.
2. **The split is by random users, not by time.** Typically, validation should be split based on time. We split based on user here, which means validation step may include user interactions that happened around similar times as those in training time. Random-user splits are known to *overstate* performance versus temporal splits.
3. **"Rating ≥ 4 = positive" is a crude label.** Better approach is to predict ratings instead.
4. **The reranker is evaluated on a sample of held-out users, not all of them**, so its
   numbers carry more sampling noise than the retriever's.
5. **Minimal tuning, single seed.** Hyperparameters were tuned informally against this same eval signal (a mild optimistic bias), results are from a single run/seed with no
   confidence intervals, and there is no separate held-out *test* set kept untouched from
   tuning - the reported split doubles as both dev and test.

## Future Work

Some fun things to try:

- See areas for improvement under [Retrieval](#retrieval), [Ranking](#ranking) and [Validation](#validation).
- Overkill, but use [Silvertorch](https://engineering.fb.com/2026/05/26/ml-applications/silvertorch-index-as-model-new-retrieval-paradigm-recommendation-systems/) and benchmark speed.
- Use [Generative Recommenders](https://arxiv.org/pdf/2402.17152), which would combine the retrieval+ranking stage. Would work nicely here given it's a static dataset and doesn't carry much of the cold start concerns.
- Analyze "time to good recommendation", i.e. how many ratings before we get to a "decent" precision/recall for a user, a key problem in production recommender systems.

## Repository layout

Ten flat modules, read in roughly this order:

| module | what it holds |
|---|---|
| [`data.py`](src/data.py) | download ml-32m, load ratings, `filter_positive` (rating ≥ 4), raw genre/tag loaders |
| [`features.py`](src/features.py) | the hashing trick, per-user time-sorted histories, per-movie hashed content |
| [`dataset.py`](src/dataset.py) | CSR content tables + vectorized ragged gather, causal (context, target) enumeration, disjoint train/eval user split, GPU-resident batch build |
| [`model.py`](src/model.py) | `HashedTwoTower` + `Bag` (`EmbeddingBag` pooled inputs) |
| [`losses.py`](src/losses.py) | `in_batch_softmax` with accidental-hit masking + logQ correction |
| [`serve.py`](src/serve.py) | encode catalog/users, `evaluate_inductive` (exact in-memory kNN) |
| [`train.py`](src/train.py) | `prepare` + `train` + CLI (cosine LR + warmup, checkpointing, logQ) |
| [`metrics.py`](src/metrics.py) | Recall@K / nDCG@K / MRR / HR@K |
| [`rerank.py`](src/rerank.py) | `Ranker` (late-interaction cross-encoder) + `fuse_scores` + shortlist rescoring |
| [`rerank_train.py`](src/rerank_train.py) | train the cross-encoder against the frozen tower; eval retrieval vs fused vs standalone |

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cd src
# Stage 1: train the two-tower retriever. Downloads ml-32m (~250 MB) on first run.
python train.py --root ../data --epochs 20 --batch-size 8192 --logq \
  --save-path ../checkpoints/twotower.pt

# Stage 2: train the cross-encoder reranker against the frozen tower.
python rerank_train.py --root ../data --ckpt ../checkpoints/twotower.pt \
  --batch-size 1024 --max-steps 15000 --logq --fuse-beta 0.5 \
  --save-path ../checkpoints/ranker.pt
```

## Development

```bash
pip install -r requirements-dev.txt      # black + isort + pylint + pre-commit
pre-commit install                       # optional: format/lint gate on commit
python -m pytest tests/ -q               # unit tests (need torch; run from repo root)
```

## License

[MIT](LICENSE).
