# movielens-retrieve-and-rank

A small, **readable** two-stage recommender on MovieLens-32M, in pure PyTorch - built
to be *read top to bottom*, not just run. Stage 1 is an inductive **two-tower retriever**
(bi-encoder) that scores the whole catalog with a dot product; stage 2 is a
**cross-encoder fusion reranker** that rescores the retriever's shortlist. No framework
magic, no config inheritance - ten flat modules under [`src/`](src) you can follow without
a debugger.

> **Read this first - what this is and is not.** This is a *simple, pedagogical*
> implementation. The numbers below come from a validation setup that only *loosely*
> models how a recommender would really be evaluated, and is **far from rigorous**. See
> [Validation approach and its caveats](#validation-approach-and-its-caveats) before you
> read anything into the metrics. Do not treat these as benchmark results.

## The one architecture

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

Retrieve-then-rerank is the whole point. The **bi-encoder** trades precision for speed so
it can score every movie; the **cross-encoder** is expensive but only ever sees K
candidates. The reranker's score is *fused* with the retriever's rather than replacing it -
a cross-encoder is a strong reranker but a weak retriever (more on that below).

### What makes it interesting (the sparse-feature part)
Unlike text-IR retrievers, these towers lean on **sparse/ID features** - embedding tables
over movie ids, genres, and tags, all via the **hashing trick** (a modulo/CRC32 hash, so
there is no vocabulary to build). The user side has no `user_id` row: a user is **pooled
from their interaction history** (the movie ids they liked, plus those movies' genres and
tags). Pooling the user from history rather than a learned id row is what makes the towers
**inductive** - they produce sensible embeddings for users *and* movies never seen during
training.

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

There is no package to install - the modules import each other by bare name and run from
inside `src/`. `data/` (the dataset) and `checkpoints/` are gitignored; the download flow
recreates the data on first run.

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

Handy flags: `--max-rows N` (subsample for low-memory laptops), `--no-logq` (ablate the
popularity correction), `--device cpu`, `--holdout-frac` (fraction of users held out for
eval). Both scripts print retrieval (and, for the reranker, fused) metrics at the end.

## Training approach
- **Task framing.** Implicit feedback: a rating **≥ 4** is a positive `(user, movie)`
  interaction; everything else is dropped. There is deliberately **no minimum-interaction
  filter** on users.
- **Inductive, fully feature-hashed.** No `user_id` / `movie_id` vocabularies. The user
  tower pools the hashed features of the user's history; the item tower embeds the movie id
  + its genres + tags. Separate id tables are learned for the history side vs the item side.
- **Causal, no leakage.** A training example pools a user's history *strictly before* the
  target movie, so the model never sees the answer in its own inputs.
- **In-batch softmax with logQ correction.** Negatives are the other positives in the
  batch (thousands of them, for free), with accidental-hit masking. In-batch negatives are
  popularity-biased - popular movies appear as negatives constantly and get suppressed,
  even though popular movies are disproportionately what users watch next. Subtracting
  `log(popularity)` from the negatives' logits (**logQ correction**, on by default) fixes
  this and was the single biggest quality lever in this project.
- **Two-stage.** The reranker is a late-interaction cross-encoder trained with the *same*
  in-batch softmax against a **frozen** tower, then its score is fused with the tower's
  (`z-normalize each per user, tower + β·cross`, β ≈ 0.5). It reranks the tower's top-100.
- **Compute.** The model is tiny; a single T4 GPU trains the tower in ~1.5 hr (20 epochs).
  A GPU-resident batch build keeps feature tables on-device so each step ships only tiny
  index tensors. Exact in-memory kNN (matmul + topk) is used for retrieval - at a ~10⁵
  catalog an ANN index buys nothing.

## Results
MovieLens-32M, full 32M ratings, trained on a single T4. Each user's next liked movie is
retrieved from the full catalog. **Read the [caveats](#validation-approach-and-its-caveats)
before interpreting these.**

### Stage 1 - two-tower retriever (and the logQ ablation)
Evaluated on **held-out users** (disjoint from training), retrieving from the full ~55k
catalog:

| model | held-out users | recall@10 | recall@20 | nDCG@20 | MRR |
|---|---|---|---|---|---|
| two-tower | 20,042 | 0.059 | 0.093 | 0.039 | 0.026 |
| **+ logQ correction** | 20,042 | **0.134** | **0.205** | **0.090** | **0.062** |

The logQ popularity correction roughly **doubles** every metric.

### Stage 2 - cross-encoder fusion reranker
The two-tower scores a `(user, movie)` pair with one dot product; the cross-encoder lets
the two representations interact through an MLP, catching signal a dot product cannot. It
reranks the tower's top-100, and its score is fused with the tower's. On a held-out-user
sample:

| stage | recall@10 | recall@20 | nDCG@20 |
|---|---|---|---|
| two-tower retrieval | 0.131 | 0.205 | 0.089 |
| **+ cross-encoder fusion rerank** | **0.141** | **0.210** | **0.092** |
| _Δ_ | _+7.4%_ | _+2.3%_ | _+3.5%_ |

The lift concentrates at the **top** of the list (recall@10, nDCG) - exactly where a
reranker earns its cost. Two findings shaped the design: **(1)** training the cross-encoder
only on the retriever's hard negatives fails (it just learns to spot the injected positive,
which doesn't transfer) - in-batch negatives are what teach it a real scorer; **(2)** the
cross-encoder *alone* is a **weaker retriever** than the tower (~0.181 recall@20) because it
can't use thousands of in-batch negatives cheaply. So it is kept as a fusion reranker on top
of the tower, never a replacement - the textbook "strong reranker, weak retriever".

## Validation approach and its caveats
**This is the important section.** The evaluation here is a reasonable *sanity signal*, not
a rigorous benchmark, and the numbers should be read that way.

**How it is evaluated.** A fraction of **users** is held out entirely (disjoint from
training) so that "generalizes to unseen users" is actually tested - this is the inductive
claim, and it is the one thing the setup does take seriously. For each held-out user, one of
their liked movies is the target (leave-one-out), and we measure whether the retriever/
reranker surfaces it from the full catalog (Recall@K, nDCG@K, MRR; with one target per user,
recall@k equals hit-rate@k).

**Why this is only a loose model of "real" validation - caveats, roughly in order of how
much they matter:**

1. **MovieLens has no official held-out test set or leaderboard.** The train/eval split is
   *self-constructed*. There is no external, agreed-upon benchmark to check these numbers
   against, so they are not comparable to published results and could reflect quirks of this
   particular split.
2. **The split is by random users, not by time.** A production recommender is judged on the
   *future*: train on everything up to time T, predict interactions after T. Here the target
   can be chronologically *earlier* than interactions used elsewhere in training, so there is
   mild information bleed at the population level (popularity, co-occurrence patterns) even
   though each individual example is causal. Random-user splits are known to *overstate*
   performance versus temporal splits.
3. **"Rating ≥ 4 = positive" is a crude label.** Explicit ratings are not implicit
   engagement; a real system optimizes clicks/watches/dwell, and the ≥ 4 threshold is an
   arbitrary stand-in for "the user liked it".
4. **Leave-one-out with a single target is a weak, high-variance estimate** of ranking
   quality, and it ignores that many un-interacted movies a user *would* have liked are
   scored as misses (unobserved-positive / missing-not-at-random bias). Offline retrieval
   metrics are only loosely correlated with what a real recommender optimizes for anyway.
5. **The reranker is evaluated on a *sample* of held-out users, not all of them**, so its
   numbers carry more sampling noise than the retriever's.
6. **Minimal tuning, single seed.** Hyperparameters were tuned informally against this same
   eval signal (a mild optimistic bias), results are from a single run/seed with no
   confidence intervals, and there is no separate held-out *test* set kept untouched from
   tuning - the reported split doubles as both dev and test.

None of this makes the numbers meaningless - the logQ effect and the retrieve-then-rerank
lift are large and directionally believable. But treat them as *"this pipeline learns
something real and the pieces behave as theory predicts"*, **not** as a leaderboard-grade
measurement of recommendation quality.

## Design principles
1. **Readable > clever.** A newcomer should follow any file without a debugger.
2. **One abstraction, reused.** Tower / Ranker / Retriever - the same handful of types.
3. **Serving is first-class.** Retrieval + rerank ship together, not as an afterthought.
4. **Be honest about evaluation.** See the caveats above.

## Development
```bash
pip install -r requirements-dev.txt      # black + isort + pylint + pre-commit
pre-commit install                       # optional: format/lint gate on commit
python -m pytest tests/ -q               # unit tests (need torch; run from repo root)
```
Formatting and lint config (Black line-length 88, isort `black` profile, pylint) live in
[`pyproject.toml`](pyproject.toml) - the single source of truth for your editor, the CLI,
and CI. `tests/` imports the flat `src/` modules via the root [`conftest.py`](conftest.py).

## License
[MIT](LICENSE).
