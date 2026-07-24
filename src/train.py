"""Train the inductive hashed two-tower, end-to-end.

    cd src && python train.py --root ../data --batch-size 8192 --epochs 20

Flow: download -> build per-user histories + hashed content -> hold out a fraction of
*users* for eval -> train with causal (context, target) pairs and in-batch softmax ->
evaluate retrieval on the held-out users the model never saw. Single-GPU bf16 on CUDA.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import data as D
import features as Feat
from dataset import (
    ExampleIndexDataset,
    build_bags_on_gpu,
    build_content_csr,
    build_resident_tables,
    collate_indices,
    enumerate_train_examples,
    eval_examples,
    split_users,
)
from features import HashConfig
from losses import in_batch_softmax
from model import HashedTwoTower
from serve import evaluate_inductive


@dataclass
class Data:
    histories: list  # per user, dense movie indices, time-sorted
    csr: object
    is_train_user: np.ndarray
    ex_user: np.ndarray
    ex_pos: np.ndarray
    eval_contexts: list
    eval_targets: np.ndarray
    n_movies: int
    log_q: np.ndarray  # [M] log popularity per dense movie, for the logQ correction


def prepare(
    root: str, max_rows, holdout_frac: float, max_len: int, hcfg: HashConfig
) -> Data:
    ratings = D.load_ratings(root, max_rows=max_rows)
    pos = D.filter_positive(ratings.rating)
    user, movie, ts = ratings.user[pos], ratings.movie[pos], ratings.ts[pos]

    raw_histories = Feat.build_user_histories(user, movie, ts)
    catalog_ids = np.unique(movie)  # dense index = position in this sorted array
    # Remap every history to dense movie indices (vectorized via sorted searchsorted).
    histories = [
        np.searchsorted(catalog_ids, h).astype(np.int64) for h in raw_histories
    ]

    content = Feat.build_movie_content(root, hcfg)
    csr = build_content_csr(catalog_ids, content, hcfg)
    is_train_user = split_users(len(histories), holdout_frac)

    ex_user, ex_pos = enumerate_train_examples(histories, is_train_user)
    eval_contexts, eval_targets = eval_examples(histories, is_train_user, max_len)

    # Popularity of each movie among training users -> log Q for the sampling correction.
    train_movies = np.concatenate(
        [histories[u] for u in range(len(histories)) if is_train_user[u]]
    )
    freq = np.bincount(train_movies, minlength=len(catalog_ids)).astype(np.float64)
    log_q = np.log(freq + 1.0)

    return Data(
        histories,
        csr,
        is_train_user,
        ex_user,
        ex_pos,
        eval_contexts,
        eval_targets,
        len(catalog_ids),
        log_q,
    )


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    D.download(args.root)
    hcfg = HashConfig()
    ds = prepare(args.root, args.max_rows, args.holdout_frac, args.max_len, hcfg)
    device = pick_device(args.device)
    print(
        f"train_users={int(ds.is_train_user.sum())}  eval_users={len(ds.eval_contexts)}  "
        f"catalog={ds.n_movies}  train_pairs={len(ds.ex_user)}  device={device}",
        flush=True,
    )

    model = HashedTwoTower(hcfg, dim=args.dim, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Feature tables live on the device for the whole run; each step transfers only
    # the small (user, pos) index selection and gathers the pooled bags on-device.
    tables = build_resident_tables(ds.csr, ds.histories, device)
    loader = DataLoader(
        ExampleIndexDataset(ds.ex_user, ds.ex_pos),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_indices,
        pin_memory=False,
    )
    use_bf16 = device.type == "cuda"

    n_batches = len(loader)
    total_steps = n_batches * args.epochs
    warmup = min(args.warmup_steps, max(1, total_steps - 1))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    log_q_table = (
        torch.as_tensor(ds.log_q, dtype=torch.float32, device=device)
        if args.logq
        else None
    )

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = torch.zeros((), device=device)
        t0 = time.perf_counter()
        for i, (user_idx, target_pos) in enumerate(loader, start=1):
            user_idx = user_idx.to(device, non_blocking=True)
            target_pos = target_pos.to(device, non_blocking=True)
            batch = build_bags_on_gpu(user_idx, target_pos, tables, args.max_len)
            u = batch["user"]
            item_ids = batch["item_ids"]
            it = batch["item"]
            target_raw = batch["target_raw"]
            log_q = log_q_table[target_raw] if log_q_table is not None else None
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                user_emb = model.user(*u)
                item_emb = model.item(item_ids, *it)
                loss = in_batch_softmax(
                    user_emb,
                    item_emb,
                    target_raw,
                    temperature=args.temperature,
                    log_q=log_q,
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.detach()
            if args.log_every and (i % args.log_every == 0 or i == n_batches):
                rate = (i * args.batch_size) / (time.perf_counter() - t0)
                print(
                    f"  epoch {epoch}  batch {i}/{n_batches}  loss={(running / i).item():.4f}  "
                    f"lr={sched.get_last_lr()[0]:.2e}  {rate:,.0f} pairs/s",
                    flush=True,
                )
        avg = (running / n_batches).item()

        metrics = evaluate_inductive(
            model,
            ds.n_movies,
            ds.csr,
            ds.eval_contexts,
            ds.eval_targets,
            device,
            k=args.eval_k,
        )
        line = "  ".join(f"{name}={val:.4f}" for name, val in metrics.items())
        print(f"epoch {epoch}  loss={avg:.4f}  {line}", flush=True)

        score = metrics.get(args.select_metric, next(iter(metrics.values())))
        if args.save_path and score > best_score:
            best_score = score
            _save_checkpoint(args, model, epoch, metrics, score)


def _save_checkpoint(args, model, epoch: int, metrics: dict, score: float) -> None:
    """Persist the best-so-far model + its metrics and the run config."""
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": vars(args),
        },
        args.save_path,
    )
    print(
        f"  saved best ({args.select_metric}={score:.4f}) -> {args.save_path}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the inductive MovieLens two-tower retriever."
    )
    p.add_argument("--root", default="data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--batch-size", type=int, default=8192, help="bigger = more in-batch negatives"
    )
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument(
        "--lr", type=float, default=1e-2, help="peak LR; scale up with batch size"
    )
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument(
        "--logq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="popularity (logQ) correction for in-batch negatives",
    )
    p.add_argument(
        "--max-len", type=int, default=50, help="cap on history length pooled per user"
    )
    p.add_argument(
        "--holdout-frac",
        type=float,
        default=0.1,
        help="fraction of users held out for eval",
    )
    p.add_argument(
        "--eval-k", type=int, default=50, help="retrieval depth before @k cutoffs"
    )
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="cap ratings rows (low-memory laptops)",
    )
    p.add_argument("--save-path", default=None)
    p.add_argument("--select-metric", default="recall@20")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
