"""Train the cross-encoder `Ranker` and serve it as a fusion reranker.

The two-tower retriever is a strong, cheap first stage; the cross-encoder's job is to
*correct* its top-K ordering, not to replace it. Two findings drove this design (see the
experiment log in the repo history):

  1. Training the cross-encoder only on the retriever's hard negatives fails - it learns
     to spot the odd injected positive, which doesn't transfer to reordering a shortlist
     of uniformly-plausible items. Training it with **in-batch softmax** (every other
     item in the batch is a negative, + logQ popularity correction), exactly like the
     two-tower, gives it dense signal and it learns a real scorer.
  2. Cross-encoder scores *alone* rank worse than the two-tower (a cross-encoder can't
     use thousands of in-batch negatives cheaply, so as a standalone retriever it is
     weaker). But **fusing** the two scores - z-normalize each per user, add
     `beta * cross` to the tower score - beats the retriever, most at the top of the
     list (r@10, nDCG). The tower supplies the prior, the cross head the correction.

So the served pipeline is: two-tower retrieves top-K -> cross-encoder rescoring ->
`fuse_scores` reorders. This module trains the cross-encoder for that pipeline and
reports retrieval vs fused (vs the cross-encoder standalone, for reference).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import data as D
from dataset import (
    ExampleIndexDataset,
    build_bags_on_gpu,
    build_resident_tables,
    collate_indices,
    csr_item_bags,
    csr_user_bags,
)
from features import HashConfig
from metrics import evaluate_ranking
from model import HashedTwoTower
from rerank import Ranker, fuse_scores, score_full_catalog
from serve import _to, encode_catalog
from train import pick_device, prepare


def load_frozen_tower(ckpt_path: str, hcfg: HashConfig, device) -> HashedTwoTower:
    """Load a trained two-tower checkpoint and freeze it (the first-stage retriever)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    tower = HashedTwoTower(
        hcfg, dim=cfg.get("dim", 64), hidden=cfg.get("hidden", 128)
    ).to(device)
    tower.load_state_dict(ckpt["model"])
    tower.eval()
    for prm in tower.parameters():
        prm.requires_grad_(False)
    return tower


def inbatch_cross_loss(
    ranker: Ranker,
    user_rep: Tensor,
    item_rep: Tensor,
    target: Tensor,
    temperature: float,
    log_q: Tensor | None,
    row_chunk: int,
) -> Tensor:
    """In-batch softmax with the cross head as scorer: row i must pick item i.

    Scores the full [B, B] cross-head matrix in row chunks (bounds the [chunk, B, feat]
    temporary), applies the logQ popularity correction, masks accidental hits
    (off-diagonal duplicates of a row's own target), and averages the per-row
    cross-entropy. This is the dense-negative signal the hard-negative-only variant
    lacked. O(B^2) head evals, so keep B moderate (~1024).
    """
    b = user_rep.size(0)
    device = user_rep.device
    total = user_rep.new_zeros(())
    for s in range(0, b, row_chunk):
        e = min(s + row_chunk, b)
        pe = user_rep[s:e].unsqueeze(1).expand(e - s, b, ranker.dim)
        qe = item_rep.unsqueeze(0).expand(e - s, b, ranker.dim)
        scores = ranker.cross(ranker.cross_feats(pe, qe)).squeeze(-1) / temperature
        if log_q is not None:
            scores = scores - log_q[target].unsqueeze(0)
        rows = torch.arange(s, e, device=device)
        same = target[s:e, None] == target[None, :]  # [chunk, B] accidental hits
        same[torch.arange(e - s, device=device), rows] = False  # keep the diagonal
        scores = scores.masked_fill(same, float("-inf"))
        total = total + F.cross_entropy(scores, rows, reduction="sum")
    return total / b


@torch.no_grad()
def evaluate_reranker(
    ranker: Ranker,
    tower: HashedTwoTower,
    catalog_tt: Tensor,
    ds,
    device,
    depth: int = 100,
    fuse_beta: float = 0.5,
    n_users: int = 6000,
    standalone_users: int = 2000,
) -> dict[str, dict[str, float]]:
    """Retrieval baseline vs fused reranker vs cross-encoder standalone, on held-out users.

    Fused reorders the tower's top-`depth` shortlist by `fuse_scores(tower, cross, beta)`.
    Standalone scores the whole catalog with the cross head (O(users x catalog), for
    reference only - never how we would serve).
    """
    ranker.eval()
    n = min(n_users, len(ds.eval_contexts))
    contexts, targets = ds.eval_contexts[:n], ds.eval_targets[:n]

    def catalog_reps(s: int, e: int) -> Tensor:
        ids, gen, tag = csr_item_bags(np.arange(s, e), ds.csr)  # CPU bags
        return ranker.item_rep(ids.to(device), _to(gen, device), _to(tag, device))

    item_reps = torch.cat(
        [
            catalog_reps(s, min(s + 8192, ds.n_movies))
            for s in range(0, ds.n_movies, 8192)
        ]
    )

    base, fused, rel = [], [], []
    for s in range(0, n, 1024):
        chunk = contexts[s : s + 1024]
        ubags = tuple(_to(b, device) for b in csr_user_bags(chunk, ds.csr))
        sims = tower.user(*ubags) @ catalog_tt.t()
        for r, ctx in enumerate(chunk):
            sims[r, ctx] = float("-inf")
        cand = sims.topk(depth, dim=1).indices  # [b, depth] tower order
        cross = ranker.score_shortlist(ranker.user_rep(*ubags), item_reps[cand])
        order = fuse_scores(sims.gather(1, cand), cross, fuse_beta).argsort(
            dim=1, descending=True
        )
        base.append(cand.cpu().numpy())
        fused.append(torch.gather(cand, 1, order).cpu().numpy())
        rel.extend({int(t)} for t in targets[s : s + len(chunk)])

    out = {
        "retrieval": evaluate_ranking(list(np.concatenate(base)), rel, ks=(10, 20)),
        "fused": evaluate_ranking(list(np.concatenate(fused)), rel, ks=(10, 20)),
        "ceiling": evaluate_ranking(list(np.concatenate(base)), rel, ks=(depth,)),
    }

    ns = min(standalone_users, n)
    stand = []
    for s in range(0, ns, 256):
        chunk = contexts[s : min(s + 256, ns)]
        ubags = tuple(_to(b, device) for b in csr_user_bags(chunk, ds.csr))
        scores = score_full_catalog(ranker, ranker.user_rep(*ubags), item_reps)
        for r, ctx in enumerate(chunk):
            scores[r, ctx] = float("-inf")
        stand.append(scores.topk(20, dim=1).indices.cpu().numpy())
    out["standalone"] = evaluate_ranking(
        list(np.concatenate(stand)), rel[:ns], ks=(10, 20)
    )
    ranker.train()
    return out


def train_reranker(args: argparse.Namespace) -> None:
    """Train the cross-encoder with in-batch softmax against a frozen two-tower."""
    D.download(args.root)
    hcfg = HashConfig()
    ds = prepare(args.root, args.max_rows, args.holdout_frac, args.max_len, hcfg)
    device = pick_device(args.device)
    tower = load_frozen_tower(args.ckpt, hcfg, device)
    catalog_tt = encode_catalog(tower, ds.n_movies, ds.csr, device)
    tables = build_resident_tables(ds.csr, ds.histories, device)
    log_q = (
        torch.as_tensor(ds.log_q, dtype=torch.float32, device=device)
        if args.logq
        else None
    )
    print(
        f"catalog={ds.n_movies}  eval_users={len(ds.eval_contexts)}  "
        f"train_pairs={len(ds.ex_user)}  device={device}",
        flush=True,
    )

    ranker = Ranker(
        hcfg, dim=args.dim, hidden=args.hidden, cross_hidden=args.cross_hidden
    ).to(device)
    opt = torch.optim.AdamW(
        ranker.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        ExampleIndexDataset(ds.ex_user, ds.ex_pos),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_indices,
    )

    def lr_scale(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    use_bf16 = device.type == "cuda"

    step, running, t0 = 0, 0.0, time.perf_counter()
    while step < args.max_steps:
        for user_idx, target_pos in loader:
            user_idx = user_idx.to(device, non_blocking=True)
            target_pos = target_pos.to(device, non_blocking=True)
            batch = build_bags_on_gpu(user_idx, target_pos, tables, args.max_len)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                p = ranker.user_rep(*batch["user"])
                q = ranker.item_rep(batch["item_ids"], *batch["item"])
                loss = inbatch_cross_loss(
                    ranker,
                    p,
                    q,
                    batch["target_raw"],
                    args.temperature,
                    log_q,
                    args.row_chunk,
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            step += 1
            if args.log_every and step % args.log_every == 0:
                rate = step * args.batch_size / (time.perf_counter() - t0)
                print(
                    f"  step {step}/{args.max_steps}  loss={running / args.log_every:.4f}"
                    f"  lr={sched.get_last_lr()[0]:.2e}  {rate:,.0f} pairs/s",
                    flush=True,
                )
                running = 0.0
            if step % args.eval_every == 0 or step >= args.max_steps:
                _report(
                    step,
                    evaluate_reranker(
                        ranker,
                        tower,
                        catalog_tt,
                        ds,
                        device,
                        depth=args.eval_depth,
                        fuse_beta=args.fuse_beta,
                        n_users=args.eval_users,
                        standalone_users=args.standalone_users,
                    ),
                )
            if step >= args.max_steps:
                break

    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": ranker.state_dict(), "config": vars(args)}, args.save_path)
        print(f"saved ranker -> {args.save_path}", flush=True)


def _report(step: int, m: dict) -> None:
    def line(name):
        d = m[name]
        return (
            f"{name:11s} r@10={d['recall@10']:.4f} "
            f"r@20={d['recall@20']:.4f} ndcg@20={d['ndcg@20']:.4f}"
        )

    depth_key = next(k for k in m["ceiling"] if k.startswith("recall@"))
    print(
        f"[step {step}]\n  {line('retrieval')}  (ceiling {depth_key}="
        f"{m['ceiling'][depth_key]:.4f})\n  {line('fused')}\n  {line('standalone')}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the cross-encoder fusion reranker.")
    p.add_argument("--root", default="data")
    p.add_argument("--ckpt", required=True, help="frozen two-tower checkpoint")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--cross-hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1024, help="B^2 cross-head cost")
    p.add_argument("--row-chunk", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--logq", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=15000)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument(
        "--fuse-beta", type=float, default=0.5, help="cross weight in fusion"
    )
    p.add_argument("--eval-depth", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-users", type=int, default=6000)
    p.add_argument("--standalone-users", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--save-path", default=None)
    return p


if __name__ == "__main__":
    train_reranker(build_parser().parse_args())
