#!/usr/bin/env python3
"""
ERMA: Editability-aware Relational Multi-modal Assessment
Training script
"""
import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from dataset import ERMADataset, collate_fn
from model import EMMA


IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_WORKERS   = 4
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
TOTAL_STEPS   = 1500
WARMUP_STEPS  = 500
LOG_INTERVAL  = 50
SEED          = 42


class ERMAConfig:
    CLIP_MODEL_FOR_PROMPT = "ViT-B/32"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(pred, gt):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    gt   = np.asarray(gt,   dtype=np.float64).ravel()
    if pred.size < 2 or pred.std() < 1e-8 or gt.std() < 1e-8:
        return 0.0, 0.0
    return float(spearmanr(pred, gt).correlation), float(pearsonr(pred, gt)[0])


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def lr_lambda(warmup, total):
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return fn


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    preds, gts = [], []
    for batch in loader:
        src = batch["source"].to(device, non_blocking=True)
        sur = batch["surrogate"].to(device, non_blocking=True)
        gt  = batch["label"].to(device, non_blocking=True)
        pred = model(src, sur, batch["prompt"])
        loss = criterion(pred, gt)
        bs = gt.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds.extend(pred.float().cpu().numpy().ravel().tolist())
        gts.extend(gt.float().cpu().numpy().ravel().tolist())
    srcc, plcc = metrics(preds, gts)
    return total_loss / max(n, 1), srcc, plcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     required=True,
                        help="Root directory of the dataset (e.g. /path/to/data)")
    parser.add_argument("--train_labels",  default="labels/train_labels.json")
    parser.add_argument("--val_labels",    default="labels/val_labels.json")
    parser.add_argument("--ckpt_dir",      default="checkpoints")
    parser.add_argument("--steps",         type=int, default=TOTAL_STEPS)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_start",    type=int, default=500)
    parser.add_argument("--batch_size",    type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers",   type=int, default=NUM_WORKERS)
    parser.add_argument("--resume",        default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ERMADataset(args.train_labels, args.data_root, IMG_SIZE, train=True)
    val_ds   = ERMADataset(args.val_labels,   args.data_root, IMG_SIZE, train=False)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  batch={args.batch_size}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )

    model     = EMMA(ERMAConfig()).to(device)
    criterion = nn.MSELoss()
    optim     = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda(WARMUP_STEPS, args.steps))

    start_step = 0
    best_srcc  = -1.0
    best_plcc  = -1.0

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
            if "optim" in state:
                optim.load_state_dict(state["optim"])
            if "sched" in state:
                sched.load_state_dict(state["sched"])
            start_step = int(state.get("step", 0))
            best_srcc  = float(state.get("best_srcc", -1.0))
            best_plcc  = float(state.get("best_plcc", -1.0))
        else:
            model.load_state_dict(state)
        print(f"[resume] {args.resume}  step={start_step}")

    train_iter    = infinite_loader(train_loader)
    window_loss   = deque(maxlen=LOG_INTERVAL)
    window_pred   = deque(maxlen=LOG_INTERVAL * args.batch_size)
    window_gt     = deque(maxlen=LOG_INTERVAL * args.batch_size)
    eval_interval = args.eval_interval if args.eval_interval > 0 else args.steps
    t0 = time.time()
    model.train()

    for step in range(start_step + 1, args.steps + 1):
        batch = next(train_iter)
        src = batch["source"].to(device, non_blocking=True)
        sur = batch["surrogate"].to(device, non_blocking=True)
        gt  = batch["label"].to(device, non_blocking=True)
        pred = model(src, sur, batch["prompt"])
        loss = criterion(pred, gt)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()

        window_loss.append(loss.item())
        window_pred.extend(pred.detach().float().cpu().numpy().tolist())
        window_gt.extend(gt.detach().float().cpu().numpy().tolist())

        if step % LOG_INTERVAL == 0:
            tr_srcc, tr_plcc = metrics(list(window_pred), list(window_gt))
            dt = time.time() - t0
            print(
                f"[step {step:6d}/{args.steps}] train(win={LOG_INTERVAL}b): "
                f"loss={np.mean(window_loss):.4f} SRCC={tr_srcc:.4f} PLCC={tr_plcc:.4f} | "
                f"lr={optim.param_groups[0]['lr']:.2e} | {LOG_INTERVAL/max(dt,1e-6):.2f} step/s"
            )
            t0 = time.time()

        should_eval = (
            (step >= args.eval_start and step % eval_interval == 0)
            or step == args.steps
        )
        if should_eval:
            val_loss, val_srcc, val_plcc = evaluate(model, val_loader, criterion, device)
            srcc_improved = val_srcc > best_srcc
            plcc_improved = val_plcc > best_plcc
            print(
                f"[step {step:6d}/{args.steps}] VAL: "
                f"loss={val_loss:.4f} SRCC={val_srcc:.4f} PLCC={val_plcc:.4f} | "
                f"best_SRCC={best_srcc:.4f} best_PLCC={best_plcc:.4f}"
            )
            if srcc_improved:
                best_srcc = val_srcc
            if plcc_improved:
                best_plcc = val_plcc
            if srcc_improved or plcc_improved:
                ck = {
                    "step": step,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "sched": sched.state_dict(),
                    "best_srcc": best_srcc,
                    "best_plcc": best_plcc,
                    "val_metrics": {
                        "loss": val_loss, "srcc": val_srcc, "plcc": val_plcc,
                    },
                }
                tag = f"step{step:06d}_srcc{val_srcc:.4f}_plcc{val_plcc:.4f}"
                torch.save(ck, ckpt_dir / f"erma_{tag}.pt")
                torch.save(ck, ckpt_dir / "best.pt")
                print(f"[saved] {ckpt_dir}/erma_{tag}.pt")
            model.train()
            t0 = time.time()

    print(f"[done] best_srcc={best_srcc:.4f}  best_plcc={best_plcc:.4f}")


if __name__ == "__main__":
    main()
