#!/usr/bin/env python3
"""
ERMA: Editability-aware Relational Multi-modal Assessment
Inference / evaluation script
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ERMADataset, collate_fn
from model import EMMA


IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 4


class ERMAConfig:
    CLIP_MODEL_FOR_PROMPT = "ViT-B/32"


def compute_metrics(preds, gts):
    preds = np.asarray(preds, dtype=np.float64).ravel()
    gts   = np.asarray(gts,   dtype=np.float64).ravel()
    if preds.size < 2 or preds.std() < 1e-8 or gts.std() < 1e-8:
        return 0.0, 0.0
    srcc = float(spearmanr(preds, gts).correlation)
    plcc = float(pearsonr(preds, gts)[0])
    return srcc, plcc


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ERMA on the test set."
    )
    parser.add_argument("--data_root",   required=True,
                        help="Root directory of the dataset (e.g. /path/to/data)")
    parser.add_argument("--test_labels", default="labels/test_labels.json")
    parser.add_argument("--checkpoint",  default="pretrained/erma.pt",
                        help="Path to checkpoint (weights-only .pt or full training ckpt)")
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ERMA] device={device}")

    # Load model
    model = EMMA(ERMAConfig()).to(device)
    ckpt_path = Path(args.checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        step = state.get("step", "?")
        tm   = state.get("test_metrics", state.get("val_metrics", {}))
        print(f"[ERMA] loaded checkpoint: step={step}, "
              f"saved SRCC={tm.get('srcc','?'):.4f}, PLCC={tm.get('plcc','?'):.4f}")
    else:
        model.load_state_dict(state)
        print(f"[ERMA] loaded weights from {ckpt_path.name}")
    model.eval()

    # Dataset
    test_ds = ERMADataset(args.test_labels, args.data_root, IMG_SIZE, train=False)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    print(f"[ERMA] test set: {len(test_ds)} samples")

    # Inference
    preds, gts = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            src = batch["source"].to(device, non_blocking=True)
            sur = batch["surrogate"].to(device, non_blocking=True)
            pred = model(src, sur, batch["prompt"])
            preds.extend(pred.float().cpu().numpy().ravel().tolist())
            gts.extend(batch["label"].numpy().ravel().tolist())

    srcc, plcc = compute_metrics(preds, gts)
    print(f"\n[ERMA] Test Results")
    print(f"  SRCC : {srcc:.4f}")
    print(f"  PLCC : {plcc:.4f}")
    print(f"  n    : {len(preds)}")


if __name__ == "__main__":
    main()
