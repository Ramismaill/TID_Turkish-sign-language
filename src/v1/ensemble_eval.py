"""
ensemble_eval.py — Evaluate ST-GCN + CTR-GCN + SML ensemble on validation set

Usage (2-model, backward compatible):
  python src/ensemble_eval.py ^
    --stgcn_ckpt  "C:/AUTSL_project/checkpoints/best_stgcn.pt" ^
    --ctrgcn_ckpt "C:/AUTSL_project/checkpoints/best_ctrgcn.pt" ^
    --val_csv     "C:/AUTSL_project/landmarks/val_manifest.csv"

Usage (3-model):
  python src/ensemble_eval.py ^
    --stgcn_ckpt  "C:/AUTSL_project/checkpoints/best_stgcn.pt" ^
    --ctrgcn_ckpt "C:/AUTSL_project/checkpoints/best_ctrgcn.pt" ^
    --sml_ckpt    "C:/AUTSL_project/checkpoints/best_sml.pt" ^
    --val_csv     "C:/AUTSL_project/landmarks/val_manifest.csv"
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS
from stgcn_model import STGCN
from ctrgcn_model import CTRGCN


class ValDataset(Dataset):
    def __init__(self, val_csv):
        self.df = pd.read_csv(val_csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["npy_path"]).astype(np.float32)
        x = x[:, KEEP_INDICES]
        x = x.reshape(64, NUM_NODES, 3)
        x = x.transpose(2, 0, 1)
        label = int(row["label"])
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.long)


def stgcn_input(x):
    vel = torch.zeros_like(x)
    vel[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
    vel = vel * 10.0
    return torch.cat([x, vel], dim=1)


def sml_inputs(x):
    """Compute bone and motion from joint tensor for SML inference."""
    bone_src = [p[0] for p in BONE_PAIRS]
    bone_dst = [p[1] for p in BONE_PAIRS]
    bone = x[:, :, :, bone_src] - x[:, :, :, bone_dst]
    motion = torch.zeros_like(x)
    motion[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
    return x, bone, motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stgcn_ckpt",  required=True)
    parser.add_argument("--ctrgcn_ckpt", required=True)
    parser.add_argument("--sml_ckpt",    default=None,
                        help="Optional SML checkpoint for 3-model ensemble")
    parser.add_argument("--val_csv",     required=True)
    parser.add_argument("--batch_size",  type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    has_sml = args.sml_ckpt is not None

    # Load ST-GCN
    stgcn = STGCN(num_classes=226, dropout=0.5).to(device)
    ckpt  = torch.load(args.stgcn_ckpt, map_location=device)
    stgcn.load_state_dict(ckpt["model"])
    stgcn.eval()
    print(f"ST-GCN  loaded  -- val_acc: {ckpt.get('val_acc', 0):.4f}")

    # Load CTR-GCN
    ctrgcn = CTRGCN(num_classes=226, dropout=0.4).to(device)
    ckpt2  = torch.load(args.ctrgcn_ckpt, map_location=device)
    ctrgcn.load_state_dict(ckpt2["model"])
    ctrgcn.eval()
    print(f"CTR-GCN loaded  -- val_acc: {ckpt2.get('val_acc', 0):.4f}")

    # Load SML (optional)
    sml = None
    if has_sml:
        from sml_model import SML
        sml = SML(num_classes=226, dropout=0.3).to(device)
        ckpt3 = torch.load(args.sml_ckpt, map_location=device)
        sml.load_state_dict(ckpt3["model"])
        sml.eval()
        print(f"SML     loaded  -- val_acc: {ckpt3.get('val_acc', 0):.4f}")

    val_loader = DataLoader(ValDataset(args.val_csv),
                            batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Collect all probabilities
    all_prob_s = []
    all_prob_c = []
    all_prob_m = []
    all_y      = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with autocast("cuda"):
                logits_s = stgcn(stgcn_input(x))
                logits_c = ctrgcn(x)
            all_prob_s.append(F.softmax(logits_s, dim=1).cpu())
            all_prob_c.append(F.softmax(logits_c, dim=1).cpu())

            if sml is not None:
                joint, bone, motion = sml_inputs(x)
                with autocast("cuda"):
                    logits_m = sml(joint, bone, motion)
                all_prob_m.append(F.softmax(logits_m, dim=1).cpu())

            all_y.append(y.cpu())

    all_prob_s = torch.cat(all_prob_s, dim=0)
    all_prob_c = torch.cat(all_prob_c, dim=0)
    all_y      = torch.cat(all_y,      dim=0)
    if sml is not None:
        all_prob_m = torch.cat(all_prob_m, dim=0)

    # Individual accuracies
    acc_s = (all_prob_s.argmax(1) == all_y).float().mean().item() * 100
    acc_c = (all_prob_c.argmax(1) == all_y).float().mean().item() * 100
    print(f"\nST-GCN  alone: {acc_s:.2f}%")
    print(f"CTR-GCN alone: {acc_c:.2f}%")
    if sml is not None:
        acc_m = (all_prob_m.argmax(1) == all_y).float().mean().item() * 100
        print(f"SML     alone: {acc_m:.2f}%")

    if has_sml:
        # ── 3-model grid search ──────────────────────────────────────────
        print(f"\n{'='*60}")
        print("3-Model Weight Grid Search (w_stgcn, w_ctrgcn, w_sml):")
        print(f"{'='*60}")

        best_acc     = 0
        best_weights = (0, 0, 0)

        for w_s in np.arange(0.0, 1.05, 0.1):
            for w_c in np.arange(0.0, 1.05 - w_s, 0.1):
                w_m = round(1.0 - w_s - w_c, 1)
                if w_m < -0.01:
                    continue
                prob_ens = w_s * all_prob_s + w_c * all_prob_c + w_m * all_prob_m
                acc = (prob_ens.argmax(1) == all_y).float().mean().item() * 100
                if acc > best_acc:
                    best_acc     = acc
                    best_weights = (w_s, w_c, w_m)

        w_s, w_c, w_m = best_weights
        print(f"\nBEST 3-MODEL ENSEMBLE: {best_acc:.2f}%")
        print(f"  Weights: ST-GCN={w_s:.1f}, CTR-GCN={w_c:.1f}, SML={w_m:.1f}")
        print(f"  Gain over ST-GCN : +{best_acc - acc_s:.2f}%")
        print(f"  Gain over CTR-GCN: +{best_acc - acc_c:.2f}%")
        print(f"  Gain over SML    : +{best_acc - acc_m:.2f}%")

        # Also show best 2-model combinations
        print(f"\n{'='*60}")
        print("Best 2-model combinations for reference:")
        print(f"{'='*60}")

        # ST-GCN + CTR-GCN
        best_2 = 0
        for a in np.arange(0.0, 1.05, 0.1):
            prob = a * all_prob_s + (1-a) * all_prob_c
            acc = (prob.argmax(1) == all_y).float().mean().item() * 100
            if acc > best_2:
                best_2 = acc
        print(f"  ST-GCN + CTR-GCN: {best_2:.2f}%")

        # ST-GCN + SML
        best_2 = 0
        for a in np.arange(0.0, 1.05, 0.1):
            prob = a * all_prob_s + (1-a) * all_prob_m
            acc = (prob.argmax(1) == all_y).float().mean().item() * 100
            if acc > best_2:
                best_2 = acc
        print(f"  ST-GCN + SML    : {best_2:.2f}%")

        # CTR-GCN + SML
        best_2 = 0
        for a in np.arange(0.0, 1.05, 0.1):
            prob = a * all_prob_c + (1-a) * all_prob_m
            acc = (prob.argmax(1) == all_y).float().mean().item() * 100
            if acc > best_2:
                best_2 = acc
        print(f"  CTR-GCN + SML   : {best_2:.2f}%")

        print(f"{'='*60}")

    else:
        # ── Original 2-model alpha sweep (backward compatible) ───────────
        print(f"\n{'='*50}")
        print("Alpha Sweep (ST-GCN weight / CTR-GCN weight):")
        print(f"{'='*50}")

        best_acc   = 0
        best_alpha = 0

        for alpha in np.arange(0.0, 1.05, 0.1):
            prob_ens = alpha * all_prob_s + (1 - alpha) * all_prob_c
            acc = (prob_ens.argmax(1) == all_y).float().mean().item() * 100
            marker = " <-- BEST" if acc > best_acc else ""
            print(f"  alpha={alpha:.1f} : {acc:.2f}%{marker}")
            if acc > best_acc:
                best_acc   = acc
                best_alpha = alpha

        print(f"\n{'='*50}")
        print(f"BEST ENSEMBLE : {best_acc:.2f}% at alpha={best_alpha:.1f}")
        print(f"Gain over ST-GCN: +{best_acc - acc_s:.2f}%")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
