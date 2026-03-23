"""
ensemble_sml_stgcn.py — Ensemble: SML + ST-GCN (no CTR-GCN needed)

Usage:
  python src\ensemble_sml_stgcn.py ^
    --sml_ckpt   "C:/AUTSL_project/checkpoints/best_sml.pt" ^
    --stgcn_ckpt "C:/AUTSL_project/checkpoints/best_stgcn.pt" ^
    --val_csv    "C:/AUTSL_project/landmarks/val_manifest.csv"
"""

import sys
sys.path.insert(0, 'src')

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import pandas as pd

from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS
from stgcn_model import STGCN
from sml_model import SML


class ValDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["npy_path"]).astype(np.float32)
        x = x[:, KEEP_INDICES].reshape(64, NUM_NODES, 3).transpose(2, 0, 1)
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(int(row["label"]), dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sml_ckpt",   required=True)
    parser.add_argument("--stgcn_ckpt", required=True)
    parser.add_argument("--val_csv",    required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    BONE_SRC = [p[0] for p in BONE_PAIRS]
    BONE_DST = [p[1] for p in BONE_PAIRS]

    # Load SML
    sml = SML(num_classes=226, dropout=0.3).to(device)
    ckpt = torch.load(args.sml_ckpt, map_location=device, weights_only=False)
    sml.load_state_dict(ckpt["model"])
    sml.eval()
    print(f"SML    loaded — val_acc: {ckpt.get('val_acc', 0):.4f}")

    # Load ST-GCN
    stgcn = STGCN(num_classes=226, dropout=0.5).to(device)
    ckpt2 = torch.load(args.stgcn_ckpt, map_location=device, weights_only=False)
    stgcn.load_state_dict(ckpt2["model"])
    stgcn.eval()
    print(f"ST-GCN loaded — val_acc: {ckpt2.get('val_acc', 0):.4f}")

    loader = DataLoader(ValDataset(args.val_csv),
                        batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    all_prob_sml   = []
    all_prob_stgcn = []
    all_y          = []

    print("\nComputing probabilities...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            # SML inputs
            joint  = x
            bone   = joint[:, :, :, BONE_SRC] - joint[:, :, :, BONE_DST]
            motion = torch.zeros_like(joint)
            motion[:, :, 1:, :] = joint[:, :, 1:, :] - joint[:, :, :-1, :]

            # ST-GCN inputs (joint + velocity)
            vel = torch.zeros_like(x)
            vel[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
            stgcn_x = torch.cat([x, vel * 10.0], dim=1)

            with autocast("cuda"):
                logits_sml   = sml(joint, bone, motion)
                logits_stgcn = stgcn(stgcn_x)

            all_prob_sml.append(F.softmax(logits_sml,   dim=1).cpu())
            all_prob_stgcn.append(F.softmax(logits_stgcn, dim=1).cpu())
            all_y.append(y)

    prob_sml   = torch.cat(all_prob_sml,   dim=0)
    prob_stgcn = torch.cat(all_prob_stgcn, dim=0)
    labels     = torch.cat(all_y,          dim=0)

    # Individual
    acc_sml   = (prob_sml.argmax(1)   == labels).float().mean().item() * 100
    acc_stgcn = (prob_stgcn.argmax(1) == labels).float().mean().item() * 100
    print(f"\nSML    alone: {acc_sml:.2f}%")
    print(f"ST-GCN alone: {acc_stgcn:.2f}%")

    # Alpha sweep
    print(f"\n{'='*55}")
    print("Alpha Sweep (SML weight / ST-GCN weight):")
    print(f"{'='*55}")

    best_acc   = 0
    best_alpha = 0

    for alpha in np.arange(0.0, 1.05, 0.05):
        ens = alpha * prob_sml + (1 - alpha) * prob_stgcn
        acc = (ens.argmax(1) == labels).float().mean().item() * 100
        marker = " <- BEST" if acc > best_acc else ""
        print(f"  SML={alpha:.2f} / ST-GCN={1-alpha:.2f} : {acc:.2f}%{marker}")
        if acc > best_acc:
            best_acc   = acc
            best_alpha = alpha

    print(f"\n{'='*55}")
    print(f"BEST ENSEMBLE : {best_acc:.2f}% at SML={best_alpha:.2f} / ST-GCN={1-best_alpha:.2f}")
    print(f"Gain over SML   : +{best_acc - acc_sml:.2f}%")
    print(f"Gain over ST-GCN: +{best_acc - acc_stgcn:.2f}%")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
