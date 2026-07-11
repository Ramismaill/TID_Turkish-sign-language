"""
ensemble_search.py — Fine-grained ensemble weight grid search

Usage (2-model: SML + CTR-GCN):
  python ensemble_search.py ^
    --sml_ckpt    ../checkpoints/best_sml.pt ^
    --ctrgcn_ckpt ../checkpoints/best_ctrgcn.pt ^
    --val_csv     ../landmarks/val_manifest.csv ^
    --step 0.05

Usage (3-model: SML + CTR-GCN + ST-GCN):
  python ensemble_search.py ^
    --sml_ckpt    ../checkpoints/best_sml.pt ^
    --ctrgcn_ckpt ../checkpoints/best_ctrgcn.pt ^
    --stgcn_ckpt  ../checkpoints/best_stgcn.pt ^
    --val_csv     ../landmarks/val_manifest.csv ^
    --step 0.05 ^
    --save_probs  ../results/ensemble_probs.pt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import pandas as pd

from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS
from stgcn_model import STGCN
from ctrgcn_model import CTRGCN
from sml_model import SML


# ── Dataset (same as ensemble_eval.py) ───────────────────────────────────────

class ValDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["npy_path"]).astype(np.float32)
        x = x[:, KEEP_INDICES]
        x = x.reshape(64, NUM_NODES, 3).transpose(2, 0, 1)
        label = int(row["label"])
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.long)


# ── Input transforms ─────────────────────────────────────────────────────────

def stgcn_input(x):
    vel = torch.zeros_like(x)
    vel[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
    vel = vel * 10.0
    return torch.cat([x, vel], dim=1)


def sml_inputs(x):
    bone_src = [p[0] for p in BONE_PAIRS]
    bone_dst = [p[1] for p in BONE_PAIRS]
    bone = x[:, :, :, bone_src] - x[:, :, :, bone_dst]
    motion = torch.zeros_like(x)
    motion[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
    return x, bone, motion


# ── Precompute probabilities ─────────────────────────────────────────────────

def precompute_probs(models, dataloader, device):
    """Run inference once for each model and cache softmax probabilities."""
    probs = {name: [] for name in models}
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            all_labels.append(y)

            for name, model in models.items():
                with autocast("cuda", enabled=(device == "cuda")):
                    if name == "stgcn":
                        logits = model(stgcn_input(x))
                    elif name == "sml":
                        joint, bone, motion = sml_inputs(x)
                        logits = model(joint, bone, motion)
                    else:  # ctrgcn
                        logits = model(x)
                probs[name].append(F.softmax(logits, dim=1).cpu())

    result = {name: torch.cat(p, dim=0) for name, p in probs.items()}
    result["labels"] = torch.cat(all_labels, dim=0)
    return result


# ── Grid search ──────────────────────────────────────────────────────────────

def search_2model(prob_a, prob_b, labels, step, name_a, name_b):
    """Alpha sweep between two models. Returns sorted results."""
    results = []
    for alpha in np.arange(0.0, 1.0 + step / 2, step):
        alpha = round(alpha, 4)
        if alpha > 1.0:
            alpha = 1.0
        prob = alpha * prob_a + (1 - alpha) * prob_b
        acc = (prob.argmax(1) == labels).float().mean().item() * 100
        results.append((alpha, acc))

    results.sort(key=lambda x: -x[1])
    return results


def search_3model(prob_s, prob_c, prob_m, labels, step):
    """3-model grid search with constraint w_s + w_c + w_m = 1.0."""
    results = []
    for w_s in np.arange(0.0, 1.0 + step / 2, step):
        for w_c in np.arange(0.0, 1.0 + step / 2 - w_s, step):
            w_m = round(1.0 - w_s - w_c, 4)
            if w_m < -0.001:
                continue
            w_m = max(w_m, 0.0)
            prob = w_s * prob_s + w_c * prob_c + w_m * prob_m
            acc = (prob.argmax(1) == labels).float().mean().item() * 100
            results.append((round(w_s, 4), round(w_c, 4), round(w_m, 4), acc))

    results.sort(key=lambda x: -x[3])
    return results


# ── Display ──────────────────────────────────────────────────────────────────

def print_2model_table(results, name_a, name_b, top_k=5):
    print(f"\n{'='*50}")
    print(f"2-Model Search: {name_a} + {name_b}")
    print(f"{'='*50}")
    print(f"{'Rank':>4} | {name_a:>8} | {name_b:>8} | {'Accuracy':>10}")
    print(f"{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    for i, (alpha, acc) in enumerate(results[:top_k]):
        print(f"{i+1:>4} | {alpha:>8.4f} | {1-alpha:>8.4f} | {acc:>9.2f}%")

    best_alpha, best_acc = results[0]
    print(f"\nBest: {name_a}={best_alpha:.4f}, {name_b}={1-best_alpha:.4f} -> {best_acc:.2f}%")
    return best_acc


def print_3model_table(results, top_k=5):
    print(f"\n{'='*60}")
    print("3-Model Search: SML + CTR-GCN + ST-GCN")
    print(f"{'='*60}")
    print(f"{'Rank':>4} | {'w_SML':>7} | {'w_CTR':>7} | {'w_STG':>7} | {'Accuracy':>10}")
    print(f"{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")

    for i, (w_s, w_c, w_m, acc) in enumerate(results[:top_k]):
        print(f"{i+1:>4} | {w_s:>7.4f} | {w_c:>7.4f} | {w_m:>7.4f} | {acc:>9.2f}%")

    w_s, w_c, w_m, best_acc = results[0]
    print(f"\nBest: SML={w_s:.4f}, CTR-GCN={w_c:.4f}, ST-GCN={w_m:.4f} -> {best_acc:.2f}%")
    return best_acc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-grained ensemble weight grid search"
    )
    parser.add_argument("--sml_ckpt", required=True,
                        help="SML checkpoint")
    parser.add_argument("--ctrgcn_ckpt", required=True,
                        help="CTR-GCN checkpoint")
    parser.add_argument("--stgcn_ckpt", default=None,
                        help="ST-GCN checkpoint (optional, enables 3-model)")
    parser.add_argument("--val_csv", required=True,
                        help="Validation manifest CSV")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--step", type=float, default=0.05,
                        help="Weight grid step size (default: 0.05)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Show top-K results (default: 5)")
    parser.add_argument("--save_probs", default=None,
                        help="Save best ensemble probs to this path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Step size  : {args.step}")
    has_stgcn = args.stgcn_ckpt is not None

    # ── Load models ──────────────────────────────────────────────────────
    models = {}

    sml = SML(num_classes=226, dropout=0.3).to(device)
    ckpt = torch.load(args.sml_ckpt, map_location=device, weights_only=False)
    sml.load_state_dict(ckpt["model"])
    sml.eval()
    models["sml"] = sml
    print(f"SML     loaded -- val_acc: {ckpt.get('val_acc', 0):.4f}")

    ctrgcn = CTRGCN(num_classes=226, dropout=0.4).to(device)
    ckpt2 = torch.load(args.ctrgcn_ckpt, map_location=device, weights_only=False)
    ctrgcn.load_state_dict(ckpt2["model"])
    ctrgcn.eval()
    models["ctrgcn"] = ctrgcn
    print(f"CTR-GCN loaded -- val_acc: {ckpt2.get('val_acc', 0):.4f}")

    if has_stgcn:
        stgcn = STGCN(num_classes=226, dropout=0.5).to(device)
        ckpt3 = torch.load(args.stgcn_ckpt, map_location=device, weights_only=False)
        stgcn.load_state_dict(ckpt3["model"])
        stgcn.eval()
        models["stgcn"] = stgcn
        print(f"ST-GCN  loaded -- val_acc: {ckpt3.get('val_acc', 0):.4f}")

    # ── Precompute ───────────────────────────────────────────────────────
    print("\nPrecomputing probabilities...")
    loader = DataLoader(
        ValDataset(args.val_csv),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    cached = precompute_probs(models, loader, device)
    labels = cached["labels"]

    # Individual accuracies
    print(f"\n{'='*40}")
    print("Individual Model Accuracies:")
    print(f"{'='*40}")
    acc_sml = (cached["sml"].argmax(1) == labels).float().mean().item() * 100
    acc_ctr = (cached["ctrgcn"].argmax(1) == labels).float().mean().item() * 100
    print(f"  SML     : {acc_sml:.2f}%")
    print(f"  CTR-GCN : {acc_ctr:.2f}%")
    if has_stgcn:
        acc_stg = (cached["stgcn"].argmax(1) == labels).float().mean().item() * 100
        print(f"  ST-GCN  : {acc_stg:.2f}%")

    # ── 2-model: SML + CTR-GCN ──────────────────────────────────────────
    results_sc = search_2model(
        cached["sml"], cached["ctrgcn"], labels,
        args.step, "SML", "CTR-GCN"
    )
    best_2_sc = print_2model_table(results_sc, "SML", "CTR-GCN", args.top_k)

    best_overall_acc = best_2_sc
    best_overall_probs = results_sc[0][0] * cached["sml"] + \
                         (1 - results_sc[0][0]) * cached["ctrgcn"]
    best_overall_desc = f"SML+CTR-GCN"

    if has_stgcn:
        # ── 2-model: SML + ST-GCN ───────────────────────────────────────
        results_ss = search_2model(
            cached["sml"], cached["stgcn"], labels,
            args.step, "SML", "ST-GCN"
        )
        best_2_ss = print_2model_table(results_ss, "SML", "ST-GCN", args.top_k)

        # ── 2-model: CTR-GCN + ST-GCN ───────────────────────────────────
        results_cs = search_2model(
            cached["ctrgcn"], cached["stgcn"], labels,
            args.step, "CTR-GCN", "ST-GCN"
        )
        best_2_cs = print_2model_table(results_cs, "CTR-GCN", "ST-GCN", args.top_k)

        # ── 3-model: SML + CTR-GCN + ST-GCN ─────────────────────────────
        results_3 = search_3model(
            cached["sml"], cached["ctrgcn"], cached["stgcn"],
            labels, args.step
        )
        best_3 = print_3model_table(results_3, args.top_k)

        if best_3 > best_overall_acc:
            w_s, w_c, w_m, _ = results_3[0]
            best_overall_acc = best_3
            best_overall_probs = w_s * cached["sml"] + \
                                 w_c * cached["ctrgcn"] + \
                                 w_m * cached["stgcn"]
            best_overall_desc = f"SML+CTR-GCN+ST-GCN"

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"BEST ENSEMBLE: {best_overall_desc} -> {best_overall_acc:.2f}%")
    print(f"  Gain over SML     : +{best_overall_acc - acc_sml:.2f}%")
    print(f"  Gain over CTR-GCN : +{best_overall_acc - acc_ctr:.2f}%")
    if has_stgcn:
        print(f"  Gain over ST-GCN  : +{best_overall_acc - acc_stg:.2f}%")
    print(f"{'='*60}")

    # ── Save probs ───────────────────────────────────────────────────────
    if args.save_probs:
        os.makedirs(os.path.dirname(args.save_probs) or ".", exist_ok=True)
        torch.save({
            "probs": best_overall_probs,
            "labels": labels,
            "accuracy": best_overall_acc,
            "description": best_overall_desc,
        }, args.save_probs)
        print(f"Saved: {args.save_probs}")


if __name__ == "__main__":
    main()
