"""
eval_tmsnet.py — TMS-Net evaluation on AUTSL val/test set

Place this file in:  C:\AUTSL_project\src\
Run from:           C:\AUTSL_project\

Usage (from C:\AUTSL_project\):
    python src\eval_tmsnet.py ^
        --checkpoint checkpoints\best.pth ^
        --manifest landmarks\val_manifest.csv ^
        --class_map src\class_map.json ^
        --out_dir results

Outputs (in --out_dir):
    - tmsnet_classification_report.txt
    - tmsnet_classification_report.csv
    - tmsnet_confusion_top20.png
    - tmsnet_per_class_accuracy.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

# Add this script's directory to the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS
from tmsnet_model import TMSNet


# ──────────────────────────────────────────────
# Pre-compute angle pairs (same as inference_tmsnet.py)
# ──────────────────────────────────────────────
def _build_angle_pairs(bone_pairs):
    pairs = []
    n = len(bone_pairs)
    for i in range(n):
        for j in range(i + 1, n):
            if set(bone_pairs[i]) & set(bone_pairs[j]):
                pairs.append((i, j))
    return pairs

ANGLE_PAIRS = _build_angle_pairs(BONE_PAIRS)


# ──────────────────────────────────────────────
# Dataset: loads .npy → builds 6 streams
# ──────────────────────────────────────────────
class TMSNetEvalDataset(Dataset):
    """
    Each .npy file: shape (64, 225) — 64 frames × 75 landmarks × 3 coords flat.
    We slice with KEEP_INDICES → (64, 168), reshape → (64, 56, 3),
    then build the 6 streams expected by TMS-Net.
    """
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        assert {"npy_path", "label"}.issubset(self.df.columns), \
            "CSV must contain 'npy_path' and 'label' columns."

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["npy_path"]).astype(np.float32)         # (64, 225)
        x = x[:, KEEP_INDICES].reshape(64, NUM_NODES, 3)        # (T, V, 3)
        streams = self._build_streams(x)
        label = int(row["label"])
        return streams, label

    @staticmethod
    def _build_streams(x):
        """x: (T, V, 3) → dict of 6 tensors, each (3, T, V)"""
        T, V, _ = x.shape

        # Bone: child - parent
        bone = np.zeros_like(x)
        for parent, child in BONE_PAIRS:
            bone[:, child] = x[:, child] - x[:, parent]

        # Joint motion
        jmot = np.zeros_like(x)
        jmot[:-1] = x[1:] - x[:-1]

        # Bone motion
        bmot = np.zeros_like(bone)
        bmot[:-1] = bone[1:] - bone[:-1]

        # Angles between bones sharing a joint
        bv = np.zeros((T, len(BONE_PAIRS), 3), dtype=np.float32)
        for idx, (parent, child) in enumerate(BONE_PAIRS):
            bv[:, idx] = x[:, child] - x[:, parent]

        an = np.zeros((T, V, 3), dtype=np.float32)
        for bi, bj in ANGLE_PAIRS:
            vi = bv[:, bi]; vj = bv[:, bj]
            dot = (vi * vj).sum(axis=-1)
            cosine = dot / (np.linalg.norm(vi, axis=-1).clip(1e-9) *
                            np.linalg.norm(vj, axis=-1).clip(1e-9))
            an[:, BONE_PAIRS[bi][1], 0] += cosine

        counts = np.zeros(V, dtype=np.float32)
        for bi, _ in ANGLE_PAIRS:
            counts[BONE_PAIRS[bi][1]] += 1
        an /= counts.clip(min=1)[np.newaxis, :, np.newaxis]

        amot = np.zeros_like(an)
        amot[:-1] = an[1:] - an[:-1]

        def to_ctv(a):
            # (T, V, 3) → (3, T, V)
            return torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32))

        return {
            "joint":   to_ctv(x),
            "bone":    to_ctv(bone),
            "jmotion": to_ctv(jmot),
            "bmotion": to_ctv(bmot),
            "angle":   to_ctv(an),
            "amotion": to_ctv(amot),
        }


def collate_streams(batch):
    """Stack list of (dict, label) → (dict of batched tensors, tensor of labels)"""
    streams_list, labels = zip(*batch)
    keys = streams_list[0].keys()
    out = {k: torch.stack([s[k] for s in streams_list], dim=0) for k in keys}
    return out, torch.tensor(labels, dtype=torch.long)


# ──────────────────────────────────────────────
# Inference loop
# ──────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    n_done = 0
    total = len(loader.dataset)

    for streams, y in loader:
        streams = {k: v.to(device, non_blocking=True) for k, v in streams.items()}
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(streams["joint"],   streams["bone"],
                           streams["jmotion"], streams["bmotion"],
                           streams["angle"],   streams["amotion"])
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())

        n_done += len(y)
        print(f"  {n_done}/{total} processed...", end="\r")

    print()
    return np.concatenate(all_labels), np.concatenate(all_preds)


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────
def generate_report(y_true, y_pred, class_names, out_dir):
    txt_report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    txt_path = os.path.join(out_dir, "tmsnet_classification_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_report)
    print(f"Saved: {txt_path}")

    rep_dict = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0
    )
    csv_path = os.path.join(out_dir, "tmsnet_classification_report.csv")
    pd.DataFrame(rep_dict).transpose().to_csv(csv_path)
    print(f"Saved: {csv_path}")

    acc = (y_true == y_pred).mean() * 100
    macro_f1 = rep_dict["macro avg"]["f1-score"] * 100
    weighted_f1 = rep_dict["weighted avg"]["f1-score"] * 100

    print(f"\n{'='*60}")
    print(f"  Overall Accuracy : {acc:.2f}%")
    print(f"  Macro F1         : {macro_f1:.2f}%")
    print(f"  Weighted F1      : {weighted_f1:.2f}%")
    print(f"{'='*60}\n")

    # Per-class accuracy
    cm_full = confusion_matrix(y_true, y_pred)
    per_class_acc = cm_full.diagonal() / cm_full.sum(axis=1).clip(min=1)
    pca_df = pd.DataFrame({
        "class_id": range(len(class_names)),
        "class_name": class_names,
        "accuracy": per_class_acc,
        "support": cm_full.sum(axis=1),
    })
    pca_path = os.path.join(out_dir, "tmsnet_per_class_accuracy.csv")
    pca_df.to_csv(pca_path, index=False)
    print(f"Saved: {pca_path}")

    # Top-5 worst by F1
    per_class = {n: rep_dict[n] for n in class_names if n in rep_dict}
    worst = sorted(per_class.items(), key=lambda kv: kv[1]["f1-score"])[:5]
    print("\nTop-5 worst classes (lowest F1):")
    for name, m in worst:
        print(f"  {name:30s}  F1={m['f1-score']:.3f}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"support={int(m['support'])}")

    return acc, macro_f1, weighted_f1


def plot_confusion_heatmap(y_true, y_pred, class_names, out_dir, top_k=20):
    cm = confusion_matrix(y_true, y_pred)

    confused = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                confused.append((cm[i, j], i, j))
    confused.sort(reverse=True)

    if not confused:
        print("No misclassifications found — skipping heatmap.")
        return

    top_pairs = confused[:top_k]
    involved = sorted({i for _, i, _ in top_pairs} | {j for _, _, j in top_pairs})
    sub_cm = cm[np.ix_(involved, involved)]
    sub_names = [class_names[i] if i < len(class_names) else str(i) for i in involved]

    n = len(involved)
    fig_size = max(8, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(sub_cm, cmap="YlOrRd", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(sub_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"TMS-Net — Top-{top_k} Most Confused Class Pairs")

    for i in range(n):
        for j in range(n):
            v = sub_cm[i, j]
            if v > 0:
                color = "white" if v > sub_cm.max() * 0.6 else "black"
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6, color=color)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "tmsnet_confusion_top20.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    print(f"\nTop-{top_k} confused pairs (true → predicted : count):")
    for count, i, j in top_pairs:
        tn = class_names[i] if i < len(class_names) else str(i)
        pn = class_names[j] if j < len(class_names) else str(j)
        print(f"  {tn:25s} → {pn:25s} : {count}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate TMS-Net on AUTSL val/test set")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--manifest",   required=True, help="Path to val/test manifest CSV")
    parser.add_argument("--class_map",  default="src/class_map.json")
    parser.add_argument("--out_dir",    default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--top_k",      type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load class map
    with open(args.class_map, "r", encoding="utf-8") as f:
        cmap = json.load(f)
    class_names = [cmap[str(i)] for i in range(len(cmap))]
    print(f"Classes: {len(class_names)}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = TMSNet(num_classes=len(class_names)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"  → epoch {ckpt.get('epoch', '?')}, "
          f"val_acc={ckpt.get('val_acc', 0)*100:.2f}%")

    # Dataset
    dataset = TMSNetEvalDataset(args.manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_streams,
        pin_memory=(device == "cuda"),
    )
    print(f"Samples: {len(dataset)}")

    # Inference
    print("\nRunning inference...")
    y_true, y_pred = run_inference(model, loader, device)

    # Reports + plots
    generate_report(y_true, y_pred, class_names, args.out_dir)
    plot_confusion_heatmap(y_true, y_pred, class_names, args.out_dir, top_k=args.top_k)

    print("\nDone.")


if __name__ == "__main__":
    main()
