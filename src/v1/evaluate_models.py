"""
evaluate_models.py — Comprehensive single-model evaluation

Usage:
  python evaluate_models.py ^
    --model_type sml ^
    --checkpoint ../checkpoints/best_sml.pt ^
    --manifest_csv ../landmarks/test_manifest.csv ^
    --class_map class_map.json ^
    --out_dir ../results/

Outputs:
  - classification_report.txt   (human-readable)
  - classification_report.csv   (machine-readable)
  - confusion_top20.png         (heatmap of most confused class pairs)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

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


# ── Input transforms (same as ensemble_eval.py) ─────────────────────────────

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


# ── Model loading ────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "stgcn":  {"cls": STGCN,  "kwargs": {"num_classes": 226, "dropout": 0.5}},
    "ctrgcn": {"cls": CTRGCN, "kwargs": {"num_classes": 226, "dropout": 0.4}},
    "sml":    {"cls": SML,    "kwargs": {"num_classes": 226, "dropout": 0.3}},
}


def load_model(model_type, ckpt_path, device):
    cfg = MODEL_CONFIG[model_type]
    model = cfg["cls"](**cfg["kwargs"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    val_acc = ckpt.get("val_acc", 0)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded {model_type.upper()} from epoch {epoch}, val_acc={val_acc:.4f}")
    return model


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, model_type, dataloader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            with autocast("cuda", enabled=(device == "cuda")):
                if model_type == "stgcn":
                    logits = model(stgcn_input(x))
                elif model_type == "sml":
                    joint, bone, motion = sml_inputs(x)
                    logits = model(joint, bone, motion)
                else:  # ctrgcn
                    logits = model(x)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


# ── Classification report ────────────────────────────────────────────────────

def generate_report(y_true, y_pred, class_names, out_dir):
    # Text report
    report_txt = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    txt_path = os.path.join(out_dir, "classification_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)
    print(f"Saved: {txt_path}")

    # CSV report
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0
    )
    csv_path = os.path.join(out_dir, "classification_report.csv")
    pd.DataFrame(report_dict).transpose().to_csv(csv_path)
    print(f"Saved: {csv_path}")

    # Overall metrics
    acc = (y_true == y_pred).mean() * 100
    macro_f1 = report_dict["macro avg"]["f1-score"] * 100
    weighted_f1 = report_dict["weighted avg"]["f1-score"] * 100
    print(f"\nOverall Accuracy : {acc:.2f}%")
    print(f"Macro F1         : {macro_f1:.2f}%")
    print(f"Weighted F1      : {weighted_f1:.2f}%")

    # Top-5 worst classes by F1
    per_class = {
        name: report_dict[name]
        for name in class_names if name in report_dict
    }
    sorted_by_f1 = sorted(per_class.items(), key=lambda kv: kv[1]["f1-score"])
    print("\nTop-5 worst classes (lowest F1):")
    for name, metrics in sorted_by_f1[:5]:
        print(f"  {name}: F1={metrics['f1-score']:.3f}  "
              f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"support={int(metrics['support'])}")


# ── Top-K confused pairs heatmap ─────────────────────────────────────────────

def plot_confusion_heatmap(y_true, y_pred, class_names, out_dir, top_k=20):
    cm = confusion_matrix(y_true, y_pred)

    # Find off-diagonal confused pairs
    confused = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                confused.append((cm[i, j], i, j))
    confused.sort(reverse=True)

    if not confused:
        print("No misclassifications found — skipping heatmap.")
        return

    # Get unique class indices from top-K pairs
    top_pairs = confused[:top_k]
    involved = set()
    for _, i, j in top_pairs:
        involved.add(i)
        involved.add(j)
    involved = sorted(involved)

    # Extract sub-matrix
    sub_cm = cm[np.ix_(involved, involved)]
    sub_names = [class_names[i] if i < len(class_names) else str(i)
                 for i in involved]

    # Plot
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
    ax.set_title(f"Top-{top_k} Most Confused Class Pairs")

    # Annotate cells with counts
    for i in range(n):
        for j in range(n):
            val = sub_cm[i, j]
            if val > 0:
                color = "white" if val > sub_cm.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=6, color=color)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "confusion_top20.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Print top confused pairs
    print(f"\nTop-{top_k} confused pairs (true -> predicted : count):")
    for count, i, j in top_pairs:
        true_name = class_names[i] if i < len(class_names) else str(i)
        pred_name = class_names[j] if j < len(class_names) else str(j)
        print(f"  {true_name} -> {pred_name} : {count}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single model on test/val set"
    )
    parser.add_argument("--model_type", required=True,
                        choices=["stgcn", "ctrgcn", "sml"])
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_*.pt checkpoint")
    parser.add_argument("--manifest_csv", required=True,
                        help="Path to test or val manifest CSV")
    parser.add_argument("--class_map", default="class_map.json",
                        help="Path to class_map.json")
    parser.add_argument("--out_dir", default="../results/",
                        help="Directory for output files")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of confused pairs for heatmap")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load class names
    with open(args.class_map, "r") as f:
        cmap = json.load(f)
    class_names = [cmap[str(i)] for i in range(len(cmap))]
    print(f"Classes: {len(class_names)}")

    # Load model
    model = load_model(args.model_type, args.checkpoint, device)

    # DataLoader
    loader = DataLoader(
        ValDataset(args.manifest_csv),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"Samples: {len(loader.dataset)}")

    # Inference
    print("\nRunning inference...")
    y_true, y_pred = run_inference(model, args.model_type, loader, device)

    # Reports
    print(f"\n{'='*60}")
    generate_report(y_true, y_pred, class_names, args.out_dir)

    print(f"\n{'='*60}")
    plot_confusion_heatmap(y_true, y_pred, class_names, args.out_dir,
                           top_k=args.top_k)

    print(f"\n{'='*60}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
