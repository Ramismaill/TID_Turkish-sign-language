"""
plot_metrics.py — Parse training logs and plot Loss / Accuracy curves

Usage:
  python src/plot_metrics.py --log train_sml_log.txt --out sml_metrics.png
  python src/plot_metrics.py --log train_sml_log.txt --out sml_metrics.png --title "SML Training"

Accepts raw terminal output from train_sml.py / train_ctrgcn.py / train_stgcn.py.
Parses lines like:
  Epoch 001 | train_loss=4.8204 train_acc=0.0260 | val_loss=3.8949 val_acc=0.1304
"""

import argparse
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    """Parse training log file and return lists of metrics."""
    pattern = re.compile(
        r"Epoch\s+(\d+)\s*\|\s*"
        r"train_loss=([\d.]+)\s+train_acc=([\d.]+)\s*\|\s*"
        r"val_loss=([\d.]+)\s+val_acc=([\d.]+)"
    )

    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                train_acc.append(float(m.group(3)))
                val_loss.append(float(m.group(4)))
                val_acc.append(float(m.group(5)))

    if not epochs:
        print("ERROR: No training log lines found. Expected format:")
        print("  Epoch 001 | train_loss=X.XXXX train_acc=X.XXXX | val_loss=X.XXXX val_acc=X.XXXX")
        sys.exit(1)

    return {
        "epoch": np.array(epochs),
        "train_loss": np.array(train_loss),
        "train_acc": np.array(train_acc) * 100,
        "val_loss": np.array(val_loss),
        "val_acc": np.array(val_acc) * 100,
    }


def plot_metrics(metrics, out_path, title="Training Metrics"):
    """Create side-by-side Loss and Accuracy plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    epochs = metrics["epoch"]

    # ── Loss Plot ────────────────────────────────────────────────────────
    ax1.plot(epochs, metrics["train_loss"], "o-", color="#2196F3",
             markersize=3, linewidth=1.5, label="Train Loss")
    ax1.plot(epochs, metrics["val_loss"], "s-", color="#F44336",
             markersize=3, linewidth=1.5, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(epochs[0], epochs[-1])

    # ── Accuracy Plot ────────────────────────────────────────────────────
    ax2.plot(epochs, metrics["train_acc"], "o-", color="#2196F3",
             markersize=3, linewidth=1.5, label="Train Acc")
    ax2.plot(epochs, metrics["val_acc"], "s-", color="#F44336",
             markersize=3, linewidth=1.5, label="Val Acc")

    # Best val_acc marker
    best_idx = np.argmax(metrics["val_acc"])
    best_epoch = epochs[best_idx]
    best_val = metrics["val_acc"][best_idx]
    ax2.annotate(f"Best: {best_val:.1f}% (E{best_epoch})",
                 xy=(best_epoch, best_val),
                 xytext=(best_epoch + 2, best_val - 5),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 fontsize=9, color="green", fontweight="bold")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curve")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(epochs[0], epochs[-1])
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  Epochs: {len(epochs)}")
    print(f"  Best val_acc: {best_val:.2f}% at epoch {best_epoch}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument("--out", default="metrics.png", help="Output image path")
    parser.add_argument("--title", default="Training Metrics", help="Plot title")
    args = parser.parse_args()

    metrics = parse_log(args.log)
    plot_metrics(metrics, args.out, args.title)


if __name__ == "__main__":
    main()
