"""
train_ctrgcn.py — Multi-Stream CTR-GCN Training for AUTSL (Target: 96-97%)

Streams: Joint + Bone + Velocity (computed on-the-fly)
Input to model: (B, 3, 64, 56) — joint only, bone/vel computed inside model
Key upgrades vs ST-GCN:
  - Dynamic channel-wise topology refinement
  - Multi-scale temporal convolution
  - 3-stream fusion (joint + bone + velocity)
  - Larger batch size (64) — RTX 5060 Ti 16GB
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import pandas as pd

from graph import KEEP_INDICES, NUM_NODES
from ctrgcn_model import CTRGCN
from augmentations import LandmarkAugment


# ── Dataset ───────────────────────────────────────────────────────────────────

class AUTSLGraphDataset(Dataset):
    """
    Returns joint coordinates only: (3, 64, 56)
    Bone and velocity computed inside CTRGCN.forward()
    """

    def __init__(self, manifest_csv: str, augment=None):
        self.df = pd.read_csv(manifest_csv)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["npy_path"]).astype(np.float32)  # (64, 225)

        if self.augment is not None:
            x = self.augment(x)

        x = x[:, KEEP_INDICES]                           # (64, 168)
        x = x.reshape(64, NUM_NODES, 3)                  # (64, 56, 3)
        x = x.transpose(2, 0, 1)                         # (3, 64, 56)

        label = int(row["label"])
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.long)


# ── Helpers ───────────────────────────────────────────────────────────────────

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best        = -1.0
        self.count       = 0
        self.should_stop = False

    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best  = metric
            self.count = 0
            return True
        self.count += 1
        if self.count >= self.patience:
            self.should_stop = True
        return False


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            b = x.size(0)
            total_loss += loss.item() * b
            total_acc  += accuracy(logits, y) * b
            n += b
    return total_loss / n, total_acc / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--val_csv",      required=True)
    parser.add_argument("--epochs",       type=int,   default=120)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--dropout",      type=float, default=0.4)
    parser.add_argument("--patience",     type=int,   default=20)
    parser.add_argument("--mixup_alpha",  type=float, default=0.2)
    parser.add_argument("--out_dir",      type=str,   default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device   : {device}")
    print(f"Nodes          : {NUM_NODES}")
    print(f"Streams        : joint + bone + velocity")
    print(f"Batch size     : {args.batch_size}")
    print(f"Dropout        : {args.dropout}")
    print(f"Weight decay   : {args.weight_decay}")

    # Datasets
    train_ds = AUTSLGraphDataset(args.manifest_csv,
                                  augment=LandmarkAugment(target_len=64))
    val_ds   = AUTSLGraphDataset(args.val_csv, augment=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=args.num_workers,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True)

    # Model
    model = CTRGCN(num_classes=226, dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1)
    scaler      = GradScaler("cuda")
    early       = EarlyStopping(patience=args.patience)
    best_path   = os.path.join(args.out_dir, "best_ctrgcn.pt")
    resume_path = os.path.join(args.out_dir, "resume_ctrgcn.pt")

    # Resume support
    start_epoch = 1
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        early.best  = ckpt["best"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (best={early.best:.4f})")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Mixup
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            idx = torch.randperm(x.size(0)).to(device)
            x   = lam * x + (1 - lam) * x[idx]

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = model(x)
                loss   = lam * criterion(logits, y) + \
                         (1 - lam) * criterion(logits, y[idx])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch - 1 + step / max(1, len(train_loader)))

            b = x.size(0)
            running_loss += loss.item() * b
            running_acc  += accuracy(logits.detach(), y) * b
            n += b

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} | "
              f"train_loss={running_loss/n:.4f} train_acc={running_acc/n:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if early.step(val_acc):
            torch.save({"model":     model.state_dict(),
                        "val_acc":   val_acc,
                        "epoch":     epoch,
                        "optimizer": optimizer.state_dict()}, best_path)
            print(f"  -> Best saved ({val_acc:.4f}) at epoch {epoch}")

        torch.save({"model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch":     epoch,
                    "best":      early.best}, resume_path)

        if early.should_stop:
            print("Early stopping triggered.")
            break

    print(f"\nTraining complete. Best val_acc = {early.best:.4f}")


if __name__ == "__main__":
    main()
