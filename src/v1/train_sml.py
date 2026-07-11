"""
train_sml.py — SML (Skeleton-based Multi-feature Learning) Training for AUTSL

Streams: Joint + Bone + Motion (pre-computed in dataset)
Input to model: 3 × (B, 3, 64, 56)
Key differences vs CTR-GCN:
  - Streams pre-computed in dataset (augmented joints -> derived bone/motion)
  - Cross-stream attention fusion (streams inform each other)
  - 3 lighter encoders (8 blocks) instead of 4 heavier (10 blocks)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import pandas as pd

from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS
from sml_model import SML
from augmentations import LandmarkAugment


# ── Dataset ───────────────────────────────────────────────────────────────────

class AUTSLSMLDataset(Dataset):
    """
    Returns 3 pre-computed streams: joint, bone, motion
    Each of shape (3, 64, 56).

    Bone and motion are derived from augmented joints, ensuring
    augmentation consistency across all streams.
    """

    def __init__(self, manifest_csv: str, augment=None):
        self.df = pd.read_csv(manifest_csv)
        self.augment = augment
        # Pre-build bone index arrays from shared BONE_PAIRS
        self.bone_src = np.array([p[0] for p in BONE_PAIRS])
        self.bone_dst = np.array([p[1] for p in BONE_PAIRS])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["npy_path"]).astype(np.float32)  # (64, 225)

        # Augment raw landmarks first (before deriving bone/motion)
        if self.augment is not None:
            x = self.augment(x)

        # Slice and reshape
        x = x[:, KEEP_INDICES]                           # (64, 168)
        x = x.reshape(64, NUM_NODES, 3)                  # (64, 56, 3)
        x = x.transpose(2, 0, 1)                         # (3, 64, 56)

        joint = torch.tensor(x, dtype=torch.float32)

        # Bone: child - parent (same formula as CTR-GCN._compute_bone)
        bone = joint[:, :, self.bone_src] - joint[:, :, self.bone_dst]

        # Motion: temporal difference (same formula as CTR-GCN._compute_motion)
        motion = torch.zeros_like(joint)
        motion[:, 1:, :] = joint[:, 1:, :] - joint[:, :-1, :]

        label = int(row["label"])
        return (joint, bone, motion), torch.tensor(label, dtype=torch.long)


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
        for (joint, bone, motion), y in loader:
            joint  = joint.to(device)
            bone   = bone.to(device)
            motion = motion.to(device)
            y = y.to(device)

            logits = model(joint, bone, motion)
            loss   = criterion(logits, y)
            b = joint.size(0)
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
    parser.add_argument("--batch_size",   type=int,   default=48)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--patience",     type=int,   default=20)
    parser.add_argument("--mixup_alpha",  type=float, default=0.2)
    parser.add_argument("--out_dir",      type=str,   default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device   : {device}")
    print(f"Nodes          : {NUM_NODES}")
    print(f"Streams        : joint + bone + motion (pre-computed)")
    print(f"Fusion         : cross-stream attention")
    print(f"Batch size     : {args.batch_size}")
    print(f"Dropout        : {args.dropout}")
    print(f"Weight decay   : {args.weight_decay}")

    # Datasets
    train_ds = AUTSLSMLDataset(args.manifest_csv,
                                augment=LandmarkAugment(target_len=64))
    val_ds   = AUTSLSMLDataset(args.val_csv, augment=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=args.num_workers,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True)

    # Model
    model = SML(num_classes=226, dropout=args.dropout).to(device)
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
    best_path   = os.path.join(args.out_dir, "best_sml.pt")
    resume_path = os.path.join(args.out_dir, "resume_sml.pt")

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

        for step, ((joint, bone, motion), y) in enumerate(train_loader):
            joint  = joint.to(device)
            bone   = bone.to(device)
            motion = motion.to(device)
            y = y.to(device)

            # Mixup — apply same lambda and index to all 3 streams
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            idx = torch.randperm(joint.size(0)).to(device)
            joint  = lam * joint  + (1 - lam) * joint[idx]
            bone   = lam * bone   + (1 - lam) * bone[idx]
            motion = lam * motion + (1 - lam) * motion[idx]

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = model(joint, bone, motion)
                loss   = lam * criterion(logits, y) + \
                         (1 - lam) * criterion(logits, y[idx])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch - 1 + step / max(1, len(train_loader)))

            b = joint.size(0)
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
