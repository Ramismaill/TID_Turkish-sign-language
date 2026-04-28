"""
Training script for TMS-Net on AUTSL dataset.
6 streams computed on-the-fly in the dataset loader:
  1. joint        — raw (x,y,z) coords, shape (3, T, V)
  2. bone         — bone vectors
  3. joint_motion — temporal diff of joint
  4. bone_motion  — temporal diff of bone
  5. angle        — cosine angle between adjacent bones
  6. angle_motion — temporal diff of angle

Usage:
  python src/train_tmsnet.py \
    --manifest_csv "C:/AUTSL_project/landmarks/train_manifest.csv" \
    --val_csv      "C:/AUTSL_project/landmarks/val_manifest.csv"   \
    --epochs 120 --batch_size 32 --num_workers 4                   \
    --out_dir "C:/AUTSL_project/checkpoints"
"""

import os, sys, argparse, math, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmsnet_model import TMSNet
from augmentations import LandmarkAugment
from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS


# ─────────────────────────────────────────────
# 6-Stream Feature Computation
# ─────────────────────────────────────────────

def compute_bone(joint):
    """
    joint: (T, V, 3)  →  bone: (T, V, 3)
    bone[v] = joint[v] - joint[parent(v)]
    root node (0) gets zero bone vector.
    """
    bone = np.zeros_like(joint)
    for child, parent in BONE_PAIRS:
        if child < NUM_NODES and parent < NUM_NODES:
            bone[:, child, :] = joint[:, child, :] - joint[:, parent, :]
    return bone


def compute_angle(bone):
    """
    bone: (T, V, 3)  →  angle: (T, V, 3)
    For each node, compute cosine angle between its bone and its parent's bone.
    Stored as (cos_theta, sin_theta, 0) to give a 3D representation.
    Scale-invariant — captures wrist/finger orientation regardless of distance.
    """
    angle = np.zeros_like(bone)
    for child, parent in BONE_PAIRS:
        if child < NUM_NODES and parent < NUM_NODES:
            b1 = bone[:, child, :]    # (T, 3)
            b2 = bone[:, parent, :]   # (T, 3)
            n1 = np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8
            n2 = np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8
            cos_t = np.sum(b1 / n1 * b2 / n2, axis=-1, keepdims=True).clip(-1, 1)
            sin_t = np.sqrt(1 - cos_t ** 2 + 1e-8)
            angle[:, child, 0:1] = cos_t
            angle[:, child, 1:2] = sin_t
    return angle


def temporal_diff(x, scale=1.0):
    """
    x: (T, V, 3)  →  diff: (T, V, 3)
    diff[t] = x[t] - x[t-1];  diff[0] = 0
    """
    d = np.zeros_like(x)
    d[1:] = (x[1:] - x[:-1]) * scale
    return d


def build_streams(npy_path, augment=None):
    """
    Load .npy file and compute all 6 streams.
    Returns dict of tensors, each (3, T, V).
    """
    raw = np.load(npy_path).astype(np.float32)        # (T, 225)

    # Apply landmark augmentation (on raw 225 features)
    if augment is not None:
        raw = augment(raw)

    # Slice to 56 nodes → (T, V, 3)
    raw_pruned = raw[:, KEEP_INDICES]                  # (T, 168)
    joint = raw_pruned.reshape(raw_pruned.shape[0], NUM_NODES, 3)  # (T, V, 3)

    # Compute derived streams
    bone    = compute_bone(joint)
    jmotion = temporal_diff(joint, scale=1.0)
    bmotion = temporal_diff(bone,  scale=1.0)
    angle   = compute_angle(bone)
    amotion = temporal_diff(angle, scale=1.0)

    # Convert to (3, T, V) tensors
    def to_tensor(arr):
        return torch.from_numpy(arr.transpose(2, 0, 1))   # (3, T, V)

    return {
        'joint':   to_tensor(joint),
        'bone':    to_tensor(bone),
        'jmotion': to_tensor(jmotion),
        'bmotion': to_tensor(bmotion),
        'angle':   to_tensor(angle),
        'amotion': to_tensor(amotion),
    }


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class AUTSLTMSDataset(Dataset):
    def __init__(self, manifest_csv, augment=False):
        self.df = pd.read_csv(manifest_csv)
        self.aug = LandmarkAugment() if augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = row['npy_path']
        label = int(row['label'])
        streams = build_streams(path, self.aug)
        return streams, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    streams_list, labels = zip(*batch)
    keys = streams_list[0].keys()
    collated = {k: torch.stack([s[k] for s in streams_list]) for k in keys}
    return collated, torch.stack(labels)


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.best = -1
        self.counter = 0

    def __call__(self, val_acc):
        if val_acc > self.best:
            self.best = val_acc
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for streams, y in loader:
            streams = {k: v.to(device) for k, v in streams.items()}
            y = y.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(
                    streams['joint'],  streams['bone'],
                    streams['jmotion'], streams['bmotion'],
                    streams['angle'],  streams['amotion']
                )
                loss = criterion(logits, y)
            b = y.size(0)
            total_loss += loss.item() * b
            correct    += (logits.argmax(1) == y).sum().item()
            total      += b
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_csv', required=True)
    parser.add_argument('--val_csv',      required=True)
    parser.add_argument('--epochs',       type=int,   default=120)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout',      type=float, default=0.4)
    parser.add_argument('--mixup_alpha',  type=float, default=0.2)
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--patience',     type=int,   default=20)
    parser.add_argument('--out_dir',      default='C:/AUTSL_project/checkpoints')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Data ──
    train_ds = AUTSLTMSDataset(args.manifest_csv, augment=True)
    val_ds   = AUTSLTMSDataset(args.val_csv,      augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True)

    # ── Model ──
    model     = TMSNet(num_classes=226, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6
    )
    scaler = GradScaler()
    early  = EarlyStopping(patience=args.patience)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_path   = os.path.join(args.out_dir, 'best_tmsnet.pt')
    resume_path = os.path.join(args.out_dir, 'resume_tmsnet.pt')

    print(f"Using device   : {device}")
    print(f"Nodes          : {NUM_NODES}")
    print(f"Streams        : joint + bone + jmotion + bmotion + angle + amotion")
    print(f"Batch size     : {args.batch_size}")
    print(f"Dropout        : {args.dropout}")
    print(f"Weight decay   : {args.weight_decay}")
    print(f"Trainable parameters: {num_params:,}")

    # ── Resume ──
    start_epoch = 1
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        early.best = ckpt['best']
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (best={early.best:.4f})")

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for streams, y in train_loader:
            streams = {k: v.to(device) for k, v in streams.items()}
            y = y.to(device)

            # Mixup
            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                idx = torch.randperm(y.size(0), device=device)
                streams = {k: lam * v + (1 - lam) * v[idx] for k, v in streams.items()}

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(
                    streams['joint'],  streams['bone'],
                    streams['jmotion'], streams['bmotion'],
                    streams['angle'],  streams['amotion']
                )
                if args.mixup_alpha > 0:
                    loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[idx])
                else:
                    loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            b = y.size(0)
            total_loss += loss.item() * b
            correct    += (logits.argmax(1) == y).sum().item()
            total      += b

        scheduler.step()

        train_loss = total_loss / total
        train_acc  = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Save best
        if val_acc > early.best:
            torch.save({
                'model':   model.state_dict(),
                'val_acc': val_acc,
                'epoch':   epoch,
                'optimizer': optimizer.state_dict()
            }, best_path)
            print(f"  -> Best saved ({val_acc:.4f}) at epoch {epoch}")

        # Save resume
        torch.save({
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch':     epoch,
            'best':      early.best
        }, resume_path)

        if early(val_acc):
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nTraining complete. Best val_acc: {early.best:.4f}")
    print(f"Model saved: {best_path}")


if __name__ == '__main__':
    main()