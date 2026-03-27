"""
ensemble_tmsnet_sml.py — Ensemble: TMS-Net + SML

Usage:
  python src\ensemble_tmsnet_sml.py ^
    --tmsnet_ckpt "C:/AUTSL_project/checkpoints/best_tmsnet.pth" ^
    --sml_ckpt    "C:/AUTSL_project/checkpoints/best_sml.pt" ^
    --val_csv     "C:/AUTSL_project/landmarks/val_manifest.csv"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import pandas as pd

from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS, ADJACENCY
from sml_model import SML
from tmsnet_model import TMSNet


# ── helpers ───────────────────────────────────────────────────────────────────

BONE_SRC = [p[0] for p in BONE_PAIRS]
BONE_DST = [p[1] for p in BONE_PAIRS]


def get_angle_pairs(bone_pairs):
    pairs = []
    n = len(bone_pairs)
    for i in range(n):
        for j in range(i + 1, n):
            if set(bone_pairs[i]) & set(bone_pairs[j]):
                pairs.append((i, j))
    return pairs


ANGLE_PAIRS = get_angle_pairs(BONE_PAIRS)


def compute_tmsnet_streams(joint_np):
    """joint_np: (T, V, 3) → dict of (3, T, V) tensors"""
    T, V, _ = joint_np.shape

    bone = np.zeros_like(joint_np)
    for parent, child in BONE_PAIRS:
        bone[:, child] = joint_np[:, child] - joint_np[:, parent]

    jmotion = np.zeros_like(joint_np)
    jmotion[:-1] = joint_np[1:] - joint_np[:-1]

    bmotion = np.zeros_like(bone)
    bmotion[:-1] = bone[1:] - bone[:-1]

    bone_vecs = np.zeros((T, len(BONE_PAIRS), 3), dtype=np.float32)
    for idx, (parent, child) in enumerate(BONE_PAIRS):
        bone_vecs[:, idx] = joint_np[:, child] - joint_np[:, parent]

    angle_node   = np.zeros((T, V, 3), dtype=np.float32)
    amotion_node = np.zeros((T, V, 3), dtype=np.float32)
    for k, (bi, bj) in enumerate(ANGLE_PAIRS):
        vi = bone_vecs[:, bi]
        vj = bone_vecs[:, bj]
        dot    = (vi * vj).sum(axis=-1)
        norm_i = np.linalg.norm(vi, axis=-1).clip(min=1e-9)
        norm_j = np.linalg.norm(vj, axis=-1).clip(min=1e-9)
        cosine = dot / (norm_i * norm_j)
        child_node = BONE_PAIRS[bi][1]
        angle_node[:, child_node, 0] += cosine

    counts = np.zeros(V, dtype=np.float32)
    for k, (bi, _) in enumerate(ANGLE_PAIRS):
        counts[BONE_PAIRS[bi][1]] += 1
    counts = counts.clip(min=1)
    angle_node /= counts[np.newaxis, :, np.newaxis]
    amotion_node[:-1] = angle_node[1:] - angle_node[:-1]

    def ctv(a): return torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32))

    return {
        'joint'   : ctv(joint_np),
        'bone'    : ctv(bone),
        'jmotion' : ctv(jmotion),
        'bmotion' : ctv(bmotion),
        'angle'   : ctv(angle_node),
        'amotion' : ctv(amotion_node),
    }


# ── Dataset ───────────────────────────────────────────────────────────────────

class EnsembleDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # Detect path and label columns
        path_col  = next((c for c in ['npy_path', 'path', 'filepath'] if c in df.columns), df.columns[0])
        label_col = next((c for c in ['label', 'sign', 'class_id']   if c in df.columns), df.columns[1])
        self.samples = [(row[path_col], int(row[label_col])) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        raw = np.load(path).astype(np.float32)             # (64, 225)
        raw = raw[:, KEEP_INDICES].reshape(64, NUM_NODES, 3)  # (64, 56, 3)

        # TMS-Net streams
        tms = compute_tmsnet_streams(raw)                  # dict of (3,64,56)

        # SML streams — must match SML training exactly
        joint_t  = tms['joint']                            # (3, 64, 56)
        # SML bone: indexed by bone position, not node position
        bone_t   = joint_t[:, :, BONE_SRC] - joint_t[:, :, BONE_DST]  # (3,64,56)
        # SML motion: temporal difference
        motion_t = torch.zeros_like(joint_t)
        motion_t[:, 1:, :] = joint_t[:, 1:, :] - joint_t[:, :-1, :]

        return tms, joint_t, bone_t, motion_t, label


def collate_fn(batch):
    tms_keys = batch[0][0].keys()
    tms = {k: torch.stack([b[0][k] for b in batch]) for k in tms_keys}
    joint  = torch.stack([b[1] for b in batch])
    bone   = torch.stack([b[2] for b in batch])
    motion = torch.stack([b[3] for b in batch])
    labels = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return tms, joint, bone, motion, labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmsnet_ckpt', required=True)
    parser.add_argument('--sml_ckpt',    required=True)
    parser.add_argument('--val_csv',     required=True)
    parser.add_argument('--batch_size',  type=int, default=32)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load TMS-Net
    tmsnet = TMSNet(num_classes=226, dropout=0.4).to(device)
    ckpt = torch.load(args.tmsnet_ckpt, map_location=device, weights_only=False)
    tmsnet.load_state_dict(ckpt['model'])
    tmsnet.eval()
    print(f"TMS-Net loaded — val_acc: {ckpt.get('val_acc', 0)*100:.2f}%")

    # ── Load SML
    sml = SML(num_classes=226, dropout=0.3).to(device)
    ckpt2 = torch.load(args.sml_ckpt, map_location=device, weights_only=False)
    sml.load_state_dict(ckpt2['model'])
    sml.eval()
    print(f"SML    loaded — val_acc: {ckpt2.get('val_acc', 0)*100:.2f}%")

    loader = DataLoader(
        EnsembleDataset(args.val_csv),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    all_prob_tms = []
    all_prob_sml = []
    all_y        = []

    print("\nComputing probabilities...")
    with torch.no_grad():
        for tms, joint, bone, motion, y in loader:
            tms    = {k: v.to(device) for k, v in tms.items()}
            joint  = joint.to(device)
            bone   = bone.to(device)
            motion = motion.to(device)

            with autocast('cuda'):
                logits_tms = tmsnet(**tms)
                logits_sml = sml(joint, bone, motion)

            all_prob_tms.append(F.softmax(logits_tms, dim=1).cpu())
            all_prob_sml.append(F.softmax(logits_sml, dim=1).cpu())
            all_y.append(y)

    prob_tms = torch.cat(all_prob_tms, dim=0)
    prob_sml = torch.cat(all_prob_sml, dim=0)
    labels   = torch.cat(all_y,        dim=0)

    acc_tms = (prob_tms.argmax(1) == labels).float().mean().item() * 100
    acc_sml = (prob_sml.argmax(1) == labels).float().mean().item() * 100
    print(f"\nTMS-Net alone: {acc_tms:.2f}%")
    print(f"SML     alone: {acc_sml:.2f}%")

    # ── Alpha sweep
    print(f"\n{'='*55}")
    print("Alpha Sweep  (TMS-Net weight  /  SML weight):")
    print(f"{'='*55}")
    best_acc, best_alpha = 0, 0
    for alpha in np.arange(0.0, 1.05, 0.05):
        ens = alpha * prob_tms + (1 - alpha) * prob_sml
        acc = (ens.argmax(1) == labels).float().mean().item() * 100
        marker = " <- BEST" if acc > best_acc else ""
        print(f"  TMS={alpha:.2f} / SML={1-alpha:.2f} : {acc:.2f}%{marker}")
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha

    print(f"\n{'='*55}")
    print(f"BEST ENSEMBLE : {best_acc:.2f}%  (TMS={best_alpha:.2f} / SML={1-best_alpha:.2f})")
    print(f"Gain over TMS-Net : +{best_acc - acc_tms:.2f}%")
    print(f"Gain over SML     : +{best_acc - acc_sml:.2f}%")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()