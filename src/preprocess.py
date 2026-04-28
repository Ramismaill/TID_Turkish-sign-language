"""
preprocess.py
Converts AUTSL parquet landmark files → .npy arrays (64, 225) for fast training.
Run this ONCE before training.

Usage:
  python preprocess.py ^
    --data_root "D:/AUTSL_DATASET_KAGGLE/archive/AUSTL_processed_landmark" ^
    --out_root  "C:/AUTSL_project/landmarks" ^
    --target_len 64
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

POSE_LM = 33
HAND_LM = 21
FEAT_DIM = (POSE_LM + HAND_LM + HAND_LM) * 3   # 225
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12


# ── helpers ────────────────────────────────────────────────────────────────────

def extract_type(frame_df, lm_type, expected_n):
    sub = frame_df[frame_df["type"] == lm_type].sort_values("landmark_index")
    out = np.zeros((expected_n, 3), dtype=np.float32)
    vals = sub[["x", "y", "z"]].values.astype(np.float32)
    n = min(expected_n, len(vals))
    out[:n] = vals[:n]
    return out


def parquet_to_array(parquet_path, target_len=64):
    """Read one parquet file → (target_len, 225) float32 array."""
    df = pd.read_parquet(parquet_path)

    # keep only the three types we need (skip face: too many landmarks, not needed)
    df = df[df["type"].isin(["pose", "left_hand", "right_hand"])].copy()

    # fill NaN landmarks with 0.0 (undetected landmarks)
    df[["x", "y", "z"]] = df[["x", "y", "z"]].fillna(0.0)

    frames = sorted(df["frame"].unique())
    seq = []

    for fid in frames:
        fdf = df[df["frame"] == fid]
        pose = extract_type(fdf, "pose",       POSE_LM)   # (33,3)
        lh   = extract_type(fdf, "left_hand",  HAND_LM)   # (21,3)
        rh   = extract_type(fdf, "right_hand", HAND_LM)   # (21,3)
        vec  = np.concatenate([pose.reshape(-1),
                               lh.reshape(-1),
                               rh.reshape(-1)])             # (225,)
        seq.append(vec)

    if len(seq) == 0:
        return np.zeros((target_len, FEAT_DIM), dtype=np.float32)

    seq = np.stack(seq).astype(np.float32)   # (T, 225)
    seq = normalize_shoulders(seq)
    seq = resample(seq, target_len)
    return seq


def normalize_shoulders(seq):
    """Subtract shoulder midpoint from every frame."""
    T = seq.shape[0]
    xyz = seq.reshape(T, -1, 3)                        # (T, 75, 3)
    ls  = xyz[:, LEFT_SHOULDER,  :]                    # (T, 3)
    rs  = xyz[:, RIGHT_SHOULDER, :]
    center = (ls + rs) / 2.0                           # (T, 3)

    for t in range(T):
        if not (np.allclose(ls[t], 0.0) and np.allclose(rs[t], 0.0)):
            xyz[t] -= center[t]                        # broadcasts (75,3) - (3,)

    return xyz.reshape(T, -1).astype(np.float32)


def resample(seq, target_len):
    """Linear interpolation to a fixed number of frames."""
    t, d = seq.shape
    if t == target_len:
        return seq
    if t == 1:
        return np.repeat(seq, target_len, axis=0).astype(np.float32)
    old_idx = np.linspace(0, 1, t)
    new_idx = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, d), dtype=np.float32)
    for i in range(d):
        out[:, i] = np.interp(new_idx, old_idx, seq[:, i])
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def process_split(split, data_root, out_root, target_len):
    csv_path = data_root / f"{split}.csv"
    df = pd.read_csv(csv_path)

    out_split = out_root / split
    out_split.mkdir(parents=True, exist_ok=True)

    manifest = []
    errors   = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}"):
        rel_path   = row["path"]          # e.g. "train/0/1.parquet"
        label      = int(row["sign"])
        parquet_fp = data_root / rel_path

        if not parquet_fp.exists():
            print(f"[WARN] missing: {parquet_fp}")
            errors += 1
            continue

        try:
            arr = parquet_to_array(parquet_fp, target_len=target_len)
        except Exception as e:
            print(f"[ERROR] {parquet_fp}: {e}")
            errors += 1
            continue

        # mirror the sub-folder structure  (participant_id/sequence_id.npy)
        npy_rel  = Path(rel_path).with_suffix(".npy")          # train/0/1.npy
        npy_path = out_split / npy_rel.relative_to(split)      # C:/…/train/0/1.npy
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, arr)

        manifest.append({"npy_path": str(npy_path), "label": label})

    manifest_df = pd.DataFrame(manifest)
    manifest_csv = out_root / f"{split}_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\n[{split}] Done. {len(manifest_df)} samples | {errors} errors")
    print(f"  Manifest: {manifest_csv}")
    return manifest_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, required=True,
                        help="Root with train/ val/ test/ folders + CSVs")
    parser.add_argument("--out_root",   type=str, required=True,
                        help="Output root for .npy files (use C: SSD)")
    parser.add_argument("--target_len", type=int, default=64)
    parser.add_argument("--splits",     nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        if (data_root / f"{split}.csv").exists():
            process_split(split, data_root, out_root, args.target_len)
        else:
            print(f"[SKIP] {split}.csv not found")


if __name__ == "__main__":
    main()
