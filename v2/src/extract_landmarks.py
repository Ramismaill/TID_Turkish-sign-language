import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import mediapipe as mp

POSE_LM = 33
HAND_LM = 21
FEAT_DIM = (POSE_LM + HAND_LM + HAND_LM) * 3  # 225
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12


def lm_list_to_xyz(landmarks, expected_count):
    if landmarks is None:
        return np.zeros((expected_count, 3), dtype=np.float32)
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    if arr.shape[0] != expected_count:
        out = np.zeros((expected_count, 3), dtype=np.float32)
        n = min(expected_count, arr.shape[0])
        out[:n] = arr[:n]
        return out
    return arr


def normalize_by_shoulder_center(frame_vec):
    xyz = frame_vec.reshape(-1, 3)
    pose = xyz[:POSE_LM]
    ls = pose[LEFT_SHOULDER_IDX]
    rs = pose[RIGHT_SHOULDER_IDX]
    if np.allclose(ls, 0.0) and np.allclose(rs, 0.0):
        return frame_vec
    center = (ls + rs) / 2.0
    xyz = xyz - center
    return xyz.reshape(-1).astype(np.float32)


def resample_sequence(seq, target_len=64):
    t, d = seq.shape
    if t == target_len:
        return seq.astype(np.float32)
    if t == 0:
        return np.zeros((target_len, d), dtype=np.float32)
    if t == 1:
        return np.repeat(seq, target_len, axis=0).astype(np.float32)
    old_idx = np.linspace(0, 1, t)
    new_idx = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, d), dtype=np.float32)
    for i in range(d):
        out[:, i] = np.interp(new_idx, old_idx, seq[:, i])
    return out


def extract_video_landmarks(video_path, holistic, target_len=64):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        pose_xyz = lm_list_to_xyz(results.pose_landmarks, POSE_LM)
        left_xyz = lm_list_to_xyz(results.left_hand_landmarks, HAND_LM)
        right_xyz = lm_list_to_xyz(results.right_hand_landmarks, HAND_LM)
        vec = np.concatenate([pose_xyz.reshape(-1), left_xyz.reshape(-1), right_xyz.reshape(-1)], axis=0)
        vec = normalize_by_shoulder_center(vec)
        frames.append(vec)
    cap.release()
    if len(frames) == 0:
        seq = np.zeros((1, FEAT_DIM), dtype=np.float32)
    else:
        seq = np.stack(frames).astype(np.float32)
    seq = resample_sequence(seq, target_len=target_len)
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_root", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--target_len", type=int, default=64)
    args = parser.parse_args()

    videos_root = Path(args.videos_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    mp_holistic = mp.solutions.holistic
    manifest = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            rel_name = str(row["filename"])
            label = int(row["label"])
            video_path = videos_root / rel_name
            if not video_path.exists():
                if video_path.suffix == "":
                    video_path = video_path.with_suffix(".mp4")
            if not video_path.exists():
                print(f"[WARN] Missing video: {rel_name}")
                continue
            seq = extract_video_landmarks(video_path, holistic, target_len=args.target_len)
            out_path = out_root / Path(rel_name).with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, seq)
            manifest.append({"npy_path": str(out_path), "label": label})

    manifest_df = pd.DataFrame(manifest)
    manifest_file = out_root / "manifest.csv"
    manifest_df.to_csv(manifest_file, index=False)
    print(f"Done. {len(manifest_df)} samples saved to {manifest_file}")


if __name__ == "__main__":
    main()