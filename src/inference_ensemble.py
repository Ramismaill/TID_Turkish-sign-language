"""
Real-time TMS-Net + SML Ensemble Inference
Combines both models at inference time for 95.13% accuracy.

Usage:
  python src/inference_ensemble.py \
    --sml_ckpt    "C:/AUTSL_project/checkpoints/best_sml.pt"    \
    --tmsnet_ckpt "C:/AUTSL_project/checkpoints/best_tmsnet.pt" \
    --class_map   "C:/AUTSL_project/src/class_map.json"         \
    --alpha 0.6

  alpha = weight for TMS-Net (0.6 = 60% TMS-Net, 40% SML)
"""

import os, sys, argparse, json, collections
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sml_model import SML
from tmsnet_model import TMSNet
from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS

try:
    import mediapipe as mp
except ImportError:
    print("Install mediapipe: pip install mediapipe==0.10.9")
    sys.exit(1)


# ─────────────────────────────────────────────
# Feature computation (same as train_tmsnet.py)
# ─────────────────────────────────────────────

def compute_bone(joint):
    bone = np.zeros_like(joint)
    for child, parent in BONE_PAIRS:
        if child < NUM_NODES and parent < NUM_NODES:
            bone[:, child, :] = joint[:, child, :] - joint[:, parent, :]
    return bone


def compute_angle(bone):
    angle = np.zeros_like(bone)
    for child, parent in BONE_PAIRS:
        if child < NUM_NODES and parent < NUM_NODES:
            b1 = bone[:, child, :]
            b2 = bone[:, parent, :]
            n1 = np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8
            n2 = np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8
            cos_t = np.sum(b1/n1 * b2/n2, axis=-1, keepdims=True).clip(-1, 1)
            sin_t = np.sqrt(1 - cos_t**2 + 1e-8)
            angle[:, child, 0:1] = cos_t
            angle[:, child, 1:2] = sin_t
    return angle


def temporal_diff(x):
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    return d


def extract_landmarks(results):
    """Extract 225 MediaPipe landmarks → (225,) array."""
    frame = np.zeros(225, dtype=np.float32)
    # Pose: 33 nodes × 3 = indices 0:99
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            if i < 33:
                frame[i*3]   = lm.x
                frame[i*3+1] = lm.y
                frame[i*3+2] = lm.z
    # Left hand: 21 nodes × 3 = indices 99:162
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            frame[99 + i*3]   = lm.x
            frame[99 + i*3+1] = lm.y
            frame[99 + i*3+2] = lm.z
    # Right hand: 21 nodes × 3 = indices 162:225
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            frame[162 + i*3]   = lm.x
            frame[162 + i*3+1] = lm.y
            frame[162 + i*3+2] = lm.z
    return frame


def buffer_to_streams(buffer):
    """Convert 64-frame buffer to all 6 TMS-Net streams + SML input."""
    raw = np.array(buffer, dtype=np.float32)   # (64, 225)

    # SML input: (6, 64, 56) — joint + velocity
    pruned = raw[:, KEEP_INDICES]              # (64, 168)
    joint_flat = pruned.reshape(64, NUM_NODES, 3)
    vel = np.zeros_like(joint_flat)
    vel[1:] = joint_flat[1:] - joint_flat[:-1]
    sml_input = np.concatenate([
        joint_flat.transpose(2, 0, 1),        # (3, 64, 56)
        vel.transpose(2, 0, 1) * 10.0         # (3, 64, 56)
    ], axis=0)                                 # (6, 64, 56)

    # TMS-Net streams
    joint   = joint_flat                       # (64, 56, 3)
    bone    = compute_bone(joint)
    jmotion = temporal_diff(joint)
    bmotion = temporal_diff(bone)
    angle   = compute_angle(bone)
    amotion = temporal_diff(angle)

    def to_t(arr):
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, 64, 56)

    return (
        torch.from_numpy(sml_input).unsqueeze(0),   # (1, 6, 64, 56)
        to_t(joint), to_t(bone), to_t(jmotion),
        to_t(bmotion), to_t(angle), to_t(amotion)
    )


def is_moving(buffer):
    if len(buffer) < 8:
        return False
    recent = np.array(list(buffer)[-8:])
    pruned = recent[:, KEEP_INDICES].reshape(8, NUM_NODES, 3)
    hands  = pruned[:, 14:, :]     # hand nodes
    return np.std(hands) > 0.008


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sml_ckpt',    required=True)
    parser.add_argument('--tmsnet_ckpt', required=True)
    parser.add_argument('--class_map',   required=True)
    parser.add_argument('--alpha',       type=float, default=0.6,
                        help='TMS-Net weight (1-alpha = SML weight)')
    parser.add_argument('--conf_thresh', type=float, default=0.55)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load class map
    with open(args.class_map) as f:
        class_map = json.load(f)

    # Load SML
    sml = SML(num_classes=226, dropout=0.3).to(device)
    ckpt = torch.load(args.sml_ckpt, map_location=device, weights_only=False)
    sml.load_state_dict(ckpt['model'])
    sml.eval()
    print(f"SML    loaded — val_acc: {ckpt.get('val_acc', 'N/A'):.4f}")

    # Load TMS-Net
    tms = TMSNet(num_classes=226, dropout=0.4).to(device)
    ckpt2 = torch.load(args.tmsnet_ckpt, map_location=device, weights_only=False)
    tms.load_state_dict(ckpt2['model'])
    tms.eval()
    print(f"TMS-Net loaded — val_acc: {ckpt2.get('val_acc', 'N/A'):.4f}")
    print(f"Ensemble alpha : TMS-Net={args.alpha:.1f} / SML={1-args.alpha:.1f}")
    print("Press Q=quit | C=clear | D=debug")

    # MediaPipe
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # State
    frame_buffer  = collections.deque(maxlen=64)
    gloss_seq     = []
    last_pred     = None
    stable_count  = 0
    frame_count   = 0
    debug_mode    = True
    current_sign  = ""
    current_conf  = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam found.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Extract landmarks
        lm = extract_landmarks(results)
        frame_buffer.append(lm)

        active = is_moving(frame_buffer)

        # Inference every 4 frames when buffer full
        if len(frame_buffer) == 64 and frame_count % 4 == 0 and active:
            sml_inp, j, b, jm, bm, ang, am = buffer_to_streams(frame_buffer)

            with torch.no_grad():
                # SML
                p_sml = F.softmax(sml(sml_inp.to(device)), dim=1)
                # TMS-Net
                p_tms = F.softmax(tms(
                    j.to(device), b.to(device), jm.to(device),
                    bm.to(device), ang.to(device), am.to(device)
                ), dim=1)

            # Weighted ensemble
            p_ens = args.alpha * p_tms + (1 - args.alpha) * p_sml
            conf, idx = torch.max(p_ens, 1)
            conf  = conf.item()
            idx   = idx.item()
            label = class_map.get(str(idx), f"SIGN_{idx:03d}")

            if conf >= args.conf_thresh:
                current_sign = label
                current_conf = conf

                if label == last_pred:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_pred = label

                if stable_count == 1:
                    if not gloss_seq or gloss_seq[-1] != label:
                        gloss_seq.append(label)
                        print(f"[SIGN] {label} ({conf*100:.1f}%)")
                        if len(gloss_seq) > 8:
                            gloss_seq.pop(0)

        # ── Draw UI ──
        h, w = frame.shape[:2]

        # Status dot
        dot_color = (0, 255, 100) if active else (0, 0, 200)
        cv2.circle(frame, (w - 20, 20), 10, dot_color, -1)

        # Debug panel — top-left: top-3 predictions
        if debug_mode and current_sign:
            cv2.putText(frame, f"TOP: {current_sign} ({current_conf*100:.0f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # History — small gray text above subtitle
        if len(gloss_seq) > 1:
            history = "  ".join(gloss_seq[:-1])
            cv2.putText(frame, history,
                        (w//2 - 200, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (160, 160, 160), 1, cv2.LINE_AA)

        # Main subtitle — bottom center, large white text
        if gloss_seq:
            text = gloss_seq[-1].upper()
            font_scale = 1.4
            thickness  = 3
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
            tx = (w - tw) // 2
            ty = h - 30

            # Shadow
            cv2.putText(frame, text, (tx+2, ty+2),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness+2, cv2.LINE_AA)
            # Text
            cv2.putText(frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Model label top-right
        label_text = f"TMS+SML [{args.alpha:.0%}/{1-args.alpha:.0%}]"
        cv2.putText(frame, label_text, (w - 260, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 255), 1)

        cv2.imshow("TMS-Net + SML Ensemble | TID Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            gloss_seq.clear()
            last_pred    = None
            stable_count = 0
            current_sign = ""
            print("[INFO] Cleared")
        elif key == ord('d'):
            debug_mode = not debug_mode

    print(f"Final: {gloss_seq}")
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()


if __name__ == '__main__':
    main()
