"""
inference_sml.py — Real-time Turkish Sign Language Recognition using SML model
Same pipeline as inference.py but uses SML (3-stream: joint + bone + motion)

Usage:
  python src\inference_sml.py ^
    --checkpoint "C:/AUTSL_project/checkpoints/best_sml.pt" ^
    --class_map  "C:/AUTSL_project/src/class_map.json"
"""

import cv2
import torch
import numpy as np
import collections
import json
import argparse
import sys
from torch.amp import autocast

import mediapipe as mp

sys.path.insert(0, 'src')
from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS
from sml_model import SML

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 64
STRIDE       = 4
CONF_THRESH  = 0.50
STABLE_COUNT = 1
IDLE_THRESH  = 0.008
NUM_CLASSES  = 226

POSE_LM        = 33
HAND_LM        = 21
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

BONE_SRC = [p[0] for p in BONE_PAIRS]
BONE_DST = [p[1] for p in BONE_PAIRS]


def extract_landmarks(results):
    def lm_to_xyz(lm_obj, n):
        if lm_obj is None:
            return np.zeros((n, 3), dtype=np.float32)
        arr = np.array([[l.x, l.y, l.z] for l in lm_obj.landmark],
                       dtype=np.float32)
        if arr.shape[0] != n:
            out = np.zeros((n, 3), dtype=np.float32)
            out[:min(n, arr.shape[0])] = arr[:min(n, arr.shape[0])]
            return out
        return arr

    pose = lm_to_xyz(results.pose_landmarks,       POSE_LM)
    lh   = lm_to_xyz(results.left_hand_landmarks,  HAND_LM)
    rh   = lm_to_xyz(results.right_hand_landmarks, HAND_LM)

    vec = np.concatenate([pose.reshape(-1), lh.reshape(-1), rh.reshape(-1)])

    ls = pose[LEFT_SHOULDER]
    rs = pose[RIGHT_SHOULDER]
    if not (np.allclose(ls, 0) and np.allclose(rs, 0)):
        center = (ls + rs) / 2.0
        xyz = vec.reshape(-1, 3)
        xyz -= center
        vec = xyz.reshape(-1)

    return vec.astype(np.float32)


def prepare_inputs(frame_buffer):
    """Returns joint, bone, motion tensors for SML — each (1, 3, 64, 56)"""
    x = np.stack(list(frame_buffer), axis=0)   # (64, 225)
    x = x[:, KEEP_INDICES]                      # (64, 168)
    x = x.reshape(64, NUM_NODES, 3)             # (64, 56, 3)
    x = x.transpose(2, 0, 1)                    # (3, 64, 56)

    joint = torch.tensor(x, dtype=torch.float32)

    # Bone: child - parent
    bone = joint[:, :, BONE_SRC] - joint[:, :, BONE_DST]

    # Motion: temporal difference
    motion = torch.zeros_like(joint)
    motion[:, 1:, :] = joint[:, 1:, :] - joint[:, :-1, :]

    return (joint.unsqueeze(0),   # (1, 3, 64, 56)
            bone.unsqueeze(0),
            motion.unsqueeze(0))


def is_moving(frame_buffer):
    if len(frame_buffer) < 8:
        return False
    recent = np.stack(list(frame_buffer)[-8:])
    sliced = recent[:, KEEP_INDICES].reshape(8, NUM_NODES, 3)
    hands  = sliced[:, 14:56, :]
    return np.std(hands) > IDLE_THRESH


def draw_hud(frame, gloss_sequence, current_sign, confidence, is_active, top3):
    h, w = frame.shape[:2]

    # Bottom subtitle bar
    bar_h   = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if current_sign and is_active and confidence > CONF_THRESH:
        text       = current_sign.upper()
        font_scale = 1.8
        thickness  = 3
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = (w - tw) // 2
        ty = h - bar_h + th + 12
        cv2.putText(frame, text, (tx+2, ty+2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+2)
        color = (0, 200, 255) if confidence > 0.70 else (255, 200, 0)
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    # Model label
    cv2.putText(frame, "SML Model", (10, h - bar_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # History
    if gloss_sequence:
        history = "  .  ".join(gloss_sequence[-6:])
        (hw, hh), _ = cv2.getTextSize(history, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        hx = (w - hw) // 2
        hy = h - bar_h - 10
        cv2.putText(frame, history, (hx+1, hy+1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, history, (hx, hy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Top-3
    if top3 and is_active:
        for i, (name, conf) in enumerate(top3):
            label = f"{i+1}. {name} {conf*100:.0f}%"
            cv2.putText(frame, label, (10, 30 + i*28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 255) if i == 0 else (180, 180, 180), 2)

    # Status dot
    color = (0, 255, 0) if is_active else (0, 0, 255)
    cv2.circle(frame, (w - 20, 20), 10, color, -1)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--class_map",  required=True)
    parser.add_argument("--camera",     type=int,   default=0)
    parser.add_argument("--dropout",    type=float, default=0.3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.class_map, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    model = SML(num_classes=NUM_CLASSES, dropout=args.dropout).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"SML Model loaded. Best val_acc: {ckpt.get('val_acc', 'N/A')}")

    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils

    frame_buffer   = collections.deque(maxlen=WINDOW_SIZE)
    gloss_sequence = []
    last_pred      = None
    stable_count   = 0
    frame_count    = 0
    current_sign   = None
    current_conf   = 0.0
    top3_preds     = []

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press Q=quit | C=clear | D=toggle debug")
    show_debug = True

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            vec = extract_landmarks(results)
            frame_buffer.append(vec)

            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2),
                mp_drawing.DrawingSpec(color=(0, 150, 200), thickness=1))
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 150, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(200, 100, 0), thickness=1))
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(100, 100, 255), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(100, 100, 200), thickness=1))

            active = is_moving(frame_buffer)

            if len(frame_buffer) == WINDOW_SIZE and frame_count % STRIDE == 0:
                joint, bone, motion = prepare_inputs(frame_buffer)
                joint  = joint.to(device)
                bone   = bone.to(device)
                motion = motion.to(device)

                with torch.no_grad():
                    with autocast("cuda"):
                        logits = model(joint, bone, motion)
                    prob = torch.softmax(logits, dim=1)[0]

                top3_vals, top3_idx = torch.topk(prob, 3)
                top3_preds = [
                    (class_map.get(str(top3_idx[i].item()), f"CLS_{top3_idx[i].item()}"),
                     top3_vals[i].item())
                    for i in range(3)
                ]

                conf_val  = top3_vals[0].item()
                pred_val  = top3_idx[0].item()
                sign_name = class_map.get(str(pred_val), f"CLS_{pred_val}")

                current_sign = sign_name
                current_conf = conf_val

                if conf_val > CONF_THRESH:
                    if sign_name == last_pred:
                        stable_count += 1
                    else:
                        stable_count = 0
                        last_pred    = sign_name

                    if stable_count >= STABLE_COUNT:
                        if not gloss_sequence or gloss_sequence[-1] != sign_name:
                            gloss_sequence.append(sign_name)
                            print(f"[SIGN] {sign_name} ({conf_val*100:.1f}%)")
                            stable_count = 0
                else:
                    stable_count = 0

            frame = draw_hud(frame, gloss_sequence,
                             current_sign if active else None,
                             current_conf, active,
                             top3_preds if show_debug and active else [])

            cv2.imshow("TID - SML Model", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                gloss_sequence.clear()
                print("[INFO] Cleared")
            elif key == ord('d'):
                show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final: {gloss_sequence}")


if __name__ == "__main__":
    main()
