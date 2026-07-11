"""
inference_tmsnet.py
Production-quality live Turkish Sign Language Recognition — TMS-Net
OOP architecture with proper sign segmentation and temporal smoothing.

Usage:
  python src\inference_tmsnet.py ^
    --checkpoint "C:/AUTSL_project/checkpoints/best.pth" ^
    --class_map  "C:/AUTSL_project/src/class_map.json"
"""

import cv2
import torch
import numpy as np
import collections
import json
import argparse
import sys
import os
import time
from torch.amp import autocast
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS
from tmsnet_model import TMSNet

WINDOW_SIZE    = 64
STRIDE         = 2
CONF_THRESH    = 0.45
IDLE_THRESH    = 0.006
MOTION_THRESH  = 0.012
NUM_CLASSES    = 226
POSE_LM        = 33
HAND_LM        = 21
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

def _build_angle_pairs(bone_pairs):
    pairs = []
    n = len(bone_pairs)
    for i in range(n):
        for j in range(i + 1, n):
            if set(bone_pairs[i]) & set(bone_pairs[j]):
                pairs.append((i, j))
    return pairs

ANGLE_PAIRS = _build_angle_pairs(BONE_PAIRS)
BONE_SRC    = [p[0] for p in BONE_PAIRS]
BONE_DST    = [p[1] for p in BONE_PAIRS]


class LandmarkExtractor:
    def __init__(self, model_complexity=1):
        self._mp = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, bgr_frame):
        rgb     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self._mp.process(rgb)
        return self._to_vec(results), results

    def _lm_xyz(self, lm_obj, n):
        if lm_obj is None:
            return np.zeros((n, 3), dtype=np.float32)
        arr = np.array([[l.x, l.y, l.z] for l in lm_obj.landmark], dtype=np.float32)
        out = np.zeros((n, 3), dtype=np.float32)
        out[:min(n, len(arr))] = arr[:min(n, len(arr))]
        return out

    def _to_vec(self, results):
        pose = self._lm_xyz(results.pose_landmarks,       POSE_LM)
        lh   = self._lm_xyz(results.left_hand_landmarks,  HAND_LM)
        rh   = self._lm_xyz(results.right_hand_landmarks, HAND_LM)
        vec  = np.concatenate([pose.reshape(-1), lh.reshape(-1), rh.reshape(-1)])
        ls, rs = pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER]
        if not (np.allclose(ls, 0) and np.allclose(rs, 0)):
            center = (ls + rs) / 2.0
            xyz    = vec.reshape(-1, 3)
            xyz   -= center
            vec    = xyz.reshape(-1)
        return vec.astype(np.float32)

    def close(self):
        self._mp.close()


class MotionDetector:
    IDLE    = 'IDLE'
    SIGNING = 'SIGNING'

    def __init__(self, history_len=10):
        self._history = collections.deque(maxlen=history_len)
        self._state   = self.IDLE

    def update(self, vec):
        sliced = vec[KEEP_INDICES].reshape(NUM_NODES, 3)
        motion = float(np.std(sliced[14:56]))
        self._history.append(motion)
        avg = float(np.mean(self._history)) if self._history else 0.0
        if avg > MOTION_THRESH:
            self._state = self.SIGNING
        elif avg < IDLE_THRESH:
            self._state = self.IDLE
        return self._state

    @property
    def state(self):
        return self._state


class StreamBuilder:
    def build(self, buffer, device):
        x   = buffer[:, KEEP_INDICES].reshape(64, NUM_NODES, 3)
        T, V, _ = x.shape
        bone = self._bone(x)
        jmot = self._motion(x)
        bmot = self._motion(bone)
        angle, amot = self._angle_streams(x, T, V)

        def ctv(a):
            t = torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32))
            return t.unsqueeze(0).to(device)

        return {'joint': ctv(x), 'bone': ctv(bone), 'jmotion': ctv(jmot),
                'bmotion': ctv(bmot), 'angle': ctv(angle), 'amotion': ctv(amot)}

    def _bone(self, x):
        bone = np.zeros_like(x)
        for parent, child in BONE_PAIRS:
            bone[:, child] = x[:, child] - x[:, parent]
        return bone

    def _motion(self, x):
        mot = np.zeros_like(x)
        mot[:-1] = x[1:] - x[:-1]
        return mot

    def _angle_streams(self, x, T, V):
        bv = np.zeros((T, len(BONE_PAIRS), 3), dtype=np.float32)
        for idx, (parent, child) in enumerate(BONE_PAIRS):
            bv[:, idx] = x[:, child] - x[:, parent]
        an = np.zeros((T, V, 3), dtype=np.float32)
        for bi, bj in ANGLE_PAIRS:
            vi = bv[:, bi]; vj = bv[:, bj]
            dot = (vi * vj).sum(axis=-1)
            cosine = dot / (np.linalg.norm(vi, axis=-1).clip(1e-9) *
                            np.linalg.norm(vj, axis=-1).clip(1e-9))
            an[:, BONE_PAIRS[bi][1], 0] += cosine
        counts = np.zeros(V, dtype=np.float32)
        for bi, _ in ANGLE_PAIRS:
            counts[BONE_PAIRS[bi][1]] += 1
        an /= counts.clip(min=1)[np.newaxis, :, np.newaxis]
        am = np.zeros_like(an); am[:-1] = an[1:] - an[:-1]
        return an, am


class PredictionSmoother:
    def __init__(self, window=4, conf_thresh=CONF_THRESH):
        self._probs  = collections.deque(maxlen=window)
        self._thresh = conf_thresh

    def update(self, probs):
        self._probs.append(probs)
        avg  = torch.stack(list(self._probs)).mean(0)
        conf, idx = avg.max(0)
        if conf.item() >= self._thresh:
            return idx.item(), conf.item()
        return None, 0.0

    def top3(self, class_map):
        if not self._probs:
            return []
        avg = torch.stack(list(self._probs)).mean(0)
        vals, idxs = avg.topk(3)
        return [(class_map.get(i.item(), str(i.item())), v.item())
                for i, v in zip(idxs, vals)]

    def reset(self):
        self._probs.clear()


class HUDRenderer:
    def render(self, frame, current_sign, confidence, state, top3, history, fps):
        h, w = frame.shape[:2]
        BAR_H = 160

        # ── Solid black bottom bar ──────────────────────────────────────────
        cv2.rectangle(frame, (0, h - BAR_H), (w, h), (0, 0, 0), -1)

        # ── Top-left info ───────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (320, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"TMS-Net 94.70% | {fps:.0f}fps",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)
        col = (0, 255, 0) if state == MotionDetector.SIGNING else (120, 120, 120)
        cv2.putText(frame, state,
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        # ── Main prediction (large, bottom-center) ──────────────────────────
        if current_sign and confidence > CONF_THRESH:
            conf = confidence
            text  = current_sign.upper()
            fs, th = 3.5, 5
            (tw, th_px), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx = (w - tw) // 2
            ty = h - 55
            # Black outline
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), th+4)
            # Coloured text
            color = (0, 230, 255) if conf > 0.75 else (255, 200, 50)
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th)

            # Confidence bar inside black bar
            bw = int((w - 100) * conf)
            cv2.rectangle(frame, (50, h-18), (w-50, h-6), (60,60,60), -1)
            cv2.rectangle(frame, (50, h-18), (50+bw, h-6), color, -1)

        # ── Alternatives (inside black bar, small text) ─────────────────────
        if top3 and state == MotionDetector.SIGNING:
            alts = [f"{label} {prob*100:.0f}%" for label, prob in top3[:3]]
            alt_txt = "  |  ".join(alts)
            (aw, _), _ = cv2.getTextSize(alt_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ax = (w - aw) // 2
            cv2.putText(frame, alt_txt, (ax, h - BAR_H + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── History (inside black bar) ──────────────────────────────────────
        if history:
            hist = "  >  ".join(list(history)[-5:])
            (hw, _), _ = cv2.getTextSize(hist, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            hx = (w - hw) // 2
            cv2.putText(frame, hist, (hx, h - BAR_H + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return frame


class SignLanguageRecognizer:
    def __init__(self, checkpoint, class_map_path, camera_idx=0, conf_thresh=CONF_THRESH):
        with open(class_map_path, encoding='utf-8') as f:
            raw = json.load(f)
        self.class_map = {int(k): v for k, v in raw.items()}

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device : {self.device}")

        self.model = TMSNet(num_classes=NUM_CLASSES, dropout=0.4).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        print(f"Model  : val_acc={ckpt.get('val_acc',0)*100:.2f}%")

        self.extractor    = LandmarkExtractor()
        self.motion       = MotionDetector()
        self.builder      = StreamBuilder()
        self.smoother     = PredictionSmoother(window=4, conf_thresh=conf_thresh)
        self.hud          = HUDRenderer()
        self.frame_buffer = collections.deque(maxlen=WINDOW_SIZE)
        self.history      = collections.deque(maxlen=10)
        self.current_sign = None
        self.confidence   = 0.0
        self.frame_count  = 0
        self._last_sign   = None
        self._sign_count  = 0
        self._fps_t       = time.time()
        self._fps         = 0.0
        self._fps_frames  = 0

        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def _predict(self):
        buf    = np.stack(list(self.frame_buffer))
        inputs = self.builder.build(buf, self.device)
        with torch.no_grad(), autocast(self.device):
            logits = self.model(**inputs)
            probs  = torch.softmax(logits, dim=1)[0].cpu()
        return probs

    def _update_sign(self, label_idx, conf):
        label = self.class_map.get(label_idx, str(label_idx))
        self.confidence = conf
        if label == self._last_sign:
            self._sign_count += 1
        else:
            self._last_sign  = label
            self._sign_count = 1
        if self._sign_count >= 2 and label != self.current_sign:
            self.current_sign = label
            self.history.append(label)
            self._sign_count  = 0

    def run(self):
        print("Press Q to quit | C to clear history")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_count += 1
            self._fps_frames += 1
            now = time.time()
            if now - self._fps_t >= 1.0:
                self._fps        = self._fps_frames / (now - self._fps_t + 1e-9)
                self._fps_t      = now
                self._fps_frames = 0

            vec, results = self.extractor.process(frame)
            self.frame_buffer.append(vec)
            state = self.motion.update(vec)

            # Draw skeleton
            mp_draw = mp.solutions.drawing_utils
            mp_styles = mp.solutions.drawing_styles
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp.solutions.holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,128), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0,200,100), thickness=2))
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, results.left_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(255,200,0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(200,150,0), thickness=2))
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, results.right_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,180,255), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0,140,200), thickness=2))

            if (len(self.frame_buffer) == WINDOW_SIZE
                    and self.frame_count % STRIDE == 0
                    and state == MotionDetector.SIGNING):
                probs          = self._predict()
                label_idx, conf = self.smoother.update(probs)
                if label_idx is not None:
                    self._update_sign(label_idx, conf)
            elif state == MotionDetector.IDLE:
                self.smoother.reset()

            top3  = self.smoother.top3(self.class_map)
            frame = self.hud.render(frame, self.current_sign, self.confidence,
                                    state, top3, self.history, self._fps)
            cv2.imshow('TMS-Net | Turk Isaret Dili Canli Ceviri (94.70%)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.history.clear()
                self.current_sign = None
                self.smoother.reset()
        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--class_map',   required=True)
    parser.add_argument('--camera',      type=int,   default=0)
    parser.add_argument('--conf',        type=float, default=CONF_THRESH)
    args = parser.parse_args()

    SignLanguageRecognizer(
        checkpoint     = args.checkpoint,
        class_map_path = args.class_map,
        camera_idx     = args.camera,
        conf_thresh    = args.conf,
    ).run()


if __name__ == '__main__':
    main()
