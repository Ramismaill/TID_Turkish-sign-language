"""
inference_tmsnet_llm.py (v3)
Integrated live demo: TMS-Net sign recognition + Qwen-7B Turkish sentence generation.

v3 changes (performance):
- Resizable, larger display window (1280x720 default)
- MediaPipe model_complexity=0 (~2x faster skeleton extraction)
- LLM uses dedicated 4 threads (no fighting with webcam loop)
- Frame upscaling for display only (capture stays at 640x480 for speed)

v2 features (kept):
- Bug fix: clear current_sign after auto-translate
- Translation history logged to logs/translations.log

Usage:
  python src/inference_tmsnet_llm.py ^
    --checkpoint checkpoints/best.pth ^
    --class_map class_map.json ^
    --llm_model models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf ^
    --camera 1

Keys:
  q     = quit
  r     = reset buffer + last sentence
  Enter = translate buffer to sentence (immediate)
  c     = clear sign history
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
import threading
from torch.amp import autocast
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import KEEP_INDICES, NUM_NODES, BONE_PAIRS
from tmsnet_model import TMSNet
from llm_translator import GlossToTurkish

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

AUTO_TRANSLATE_IDLE_SEC = 2.0
MIN_BUFFER_SIZE         = 1
MAX_BUFFER_SIZE         = 8

# v3 performance
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720
LLM_THREADS    = 4   # Dedicated LLM threads (avoids fighting with webcam loop)


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
    """v3: model_complexity=0 for ~2x speedup. Was 1."""
    def __init__(self, model_complexity=0):
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


class TranslationManager:
    def __init__(self, llm_translator, idle_threshold=AUTO_TRANSLATE_IDLE_SEC,
                 on_complete_callback=None):
        self.translator     = llm_translator
        self.gloss_buffer   = []
        self.last_sentence  = ""
        self.last_glosses   = []
        self.translate_ms   = 0.0
        self.idle_threshold = idle_threshold
        self.on_complete    = on_complete_callback

        self._lock          = threading.Lock()
        self._is_translating = False
        self._last_sign_time = time.time()
        self._idle_start    = None

    def add_gloss(self, gloss):
        with self._lock:
            if self.gloss_buffer and self.gloss_buffer[-1] == gloss:
                return
            if len(self.gloss_buffer) >= MAX_BUFFER_SIZE:
                return
            self.gloss_buffer.append(gloss)
            self._last_sign_time = time.time()
            self._idle_start    = None

    def update_idle(self, is_idle):
        with self._lock:
            if is_idle:
                if self._idle_start is None:
                    self._idle_start = time.time()
            else:
                self._idle_start = None

    def should_auto_translate(self):
        with self._lock:
            if self._is_translating:
                return False
            if len(self.gloss_buffer) < MIN_BUFFER_SIZE:
                return False
            if self._idle_start is None:
                return False
            if (time.time() - self._idle_start) < self.idle_threshold:
                return False
            return True

    def trigger_translate(self):
        with self._lock:
            if self._is_translating:
                return False
            if not self.gloss_buffer:
                return False
            self._is_translating = True
            buffer_copy = list(self.gloss_buffer)

        thread = threading.Thread(
            target=self._translate_worker,
            args=(buffer_copy,),
            daemon=True,
        )
        thread.start()
        return True

    def _translate_worker(self, buffer_copy):
        try:
            sentence, elapsed = self.translator.translate(buffer_copy)
            if not sentence or len(sentence) < 2:
                sentence = " ".join(buffer_copy) + " (ceviri yok)"
                elapsed  = 0.0
        except Exception as e:
            print(f"[LLM] Translation error: {e}")
            sentence = " ".join(buffer_copy) + " (hata)"
            elapsed  = 0.0

        with self._lock:
            self.last_sentence  = sentence
            self.last_glosses   = buffer_copy
            self.translate_ms   = elapsed * 1000
            self.gloss_buffer.clear()
            self._idle_start    = None
            self._is_translating = False

        if self.on_complete:
            try:
                self.on_complete()
            except Exception as e:
                print(f"[Translation] on_complete callback error: {e}")

    def reset(self):
        with self._lock:
            self.gloss_buffer.clear()
            self.last_sentence = ""
            self.last_glosses  = []
            self._idle_start   = None

    def snapshot(self):
        with self._lock:
            return {
                'buffer':       list(self.gloss_buffer),
                'sentence':     self.last_sentence,
                'last_glosses': list(self.last_glosses),
                'translate_ms': self.translate_ms,
                'is_busy':      self._is_translating,
            }


class HUDRenderer:
    """v3: Larger fonts and bar to match bigger display window."""

    def render(self, frame, current_sign, confidence, state, top3,
               history, fps, trans_state):
        h, w = frame.shape[:2]
        BAR_H = 220   # v3: bigger bar for bigger window

        cv2.rectangle(frame, (0, h - BAR_H), (w, h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (640, 110), (0, 0, 0), -1)

        # Top-left
        cv2.putText(frame, f"TMS-Net 94.70% + Qwen-7B | {fps:.0f}fps",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
        col = (0, 255, 0) if state == MotionDetector.SIGNING else (120, 120, 120)
        cv2.putText(frame, state, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        if trans_state['is_busy']:
            cv2.putText(frame, "LLM: translating...", (10, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        else:
            cv2.putText(frame, "ENTER=translate  R=reset  Q=quit",
                        (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Top-right: gloss buffer
        if trans_state['buffer']:
            buf_text = "Buffer: " + " + ".join(trans_state['buffer'])
            (tw, _), _ = cv2.getTextSize(buf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            tx = w - tw - 25
            cv2.rectangle(frame, (tx - 12, 5), (w - 5, 50), (0, 0, 0), -1)
            cv2.putText(frame, buf_text, (tx, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)

        # Middle: current sign
        if current_sign and confidence > CONF_THRESH and state == MotionDetector.SIGNING:
            text  = current_sign.upper()
            fs, th = 2.0, 4
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx = (w - tw) // 2
            ty = h - BAR_H + 50
            cv2.putText(frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 4)
            color = (0, 230, 255) if confidence > 0.75 else (255, 200, 50)
            cv2.putText(frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, th)

        # Top-3
        if top3 and state == MotionDetector.SIGNING:
            alts = [f"{label} {prob*100:.0f}%" for label, prob in top3[:3]]
            alt_txt = "  |  ".join(alts)
            (aw, _), _ = cv2.getTextSize(alt_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            ax = (w - aw) // 2
            cv2.putText(frame, alt_txt, (ax, h - BAR_H + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Bottom: TURKISH SENTENCE
        if trans_state['sentence']:
            sent = trans_state['sentence']
            fs, th = 1.3, 2
            (tw, _), _ = cv2.getTextSize(sent, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            while tw > w - 40 and fs > 0.6:
                fs -= 0.1
                (tw, _), _ = cv2.getTextSize(sent, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx = (w - tw) // 2
            ty = h - 70
            cv2.putText(frame, sent, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 5)
            cv2.putText(frame, sent, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
            if trans_state['translate_ms'] > 0:
                info = f"({trans_state['translate_ms']:.0f}ms)"
                cv2.putText(frame, info, (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return frame


class SignLanguageRecognizerLLM:
    def __init__(self, checkpoint, class_map_path, llm_model_path,
                 camera_idx=1, conf_thresh=CONF_THRESH,
                 display_width=DISPLAY_WIDTH, display_height=DISPLAY_HEIGHT):
        with open(class_map_path, encoding='utf-8') as f:
            raw = json.load(f)
        self.class_map = {int(k): v for k, v in raw.items()}

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[TMS-Net] Device: {self.device}")
        self.model = TMSNet(num_classes=NUM_CLASSES, dropout=0.4).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        print(f"[TMS-Net] val_acc = {ckpt.get('val_acc', 0)*100:.2f}%")

        # v3: dedicated LLM threads to avoid fighting webcam loop
        translator = GlossToTurkish(
            llm_model_path,
            history_log="logs/translations.log",
            n_threads=LLM_THREADS,
        )
        translator.warmup()

        self.trans_mgr = TranslationManager(
            translator,
            on_complete_callback=self._on_translation_complete,
        )

        self.extractor    = LandmarkExtractor(model_complexity=0)  # v3: faster
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

        self._sign_lock = threading.Lock()

        # v3: display dimensions
        self.display_width  = display_width
        self.display_height = display_height
        self.window_name    = 'TMS-Net + Qwen-7B | Turkish Sign Language Live Translator'

        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_MSMF)

    def _on_translation_complete(self):
        with self._sign_lock:
            self.current_sign = None
            self._last_sign   = None
            self._sign_count  = 0
        print("[Demo] Sentence ready. Buffer cleared. Sign again.")

    def _predict(self):
        buf    = np.stack(list(self.frame_buffer))
        inputs = self.builder.build(buf, self.device)
        with torch.no_grad(), autocast(self.device):
            logits = self.model(**inputs)
            probs  = torch.softmax(logits, dim=1)[0].cpu()
        return probs

    def _update_sign(self, label_idx, conf):
        label = self.class_map.get(label_idx, str(label_idx))
        with self._sign_lock:
            self.confidence = conf
            if label == self._last_sign:
                self._sign_count += 1
            else:
                self._last_sign  = label
                self._sign_count = 1
            if self._sign_count >= 2 and label != self.current_sign:
                self.current_sign = label
                self.history.append(label)
                self.trans_mgr.add_gloss(label)
                self._sign_count = 0

    def run(self):
        print("\n" + "=" * 60)
        print("Live Demo: Sign Language -> Turkish Sentence (v3)")
        print("=" * 60)
        print("Keys:  Q=quit  |  R=reset  |  ENTER=translate  |  C=clear history")
        print(f"Display: {self.display_width}x{self.display_height}")
        print("=" * 60 + "\n")

        # v3: create resizable window with explicit size
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("[Webcam] Frame read failed.")
                break

            self.frame_count += 1
            self._fps_frames += 1
            now = time.time()
            if now - self._fps_t >= 1.0:
                self._fps        = self._fps_frames / (now - self._fps_t + 1e-9)
                self._fps_t      = now
                self._fps_frames = 0

            # v3: upscale display frame BEFORE drawing on it
            # (capture stays at native 640x480 for MediaPipe speed)
            display_frame = cv2.resize(frame, (self.display_width, self.display_height),
                                        interpolation=cv2.INTER_LINEAR)

            # MediaPipe processes ORIGINAL small frame (faster)
            vec, results = self.extractor.process(frame)
            self.frame_buffer.append(vec)
            state = self.motion.update(vec)

            self.trans_mgr.update_idle(state == MotionDetector.IDLE)

            if self.trans_mgr.should_auto_translate():
                self.trans_mgr.trigger_translate()

            # Draw skeleton on display_frame (the big one)
            mp_draw = mp.solutions.drawing_utils
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    display_frame, results.pose_landmarks,
                    mp.solutions.holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 255, 128), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 200, 100), thickness=2))
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(
                    display_frame, results.left_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(255, 200, 0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(200, 150, 0), thickness=2))
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(
                    display_frame, results.right_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 180, 255), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 140, 200), thickness=2))

            if (len(self.frame_buffer) == WINDOW_SIZE and
                self.frame_count % STRIDE == 0 and
                state == MotionDetector.SIGNING):
                probs = self._predict()
                label_idx, conf = self.smoother.update(probs)
                if label_idx is not None:
                    self._update_sign(label_idx, conf)
            elif state == MotionDetector.IDLE:
                self.smoother.reset()

            top3 = self.smoother.top3(self.class_map)
            trans_state = self.trans_mgr.snapshot()
            display_frame = self.hud.render(display_frame, self.current_sign, self.confidence,
                                             state, top3, self.history, self._fps,
                                             trans_state)

            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.trans_mgr.reset()
                with self._sign_lock:
                    self.current_sign = None
                    self._last_sign   = None
                    self._sign_count  = 0
                self.smoother.reset()
                print("[Reset] Buffer and sentence cleared.")
            elif key == ord('c'):
                self.history.clear()
                with self._sign_lock:
                    self.current_sign = None
                self.smoother.reset()
                print("[Clear] History cleared.")
            elif key == 13:  # ENTER
                if self.trans_mgr.trigger_translate():
                    print("[Translate] Triggered manually.")

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()


def main():
    parser = argparse.ArgumentParser(description='TMS-Net + LLM live demo (v3)')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--class_map', required=True)
    parser.add_argument('--llm_model', required=True)
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--conf', type=float, default=CONF_THRESH)
    parser.add_argument('--width',  type=int, default=DISPLAY_WIDTH,
                        help='Display window width (default 1280)')
    parser.add_argument('--height', type=int, default=DISPLAY_HEIGHT,
                        help='Display window height (default 720)')
    args = parser.parse_args()

    SignLanguageRecognizerLLM(
        checkpoint     = args.checkpoint,
        class_map_path = args.class_map,
        llm_model_path = args.llm_model,
        camera_idx     = args.camera,
        conf_thresh    = args.conf,
        display_width  = args.width,
        display_height = args.height,
    ).run()


if __name__ == "__main__":
    main()
