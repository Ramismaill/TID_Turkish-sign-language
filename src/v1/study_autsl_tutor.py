"""
study_autsl_tutor.py (v6)
TID Sign Language Tutor (Phase 1) - Single Word Mode

v6 CHANGE (key remap only):
- 'P' (previous word) -> 'B'  (more intuitive: "Back")
- 'N' (next word) unchanged
- All other behavior identical to v5

v5 features (kept):
- New default DTW bounds: good=1.5, bad=4.0
- All CLI overrides
- 7-second recording

Usage:
  python src/study_autsl_tutor.py --class_map class_map.json ^
    --reference_dir reference_landmarks ^
    --llm_model models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf ^
    --camera 1 --start_word acele
"""

import os
import sys
import time
import json
import argparse
import collections
import threading
from pathlib import Path

import torch  # MUST be before llm_translator → loads CUDA runtime DLLs that llama.dll depends on
import cv2
import numpy as np
import mediapipe as mp
from dtw import dtw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import KEEP_INDICES, NUM_NODES
from llm_translator import GlossToTurkish

# ============ Constants ============
RECORD_FRAMES   = 64
COUNTDOWN_SEC   = 3
PANEL_WIDTH     = 640
PANEL_HEIGHT    = 720
TOTAL_WIDTH     = PANEL_WIDTH * 2
TOTAL_HEIGHT    = PANEL_HEIGHT
POSE_LM         = 33
HAND_LM         = 21
LEFT_SHOULDER   = 11
RIGHT_SHOULDER  = 12

REF_ANIM_FPS    = 20

DEFAULT_RECORD_TIME    = 7.0
DEFAULT_GOOD_DIST      = 1.5
DEFAULT_BAD_DIST       = 4.0
STRICT_GOOD_DIST       = 0.05
STRICT_BAD_DIST        = 0.50

EXCELLENT_THRESHOLD    = 60
GOOD_THRESHOLD         = 35

STATE_IDLE      = 'IDLE'
STATE_COUNTDOWN = 'COUNTDOWN'
STATE_RECORDING = 'RECORDING'
STATE_SCORING   = 'SCORING'
STATE_RESULT    = 'RESULT'

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


# ============ Reference Loader ============
class ReferenceLibrary:
    def __init__(self, reference_dir, class_map):
        self.reference_dir = Path(reference_dir)
        self.class_map     = class_map
        self.reference_cache = {}
        self._scan()

    def _scan(self):
        if not self.reference_dir.exists():
            print(f"[Reference] Warning: {self.reference_dir} not found")
            return
        files = list(self.reference_dir.glob("cls*.npy"))
        print(f"[Reference] Found {len(files)} reference files in {self.reference_dir}")
        if files:
            print(f"[Reference] Sample filename: {files[0].name}")

    def get(self, class_idx):
        if class_idx in self.reference_cache:
            return self.reference_cache[class_idx]

        samples = []
        for sample_idx in range(1, 5):
            path = self.reference_dir / f"cls{class_idx:03d}_{sample_idx}.npy"
            if path.exists():
                try:
                    arr = np.load(path)
                    samples.append(arr)
                except Exception as e:
                    print(f"[Reference] Failed to load {path}: {e}")

        if not samples:
            print(f"[Reference] No samples found for class {class_idx}")
            return None

        result = {
            'primary':   samples[0],
            'all':       samples,
            'count':     len(samples),
        }
        self.reference_cache[class_idx] = result
        print(f"[Reference] Loaded {len(samples)} samples for class {class_idx} "
              f"(shape: {samples[0].shape})")
        return result


# ============ Skeleton Renderer ============
class SkeletonRenderer:
    @staticmethod
    def _auto_fit_points(xyz, w, h):
        valid_mask = ~np.all(np.isclose(xyz, 0), axis=1)
        valid_xyz = xyz[valid_mask]

        if len(valid_xyz) < 3:
            return [None] * len(xyz)

        xs = valid_xyz[:, 0]
        ys = valid_xyz[:, 1]
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        xrange = max(0.1, xmax - xmin)
        yrange = max(0.1, ymax - ymin)

        def normalize(p):
            if np.allclose(p, 0):
                return None
            x = (p[0] - xmin) / xrange
            y = (p[1] - ymin) / yrange
            x = 0.1 + 0.8 * x
            y = 0.1 + 0.8 * y
            return (x, y)

        return [normalize(xyz[i]) for i in range(len(xyz))]

    @staticmethod
    def _draw_landmarks(canvas, points, connections, color, thickness=2,
                        circle_radius=4, w=PANEL_WIDTH, h=PANEL_HEIGHT):
        for a, b in connections:
            if a < len(points) and b < len(points):
                pa, pb = points[a], points[b]
                if pa is not None and pb is not None:
                    xa, ya = int(pa[0] * w), int(pa[1] * h)
                    xb, yb = int(pb[0] * w), int(pb[1] * h)
                    cv2.line(canvas, (xa, ya), (xb, yb), color, thickness)
        for p in points:
            if p is not None:
                x, y = int(p[0] * w), int(p[1] * h)
                cv2.circle(canvas, (x, y), circle_radius, color, -1)

    @staticmethod
    def render_frame(landmark_vec, w=PANEL_WIDTH, h=PANEL_HEIGHT, bg_color=(20, 20, 30)):
        canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)

        try:
            xyz = landmark_vec.reshape(-1, 3)
        except Exception:
            return canvas

        if xyz.shape[0] < POSE_LM + 2 * HAND_LM:
            return canvas

        all_pts = SkeletonRenderer._auto_fit_points(xyz, w, h)

        pose_pts = all_pts[:POSE_LM]
        lh_pts   = all_pts[POSE_LM:POSE_LM+HAND_LM]
        rh_pts   = all_pts[POSE_LM+HAND_LM:POSE_LM+2*HAND_LM]

        SkeletonRenderer._draw_landmarks(canvas, pose_pts, POSE_CONNECTIONS,
                                          (0, 255, 128), 3, 5, w, h)
        SkeletonRenderer._draw_landmarks(canvas, lh_pts, HAND_CONNECTIONS,
                                          (0, 200, 255), 2, 4, w, h)
        SkeletonRenderer._draw_landmarks(canvas, rh_pts, HAND_CONNECTIONS,
                                          (255, 180, 0), 2, 4, w, h)

        return canvas


# ============ DTW Scorer ============
class DTWScorer:
    def __init__(self, good_dist=DEFAULT_GOOD_DIST, bad_dist=DEFAULT_BAD_DIST):
        self.good_dist = good_dist
        self.bad_dist  = bad_dist
        print(f"[DTW] Calibration: good<={good_dist}, bad>={bad_dist}")

    def score(self, user_seq, reference_seq):
        try:
            if user_seq.shape[1] >= len(KEEP_INDICES):
                u = user_seq[:, KEEP_INDICES]
            else:
                u = user_seq
            if reference_seq.shape[1] >= len(KEEP_INDICES):
                r = reference_seq[:, KEEP_INDICES]
            else:
                r = reference_seq

            alignment = dtw(u, r, distance_only=True,
                            step_pattern='symmetric2')
            distance = alignment.distance
            path_len = max(len(u), len(r))
            normalized = distance / max(path_len, 1)

            if normalized <= self.good_dist:
                score = 100.0
            elif normalized >= self.bad_dist:
                score = 0.0
            else:
                score = 100.0 * (self.bad_dist - normalized) / (self.bad_dist - self.good_dist)

            return float(score), float(normalized)
        except Exception as e:
            print(f"[DTW] Error: {e}")
            return 0.0, 0.0

    def best_against(self, user_seq, reference_samples):
        best_score = 0.0
        best_norm  = float('inf')
        for ref in reference_samples:
            s, n = self.score(user_seq, ref)
            print(f"[DTW] vs sample: distance={n:.4f}, score={s:.1f}")
            if s > best_score:
                best_score = s
                best_norm  = n
        return best_score, best_norm


# ============ LLM Coach ============
class LLMCoach:
    def __init__(self, llm_translator):
        self.llm = llm_translator

    def get_feedback(self, word, score):
        return self._fallback(word, score)

    def _fallback(self, word, score):
        word_upper = word.upper()
        if score >= EXCELLENT_THRESHOLD:
            return f"Mukemmel! '{word_upper}' isaretini cok iyi yaptin."
        elif score >= GOOD_THRESHOLD:
            return f"Iyi gidiyorsun. '{word_upper}' icin biraz daha pratik yap."
        elif score >= 20:
            return f"Hareketleri biraz daha net yap. Referansi tekrar izle."
        else:
            return f"Tekrar dene. Sol panelden hareketleri dikkatlice izle."


# ============ Live Capture ============
class LiveCapture:
    def __init__(self, camera_idx=1):
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_MSMF)
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def grab(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        vec = self._to_vec(results)
        return frame, results, vec

    def _to_vec(self, results):
        pose = self._lm_xyz(results.pose_landmarks, POSE_LM)
        lh   = self._lm_xyz(results.left_hand_landmarks, HAND_LM)
        rh   = self._lm_xyz(results.right_hand_landmarks, HAND_LM)
        vec  = np.concatenate([pose.reshape(-1), lh.reshape(-1), rh.reshape(-1)])
        ls, rs = pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER]
        if not (np.allclose(ls, 0) and np.allclose(rs, 0)):
            center = (ls + rs) / 2.0
            xyz = vec.reshape(-1, 3)
            xyz -= center
            vec = xyz.reshape(-1)
        return vec.astype(np.float32)

    def _lm_xyz(self, lm_obj, n):
        if lm_obj is None:
            return np.zeros((n, 3), dtype=np.float32)
        arr = np.array([[l.x, l.y, l.z] for l in lm_obj.landmark], dtype=np.float32)
        out = np.zeros((n, 3), dtype=np.float32)
        out[:min(n, len(arr))] = arr[:min(n, len(arr))]
        return out

    def release(self):
        self.cap.release()
        self.holistic.close()


# ============ Tutor App ============
class TutorApp:

    def __init__(self, class_map_path, reference_dir, llm_model, camera_idx=1,
                 starting_word=None, record_seconds=DEFAULT_RECORD_TIME,
                 good_dist=DEFAULT_GOOD_DIST, bad_dist=DEFAULT_BAD_DIST,
                 strict=False):
        with open(class_map_path, encoding='utf-8') as f:
            raw = json.load(f)
        self.class_map = {int(k): v for k, v in raw.items()}
        self.classes = sorted(self.class_map.items(), key=lambda x: x[0])

        self.current_idx = 0
        if starting_word:
            for i, (idx, name) in enumerate(self.classes):
                if name.lower() == starting_word.lower():
                    self.current_idx = i
                    print(f"[Tutor] Starting at word: {name} (class {idx})")
                    break

        self.record_seconds = record_seconds
        print(f"[Tutor] Recording duration: {self.record_seconds}s")

        if strict:
            print("[Tutor] Mode: STRICT (objective evaluation)")
            self.scorer = DTWScorer(STRICT_GOOD_DIST, STRICT_BAD_DIST)
        else:
            print("[Tutor] Mode: LEARNING (calibrated for beginners)")
            self.scorer = DTWScorer(good_dist, bad_dist)

        self.refs   = ReferenceLibrary(reference_dir, self.class_map)
        self.coach  = LLMCoach(GlossToTurkish(llm_model, n_threads=4))
        self.coach.llm.warmup()
        self.cap    = LiveCapture(camera_idx)

        self.state            = STATE_IDLE
        self.recording_buf    = []
        self.recording_start  = 0.0
        self.countdown_start  = 0.0
        self.last_score       = None
        self.last_norm_dist   = None
        self.last_feedback    = ""
        self.attempts         = 0
        self.ref_anim_idx     = 0
        self.last_anim_step   = time.time()
        self._lock            = threading.Lock()
        self._is_scoring      = False
        self._recording_announced = False

        self.window_name = 'TID Tutor (Phase 1) v6 - Sign Language Education Platform'

    def _current_word(self):
        idx, name = self.classes[self.current_idx]
        return idx, name

    def _draw_left_panel(self):
        idx, name = self._current_word()
        ref = self.refs.get(idx)
        if ref is None:
            canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), (20, 20, 30), dtype=np.uint8)
            cv2.putText(canvas, "Bu kelime icin referans yok",
                        (60, PANEL_HEIGHT//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            return canvas

        primary = ref['primary']
        T = primary.shape[0]

        now = time.time()
        if now - self.last_anim_step >= (1.0 / REF_ANIM_FPS):
            self.ref_anim_idx = (self.ref_anim_idx + 1) % T
            self.last_anim_step = now

        frame_vec = primary[self.ref_anim_idx]
        canvas = SkeletonRenderer.render_frame(frame_vec)

        cv2.rectangle(canvas, (0, 0), (PANEL_WIDTH, 50), (0, 0, 0), -1)
        cv2.putText(canvas, f"REFERANS: {name.upper()}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2)
        cv2.putText(canvas, f"frame {self.ref_anim_idx + 1}/{T}",
                    (PANEL_WIDTH - 130, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return canvas

    def _draw_right_panel(self, frame, results):
        canvas = cv2.resize(frame, (PANEL_WIDTH, PANEL_HEIGHT))

        mp_draw = mp.solutions.drawing_utils
        if results and results.pose_landmarks:
            mp_draw.draw_landmarks(
                canvas, results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2))
        if results and results.left_hand_landmarks:
            mp_draw.draw_landmarks(
                canvas, results.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 150, 200), thickness=2))
        if results and results.right_hand_landmarks:
            mp_draw.draw_landmarks(
                canvas, results.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(255, 180, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(200, 130, 0), thickness=2))

        cv2.rectangle(canvas, (0, 0), (PANEL_WIDTH, 50), (0, 0, 0), -1)
        cv2.putText(canvas, "SIZIN DENEMENIZ (CANLI)",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 200, 50), 2)

        if self.state == STATE_COUNTDOWN:
            elapsed = time.time() - self.countdown_start
            remaining = COUNTDOWN_SEC - int(elapsed)
            txt = str(remaining) if remaining > 0 else "BASLA!"
            cv2.putText(canvas, txt,
                        (PANEL_WIDTH//2 - 100, PANEL_HEIGHT//2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 0), 12)
            cv2.putText(canvas, txt,
                        (PANEL_WIDTH//2 - 100, PANEL_HEIGHT//2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 6)
        elif self.state == STATE_RECORDING:
            elapsed = time.time() - self.recording_start
            progress = min(1.0, elapsed / self.record_seconds)
            cv2.rectangle(canvas, (50, 60), (PANEL_WIDTH-50, 90), (40, 40, 40), -1)
            cv2.rectangle(canvas, (50, 60), (50 + int((PANEL_WIDTH-100)*progress), 90),
                          (0, 0, 255), -1)
            time_left = max(0, self.record_seconds - elapsed)
            cv2.putText(canvas, f"KAYIT: {time_left:.1f}s",
                        (PANEL_WIDTH//2 - 110, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)

        return canvas

    def _draw_bottom_overlay(self, combined):
        h = combined.shape[0]
        BAR_H = 140
        cv2.rectangle(combined, (0, h - BAR_H), (TOTAL_WIDTH, h), (0, 0, 0), -1)

        idx, name = self._current_word()
        info = f"Hedef: {name.upper()}  ({self.current_idx + 1}/{len(self.classes)})"
        cv2.putText(combined, info, (20, h - BAR_H + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f"Deneme: {self.attempts}",
                    (20, h - BAR_H + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        if self.state == STATE_RESULT and self.last_score is not None:
            score = int(self.last_score)
            if score >= EXCELLENT_THRESHOLD:
                color, label = (0, 255, 0), "MUKEMMEL!"
            elif score >= GOOD_THRESHOLD:
                color, label = (0, 220, 255), "IYI"
            else:
                color, label = (0, 100, 255), "TEKRAR DENE"
            score_text = f"%{score}  {label}"
            (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            tx = (TOTAL_WIDTH - tw) // 2
            cv2.putText(combined, score_text, (tx, h - BAR_H + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
            cv2.putText(combined, score_text, (tx, h - BAR_H + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            if self.last_feedback:
                fb = self.last_feedback[:90]
                (tw, _), _ = cv2.getTextSize(fb, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                tx = (TOTAL_WIDTH - tw) // 2
                cv2.putText(combined, fb, (tx, h - BAR_H + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            if self.last_norm_dist is not None:
                debug = f"DTW dist={self.last_norm_dist:.3f}"
                cv2.putText(combined, debug, (TOTAL_WIDTH - 200, h - BAR_H + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        elif self.state == STATE_SCORING:
            text = "Hesaplaniyor..."
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            tx = (TOTAL_WIDTH - tw) // 2
            cv2.putText(combined, text, (tx, h - BAR_H + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        elif self.state == STATE_IDLE:
            # v6: B replaces P
            text = "SPACE: Basla     R: Tekrar     N/B: Sonraki/Onceki kelime     Q: Cikis"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx = (TOTAL_WIDTH - tw) // 2
            cv2.putText(combined, text, (tx, h - BAR_H + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return combined

    def _start_countdown(self):
        if self.state in (STATE_COUNTDOWN, STATE_RECORDING, STATE_SCORING):
            return
        self.state = STATE_COUNTDOWN
        self.countdown_start = time.time()
        self.recording_buf.clear()
        self._recording_announced = False
        print(f"[Tutor] Countdown started for {self._current_word()[1]}")

    def _check_countdown(self):
        if self.state == STATE_COUNTDOWN:
            elapsed = time.time() - self.countdown_start
            if elapsed >= COUNTDOWN_SEC:
                self.state = STATE_RECORDING
                self.recording_start = time.time()
                if not self._recording_announced:
                    print(f"[Tutor] Recording started for {self._current_word()[1]} ({self.record_seconds}s)")
                    self._recording_announced = True

    def _record_frame(self, vec):
        if self.state == STATE_RECORDING and vec is not None:
            self.recording_buf.append(vec.copy())
            elapsed = time.time() - self.recording_start
            if elapsed >= self.record_seconds or len(self.recording_buf) >= RECORD_FRAMES * 5:
                self._finalize_recording()

    def _finalize_recording(self):
        if len(self.recording_buf) == 0:
            self.state = STATE_IDLE
            return
        while len(self.recording_buf) < RECORD_FRAMES:
            self.recording_buf.append(self.recording_buf[-1])
        if len(self.recording_buf) > RECORD_FRAMES:
            indices = np.linspace(0, len(self.recording_buf) - 1, RECORD_FRAMES).astype(int)
            self.recording_buf = [self.recording_buf[i] for i in indices]
        print(f"[Tutor] Recording complete: {len(self.recording_buf)} frames (sampled from raw)")
        self.state = STATE_SCORING
        self._score_in_background()

    def _score_in_background(self):
        idx, name = self._current_word()
        ref = self.refs.get(idx)
        if ref is None:
            self.last_score    = 0
            self.last_feedback = "Bu kelime icin referans bulunamadi."
            self.state         = STATE_RESULT
            self.attempts     += 1
            return

        user_seq = np.stack(self.recording_buf)

        def worker():
            print(f"[Tutor] Scoring user_seq shape={user_seq.shape} vs {len(ref['all'])} reference samples")
            score, normalized = self.scorer.best_against(user_seq, ref['all'])
            feedback = self.coach.get_feedback(name, score)
            with self._lock:
                self.last_score      = score
                self.last_norm_dist   = normalized
                self.last_feedback   = feedback
                self.state           = STATE_RESULT
                self.attempts       += 1
                self._is_scoring     = False
            print(f"[Tutor] {name}: score={score:.1f}%, dist={normalized:.4f}, feedback={feedback}")

        self._is_scoring = True
        threading.Thread(target=worker, daemon=True).start()

    def _next_word(self):
        self.current_idx = (self.current_idx + 1) % len(self.classes)
        self._reset_state()
        print(f"[Tutor] Next: {self._current_word()[1]}")

    def _prev_word(self):
        self.current_idx = (self.current_idx - 1) % len(self.classes)
        self._reset_state()
        print(f"[Tutor] Back: {self._current_word()[1]}")

    def _reset_state(self):
        self.state         = STATE_IDLE
        self.last_score    = None
        self.last_norm_dist = None
        self.last_feedback = ""
        self.attempts      = 0
        self.recording_buf.clear()
        self.ref_anim_idx  = 0
        self._recording_announced = False

    def run(self):
        print("\n" + "=" * 60)
        print("TID Tutor (Phase 1) v6")
        print("=" * 60)
        print("Keys:")
        print(f"  SPACE  : Start (3-2-1 then {self.record_seconds:.0f} sec recording)")
        print("  R      : Retry same word")
        print("  N      : Next word")
        print("  B      : Back (previous word)")
        print("  Q      : Quit")
        print("=" * 60)
        print(f"Starting word: {self._current_word()[1]}")
        print("=" * 60 + "\n")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, TOTAL_WIDTH, TOTAL_HEIGHT)

        try:
            while True:
                frame, results, vec = self.cap.grab()
                if frame is None:
                    print("[Capture] Frame read failed.")
                    break

                self._check_countdown()
                if self.state == STATE_RECORDING:
                    self._record_frame(vec)

                left  = self._draw_left_panel()
                right = self._draw_right_panel(frame, results)
                combined = np.hstack([left, right])
                combined = self._draw_bottom_overlay(combined)

                cv2.imshow(self.window_name, combined)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self._start_countdown()
                elif key == ord('r'):
                    self._reset_state()
                elif key == ord('n'):
                    self._next_word()
                elif key == ord('b'):  # v6: was 'p', now 'b'
                    self._prev_word()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n[Tutor] Session ended.")


def main():
    parser = argparse.ArgumentParser(description='TID Tutor Phase 1 v6')
    parser.add_argument('--class_map', required=True)
    parser.add_argument('--reference_dir', default='reference_landmarks')
    parser.add_argument('--llm_model', required=True)
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--start_word', default=None)
    parser.add_argument('--record_seconds', type=float, default=DEFAULT_RECORD_TIME,
                        help=f'Recording time in seconds (default {DEFAULT_RECORD_TIME})')
    parser.add_argument('--good_dist', type=float, default=DEFAULT_GOOD_DIST,
                        help=f'DTW distance for 100%% (default {DEFAULT_GOOD_DIST})')
    parser.add_argument('--bad_dist', type=float, default=DEFAULT_BAD_DIST,
                        help=f'DTW distance for 0%% (default {DEFAULT_BAD_DIST})')
    parser.add_argument('--strict', action='store_true',
                        help='Use strict bounds (0.05/0.50) for objective evaluation')
    args = parser.parse_args()

    app = TutorApp(
        class_map_path = args.class_map,
        reference_dir  = args.reference_dir,
        llm_model      = args.llm_model,
        camera_idx     = args.camera,
        starting_word  = args.start_word,
        record_seconds = args.record_seconds,
        good_dist      = args.good_dist,
        bad_dist       = args.bad_dist,
        strict         = args.strict,
    )
    app.run()


if __name__ == "__main__":
    main()
