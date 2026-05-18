"""
sentence_tutor.py (v3)
TID Sentence Tutor (Phase 2) - Practice full Turkish sentences word by word.

v3 CHANGE (key remap only):
- 'B' replaces 'P'-like back. (Note: sentence_tutor never had 'P' for prev,
  but this version aligns key vocabulary with study_autsl_tutor v6.)
- 'N' for next word, 'R' for reset, 'Q' for quit unchanged

v2 features (kept):
- Shared LLM instance (saves 4.4 GB RAM)
- 150-word vocab in prompt

Usage:
  python src/sentence_tutor.py ^
    --class_map class_map.json ^
    --reference_dir reference_landmarks ^
    --llm_model models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf ^
    --camera 1 ^
    --theme yemek
"""

import os
import sys
import time
import json
import argparse
import collections
import threading
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import KEEP_INDICES, NUM_NODES
from llm_translator import GlossToTurkish
from study_autsl_tutor import (
    ReferenceLibrary, SkeletonRenderer, DTWScorer, LiveCapture,
    PANEL_WIDTH, PANEL_HEIGHT, COUNTDOWN_SEC,
    POSE_LM, HAND_LM,
    DEFAULT_RECORD_TIME, DEFAULT_GOOD_DIST, DEFAULT_BAD_DIST,
    REF_ANIM_FPS, RECORD_FRAMES,
    EXCELLENT_THRESHOLD, GOOD_THRESHOLD,
    STATE_IDLE, STATE_COUNTDOWN, STATE_RECORDING, STATE_SCORING, STATE_RESULT,
    POSE_CONNECTIONS, HAND_CONNECTIONS,
)

TOTAL_WIDTH    = PANEL_WIDTH * 2
TOTAL_HEIGHT   = PANEL_HEIGHT
MAX_ATTEMPTS_PER_WORD = 3
MIN_VALID_GLOSSES = 2
MAX_LLM_REGEN_TRIES = 3

PHASE_THEME       = 'THEME'
PHASE_GENERATING  = 'GENERATING'
PHASE_PRACTICING  = 'PRACTICING'
PHASE_FINAL       = 'FINAL'


class SentenceGenerator:
    def __init__(self, llm_instance, vocabulary):
        self.llm = llm_instance
        self.vocabulary = vocabulary
        print(f"[Sentence] Using shared LLM instance, vocab size: {len(vocabulary)}")

    def generate(self, theme):
        vocab_sample = sorted(list(self.vocabulary))

        prompt = f"""<|im_start|>system
Sen bir Turk Isaret Dili (TID) ogretmenisin. Verilen tema icin ogrenciye pratik yapmak uzere kisa bir Turkce cumle uretirsin.

KURALLAR:
- SADECE asagidaki kelime listesinden kelimeler kullan.
- Cumle 2-5 kelime arasi olsun.
- Tema ile ilgili anlamli bir cumle olsun.
- TID grameri: fiil sonda olur.
- Sadece cumleyi yaz, aciklama ekleme.

KULLANILABILIR KELIMELER:
{', '.join(vocab_sample[:150])}

ORNEKLER:
Tema: yemek -> Cumle: ben su icmek
Tema: aile -> Cumle: anne baba ben
Tema: okul -> Cumle: ben okul gitmek
Tema: selamlama -> Cumle: merhaba arkadas
<|im_end|>
<|im_start|>user
Tema: {theme}
Cumle:<|im_end|>
<|im_start|>assistant
"""

        t0 = time.time()
        out = self.llm(
            prompt,
            max_tokens=30,
            temperature=0.6,
            stop=["<|im_end|>", "\n", "Tema:"],
            echo=False,
        )
        elapsed = time.time() - t0

        raw = out["choices"][0]["text"].strip()
        raw = raw.replace("Cumle:", "").strip()
        raw = raw.lower()

        print(f"[Sentence] LLM generated: '{raw}' ({elapsed:.1f}s)")
        return raw

    def extract_glosses(self, sentence):
        words = sentence.replace(",", " ").replace(".", " ").replace("?", " ").split()
        words = [w.strip().lower() for w in words if w.strip()]

        valid = [w for w in words if w in self.vocabulary]
        invalid = [w for w in words if w not in self.vocabulary]

        if invalid:
            print(f"[Sentence] Filtered out unknown words: {invalid}")

        return valid

    def generate_with_retries(self, theme):
        glosses = []
        raw = ""
        for attempt in range(1, MAX_LLM_REGEN_TRIES + 1):
            print(f"[Sentence] Attempt {attempt}/{MAX_LLM_REGEN_TRIES}")
            raw = self.generate(theme)
            glosses = self.extract_glosses(raw)
            if len(glosses) >= MIN_VALID_GLOSSES:
                return raw, glosses
            print(f"[Sentence] Only {len(glosses)} valid glosses, retrying...")
        print(f"[Sentence] Using best-effort glosses: {glosses}")
        return raw, glosses


class WordSession:
    def __init__(self, word, class_idx):
        self.word      = word
        self.class_idx = class_idx
        self.attempts  = []
        self.best_score = 0.0
        self.best_dist  = float('inf')

    def add_attempt(self, score, distance):
        self.attempts.append((score, distance))
        if score > self.best_score:
            self.best_score = score
            self.best_dist  = distance

    @property
    def attempts_used(self):
        return len(self.attempts)

    @property
    def attempts_remaining(self):
        return max(0, MAX_ATTEMPTS_PER_WORD - len(self.attempts))

    @property
    def is_complete(self):
        return len(self.attempts) >= MAX_ATTEMPTS_PER_WORD


class SentenceTutorApp:

    def __init__(self, class_map_path, reference_dir, llm_model, camera_idx=1,
                 theme=None, record_seconds=DEFAULT_RECORD_TIME,
                 good_dist=DEFAULT_GOOD_DIST, bad_dist=DEFAULT_BAD_DIST):
        with open(class_map_path, encoding='utf-8') as f:
            raw = json.load(f)
        self.class_map = {int(k): v for k, v in raw.items()}
        self.name_to_idx = {name.lower(): idx for idx, name in self.class_map.items()}
        self.vocab = set(self.name_to_idx.keys())

        print(f"[App] Loaded {len(self.vocab)} known signs")

        self.refs = ReferenceLibrary(reference_dir, self.class_map)
        self.scorer = DTWScorer(good_dist, bad_dist)
        self.cap = LiveCapture(camera_idx)
        self.record_seconds = record_seconds

        self.translator = GlossToTurkish(llm_model, n_threads=4)
        self.translator.warmup()
        self.gen = SentenceGenerator(self.translator.llm, self.vocab)

        self.theme = theme
        self.raw_sentence = ""
        self.glosses = []
        self.sessions = []
        self.current_word_idx = 0
        self.phase = PHASE_THEME

        self.state            = STATE_IDLE
        self.recording_buf    = []
        self.recording_start  = 0.0
        self.countdown_start  = 0.0
        self.last_score       = None
        self.last_norm_dist   = None
        self.last_feedback    = ""
        self.ref_anim_idx     = 0
        self.last_anim_step   = time.time()
        self._lock            = threading.Lock()
        self._is_scoring      = False
        self._recording_announced = False

        self.composite_score = 0.0
        self.final_feedback  = ""

        self.window_name = 'TID Sentence Tutor (Phase 2) v3 - Education Platform'

    def _current_session(self):
        if self.current_word_idx < len(self.sessions):
            return self.sessions[self.current_word_idx]
        return None

    def _setup_sentence(self, theme):
        print(f"\n[App] Theme: {theme}")
        self.theme = theme
        self.phase = PHASE_GENERATING
        self.raw_sentence, self.glosses = self.gen.generate_with_retries(theme)

        if len(self.glosses) < MIN_VALID_GLOSSES:
            print(f"[App] WARNING: only {len(self.glosses)} valid glosses, proceeding anyway")
            if not self.glosses:
                fallback = ['ben', 'su', 'icmek']
                self.glosses = [w for w in fallback if w in self.vocab]
                print(f"[App] Using fallback glosses: {self.glosses}")

        self.sessions = []
        for word in self.glosses:
            class_idx = self.name_to_idx.get(word.lower())
            if class_idx is not None:
                self.sessions.append(WordSession(word, class_idx))

        print(f"[App] Sentence: {self.raw_sentence}")
        print(f"[App] Practicing {len(self.sessions)} words: {[s.word for s in self.sessions]}")

        self.current_word_idx = 0
        self.phase = PHASE_PRACTICING
        self._reset_attempt_state()

    def _reset_attempt_state(self):
        self.state = STATE_IDLE
        self.last_score = None
        self.last_norm_dist = None
        self.last_feedback = ""
        self.recording_buf.clear()
        self.ref_anim_idx = 0
        self._recording_announced = False

    def _draw_left_panel(self):
        sess = self._current_session()
        if sess is None or self.phase == PHASE_FINAL:
            return self._draw_final_left()

        ref = self.refs.get(sess.class_idx)
        if ref is None:
            canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), (20, 20, 30), dtype=np.uint8)
            cv2.putText(canvas, f"Referans yok: {sess.word}",
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
        cv2.putText(canvas, f"REFERANS: {sess.word.upper()}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2)
        cv2.putText(canvas, f"frame {self.ref_anim_idx + 1}/{T}",
                    (PANEL_WIDTH - 130, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return canvas

    def _draw_final_left(self):
        canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), (20, 20, 30), dtype=np.uint8)

        cv2.putText(canvas, "CUMLE TAMAMLANDI",
                    (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 128), 2)
        cv2.putText(canvas, f"\"{self.raw_sentence}\"",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        y = 160
        cv2.putText(canvas, "Kelime Skorlari:",
                    (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 40
        for sess in self.sessions:
            score = int(sess.best_score)
            if score >= EXCELLENT_THRESHOLD:
                col = (0, 255, 0)
            elif score >= GOOD_THRESHOLD:
                col = (0, 220, 255)
            else:
                col = (0, 100, 255)
            txt = f"  {sess.word.upper():<15s}  %{score:3d}  ({sess.attempts_used} deneme)"
            cv2.putText(canvas, txt, (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            y += 35

        y += 20
        cv2.line(canvas, (50, y - 10), (PANEL_WIDTH - 50, y - 10), (100, 100, 100), 1)
        comp_score = int(self.composite_score)
        if comp_score >= EXCELLENT_THRESHOLD:
            col, label = (0, 255, 0), "MUKEMMEL"
        elif comp_score >= GOOD_THRESHOLD:
            col, label = (0, 220, 255), "IYI"
        else:
            col, label = (0, 100, 255), "PRATIK GEREKIYOR"
        cv2.putText(canvas, f"GENEL: %{comp_score} {label}",
                    (50, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)

        if self.final_feedback:
            y += 80
            words = self.final_feedback.split()
            line = ""
            lines = []
            for w in words:
                if len(line) + len(w) > 38:
                    lines.append(line)
                    line = w
                else:
                    line = line + " " + w if line else w
            if line:
                lines.append(line)
            for ln in lines[:3]:
                cv2.putText(canvas, ln, (50, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 100), 1)
                y += 28

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
        cv2.putText(canvas, "SIZIN DENEMENIZ",
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
        BAR_H = 160
        cv2.rectangle(combined, (0, h - BAR_H), (TOTAL_WIDTH, h), (0, 0, 0), -1)

        if self.phase == PHASE_FINAL:
            text = "R: Yeni cumle      N: Yeni tema      Q: Cikis"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            tx = (TOTAL_WIDTH - tw) // 2
            cv2.putText(combined, text, (tx, h - BAR_H + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            return combined

        sess = self._current_session()
        if sess is None:
            return combined

        cv2.putText(combined, f"CUMLE: {self.raw_sentence}",
                    (20, h - BAR_H + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2)
        cv2.putText(combined, f"Adim {self.current_word_idx + 1}/{len(self.sessions)}: "
                              f"{sess.word.upper()}  "
                              f"(Deneme {sess.attempts_used + 1}/{MAX_ATTEMPTS_PER_WORD})",
                    (20, h - BAR_H + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if sess.best_score > 0:
            cv2.putText(combined, f"En iyi: %{int(sess.best_score)}",
                        (20, h - BAR_H + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if self.state == STATE_RESULT and self.last_score is not None:
            score = int(self.last_score)
            if score >= EXCELLENT_THRESHOLD:
                color, label = (0, 255, 0), "MUKEMMEL!"
            elif score >= GOOD_THRESHOLD:
                color, label = (0, 220, 255), "IYI"
            else:
                color, label = (0, 100, 255), "TEKRAR DENE"
            score_text = f"%{score}  {label}"
            (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            tx = (TOTAL_WIDTH - tw) // 2
            cv2.putText(combined, score_text, (tx, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
            cv2.putText(combined, score_text, (tx, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            if sess.attempts_used >= MAX_ATTEMPTS_PER_WORD:
                hint = "Bu kelime tamam. N: sonraki kelime"
            elif score >= EXCELLENT_THRESHOLD:
                hint = "Cok iyi! N: sonraki kelime, R: tekrar dene"
            else:
                hint = "Tekrar deneyebilirsin (SPACE) veya N: sonraki kelime"
            cv2.putText(combined, hint, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        elif self.state == STATE_SCORING:
            text = "Hesaplaniyor..."
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            tx = (TOTAL_WIDTH - tw) // 2
            cv2.putText(combined, text, (tx, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        elif self.state == STATE_IDLE:
            # v3: B replaces P (matching study_autsl_tutor v6)
            text = "SPACE: Basla     N: Sonraki kelime     B: Onceki kelime     Q: Cikis"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx = (TOTAL_WIDTH - tw) // 2
            cv2.putText(combined, text, (tx, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return combined

    def _start_countdown(self):
        if self.state in (STATE_COUNTDOWN, STATE_RECORDING, STATE_SCORING):
            return
        sess = self._current_session()
        if sess is None or sess.is_complete:
            print("[App] Word complete or no session")
            return
        self.state = STATE_COUNTDOWN
        self.countdown_start = time.time()
        self.recording_buf.clear()
        self._recording_announced = False
        print(f"[App] Countdown for {sess.word} (attempt {sess.attempts_used + 1}/{MAX_ATTEMPTS_PER_WORD})")

    def _check_countdown(self):
        if self.state == STATE_COUNTDOWN:
            elapsed = time.time() - self.countdown_start
            if elapsed >= COUNTDOWN_SEC:
                self.state = STATE_RECORDING
                self.recording_start = time.time()
                if not self._recording_announced:
                    sess = self._current_session()
                    print(f"[App] Recording {sess.word} ({self.record_seconds}s)")
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
        print(f"[App] Recording complete: {len(self.recording_buf)} frames")
        self.state = STATE_SCORING
        self._score_in_background()

    def _score_in_background(self):
        sess = self._current_session()
        if sess is None:
            return
        ref = self.refs.get(sess.class_idx)
        if ref is None:
            self.last_score    = 0
            self.state         = STATE_RESULT
            sess.add_attempt(0, 999)
            return

        user_seq = np.stack(self.recording_buf)

        def worker():
            print(f"[App] Scoring {sess.word}...")
            score, normalized = self.scorer.best_against(user_seq, ref['all'])
            with self._lock:
                self.last_score      = score
                self.last_norm_dist  = normalized
                sess.add_attempt(score, normalized)
                self.state           = STATE_RESULT
                self._is_scoring     = False
            print(f"[App] {sess.word}: %{score:.0f} (best: %{sess.best_score:.0f})")

        self._is_scoring = True
        threading.Thread(target=worker, daemon=True).start()

    def _next_word(self):
        if self.phase == PHASE_FINAL:
            return
        if self.current_word_idx + 1 < len(self.sessions):
            self.current_word_idx += 1
            self._reset_attempt_state()
            sess = self._current_session()
            print(f"[App] Moving to word {self.current_word_idx + 1}: {sess.word}")
        else:
            self._finalize_sentence()

    def _prev_word(self):
        """v3: B key navigates back to previous word (if not at first)."""
        if self.phase == PHASE_FINAL:
            return
        if self.current_word_idx > 0:
            self.current_word_idx -= 1
            self._reset_attempt_state()
            sess = self._current_session()
            print(f"[App] Back to word {self.current_word_idx + 1}: {sess.word}")

    def _finalize_sentence(self):
        scored_sessions = [s for s in self.sessions if s.attempts_used > 0]
        if scored_sessions:
            self.composite_score = sum(s.best_score for s in scored_sessions) / len(scored_sessions)
        else:
            self.composite_score = 0.0

        print(f"\n[App] Sentence complete!")
        print(f"[App] Composite score: %{self.composite_score:.1f}")
        for s in self.sessions:
            print(f"[App]   {s.word:15s}: %{s.best_score:5.1f} ({s.attempts_used} attempts)")

        def feedback_worker():
            try:
                comp = int(self.composite_score)
                if comp >= EXCELLENT_THRESHOLD:
                    glosses = ["mukemmel", "cumle", "iyi"]
                elif comp >= GOOD_THRESHOLD:
                    glosses = ["iyi", "biraz", "pratik"]
                else:
                    glosses = ["pratik", "tekrar", "denemek"]
                fb, _ = self.translator.translate(glosses, max_tokens=30)
                with self._lock:
                    self.final_feedback = fb or "Cumleyi tamamladin!"
            except Exception as e:
                print(f"[App] Feedback error: {e}")
                with self._lock:
                    self.final_feedback = "Cumleyi tamamladin! Pratik yapmaya devam et."

        threading.Thread(target=feedback_worker, daemon=True).start()
        self.phase = PHASE_FINAL

    def _reset_sentence(self):
        for s in self.sessions:
            s.attempts.clear()
            s.best_score = 0.0
            s.best_dist = float('inf')
        self.current_word_idx = 0
        self._reset_attempt_state()
        self.composite_score = 0.0
        self.final_feedback = ""
        self.phase = PHASE_PRACTICING

    def _ask_theme_console(self):
        print("\n" + "=" * 50)
        print("Yeni cumle icin tema gir (ornek: yemek, aile, okul):")
        try:
            theme = input("Tema: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not theme:
            theme = "selamlama"
        return theme

    def run(self):
        print("\n" + "=" * 60)
        print("TID Sentence Tutor (Phase 2) v3")
        print("=" * 60)
        print("Workflow:")
        print("  1. Tema gir (yemek, aile, okul, vs.)")
        print("  2. LLM Turkce cumle uretir")
        print(f"  3. Her kelime icin {MAX_ATTEMPTS_PER_WORD} deneme hakkin var")
        print("  4. Cumle bittiginde toplam skor gosterilir")
        print()
        print("Keys (in window):")
        print("  SPACE  : Start practicing current word")
        print("  N      : Next word (or finish sentence)")
        print("  B      : Back (previous word)")
        print("  R      : Reset (retry from word 1)")
        print("  Q      : Quit")
        print("=" * 60 + "\n")

        if self.theme:
            print(f"[App] Using starting theme: {self.theme}")
            self._setup_sentence(self.theme)
        else:
            theme = self._ask_theme_console()
            if not theme:
                print("[App] No theme provided, exiting.")
                return
            self._setup_sentence(theme)

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

                left = self._draw_left_panel()
                right = self._draw_right_panel(frame, results)
                combined = np.hstack([left, right])
                combined = self._draw_bottom_overlay(combined)

                cv2.imshow(self.window_name, combined)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if self.phase == PHASE_PRACTICING:
                        self._start_countdown()
                elif key == ord('n'):
                    if self.phase == PHASE_PRACTICING:
                        self._next_word()
                    elif self.phase == PHASE_FINAL:
                        cv2.destroyWindow(self.window_name)
                        new_theme = self._ask_theme_console()
                        if new_theme is None:
                            break
                        self._setup_sentence(new_theme)
                        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(self.window_name, TOTAL_WIDTH, TOTAL_HEIGHT)
                elif key == ord('b'):  # v3: NEW - back to previous word
                    if self.phase == PHASE_PRACTICING:
                        self._prev_word()
                elif key == ord('r'):
                    if self.phase == PHASE_PRACTICING:
                        self._reset_sentence()
                    elif self.phase == PHASE_FINAL:
                        self._reset_sentence()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n[App] Session ended.")


def main():
    parser = argparse.ArgumentParser(description='TID Sentence Tutor Phase 2 v3')
    parser.add_argument('--class_map', required=True)
    parser.add_argument('--reference_dir', default='reference_landmarks')
    parser.add_argument('--llm_model', required=True)
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--theme', default=None)
    parser.add_argument('--record_seconds', type=float, default=DEFAULT_RECORD_TIME)
    parser.add_argument('--good_dist', type=float, default=DEFAULT_GOOD_DIST)
    parser.add_argument('--bad_dist', type=float, default=DEFAULT_BAD_DIST)
    args = parser.parse_args()

    app = SentenceTutorApp(
        class_map_path = args.class_map,
        reference_dir  = args.reference_dir,
        llm_model      = args.llm_model,
        camera_idx     = args.camera,
        theme          = args.theme,
        record_seconds = args.record_seconds,
        good_dist      = args.good_dist,
        bad_dist       = args.bad_dist,
    )
    app.run()


if __name__ == "__main__":
    main()
