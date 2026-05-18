"""
src/v2/text_to_sign.py — Turkish text → sign landmark pipeline

Loads sign_dictionary.json, resolves Turkish word variants and suffixes,
loads corresponding .npy landmark arrays from reference_landmarks/.

Usage (smoke test):
    cd C:\\sign_language
    python src/v2/text_to_sign.py

Author: Ram Ismail, Muhammet Ay
Date: 2026-05-18
"""

import json
import re
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np

# ── Path resolution ────────────────────────────────────────────────────────────
# Works whether called as:
#   python src/v2/text_to_sign.py          (from project root)
#   python text_to_sign.py                 (from src/v2/)
SCRIPT_DIR   = Path(__file__).resolve().parent          # src/v2/
PROJECT_ROOT = SCRIPT_DIR.parent.parent                 # C:\sign_language\

DEFAULT_DICT_PATH      = SCRIPT_DIR / "sign_dictionary.json"
DEFAULT_REFERENCE_DIR  = PROJECT_ROOT / "reference_landmarks"

# ── Turkish suffix stripping (prototype-level) ─────────────────────────────────
# Covers the most common agglutinative suffixes for the 15-word vocab.
# Day 9+ will replace this with a proper morphological analyser.
_TURKISH_SUFFIXES = [
    # Accusative / dative / locative / ablative / genitive
    "dan", "den", "da", "de", "ya", "ye", "a", "e",
    # Possessive
    "ım", "im", "um", "üm", "m",
    "ın", "in", "un", "ün", "n",
    # Plural
    "lar", "ler",
    # Verb personal endings (simple present)
    "ıyor", "iyor", "uyor", "üyor",
    "yor",
    # Future
    "acak", "ecek",
    # Past
    "dı", "di", "du", "dü", "tı", "ti", "tu", "tü",
]

# Sort longest-first so we strip the most specific suffix possible
_TURKISH_SUFFIXES = sorted(_TURKISH_SUFFIXES, key=len, reverse=True)


def _strip_suffix(word: str) -> str:
    """Return the stem of a Turkish word by stripping one known suffix."""
    for suffix in _TURKISH_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[: len(word) - len(suffix)]
    return word


def _normalise(text: str) -> str:
    """Lowercase + strip punctuation."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


# ── Pipeline ───────────────────────────────────────────────────────────────────

class TextToSignPipeline:
    """
    Translates a Turkish sentence into a sequence of (word, landmarks) pairs.

    landmarks shape: (n_frames, 225)  — 75 MediaPipe Holistic keypoints × 3 coords
    Unknown words produce (word, None).
    """

    def __init__(
        self,
        dictionary_path: str | Path = DEFAULT_DICT_PATH,
        reference_dir:   str | Path = DEFAULT_REFERENCE_DIR,
    ):
        self.reference_dir = Path(reference_dir)
        self.dictionary: dict = self._load_dict(Path(dictionary_path))

        # Build reverse variant map:  "sağol" → "teşekkür ederim"
        self._variant_map: dict[str, str] = {}
        for canonical, entry in self.dictionary.items():
            if canonical.startswith("_"):
                continue
            for v in entry.get("variants", []):
                self._variant_map[_normalise(v)] = canonical

        print(f"[TextToSignPipeline] Loaded {len(self.dictionary) - 1} signs "
              f"from {dictionary_path}")
        print(f"[TextToSignPipeline] Reference dir: {self.reference_dir}")

    # ── Public API ──────────────────────────────────────────────────────────────

    def translate(self, text: str) -> list[tuple[str, Optional[np.ndarray]]]:
        """
        Tokenise text and return [(canonical_word, landmarks_or_None), ...].
        """
        tokens = _normalise(text).split()
        results = []
        i = 0
        while i < len(tokens):
            # Try 2-word phrase first (e.g. "teşekkür ederim")
            if i + 1 < len(tokens):
                phrase = tokens[i] + " " + tokens[i + 1]
                canonical = self._lookup_canonical(phrase)
                if canonical is not None:
                    landmarks = self._load_landmarks(canonical)
                    results.append((canonical, landmarks))
                    i += 2
                    continue
            # Single word
            canonical = self._lookup_canonical(tokens[i])
            landmarks = self._load_landmarks(canonical) if canonical else None
            results.append((canonical or tokens[i], landmarks))
            i += 1
        return results

    # ── Internals ───────────────────────────────────────────────────────────────

    def _load_dict(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Dictionary not found: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _lookup_canonical(self, word: str) -> Optional[str]:
        """
        Resolve a word to a canonical dictionary key.
        Order: exact match → variant map → suffix-stripped exact → suffix-stripped variant
        """
        w = _normalise(word)

        # 1. Exact match
        if w in self.dictionary:
            return w

        # 2. Variant map
        if w in self._variant_map:
            return self._variant_map[w]

        # 3. Suffix strip then exact
        stem = _strip_suffix(w)
        if stem != w:
            if stem in self.dictionary:
                return stem
            if stem in self._variant_map:
                return self._variant_map[stem]

        return None

    def _load_landmarks(self, canonical: str) -> Optional[np.ndarray]:
        """
        Load reference .npy files for a canonical word and return their mean.

        Each file is (64, 225). We average across all available references
        so the output is always (64, 225) — never (192, 225).
        Averaging reduces per-recording noise while keeping a single clean sequence.

        Returns shape (64, 225) float32, or None if no files found.
        """
        entry = self.dictionary.get(canonical)
        if entry is None:
            return None

        arrays = []
        for rel_path in entry.get("reference_files", []):
            full_path = self.reference_dir.parent / rel_path  # PROJECT_ROOT / rel_path
            if full_path.exists():
                arr = np.load(str(full_path)).astype(np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                arrays.append(arr)
            else:
                print(f"  [WARNING] Missing: {full_path}")

        if not arrays:
            return None

        # All refs same shape → mean across recordings, output stays (64, 225)
        stacked = np.stack(arrays, axis=0)   # (n_refs, 64, 225)
        return stacked.mean(axis=0)           # (64, 225)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = TextToSignPipeline()

    test_sentences = [
        "merhaba",
        "Ben seni seviyorum",
        "teşekkür ederim iyi",
        "okula gidiyorum",
        "sağol hayır evet",
        "anne evde yemek",
    ]

    print("\n" + "=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    total_found = 0
    total_missing = 0

    for sentence in test_sentences:
        print(f"\nInput : {sentence!r}")
        results = pipeline.translate(sentence)
        for word, lm in results:
            if lm is not None:
                print(f"  ✓  {word:20s}  shape={lm.shape}")
                total_found += 1
            else:
                print(f"  ✗  {word:20s}  (not in vocab)")
                total_missing += 1

    print("\n" + "=" * 60)
    print(f"Found: {total_found}  |  Missing: {total_missing}")
    if total_missing == 0:
        print("ALL SIGNS LOADED ✓")
    else:
        print("Some words not in vocab — expected for out-of-dict tokens.")
    print("=" * 60)
