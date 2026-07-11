"""
src/v2/test_translate_full.py — Full 15-word /translate + .npy layout verification

Day 3 task §4.2: call all 15 vocabulary words individually, confirm each loads
with shape (64, 225). Also inspects the raw .npy files to verify the landmark
layout (75 keypoints × 3) before Day 4 Kalidokit retargeting is written.

Usage:
    cd C:\\sign_language
    conda run -n isaret_dili python src/v2/test_translate_full.py

Author: Ram Ismail
Date: 2026-06-02
"""

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from text_to_sign import TextToSignPipeline, DEFAULT_DICT_PATH, DEFAULT_REFERENCE_DIR

EXPECTED_SHAPE = (64, 225)


def test_all_canonical_words(pipeline: TextToSignPipeline) -> dict:
    """Translate every canonical dictionary key, check shape == (64, 225)."""
    print("=" * 64)
    print("TEST 1 — All 15 canonical words load with shape (64, 225)")
    print("=" * 64)

    results = {"ok": [], "wrong_shape": [], "missing": []}

    for canonical, entry in pipeline.dictionary.items():
        if canonical.startswith("_"):
            continue
        out = pipeline.translate(canonical)
        # translate returns [(word, landmarks)]
        _, lm = out[0]
        if lm is None:
            print(f"  [X] {canonical:18s}  NO LANDMARKS (files missing)")
            results["missing"].append(canonical)
        elif lm.shape != EXPECTED_SHAPE:
            print(f"  [!] {canonical:18s}  shape={lm.shape}  EXPECTED {EXPECTED_SHAPE}")
            results["wrong_shape"].append((canonical, lm.shape))
        else:
            print(f"  [OK] {canonical:18s} shape={lm.shape}")
            results["ok"].append(canonical)

    return results


def test_demo_sentences(pipeline: TextToSignPipeline) -> None:
    """Run the 6 fixed demo sentences through the full pipeline."""
    print("\n" + "=" * 64)
    print("TEST 2 — 6 fixed demo sentences")
    print("=" * 64)

    demo = [
        "Merhaba",
        "Teşekkür ederim",
        "Ben seni seviyorum",
        "Anne evde yemek",
        "İyi günler",          # 'günler' unknown — expected skip
        "Okula gidiyorum",     # 'okula' suffix-strip, 'gidiyorum' unknown
    ]

    for sentence in demo:
        out = pipeline.translate(sentence)
        rendered = []
        for word, lm in out:
            tag = f"{word}({lm.shape[0]}f)" if lm is not None else f"{word}[?]"
            rendered.append(tag)
        print(f"  '{sentence}'  ->  {', '.join(rendered)}")


def inspect_raw_npy() -> None:
    """
    Inspect a few raw .npy files to verify the actual landmark layout.

    Day 4 Kalidokit retargeting needs to know:
      - total values per frame (should be 225 = 75 keypoints × 3)
      - value range (normalized? pixel space?)
      - whether 75 = 33 pose + 21 left hand + 21 right hand (handoff §12 assumption)
    """
    print("\n" + "=" * 64)
    print("TEST 3 — Raw .npy layout inspection (for Day 4 retargeting)")
    print("=" * 64)

    samples = ["cls173_1.npy", "cls196_1.npy", "cls014_1.npy"]
    ref_dir = Path(DEFAULT_REFERENCE_DIR)

    for name in samples:
        p = ref_dir / name
        if not p.exists():
            print(f"  [X] {name} not found at {p}")
            continue
        arr = np.load(str(p))
        per_frame = arr.shape[-1] if arr.ndim >= 2 else arr.shape[0]
        kp = per_frame / 3
        print(f"\n  {name}")
        print(f"    raw shape : {arr.shape}   dtype={arr.dtype}")
        print(f"    per-frame : {per_frame} values  ->  {kp:.1f} keypoints (×3)")
        print(f"    value range: min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")
        # Layout hypothesis check: 75 kp = 33 pose + 21 + 21
        if abs(kp - 75) < 0.5:
            print(f"    -> 75 keypoints: consistent with [33 pose + 21 Lhand + 21 Rhand] (handoff §12)")
        elif abs(kp - 543) < 0.5:
            print(f"    -> 543 keypoints: FULL holistic incl. 468 face (main.ts:173 comment)")
        else:
            print(f"    -> UNEXPECTED keypoint count — layout must be confirmed manually")


def main() -> int:
    print(f"Dictionary : {DEFAULT_DICT_PATH}")
    print(f"References : {DEFAULT_REFERENCE_DIR}\n")

    pipeline = TextToSignPipeline()

    r1 = test_all_canonical_words(pipeline)
    test_demo_sentences(pipeline)
    inspect_raw_npy()

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  OK (shape 64,225) : {len(r1['ok'])}/15")
    print(f"  Wrong shape       : {len(r1['wrong_shape'])}  {r1['wrong_shape']}")
    print(f"  Missing landmarks : {len(r1['missing'])}  {r1['missing']}")

    if not r1["wrong_shape"] and not r1["missing"]:
        print("\n  ALL 15 WORDS OK — data layer is clean for Day 4.")
        return 0
    else:
        print("\n  ISSUES FOUND — fix before Day 5 gate.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
