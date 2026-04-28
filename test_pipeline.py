"""
Pipeline Testi - TMS-Net + SML + ST-GCN okul PC'sinde calisiyor mu?
ZIP acildiktan sonra C:\AUTSL_final\ klasorunde calistir.
"""
import os
import sys
import json
from pathlib import Path

print("=" * 60)
print("  Pipeline Testi")
print("=" * 60)

errors = []

# --- 1. Dosya kontrolu ---
print("\n[1] Dosya yapisi:")
checks = [
    ("src/tmsnet_model.py", "TMS-Net model"),
    ("src/sml_model.py", "SML model"),
    ("src/stgcn_model.py", "ST-GCN model"),
    ("src/graph.py", "Graph"),
    ("src/augmentations.py", "Augmentations"),
    ("src/inference_tmsnet.py", "TMS-Net inference"),
    ("src/ensemble_tmsnet_sml.py", "Ensemble"),
    ("checkpoints/best.pth", "TMS-Net ckpt"),
    ("checkpoints/best_sml.pt", "SML ckpt"),
    ("checkpoints/best_stgcn.pt", "ST-GCN ckpt"),
    ("class_map.json", "Class map (226 Turkce sign)"),
]

for path, desc in checks:
    p = Path(path)
    if p.exists():
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            if size_kb > 1000:
                print(f"    [OK] {path}  ({size_kb/1024:.1f} MB) - {desc}")
            else:
                print(f"    [OK] {path}  ({size_kb:.1f} KB) - {desc}")
        else:
            print(f"    [OK] {path} - {desc}")
    else:
        errors.append(f"Eksik: {path}")
        print(f"    [X]  {path} - {desc}")

# --- 2. class_map ---
print("\n[2] class_map.json:")
try:
    with open("class_map.json", "r", encoding="utf-8") as f:
        cm = json.load(f)
    print(f"    [OK] {len(cm)} sinif")
    print(f"    Ilk 3:  0='{cm.get('0')}', 1='{cm.get('1')}', 2='{cm.get('2')}'")
    print(f"    Son 3:  223='{cm.get('223')}', 224='{cm.get('224')}', 225='{cm.get('225')}'")
    if len(cm) != 226:
        errors.append(f"226 sinif beklenirken {len(cm)} bulundu")
except Exception as e:
    errors.append(f"class_map: {e}")

# --- 3. Checkpoint yukleme ---
print("\n[3] Checkpoint yukleme:")
try:
    import torch
    for name, path in [("TMS-Net", "checkpoints/best.pth"),
                       ("SML", "checkpoints/best_sml.pt"),
                       ("ST-GCN", "checkpoints/best_stgcn.pt")]:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                keys = list(ckpt.keys())[:3]
                print(f"    [OK] {name}: keys={keys}")
            else:
                print(f"    [OK] {name}: yuklu")
        else:
            print(f"    [X]  {name} checkpoint eksik")
except Exception as e:
    errors.append(f"Checkpoint: {e}")

# --- 4. MediaPipe ---
print("\n[4] MediaPipe:")
try:
    import mediapipe as mp
    h = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
    print(f"    [OK] Holistic yuklendi")
    h.close()
except Exception as e:
    errors.append(f"MediaPipe: {e}")

# --- 5. Reference landmarks ---
print("\n[5] Referans landmark'lar:")
ref_dir = Path("reference_landmarks")
if ref_dir.exists():
    npy_files = list(ref_dir.glob("*.npy"))
    print(f"    [OK] {len(npy_files)} .npy dosyasi")
    if npy_files:
        # Sinif numaralari
        import re
        classes = set()
        for f in npy_files:
            m = re.match(r"cls(\d+)_", f.name)
            if m:
                classes.add(int(m.group(1)))
        print(f"    {len(classes)} farkli sinif kapsaniyor")
        # Ornek yukle
        import numpy as np
        arr = np.load(npy_files[0])
        print(f"    Ornek sekil: {arr.shape}, dtype: {arr.dtype}")
else:
    print(f"    [UYARI] reference_landmarks/ yok - DTW icin gerekli")

# --- 6. Webcam ---
print("\n[6] Webcam:")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"    [OK] {w}x{h}")
        else:
            print(f"    [UYARI] Acildi ama frame yok")
        cap.release()
    else:
        print(f"    [UYARI] Webcam yok (demo icin gerekli)")
except Exception as e:
    print(f"    [UYARI] {e}")

# --- OZET ---
print("\n" + "=" * 60)
if errors:
    print(f"  [X] {len(errors)} HATA:")
    for e in errors: print(f"      - {e}")
    print("\n  ZIP icerigini kontrol et.")
else:
    print("  [OK] PIPELINE HAZIR!")
    print("     Sonraki adim: Faz 1 - surekli tanima kodu")
print("=" * 60)

sys.exit(1 if errors else 0)
