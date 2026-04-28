"""
Kurulum Dogrulama - Okul PC
"""
import sys

print("=" * 60)
print("  TID Final - Kurulum Dogrulama")
print("=" * 60)

errors, warnings = [], []

print(f"\n[1] Python: {sys.version.split()[0]}")
if sys.version_info < (3, 9) or sys.version_info >= (3, 12):
    warnings.append(f"Python 3.10 tavsiye edilir")

print("\n[2] PyTorch + CUDA:")
try:
    import torch
    print(f"    PyTorch: {torch.__version__}")
    print(f"    CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA version: {torch.version.cuda}")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    VRAM: {vram:.1f} GB")
        x = torch.randn(100, 100).cuda()
        y = x @ x
        print(f"    [OK] CUDA hesaplama calisiyor")
    else:
        errors.append("CUDA yok - PyTorch CPU-only olabilir")
except ImportError as e:
    errors.append(f"PyTorch: {e}")

print("\n[3] Ana paketler:")
for pkg, mod in [("numpy", "numpy"), ("mediapipe", "mediapipe"),
                  ("opencv-python", "cv2"), ("pandas", "pandas"),
                  ("matplotlib", "matplotlib"), ("scipy", "scipy"),
                  ("einops", "einops"), ("tqdm", "tqdm")]:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"    [OK] {pkg}: {ver}")
    except ImportError:
        errors.append(f"Paket eksik: {pkg}")

print("\n[4] LLM paketleri:")
for pkg, mod in [("llama-cpp-python", "llama_cpp"),
                  ("gradio", "gradio"),
                  ("sentencepiece", "sentencepiece")]:
    try:
        m = __import__(mod)
        print(f"    [OK] {pkg}")
    except ImportError:
        warnings.append(f"{pkg} eksik (download_llm.bat sonrasi kurulur)")

print("\n[5] Degerlendirme paketleri:")
for pkg, mod in [("sacrebleu", "sacrebleu"), ("dtw-python", "dtw")]:
    try:
        m = __import__(mod)
        print(f"    [OK] {pkg}")
    except ImportError:
        warnings.append(f"{pkg} eksik")

print("\n" + "=" * 60)
if errors:
    print(f"  [X] {len(errors)} HATA:")
    for e in errors: print(f"      - {e}")
if warnings:
    print(f"  [!] {len(warnings)} UYARI:")
    for w in warnings: print(f"      - {w}")
if not errors and not warnings:
    print("  [OK] HER SEY HAZIR!")
elif not errors:
    print("  [OK] Ana kurulum tamam")
print("=" * 60)

sys.exit(1 if errors else 0)
