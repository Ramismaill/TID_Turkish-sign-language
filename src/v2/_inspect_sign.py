"""Quick diagnostic: is the stored sign data actually moving / one-or-two-handed /
distinct between signs? Separates DATA quality from RETARGETING quality."""
import sys, numpy as np
sys.path.insert(0, r"C:\sign_language\src\v2")
from text_to_sign import TextToSignPipeline

p = TextToSignPipeline()

def inspect(word):
    arr = p._load_landmarks(p._resolve(word)[0] if hasattr(p, "_resolve") else word)
    # fallback: use translate path
    if arr is None:
        arr = p._load_landmarks(word)
    if arr is None:
        print(f"  {word}: NO DATA"); return None
    arr = np.asarray(arr)                      # (64, 225)
    pose  = arr[:, 0:99].reshape(64, 33, 3)
    left  = arr[:, 99:162].reshape(64, 21, 3)
    right = arr[:, 162:225].reshape(64, 21, 3)
    # motion = how much each block moves across the 64 frames
    def motion(block): return float(np.mean(np.std(block, axis=0)))
    # "active" = not collapsed/degenerate (variance within a frame)
    def active(block): return float(np.mean(np.std(block, axis=1)))
    lw = pose[:, 15, :]; rw = pose[:, 16, :]   # wrists
    print(f"  {word:16s} shape={arr.shape}")
    print(f"    frame-to-frame motion : pose={motion(pose):.4f}  L={motion(left):.4f}  R={motion(right):.4f}")
    print(f"    within-frame spread   : L={active(left):.4f}  R={active(right):.4f}  (~0 => collapsed/degenerate hand)")
    print(f"    L-wrist Y range       : {lw[:,1].min():.3f}..{lw[:,1].max():.3f}   R-wrist Y range: {rw[:,1].min():.3f}..{rw[:,1].max():.3f}")
    return arr

print("=== Per-sign diagnostic ===")
a = inspect("merhaba")
b = inspect("seviyorum")
c = inspect("evet")

if a is not None and b is not None:
    diff = float(np.mean(np.abs(a - b)))
    print(f"\n  merhaba vs seviyorum mean abs diff = {diff:.4f}   (~0 => signs are identical = BUG)")
