"""Ram's Claude check: is finger DATA real, or degenerate every frame?
Decides data-vs-code for the flat-finger blocker."""
import sys, numpy as np
sys.path.insert(0, r"C:\sign_language\src\v2")
from text_to_sign import TextToSignPipeline

p = TextToSignPipeline()

for word in ["merhaba", "seviyorum", "evet", "iyi", "anne"]:
    arr = p._load_landmarks(word)
    if arr is None:
        print(f"{word:12s}: NO DATA"); continue
    arr = np.asarray(arr)
    left  = arr[:, 99:162].reshape(64, 21, 3)
    right = arr[:, 162:225].reshape(64, 21, 3)
    lv = left.var(axis=(1, 2))      # per-frame variance (64,)
    rv = right.var(axis=(1, 2))
    l_deg = int((lv < 1e-6).sum())
    r_deg = int((rv < 1e-6).sum())
    print(f"{word:12s}: L var[min..max]={lv.min():.5f}..{lv.max():.5f} deg={l_deg}/64 | "
          f"R var[min..max]={rv.min():.5f}..{rv.max():.5f} deg={r_deg}/64")
print("\n-> var > 0 and deg low  => DATA is real, fingers are a CODE/axis issue (fixed: axis='y')")
print("-> var ~0 / deg ~64     => DATA is dead, no finger info (would need re-extract / Plan B)")
