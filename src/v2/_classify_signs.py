"""Predict per-sign animation quality from data: which hand is active, how much
motion. Tells us which of the 15 signs will show a clear handshape (GO/NO-GO prep)."""
import sys, numpy as np
sys.path.insert(0, r"C:\sign_language\src\v2")
from text_to_sign import TextToSignPipeline

p = TextToSignPipeline()
VOCAB = ["merhaba", "teşekkür ederim", "evet", "hayır", "iyi", "kötü", "ben",
         "sen", "seviyorum", "yardım", "içmek", "yemek", "okul", "ev", "anne"]

def motion(block):  # mean frame-to-frame std => how much it moves over the 64 frames
    return float(np.mean(np.std(block, axis=0)))

print(f"{'word':16s} {'L-mot':>6s} {'R-mot':>6s}  class")
print("-" * 50)
for w in VOCAB:
    arr = p._load_landmarks(w)
    if arr is None:
        print(f"{w:16s}  NO DATA"); continue
    arr = np.asarray(arr)
    L = motion(arr[:, 99:162].reshape(64, 21, 3))
    R = motion(arr[:, 162:225].reshape(64, 21, 3))
    if max(L, R) < 0.04:
        cls = "LOW motion (weak)"
    elif L > 0.07 and R > 0.07:
        cls = "TWO-HANDED (verify R curl)"
    elif L > R * 1.6:
        cls = "left-dominant"
    elif R > L * 1.6:
        cls = "RIGHT-dominant (verify R curl)"
    else:
        cls = "mixed"
    print(f"{w:16s} {L:6.3f} {R:6.3f}  {cls}")
