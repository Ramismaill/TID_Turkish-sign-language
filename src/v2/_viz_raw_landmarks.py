"""Render the RAW stored landmarks as a skeleton GIF — bypasses ALL avatar/retargeting.
Answers: is the SOURCE data a recognizable sign, or is our retargeting losing it?"""
import sys, numpy as np
sys.path.insert(0, r"C:\sign_language\src\v2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from text_to_sign import TextToSignPipeline

WORD = sys.argv[1] if len(sys.argv) > 1 else "merhaba"
p = TextToSignPipeline()
arr = np.asarray(p._load_landmarks(WORD))          # (64, 225)
pose  = arr[:, 0:99].reshape(64, 33, 3)
left  = arr[:, 99:162].reshape(64, 21, 3)
right = arr[:, 162:225].reshape(64, 21, 3)

POSE_CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(0,11),(0,12)]
HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]

fig, ax = plt.subplots(figsize=(5, 6))

def draw(f):
    ax.clear()
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(0.6, -0.8)   # y inverted (image y-down) -> head up
    ax.set_title(f"RAW source: '{WORD}'  frame {f}/63", fontsize=10)
    ax.set_aspect("equal"); ax.axis("off")
    P = pose[f]
    for a, b in POSE_CONN:
        ax.plot([P[a,0],P[b,0]],[P[a,1],P[b,1]], "-", color="#3355aa", lw=2)
    ax.scatter(P[:,0], P[:,1], s=8, color="#3355aa")
    for H, col in [(left[f], "#cc3333"), (right[f], "#33aa33")]:
        for a, b in HAND_CONN:
            ax.plot([H[a,0],H[b,0]],[H[a,1],H[b,1]], "-", color=col, lw=1)
        ax.scatter(H[:,0], H[:,1], s=6, color=col)

anim = FuncAnimation(fig, draw, frames=64, interval=50)
out = rf"C:\Users\Muhammet\Downloads\raw_{WORD.replace(' ','_')}.gif"
anim.save(out, writer=PillowWriter(fps=20))
print("saved:", out)
print(f"L-wrist(15) y range: {pose[:,15,1].min():.3f}..{pose[:,15,1].max():.3f}  (0=shoulder, negative=above)")
print(f"R-wrist(16) y range: {pose[:,16,1].min():.3f}..{pose[:,16,1].max():.3f}")
print(f"nose(0) y: {pose[:,0,1].mean():.3f}   (head reference)")
