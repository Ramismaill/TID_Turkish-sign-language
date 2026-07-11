# TID — Day 4 Locked Decisions (K1–K5)

> **Status:** LOCKED — two-Claude alignment reached (Muhammet's + Ram's sessions), 2 Haziran 2026.
> **Supersedes:** the visibility-injection / `solve()`-based drafts in `Ram_Claude_Context_Day4.md` §5 and `tid_pipeline.txt`.
> **Source of truth:** verified against installed `kalidokit@1.1.5` (`node_modules/kalidokit/dist/`).
> Authors: Ram İsmail & Muhammet Ay.

---

## Locked decisions

| # | Decision | Result |
|---|---|---|
| **K1** | solve() vs bypass | **Bypass:** `Pose.calcArms` + `Hand.solve` directly. + wrist-ownership rule. |
| **K2** | calcHips | **Skip** — torso/spine fully static (sign = upper body, no drift). |
| **K3** | np.mean vs single-ref | **Single-reference** (`arrays[0]`). Applied + tested ✓. |
| **K4** | VRoid avatar | **One avatar**, today, whoever is faster. Engineering reqs below. |
| **K5** | Day 4 order | pose-only POC → hand → full sign, with euler-log instrumentation. |

---

## Why bypass `solve()` (K1) — source-verified

- `calcArms` (`kalidokit.es.js:587-627`) computes arm rotations from **relative landmark vectors** (`findRotation`/`angleBetween3DCoords`, lines 589-604) → translation + scale invariant → shoulder-centering does NOT corrupt angles.
- `calcArms` calls `rigArm` **internally** (lines 606-607) → returned `UpperArm/LowerArm/Hand` are already human-limit clamped.
- The off-screen/visibility freeze (`es.js:838-849`) lives in `solve()`, **after** calcArms.

**Consequences of the bypass:**
1. Off-screen freeze gone.
2. **`visibility:1.0` injection UNNECESESARY** — only `solve()`'s guard reads visibility. (calcArms / Hand.solve do not.)
3. `lm3d[15].y > 0.1` false-freeze risk + coordinate-system anxiety → gone for arms/hands.

## Wrist-ownership rule (K1) — prevents double rotation / seam

`calcArms` returns `Hand.l/r` (wrist orient from pose lm[15-20]) AND `Hand.solve` returns wrist + 15 finger joints. Both writing the wrist bone = double rotation.

**Rule (canonical Kalidokit demo pattern):**
- `UpperArm`, `LowerArm` (l/r) ← `calcArms`
- Wrist **bend (x, y)** + all fingers ← `Hand.solve`
- Wrist **twist (z)** ← `calcArms.Hand.z` (forearm roll)
```js
rigRotation(vrm, 'leftHand', { x: lh.LeftWrist.x, y: lh.LeftWrist.y, z: arms.Hand.l.z });
```
→ Do NOT write the full `calcArms.Hand` to the wrist bone; only borrow its `.z`. Verify in POC euler-log; drop z if pose-z is noisy on stored data.

## K3 — single-reference (applied)

`text_to_sign.py:_load_landmarks` now returns `arrays[0]` (first available reference), not the mean. Reason: AUTSL's 3 refs are temporally unaligned; frame-frame averaging blends sign phases → anatomically-impossible "ghost" handshape (worst for fingers). A single real recording = coherent, readable handshape.

⚠️ **Do NOT `git revert b5732a8`** to "undo the mean" — that restores `np.vstack` → `(192,225)` bug. The correct change is the one-line `mean → arrays[0]` (done). Test: 15/15 still `(64,225)` ✓.

Future: per-word **medoid** (ref closest to the other two) if a word's first ref is a bad take.

## K4 — VRoid avatar requirements

- Full finger rig: 5 fingers × 3 joints (VRoid default has this)
- **Confirm VRM version** (1.0 vs 0.x) → decides thumb bone naming code path:
  - VRM 1.0: `thumbMetacarpal / thumbProximal / thumbDistal`
  - Kalidokit output: `ThumbProximal / ThumbIntermediate / ThumbDistal`
- Keep poly count modest (30 FPS target on RTX 4060 Laptop)
- Style: realistic / neutral

---

## Locked Day 4 architecture (data flow)

```
flat[225]
  → parseStoredFrame:
       pose      = flat[0:99]    → 33 × {x,y,z}   (FULL array — calcArms indexes lm[11..20])
       leftHand  = flat[99:162]  → 21 × {x,y,z}
       rightHand = flat[162:225] → 21 × {x,y,z}
       (+ degenerate-hand flag · visibility NOT needed on this path)
  → arms = Kalidokit.Pose.calcArms(pose)          // clamped, off-screen-free, relative-vector
  → lh   = degenerate(leftHand)  ? lastLH : Kalidokit.Hand.solve(leftHand,  'Left')
  → rh   = degenerate(rightHand) ? lastRH : Kalidokit.Hand.solve(rightHand, 'Right')
  → DO NOT call calcHips — torso/spine rest (static)
  → applyRig(vrm):
       UpperArm.l/r ← arms ; LowerArm.l/r ← arms
       wrist bend(x,y)+fingers ← lh/rh ; wrist twist(z) ← arms.Hand.z
       (thumb bone rename per VRM version)
```

## Fail-fast sequence (final)

```
0. ✅ Source verified · two sessions aligned · K3 applied+tested
1. VRoid avatar in place (BLOCKER — can't test fingers without finger bones); confirm VRM version
2. parseStoredFrame (225→pose/L/R) + degenerate flag
3. POSE-ONLY: calcArms only → do arms move on the real avatar? (isolate one variable)
4. Add hands: Hand.solve + degenerate skip + wrist-ownership rule
5. One sign (STATIC, distinct handshape) → euler-log per bone → full sign
6. 🚦 Day 5 GO/NO-GO (focus: finger readability)
```

## Implementation notes for Day 4

- Base the `rigRotation`/`rigPosition` helpers on the **official Kalidokit VRM demo** (bone-local space + dampener/lerp); validate the `tid_pipeline.txt` AI-draft against it — do not write axis mapping from scratch.
- Instrument: log every bone's computed euler (console/JSON) in the POC. Tune the Kalidokit-euler → VRM-bone axis/sign mapping + per-bone dampeners **from the log**, not by eyeball. This is the main remaining effort to Day 5.
- IEEE/reproducibility: pin `kalidokit@1.1.5` + the `@pixiv/three-vrm` version. Kalidokit is no longer actively maintained.

---

*Next action (both sessions): VRoid avatar → pose-only POC (`calcArms`) → confirm arms move.*
