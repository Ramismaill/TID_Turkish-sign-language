# TID — Development Journal

> Branch `feature/text-to-sign-v2`. One entry per work session.
> Authors: Ram Ismail, Muhammet Ay.

---

## Day 3 — 2026-06-02 (resume after ~2-week gap)

**Focus:** verify data layer before writing Kalidokit retargeting (Day 4). No avatar yet.

### §4.2 — Full /translate + shape test (PASS)
- New test: `src/v2/test_translate_full.py` (run: `python src/v2/test_translate_full.py` in `isaret_dili` env).
- **15/15** canonical words load with shape **(64, 225)**. No broken `.npy`, no missing files.
- 6 fixed demo sentences behave as expected:
  - `İyi günler` → iyi + günler[unknown]
  - `Okula gidiyorum` → okul (suffix-strip) + gidiyorum[unknown]

### Landmark layout — VERIFIED (was ambiguous)
- Source of truth: `src/v1/extract_landmarks.py:68`
  ```python
  vec = np.concatenate([pose_xyz, left_xyz, right_xyz])  # 33 + 21 + 21 = 75 kp
  ```
- **225 = 75 keypoints × 3.** Order: `[0:99]` pose(33) · `[99:162]` left hand(21) · `[162:225]` right hand(21). **NO face.**
- Confirms handoff §12 Kalidokit slicing: `pose=0:33, leftHand=33:54, rightHand=54:75`. Safe to use.
- Fixed misleading comment in `tid-frontend/src/main.ts` (it claimed a 468-point face block — wrong).

### Normalization — IMPORTANT for Day 4-5
- `extract_landmarks.py:28-37` `normalize_by_shoulder_center`: every frame is shifted so the shoulder
  midpoint (pose lm 11 & 12) is the origin. Values are therefore centered, range ~[-1, 1] (verified:
  cls173/cls196/cls014 min≈-0.99, max≈0.91, mean≈0).
- **Likely OK:** Kalidokit derives joint angles from *relative* landmark vectors → invariant to a global
  shift. So shoulder-centering probably does NOT break Pose.solve / Hand.solve.
- **Gotcha to handle in Day 4 helper:** undetected hands are written as zeros in extraction, which after
  shoulder-centering become `-center` → the whole hand collapses to a single point for that frame. If
  fingers look broken at Day 5, suspect these degenerate frames, NOT Kalidokit. Mitigation: detect a
  collapsed/degenerate hand per frame and hold the last good pose instead of calling Hand.solve.
- Also: extraction stores only (x,y,z) — no `visibility` field. Kalidokit Pose.solve can run without it
  but quality may drop; revisit at Day 5 gate if body is noisy.

### §4.1 — GitHub research (RESOLVED via external deep research — 5 reports)
Live WebSearch was 529-overloaded, so research was run externally (Gemini Deep Research, Sider, Qwen).
5 PDF reports analysed (extracted text in `Downloads/_extracted/`). They CONTRADICTED on the key
coordinate question; the source-citing report (`deep_research_backing`, quotes
`kalidokit/src/PoseSolver/index.ts`) is decisive.

**VERDICT — feeding our stored, shoulder-centered (64,225) data to Kalidokit: FEASIBLE with 3 fixes.**
(NOT "low feasibility" as the pessimistic report claimed; NOT plug-and-play as the optimistic one claimed.)
- Joint/finger ANGLE math is invariant to translation (shoulder-centering) AND uniform scale — Kalidokit
  uses relative landmark vectors + normalized dot products. So our origin shift does NOT corrupt angles. ✓
- BUT Kalidokit has non-invariant gating that breaks our data unless handled:
  1. **VISIBILITY (critical):** source does `(lm[i].visibility ?? 0) < 0.23` → limb flagged off-screen →
     `Arms.UpperArm.r.multiply(0)` (rotation zeroed, avatar frozen in rest pose). Our .npy has NO
     visibility → MUST inject `visibility: 1.0` per landmark in the parse step.
  2. **Absolute off-screen thresholds:** source does `lm3d[15].y > 0.1`, `lm2d[15].y > 0.995` (calibrated
     for [0,1] image space). Our centered ~[-1,1] coords may trip these → false off-screen. Verify
     empirically at Day 5; if it misfires, shift coords toward [0,1] image space before solve.
  3. **Hips position drift:** Hips.position is absolute (assumes metres) → mute/anchor hip translation.
- `runtime:'mediapipe'` expects x,y in [0,1]; `tfjs` divides by imageSize. For stored data pass a mock
  `imageSize:{width,height}` to avoid a missing-video-element error.

**INTERACTION with our Day-3 degenerate-frame finding (original — reports didn't know our extraction
zero-fills undetected hands):** do NOT blindly inject `visibility:1.0` everywhere. For frames where a hand
was undetected (all-zeros pre-centering → collapsed point post-centering), detect it and skip `Hand.solve`
for that hand (hold last good pose), else Kalidokit computes finger angles from identical points → NaN/
garbage fingers.

**Concrete Day-4 code exists** (`tid_pipeline.txt`): production `rigRotation`/`rigPosition`/`applyRig` for
three-vrm v2/v3 + Kalidokit 1.1.x, incl. the VRM-1.0 thumb-bone rename (Kalidokit `ThumbProximal` →
VRM `thumbMetacarpal`), and `parseStoredFrame(225 → pose/L/R)`. **TREAT AS DRAFT — AI-generated, verify
against real Kalidokit 1.1.5 source** (`kalidokit/src/PoseSolver/index.ts`, `HandSolver/index.ts`) before
trusting. One report had a wrong URL (correct: `ZhengdiYu/SignAvatars`, ECCV 2024).

**PRIOR ART / NOVELTY (IEEE Related Work):** no Kalidokit+three-vrm sign-language *playback* project exists
→ novelty confirmed (all 5 reports agree). Closest: three.js forum "Kalidokit pre-prepared list of coord"
(#61291), `kevinjosethomas/sign-language-processing` (issue #5), `europanite/webcam_to_avatar`, Adrien
Lefebvre SL-AR game. Alternatives: SMPL-X (`ZhengdiYu/SignAvatars`, DexAvatar, Neural Sign Actors, SGNify),
HamNoSys/SiGML (JASigning/CWASA — UEA), neural T2S (T2S-GPT, SignLLM, SignGPT).

### Next session — Day 4 (ready, de-risked)
1. `parseStoredFrame(flat225)` → pose[33]/leftHand[21]/rightHand[21] as `{x,y,z,visibility}`, WITH
   degenerate-hand detection (zero-fill → skip Hand.solve / hold last pose).
2. `Pose.solve(pose,pose,{runtime:'mediapipe'})` + `Hand.solve(L/R)`; mock `imageSize`; mute hips position.
3. `applyRig` (adapt `tid_pipeline` helper, verified against real Kalidokit source) → `getNormalizedBoneNode`.
4. POC FIRST: one frame → inspect `rigRotation`; if limbs sane, run full sign → Day 5 GO/NO-GO.
5. **Human task (Muhammet/Ram):** VRoid realistic export → `tid-frontend/public/avatars/avatar.vrm`.

**Status:** data layer clean, layout + normalization understood. Critical path is unblocked toward the
Day 5 GO/NO-GO gate. Remaining blocker for a real avatar on screen is the VRoid export (manual).
