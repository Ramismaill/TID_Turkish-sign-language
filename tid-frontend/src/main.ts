/**
 * TID Text-to-Sign Avatar Platform — Frontend
 * main.ts
 *
 * Stack: Vite + TypeScript + Three.js + @pixiv/three-vrm + Kalidokit
 *
 * Day 2: Scene setup, VRM placeholder, API connection, animation queue.
 * Day 4: Kalidokit retargeting (calcArms bypass + Hand.solve), wrist-ownership,
 *         degenerate-frame skip, finger bone mapping (VRM 1.0), euler-log.
 * Day 9+: Real VRM file, LLM gloss generation wired in.
 *
 * Authors: Ram Ismail, Muhammet Ay
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Pose, Hand } from 'kalidokit';
import { SkeletonFigure } from './skeleton';
import { applyArmIK, computeActiveHands, DEFAULT_IK, type IKOpts } from './armIK';

// ── Config ────────────────────────────────────────────────────────────────────

const API_BASE = 'http://localhost:8000';
const FPS = 30;
const FRAME_INTERVAL_MS = 1000 / FPS;

// Landmark layout (verified: src/v1/extract_landmarks.py:68):
//   225 = 75 keypoints × 3 → [0:99] pose(33) · [99:162] left hand(21) · [162:225] right hand(21)
//   Coords are shoulder-centered (origin = midpoint of pose lm 11 & 12), range ~[-1,1]. NO face.
const POSE_COUNT = 33;
const HAND_COUNT = 21;

// ── Types ─────────────────────────────────────────────────────────────────────

interface SignResponse {
  word: string;
  input_word: string;
  frame_count: number;
  fps: number;
  duration_ms: number;
  landmarks: number[][];   // [frame][225 floats]
}

interface TranslateResponse {
  input: string;
  total_signs: number;
  unknown_words: string[];
  signs: SignResponse[];
}

// ── Three.js scene setup ──────────────────────────────────────────────────────

const canvas = document.getElementById('avatar-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setSize(canvas.clientWidth, canvas.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(
  35,
  canvas.clientWidth / canvas.clientHeight,
  0.1,
  100
);
camera.position.set(0, 1.4, 3.0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 1.0;
controls.maxDistance = 6.0;
controls.update();

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
directionalLight.position.set(1, 2, 2);
scene.add(directionalLight);

const fillLight = new THREE.DirectionalLight(0x8888ff, 0.4);
fillLight.position.set(-2, 1, -1);
scene.add(fillLight);

// Floor grid (subtle)
const gridHelper = new THREE.GridHelper(4, 10, 0x333355, 0x222244);
scene.add(gridHelper);

// ── VRM placeholder (box figure) ─────────────────────────────────────────────
// Shown until a real VRM file is loaded via VRoid Studio export.
// Replace by dropping a .vrm file into public/avatars/ and updating VRM_PATH.

const VRM_PATH = '/avatars/avatar.vrm';   // put your VRoid export here

let vrm: VRM | null = null;

function createPlaceholderFigure(): THREE.Group {
  const group = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: 0x6c63ff });

  // Body
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.5, 0.2), mat);
  body.position.y = 1.0;
  group.add(body);

  // Head
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.25, 0.25), mat);
  head.position.y = 1.45;
  group.add(head);

  // Left arm
  const lArm = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.45, 0.12), mat);
  lArm.position.set(-0.28, 1.0, 0);
  group.add(lArm);

  // Right arm
  const rArm = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.45, 0.12), mat);
  rArm.position.set(0.28, 1.0, 0);
  group.add(rArm);

  // Legs
  const lLeg = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.5, 0.16), mat);
  lLeg.position.set(-0.12, 0.5, 0);
  group.add(lLeg);

  const rLeg = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.5, 0.16), mat);
  rLeg.position.set(0.12, 0.5, 0);
  group.add(rLeg);

  return group;
}

const placeholder = createPlaceholderFigure();
scene.add(placeholder);

// ── Skeleton figure: faithful raw-landmark renderer (TID.mode = 'skeleton') ──
// Bypasses ALL retargeting — draws the stored landmarks directly. "The GIF, in 3D."
const skeleton = new SkeletonFigure();
scene.add(skeleton.group);
(window as any).SKEL = skeleton;   // live tuning: SKEL.setOpts({ scale: 1.6, yOffset: 1.1, zScale: 0, flipX: -1 })

function updateModeVisibility() {
  const isSkel = TID.mode === 'skeleton';
  skeleton.setVisible(isSkel);
  placeholder.visible = !isSkel && !vrm;
  if (vrm) vrm.scene.visible = !isSkel;
}

// Try loading real VRM
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

loader.load(
  VRM_PATH,
  (gltf) => {
    vrm = gltf.userData.vrm as VRM;
    // VRM 1.0 faces +Z = toward our camera (same frame as the skeleton + landmark data).
    // Math.PI here would turn it to face AWAY → IK hands land on the avatar's back.
    vrm.scene.rotation.y = 0;
    scene.add(vrm.scene);
    scene.remove(placeholder);
    setStatus('Avatar yüklendi ✓', 'success');
    console.log('[TID] VRM loaded:', vrm);

    // Day 4 Adım 1: log VRM version + validate finger bones
    const meta = (vrm as any).meta;
    console.log('[TID] VRM meta:', meta);
    console.log('[TID] VRM version:', meta?.metaVersion ?? 'unknown');

    // Validate 5×3 finger bones exist (each side)
    const fingerBones = [
      'leftThumbMetacarpal', 'leftThumbProximal', 'leftThumbDistal',
      'leftIndexProximal', 'leftIndexIntermediate', 'leftIndexDistal',
      'leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal',
      'leftRingProximal', 'leftRingIntermediate', 'leftRingDistal',
      'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal',
      'rightThumbMetacarpal', 'rightThumbProximal', 'rightThumbDistal',
      'rightIndexProximal', 'rightIndexIntermediate', 'rightIndexDistal',
      'rightMiddleProximal', 'rightMiddleIntermediate', 'rightMiddleDistal',
      'rightRingProximal', 'rightRingIntermediate', 'rightRingDistal',
      'rightLittleProximal', 'rightLittleIntermediate', 'rightLittleDistal',
    ];
    const missing = fingerBones.filter(
      b => !vrm!.humanoid.getNormalizedBoneNode(b as VRMHumanBoneName)
    );
    if (missing.length === 0) {
      console.log('[TID] ✅ All 30 finger bones present');
    } else {
      console.warn('[TID] ⚠️ Missing finger bones:', missing);
    }
  },
  undefined,
  () => {
    // No VRM file yet — placeholder stays, that's fine for Day 2
    console.log('[TID] No VRM file found at', VRM_PATH, '— using placeholder');
  }
);

// ── Animation queue ───────────────────────────────────────────────────────────

interface QueuedSign {
  word: string;
  landmarks: number[][];
  durationMs: number;
}

let animQueue: QueuedSign[] = [];
let lastSigns: QueuedSign[] = [];   // last played set, replayed when TID.loop = true
let currentSign: QueuedSign | null = null;
let currentFrame = 0;
let lastFrameTime = 0;
let isPlaying = false;
let signStartTs = 0;   // for frame-ticker timing log (Blocker 2 diagnostic)
let currentActiveHands = { left: true, right: true };   // which hands the IK drives (rest the other)

// ── Day 4: Kalidokit retargeting infrastructure ──────────────────────────────
// Architecture: docs/Day4_Decisions_Locked.md (K1–K5, two-Claude alignment)
// Source verified against: node_modules/kalidokit@1.1.5/dist/kalidokit.es.js
//
// K1: bypass solve() → use Pose.calcArms + Hand.solve directly
//     (no off-screen guard, no visibility check on this path)
// K2: skip calcHips → torso/spine stays static
// K3: single-reference (applied in backend)

interface LandmarkXYZ { x: number; y: number; z: number }

/** Convert flat[225] → structured pose/hand arrays for Kalidokit */
function parseStoredFrame(flat: number[]): {
  pose: LandmarkXYZ[];
  leftHand: LandmarkXYZ[];
  rightHand: LandmarkXYZ[];
  leftDegenerate: boolean;
  rightDegenerate: boolean;
} {
  const toArray = (start: number, count: number): LandmarkXYZ[] =>
    Array.from({ length: count }, (_, i) => ({
      x: flat[start + i * 3],
      y: flat[start + i * 3 + 1],
      z: flat[start + i * 3 + 2],
    }));

  // Offsets match extract_landmarks.py:68 → np.concatenate([pose_xyz, left_xyz, right_xyz])
  const pose      = toArray(0,   POSE_COUNT);   // flat[0:99]   → 33 × {x,y,z}
  const leftHand  = toArray(99,  HAND_COUNT);   // flat[99:162]  → 21 × {x,y,z}
  const rightHand = toArray(162, HAND_COUNT);   // flat[162:225] → 21 × {x,y,z}

  return {
    pose,
    leftHand,
    rightHand,
    leftDegenerate:  isHandDegenerate(leftHand),
    rightDegenerate: isHandDegenerate(rightHand),
  };
}

/**
 * Detect degenerate hand frames.
 * Undetected hands → zeros in extraction → after shoulder-centering, all 21 points
 * collapse to ~same coord. Hand.solve on these → NaN finger angles.
 * Threshold 0.05: any real hand pose has spread > 0.05 in both x and y.
 */
function isHandDegenerate(pts: LandmarkXYZ[]): boolean {
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return (maxX - minX) < 0.05 && (maxY - minY) < 0.05;
}

/**
 * Apply euler rotation to a VRM bone with dampener + slerp interpolation.
 * Based on canonical Kalidokit VRM demo pattern (bone-local space).
 * three-vrm v3 uses getNormalizedBoneNode with camelCase string names.
 */
function rigRotation(
  vrmModel: VRM,
  boneName: string,
  rotation: { x: number; y: number; z: number },
  dampener = 1.0,
  lerpAmount = 0.3,
) {
  const bone = vrmModel.humanoid.getNormalizedBoneNode(boneName as VRMHumanBoneName);
  if (!bone) return;

  const euler = new THREE.Euler(
    rotation.x * dampener,
    rotation.y * dampener,
    rotation.z * dampener,
    'XYZ',
  );
  const target = new THREE.Quaternion().setFromEuler(euler);
  bone.quaternion.slerp(target, lerpAmount);
}

/**
 * Finger bone mapping: Kalidokit Hand.solve key → VRM 1.0 bone name.
 * Critical: thumb bones are RENAMED in VRM 1.0 vs Kalidokit:
 *   Kalidokit ThumbProximal     → VRM thumbMetacarpal
 *   Kalidokit ThumbIntermediate → VRM thumbProximal
 *   Kalidokit ThumbDistal       → VRM thumbDistal
 * Other fingers: just lowercase the side prefix.
 */
const FINGER_BONE_MAP: Record<string, string> = {
  // Left thumb (VRM 1.0 rename)
  LeftThumbProximal:      'leftThumbMetacarpal',
  LeftThumbIntermediate:  'leftThumbProximal',
  LeftThumbDistal:        'leftThumbDistal',
  // Left index
  LeftIndexProximal:      'leftIndexProximal',
  LeftIndexIntermediate:  'leftIndexIntermediate',
  LeftIndexDistal:        'leftIndexDistal',
  // Left middle
  LeftMiddleProximal:     'leftMiddleProximal',
  LeftMiddleIntermediate: 'leftMiddleIntermediate',
  LeftMiddleDistal:       'leftMiddleDistal',
  // Left ring
  LeftRingProximal:       'leftRingProximal',
  LeftRingIntermediate:   'leftRingIntermediate',
  LeftRingDistal:         'leftRingDistal',
  // Left little
  LeftLittleProximal:     'leftLittleProximal',
  LeftLittleIntermediate: 'leftLittleIntermediate',
  LeftLittleDistal:       'leftLittleDistal',
  // Right thumb (VRM 1.0 rename)
  RightThumbProximal:     'rightThumbMetacarpal',
  RightThumbIntermediate: 'rightThumbProximal',
  RightThumbDistal:       'rightThumbDistal',
  // Right index
  RightIndexProximal:     'rightIndexProximal',
  RightIndexIntermediate: 'rightIndexIntermediate',
  RightIndexDistal:       'rightIndexDistal',
  // Right middle
  RightMiddleProximal:    'rightMiddleProximal',
  RightMiddleIntermediate:'rightMiddleIntermediate',
  RightMiddleDistal:      'rightMiddleDistal',
  // Right ring
  RightRingProximal:      'rightRingProximal',
  RightRingIntermediate:  'rightRingIntermediate',
  RightRingDistal:        'rightRingDistal',
  // Right little
  RightLittleProximal:    'rightLittleProximal',
  RightLittleIntermediate:'rightLittleIntermediate',
  RightLittleDistal:      'rightLittleDistal',
};

// Retargeting state: hold last good hand rig for degenerate frames
let lastLeftHandRig:  Record<string, { x: number; y: number; z: number }> | null = null;
let lastRightHandRig: Record<string, { x: number; y: number; z: number }> | null = null;

// ── Live tuning harness (Day 4 Adım 5: axis reconciliation) ───────────────────
// Kalidokit euler ↔ VRM bone axis/sign differ; values need per-axis scaling.
// Tune LIVE in the browser console while a sign loops:
//   TID.loop = true        → repeat the last sign continuously (so you can watch + tune)
//   TID.ua.z = 1.2         → upper-arm Z multiplier (etc. for .x .y, and TID.la / TID.wr)
//   TID.fingers = 0.8      → finger bend multiplier
//   TID.eulerLog = true    → dump one frame's raw euler to console, then auto-off
//   TID.reset()            → snap arms/hands back to T-pose (rest)
interface TIDTuning {
  ua: { x: number; y: number; z: number };   // upper arm per-axis multiplier
  la: { x: number; y: number; z: number };   // lower arm
  wr: { x: number; y: number; z: number };   // wrist
  fingers: number;                            // finger bend multiplier
  fingerAxis: 'x' | 'y' | 'z';                // which local axis curls a finger (VRM 1.0 test)
  fingerSign: number;                         // +1 / -1 curl direction
  testCurl: number | null;                    // if set, force ALL fingers to this curl (axis test)
  loop: boolean;
  eulerLog: boolean;
  mode: 'vrm' | 'skeleton' | 'ik';            // 'skeleton'=raw landmarks · 'ik'=position-based arm IK
  ik: IKOpts;                                 // tunable landmark→world axis mapping for IK arms
  reset?: () => void;
}
declare global { interface Window { TID: TIDTuning } }

const TID: TIDTuning = (window.TID = window.TID || {
  ua: { x: 0.6, y: 0.25, z: -0.6 },   // z negated: Kalidokit↔VRM1.0 axis flip (verified live)
  la: { x: 0.7, y: 0.3,  z: -0.6 },   // same flip hypothesis (LowerArm.z often ≈0 anyway)
  wr: { x: 1.0, y: 1.0,  z: 1.0 },
  fingers: 1.0,
  fingerAxis: 'y',   // VERIFIED via live test: 'y' curls fingers into palm on VRM 1.0
  fingerSign: 1,
  testCurl: null,
  loop: false,
  eulerLog: false,
  mode: 'ik',         // testing approach B (position-based arm IK); 'skeleton'/'vrm' to compare
  ik: { ...DEFAULT_IK },
});

// Snap all driven bones back to rest (T-pose). Useful while tuning: TID.reset()
TID.reset = () => {
  if (!vrm) return;
  const bones = [
    'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm',
    'leftHand', 'rightHand', ...Object.values(FINGER_BONE_MAP),
  ];
  for (const b of bones) {
    const node = vrm.humanoid.getNormalizedBoneNode(b as VRMHumanBoneName);
    if (node) node.quaternion.identity();
  }
  console.log('[TID] reset to T-pose');
};

// ── End retargeting infrastructure ───────────────────────────────────────────

function applyLandmarksToPlaceholder(landmarks: number[], _frameIdx: number) {
  // Day 2: simple proof-of-concept — wiggle arms based on wrist Y coords
  // landmarks layout (225 = 75 keypoints × 3 — verified in src/v1/extract_landmarks.py:68):
  //   [0:99] pose 33 · [99:162] left hand 21 · [162:225] right hand 21  (NO face)
  //   coords are shoulder-centered (origin = midpoint of pose lm 11 & 12), range ~[-1,1]
  // We only use pose landmarks 15 (left wrist) and 16 (right wrist) for now.

  const poseStart = 0;
  const leftWristIdx  = poseStart + 15 * 3;   // x,y,z
  const rightWristIdx = poseStart + 16 * 3;

  const ly = landmarks[leftWristIdx + 1]  ?? 0;   // Y coord
  const ry = landmarks[rightWristIdx + 1] ?? 0;

  // Map [-1..1] landmark space to rotation
  const lArm = placeholder.children[2] as THREE.Mesh;
  const rArm = placeholder.children[3] as THREE.Mesh;

  if (lArm) lArm.rotation.z =  ly * 1.2;
  if (rArm) rArm.rotation.z = -ry * 1.2;
}

function applyLandmarksToVRM(vrmModel: VRM, landmarks: number[], active?: { left: boolean; right: boolean }) {
  // Day 4: Real Kalidokit retargeting
  // K1: bypass solve() → calcArms + Hand.solve directly (no off-screen/visibility freeze)
  // K2: no calcHips → torso static
  // Wrist-ownership: bend (x,y) ← Hand.solve, twist (z) ← calcArms.Hand
  // See: docs/Day4_Decisions_Locked.md

  const { pose, leftHand, rightHand, leftDegenerate, rightDegenerate } = parseStoredFrame(landmarks);

  // ── Arms (K1: calcArms directly, NO solve()) ──────────────────────────────
  // calcArms: relative vectors (es.js:589-604), translation+scale invariant.
  // Returns clamped euler angles (rigArm called internally, es.js:606-607).
  const arms = Pose.calcArms(pose as any);

  // ── Left hand ─────────────────────────────────────────────────────────────
  let leftRig = lastLeftHandRig;
  if (!leftDegenerate) {
    const solved = Hand.solve(leftHand as any, 'Left');
    if (solved) {
      leftRig = solved as Record<string, { x: number; y: number; z: number }>;
      lastLeftHandRig = leftRig;
    }
  }

  // ── Right hand ────────────────────────────────────────────────────────────
  let rightRig = lastRightHandRig;
  if (!rightDegenerate) {
    const solved = Hand.solve(rightHand as any, 'Right');
    if (solved) {
      rightRig = solved as Record<string, { x: number; y: number; z: number }>;
      lastRightHandRig = rightRig;
    }
  }

  // ── Euler-log instrumentation (Adım 5) ────────────────────────────────────
  // Usage: console → TID.eulerLog = true → play a sign. Logs one frame, auto-off.
  if (TID.eulerLog) {
    console.log('[TID euler-log]', JSON.stringify({
      arms: {
        UpperArm_L: { x: arms.UpperArm.l.x, y: arms.UpperArm.l.y, z: arms.UpperArm.l.z },
        UpperArm_R: { x: arms.UpperArm.r.x, y: arms.UpperArm.r.y, z: arms.UpperArm.r.z },
        LowerArm_L: { x: arms.LowerArm.l.x, y: arms.LowerArm.l.y, z: arms.LowerArm.l.z },
        LowerArm_R: { x: arms.LowerArm.r.x, y: arms.LowerArm.r.y, z: arms.LowerArm.r.z },
        Hand_L: { x: arms.Hand.l.x, y: arms.Hand.l.y, z: arms.Hand.l.z },
        Hand_R: { x: arms.Hand.r.x, y: arms.Hand.r.y, z: arms.Hand.r.z },
      },
      leftRig,
      rightRig,
      leftDegenerate,
      rightDegenerate,
    }, null, 2));
    TID.eulerLog = false;  // one-shot
  }

  // ── Apply arms to VRM bones (live-tuned via TID.ua / TID.la) ───────────────
  // K2: NO calcHips, NO spine — torso stays at rest (sign language = upper body)
  // Per-axis multipliers reconcile Kalidokit euler vs VRM bone axes (see TID harness).
  const { ua, la, wr } = TID;
  const UA_L = arms.UpperArm.l, UA_R = arms.UpperArm.r;
  const LO_L = arms.LowerArm.l, LO_R = arms.LowerArm.r;

  // In 'ik' mode the arms are driven by applyArmIK (position-based) — skip Kalidokit arms here.
  if (TID.mode !== 'ik') {
    rigRotation(vrmModel, 'leftUpperArm',  { x: UA_L.x * ua.x, y: UA_L.y * ua.y, z: UA_L.z * ua.z }, 1.0, 0.4);
    rigRotation(vrmModel, 'rightUpperArm', { x: UA_R.x * ua.x, y: UA_R.y * ua.y, z: UA_R.z * ua.z }, 1.0, 0.4);
    rigRotation(vrmModel, 'leftLowerArm',  { x: LO_L.x * la.x, y: LO_L.y * la.y, z: LO_L.z * la.z }, 1.0, 0.4);
    rigRotation(vrmModel, 'rightLowerArm', { x: LO_R.x * la.x, y: LO_R.y * la.y, z: LO_R.z * la.z }, 1.0, 0.4);
  }

  // ── Finger application with VRM-axis routing (live-testable) ───────────────
  // Kalidokit puts non-thumb finger curl in .z (built for VRM 0.x). On VRM 1.0
  // the curl axis may differ → route .z to TID.fingerAxis. Test live:
  //   TID.testCurl = 0.8 ; TID.fingerAxis = 'x' (or 'y','z')  → which axis curls fingers?
  const applyFingers = (rig: Record<string, { x: number; y: number; z: number }>, prefix: 'Left' | 'Right') => {
    for (const [kKey, vrmBone] of Object.entries(FINGER_BONE_MAP)) {
      if (!kKey.startsWith(prefix)) continue;
      const r = rig[kKey];
      if (!r) continue;
      const e: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 };
      if (TID.testCurl != null) {
        e[TID.fingerAxis] = TID.testCurl;                        // force-curl test (ignore data)
      } else if (kKey.includes('Thumb')) {
        e.x = r.x * TID.fingers; e.y = r.y * TID.fingers; e.z = r.z * TID.fingers;
      } else {
        e[TID.fingerAxis] = r.z * TID.fingers * TID.fingerSign;  // route curl to VRM axis
      }
      rigRotation(vrmModel, vrmBone, e, 1.0, 0.5);
    }
  };

  // ── Apply left hand + fingers (skip if this hand is resting in IK mode) ────
  if (leftRig && (!active || active.left)) {
    const lw = leftRig['LeftWrist'];   // K1 wrist: bend (x,y) ← Hand.solve, twist (z) ← calcArms
    if (lw) rigRotation(vrmModel, 'leftHand', { x: lw.x * wr.x, y: lw.y * wr.y, z: arms.Hand.l.z * wr.z }, 1.0, 0.4);
    applyFingers(leftRig, 'Left');
  }

  // ── Apply right hand + fingers (skip if this hand is resting in IK mode) ───
  if (rightRig && (!active || active.right)) {
    const rw = rightRig['RightWrist'];
    if (rw) rigRotation(vrmModel, 'rightHand', { x: rw.x * wr.x, y: rw.y * wr.y, z: arms.Hand.r.z * wr.z }, 1.0, 0.4);
    applyFingers(rightRig, 'Right');
  }
}

function tickAnimation(timestamp: number) {
  if (!isPlaying) return;

  if (!currentSign && animQueue.length > 0) {
    currentSign = animQueue.shift()!;
    currentFrame = 0;
    updateWordDisplay(currentSign.word);
  }

  if (!currentSign) {
    if (TID.loop && lastSigns.length > 0) {
      animQueue = lastSigns.map(s => ({ ...s }));   // re-queue for continuous tuning
      currentSign = animQueue.shift()!;
      currentFrame = 0;
      updateWordDisplay(currentSign.word);
    } else {
      isPlaying = false;
      setStatus('Animasyon tamamlandı ✓', 'success');
      setPlayButtonState(false);
      return;
    }
  }

  if (timestamp - lastFrameTime >= FRAME_INTERVAL_MS) {
    if (currentFrame === 0) {
      signStartTs = timestamp;
      currentActiveHands = computeActiveHands(currentSign.landmarks, TID.ik.gate);
      console.log(`[TID] '${currentSign.word}' active hands:`, currentActiveHands);
    }
    const frame = currentSign.landmarks[currentFrame];
    if (frame) {
      if (TID.mode === 'skeleton') {
        skeleton.update(frame);                       // faithful: draw raw landmarks directly
      } else if (vrm) {
        if (TID.mode === 'ik') applyArmIK(vrm, frame, TID.ik, currentActiveHands);   // position-based arms (B)
        applyLandmarksToVRM(vrm, frame, TID.mode === 'ik' ? currentActiveHands : undefined);   // wrist+fingers (skips resting hand)
        vrm.update(FRAME_INTERVAL_MS / 1000);
      } else {
        applyLandmarksToPlaceholder(frame, currentFrame);
      }
    }

    currentFrame++;
    lastFrameTime = timestamp;

    if (currentFrame >= currentSign.landmarks.length) {
      // Blocker 2 diagnostic: did all 64 frames stream over ~2s, or snap instantly?
      const elapsed = Math.round(timestamp - signStartTs);
      console.log(`[TID] played '${currentSign.word}': ${currentFrame} frames in ${elapsed}ms (target ~${Math.round(currentFrame * FRAME_INTERVAL_MS)}ms)`);
      currentSign = null;
    }
  }
}

// ── Render loop ───────────────────────────────────────────────────────────────

function animate(timestamp: number) {
  requestAnimationFrame(animate);
  controls.update();
  updateModeVisibility();
  tickAnimation(timestamp);
  renderer.render(scene, camera);
}
animate(0);

// ── API calls ─────────────────────────────────────────────────────────────────

async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    return data.pipeline_loaded === true;
  } catch {
    return false;
  }
}

async function translateText(text: string): Promise<TranslateResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.error('[TID] translate error:', e);
    return null;
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function setStatus(msg: string, type: 'info' | 'success' | 'error' | 'loading' = 'info') {
  const el = document.getElementById('status')!;
  el.textContent = msg;
  el.className = `status ${type}`;
}

function setPlayButtonState(playing: boolean) {
  const btn = document.getElementById('play-btn') as HTMLButtonElement;
  btn.disabled = playing;
  btn.textContent = playing ? 'Oynatılıyor...' : '▶ Çevir & Oynat';
}

function updateWordDisplay(word: string) {
  const el = document.getElementById('current-word')!;
  el.textContent = word;
}

function renderVocabList(words: string[]) {
  const el = document.getElementById('vocab-list')!;
  el.innerHTML = words
    .map(w => `<span class="vocab-chip">${w}</span>`)
    .join('');
}

// ── Main UI logic ─────────────────────────────────────────────────────────────

const playBtn = document.getElementById('play-btn') as HTMLButtonElement;
const textInput = document.getElementById('text-input') as HTMLTextAreaElement;

playBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) {
    setStatus('Lütfen bir metin girin.', 'error');
    return;
  }

  setStatus('API\'ye gönderiliyor...', 'loading');
  setPlayButtonState(true);

  const result = await translateText(text);
  if (!result) {
    setStatus('Backend\'e ulaşılamadı. Sunucunun çalıştığından emin ol.', 'error');
    setPlayButtonState(false);
    return;
  }

  if (result.total_signs === 0) {
    setStatus(`Hiçbir kelime sözlükte bulunamadı. Bilinmeyenler: ${result.unknown_words.join(', ')}`, 'error');
    setPlayButtonState(false);
    return;
  }

  // Build animation queue
  animQueue = result.signs.map(s => ({
    word: s.word,
    landmarks: s.landmarks,
    durationMs: s.duration_ms,
  }));
  lastSigns = animQueue.map(s => ({ ...s }));   // remember for TID.loop

  const unknownMsg = result.unknown_words.length > 0
    ? ` (atlandı: ${result.unknown_words.join(', ')})`
    : '';

  setStatus(`${result.total_signs} işaret oynatılıyor${unknownMsg}`, 'loading');
  isPlaying = true;
  lastFrameTime = 0;
});

// ── Init ──────────────────────────────────────────────────────────────────────

(async () => {
  setStatus('Backend kontrol ediliyor...', 'loading');
  const healthy = await checkHealth();

  if (healthy) {
    setStatus('Backend bağlantısı OK ✓', 'success');

    // Load vocab list
    try {
      const res = await fetch(`${API_BASE}/vocab`);
      const data = await res.json();
      renderVocabList(data.words);
    } catch {
      // Non-fatal
    }
  } else {
    setStatus('Backend çevrimdışı — uvicorn\'u başlat: uvicorn src.v2.server:app --reload --port 8000', 'error');
  }
})();

// ── Resize handler ────────────────────────────────────────────────────────────

window.addEventListener('resize', () => {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});
