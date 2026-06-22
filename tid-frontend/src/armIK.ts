/**
 * armIK.ts — Position-based two-bone IK + face-contact anchor (approach B).
 *
 * 1. Place the wrist at a TARGET scaled to the avatar's arm, anchored at its
 *    shoulder → reproduces the pose at the right proportions.
 * 2. FACE-CONTACT ANCHOR: when the landmark wrist nears the face, pull the target
 *    to head height beside the avatar's head → the hand reaches the temple
 *    (recognizability core for hand-to-face signs like "merhaba"), regardless of
 *    the signer-vs-avatar proportion mismatch.
 * 3. Analytic elbow (law of cosines) bending toward the (mapped) signer elbow.
 *
 * Mapping mirrors the faithful skeleton (x, −y, z·fz). Fingers = Kalidokit. Torso static.
 *
 * Authors: Ram Ismail, Muhammet Ay
 */
import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';

export interface IKOpts { fx: number; fy: number; fz: number; lerp: number; face: number; faceDist: number; gate: number; smooth: number; }
// face: 0 → anchor OFF by default. The 0.55 radius pulled chest-level hands to the head and
// broke 5/6 GO/NO-GO signs; re-enable per-test with TID.ik.face = 0.55 until the smart anchor.
// faceDist: head bone origin is the CENTER of the head — the temple is ~9cm out, so the old
// 0.02 put the hand target inside the head mesh.
// smooth: 0.25 lagged the target ~270ms — turned fast repetitive signs (hayır, evet)
// to mush. 0.5 ≈ 110ms: still kills jitter, keeps the rhythm.
export const DEFAULT_IK: IKOpts = { fx: 1, fy: -1, fz: -0.5, lerp: 0.6, face: 0, faceDist: 0.09, gate: 0.07, smooth: 0.5 };

// reusable temporaries
const _restDir = new THREE.Vector3();
const _pq = new THREE.Quaternion();
const _tLocal = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _sh = new THREE.Vector3(), _elR = new THREE.Vector3(), _wrR = new THREE.Vector3();
const _se = new THREE.Vector3(), _ew = new THREE.Vector3(), _sw = new THREE.Vector3();
const _target = new THREE.Vector3(), _D = new THREE.Vector3(), _pole = new THREE.Vector3();
const _bend = new THREE.Vector3(), _elbow = new THREE.Vector3(), _qa = new THREE.Quaternion();
const _aim = new THREE.Vector3();
const _head = new THREE.Vector3(), _faceTgt = new THREE.Vector3(), _hd = new THREE.Vector3();
const _identity = new THREE.Quaternion();
const _smoothTgt: Record<string, THREE.Vector3> = {};   // per-arm smoothed target (anti-jitter)

/** Clear per-arm smoothing state. Call when a NEW sign starts — otherwise the arm
 *  glides in from the PREVIOUS sign's last target and corrupts its first frames. */
export function resetIKState() {
  for (const k of Object.keys(_smoothTgt)) delete _smoothTgt[k];
}

/** Rotate `bone` so its rest child-direction points along targetDirWorld. */
function aimBone(bone: THREE.Object3D, child: THREE.Object3D, targetDirWorld: THREE.Vector3, lerp: number) {
  _restDir.copy(child.position).normalize();
  if (_restDir.lengthSq() < 1e-8 || targetDirWorld.lengthSq() < 1e-8) return;
  bone.parent!.getWorldQuaternion(_pq);
  _tLocal.copy(targetDirWorld).applyQuaternion(_pq.invert()).normalize();
  _q.setFromUnitVectors(_restDir, _tLocal);
  bone.quaternion.slerp(_q, lerp);
}

/** Mapped landmark direction (pose index a → b) in VRM world axes. */
function mapDir(flat: number[], a: number, b: number, o: IKOpts, out: THREE.Vector3) {
  const ia = a * 3, ib = b * 3;
  return out.set(
    (flat[ib]     - flat[ia])     * o.fx,
    (flat[ib + 1] - flat[ia + 1]) * o.fy,
    (flat[ib + 2] - flat[ia + 2]) * o.fz,
  );
}

function solveArm(
  vrm: VRM, ua: string, la: string, hand: string,
  flat: number[], Si: number, Ei: number, Wi: number, o: IKOpts,
  signerLen?: number,
) {
  const h = vrm.humanoid;
  const uaB = h.getNormalizedBoneNode(ua as VRMHumanBoneName);
  const laB = h.getNormalizedBoneNode(la as VRMHumanBoneName);
  const hB  = h.getNormalizedBoneNode(hand as VRMHumanBoneName);
  if (!uaB || !laB || !hB) return;

  uaB.getWorldPosition(_sh);
  laB.getWorldPosition(_elR);
  hB.getWorldPosition(_wrR);
  const L1 = _sh.distanceTo(_elR);
  const L2 = _elR.distanceTo(_wrR);
  const reach = L1 + L2;

  // Scale signer arm → avatar reach. Use the PER-SIGN constant length when given:
  // per-frame lengths collapse when the arm points at the camera (foreshortened x/y
  // + damped z) → scale balloons → target overshoots. A constant keeps the wrist
  // trajectory a uniform copy of the signer's (i.e. exactly the faithful skeleton).
  mapDir(flat, Si, Ei, o, _se);   // still needed below as the elbow pole
  mapDir(flat, Ei, Wi, o, _ew);
  const len = (signerLen && signerLen > 1e-3) ? signerLen : (_se.length() + _ew.length() + 1e-6);
  const scale = reach / len;

  // Proportional wrist target
  mapDir(flat, Si, Wi, o, _sw);
  _target.copy(_sh).addScaledVector(_sw, scale);

  // FACE-CONTACT ANCHOR: if the landmark wrist is near the nose, pull the target to
  // head height beside the avatar's head (the temple) by proximity.
  if (o.face > 0) {
    const headB = h.getNormalizedBoneNode('head' as VRMHumanBoneName);
    if (headB) {
      const dn = Math.hypot(flat[Wi*3] - flat[0], flat[Wi*3+1] - flat[1], flat[Wi*3+2] - flat[2]);
      const t = 1 - dn / o.face;
      if (t > 0) {
        headB.getWorldPosition(_head);
        _hd.copy(_sw); _hd.y = 0;                          // side from STABLE shoulder→wrist (no jitter near head)
        if (_hd.lengthSq() > 1e-6) _hd.normalize(); else _hd.set(0, 0, 1);
        _faceTgt.copy(_head).addScaledVector(_hd, o.faceDist);   // temple at head height, faceDist beside it
        _target.lerp(_faceTgt, Math.min(1, t));
      }
    }
  }

  // Temporal smoothing of the target position (kills trembling/jitter)
  let st = _smoothTgt[ua];
  if (!st) st = _smoothTgt[ua] = _target.clone();
  else { st.lerp(_target, o.smooth); _target.copy(st); }

  // Law of cosines → elbow position so the wrist reaches the target
  _D.copy(_target).sub(_sh);
  let d = _D.length();
  _D.divideScalar(d || 1);
  d = THREE.MathUtils.clamp(d, Math.abs(L1 - L2) + 1e-4, reach - 1e-4);
  const cosA = THREE.MathUtils.clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1);
  const A = Math.acos(cosA);

  _bend.copy(_D).cross(_pole.copy(_se));
  if (_bend.lengthSq() < 1e-8) _bend.set(0, 0, 1);
  _bend.normalize();
  _qa.setFromAxisAngle(_bend, A);
  _elbow.copy(_D).applyQuaternion(_qa).multiplyScalar(L1).add(_sh);

  aimBone(uaB, laB, _aim.copy(_elbow).sub(_sh), o.lerp);
  aimBone(laB, hB, _aim.copy(_target).sub(_elbow), o.lerp);
}

const _down = new THREE.Vector3(0, -1, 0);

/** Let an inactive arm hang naturally at the side (aim both segments down). */
function restArm(vrm: VRM, ua: string, la: string, hand: string, lerp: number) {
  const h = vrm.humanoid;
  const uaB = h.getNormalizedBoneNode(ua as VRMHumanBoneName);
  const laB = h.getNormalizedBoneNode(la as VRMHumanBoneName);
  const hB  = h.getNormalizedBoneNode(hand as VRMHumanBoneName);
  if (!uaB || !laB || !hB) return;
  aimBone(uaB, laB, _down, lerp);
  aimBone(laB, hB, _down, lerp);
  hB.quaternion.slerp(_identity, lerp);   // neutral wrist (don't keep Kalidokit's cocked pose)
}

/** Drive ACTIVE arms via IK; rest inactive ones (so only the signing hand moves). */
export function applyArmIK(
  vrm: VRM, flat: number[], o: IKOpts = DEFAULT_IK,
  active: { left: boolean; right: boolean } = { left: true, right: true },
  reach?: { left: number; right: number },
) {
  if (active.left) solveArm(vrm, 'leftUpperArm', 'leftLowerArm', 'leftHand', flat, 11, 13, 15, o, reach?.left);
  else             restArm(vrm, 'leftUpperArm', 'leftLowerArm', 'leftHand', o.lerp);

  if (active.right) solveArm(vrm, 'rightUpperArm', 'rightLowerArm', 'rightHand', flat, 12, 14, 16, o, reach?.right);
  else              restArm(vrm, 'rightUpperArm', 'rightLowerArm', 'rightHand', o.lerp);
}

/** Per-sign constant signer arm length (mapped space), per arm. 95th percentile of
 *  |S→E|+|E→W| across frames ≈ the arm at its most extended, robust to noise spikes.
 *  Underestimating slightly (arm never fully extends) only scales the whole
 *  trajectory uniformly — shape is preserved, unlike per-frame scaling. */
export function computeArmReach(frames: number[][], o: IKOpts): { left: number; right: number } {
  const p95 = (a: number[]) => {
    const s = [...a].sort((x, y) => x - y);
    return s[Math.min(s.length - 1, Math.floor(s.length * 0.95))];
  };
  const chain = (f: number[], Si: number, Ei: number, Wi: number) => {
    mapDir(f, Si, Ei, o, _se);
    mapDir(f, Ei, Wi, o, _ew);
    return _se.length() + _ew.length();
  };
  const L: number[] = [], R: number[] = [];
  for (const f of frames) {
    L.push(chain(f, 11, 13, 15));
    R.push(chain(f, 12, 14, 16));
  }
  const out = { left: p95(L), right: p95(R) };
  console.log(`[TID] signer arm reach  L=${out.left.toFixed(3)}  R=${out.right.toFixed(3)}  (per-sign constant)`);
  return out;
}

/** Per-hand activity from wrist motion across a sign's frames (for rest-gating). */
export function computeActiveHands(frames: number[][], gate: number): { left: boolean; right: boolean } {
  if (gate <= 0 || frames.length < 2) return { left: true, right: true };
  const motion = (idx: number) => {
    const n = frames.length;
    let mx = 0, my = 0, mz = 0;
    for (const f of frames) { mx += f[idx * 3]; my += f[idx * 3 + 1]; mz += f[idx * 3 + 2]; }
    mx /= n; my /= n; mz /= n;
    let v = 0;
    for (const f of frames) {
      v += (f[idx*3]-mx)**2 + (f[idx*3+1]-my)**2 + (f[idx*3+2]-mz)**2;
    }
    return Math.sqrt(v / n);
  };
  const mL = motion(15), mR = motion(16);
  console.log(`[TID] wrist motion  L=${mL.toFixed(3)}  R=${mR.toFixed(3)}  (gate ${gate})`);
  return { left: mL > gate, right: mR > gate };
}
