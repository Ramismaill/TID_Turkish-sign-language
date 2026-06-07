/**
 * SkeletonFigure — renders raw landmark frames DIRECTLY in 3D (no retargeting).
 *
 * This is "the GIF character, in 3D": joints = spheres, bones = colored lines,
 * driven straight from the stored (225-float) landmark positions. Faithful by
 * construction — it IS the source data, so it reproduces the sign exactly.
 *
 * Layout (verified src/v1/extract_landmarks.py:68): 225 = 75 kp × 3
 *   [0:99] pose(33) · [99:162] left hand(21) · [162:225] right hand(21)
 * Coords are shoulder-centered, image y-DOWN, range ~[-1,1].
 *
 * Authors: Ram Ismail, Muhammet Ay
 */
import * as THREE from 'three';

// MediaPipe pose connections (upper body only — no legs needed for signing)
const POSE_BONES: [number, number][] = [
  [11, 12],            // shoulders
  [11, 13], [13, 15],  // left arm: shoulder→elbow→wrist
  [12, 14], [14, 16],  // right arm
  [11, 23], [12, 24],  // torso sides
  [23, 24],            // hips
  [0, 11], [0, 12],    // head(nose)→shoulders
];

// MediaPipe hand connections (21 pts: wrist + 5 fingers × 4)
const HAND_BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],         // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],         // index
  [0, 9], [9, 10], [10, 11], [11, 12],    // middle
  [0, 13], [13, 14], [14, 15], [15, 16],  // ring
  [0, 17], [17, 18], [18, 19], [19, 20],  // little
];

// Pose landmarks shown as joint spheres (upper body + head)
const POSE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24];

export interface SkeletonOpts {
  scale: number;    // overall size
  yOffset: number;  // vertical placement (shoulders sit here)
  zScale: number;   // depth scaling (MediaPipe z is noisy → damp it)
  flipX: number;    // +1 or -1 (mirror)
}
export const DEFAULT_SKELETON_OPTS: SkeletonOpts = { scale: 1.4, yOffset: 1.2, zScale: 0.5, flipX: 1 };

const COL_POSE  = new THREE.Color(0x5b8def);  // blue
const COL_LEFT  = new THREE.Color(0xe24a4a);  // red
const COL_RIGHT = new THREE.Color(0x3fb950);  // green

export class SkeletonFigure {
  group = new THREE.Group();
  private opts: SkeletonOpts;
  private joints: THREE.InstancedMesh;
  private lines: THREE.LineSegments;
  private linePos: Float32Array;
  private dummy = new THREE.Object3D();
  private _a = new THREE.Vector3();
  private _b = new THREE.Vector3();

  constructor(opts: SkeletonOpts = DEFAULT_SKELETON_OPTS) {
    this.opts = { ...opts };

    // ── Joint spheres (instanced): pose joints, then 21 left, then 21 right ──
    const jointCount = POSE_JOINTS.length + 21 + 21;            // 9 + 21 + 21 = 51
    const sphere = new THREE.SphereGeometry(1, 10, 10);         // unit; scaled per-instance
    const jmat = new THREE.MeshBasicMaterial();
    this.joints = new THREE.InstancedMesh(sphere, jmat, jointCount);
    this.joints.frustumCulled = false;
    let idx = 0;
    for (let i = 0; i < POSE_JOINTS.length; i++) this.joints.setColorAt(idx++, COL_POSE);
    for (let i = 0; i < 21; i++) this.joints.setColorAt(idx++, COL_LEFT);
    for (let i = 0; i < 21; i++) this.joints.setColorAt(idx++, COL_RIGHT);
    if (this.joints.instanceColor) this.joints.instanceColor.needsUpdate = true;
    // Start hidden (zero-scale) until the first update() supplies real positions.
    this.dummy.scale.setScalar(0);
    this.dummy.updateMatrix();
    for (let i = 0; i < jointCount; i++) this.joints.setMatrixAt(i, this.dummy.matrix);
    this.joints.instanceMatrix.needsUpdate = true;
    this.group.add(this.joints);

    // ── Bone lines (one LineSegments, vertex-colored) ──
    const segCount = POSE_BONES.length + HAND_BONES.length * 2; // 10 + 40 = 50
    this.linePos = new Float32Array(segCount * 2 * 3);
    const lcol = new Float32Array(segCount * 2 * 3);
    let c = 0;
    const pushCol = (col: THREE.Color) => {
      for (let k = 0; k < 2; k++) { lcol[c++] = col.r; lcol[c++] = col.g; lcol[c++] = col.b; }
    };
    for (let i = 0; i < POSE_BONES.length; i++) pushCol(COL_POSE);
    for (let i = 0; i < HAND_BONES.length; i++) pushCol(COL_LEFT);
    for (let i = 0; i < HAND_BONES.length; i++) pushCol(COL_RIGHT);
    const lgeo = new THREE.BufferGeometry();
    lgeo.setAttribute('position', new THREE.BufferAttribute(this.linePos, 3));
    lgeo.setAttribute('color', new THREE.BufferAttribute(lcol, 3));
    this.lines = new THREE.LineSegments(lgeo, new THREE.LineBasicMaterial({ vertexColors: true }));
    this.lines.frustumCulled = false;
    this.group.add(this.lines);
  }

  /** Map a landmark (base offset + local index) into world space. */
  private map(flat: number[], base: number, i: number, out: THREE.Vector3) {
    const o = base + i * 3;
    out.set(
      flat[o] * this.opts.scale * this.opts.flipX,
      -flat[o + 1] * this.opts.scale + this.opts.yOffset,  // flip y: image-down → world-up
      -flat[o + 2] * this.opts.scale * this.opts.zScale,
    );
  }

  private setJoint(i: number, v: THREE.Vector3, radius: number) {
    this.dummy.position.copy(v);
    this.dummy.scale.setScalar(radius);
    this.dummy.rotation.set(0, 0, 0);
    this.dummy.updateMatrix();
    this.joints.setMatrixAt(i, this.dummy.matrix);
  }

  /** Drive the whole figure from one 225-float landmark frame. */
  update(flat: number[]) {
    const v = this._a;
    let ji = 0;
    for (const p of POSE_JOINTS) { this.map(flat, 0, p, v); this.setJoint(ji++, v, 0.022); }
    for (let h = 0; h < 21; h++) { this.map(flat, 99, h, v); this.setJoint(ji++, v, 0.011); }
    for (let h = 0; h < 21; h++) { this.map(flat, 162, h, v); this.setJoint(ji++, v, 0.011); }
    this.joints.instanceMatrix.needsUpdate = true;

    let li = 0;
    const a = this._a, b = this._b;
    const seg = (base: number, i: number, j: number) => {
      this.map(flat, base, i, a); this.map(flat, base, j, b);
      this.linePos[li++] = a.x; this.linePos[li++] = a.y; this.linePos[li++] = a.z;
      this.linePos[li++] = b.x; this.linePos[li++] = b.y; this.linePos[li++] = b.z;
    };
    for (const [i, j] of POSE_BONES) seg(0, i, j);
    for (const [i, j] of HAND_BONES) seg(99, i, j);
    for (const [i, j] of HAND_BONES) seg(162, i, j);
    (this.lines.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
  }

  setVisible(b: boolean) { this.group.visible = b; }
  setOpts(o: Partial<SkeletonOpts>) { Object.assign(this.opts, o); }
}
