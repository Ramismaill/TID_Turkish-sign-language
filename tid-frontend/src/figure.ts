/**
 * figure.ts — EnrichedFigure: stylized volumetric character driven DIRECTLY by the
 * raw landmark frames. Same mapping as SkeletonFigure → faithful by construction
 * (NO retargeting) — but with a torso, head, capsule arms and volumetric fingers,
 * so it reads as a designed character instead of a debug skeleton.
 *
 * Design follows the avatar research consensus: legibility > realism, contrast is
 * the dominant variable → skin-tone hands/head vs solid teal clothing vs the dark
 * scene background. The torso volume restores the "location parameter" (is the
 * hand ON the chest or floating?) that a bare line skeleton loses.
 *
 * Authors: Ram Ismail, Muhammet Ay
 */
import * as THREE from 'three';
import { HAND_BONES, DEFAULT_SKELETON_OPTS, type SkeletonOpts } from './skeleton';

// Arm segments only — the shoulder line / torso sides are covered by the torso box
const ARM_BONES: [number, number][] = [
  [11, 13], [13, 15],  // left arm: shoulder→elbow→wrist
  [12, 14], [14, 16],  // right arm
];

// ── Palette (research: skin ≠ clothing ≠ background, solid colors, no pattern) ──
const COL_CLOTH = new THREE.Color(0x2e7d8c);  // teal shirt: arms + torso
const COL_SKIN  = new THREE.Color(0xf2c298);  // head + hands
const COL_EYE   = new THREE.Color(0x22262e);

// ── Radii (meters, world space) ──
const R_ARM       = 0.038;
const R_NECK      = 0.042;
const R_PALM_SEG  = 0.015;   // wrist→knuckle connectors (gives the palm volume)
const R_FINGER    = 0.0095;
const R_SHOULDER  = 0.052;
const R_ELBOW     = 0.04;
const R_WRIST     = 0.026;
const R_HAND_JOINT = 0.011;
const R_HEAD      = 0.105;

const UP = new THREE.Vector3(0, 1, 0);

// Palm connectors in HAND_BONES: [0,1],[0,5],[0,9],[0,13],[0,17] → indexes 0,4,8,12,16
const PALM_SEG_IDX = new Set([0, 4, 8, 12, 16]);

export class EnrichedFigure {
  group = new THREE.Group();
  private opts: SkeletonOpts;
  private limbs: THREE.InstancedMesh;     // cylinders: 4 arm + 1 neck + 40 hand segments
  private joints: THREE.InstancedMesh;    // spheres: shoulders/elbows/wrists + 42 hand pts
  private torso: THREE.Mesh;
  private head: THREE.Group;
  private dummy = new THREE.Object3D();
  private _a = new THREE.Vector3();
  private _b = new THREE.Vector3();
  private _dir = new THREE.Vector3();
  private _midSh = new THREE.Vector3();
  private _midHip = new THREE.Vector3();
  // Hold the last good hand block per side — degenerate frames (hand not detected →
  // all 21 pts collapse) would otherwise make the whole hand vanish for a frame.
  private lastLeft:  number[] | null = null;
  private lastRight: number[] | null = null;

  constructor(opts: SkeletonOpts = DEFAULT_SKELETON_OPTS) {
    this.opts = { ...opts };
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0 });

    // ── Limb cylinders: [0..3] arms · [4] neck · [5..24] left hand · [25..44] right ──
    const segCount = ARM_BONES.length + 1 + HAND_BONES.length * 2;
    this.limbs = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 12), mat.clone(), segCount);
    this.limbs.frustumCulled = false;
    for (let i = 0; i < ARM_BONES.length; i++) this.limbs.setColorAt(i, COL_CLOTH);
    this.limbs.setColorAt(ARM_BONES.length, COL_SKIN);                               // neck
    for (let i = 0; i < HAND_BONES.length * 2; i++) this.limbs.setColorAt(ARM_BONES.length + 1 + i, COL_SKIN);
    if (this.limbs.instanceColor) this.limbs.instanceColor.needsUpdate = true;
    this.group.add(this.limbs);

    // ── Joint spheres: [0,1] shoulders · [2,3] elbows · [4,5] wrists · [6..47] L hand · [48..89] R ──
    const jointCount = 6 + 21 + 21;
    this.joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 10), mat.clone(), jointCount);
    this.joints.frustumCulled = false;
    this.joints.setColorAt(0, COL_CLOTH); this.joints.setColorAt(1, COL_CLOTH);      // shoulders
    this.joints.setColorAt(2, COL_CLOTH); this.joints.setColorAt(3, COL_CLOTH);      // elbows (sleeve)
    for (let i = 4; i < jointCount; i++) this.joints.setColorAt(i, COL_SKIN);        // wrists + hands
    if (this.joints.instanceColor) this.joints.instanceColor.needsUpdate = true;
    this.group.add(this.joints);

    // Start hidden (zero scale) until the first update()
    this.dummy.scale.setScalar(0);
    this.dummy.updateMatrix();
    for (let i = 0; i < segCount; i++) this.limbs.setMatrixAt(i, this.dummy.matrix);
    for (let i = 0; i < jointCount; i++) this.joints.setMatrixAt(i, this.dummy.matrix);
    this.limbs.instanceMatrix.needsUpdate = true;
    this.joints.instanceMatrix.needsUpdate = true;

    // ── Torso: one rounded-ish box, shoulders→hips (unit box, scaled per frame) ──
    const torsoMat = mat.clone(); torsoMat.color = COL_CLOTH.clone();
    this.torso = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), torsoMat);
    this.torso.visible = false;
    this.group.add(this.torso);

    // ── Head: skin sphere + two eyes, positioned (not rotated) from the nose lm ──
    this.head = new THREE.Group();
    const headMat = mat.clone(); headMat.color = COL_SKIN.clone();
    this.head.add(new THREE.Mesh(new THREE.SphereGeometry(R_HEAD, 20, 16), headMat));
    const eyeMat = mat.clone(); eyeMat.color = COL_EYE.clone();
    for (const sx of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.012, 8, 8), eyeMat);
      eye.position.set(sx * 0.036, 0.012, R_HEAD * 0.82);
      this.head.add(eye);
    }
    this.head.visible = false;
    this.group.add(this.head);
  }

  /** Map a landmark (base offset + local index) into world space — IDENTICAL to SkeletonFigure. */
  private map(flat: number[], base: number, i: number, out: THREE.Vector3) {
    const o = base + i * 3;
    out.set(
      flat[o] * this.opts.scale * this.opts.flipX,
      -flat[o + 1] * this.opts.scale + this.opts.yOffset,
      -flat[o + 2] * this.opts.scale * this.opts.zScale,
    );
  }

  private setSeg(i: number, a: THREE.Vector3, b: THREE.Vector3, r: number) {
    const len = a.distanceTo(b);
    this.dummy.position.copy(a).add(b).multiplyScalar(0.5);
    if (len < 1e-5) {
      this.dummy.scale.setScalar(0);            // degenerate segment → hide
    } else {
      this.dummy.scale.set(r, len, r);
      this.dummy.quaternion.setFromUnitVectors(UP, this._dir.copy(b).sub(a).normalize());
    }
    this.dummy.updateMatrix();
    this.limbs.setMatrixAt(i, this.dummy.matrix);
  }

  private setJoint(i: number, v: THREE.Vector3, r: number) {
    this.dummy.position.copy(v);
    this.dummy.scale.setScalar(r);
    this.dummy.quaternion.identity();
    this.dummy.updateMatrix();
    this.joints.setMatrixAt(i, this.dummy.matrix);
  }

  /** Same spread test as main.ts: undetected hand → 21 pts collapse to one coord. */
  private static handDegenerate(flat: number[], base: number): boolean {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < 21; i++) {
      const x = flat[base + i * 3], y = flat[base + i * 3 + 1];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
    return (maxX - minX) < 0.05 && (maxY - minY) < 0.05;
  }

  /** Drive the whole figure from one 225-float landmark frame. */
  update(flat: number[]) {
    const a = this._a, b = this._b;

    // ── Degenerate-hand hold: swap in the last good 63-float block per side ──
    let src = flat;
    const leftBad = EnrichedFigure.handDegenerate(flat, 99);
    const rightBad = EnrichedFigure.handDegenerate(flat, 162);
    if (leftBad || rightBad) {
      src = flat.slice();
      if (leftBad && this.lastLeft) src.splice(99, 63, ...this.lastLeft);
      if (rightBad && this.lastRight) src.splice(162, 63, ...this.lastRight);
    }
    if (!leftBad)  this.lastLeft = flat.slice(99, 162);
    if (!rightBad) this.lastRight = flat.slice(162, 225);

    // ── Torso: box spanning shoulders (lm 11,12) → hips (lm 23,24) ──
    this.map(src, 0, 11, a); this.map(src, 0, 12, b);
    this._midSh.copy(a).add(b).multiplyScalar(0.5);
    const shoulderW = a.distanceTo(b);
    this.setJoint(0, a, R_SHOULDER);
    this.setJoint(1, b, R_SHOULDER);
    this.map(src, 0, 23, a); this.map(src, 0, 24, b);
    this._midHip.copy(a).add(b).multiplyScalar(0.5);
    const torsoH = this._midSh.distanceTo(this._midHip);
    this.torso.position.copy(this._midSh).add(this._midHip).multiplyScalar(0.5);
    this.torso.scale.set(shoulderW * 1.25, torsoH * 1.2, 0.14);
    this._dir.copy(this._midSh).sub(this._midHip).normalize();
    this.torso.quaternion.setFromUnitVectors(UP, this._dir);
    this.torso.visible = true;

    // ── Head: sphere behind/above the nose lm (lm 0) + neck from mid-shoulder ──
    this.map(src, 0, 0, a);
    a.y += 0.03; a.z -= 0.045;            // nose is the FRONT of the head → recess the center
    this.head.position.copy(a);
    this.head.visible = true;
    this.setSeg(ARM_BONES.length, this._midSh, a, R_NECK);   // neck

    // ── Arms (capsules) + elbow/wrist spheres ──
    for (let s = 0; s < ARM_BONES.length; s++) {
      this.map(src, 0, ARM_BONES[s][0], a);
      this.map(src, 0, ARM_BONES[s][1], b);
      this.setSeg(s, a, b, R_ARM);
    }
    this.map(src, 0, 13, a); this.setJoint(2, a, R_ELBOW);
    this.map(src, 0, 14, a); this.setJoint(3, a, R_ELBOW);
    this.map(src, 0, 15, a); this.setJoint(4, a, R_WRIST);
    this.map(src, 0, 16, a); this.setJoint(5, a, R_WRIST);

    // ── Hands: volumetric fingers + palm connectors + joint spheres ──
    let segI = ARM_BONES.length + 1;
    let jointI = 6;
    for (const base of [99, 162]) {
      for (let s = 0; s < HAND_BONES.length; s++) {
        this.map(src, base, HAND_BONES[s][0], a);
        this.map(src, base, HAND_BONES[s][1], b);
        this.setSeg(segI++, a, b, PALM_SEG_IDX.has(s) ? R_PALM_SEG : R_FINGER);
      }
      for (let h = 0; h < 21; h++) {
        this.map(src, base, h, a);
        this.setJoint(jointI++, a, h === 0 ? R_WRIST * 0.9 : R_HAND_JOINT);
      }
    }

    this.limbs.instanceMatrix.needsUpdate = true;
    this.joints.instanceMatrix.needsUpdate = true;
  }

  setVisible(v: boolean) { this.group.visible = v; }
  setOpts(o: Partial<SkeletonOpts>) { Object.assign(this.opts, o); }
}
