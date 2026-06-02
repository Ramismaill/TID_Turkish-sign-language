/**
 * TID Text-to-Sign Avatar Platform — Frontend
 * main.ts
 *
 * Stack: Vite + TypeScript + Three.js + @pixiv/three-vrm + Kalidokit
 *
 * Day 2: Scene setup, VRM placeholder, API connection, animation queue.
 * Day 9+: Real VRM file, LLM gloss generation wired in.
 *
 * Authors: Ram Ismail, Muhammet Ay
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// ── Config ────────────────────────────────────────────────────────────────────

const API_BASE = 'http://localhost:8000';
const FPS = 30;
const FRAME_INTERVAL_MS = 1000 / FPS;

// MediaPipe Holistic landmark layout (used by Day 4 retargeting):
//   225 = 75 keypoints × 3  →  [0:33] pose · [33:54] left hand · [54:75] right hand
//   (count constants will be (re)introduced in Day 4 where they are actually used)

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

// Try loading real VRM
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

loader.load(
  VRM_PATH,
  (gltf) => {
    vrm = gltf.userData.vrm as VRM;
    vrm.scene.rotation.y = Math.PI;  // face camera
    scene.add(vrm.scene);
    scene.remove(placeholder);
    setStatus('Avatar yüklendi ✓', 'success');
    console.log('[TID] VRM loaded:', vrm);
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
let currentSign: QueuedSign | null = null;
let currentFrame = 0;
let lastFrameTime = 0;
let isPlaying = false;

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

function applyLandmarksToVRM(vrm: VRM, landmarks: number[]) {
  // Stub — full kalidokit retargeting in Day 4-5
  // For now: drive left/right upper arm rotation from pose landmarks
  const leftShoulder  = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.LeftUpperArm);
  const rightShoulder = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.RightUpperArm);

  const ly = landmarks[15 * 3 + 1] ?? 0;
  const ry = landmarks[16 * 3 + 1] ?? 0;

  if (leftShoulder)  leftShoulder.rotation.z  =  ly * 1.5;
  if (rightShoulder) rightShoulder.rotation.z = -ry * 1.5;
}

function tickAnimation(timestamp: number) {
  if (!isPlaying) return;

  if (!currentSign && animQueue.length > 0) {
    currentSign = animQueue.shift()!;
    currentFrame = 0;
    updateWordDisplay(currentSign.word);
  }

  if (!currentSign) {
    isPlaying = false;
    setStatus('Animasyon tamamlandı ✓', 'success');
    setPlayButtonState(false);
    return;
  }

  if (timestamp - lastFrameTime >= FRAME_INTERVAL_MS) {
    const frame = currentSign.landmarks[currentFrame];
    if (frame) {
      if (vrm) {
        applyLandmarksToVRM(vrm, frame);
        vrm.update(FRAME_INTERVAL_MS / 1000);
      } else {
        applyLandmarksToPlaceholder(frame, currentFrame);
      }
    }

    currentFrame++;
    lastFrameTime = timestamp;

    if (currentFrame >= currentSign.landmarks.length) {
      currentSign = null;
    }
  }
}

// ── Render loop ───────────────────────────────────────────────────────────────

function animate(timestamp: number) {
  requestAnimationFrame(animate);
  controls.update();
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
