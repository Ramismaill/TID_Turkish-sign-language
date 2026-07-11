# TID Projesi — Ram'ın Claude Oturumu İçin Tam Bağlam (Day 4 başlangıcı)

> **Bu dosyayı Claude'a ver:** Tüm geçmişi, kararları ve Day 4'ün tam teknik planını içerir.
> **Tarih:** 2 Haziran 2026 | **Branch:** `feature/text-to-sign-v2` | **Son commit:** `50bd0e8`
>
> ⚠️ **GÜNCELLEME (2 Haz akşamı):** Bu dokümanın §5 retargeting planı (visibility:1.0 enjeksiyonu + `solve()` kullanımı) **ARTIK GÜNCEL DEĞİL.** İki Claude oturumu kaynak-doğrulamasıyla `solve()` bypass'ına (`calcArms` + `Hand.solve`) karar verdi → visibility enjeksiyonu gereksiz, calcHips skip, single-ref. **Kilitli kararlar için → `docs/Day4_Decisions_Locked.md`.**

---

## 1. Sen Kimsin, Proje Ne

**Ram İsmail** — TID projesinin co-lead'i. Muhammet Ay ile birlikte yürütüyorsunuz.

**Proje:** Türkçe metin → Türk İşaret Dili (TİD) → 3D VRM avatar animasyonu.
Web tabanlı, tarayıcıda çalışır. FEE306 Applied Neural Networks dersi, İstanbul Topkapı Üniversitesi.

**Repo:** `github.com/Ramismaill/TID_Turkish-sign-language`
**Branch:** `feature/text-to-sign-v2`
**Proje kökü (MM PC):** `C:\sign_language\`

---

## 2. Stack (kilitli — tartışmaya açma)

| Katman | Teknoloji |
|---|---|
| Backend | FastAPI + uvicorn, Python 3.11, conda env `isaret_dili` |
| Landmark verisi | AUTSL dataset, 678 `.npy`, padded format (`cls014_1.npy`) |
| Frontend | Vite + TypeScript + Three.js + `@pixiv/three-vrm` ^2.1.0 + Kalidokit 1.1.5 |
| Avatar | VRoid Studio → VRM (realistic preset, henüz export edilmedi) |
| LLM | Qwen2.5-7B Q4_K_M (Day 13'te entegre edilecek) |
| GPU | MM PC: RTX 4060 Laptop 8GB VRAM |

---

## 3. Klasör Yapısı

```
C:\sign_language\
├── src\
│   ├── v1\              # TMS-Net 94.70% recognition (bitti, dokunma)
│   ├── v2\
│   │   ├── server.py              # FastAPI (4 endpoint, çalışıyor)
│   │   ├── text_to_sign.py        # Pipeline (çalışıyor)
│   │   ├── sign_dictionary.json   # 15 kelime vocab
│   │   └── test_translate_full.py # Day 3 test (15/15 geçiyor)
│   └── shared\          # Boş — Day 13'te llm_singleton gelecek
├── tid-frontend\
│   └── src\
│       ├── main.ts       # Three.js scene + animation queue + API client
│       └── style.css
├── reference_landmarks\ # 678 .npy (gitignore'da)
├── JOURNAL.md            # Geliştirme günlüğü (Day 3 girişi var)
└── docs\
    └── Ram_Claude_Context_Day4.md  # Bu dosya
```

---

## 4. Ne Bitti (Day 1-3)

### Backend — TAMAM, dokunma
- `/health` `/vocab` `/sign/{word}` `/translate` — 4 endpoint çalışıyor
- 15/15 kelime `(64, 225)` shape ile yükleniyor
- Variant lookup: `sağol → teşekkür ederim`, suffix-strip: `okula → okul`
- Canlı test edildi (tarayıcıdan da doğrulandı)

### Frontend — TAMAM ama placeholder
- Vite dev server (`localhost:5173`) çalışıyor
- Placeholder kutu-adam var, kolları bilek-Y'ye göre hafifçe döndürüyor
- Animation queue (FIFO), 30 FPS ticker, API bağlantısı hepsi OK
- `applyLandmarksToVRM` fonksiyonu **STUB** — sadece üst-kol rotasyonu, gerçek retargeting YOK
- `npm run build` de geçiyor (exit 0)

### Landmark layout — KANITLANDI
Kaynak: `src/v1/extract_landmarks.py:68`
```python
vec = np.concatenate([pose_xyz, left_xyz, right_xyz])  # 33 + 21 + 21 = 75 kp × 3 = 225
```
- `[0:99]` → pose (33 kp)
- `[99:162]` → sol el (21 kp)
- `[162:225]` → sağ el (21 kp)
- **Face YOK.** Değer aralığı ~[-1, 1], **omuz-merkezli** normalize.

---

## 5. Day 4 — Şimdi Yapılacak (Kalidokit Retargeting)

### 5.1 Hedef

`applyLandmarksToVRM` fonksiyonunu gerçek hale getir:
```
flat[225] → {pose, leftHand, rightHand} → Kalidokit → VRM kemik rotasyonları
```

### 5.2 Kritik Bulgular (araştırma + kod incelemesinden)

**A) Visibility enjeksiyonu ZORUNLU (en kritik):**
Kalidokit kaynak kodu (`PoseSolver/index.ts`):
```javascript
const rightHandOff = lm3d[15].y > 0.1 || (lm3d[15].visibility ?? 0) < 0.23 || lm2d[15].y > 0.995;
if (rightHandOff) Arms.UpperArm.r = Arms.UpperArm.r.multiply(0);  // ← rotasyonu SIFIRLIYOR
```
Bizim `.npy`'de `visibility` alanı YOK → `(visibility ?? 0) = 0 < 0.23` → **avatar rest pozunda DONAR**.
**Çözüm:** Parse adımında her landmark'a `visibility: 1.0` enjekte et.

**B) Dejenere el frame'leri (bizim özgün bulgumuz):**
`extract_landmarks.py`'de tespit edilemeyen el → sıfır yazılır → omuz-merkezleme sonrası `-center`'a çöker → tüm el tek noktada.
Bu frame'de `visibility:1.0` enjekte edersek Kalidokit aynı noktalardan parmak açısı hesaplar → NaN/bozuk.
**Çözüm:** El landmark'larının tümü `|x|,|y|,|z| < 0.01` ise → dejenere kabul et → `Hand.solve` ÇAĞIRMA → son iyi pozu tut.

**C) Hips pozisyon drift:**
`Hips.position` mutlak koordinat varsayar (metre). Bizim ~[-1,1] veri avatar'ı ekranın dışına sürükler.
**Çözüm:** `rigPosition('hips', ...)` çağrısında `y += 1.0` yap ve dampener düşük tut. Ya da tamamen mute et.

**D) Açı matematiği güvenli:**
Kalidokit göreli vektörler + normalize dot product kullandığından omuz-merkezleme (translation) ve uniform scale açıları BOZMAZ. Bu kanıtlandı.

**E) `runtime:'mediapipe'` vs `'tfjs'`:**
`mediapipe` modu x,y'yi [0,1] bekler. `tfjs` modu imageSize'a böler.
Canlı video olmadığı için mock imageSize ver:
```javascript
Pose.solve(pose3D, pose3D, { runtime: 'tfjs', imageSize: { width: 640, height: 480 }, enableLegs: false })
```
**Not:** Hem `lm3d` hem `lm2d` parametresi var — biz ikisine de aynı diziyi veriyoruz (2D yok, 3D var).

### 5.3 three-vrm v2 API — eski vs yeni (BREAKİNG CHANGE)

| Eski (v0.x) | Yeni (v2.x — KULLAN) |
|---|---|
| `getBoneNode(VRMSchema.HumanoidBoneName[name])` | `getNormalizedBoneNode('leftUpperArm')` |
| PascalCase enum | camelCase string |

Bizim `main.ts:195` zaten `getNormalizedBoneNode` kullanıyor ✓

**Thumb bone rename (VRM 1.0):**
| Kalidokit key | VRM v2 bone name |
|---|---|
| `LeftThumbProximal` | `'leftThumbMetacarpal'` |
| `LeftThumbIntermediate` | `'leftThumbProximal'` |
| `LeftThumbDistal` | `'leftThumbDistal'` |

### 5.4 Tam Kemik Mapping (Kalidokit → VRM v2 camelCase)

```typescript
const FINGER_BONE_MAP: Record<string, string> = {
  // Sol el
  LeftThumbProximal: 'leftThumbMetacarpal',
  LeftThumbIntermediate: 'leftThumbProximal',
  LeftThumbDistal: 'leftThumbDistal',
  LeftIndexProximal: 'leftIndexProximal',
  LeftIndexIntermediate: 'leftIndexIntermediate',
  LeftIndexDistal: 'leftIndexDistal',
  LeftMiddleProximal: 'leftMiddleProximal',
  LeftMiddleIntermediate: 'leftMiddleIntermediate',
  LeftMiddleDistal: 'leftMiddleDistal',
  LeftRingProximal: 'leftRingProximal',
  LeftRingIntermediate: 'leftRingIntermediate',
  LeftRingDistal: 'leftRingDistal',
  LeftLittleProximal: 'leftLittleProximal',
  LeftLittleIntermediate: 'leftLittleIntermediate',
  LeftLittleDistal: 'leftLittleDistal',
  // Sağ el (aynı pattern, Right prefix)
  RightThumbProximal: 'rightThumbMetacarpal',
  RightThumbIntermediate: 'rightThumbProximal',
  RightThumbDistal: 'rightThumbDistal',
  RightIndexProximal: 'rightIndexProximal',
  RightIndexIntermediate: 'rightIndexIntermediate',
  RightIndexDistal: 'rightIndexDistal',
  RightMiddleProximal: 'rightMiddleProximal',
  RightMiddleIntermediate: 'rightMiddleIntermediate',
  RightMiddleDistal: 'rightMiddleDistal',
  RightRingProximal: 'rightRingProximal',
  RightRingIntermediate: 'rightRingIntermediate',
  RightRingDistal: 'rightRingDistal',
  RightLittleProximal: 'rightLittleProximal',
  RightLittleIntermediate: 'rightLittleIntermediate',
  RightLittleDistal: 'rightLittleDistal',
};
```

### 5.5 Day 4 — Yazılacak Fonksiyonlar (main.ts içine)

**1. `parseStoredFrame`** — flat 225 → yapılandırılmış landmark objeleri:
```typescript
const POSE_N = 33, HAND_N = 21;

function isHandDegenerate(pts: Array<{x:number,y:number,z:number}>): boolean {
  // El tespit edilememiş frame: extraction'da sıfır → omuz-merkezleme → -center
  // Tüm noktalar birbirine çok yakınsa dejenere say
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const rangeX = Math.max(...xs) - Math.min(...xs);
  const rangeY = Math.max(...ys) - Math.min(...ys);
  return rangeX < 0.05 && rangeY < 0.05;
}

function parseStoredFrame(flat: number[]) {
  const toLM = (start: number, n: number) =>
    Array.from({length: n}, (_, i) => ({
      x: flat[start + i*3],
      y: flat[start + i*3 + 1],
      z: flat[start + i*3 + 2],
      visibility: 1.0,          // ZORUNLU — yoksa Kalidokit dondurur
    }));

  return {
    pose:      toLM(0,   POSE_N),   // [0:99]
    leftHand:  toLM(99,  HAND_N),   // [99:162]
    rightHand: toLM(162, HAND_N),   // [162:225]
  };
}
```

**2. `rigRotation` / `rigPosition` helpers:**
```typescript
import * as THREE from 'three';

const rigRotation = (
  vrm: VRM, boneName: string,
  rotation = {x:0, y:0, z:0, rotationOrder:'XYZ'},
  dampener = 1, lerpAmount = 0.3
) => {
  const bone = vrm.humanoid.getNormalizedBoneNode(boneName as any);
  if (!bone) return;
  const euler = new THREE.Euler(
    rotation.x * dampener,
    rotation.y * dampener,
    rotation.z * dampener,
    (rotation as any).rotationOrder || 'XYZ'
  );
  bone.quaternion.slerp(new THREE.Quaternion().setFromEuler(euler), lerpAmount);
};

const rigPosition = (
  vrm: VRM, boneName: string,
  position = {x:0, y:0, z:0},
  dampener = 1, lerpAmount = 0.3
) => {
  const bone = vrm.humanoid.getNormalizedBoneNode(boneName as any);
  if (!bone) return;
  bone.position.lerp(
    new THREE.Vector3(position.x * dampener, position.y * dampener, position.z * dampener),
    lerpAmount
  );
};
```

**3. `applyRig`:**
```typescript
// Last good hand poses (degenerate frame'lerde tutulur)
let lastLeftHandRig: any = null;
let lastRightHandRig: any = null;

import * as Kalidokit from 'kalidokit';

function applyLandmarksToVRM(vrm: VRM, flat: number[]) {
  const { pose, leftHand, rightHand } = parseStoredFrame(flat);

  // Pose
  const riggedPose = Kalidokit.Pose.solve(pose as any, pose as any, {
    runtime: 'tfjs',
    imageSize: { width: 640, height: 480 },
    enableLegs: false,
  });

  // Sol el — dejenere ise son iyi pozu koru
  let riggedLeft = lastLeftHandRig;
  if (!isHandDegenerate(leftHand)) {
    riggedLeft = Kalidokit.Hand.solve(leftHand as any, 'Left');
    if (riggedLeft) lastLeftHandRig = riggedLeft;
  }

  // Sağ el
  let riggedRight = lastRightHandRig;
  if (!isHandDegenerate(rightHand)) {
    riggedRight = Kalidokit.Hand.solve(rightHand as any, 'Right');
    if (riggedRight) lastRightHandRig = riggedRight;
  }

  // Gövde
  if (riggedPose) {
    rigRotation(vrm, 'hips',         riggedPose.Hips.rotation, 0.7, 0.3);
    // Hips position — anchor et (drift önle)
    const hb = vrm.humanoid.getNormalizedBoneNode('hips' as any);
    if (hb) hb.position.set(0, 0, 0);   // sabit tut

    rigRotation(vrm, 'chest',        riggedPose.Spine, 0.25, 0.3);
    rigRotation(vrm, 'spine',        riggedPose.Spine, 0.45, 0.3);
    rigRotation(vrm, 'leftUpperArm', riggedPose.LeftUpperArm,  1.0, 0.4);
    rigRotation(vrm, 'leftLowerArm', riggedPose.LeftLowerArm,  1.0, 0.4);
    rigRotation(vrm, 'rightUpperArm',riggedPose.RightUpperArm, 1.0, 0.4);
    rigRotation(vrm, 'rightLowerArm',riggedPose.RightLowerArm, 1.0, 0.4);
  }

  // Sol el + parmaklar
  if (riggedLeft) {
    rigRotation(vrm, 'leftHand', {
      x: riggedLeft.LeftWrist.x,
      y: riggedLeft.LeftWrist.y,
      z: riggedPose?.LeftHand?.z ?? riggedLeft.LeftWrist.z,
    }, 1.0, 0.6);
    for (const [kKey, vKey] of Object.entries(FINGER_BONE_MAP)) {
      if (kKey.startsWith('Left') && riggedLeft[kKey]) {
        rigRotation(vrm, vKey, riggedLeft[kKey], 1.0, 0.7);
      }
    }
  }

  // Sağ el + parmaklar
  if (riggedRight) {
    rigRotation(vrm, 'rightHand', {
      x: riggedRight.RightWrist.x,
      y: riggedRight.RightWrist.y,
      z: riggedPose?.RightHand?.z ?? riggedRight.RightWrist.z,
    }, 1.0, 0.6);
    for (const [kKey, vKey] of Object.entries(FINGER_BONE_MAP)) {
      if (kKey.startsWith('Right') && riggedRight[kKey]) {
        rigRotation(vrm, vKey, riggedRight[kKey], 1.0, 0.7);
      }
    }
  }

  vrm.update(1/30);
}
```

### 5.6 Day 4 — Uygulama Sırası

1. **Kalidokit import'u ekle** (main.ts başına):
   ```typescript
   import * as Kalidokit from 'kalidokit';
   ```
2. **FINGER_BONE_MAP, parseStoredFrame, isHandDegenerate, rigRotation, rigPosition, applyRig** fonksiyonlarını main.ts'e ekle (§5.4-5.5).
3. **`applyLandmarksToVRM` stub'ını** (main.ts:192-203) gerçek implementasyonla değiştir.
4. **POC testi** (VRM dosyası olmadan da çalışır — placeholder'da değil, VRM yüklenince devreye girer):
   - `avatar.vrm` koyduktan sonra tek frame'i konsola bas, `riggedPose`/`riggedLeft` undefined mı diye kontrol et.
5. **Full sign test** → `merhaba` animasyonunu oynat → kollar/eller hareket ediyor mu?
6. **Day 5 GO/NO-GO** → parmaklar eklemleniyor mu, işaret okunuyor mu?

---

## 6. Day 5 GO/NO-GO Kapısı

| Soru | EVET → | HAYIR → |
|---|---|---|
| Gövde hareketi kabaca doğru mu? | Devam | Landmark kalitesini araştır |
| Parmaklar eklemleniyor mu (donmamış)? | Devam | **Plan B** |
| Sağır biri işareti tanır mı? | Devam | **Plan B** |
| Render ≥ 30 FPS? | Devam | Optimize (shadow/light azalt) |

**Plan B (artarak):**
1. Kalidokit `Hand.solve` bırak, MediaPipe→VRM parmak mapping'i elle yaz (+2 gün)
2. AUTSL landmark normalize, düşük-confidence frame'leri at (+1 gün)
3. pyrender + SMPL-X Python stack'e geç (+3 gün, browser avantajı gider)

---

## 7. Çalıştırma Komutları

**Backend (Anaconda Prompt #1):**
```bat
conda activate isaret_dili
cd C:\sign_language
python src\v2\server.py
```
→ `http://localhost:8000/docs` (Swagger)

**Frontend (Anaconda Prompt #2):**
```bat
cd C:\sign_language\tid-frontend
npm run dev
```
→ `http://localhost:5173`

**Test:**
```bat
conda activate isaret_dili
cd C:\sign_language
python src\v2\test_translate_full.py
```
→ `OK (shape 64,225) : 15/15`

---

## 8. Değişmez Kurallar (tartışmaya açma)

- Tüm çıktılar/kod/dosyalar **Ram İsmail** adına (asla "Rakan" yazma)
- Day 1-2 bitti — backend + frontend scaffold tekrar yazma
- Mimari kilitli: Three.js + VRM + Kalidokit · padded naming · MM PC primary · eller > yüz
- 6 demo cümlesi sabit: `Merhaba` · `Teşekkür ederim` · `Ben seni seviyorum` · `Anne evde yemek` · `İyi günler` · `Okula gidiyorum`
- Day 5 GO/NO-GO projenin kritik kapısı — bu karara takım + AI birlikte veriyor

---

## 9. Prior Art (IEEE Related Work için)

**Tier 1 — Kalidokit + three-vrm + işaret dili:** Proje YOK → novelty doğrulandı.

**En yakın benzer çalışmalar:**
- three.js forum: "Kalidokit pre-prepared list of coord" (#61291) — stored landmarks sorusu
- `kevinjosethomas/sign-language-processing` issue #5 — Kalidokit ASL için
- `europanite/webcam_to_avatar` — VRoid + Kalidokit (canlı, işaret değil)
- Adrien Lefebvre — SL AR game (Kalidokit + BiLSTM)

**Tier 2 — Alternatif paradigmalar:**
- SMPL-X: `ZhengdiYu/SignAvatars` (ECCV 2024)
- HamNoSys/SiGML: JASigning, CWASA (UEA Virtual Humans)
- Neural T2S: T2S-GPT, SignLLM (ICCV 2025), SignGPT

---

*Bu dosya Muhammet'in Claude oturumundan üretildi — Day 3 sonu (2 Haziran 2026)*
*Day 4'te güncellenecek.*
