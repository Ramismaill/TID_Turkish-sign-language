# TİD — Çift Yönlü Türk İşaret Dili İletişim Platformu

**Ders:** FET306 — Uygulamalı Yapay Sinir Ağları, İstanbul Topkapı Üniversitesi
**Grup üyeleri:**
- Ram İsmail — 24040301052
- Muhammet Ay — 23040301137

---

## 1. Proje Özeti

Türk İşaret Dili (TİD) ile yazılı Türkçe arasında **iki yönde** köprü kuran bir platform:

- **Tanıma (işaret → Türkçe metin):** Kameradan MediaPipe Holistic ile çıkarılan beden/el
  landmark'ları, iskelet tabanlı çizge evrişimli ağ **TMS-Net** ile sınıflandırılır
  (AUTSL, 226 izole işaret, **%94.70** doğrulama doğruluğu). Tahmin edilen işaret gloss'ları,
  yerel **Qwen2.5-7B** dil modeliyle akıcı Türkçe cümleye çevrilir.
- **Sentez (Türkçe metin → işaret):** Türkçe metin, hafif biçimbilimsel normalleştirme ile
  226 kelimelik sözlük üzerinden referans landmark dizilerine eşlenir ve tarayıcıda
  **sadık iskelet ("Cin Ali") oynatıcısı** ile gerçek zamanlı (30 fps) görüntülenir.
- **Öz-çalışma (öğretici):** Referans iskelet ile öğrencinin canlı görüntüsü yan yana;
  **DTW (Dinamik Zaman Bükümlemesi)** ile benzerlik skoru verilir.

İki taraf da ortak bir **iskelet temsili** (64 kare × 225 değer = 33 poz + 21 sol el + 21 sağ el) üzerinde çalışır.

---

## 2. Gereksinimler

- **Python** (conda ortamı `isaret_dili`) — PyTorch (CUDA), MediaPipe, FastAPI, llama-cpp-python, dtw-python, numpy
- **Node.js** 18+ (frontend için)
- **GPU** (önerilen): CUDA destekli NVIDIA (geliştirme: RTX 4060 Laptop; eğitim: RTX 5060 Ti)
- **Model ağırlıkları / veri** (büyük; depo dışı):
  - `checkpoints/best.pth` — TMS-Net (%94.70)
  - `models/llm/qwen2.5-7b-instruct-q4_k_m-*.gguf` — Qwen2.5-7B (2 parça)
    İndir: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
  - `reference_landmarks/*.npy` — 678 referans (226 sınıf × 3)
  - `class_map.json` — 226 sınıf eşlemesi

### Kurulum

```bash
# Python ortamı
conda env create -f environment.yml        # veya: pip install -r requirements.txt
conda activate isaret_dili

# Frontend bağımlılıkları
cd tid-frontend
npm install
cd ..
```

---

## 3. Çalıştırma

### A) Sentez — Türkçe metin → avatar (web)

İki terminal gerekir.

**Terminal 1 — Backend (FastAPI):**
```bash
conda activate isaret_dili
cd C:\sign_language
uvicorn src.v2.server:app --reload --host 0.0.0.0 --port 8000
# Açılışta "Loaded 226 signs" görmelisiniz.
```

**Terminal 2 — Frontend (Vite):**
```bash
cd C:\sign_language\tid-frontend
npm run dev
# Tarayıcı: http://localhost:5173
```

Bir hazır cümleye tıklayıp **▶ Çevir & Oynat** deyin. Tarayıcı konsolundan canlı ayar:
`TID.mode = 'skeleton' | 'ik' | 'figure'`, `SKEL.setOpts({ scale, yOffset, zScale })`.

### B) Tanıma + Öğretici (kamera)

Tek terminal yeter. Hepsinden önce:
```bash
conda activate isaret_dili
cd C:\sign_language
```

**Kelime öğretici (split-screen referans + DTW skor):**
```bash
python src\v1\study_autsl_tutor.py --class_map class_map.json ^
  --reference_dir reference_landmarks ^
  --llm_model models\llm\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --camera 0
```

**Cümle öğretici (tema bazlı):**
```bash
python src\v1\sentence_tutor.py --class_map class_map.json ^
  --reference_dir reference_landmarks ^
  --llm_model models\llm\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --camera 0 --theme aile
```

**Canlı tanıma (kamera → Türkçe cümle):**
```bash
python src\v1\inference_tmsnet_llm.py --checkpoint checkpoints\best.pth ^
  --class_map class_map.json ^
  --llm_model models\llm\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --camera 0
```

> `--camera 0` dahili kamera; harici için `1` deneyin. PowerShell'de satır birleştirici `^`
> yerine backtick (`` ` ``) kullanın; cmd/Anaconda Prompt'ta yukarıdaki haliyle çalışır.

---

## 4. Klasör Yapısı

```
sign_language/
├── src/
│   ├── v1/                     # Tanıma + öğretici (PyTorch, MediaPipe, DTW, LLM)
│   │   ├── tmsnet_model.py     #   TMS-Net (6 akış) — dağıtılan model
│   │   ├── sml_model.py        #   SML (3 akış)
│   │   ├── stgcn_model.py      #   ST-GCN (temel)
│   │   ├── graph.py            #   56-düğüm iskelet çizgesi
│   │   ├── inference_tmsnet_llm.py   # canlı tanıma → Türkçe
│   │   ├── study_autsl_tutor.py      # kelime öğretici (DTW)
│   │   └── sentence_tutor.py         # cümle öğretici
│   └── v2/                     # Sentez (metin → işaret)
│       ├── server.py           #   FastAPI backend
│       ├── text_to_sign.py     #   metin → landmark hattı
│       ├── sign_dictionary.json#   226 kelimelik sözlük
│       └── expand_dictionary.py#   sözlük üretimi + kalite taraması
├── tid-frontend/               # Vite + TypeScript + Three.js ön yüz
│   └── src/{main,skeleton,figure,armIK}.ts
├── checkpoints/                # model ağırlıkları (best.pth = TMS-Net)
├── models/llm/                 # Qwen2.5-7B GGUF
├── reference_landmarks/        # 678 .npy referans
├── class_map.json              # 226 sınıf
└── docs/                       # kararlar, kalite raporu, IEEE rapor üreticisi
```

---

## 5. Sonuçlar (özet)

| Model | Akış | Doğr. (AUTSL val, 226 sınıf) |
|---|---|---|
| **TMS-Net** | 6 | **%94.70** (dağıtılan) |
| SML | 3 | %93.39 |
| ST-GCN | temel | %89.04 |

- Sentez sözlüğü kalite taraması: 226 kelimenin **216'sı** sorunsuz oynatılabilir.
- Sadık iskelet oynatıcısı retargeting içermez → her işaret birebir, 30 fps.

Ayrıntılar için IEEE raporuna bakın (`TID_Final_Report_EN.docx` / `TID_Final_Raporu_TR.docx`).
