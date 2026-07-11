# TİD — Bidirectional Turkish Sign Language Communication Platform

**Team:**
- Ram İsmail
- Muhammet Ay

---

## 1. Project Overview

A platform that bridges Turkish Sign Language (TİD) and written Turkish **in both directions**:

- **Recognition (sign → Turkish text):** Body/hand landmarks are extracted from the camera with MediaPipe Holistic and classified by the skeleton-based graph convolutional network **TMS-Net** (AUTSL, 226 isolated signs, **94.70%** validation accuracy; **95.13%** with the TMS-Net + SML ensemble). Predicted sign glosses are converted into fluent Turkish sentences by a locally-run **Qwen2.5-7B** language model.
- **Synthesis (Turkish text → sign):** Turkish text is mapped through light morphological normalization onto a 226-word dictionary of reference landmark sequences and rendered in the browser by a **faithful skeleton ("Cin Ali") player** in real time (30 fps).
- **Self-study (tutor):** A reference skeleton and the student's live camera feed side by side; **DTW (Dynamic Time Warping)** provides a similarity score.

Both directions operate on a shared **skeleton representation** (64 frames × 225 values = 33 pose + 21 left hand + 21 right hand).

> **Status:** Final submission complete — this chapter of the project is closed. Design decisions, constraints, and negative results (why rigged-avatar retargeting was abandoned in favor of faithful landmark playback) are documented in the IEEE report.

---

## 2. Requirements

- **Python** (conda environment `isaret_dili`) — PyTorch (CUDA), MediaPipe, FastAPI, llama-cpp-python, dtw-python, numpy
- **Node.js** 18+ (for the frontend)
- **GPU** (recommended): CUDA-capable NVIDIA (development: RTX 4060 Laptop; training: RTX 5060 Ti)
- **Model weights / data** (large; not included in the repo):
  - `checkpoints/best.pth` — TMS-Net (94.70%)
  - `models/llm/qwen2.5-7b-instruct-q4_k_m-*.gguf` — Qwen2.5-7B (2 parts)
    Download: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
  - `reference_landmarks/*.npy` — 678 references (226 classes × 3)
  - `class_map.json` — 226-class mapping

### Installation

# Python environment
conda env create -f environment.yml        # or: pip install -r requirements.txt
conda activate isaret_dili

# Frontend dependencies
cd tid-frontend
npm install
cd ..

---

## 3. Running

### A) Synthesis — Turkish text → avatar (web)

Two terminals are required.

**Terminal 1 — Backend (FastAPI):**

conda activate isaret_dili
cd C:\sign_language
uvicorn src.v2.server:app --reload --host 0.0.0.0 --port 8000
# You should see "Loaded 226 signs" on startup.

**Terminal 2 — Frontend (Vite):**

cd C:\sign_language\tid-frontend
npm run dev
# Browser: http://localhost:5173

Click a preset sentence and press **▶ Çevir & Oynat** (Translate & Play). Live tuning from the browser console:
`TID.mode = 'skeleton' | 'ik' | 'figure'`, `SKEL.setOpts({ scale, yOffset, zScale })`.

### B) Recognition + Tutor (camera)

A single terminal is enough. Before each command:

conda activate isaret_dili
cd C:\sign_language

**Word tutor (split-screen reference + DTW score):**

python src\v1\study_autsl_tutor.py --class_map class_map.json ^
  --reference_dir reference_landmarks ^
  --llm_model models\llm\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --camera 0

**Sentence tutor (theme-based):**

python src\v1\sentence_tutor.py --class_map class_map.json ^
  --reference_dir reference_landmarks ^
  --llm_model models\llm\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --camera 0 --theme aile

**Live recognition (camera → Turkish sentence):**

python src\v1\inference_tmsnet_llm.py --checkpoint checkpoints\best.pth ^
  --class_map class_map.json ^
  --llm_model models\llm\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --camera 0

> `--camera 0` is the built-in camera; try `1` for an external one. In PowerShell, replace the `^` line continuation with a backtick (`` ` ``); the commands above work as-is in cmd/Anaconda Prompt.

---

## 4. Repository Structure

sign_language/
├── src/
│   ├── v1/                     # Recognition + tutor (PyTorch, MediaPipe, DTW, LLM)
│   │   ├── tmsnet_model.py     #   TMS-Net (6 streams) — deployed model
│   │   ├── sml_model.py        #   SML (3 streams)
│   │   ├── stgcn_model.py      #   ST-GCN (baseline)
│   │   ├── graph.py            #   56-node skeleton graph
│   │   ├── inference_tmsnet_llm.py   # live recognition → Turkish
│   │   ├── study_autsl_tutor.py      # word tutor (DTW)
│   │   └── sentence_tutor.py         # sentence tutor
│   └── v2/                     # Synthesis (text → sign)
│       ├── server.py           #   FastAPI backend
│       ├── text_to_sign.py     #   text → landmark pipeline
│       ├── sign_dictionary.json#   226-word dictionary
│       └── expand_dictionary.py#   dictionary generation + quality scan
├── tid-frontend/               # Vite + TypeScript + Three.js frontend
│   └── src/{main,skeleton,figure,armIK}.ts
├── checkpoints/                # model weights (best.pth = TMS-Net)
├── models/llm/                 # Qwen2.5-7B GGUF
├── reference_landmarks/        # 678 .npy references
├── class_map.json              # 226 classes
└── docs/                       # decisions, quality report, IEEE report builder

---

## 5. Results (summary)

| Model | Streams | Acc. (AUTSL val, 226 classes) |
|---|---|---|
| **TMS-Net + SML Ensemble** | 6 + 3 | **95.13%** (best) |
| TMS-Net | 6 | 94.70% (deployed) |
| SML | 3 | 93.39% |
| ST-GCN | baseline | 89.04% |

- Synthesis dictionary quality scan: **216 of 226** words play back cleanly.
- The faithful skeleton player uses no retargeting → every sign is reproduced exactly, at 30 fps.
- An honest negative result: both angle-based (Kalidokit) and analytic position-based IK retargeting onto a rigged VRM avatar failed to preserve spatial fidelity (the *location* parameter, which is phonological in sign language). Faithful playback was chosen over embodiment — consistent with the literature, where legibility matters more than photorealism.

See the IEEE report for details (`TID_Final_Report_EN.docx` / `TID_Final_Raporu_TR.docx`).
