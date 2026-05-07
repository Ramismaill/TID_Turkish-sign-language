# Turkish Sign Language Recognition & Education Platform (TİD)

Real-time sign language recognition and interactive learning system powered by deep learning and LLM coaching.

**Authors:** Ram Ismail · Muhammet Ay  
**University:** İstanbul Topkapı Üniversitesi — AI/ML Engineering  
**Dataset:** AUTSL (226 Turkish sign classes, 36,302 samples, 43 signers)

---

## Best Result: 95.13% Accuracy
TMS-Net + SML Weighted Ensemble on AUTSL validation set.

---

## What This Project Does

A camera captures hand and body movements. MediaPipe extracts a 56-node skeleton. A deep learning model classifies the sign in real time. A local LLM (Qwen2.5-7B) converts the recognized glosses into natural Turkish sentences.

On top of the recognition pipeline, we built a sign language education platform where users practice individual signs and full sentences — guided by skeleton references, DTW similarity scoring, and Turkish coaching feedback.

---

## Model Results

| Model | Accuracy |
|-------|----------|
| CNN + GRU | ~80% |
| Transformer | 85.85% |
| ST-GCN | 91.33% |
| SML | 93.39% |
| TMS-Net | 94.70% |
| TMS-Net + SML Ensemble | **95.13%** |

---

## TMS-Net Architecture

```
Input (64 frames × 56 nodes × 3 coords)
        │
        ├── Stream 1: Joint coordinates
        ├── Stream 2: Bone vectors
        ├── Stream 3: Joint motion (velocity)
        ├── Stream 4: Bone motion
        ├── Stream 5: Joint angles
        └── Stream 6: Angle motion
                │
    Multi-Scale Temporal Convolution (3 / 7 / 13 frame kernels)
                │
    Cross-Stream Attention Fusion
                │
    Classification → 226 Turkish sign classes
```

---

## Education Platform

### Word Tutor (v7)
Practice individual signs with real-time feedback:
- Left panel: skeleton reference animation for the target sign
- Right panel: your live webcam feed with skeleton overlay
- DTW similarity scoring (good ≤ 1.5, bad ≥ 4.0)
- Motion gate to detect if you actually signed (std threshold: 0.012)
- LLM coaching feedback in Turkish

```bash
python src/study_autsl_tutor.py \
  --class_map class_map.json \
  --reference_dir reference_landmarks \
  --llm_model models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --camera 0 \
  --start_word acele
```

Keys: `SPACE` start · `N` next word · `B` previous · `R` retry · `Q` quit

### Sentence Tutor (v4)
Practice full Turkish sentences theme by theme:
- Enter a theme (yemek, aile, okul...)
- LLM generates a sentence using only known vocabulary
- Practice each word in sequence, 3 attempts per word
- Composite score + Turkish feedback at the end

```bash
python src/sentence_tutor.py \
  --class_map class_map.json \
  --reference_dir reference_landmarks \
  --llm_model models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --camera 0 \
  --theme yemek
```

### Live Translation
Sign continuously — recognized glosses are converted to Turkish sentences in real time.

```bash
python src/inference_tmsnet_llm.py \
  --checkpoint checkpoints/best.pth \
  --class_map class_map.json \
  --llm_model models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --camera 0
```

---

## Installation

```bash
conda create -n AUTSL python=3.10 -y
conda activate AUTSL

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mediapipe==0.10.11 opencv-python==4.10.0.84 numpy==1.26.4
pip install einops dtw-python sacrebleu sentencepiece gradio
pip install llama-cpp-python==0.2.90 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

LLM model (download separately):
- Model: Qwen2.5-7B-Instruct-Q4_K_M
- Source: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
- Place at: `models/llm/`

```bash
python verify_setup.py
```

---

## Repository Structure

```
├── src/
│   ├── tmsnet_model.py          # TMS-Net (6 streams, multi-scale)
│   ├── sml_model.py             # SML model
│   ├── stgcn_model.py           # ST-GCN model
│   ├── inference_tmsnet_llm.py  # Live translation demo
│   ├── study_autsl_tutor.py     # Word tutor v7
│   ├── sentence_tutor.py        # Sentence tutor v4
│   ├── llm_translator.py        # Qwen gloss → Turkish
│   ├── graph.py                 # 56-node skeleton graph
│   └── ...
├── reference_landmarks/         # 678 reference .npy files (226 classes × 3 samples)
├── class_map.json               # 226 Turkish sign names
├── requirements.txt
└── verify_setup.py
```

Model checkpoints (`*.pth`) and LLM weights (`*.gguf`) are not included due to size. See installation above.

---

## Roadmap

- Avatar animation renderer — replace the skeleton with an animated character for clearer sign demonstration
- Mobile application — on-device tutoring with the full pipeline

---

## References

- AUTSL Dataset: Sincan & Keles, 2020
- TMS-Net: Deng et al., Neurocomputing, vol. 572, 2024
- ST-GCN: Yan et al., AAAI 2018
- MediaPipe: Google, 2020
- Qwen2.5: Alibaba Cloud, 2024
