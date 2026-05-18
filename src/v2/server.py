"""
src/v2/server.py — FastAPI backend for TID Text-to-Sign Avatar Platform

Day 2 skeleton (initial routes for /health, /vocab, /sign/{word}, /translate)
Extended Day 9+ when full LLM gloss generation is wired in.

Run from project root:
    cd C:\\sign_language
    uvicorn src.v2.server:app --reload --host 0.0.0.0 --port 8000

Then open:
    http://localhost:8000/health         → status check
    http://localhost:8000/docs           → auto-generated Swagger UI
    http://localhost:8000/vocab          → list all 15 known words
    http://localhost:8000/sign/merhaba   → landmark sequence for one word

Author: Ram Ismail, Muhammet Ay
Date: 2026-05-18
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import Optional, List

# ============================================================
# PATH SETUP — must come before local imports
# ============================================================

SCRIPT_DIR   = Path(__file__).resolve().parent          # C:\sign_language\src\v2
PROJECT_ROOT = SCRIPT_DIR.parent.parent                 # C:\sign_language
SRC_DIR      = PROJECT_ROOT / "src"                     # C:\sign_language\src

sys.path.insert(0, str(SRC_DIR))

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from v2.text_to_sign import TextToSignPipeline

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("tid-server")

# ============================================================
# CONFIG
# ============================================================

REFERENCE_LANDMARKS_DIR = PROJECT_ROOT / "reference_landmarks"
DICTIONARY_PATH         = SCRIPT_DIR / "sign_dictionary.json"

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# ============================================================
# NAMING FORMAT AUTO-DETECT
# ============================================================

def detect_naming_format(ref_dir: Path) -> str:
    if not ref_dir.exists():
        log.error(f"Reference dir does not exist: {ref_dir}")
        return 'unknown'
    files = list(ref_dir.glob("cls*.npy"))
    if not files:
        log.error(f"No cls*.npy files found in {ref_dir}")
        return 'unknown'
    for f in files:
        name = f.name
        if re.match(r'cls0\d+_\d+\.npy', name):
            return 'padded'
        m = re.match(r'cls(\d+)_\d+\.npy', name)
        if m and int(m.group(1)) < 100:
            return 'unpadded'
    return 'unknown'


NAMING_FORMAT = detect_naming_format(REFERENCE_LANDMARKS_DIR)
log.info(f"Detected naming format: {NAMING_FORMAT}")


def class_id_to_filename(cls_id: int, sample: int = 1) -> str:
    if NAMING_FORMAT == 'padded':
        return f"cls{cls_id:03d}_{sample}.npy"
    return f"cls{cls_id}_{sample}.npy"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="TID Text-to-Sign API",
    description="Backend for Turkish text → sign language avatar platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[TextToSignPipeline] = None


@app.on_event("startup")
async def startup_event():
    global pipeline
    log.info("Loading TextToSignPipeline...")
    try:
        pipeline = TextToSignPipeline(
            dictionary_path=str(DICTIONARY_PATH),
            reference_dir=str(REFERENCE_LANDMARKS_DIR),
        )
        log.info(f"Pipeline loaded. {len(pipeline.dictionary) - 1} signs available.")
    except Exception as e:
        log.exception("Pipeline load failed")


# ============================================================
# RESPONSE MODELS
# ============================================================

class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    naming_format: str
    project_root: str
    reference_landmarks_dir: str
    vocab_size: int


class VocabResponse(BaseModel):
    count: int
    words: List[str]


class SignResponse(BaseModel):
    word: str
    input_word: str
    frame_count: int
    fps: int
    duration_ms: int
    landmarks: List[List[float]]


class TranslateRequest(BaseModel):
    text: str = Field(..., description="Turkish input sentence", min_length=1, max_length=500)


class TranslateResponse(BaseModel):
    input: str
    total_signs: int
    unknown_words: List[str]
    signs: List[SignResponse]


# ============================================================
# ROUTES
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if pipeline is not None else "degraded",
        pipeline_loaded=pipeline is not None,
        naming_format=NAMING_FORMAT,
        project_root=str(PROJECT_ROOT),
        reference_landmarks_dir=str(REFERENCE_LANDMARKS_DIR),
        vocab_size=(len(pipeline.dictionary) - 1) if pipeline else 0,
    )


@app.get("/vocab", response_model=VocabResponse)
async def vocab():
    if pipeline is None:
        raise HTTPException(503, "Pipeline not loaded — check /health for diagnostics")
    words = [k for k in pipeline.dictionary.keys() if not k.startswith('_')]
    return VocabResponse(count=len(words), words=sorted(words))


@app.get("/sign/{word}", response_model=SignResponse)
async def get_sign(word: str):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not loaded")
    canonical = pipeline._lookup_canonical(word)
    if canonical is None:
        raise HTTPException(404, f"Word '{word}' not in vocabulary")
    landmarks = pipeline._load_landmarks(canonical)
    if landmarks is None:
        raise HTTPException(500, f"Reference landmarks missing for '{canonical}'")
    entry = pipeline.dictionary[canonical]
    return SignResponse(
        word=canonical,
        input_word=word,
        frame_count=int(landmarks.shape[0]),
        fps=30,
        duration_ms=entry.get("duration_ms", 1500),
        landmarks=landmarks.tolist(),
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not loaded")
    sequence = pipeline.translate(req.text)
    signs: List[SignResponse] = []
    unknown: List[str] = []
    for word, landmarks in sequence:
        if landmarks is None:
            unknown.append(word)
            continue
        entry = pipeline.dictionary.get(word, {})
        signs.append(SignResponse(
            word=word,
            input_word=word,
            frame_count=int(landmarks.shape[0]),
            fps=30,
            duration_ms=entry.get("duration_ms", 1500),
            landmarks=landmarks.tolist(),
        ))
    return TranslateResponse(
        input=req.text,
        total_signs=len(signs),
        unknown_words=unknown,
        signs=signs,
    )


# ============================================================
# DEV ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
