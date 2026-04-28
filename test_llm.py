"""
LLM Hizli Test - Qwen2.5-7B yukleme + ornek gloss -> Turkce cevirisi
"""
import os
import time
from pathlib import Path

# Windows'ta CUDA DLL'lerini bulabilmesi icin PyTorch'un lib klasorunu ekle
# (llama-cpp-python CUDA wheel kendi cudart/cublas getirmez, PyTorch'unkini kullanir)
import torch as _torch
_torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
if os.name == "nt" and os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)

MODEL_PATH = Path("models/llm/qwen2.5-7b-instruct-q4_k_m.gguf")
# Split GGUF (multi-part) destekleme: tek dosya yoksa ilk parcayi kullan
if not MODEL_PATH.exists():
    SPLIT_PATH = Path("models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
    if SPLIT_PATH.exists():
        MODEL_PATH = SPLIT_PATH

print("=" * 60)
print("  LLM Hizli Test")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"[HATA] Model bulunamadi: {MODEL_PATH}")
    print("Once download_llm.bat calistir.")
    exit(1)

size_gb = MODEL_PATH.stat().st_size / 1e9
print(f"\n[1] Model: {MODEL_PATH.name} ({size_gb:.2f} GB)")

print("\n[2] llama-cpp-python yukleniyor...")
from llama_cpp import Llama

print("\n[3] Model yukleniyor (GPU'ya offload)...")
t0 = time.time()
llm = Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=512,
    verbose=False,
)
print(f"    [OK] Yukleme: {time.time()-t0:.1f} sn")

# Gercek AUTSL sinif isimlerinden ornek gloss dizileri
print("\n[4] Ornek gloss -> Turkce:")
test_cases = [
    ["ben", "su", "icmek"],
    ["nerede", "tuvalet"],
    ["sen", "nasil"],
    ["tesekkur", "ederim"],
    ["yarin", "okul", "gitmek"],
]

system_prompt = """Sen bir Turk Isaret Dili (TID) cevirmenisin.
Verilen TID gloss dizisini dogal ve dilbilgisi acisindan dogru Turkce cumleye cevir.

Ornekler:
ben ev gitmek -> Eve gidiyorum.
sen isim ne -> Adin ne?
ben okul sevmek -> Okulu seviyorum.
dun yagmur yagmak -> Dun yagmur yagdi.
nerede hastane -> Hastane nerede?

Sadece Turkce cumleyi ver, aciklama yapma."""

for gloss_list in test_cases:
    gloss_str = " ".join(gloss_list)
    prompt = f"{system_prompt}\n\n{gloss_str} ->"
    t0 = time.time()
    out = llm(prompt, max_tokens=50, temperature=0.3, stop=["\n"])
    ms = (time.time() - t0) * 1000
    text = out["choices"][0]["text"].strip()
    print(f"    {gloss_str:30s} -> {text}  [{ms:.0f}ms]")

# Hiz testi
print("\n[5] Hiz testi:")
t0 = time.time()
out = llm("Turkce bir cumle yaz:", max_tokens=100, temperature=0.7)
elapsed = time.time() - t0
tokens = out["usage"]["completion_tokens"]
tps = tokens / elapsed
print(f"    {tokens} token / {elapsed:.1f}sn = {tps:.1f} token/sn")

if tps < 10:
    print(f"    [UYARI] Yavas - GPU offload calismiyor olabilir")
elif tps < 25:
    print(f"    [OK] Kabul edilebilir")
else:
    print(f"    [OK] Mukemmel - GPU tam calisiyor")

print("\n" + "=" * 60)
print("  LLM CALISIYOR!")
print("=" * 60)
