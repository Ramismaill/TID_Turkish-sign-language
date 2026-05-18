"""
llm_translator.py (v3)
Translates TID gloss list into a Turkish sentence.
Uses Qwen2.5-7B-Instruct GGUF model via llama-cpp-python.

v3 changes:
- n_threads parameter (avoid thread fighting in integrated demo)

v2 features (kept):
- 14 diverse few-shot examples
- Lowercase normalization
- Output cleanup (strips parentheticals, explanations)
- Optional history logging to file
- warmup() method
"""

import re
import time
from datetime import datetime
from pathlib import Path
from llama_cpp import Llama


SYSTEM_PROMPT = """Sen bir Turk Isaret Dili (TID) cevirmenisin.
Verilen TID kelimelerini (gloss) dogal, akici bir Turkce cumleye cevirirsin.
TID gramerinde fiiller cumlenin sonunda olur ve zamirler dusebilir.

KURALLAR:
- Sadece tek bir Turkce cumle uret.
- Aciklama, parantez veya not yazma.
- Ingilizce kullanma.
- Cumle bir nokta veya soru isareti ile bitsin.
- Eger sadece bir tek kelime verilirse, onu kibar bir cumle yap."""


FEW_SHOT_EXAMPLES = [
    {"gloss": "ben su icmek",       "tr": "Su iciyorum."},
    {"gloss": "nerede tuvalet",     "tr": "Tuvalet nerede?"},
    {"gloss": "tesekkur etmek",     "tr": "Tesekkur ederim."},
    {"gloss": "ben okul gitmek",    "tr": "Okula gidiyorum."},
    {"gloss": "nasil sen",          "tr": "Nasilsin?"},
    {"gloss": "ben ad rakan",       "tr": "Adim Rakan."},
    {"gloss": "yardim sen ben",     "tr": "Bana yardim eder misin?"},
    {"gloss": "merhaba",            "tr": "Merhaba!"},
    {"gloss": "evet",               "tr": "Evet."},
    {"gloss": "hayir",               "tr": "Hayir."},
    {"gloss": "ben aclik",          "tr": "Acim."},
    {"gloss": "saat kac",           "tr": "Saat kac?"},
    {"gloss": "anne ben sevmek",    "tr": "Annemi seviyorum."},
    {"gloss": "ben mutlu",          "tr": "Mutluyum."},
]


class GlossToTurkish:
    """Translates a TID gloss list into a fluent Turkish sentence."""

    def __init__(self, model_path, n_gpu_layers=0, n_ctx=2048, verbose=False,
                 history_log=None, n_threads=None):
        """
        Args:
            model_path: Path to GGUF file (first part if multi-part)
            n_gpu_layers: Layers to offload to GPU (0 = CPU only)
            n_ctx: Context window size
            verbose: Detailed logging
            history_log: Path to save translation history
            n_threads: CPU threads for LLM (None = auto/all cores)
                       Use 4 for integrated demo to leave cores for webcam loop
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"[LLM] Loading model: {path.name}")
        if n_threads:
            print(f"[LLM] Using {n_threads} CPU threads")
        t0 = time.time()

        kwargs = dict(
            model_path=str(path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        if n_threads:
            kwargs['n_threads'] = n_threads

        self.llm = Llama(**kwargs)
        print(f"[LLM] Loaded in {time.time() - t0:.1f}s")

        self.history_log = history_log
        if self.history_log:
            log_path = Path(self.history_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n=== Session start: {datetime.now().isoformat()} ===\n")

    def warmup(self):
        """Run a dummy translation to load KV cache. Call after __init__."""
        print("[LLM] Warming up...")
        t0 = time.time()
        self.translate(["test"])
        print(f"[LLM] Warmup done in {time.time() - t0:.1f}s")

    def _build_prompt(self, gloss_list):
        """Build a few-shot prompt with normalized input."""
        gloss_str = " ".join(g.strip().lower() for g in gloss_list)

        examples_block = "\n\n".join([
            f"Isaretler: {ex['gloss']}\nCumle: {ex['tr']}"
            for ex in FEW_SHOT_EXAMPLES
        ])

        prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{examples_block}

Isaretler: {gloss_str}
Cumle:<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def _clean_output(self, text):
        """Clean LLM output."""
        text = text.split("\n")[0].strip()
        text = re.sub(r"^[Cc]umle\s*:\s*", "", text)
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip('"\'`')
        return text

    def translate(self, gloss_list, max_tokens=40, temperature=0.3):
        """Translate gloss list to Turkish sentence."""
        if not gloss_list:
            return "", 0.0

        prompt = self._build_prompt(gloss_list)
        t0 = time.time()
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "\n\n", "Isaretler:"],
            echo=False,
        )
        elapsed = time.time() - t0

        raw = out["choices"][0]["text"]
        sentence = self._clean_output(raw)

        if self.history_log and sentence:
            try:
                with open(self.history_log, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    f.write(f"[{timestamp}] {gloss_list} -> {sentence} ({elapsed:.2f}s)\n")
            except Exception:
                pass

        return sentence, elapsed


if __name__ == "__main__":
    MODEL_PATH = "models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

    translator = GlossToTurkish(MODEL_PATH, history_log="logs/translations.log")
    translator.warmup()

    test_cases = [
        ["ben", "su", "icmek"],
        ["nerede", "tuvalet"],
        ["abla", "okul", "gitmek"],
        ["nasil", "sen"],
        ["ben", "kitap", "okumak"],
        ["yardim", "lutfen"],
        ["merhaba"],
        ["ben", "anne", "sevmek"],
        ["hayir", "sen"],
        ["ben", "yorgun"],
    ]

    print("\n" + "=" * 60)
    print("TID Gloss -> Turkish Sentence Translation Test (v3)")
    print("=" * 60)

    total_time = 0
    for i, gloss in enumerate(test_cases, 1):
        sentence, elapsed = translator.translate(gloss)
        total_time += elapsed
        print(f"\n[{i}] Gloss: {gloss}")
        print(f"    Sentence: {sentence}")
        print(f"    Time: {elapsed:.2f}s")

    print(f"\n{'=' * 60}")
    print(f"Total: {total_time:.1f}s for {len(test_cases)} sentences = "
          f"avg {total_time/len(test_cases):.1f}s/sentence")
    print(f"History saved to: logs/translations.log")
    print("=" * 60)
