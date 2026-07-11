# -*- coding: utf-8 -*-
"""
src/v2/expand_dictionary.py — sign_dictionary.json: 15 → 226 kelime.

class_map.json'daki TÜM AUTSL sınıflarından otomatik sözlük girdisi üretir:
  - ASCII isim → düzgün Türkçe görünüm ("tesekkur" → "teşekkür", "agac" → "ağaç")
  - Eldeki 15 EL YAZIMI girdiye DOKUNMAZ (class_idx'leri atlanır)
  - reference_landmarks/ içinde gerçekten var olan dosyaları bağlar
  - Her kelimenin İLK referansını kalite taramasından geçirir
    (hareket miktarı + dejenere-el oranı + NaN) → docs/dictionary_quality_report.md

Çalıştır:
    cd C:\\sign_language
    python src/v2/expand_dictionary.py

Authors: Ram Ismail, Muhammet Ay
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CLASS_MAP    = PROJECT_ROOT / "class_map.json"
DICT_PATH    = SCRIPT_DIR / "sign_dictionary.json"
REF_DIR      = PROJECT_ROOT / "reference_landmarks"
REPORT_PATH  = PROJECT_ROOT / "docs" / "dictionary_quality_report.md"

DEFAULT_DURATION_MS = 1400

# ── ASCII → Türkçe görünüm düzeltmeleri ────────────────────────────────────────
# Sadece düzeltme gerekenler; listede olmayanlar için isim aynen (alt çizgi → boşluk).
TURKISH = {
    "acikmak": "acıkmak", "agabey": "ağabey", "agac": "ağaç", "agir": "ağır",
    "aglamak": "ağlamak", "akilli": "akıllı", "akilsiz": "akılsız",
    "alisveris": "alışveriş", "arkadas": "arkadaş", "ataturk": "atatürk",
    "ayakkabi": "ayakkabı", "ayni": "aynı", "bahce": "bahçe",
    "calismak": "çalışmak", "carsamba": "çarşamba", "catal": "çatal",
    "cay": "çay", "caydanlik": "çaydanlık", "cekic": "çekiç",
    "cirkin": "çirkin", "cocuk": "çocuk", "corba": "çorba", "cuzdan": "cüzdan",
    "degistirmek": "değiştirmek", "dugun": "düğün", "dun": "dün",
    "dusman": "düşman", "fotograf": "fotoğraf", "gecmis": "geçmiş",
    "gecmis_olsun": "geçmiş olsun", "gomlek": "gömlek", "gormek": "görmek",
    "gostermek": "göstermek", "gulmek": "gülmek", "hakli": "haklı",
    "hali": "halı", "hayirli_olsun": "hayırlı olsun", "hic": "hiç",
    "hoscakal": "hoşça kal", "igne": "iğne", "ilac": "ilaç", "isik": "ışık",
    "kacmak": "kaçmak", "kahvalti": "kahvaltı", "kapi": "kapı",
    "kardes": "kardeş", "kavsak": "kavşak", "keske": "keşke",
    "kiyma": "kıyma", "kiz": "kız", "komur": "kömür", "kopek": "köpek",
    "kopru": "köprü", "maas": "maaş", "masallah": "maşallah",
    "mudur": "müdür", "nasil": "nasıl", "ogretmen": "öğretmen",
    "oruc": "oruç", "ozur_dilemek": "özür dilemek", "pastirma": "pastırma",
    "persembe": "perşembe", "salca": "salça", "sali": "salı",
    "sampiyon": "şampiyon", "sapka": "şapka", "savas": "savaş",
    "seker": "şeker", "semsiye": "şemsiye", "seytan": "şeytan",
    "soylemek": "söylemek", "soz": "söz", "sut": "süt", "tatli": "tatlı",
    "turkiye": "türkiye", "uzgun": "üzgün", "yakin": "yakın",
    "yalniz": "yalnız", "yanlis": "yanlış", "yarabandi": "yara bandı",
    "yarin": "yarın", "yastik": "yastık", "yavas": "yavaş",
    "yemek_pisirmek": "yemek pişirmek", "yildiz": "yıldız",
}

# Bitişik/ayrık yazım varyantları (canonical → ekstra varyantlar)
EXTRA_VARIANTS = {
    "hoşça kal": ["hoşçakal"],
    "yara bandı": ["yarabandı"],
    "afiyet olsun": ["afiyet"],
    "özür dilemek": ["özür", "pardon"],
    "rica etmek": ["rica"],
}


def display_name(ascii_name: str) -> str:
    if ascii_name in TURKISH:
        return TURKISH[ascii_name]
    return ascii_name.replace("_", " ")


def quality_scan(entry: dict) -> dict:
    """K3 gereği kullanılan İLK referansı tara: hareket + dejenere oran + NaN."""
    arr = None
    for rel in entry.get("reference_files", []):
        p = PROJECT_ROOT / rel
        if p.exists():
            arr = np.load(str(p)).astype(np.float32)
            break
    if arr is None:
        return {"status": "NO_DATA", "L": 0.0, "R": 0.0, "degen": 1.0}
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n = arr.shape[0]
    L = arr[:, 99:162].reshape(n, 21, 3)
    R = arr[:, 162:225].reshape(n, 21, 3)
    motion = lambda b: float(np.mean(np.std(b, axis=0)))
    mL, mR = motion(L), motion(R)

    def degen_ratio(b):  # el tespit edilemeyen frame oranı (xy yayılımı < 0.05)
        spread = b[:, :, :2].max(axis=1) - b[:, :, :2].min(axis=1)  # (T, 2)
        return float(np.mean((spread < 0.05).all(axis=1)))

    active = L if mL >= mR else R
    dg = degen_ratio(active)

    if np.isnan(arr).any():
        status = "NAN"
    elif max(mL, mR) < 0.04:
        status = "WEAK"        # az hareket → animasyon zayıf görünebilir
    elif dg > 0.5:
        status = "DEGEN"       # aktif el frame'lerin yarısında tespit edilememiş
    else:
        status = "OK"
    return {"status": status, "L": mL, "R": mR, "degen": dg}


def main():
    class_map: dict[str, str] = json.loads(CLASS_MAP.read_text(encoding="utf-8"))
    dictionary: dict = json.loads(DICT_PATH.read_text(encoding="utf-8"))

    manual_idx = {e["class_idx"] for k, e in dictionary.items() if not k.startswith("_")}
    print(f"Mevcut el-yazımı girdi: {len(manual_idx)} (class_idx korunuyor)")

    added, skipped_manual, no_files = [], [], []
    for cls_str, ascii_name in sorted(class_map.items(), key=lambda kv: int(kv[0])):
        idx = int(cls_str)
        if idx in manual_idx:
            skipped_manual.append(ascii_name)
            continue

        canonical = display_name(ascii_name)
        if canonical in dictionary:                 # isim çakışması güvenliği
            print(f"  [SKIP] '{canonical}' zaten sözlükte (cls {idx})")
            continue

        refs = [f"reference_landmarks/cls{idx:03d}_{s}.npy" for s in (1, 2, 3)
                if (REF_DIR / f"cls{idx:03d}_{s}.npy").exists()]
        if not refs:
            no_files.append(f"{ascii_name} (cls {idx})")
            continue

        variants = list(EXTRA_VARIANTS.get(canonical, []))
        ascii_spaced = ascii_name.replace("_", " ")
        if ascii_spaced != canonical:               # klavyede Türkçe karakter yoksa da bulunsun
            variants.append(ascii_spaced)

        dictionary[canonical] = {
            "class_idx": idx,
            "autsl_name": ascii_name,
            "reference_files": refs,
            "duration_ms": DEFAULT_DURATION_MS,
            "category": "autsl_auto",
            "variants": variants,
        }
        added.append(canonical)

    # _meta güncelle
    words = [k for k in dictionary if not k.startswith("_")]
    dictionary["_meta"]["version"] = "1.1-full-autsl"
    dictionary["_meta"]["total_words"] = len(words)
    dictionary["_meta"]["date"] = date.today().isoformat()
    dictionary["_meta"]["note"] = (
        "15 el-yazımı girdi + class_map.json'dan otomatik genişletme "
        "(expand_dictionary.py). Otomatik girdiler: category=autsl_auto."
    )

    DICT_PATH.write_text(json.dumps(dictionary, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(f"Yazıldı: {DICT_PATH}  →  {len(words)} kelime "
          f"(+{len(added)} yeni, {len(skipped_manual)} el-yazımı korundu)")
    if no_files:
        print(f"  [UYARI] Referans dosyası bulunamayan {len(no_files)} sınıf: {no_files}")

    # ── Kalite taraması (tüm kelimeler, K3'ün kullandığı ilk referans) ──────────
    print("Kalite taraması...")
    rows = []
    for w in sorted(words):
        q = quality_scan(dictionary[w])
        rows.append((w, dictionary[w]["class_idx"], q))

    order = {"NAN": 0, "NO_DATA": 1, "DEGEN": 2, "WEAK": 3, "OK": 4}
    rows.sort(key=lambda r: (order[r[2]["status"]], r[0]))
    counts: dict[str, int] = {}
    for _, _, q in rows:
        counts[q["status"]] = counts.get(q["status"], 0) + 1

    lines = [
        "# Sözlük Kalite Raporu (ilk referans — K3'ün oynattığı dosya)",
        "",
        f"Üretim: `expand_dictionary.py` · {date.today().isoformat()} · {len(rows)} kelime",
        "",
        "| Durum | Adet | Anlamı |",
        "|---|---|---|",
        f"| OK | {counts.get('OK', 0)} | Oynatılabilir |",
        f"| WEAK | {counts.get('WEAK', 0)} | Az hareket — animasyon sönük olabilir |",
        f"| DEGEN | {counts.get('DEGEN', 0)} | Aktif el frame'lerin >%50'sinde tespit edilememiş |",
        f"| NAN | {counts.get('NAN', 0)} | Bozuk veri |",
        f"| NO_DATA | {counts.get('NO_DATA', 0)} | Dosya yok |",
        "",
        "Demo cümlelerini OK kelimelerden kur; WEAK/DEGEN kelimelerde gerekirse",
        "2./3. referansa geçilebilir (text_to_sign.py — per-word medoid, K3 notu).",
        "",
        "| Kelime | cls | Durum | L-mot | R-mot | degen |",
        "|---|---|---|---|---|---|",
    ]
    for w, idx, q in rows:
        lines.append(f"| {w} | {idx} | {q['status']} | {q['L']:.3f} | {q['R']:.3f} | {q['degen']:.2f} |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Rapor: {REPORT_PATH}")
    print("Özet:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
