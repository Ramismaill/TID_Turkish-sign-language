# Ram — Claude Oturumunu Nasıl Başlatırsın

## Tek adım

1. `docs/Ram_Claude_Context_Day4.md` dosyasını aç
2. Tüm içeriği kopyala (Ctrl+A → Ctrl+C)
3. Claude'a yeni oturum aç
4. Şunu yaz:

```
Aşağıdaki dosya TID projemizin tam bağlamı.
Oku, özet ver, Day 4'ü başlatalım.

[BURAYA KOPYALADIĞINI YAPISTIR]
```

Hepsi bu. Claude projenin tüm geçmişini, Day 3 bulgularını ve Day 4'ün
tam teknik planını (kod dahil) görecek — sıfırdan anlatmak zorunda kalmayacaksın.

---

## Dosya ne içeriyor

- Proje stack + klasör yapısı
- Day 1-3 arası ne bitti (backend, frontend, testler)
- **Landmark layout kanıtı** (33 pose + 21 sol + 21 sağ)
- **Day 4 için kritik bulgular:**
  - Kalidokit'e visibility:1.0 enjekte etmek ZORUNLU (yoksa avatar donuyor)
  - Dejenere el frame'i tespiti (Hand.solve çağırma)
  - Hips drift sorunu ve çözümü
- **Hazır taslak kod:** parseStoredFrame, rigRotation, applyRig (three-vrm v2)
- Kemik mapping tablosu (30 parmak kemiği dahil)
- Day 5 GO/NO-GO kriterleri
- Prior art / IEEE Related Work notları

---

## Çalıştırma komutları (hatırlatma)

**Backend:**
```bat
conda activate isaret_dili
cd C:\sign_language
python src\v2\server.py
```

**Frontend:**
```bat
cd C:\sign_language\tid-frontend
npm run dev
```

**Test:**
```bat
conda activate isaret_dili
cd C:\sign_language
python src\v2\test_translate_full.py
```

---

## Önemli kural

Tüm kod ve dosyalar **Ram İsmail** adına üretilecek.
Claude'a bunu belirt: *"Tüm çıktılar Ram İsmail adına."*
