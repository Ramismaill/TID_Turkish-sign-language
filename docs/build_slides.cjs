/* TID final presentation (Turkish) — dark, product-matched theme.
   Run: node docs/build_slides.cjs   (requires global pptxgenjs) */
const path = require("path");
const pptxgen = require("C:/Users/Muhammet/AppData/Roaming/npm/node_modules/pptxgenjs");
const FIG = "C:/Users/Muhammet/Desktop/tid_figs";

const C = {
  bg: "0F1020", card: "1B1C33", card2: "232544",
  pur: "8B7FFF", purDim: "6C63FF", mint: "3FB6A8",
  blue: "5B8DEF", red: "E24A4A", green: "3FB950",
  tx: "EAECF6", mut: "9AA3B8", line: "33365A",
};
const HF = "Trebuchet MS", BF = "Calibri";

const p = new pptxgen();
p.defineLayout({ name: "W", width: 13.33, height: 7.5 });
p.layout = "W";
p.theme = { headFontFace: HF, bodyFontFace: BF };

const W = 13.33, H = 7.5;

function bg(s, color = C.bg) { s.background = { color }; }
function dots(s, x, y, r = 0.07) {
  const cols = [C.blue, C.red, C.green];
  cols.forEach((c, i) => s.addShape("ellipse", { x: x + i * (r * 2 + 0.06), y, w: r * 2, h: r * 2, fill: { color: c }, line: { type: "none" } }));
}
const trUpper = (s) => s.replace(/i/g, "İ").replace(/ı/g, "I").toUpperCase();
function kicker(s, text) {
  s.addText(trUpper(text), { x: 0.6, y: 0.42, w: 9, h: 0.3, fontFace: HF, fontSize: 12, bold: true, color: C.mint, charSpacing: 2 });
}
function title(s, text, w = 12.1) {
  s.addText(text, { x: 0.6, y: 0.72, w, h: 1.0, fontFace: HF, fontSize: 30, bold: true, color: C.tx, lineSpacing: 32 });
}
function pageno(s, n) {
  s.addText(String(n), { x: W - 0.7, y: H - 0.5, w: 0.4, h: 0.3, fontFace: BF, fontSize: 10, color: C.mut, align: "right" });
  dots(s, 0.6, H - 0.48);
}
function card(s, x, y, w, h, fill = C.card) {
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: C.line, width: 1 } });
}

// ── 1. TITLE ──
let s = p.addSlide(); bg(s);
s.addShape("rect", { x: 0, y: 0, w: W, h: 0.18, fill: { color: C.purDim }, line: { type: "none" } });
dots(s, 0.6, 1.0, 0.1);
s.addText("ÇİFT YÖNLÜ TÜRK İŞARET DİLİ\nİLETİŞİM PLATFORMU", {
  x: 0.6, y: 1.6, w: 12.1, h: 2.0, fontFace: HF, fontSize: 40, bold: true, color: C.tx, lineSpacing: 46 });
s.addText("Gerçek zamanlı iskelet tabanlı tanıma  +  sadık landmark tabanlı işaret sentezi", {
  x: 0.62, y: 3.7, w: 12, h: 0.5, fontFace: BF, fontSize: 18, color: C.pur, italic: true });
card(s, 0.6, 4.7, 7.2, 1.7, C.card);
s.addText([
  { text: "Ram İsmail", options: { bold: true, color: C.tx } }, { text: "  —  24040301052\n", options: { color: C.mut } },
  { text: "Muhammet Ay", options: { bold: true, color: C.tx } }, { text: "  —  23040301147", options: { color: C.mut } },
], { x: 0.9, y: 4.95, w: 6.6, h: 1.2, fontFace: BF, fontSize: 18, lineSpacing: 30, valign: "middle" });
s.addText("FET306 — Uygulamalı Yapay Sinir Ağları\nİstanbul Topkapı Üniversitesi", {
  x: 8.1, y: 4.95, w: 4.6, h: 1.2, fontFace: BF, fontSize: 14, color: C.mut, align: "right", valign: "middle", lineSpacing: 24 });
s.addNotes("Açılış (Muhammet). 30 sn: İki yönlü TİD platformu — hem işareti Türkçeye çeviriyor hem Türkçeyi işarete. Tek slaytta projeyi konumlandır.");

// ── 2. PROBLEM & MOTIVATION ──
s = p.addSlide(); bg(s); kicker(s, "Problem & Motivasyon"); title(s, "Neden çift yönlü?");
card(s, 0.6, 2.0, 3.7, 4.6, C.card);
s.addText("%4,5", { x: 0.6, y: 2.2, w: 3.7, h: 0.8, fontFace: HF, fontSize: 46, bold: true, color: C.pur, align: "center" });
s.addText("TÜİK 2016 — 15+ yaş nüfusta işitme güçlüğü oranı", { x: 0.85, y: 2.98, w: 3.2, h: 0.7, fontFace: BF, fontSize: 12.5, color: C.tx, align: "center" });
s.addText("~89.000", { x: 0.6, y: 3.85, w: 3.7, h: 0.8, fontFace: HF, fontSize: 46, bold: true, color: C.mint, align: "center" });
s.addText("TİD kullanan Sağır topluluğu (dilbilim tahmini)", { x: 0.85, y: 4.63, w: 3.2, h: 0.7, fontFace: BF, fontSize: 12.5, color: C.tx, align: "center" });
s.addText("İşiten çoğunlukla iletişim, sınırlı sayıdaki insan tercümana bağlı.", { x: 0.85, y: 5.55, w: 3.2, h: 0.95, fontFace: BF, fontSize: 12, color: C.mut, align: "center" });
const rows2 = [
  [C.blue, "İşaret → Türkçe", "Sağır bireyin işaretini işitenin okuyabileceği metne çevir."],
  [C.green, "Türkçe → İşaret", "Yazılı Türkçeyi Sağır bireyin görebileceği işarete dönüştür."],
  [C.red, "Öğrenme açığı", "TİD kaynakları ASL/DGS’ye göre çok kısıtlı; öz-çalışma aracı yok denecek kadar az."],
];
rows2.forEach((r, i) => {
  const y = 2.0 + i * 1.55;
  card(s, 4.6, y, 8.1, 1.4, C.card);
  s.addShape("ellipse", { x: 4.85, y: y + 0.45, w: 0.5, h: 0.5, fill: { color: r[0] }, line: { type: "none" } });
  s.addText(r[1], { x: 5.6, y: y + 0.18, w: 6.9, h: 0.45, fontFace: HF, fontSize: 18, bold: true, color: C.tx });
  s.addText(r[2], { x: 5.6, y: y + 0.62, w: 6.9, h: 0.65, fontFace: BF, fontSize: 14, color: C.mut });
});
pageno(s, 2);
s.addNotes("Ram. 1 dk: İletişim açığı + iki yönlü ihtiyaç + öğrenme açığı. Üç madde projenin üç bileşenine karşılık geliyor.");

// ── 3. SYSTEM OVERVIEW ──
s = p.addSlide(); bg(s); kicker(s, "Sistem Genel Bakışı"); title(s, "Tek bir iskelet temsili, iki yön");
s.addImage({ path: path.join(FIG, "fig1_arch_tr.png"), x: 0.7, y: 1.85, w: 8.4, h: 4.08 });
s.addText("Şekil 1 — Çift yönlü mimari", { x: 0.7, y: 5.95, w: 8.4, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: C.mut, align: "center" });
card(s, 9.4, 1.95, 3.3, 4.3, C.card);
s.addText([
  { text: "Ortak temsil\n", options: { bold: true, color: C.mint, fontSize: 15 } },
  { text: "MediaPipe Holistic landmark’ları: 64 kare × 225 değer (poz + iki el).\n\n", options: { color: C.mut, fontSize: 13 } },
  { text: "Tanıma\n", options: { bold: true, color: C.blue, fontSize: 15 } },
  { text: "GCN → gloss → LLM → Türkçe cümle.\n\n", options: { color: C.mut, fontSize: 13 } },
  { text: "Sentez\n", options: { bold: true, color: C.green, fontSize: 15 } },
  { text: "Metin → sözlük → landmark → sadık 3B oynatıcı.", options: { color: C.mut, fontSize: 13 } },
], { x: 9.65, y: 2.15, w: 2.85, h: 3.9, fontFace: BF, lineSpacing: 18, valign: "top" });
pageno(s, 3);
s.addNotes("Muhammet. 1 dk: Tüm sistemin tek slaytta haritası. Vurgula: iki dal aynı landmark temsilini paylaşıyor — mimari sadeliğin anahtarı bu.");

// ── 4. DATA & REPRESENTATION ──
s = p.addSlide(); bg(s); kicker(s, "Veri Seti & Temsil"); title(s, "AUTSL + MediaPipe Holistic");
s.addImage({ path: path.join(FIG, "fig2_holistic.png"), x: 8.7, y: 1.9, w: 4.0, h: 3.02 });
s.addText("Şekil 2 — Holistic landmark’ları", { x: 8.7, y: 4.95, w: 4.0, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: C.mut, align: "center" });
const stats4 = [["226", "izole TİD sınıfı (AUTSL)"], ["64 × 225", "kare × özellik (poz + 2 el)"], ["56", "düğümlü iskelet çizgesi (yüz/bacak atıldı)"]];
stats4.forEach((st, i) => {
  const y = 2.0 + i * 1.0;
  s.addText(st[0], { x: 0.6, y, w: 2.6, h: 0.85, fontFace: HF, fontSize: 30, bold: true, color: C.pur, align: "right", valign: "middle" });
  s.addText(st[1], { x: 3.4, y, w: 4.9, h: 0.85, fontFace: BF, fontSize: 15, color: C.tx, valign: "middle" });
});
card(s, 0.6, 5.2, 7.7, 1.4, C.card);
s.addText("Omuz-merkezli normalizasyon → işaretçinin karedeki konumundan bağımsız. Diziler 64 kareye yeniden örneklenir.", { x: 0.85, y: 5.35, w: 7.2, h: 1.1, fontFace: BF, fontSize: 14, color: C.mut, valign: "middle" });
pageno(s, 4);
s.addNotes("Ram. 1 dk: Veri = AUTSL 226 sınıf. MediaPipe Holistic ile landmark. Normalizasyon ve 56-düğüm alt çizgesi. Şekil 2 gerçek çıktımız.");

// ── 5. RECOGNITION RESULTS ──
s = p.addSlide(); bg(s); kicker(s, "Tanıma — Sonuçlar"); title(s, "İskelet tabanlı GCN modelleri");
s.addText("%94.70", { x: 0.6, y: 2.2, w: 4.6, h: 1.5, fontFace: HF, fontSize: 72, bold: true, color: C.mint, align: "center" });
s.addText("TMS-Net doğrulama doğruluğu\n(AUTSL, 226 sınıf)", { x: 0.6, y: 3.7, w: 4.6, h: 0.9, fontFace: BF, fontSize: 16, color: C.tx, align: "center", lineSpacing: 22 });
s.addText("Dağıtılan model: 6 akış (eklem, kemik, hareket, açı) + çok ölçekli zamansal evrişim + akışlar arası dikkat.", { x: 0.6, y: 4.8, w: 4.6, h: 1.4, fontFace: BF, fontSize: 14, color: C.mut, align: "center" });
const tRows = [
  [{ text: "Model", options: { bold: true, color: "FFFFFF", fill: { color: C.purDim } } }, { text: "Akış", options: { bold: true, color: "FFFFFF", fill: { color: C.purDim } } }, { text: "Doğruluk", options: { bold: true, color: "FFFFFF", fill: { color: C.purDim } } }],
  ["TMS-Net", "6", "%94.70"],
  ["SML", "3", "%93.39"],
  ["ST-GCN (temel)", "—", "%89.04"],
];
s.addTable(tRows, { x: 5.6, y: 2.4, w: 7.1, colW: [3.7, 1.4, 2.0], rowH: 0.58,
  fontFace: BF, fontSize: 16, color: C.tx, align: "center", valign: "middle",
  fill: { color: C.card }, border: { type: "solid", color: C.line, pt: 1 } });
s.addText("Çok akış + çok ölçekli zamansal modelleme → temele göre +5.66 puan. Dağıtılan model TMS-Net (%94.70).", { x: 5.6, y: 5.45, w: 7.1, h: 0.9, fontFace: BF, fontSize: 13, italic: true, color: C.mut });
pageno(s, 5);
s.addNotes("Ram. 1.5 dk: Üç model eğitildi; dağıtılan TMS-Net %94.70 en iyi sonuç. Akış + çok ölçekli zamansal katkısını vurgula.");

// ── 6. GLOSS → TURKISH (LLM) ──
s = p.addSlide(); bg(s); kicker(s, "Tanıma — Dil Modeli"); title(s, "Gloss’tan akıcı Türkçeye");
const flow = ["İşaret gloss’ları\n(ör. BEN · SEN · SEVMEK)", "Qwen2.5-7B\nyerel LLM", "“Ben seni seviyorum.”"];
const fcol = [C.blue, C.pur, C.green];
flow.forEach((t, i) => {
  const x = 0.8 + i * 4.25;
  card(s, x, 2.7, 3.6, 1.8, C.card);
  s.addShape("rect", { x, y: 2.7, w: 0.12, h: 1.8, fill: { color: fcol[i] }, line: { type: "none" } });
  s.addText(t, { x: x + 0.25, y: 2.7, w: 3.2, h: 1.8, fontFace: BF, fontSize: 16, color: C.tx, align: "center", valign: "middle" });
  if (i < 2) s.addText("→", { x: x + 3.65, y: 2.7, w: 0.6, h: 1.8, fontFace: HF, fontSize: 30, bold: true, color: C.mut, align: "center", valign: "middle" });
});
s.addText("Tanıma bir gloss dizisi üretir, dilbilgisel cümle değil. 4-bit (Q4_K_M) nicelenmiş Qwen2.5-7B, llama.cpp ile yerel olarak çalışır — gizlilik + sıfır maliyet.", { x: 0.8, y: 5.0, w: 11.7, h: 1.0, fontFace: BF, fontSize: 15, color: C.mut, align: "center" });
pageno(s, 6);
s.addNotes("Ram. 45 sn: Gloss → cümle dönüşümü neden gerekli + yerel LLM tercihi (gizlilik, maliyet).");

// ── 7. EDUCATION / DTW (v1, önce yapıldı → sentezden önce) ──
s = p.addSlide(); bg(s); kicker(s, "Eğitim Modülü"); title(s, "DTW ile öz-çalışma");
s.addImage({ path: path.join(FIG, "fig4_tutor.png"), x: 0.6, y: 1.95, w: 6.6, h: 3.74 });
s.addText("Şekil 4 — Referans (sol) ↔ öğrenci (sağ), DTW skoru", { x: 0.6, y: 5.72, w: 6.6, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: C.mut, align: "center" });
card(s, 7.5, 1.95, 5.2, 3.8, C.card);
s.addText([
  { text: "Yan yana karşılaştırma\n", options: { bold: true, color: C.mint, fontSize: 16 } },
  { text: "Solda referans işaret iskeleti, sağda öğrencinin canlı kamerası.\n\n", options: { color: C.mut, fontSize: 14 } },
  { text: "Dinamik Zaman Bükümlemesi\n", options: { bold: true, color: C.mint, fontSize: 16 } },
  { text: "Landmark dizileri hizalanır → farklı hızlardaki denemeler adil puanlanır.\n\n", options: { color: C.mut, fontSize: 14 } },
  { text: "Yorumlanabilir skor\n", options: { bold: true, color: C.mint, fontSize: 16 } },
  { text: "Yüzde + “tekrar dene” geri bildirimi; eğitmen gerekmez.", options: { color: C.mut, fontSize: 14 } },
], { x: 7.75, y: 2.15, w: 4.7, h: 3.4, fontFace: BF, lineSpacing: 18, valign: "top" });
pageno(s, 7);
s.addNotes("Ram. 1 dk: Öğretici modül (v1, projede önce yapıldı) — referans vs öğrenci + DTW skoru. DTW = zaman hizalama.");

// ── 8. SYNTHESIS ──
s = p.addSlide(); bg(s); kicker(s, "Sentez — Metin → İşaret"); title(s, "Sadık “Cin Ali” oynatıcısı");
s.addImage({ path: path.join(FIG, "fig3_renderer.png"), x: 0.6, y: 1.95, w: 7.3, h: 3.58 });
s.addText("Şekil 3 — Web ön yüzü, “arkadaş” işareti", { x: 0.6, y: 5.55, w: 7.3, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: C.mut, align: "center" });
card(s, 8.2, 1.95, 4.5, 4.5, C.card);
s.addText([
  { text: "226 kelimelik sözlük\n", options: { bold: true, color: C.green, fontSize: 16 } },
  { text: "Türkçe metin → normalizasyon + ek soyma → referans landmark dizisi.\n\n", options: { color: C.mut, fontSize: 14 } },
  { text: "Doğrudan landmark sürümü\n", options: { bold: true, color: C.green, fontSize: 16 } },
  { text: "Eklemler küre, kemikler renkli çizgi (poz mavi · sol el yeşil · sağ el kırmızı).\n\n", options: { color: C.mut, fontSize: 14 } },
  { text: "30 fps · NaN yok\n", options: { bold: true, color: C.green, fontSize: 16 } },
  { text: "Sözlüğün 216/226 kelimesi sorunsuz oynatılabilir.", options: { color: C.mut, fontSize: 14 } },
], { x: 8.45, y: 2.15, w: 4.05, h: 4.1, fontFace: BF, lineSpacing: 18, valign: "top" });
pageno(s, 8);
s.addNotes("Muhammet. 1 dk: Metin→işaret hattı + sadık oynatıcı. Renk kodlaması (mavi/yeşil/kırmızı) işaret dilinde el ayrımı için.");

// ── 8. HONEST ENGINEERING RESULT ──
s = p.addSlide(); bg(s); kicker(s, "Teknik Karar & Kısıt"); title(s, "Neden rigli avatar değil?");
const cmp = [
  ["Retargeting (denendi)", C.red, [
    "Kalidokit (açı tabanlı): mutlak el konumu kaybolur → el yüze ulaşmaz.",
    "Konum tabanlı IK: 6 işaretten yalnız 1’i doğru — oran uyumsuzluğu + perspektif kısalması.",
  ]],
  ["Sadık oynatma (seçildi)", C.green, [
    "Ham landmark birebir çizilir → retargeting yok, sadakat yöntemin doğası gereği.",
    "Tüm işaretler doğru, 30 fps. Okunabilirlik > foto-gerçekçilik (literatürle uyumlu).",
  ]],
];
cmp.forEach((col, i) => {
  const x = 0.6 + i * 6.3;
  card(s, x, 2.0, 6.0, 4.5, C.card);
  s.addShape("rect", { x, y: 2.0, w: 6.0, h: 0.7, fill: { color: i === 0 ? "3A1F2B" : "1F3A2B" }, line: { type: "none" } });
  s.addText(col[0], { x: x + 0.3, y: 2.0, w: 5.4, h: 0.7, fontFace: HF, fontSize: 18, bold: true, color: col[1], valign: "middle" });
  col[2].forEach((t, j) => {
    const y = 3.0 + j * 1.6;
    s.addShape("ellipse", { x: x + 0.35, y: y + 0.05, w: 0.18, h: 0.18, fill: { color: col[1] }, line: { type: "none" } });
    s.addText(t, { x: x + 0.7, y, w: 5.0, h: 1.5, fontFace: BF, fontSize: 14.5, color: C.tx, valign: "top", lineSpacing: 19 });
  });
});
pageno(s, 9);
s.addNotes("Muhammet. 1.5 dk: EN ÖNEMLİ slayt (özgünlük + kısıt tartışması). Dürüst negatif sonuç: iki retargeting de başarısız → sadık oynatmayı bilinçli seçtik.");

// (Eğitim/DTW slaytı sentezden önceye taşındı — slayt 7 olarak yukarıda)

// ── 10. LIVE DEMO ──
s = p.addSlide(); bg(s);
s.addShape("rect", { x: 0, y: 0, w: W, h: H, fill: { color: "14152A" }, line: { type: "none" } });
kicker(s, "Canlı Demo");
s.addText("Şimdi çalışırken görelim", { x: 0.6, y: 0.95, w: 12, h: 1.0, fontFace: HF, fontSize: 34, bold: true, color: C.tx });
const demo = [
  [C.green, "1 · Sentez (web)", "“merhaba arkadaş” yaz → Cin Ali oynatıcı işareti gerçek zamanlı yapar."],
  [C.blue, "2 · Tanıma (kamera)", "Kamera karşısında işaret → TMS-Net + LLM → Türkçe cümle."],
  [C.mint, "3 · Öğretici (DTW)", "Referansı taklit et → anlık DTW skoru + geri bildirim."],
];
demo.forEach((d, i) => {
  const y = 2.3 + i * 1.5;
  card(s, 1.0, y, 11.3, 1.3, C.card);
  s.addShape("ellipse", { x: 1.35, y: y + 0.35, w: 0.6, h: 0.6, fill: { color: d[0] }, line: { type: "none" } });
  s.addText(d[1], { x: 2.2, y: y + 0.15, w: 9.8, h: 0.5, fontFace: HF, fontSize: 19, bold: true, color: C.tx });
  s.addText(d[2], { x: 2.2, y: y + 0.62, w: 9.8, h: 0.55, fontFace: BF, fontSize: 14, color: C.mut });
});
s.addText("backend: uvicorn · frontend: npm run dev · tanıma/öğretici: python (conda: isaret_dili)", { x: 1.0, y: 6.85, w: 11.3, h: 0.3, fontFace: "Consolas", fontSize: 12, color: C.mut, align: "center" });
s.addNotes("Muhammet + Ram. 2.5 dk: CANLI DEMO. Sırayla: web sentez (Muhammet), canlı tanıma (Ram), öğretici. Komutlar önceden açık olsun.");

// ── 11. CONCLUSION ──
s = p.addSlide(); bg(s); kicker(s, "Sonuç"); title(s, "Özet");
card(s, 2.1, 2.0, 9.1, 4.5, C.card);
s.addText("Başardıklarımız", { x: 2.5, y: 2.25, w: 8.3, h: 0.5, fontFace: HF, fontSize: 20, bold: true, color: C.mint });
["Tanıma: %94.70 (AUTSL, 226 sınıf, TMS-Net) → yerel LLM ile akıcı Türkçe cümle", "Sentez: 226 kelimelik sözlük, sadık 3B “Cin Ali” oynatıcı, 30 fps", "DTW tabanlı öğretici ile eğitmensiz öz-çalışma", "Dürüst mühendislik sonucu: sadık temsil > bozuk retargeting", "Tek bir iskelet temsili üzerinde çalışan, uçtan uca çift yönlü platform"].forEach((t, i) => {
  s.addText("✓", { x: 2.55, y: 3.05 + i * 0.68, w: 0.45, h: 0.4, fontFace: BF, fontSize: 18, bold: true, color: C.green });
  s.addText(t, { x: 3.15, y: 3.0 + i * 0.68, w: 7.7, h: 0.62, fontFace: BF, fontSize: 16, color: C.tx, valign: "middle", lineSpacing: 20 });
});
pageno(s, 11);
s.addNotes("Ram. 1 dk: Projenin özeti — beş ana kazanım. Vurgu: çift yönlü + dürüst mühendislik sonucu.");

// ── 12. THANKS ──
s = p.addSlide(); bg(s);
s.addShape("rect", { x: 0, y: H - 0.18, w: W, h: 0.18, fill: { color: C.purDim }, line: { type: "none" } });
dots(s, 0.6, 1.2, 0.1);
s.addText("Teşekkürler", { x: 0.6, y: 2.4, w: 12, h: 1.2, fontFace: HF, fontSize: 48, bold: true, color: C.tx });
s.addText("Sorular?", { x: 0.62, y: 3.7, w: 12, h: 0.7, fontFace: HF, fontSize: 24, color: C.pur, italic: true });
s.addText([
  { text: "Ram İsmail (24040301052)  ·  Muhammet Ay (23040301147)\n", options: { color: C.tx, fontSize: 16, bold: true } },
  { text: "FET306 — Uygulamalı Yapay Sinir Ağları  ·  İstanbul Topkapı Üniversitesi", options: { color: C.mut, fontSize: 14 } },
], { x: 0.6, y: 5.4, w: 12, h: 1.0, fontFace: BF, lineSpacing: 24 });
s.addNotes("Kapanış. Soru-cevaba hazır olun: model seçimi, DTW eşikleri, retargeting neden başarısız, gelecek avatar.");

const OUT = "C:/Users/Muhammet/Desktop/TID_Sunum.pptx";
p.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT, "(" + p.slides.length + " slides)"));
