/* Build two IEEE-format .docx reports (Turkish + English) for the TID bidirectional
   sign-language platform. Run:  node docs/build_ieee_reports.cjs
   Requires global docx (npm i -g docx). */
const fs = require("fs");
const path = require("path");
const docx = require("C:/Users/Muhammet/AppData/Roaming/npm/node_modules/docx");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, SectionType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, ImageRun,
  LevelFormat, HeadingLevel,
} = docx;

const DESKTOP = "C:/Users/Muhammet/Desktop";
const FIGDIR = path.join(DESKTOP, "tid_figs");

// ── Page geometry (US Letter, ~0.75in margins, 2-col body) ──
const PAGE = { size: { width: 12240, height: 15840 }, margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } };
const COLGAP = 360;

// ── Helpers ──
const FONT = "Times New Roman";
const body = (text) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { line: 240, after: 0 },
  indent: { firstLine: 200 },
  children: [new TextRun({ text, font: FONT, size: 20 })],
});
const h1 = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 160, after: 80 },
  children: [new TextRun({ text, font: FONT, size: 20, smallCaps: true })],
});
const h2 = (text) => new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { before: 100, after: 40 },
  children: [new TextRun({ text, font: FONT, size: 20, italics: true })],
});
const bullet = (text) => new Paragraph({
  numbering: { reference: "ieee-bullets", level: 0 },
  alignment: AlignmentType.JUSTIFIED,
  spacing: { line: 240 },
  children: [new TextRun({ text, font: FONT, size: 20 })],
});
const ref = (n, text) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { line: 240 },
  indent: { left: 300, hanging: 300 },
  children: [new TextRun({ text: `[${n}] ${text}`, font: FONT, size: 18 })],
});

function abstractPara(lead, text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 },
    children: [
      new TextRun({ text: lead, font: FONT, size: 18, bold: true, italics: true }),
      new TextRun({ text, font: FONT, size: 18, bold: true }),
    ],
  });
}
function indexTerms(lead, text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
    children: [
      new TextRun({ text: lead, font: FONT, size: 18, bold: true, italics: true }),
      new TextRun({ text, font: FONT, size: 18, italics: true }),
    ],
  });
}

// Two-column results table
function accTable(headers, rows) {
  const W = 4600, c0 = 2600, c1 = 1000, c2 = 1000;
  const border = { style: BorderStyle.SINGLE, size: 2, color: "000000" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (txt, w, bold = false, fill = null) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: txt, font: FONT, size: 18, bold })] })],
  });
  const widths = [c0, c1, c2];
  const head = new TableRow({ tableHeader: true, children: headers.map((t, i) => cell(t, widths[i], true, "D9D9D9")) });
  const trs = rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, widths[i], i === 0)) }));
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...trs] });
}

function caption(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 },
    children: [new TextRun({ text, font: FONT, size: 16 })] });
}
function tableTitle(lines) {
  return lines.map((t, i) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: i === 0 ? 100 : 0, after: i === lines.length - 1 ? 40 : 0 },
    children: [new TextRun({ text: t, font: FONT, size: 16, smallCaps: i === 0 })] }));
}
function figure(imgPath) {
  const data = fs.readFileSync(imgPath);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80 },
    children: [new ImageRun({ type: "png", data, transformation: { width: 312, height: 154 },
      altText: { title: "System architecture", description: "Bidirectional pipeline", name: "fig1" } })] });
}
function pngSize(buf) {
  if (buf.length > 24 && buf.toString("ascii", 12, 16) === "IHDR")
    return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
  return null;
}
// Embed a screenshot if present in FIGDIR; otherwise drop a labeled placeholder so the
// slot + caption stay in place until the file is saved. Width fits one column (~320 px).
function figureOpt(file, cap, placeholder) {
  const out = [];
  if (fs.existsSync(file)) {
    const data = fs.readFileSync(file);
    const sz = pngSize(data) || { w: 16, h: 9 };
    const W = 320, H = Math.min(420, Math.round(W * sz.h / sz.w));
    out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80 },
      children: [new ImageRun({ type: "png", data, transformation: { width: W, height: H },
        altText: { title: cap, description: cap, name: path.basename(file) } })] }));
  } else {
    out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      border: { top: { style: BorderStyle.DASHED, size: 4, color: "999999", space: 8 },
                bottom: { style: BorderStyle.DASHED, size: 4, color: "999999", space: 8 },
                left: { style: BorderStyle.DASHED, size: 4, color: "999999", space: 8 },
                right: { style: BorderStyle.DASHED, size: 4, color: "999999", space: 8 } },
      children: [new TextRun({ text: placeholder, font: FONT, size: 16, italics: true, color: "777777" })] }));
  }
  out.push(caption(cap));
  return out;
}

function buildDoc(C, figPath) {
  const titleSection = {
    properties: { page: PAGE, column: { count: 1 } },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
        children: [new TextRun({ text: C.title, font: FONT, size: 44 })] }),
      ...C.authors.map(line => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
        children: line.map(seg => new TextRun({ text: seg.t, font: FONT, size: seg.s || 22, italics: !!seg.i })) })),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
    ],
  };

  const bodyChildren = [];
  bodyChildren.push(abstractPara(C.absLead, C.abstract));
  bodyChildren.push(indexTerms(C.idxLead, C.indexTerms));
  for (const sec of C.sections) {
    bodyChildren.push(h1(sec.h));
    for (const blk of sec.blocks) {
      if (blk.t === "p") bodyChildren.push(body(blk.x));
      else if (blk.t === "h2") bodyChildren.push(h2(blk.x));
      else if (blk.t === "b") bodyChildren.push(bullet(blk.x));
      else if (blk.t === "fig") { bodyChildren.push(figure(figPath)); bodyChildren.push(caption(blk.cap)); }
      else if (blk.t === "imgfig") { figureOpt(path.join(FIGDIR, blk.file), blk.cap, blk.ph).forEach(p => bodyChildren.push(p)); }
      else if (blk.t === "table") {
        tableTitle(blk.title).forEach(p => bodyChildren.push(p));
        bodyChildren.push(accTable(blk.headers, blk.rows));
        bodyChildren.push(caption(blk.cap || ""));
      }
    }
  }
  bodyChildren.push(h1(C.refHead));
  C.references.forEach((r, i) => bodyChildren.push(ref(i + 1, r)));

  const bodySection = {
    properties: { type: SectionType.CONTINUOUS, page: PAGE, column: { count: 2, space: COLGAP, equalWidth: true } },
    children: bodyChildren,
  };

  return new Document({
    numbering: { config: [{ reference: "ieee-bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
    sections: [titleSection, bodySection],
  });
}

// ===================== CONTENT: ENGLISH =====================
const EN = {
  title: "A Bidirectional Turkish Sign Language Communication Platform: Real-Time Skeleton-Based Recognition and Faithful Landmark-Driven Sign Synthesis",
  authors: [
    [{ t: "Ram İsmail (24040301052) and Muhammet Ay (23040301147)", s: 22 }],
    [{ t: "Department of Software Engineering, İstanbul Topkapı University", s: 20, i: true }],
    [{ t: "FET306 — Applied Neural Networks", s: 20, i: true }],
    [{ t: "{ramismail, muhammetay}@stu.topkapi.edu.tr", s: 18, i: true }],
  ],
  absLead: "Abstract—",
  abstract: "Turkish Sign Language (TİD) is the primary language of the Turkish Deaf community, yet automated tools that connect signed and written Turkish remain scarce. We present a bidirectional communication platform that couples sign recognition with sign synthesis. For recognition, upper-body and hand landmarks are extracted from video with MediaPipe Holistic and classified by skeleton-based graph convolutional networks; our best model, a six-stream Temporal Multi-Scale Network (TMS-Net), reaches 94.70% validation accuracy on 226 isolated AUTSL classes; the predicted glosses are turned into fluent Turkish sentences by a locally hosted Qwen2.5-7B language model. For synthesis, Turkish text is mapped through a 226-word dictionary with light morphological normalization to reference landmark sequences that are rendered in real time. We additionally report a negative engineering result: both angle-based (Kalidokit) and analytic position-based inverse-kinematics retargeting onto a rigged humanoid avatar failed to preserve spatial fidelity, so we adopt a faithful landmark-driven stick-figure renderer that reproduces every stored sign exactly at 30 frames per second. A self-study tutor that scores a learner against a reference skeleton with Dynamic Time Warping completes the platform. We discuss the design decisions, limitations, and the route toward a fully rigged expressive avatar.",
  idxLead: "Index Terms—",
  indexTerms: "Dynamic time warping, graph convolutional networks, MediaPipe, sign language recognition, sign synthesis, Turkish Sign Language.",
  sections: [
    { h: "I.  Introduction", blocks: [
      { t: "p", x: "According to the 2016 Turkey Health Survey of the Turkish Statistical Institute (TÜİK), 4.5% of the population aged 15 and over reports difficulty hearing, and linguistic surveys place Turkey’s TİD-using Deaf community at roughly 89,000 individuals. Turkish Sign Language (TİD) is the natural language of this community. Communication with the hearing majority, however, still depends on the limited availability of human interpreters. Practical assistive technology must therefore work in both directions: it must recognize signing and render it as written Turkish for hearing users, and it must turn written Turkish into visible signing for Deaf users. Most academic prototypes address only one direction, and resources for TİD in particular are far scarcer than for American or German Sign Language." },
      { t: "p", x: "This paper describes a single platform that addresses both directions and a self-study mode for learners. The system is built on a shared skeletal representation — the body and hand landmarks produced by MediaPipe Holistic — which serves as the common interface between a recognition branch and a synthesis branch (Fig. 1)." },
      { t: "fig", cap: "Fig. 1.  Bidirectional architecture. A shared MediaPipe-Holistic landmark representation links the recognition branch (top) and the synthesis branch (bottom); both are grounded in the AUTSL dataset." },
      { t: "p", x: "Our contributions are as follows:" },
      { t: "b", x: "A real-time recognition pipeline in which a six-stream TMS-Net classifier (94.70% validation accuracy on 226 AUTSL classes) feeds predicted glosses to a locally hosted large language model that produces fluent Turkish sentences." },
      { t: "b", x: "A Turkish-text-to-sign synthesis pipeline that resolves words through a 226-entry dictionary with morphological normalization and renders the corresponding reference landmarks in a web browser." },
      { t: "b", x: "An honest comparison of three embodiment strategies, showing that a faithful landmark-driven renderer outperforms two retargeting methods that lose spatial fidelity — a result we believe is useful to other practitioners." },
      { t: "b", x: "A Dynamic-Time-Warping-based tutor that scores a learner’s attempt against a reference sign for self-study." },
    ]},
    { h: "II.  Related Work and Background", blocks: [
      { t: "h2", x: "A.  Skeleton-based sign and action recognition" },
      { t: "p", x: "Graph convolutional networks treat the human skeleton as a spatio-temporal graph. ST-GCN introduced spatial graph convolutions combined with temporal convolutions over joint trajectories. Later work added multiple input streams (joint, bone, and their motion) and channel-wise topology refinement, as in CTR-GCN, consistently improving accuracy. Our models follow this lineage and extend it with multi-scale temporal kernels and cross-stream attention." },
      { t: "h2", x: "B.  Datasets and landmark extraction" },
      { t: "p", x: "AUTSL is a large isolated Turkish Sign Language dataset covering 226 signs performed by many signers. We obtain landmarks with MediaPipe Holistic, which jointly estimates pose, face and both hands; we retain the upper-body pose joints and the two 21-point hand skeletons." },
      { t: "h2", x: "C.  Sign synthesis and avatar retargeting" },
      { t: "p", x: "Signing avatars are usually animated by retargeting captured motion onto a rigged humanoid. Angle-based libraries such as Kalidokit infer joint rotations from landmarks but discard absolute end-effector position, while inverse kinematics places end-effectors but must reconcile the signer’s proportions with the avatar’s. Studies of avatar legibility report that stylization and clarity matter more than photorealism for comprehension, which motivates our final design choice." },
      { t: "h2", x: "D.  Gloss-to-text with language models" },
      { t: "p", x: "Sign recognition typically yields a sequence of glosses rather than grammatical text. We use an instruction-tuned large language model, run locally for privacy and cost, to convert glosses into fluent Turkish, following the now-common pattern of using LLMs as lightweight text normalizers." },
    ]},
    { h: "III.  Methodology: Model and Dataset", blocks: [
      { t: "h2", x: "A.  Data representation" },
      { t: "p", x: "Each video is processed by MediaPipe Holistic into per-frame landmarks: 33 pose, 21 left-hand and 21 right-hand points, each with three coordinates, giving a 225-dimensional vector per frame. Sequences are resampled to a fixed length of 64 frames and normalized so that the mid-shoulder point is the origin, which makes the representation invariant to the signer’s position in the frame. For the recognition networks we keep a 56-node subgraph — 14 upper-body pose joints plus the two hands — and discard the face and legs, yielding an input tensor of shape (3, 64, 56)." },
      { t: "imgfig", file: "fig2_holistic.png", cap: "Fig. 2.  MediaPipe Holistic landmarks extracted from a signer: full-body pose (green) and the two 21-point hand skeletons (blue/orange). The recognition model keeps the upper-body pose joints and both hands, discarding the face and legs.", ph: "[ Fig. 2 — MediaPipe Holistic landmark screenshot ]" },
      { t: "h2", x: "B.  Recognition models" },
      { t: "p", x: "We trained three skeleton-based classifiers of increasing capacity. ST-GCN is a baseline spatial-temporal graph network with a learnable adjacency residual. SML is a three-stream network (joint, bone, motion) whose streams exchange information through a cross-stream attention block. TMS-Net is a six-stream model (joint, bone, joint-motion, bone-motion, angle, angle-motion) with a multi-scale temporal convolution (fast, medium and slow kernels) and attention-based fusion across streams. All models output a 226-way classification." },
      { t: "h2", x: "C.  Gloss-to-Turkish translation" },
      { t: "p", x: "Predicted glosses are passed to Qwen2.5-7B-Instruct, quantized to 4 bits (Q4_K_M) and executed with llama.cpp on the local GPU/CPU. A fixed instruction prompt asks the model to produce a single grammatical Turkish sentence from the ordered glosses, adding function words and inflection while preserving meaning." },
      { t: "h2", x: "D.  Text-to-sign synthesis" },
      { t: "p", x: "On the synthesis side, input text is lowercased and stripped of punctuation, then each token is resolved against a dictionary of 226 words. Resolution proceeds by exact match, by a hand-curated variant map (e.g. colloquial synonyms), and finally by stripping common Turkish suffixes in longest-first order. Each dictionary entry points to reference landmark files. We deliberately use a single real recording per word rather than the mean of several: because the references are not temporally aligned, averaging blends different phases of a sign into an anatomically impossible “ghost” handshape that is especially harmful to finger readability." },
    ]},
    { h: "IV.  Experimental Setup and Implementation", blocks: [
      { t: "p", x: "Recognition models were trained in PyTorch on an NVIDIA RTX 5060 Ti (16 GB). Development and the live demo run on a laptop with an Intel Core i7-13620H, 16 GB of RAM and an NVIDIA RTX 4060 Laptop GPU. Landmark extraction uses MediaPipe; sequence alignment in the tutor uses dtw-python." },
      { t: "p", x: "The recognition demo captures webcam frames, extracts Holistic landmarks, classifies a completed sign with TMS-Net, and forwards the predicted glosses to the language model, which returns a Turkish sentence; the whole loop runs at interactive rates with CUDA acceleration." },
      { t: "p", x: "The synthesis system is a web application: a FastAPI backend exposes the dictionary and returns landmark sequences, and a Vite/TypeScript front end renders them with Three.js. The 226-word dictionary was generated automatically from the dataset class map, with Turkish orthography restored and spelling variants added. A quality scan flags words whose active hand is undetected in more than half of their frames, or whose motion is negligible, so that demonstrations can avoid weak recordings." },
      { t: "p", x: "The renderer draws joints as instanced spheres and bones as vertex-colored line segments, with a circle for the head, driven directly by the 225-value landmark frames. Because it renders the source data without retargeting, it is faithful by construction. Frames in which a hand is undetected are held at the last valid pose to avoid the hand collapsing for a frame." },
      { t: "p", x: "For completeness we also implemented two retargeting paths onto a rigged VRM 1.0 humanoid: an angle-based path using Kalidokit (bypassing its high-level solver and combining arm angles with per-hand solves), and an analytic position-based two-bone inverse-kinematics path with a law-of-cosines elbow, a per-sign constant arm-length normalization, active-hand gating, and temporal smoothing." },
    ]},
    { h: "V.  Results and Discussion", blocks: [
      { t: "table", title: ["Table I", "Recognition accuracy on the AUTSL validation set (226 classes)"],
        headers: ["Model", "Epoch", "Val. acc."],
        rows: [["TMS-Net (6-stream)", "117", "94.70%"], ["SML (3-stream)", "120", "93.39%"], ["ST-GCN (baseline)", "51", "89.04%"]],
        cap: "" },
      { t: "p", x: "Table I reports validation accuracy for the three recognition models. The six-stream TMS-Net is the strongest model at 94.70%, improving on the ST-GCN baseline by 5.66 percentage points; the lighter three-stream SML is close behind at 93.39%. The gains track the added input streams and the multi-scale temporal modeling, which together capture both slow hand-shape transitions and fast movements. TMS-Net is deployed in the live demo since it already meets our accuracy target at the lowest runtime cost." },
      { t: "table", title: ["Table II", "Synthesis dictionary quality scan (226 words)"],
        headers: ["Category", "Count", "Share"],
        rows: [["Playable (OK)", "216", "95.6%"], ["Degenerate hand", "9", "4.0%"], ["Weak motion", "1", "0.4%"]],
        cap: "" },
      { t: "p", x: "On the synthesis side, the quality scan (Table II) classifies 216 of 226 words as readily playable; nine have unreliable hand tracking and one has negligible motion. Because the renderer reproduces the stored landmarks exactly, every demonstration sentence is rendered correctly at 30 frames per second with no numerical failures; faithfulness is a property of the method rather than an empirical accuracy to be measured." },
      { t: "imgfig", file: "fig3_renderer.png", cap: "Fig. 3.  Web front end rendering the sign “arkadaş” (friend) with the faithful stick-figure renderer: pose in blue, left hand in green and right hand in red, driven directly by the stored landmarks.", ph: "[ Fig. 3 — Cin Ali renderer screenshot ]" },
      { t: "p", x: "The retargeting experiments produced an instructive negative result. The angle-based method never achieved hand-to-face contact, because joint angles alone do not constrain absolute end-effector position — the “location parameter” that is linguistically essential in signs such as the greeting “merhaba.” The position-based inverse-kinematics method placed the wrist correctly for individual cases but generalized to only one of six test signs: the mismatch between the signer’s and the avatar’s proportions, together with foreshortening when the arm points toward the camera, distorted the mapping. Faced with a rigged avatar that was unreliable and a landmark renderer that was exact, we chose fidelity over embodiment. This is consistent with the literature on signing avatars, which finds that legibility, not photorealism, governs comprehension." },
      { t: "p", x: "Finally, the self-study tutor presents a reference skeleton beside the learner’s webcam feed and scores the attempt with Dynamic Time Warping over the landmark sequences, giving an interpretable per-attempt similarity score that supports practice without an instructor." },
      { t: "imgfig", file: "fig4_tutor.png", cap: "Fig. 4.  Self-study tutor. Left: the reference sign skeleton; right: the learner’s live capture. A Dynamic-Time-Warping distance over the landmark sequences yields the similarity score shown below (here 30%).", ph: "[ Fig. 4 — DTW tutor split-screen screenshot ]" },
    ]},
    { h: "VI.  Conclusion and Future Work", blocks: [
      { t: "p", x: "We presented a bidirectional Turkish Sign Language platform that recognizes isolated signs with 94.70% accuracy and converts them to Turkish through a local language model, synthesizes signing from Turkish text with a faithful landmark renderer, and supports self-study through a Dynamic-Time-Warping tutor. A central lesson is that a faithful, if simple, representation can be preferable to an expressive but inaccurate one." },
      { t: "p", x: "Future work targets a fully rigged, stylized avatar with a corrected hand rig and ARKit-style blendshapes for the non-manual markers (facial expression, mouthing) that carry grammatical meaning; continuous, sentence-level recognition rather than isolated signs; a larger vocabulary; medoid-based selection of the most representative reference per word; and a user study measuring comprehension with Deaf participants. As a further direction, we are also exploring a “Video Signer” that combines LLM-based gloss generation with retrieval of real-signer video clips." },
    ]},
  ],
  refHead: "References",
  references: [
    "S. Yan, Y. Xiong, and D. Lin, “Spatial temporal graph convolutional networks for skeleton-based action recognition,” in Proc. AAAI, 2018.",
    "Y. Chen, Z. Zhang, C. Yuan, B. Li, Y. Deng, and W. Hu, “Channel-wise topology refinement graph convolution for skeleton-based action recognition,” in Proc. ICCV, 2021.",
    "O. M. Sincan and H. Y. Keles, “AUTSL: A large scale multi-modal Turkish sign language dataset and baseline methods,” IEEE Access, vol. 8, pp. 181340–181355, 2020.",
    "C. Lugaresi et al., “MediaPipe: A framework for building perception pipelines,” arXiv:1906.08172, 2019.",
    "H. Sakoe and S. Chiba, “Dynamic programming algorithm optimization for spoken word recognition,” IEEE Trans. Acoust., Speech, Signal Process., vol. 26, no. 1, pp. 43–49, 1978.",
    "Qwen Team, “Qwen2.5 technical report,” arXiv:2412.15115, 2024.",
    "G. Gerganov et al., “llama.cpp: LLM inference in C/C++,” 2023. [Online]. Available: https://github.com/ggerganov/llama.cpp",
    "R. Cabello et al., “three.js — JavaScript 3D library,” 2010. [Online]. Available: https://threejs.org",
    "Y. Yamamoto (pixiv), “Kalidokit: blendshape and kinematics solver for MediaPipe,” 2021. [Online]. Available: https://github.com/yeemachine/kalidokit",
    "N. Adamo-Villani and R. B. Wilbur, “Able-bodied versus stylized signing avatars: A study of legibility,” in Proc. Int. Conf. on Universal Access in HCI, 2016.",
    "H. Kacorri and M. Huenerfauth, “Evaluating a dynamic time warping based scoring metric for sign language tutoring,” in Proc. ASSETS, 2013.",
    "Turkish Statistical Institute (TÜİK), “Türkiye Sağlık Araştırması 2016 (Turkey Health Survey 2016),” TÜİK, Ankara, 2016.",
    "U. Zeshan, “Aspects of Türk İşaret Dili (Turkish Sign Language),” Sign Language & Linguistics, vol. 6, no. 1, pp. 43–75, 2003.",
  ],
};

// ===================== CONTENT: TURKISH =====================
const TR = {
  title: "Çift Yönlü Bir Türk İşaret Dili İletişim Platformu: Gerçek Zamanlı İskelet Tabanlı Tanıma ve Sadık Landmark Tabanlı İşaret Sentezi",
  authors: [
    [{ t: "Ram İsmail (24040301052) ve Muhammet Ay (23040301147)", s: 22 }],
    [{ t: "Yazılım Mühendisliği Bölümü, İstanbul Topkapı Üniversitesi", s: 20, i: true }],
    [{ t: "FET306 — Uygulamalı Yapay Sinir Ağları", s: 20, i: true }],
    [{ t: "{ramismail, muhammetay}@stu.topkapi.edu.tr", s: 18, i: true }],
  ],
  absLead: "Özet—",
  abstract: "Türk İşaret Dili (TİD), Türkiye’deki Sağır topluluğunun doğal dilidir; ancak işaretli ve yazılı Türkçeyi birbirine bağlayan otomatik araçlar hâlâ azdır. Bu çalışmada, işaret tanıma ile işaret sentezini bir araya getiren çift yönlü bir iletişim platformu sunuyoruz. Tanıma tarafında, üst beden ve el işaret noktaları videodan MediaPipe Holistic ile çıkarılıp iskelet tabanlı çizge evrişimli ağlarla sınıflandırılır; en iyi modelimiz olan altı akışlı Zamansal Çok Ölçekli Ağ (TMS-Net), 226 izole AUTSL sınıfında %94.70 doğrulama doğruluğuna ulaşır. Tahmin edilen işaret gloss’ları, yerel olarak çalışan bir Qwen2.5-7B dil modeliyle akıcı Türkçe cümlelere dönüştürülür. Sentez tarafında, Türkçe metin hafif bir biçimbilimsel normalleştirme ile 226 kelimelik bir sözlük üzerinden referans landmark dizilerine eşlenir ve gerçek zamanlı olarak görüntülenir. Ayrıca olumsuz bir mühendislik sonucunu da raporluyoruz: açı tabanlı (Kalidokit) ve analitik konum tabanlı ters kinematik retargeting yöntemlerinin ikisi de uzamsal sadakati koruyamadı; bu nedenle her depolanan işareti saniyede 30 kare ile birebir yeniden üreten, landmark tabanlı sadık bir çubuk-figür oynatıcısı benimsiyoruz. Bir öğrenciyi referans iskeletine karşı Dinamik Zaman Bükümlemesi ile puanlayan bir öz-çalışma modülü platformu tamamlar. Tasarım kararlarını, kısıtları ve tam donanımlı ifade edici bir avatara giden yolu tartışıyoruz.",
  idxLead: "Anahtar Kelimeler—",
  indexTerms: "Çizge evrişimli ağlar, dinamik zaman bükümlemesi, işaret dili tanıma, işaret sentezi, MediaPipe, Türk İşaret Dili.",
  sections: [
    { h: "I.  Giriş", blocks: [
      { t: "p", x: "Türkiye İstatistik Kurumu’nun (TÜİK) 2016 Türkiye Sağlık Araştırması’na göre 15 yaş ve üzeri nüfusun %4,5’i işitme güçlüğü bildirmektedir ve dilbilimsel araştırmalar Türkiye’deki TİD kullanan Sağır topluluğunu yaklaşık 89.000 kişi olarak tahmin etmektedir. Türk İşaret Dili (TİD) bu topluluğun doğal dilidir. Ancak işiten çoğunlukla iletişim hâlâ sınırlı sayıdaki insan tercümanlara bağlıdır. Bu nedenle işlevsel bir yardımcı teknolojinin iki yönde de çalışması gerekir: işaretlemeyi tanıyıp işiten kullanıcılar için yazılı Türkçeye dönüştürmeli, aynı zamanda yazılı Türkçeyi Sağır kullanıcılar için görünür işarete çevirmelidir. Akademik prototiplerin çoğu yalnızca tek yönü ele alır ve özellikle TİD için kaynaklar Amerikan veya Alman İşaret Diline göre çok daha kısıtlıdır." },
      { t: "p", x: "Bu makale, her iki yönü ve öğrenciler için bir öz-çalışma modunu tek bir platformda ele almaktadır. Sistem, paylaşılan bir iskelet temsiline — MediaPipe Holistic tarafından üretilen beden ve el işaret noktalarına — dayanır; bu temsil, tanıma dalı ile sentez dalı arasındaki ortak arayüz olarak görev yapar (Şekil 1)." },
      { t: "fig", cap: "Şekil 1.  Çift yönlü mimari. Paylaşılan bir MediaPipe-Holistic landmark temsili, tanıma dalı (üst) ile sentez dalını (alt) bağlar; her ikisi de AUTSL veri setine dayanır." },
      { t: "p", x: "Katkılarımız şunlardır:" },
      { t: "b", x: "Altı akışlı TMS-Net sınıflandırıcısının (226 AUTSL sınıfında %94.70 doğrulama doğruluğu) tahmin ettiği gloss’ları, akıcı Türkçe cümleler üreten yerel bir büyük dil modeline ileten gerçek zamanlı bir tanıma hattı." },
      { t: "b", x: "Kelimeleri biçimbilimsel normalleştirme ile 226 girdilik bir sözlük üzerinden çözen ve ilgili referans landmark’ları bir web tarayıcısında görüntüleyen bir Türkçe-metinden-işarete sentez hattı." },
      { t: "b", x: "Üç farklı bedenleme stratejisinin dürüst bir karşılaştırması; sadık landmark tabanlı oynatıcının, uzamsal sadakati kaybeden iki retargeting yöntemine üstünlüğünü göstermesi." },
      { t: "b", x: "Bir öğrencinin denemesini referans işaretle karşılaştırıp puanlayan, Dinamik Zaman Bükümlemesi tabanlı bir öz-çalışma modülü." },
    ]},
    { h: "II.  İlgili Çalışmalar ve Arka Plan", blocks: [
      { t: "h2", x: "A.  İskelet tabanlı işaret ve eylem tanıma" },
      { t: "p", x: "Çizge evrişimli ağlar insan iskeletini uzay-zamansal bir çizge olarak ele alır. ST-GCN, eklem yörüngeleri üzerinde uzamsal çizge evrişimlerini zamansal evrişimlerle birleştirmiştir. Sonraki çalışmalar, CTR-GCN’de olduğu gibi çoklu giriş akışları (eklem, kemik ve bunların hareketi) ve kanal bazında topoloji iyileştirmesi ekleyerek doğruluğu artırmıştır. Modellerimiz bu soydan gelir ve çok ölçekli zamansal çekirdekler ile akışlar arası dikkat mekanizmasıyla onu genişletir." },
      { t: "h2", x: "B.  Veri setleri ve landmark çıkarımı" },
      { t: "p", x: "AUTSL, çok sayıda işaretçi tarafından gerçekleştirilen 226 işareti kapsayan büyük ölçekli, izole bir Türk İşaret Dili veri setidir. Landmark’ları; poz, yüz ve iki eli birlikte kestiren MediaPipe Holistic ile elde ediyoruz; üst beden poz eklemlerini ve iki adet 21 noktalı el iskeletini koruyoruz." },
      { t: "h2", x: "C.  İşaret sentezi ve avatar retargeting" },
      { t: "p", x: "İşaret avatarları genellikle yakalanan hareketin donanımlı bir insansıya aktarılmasıyla canlandırılır. Kalidokit gibi açı tabanlı kütüphaneler eklem rotasyonlarını landmark’lardan çıkarır ancak mutlak uç-eleman konumunu yok sayar; ters kinematik ise uç-elemanı yerleştirir fakat işaretçinin oranlarını avatarınkiyle uzlaştırmak zorundadır. Avatar okunabilirliği üzerine yapılan çalışmalar, anlama için foto-gerçekçilikten çok stilizasyon ve netliğin önemli olduğunu bildirir; bu da nihai tasarım tercihimizi güdeler." },
      { t: "h2", x: "D.  Dil modelleriyle gloss’tan metne" },
      { t: "p", x: "İşaret tanıma genellikle dilbilgisel metin yerine bir gloss dizisi üretir. Gloss’ları akıcı Türkçeye dönüştürmek için, gizlilik ve maliyet nedeniyle yerel olarak çalıştırılan, talimatla ince ayar yapılmış bir büyük dil modeli kullanıyoruz." },
    ]},
    { h: "III.  Yöntem: Model ve Veri Seti", blocks: [
      { t: "h2", x: "A.  Veri temsili" },
      { t: "p", x: "Her video MediaPipe Holistic ile kare başına landmark’lara işlenir: 33 poz, 21 sol el ve 21 sağ el noktası, her biri üç koordinatlı; bu da kare başına 225 boyutlu bir vektör verir. Diziler 64 karelik sabit bir uzunluğa yeniden örneklenir ve omuz orta noktası başlangıç olacak şekilde normalleştirilir; bu, temsili işaretçinin karedeki konumundan bağımsız kılar. Tanıma ağları için 56 düğümlü bir alt çizge — 14 üst beden poz eklemi artı iki el — tutulur, yüz ve bacaklar atılır; böylece (3, 64, 56) biçiminde bir giriş tensörü elde edilir." },
      { t: "imgfig", file: "fig2_holistic.png", cap: "Şekil 2.  Bir işaretçiden çıkarılan MediaPipe Holistic landmark’ları: tüm beden pozu (yeşil) ve iki adet 21 noktalı el iskeleti (mavi/turuncu). Tanıma modeli üst beden poz eklemlerini ve iki eli tutar; yüz ve bacaklar atılır.", ph: "[ Şekil 2 — MediaPipe Holistic landmark ekran görüntüsü ]" },
      { t: "h2", x: "B.  Tanıma modelleri" },
      { t: "p", x: "Artan kapasitede üç iskelet tabanlı sınıflandırıcı eğittik. ST-GCN, öğrenilebilir bir komşuluk artığına sahip temel bir uzay-zamansal çizge ağıdır. SML, akışları bir akışlar-arası dikkat bloğuyla bilgi alışverişinde bulunan üç akışlı (eklem, kemik, hareket) bir ağdır. TMS-Net ise çok ölçekli zamansal evrişim (hızlı, orta ve yavaş çekirdekler) ve akışlar arası dikkat tabanlı füzyon içeren altı akışlı (eklem, kemik, eklem-hareket, kemik-hareket, açı, açı-hareket) bir modeldir. Tüm modeller 226 sınıflı bir çıktı verir." },
      { t: "h2", x: "C.  Gloss’tan Türkçeye çeviri" },
      { t: "p", x: "Tahmin edilen gloss’lar, 4 bite (Q4_K_M) nicelenmiş ve llama.cpp ile yerel GPU/CPU üzerinde çalıştırılan Qwen2.5-7B-Instruct modeline iletilir. Sabit bir talimat istemi, modelden sıralı gloss’lardan anlamı koruyarak, işlev sözcükleri ve çekim ekleri ekleyen tek bir dilbilgisel Türkçe cümle üretmesini ister." },
      { t: "h2", x: "D.  Metinden işarete sentez" },
      { t: "p", x: "Sentez tarafında giriş metni küçük harfe çevrilip noktalama işaretlerinden arındırılır, ardından her sözcük 226 kelimelik bir sözlüğe karşı çözülür. Çözümleme; tam eşleşme, elle hazırlanmış bir varyant haritası (ör. konuşma dilindeki eş anlamlılar) ve son olarak yaygın Türkçe eklerinin en uzundan başlayarak soyulmasıyla ilerler. Her sözlük girdisi referans landmark dosyalarına işaret eder. Kelime başına birden çok kaydın ortalaması yerine bilinçli olarak tek bir gerçek kayıt kullanıyoruz: referanslar zamansal olarak hizalı olmadığından, ortalama almak bir işaretin farklı evrelerini, özellikle parmak okunabilirliğine zarar veren, anatomik olarak imkânsız bir “hayalet” el şekline karıştırır." },
    ]},
    { h: "IV.  Deneysel Kurulum ve Uygulama", blocks: [
      { t: "p", x: "Tanıma modelleri PyTorch ile bir NVIDIA RTX 5060 Ti (16 GB) üzerinde eğitildi. Geliştirme ve canlı demo, Intel Core i7-13620H, 16 GB RAM ve NVIDIA RTX 4060 Laptop GPU’lu bir dizüstü bilgisayarda çalışır. Landmark çıkarımı MediaPipe; öğretici modüldeki dizi hizalaması dtw-python kullanır." },
      { t: "p", x: "Tanıma demosu webcam karelerini alır, Holistic landmark’larını çıkarır, tamamlanan bir işareti TMS-Net ile sınıflandırır ve tahmin edilen gloss’ları bir Türkçe cümle döndüren dil modeline iletir; tüm döngü CUDA hızlandırmasıyla etkileşimli hızlarda çalışır." },
      { t: "p", x: "Sentez sistemi bir web uygulamasıdır: bir FastAPI arka uç sözlüğü sunar ve landmark dizilerini döndürür; bir Vite/TypeScript ön yüzü bunları Three.js ile görüntüler. 226 kelimelik sözlük, veri seti sınıf haritasından otomatik olarak üretildi; Türkçe yazım geri yüklendi ve yazım varyantları eklendi. Bir kalite taraması, aktif eli karelerinin yarısından fazlasında tespit edilemeyen ya da hareketi ihmal edilebilir olan kelimeleri işaretler; böylece gösterimler zayıf kayıtlardan kaçınabilir." },
      { t: "p", x: "Oynatıcı, eklemleri örneklenmiş küreler, kemikleri köşe-renkli çizgi parçaları ve başı bir çemberle, doğrudan 225 değerli landmark kareleriyle sürülerek çizer. Kaynağı retargeting olmadan görüntülediği için yapısı gereği sadıktır. Bir elin tespit edilemediği kareler, elin bir kare boyunca çökmesini önlemek için son geçerli pozunda tutulur." },
      { t: "p", x: "Bütünlük için, donanımlı bir VRM 1.0 insansıya iki retargeting yolu da uyguladık: Kalidokit kullanan açı tabanlı bir yol (üst düzey çözücüsünü atlayıp kol açılarını el bazında çözümlerle birleştirerek) ve kosinüs teoremiyle dirsek, işaret başına sabit kol uzunluğu normalleştirmesi, aktif el geçitlemesi ve zamansal yumuşatma içeren analitik konum tabanlı iki-kemik ters kinematik bir yol." },
    ]},
    { h: "V.  Sonuçlar ve Tartışma", blocks: [
      { t: "table", title: ["Tablo I", "AUTSL doğrulama kümesinde tanıma doğruluğu (226 sınıf)"],
        headers: ["Model", "Epok", "Doğr."],
        rows: [["TMS-Net (6 akış)", "117", "%94.70"], ["SML (3 akış)", "120", "%93.39"], ["ST-GCN (temel)", "51", "%89.04"]],
        cap: "" },
      { t: "p", x: "Tablo I, üç tanıma modelinin doğrulama doğruluğunu raporlar. Altı akışlı TMS-Net %94.70 ile en güçlü modeldir ve ST-GCN temeline göre 5.66 puanlık iyileşme sağlar; daha hafif üç akışlı SML %93.39 ile hemen arkadadır. Kazanımlar, eklenen giriş akışları ve çok ölçekli zamansal modellemeyle ilişkilidir; bunlar hem yavaş el şekli geçişlerini hem de hızlı hareketleri yakalar. TMS-Net, en düşük çalışma maliyetiyle doğruluk hedefimizi zaten karşıladığından canlı demoda dağıtılan model olarak seçilmiştir." },
      { t: "table", title: ["Tablo II", "Sentez sözlüğü kalite taraması (226 kelime)"],
        headers: ["Kategori", "Sayı", "Oran"],
        rows: [["Oynatılabilir", "216", "%95.6"], ["Bozuk el izi", "9", "%4.0"], ["Zayıf hareket", "1", "%0.4"]],
        cap: "" },
      { t: "p", x: "Sentez tarafında kalite taraması (Tablo II) 226 kelimenin 216’sını doğrudan oynatılabilir olarak sınıflandırır; dokuzunda el izleme güvenilmez, birinde hareket ihmal edilebilirdir. Oynatıcı depolanan landmark’ları birebir yeniden ürettiği için her gösterim cümlesi saniyede 30 kare ile, hiçbir sayısal hata olmadan doğru şekilde görüntülenir; sadakat, ölçülmesi gereken ampirik bir doğruluk değil, yöntemin bir özelliğidir." },
      { t: "imgfig", file: "fig3_renderer.png", cap: "Şekil 3.  “arkadaş” işaretini sadık çubuk-figür oynatıcısıyla görüntüleyen web ön yüzü: poz mavi, sol el yeşil ve sağ el kırmızı; doğrudan depolanan landmark’larla sürülür.", ph: "[ Şekil 3 — Cin Ali oynatıcı ekran görüntüsü ]" },
      { t: "p", x: "Retargeting deneyleri öğretici bir olumsuz sonuç verdi. Açı tabanlı yöntem hiçbir zaman el-yüz temasına ulaşamadı; çünkü tek başına eklem açıları mutlak uç-eleman konumunu — “merhaba” gibi işaretlerde dilbilimsel olarak zorunlu olan “konum parametresini” — kısıtlamaz. Konum tabanlı ters kinematik yöntem bilek konumunu tekil durumlarda doğru yerleştirdi ancak altı test işaretinden yalnızca birine genelleşebildi: işaretçinin ve avatarın oranları arasındaki uyumsuzluk ile kol kameraya doğru işaret ettiğinde oluşan kısalma (foreshortening), eşlemeyi bozdu. Güvenilmez bir donanımlı avatar ile birebir sadık bir landmark oynatıcısı arasında, bedenleme yerine sadakati seçtik. Bu, işaret avatarları üzerine, anlamayı foto-gerçekçiliğin değil okunabilirliğin belirlediğini bulan yazınla tutarlıdır." },
      { t: "p", x: "Son olarak öz-çalışma modülü, öğrencinin webcam görüntüsünün yanında bir referans iskeletini gösterir ve denemeyi landmark dizileri üzerinden Dinamik Zaman Bükümlemesi ile puanlar; böylece eğitmen olmadan çalışmayı destekleyen, yorumlanabilir bir benzerlik skoru verir." },
      { t: "imgfig", file: "fig4_tutor.png", cap: "Şekil 4.  Öz-çalışma modülü. Sol: referans işaret iskeleti; sağ: öğrencinin canlı görüntüsü. Landmark dizileri üzerindeki Dinamik Zaman Bükümlemesi mesafesi, altta gösterilen benzerlik skorunu verir (burada %30).", ph: "[ Şekil 4 — DZB öğretici split-screen ekran görüntüsü ]" },
    ]},
    { h: "VI.  Sonuç ve Gelecek Çalışmalar", blocks: [
      { t: "p", x: "İzole işaretleri %94.70 doğrulukla tanıyıp yerel bir dil modeliyle Türkçeye dönüştüren, Türkçe metinden sadık bir landmark oynatıcısıyla işaret sentezleyen ve Dinamik Zaman Bükümlemesi tabanlı bir öğretici ile öz-çalışmayı destekleyen çift yönlü bir Türk İşaret Dili platformu sunduk. Temel bir ders, sadık — sade de olsa — bir temsilin, ifade edici ama hatalı bir temsile tercih edilebileceğidir." },
      { t: "p", x: "Gelecek çalışmalar; düzeltilmiş bir el rigi ve dilbilgisel anlam taşıyan el-dışı işaretleyiciler (yüz ifadesi, ağız hareketi) için ARKit tarzı blendshape’ler içeren tam donanımlı, stilize bir avatarı; izole işaretler yerine sürekli, cümle düzeyinde tanımayı; daha geniş bir sözlüğü; kelime başına en temsili referansın medoid tabanlı seçimini; ve Sağır katılımcılarla anlamayı ölçen bir kullanıcı çalışmasını hedefler. Ek bir yön olarak, LLM tabanlı gloss üretimini gerçek işaretçi video kliplerinin getirilmesiyle birleştiren bir “Video Signer” yaklaşımını da araştırıyoruz." },
    ]},
  ],
  refHead: "Kaynakça",
  references: EN.references,
};

(async () => {
  const jobs = [
    [EN, path.join(FIGDIR, "fig1_arch_en.png"), path.join(DESKTOP, "TID_Final_Report_EN.docx")],
    [TR, path.join(FIGDIR, "fig1_arch_tr.png"), path.join(DESKTOP, "TID_Final_Raporu_TR.docx")],
  ];
  for (const [C, fig, out] of jobs) {
    const doc = buildDoc(C, fig);
    const buf = await Packer.toBuffer(doc);
    fs.writeFileSync(out, buf);
    console.log("wrote", out, buf.length, "bytes");
  }
})();
