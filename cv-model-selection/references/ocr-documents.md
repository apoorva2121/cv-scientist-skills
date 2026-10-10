# 6. OCR and document understanding: models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

### 6A. Classic OCR engines (detection plus recognition)

| Engine | Best verified claim | Code licence | Weights licence | Latest release + date | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|
| **PaddleOCR: PP-OCRv5 / PP-OCRv6** | PP-OCRv5 (2603.24373): "merely 5 million parameters" and "competitive with many billion-parameter VLMs on standard OCR benchmarks". PP-OCRv6 (README, 3.7.0): "+4.6% detection and +5.1% recognition accuracy over PP-OCRv5", "5.2x CPU inference speedup end-to-end", 50 languages in one model (vendor claims) | Apache-2.0 | Apache-2.0 (HF `PaddlePaddle/PP-OCRv5_*`, `PP-OCRv6_*`, including ONNX exports, 2026-06) | paddleocr 3.7.0 (2026-06-11); paddlepaddle 3.3.1 (2026-03-24); active | `pip install paddleocr paddlepaddle`; **RapidOCR 3.9.2** (Apache, ONNX Runtime, no Paddle); transformers has detection classes `PPOCRV5{Mobile,Server}DetForObjectDetection` and `PPOCRV6{Small,Medium}DetForObjectDetection` | **Yes (designed for CPU)** | PaddleX 3.7.2 / PaddleOCR training |
| **docTR** | Not verified | Apache-2.0 | Apache-2.0 (repo) | python-doctr 1.1.0 (2026-08-21); Python 3.11+ | pip | Yes | Yes (reference scripts) |
| **Tesseract 5** | Not verified | Apache-2.0 | Apache-2.0 (traineddata) | 5.5.3 (2026-07-24) | System package, plus `pytesseract` 0.3.13 or `tesserocr` 2.11.0 (MIT) | Yes | tesstrain |
| EasyOCR | Not verified | Apache-2.0 | Apache-2.0 | **1.7.2 (2024-09-24)**; last commit 2025-12-05 (README) | pip | Yes | Stale |
| TrOCR | Transformer OCR for single text lines (2109.10282) | MIT (unilm) | HF `microsoft/trocr-*` **untagged** | HF 2022 | transformers `VisionEncoderDecoderModel` | base: yes | Yes (line-level) |

### 6B. End-to-end document parsing (VLM-based)

OmniDocBench v1.6 end-to-end **Overall** is `((1 - TextEdit) x 100 + Table TEDS + Formula CDM) / 3` (README, table updated 2026-09-11). Maintainer caveat: OmniDocBench is run by opendatalab, the same group as MinerU.

| Model | Size | OmniDocBench v1.6 Overall (README) | Code licence | Weights licence | Latest | Install route | CPU feasibility |
|---|---|---|---|---|---|---|---|
| TeleOCR | 1.2B | 96.91 | n/a | Apache-2.0 (HF tag only; `StarDoc-AI/TeleOCR`, 2026-08-14) | new | transformers | Small; GPU preferred |
| OvisOCR2 | 0.8B | 96.47 | n/a | Apache-2.0 (HF tag only; `ATH-MaaS/OvisOCR2`) | 2026-07-13 | transformers | Small |
| **PaddleOCR-VL-1.6** | 0.9B | **96.34** | Apache-2.0 | **Apache-2.0** (HF `PaddlePaddle/PaddleOCR-VL-1.6`, GGUF also) | 2026-05-27 (PaddleOCR 3.6.0) | `paddleocr`; transformers `PaddleOCRVLForConditionalGeneration` | 0.9B; GGUF on CPU |
| **MinerU2.5-Pro** | 1.2B | 95.75 (abstract: 95.69) | **MinerU Open Source License**: Apache-2.0 plus a separate licence above 100M MAU or USD 20M monthly revenue, plus mandatory attribution for online services | **Apache-2.0** (HF `opendatalab/MinerU2.5-Pro-2605-1.2B`) | mineru 4.0.7 (2026-09-23) | `pip install mineru` | 1.2B |
| **GLM-OCR** | 0.9B | 95.22 | n/a | **MIT** (HF `zai-org/GLM-OCR`) | 2026-01-30 | transformers `GlmOcrForConditionalGeneration` | 0.9B |
| PaddleOCR-VL-1.5 / PaddleOCR-VL | 0.9B | 94.93 / 94.18 (VL-1.5 abstract: 94.5% on OmniDocBench **v1.5**) | Apache-2.0 | Apache-2.0 | 2026-01-28 / 2025-10-16 | as above | 0.9B |
| Unlimited-OCR (Baidu) | 3B | 94.00 | n/a | MIT (HF tag only) | 2026-06-19 | transformers | 3B |
| Qianfan-OCR (Baidu) | 4B | 93.90 | n/a | Apache-2.0 (HF tag only) | 2026-03-18 | transformers `QianfanOCRForConditionalGeneration` | 4B |
| MinerU2.5 (2509) | 1.2B | 93.04 | MinerU licence | **AGPL-3.0** (HF `opendatalab/MinerU2.5-2509-1.2B`) | 2025-09-17 | mineru | 1.2B |
| dots.ocr | 3B | 90.77 | MIT | MIT (HF) | 2025-07-30 | repo | 3B |
| DeepSeek-OCR 2 / DeepSeek-OCR | 3B | 90.25 (OCR 2) | MIT | **OCR 2: Apache-2.0; OCR 1: MIT** (HF) | 2026-01-27 / 2025-10-17 | transformers `DeepseekOcr2ForConditionalGeneration` | GPU |
| HunyuanOCR | 1B | 89.95 | n/a | **Tencent Hunyuan Community: "DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA"** | 2025-11-18 | transformers | 1B |
| Qwen3-VL-235B (general VLM) | 235B | 89.78 | Apache-2.0 | Apache-2.0 | 2025-09 | transformers / vLLM | No |
| olmOCR (7B row) | 7B | 85.74 | Apache-2.0 | Apache-2.0 (`allenai/olmOCR-2-7B-1025`) | olmocr 0.4.27 (2026-03-12) | `pip install olmocr` (vLLM) | No |
| Nanonets-OCR-s / OCR2-3B | 3B | 83.61 (OCR-s) | n/a | **No licence tag; the base model is Qwen2.5-VL-3B, whose Qwen Research licence is NC** | 2025-10-13 | transformers | 3B |
| Marker (pipeline) | n/a | 78.44 (version not stated) | **Apache-2.0 since 2026-07-17** (GPL-3.0 before) | **Modified AI Pubs OpenRAIL-M: free for research, personal use and startups under $5M funding or revenue** (README) | marker-pdf 2.0.0 (2026-07-20) | pip | Slow on CPU |

**Other models not in that table.**
- **Chandra OCR 2** (Datalab): code Apache-2.0. Weights are a "modified OpenRAIL-M license: free for research, personal use, and startups under $2M funding/revenue. Cannot be used competitively with our API" (HF card).
- **Surya OCR 2** (2026-05-14): code Apache-2.0 since 2026-05-14. Weights are modified OpenRAIL-M with a $5M cap.
- **LightOnOCR-2-1B**: Apache-2.0; "state-of-the-art results on OlmOCR-Bench while being 9x smaller" (2601.14251); transformers `LightOnOcrForConditionalGeneration`.
- **GOT-OCR 2.0**: HF `stepfun-ai/GOT-OCR-2.0-hf` Apache-2.0; the GitHub repo has **no LICENSE file**; transformers `GotOcr2ForConditionalGeneration`.
- **Nougat**: code MIT, **weights CC-BY-NC-4.0**; nougat-ocr 0.1.17 (2023).
- **Donut**: MIT code and weights.
- **Florence-2**: MIT; OCR task prompts.
- **granite-docling-258M**: Apache-2.0.

### 6C. Pipelines, layout and tables

| Component | Licence (code / weights) | Latest | Notes |
|---|---|---|---|
| **Docling** (2408.09869) | MIT / layout `docling-layout-heron` Apache-2.0; `docling-models` (TableFormer) CDLA-Permissive-2.0 + Apache-2.0 | docling 2.130.0 (2026-09-22), very active | Permissive end-to-end PDF-to-structured pipeline; can call VLMs |
| DocLayout-YOLO (2410.12628) | **AGPL-3.0 code** (Ultralytics-derived; doclayout-yolo 0.0.4, 2025-06-10) / HF weights tagged Apache-2.0 (a conflict, since they were produced with AGPL code) | Last commit 2025-04-14 | Layout detection trained on the DocSynth-300K synthetic set |
| PP-DocLayoutV3 | Apache-2.0 / Apache-2.0 | HF 2026-01-20 | transformers `PPDocLayoutV3ForObjectDetection` |
| Surya layout 2 | Apache-2.0 / OpenRAIL-M (capped) | 2026-07-08 | |
| **Table Transformer (TATR)** | MIT / MIT (HF `microsoft/table-transformer-*`) | **Repo archived**; v1.0.0 (2023-02-17) | transformers `TableTransformerForObjectDetection`; PubTables-1M (2110.00061) |
| LayoutLMv3 | MIT code / **CC-BY-NC-SA-4.0 weights** | 2022 | NC trap (see R3) |

### Decision guidance: OCR and documents
- **Printed text, CPU, permissive:** PaddleOCR PP-OCRv5/v6 (via `paddleocr`, or RapidOCR for ONNX-only). Then docTR. Tesseract only for simple clean scans.
- **Full-page PDF to Markdown or JSON, permissive:**
  1. **PaddleOCR-VL-1.6** (Apache, 0.9B) or **GLM-OCR** (MIT, 0.9B).
  2. **MinerU2.5-Pro** (Apache weights; the code has MAU and revenue thresholds).
  3. **Docling** when you need a pipeline with provenance, bounding boxes and table structure.
- **Avoid in products without a licence check:** Marker, Surya and Chandra weights (revenue-capped OpenRAIL-M), MinerU2.5-2509 (AGPL), HunyuanOCR (territory-restricted), Nanonets-OCR (Qwen Research base), Nougat and LayoutLMv3 (NC), DocLayout-YOLO (AGPL code).
- **Evidence that specialists beat generalists here:** on OmniDocBench v1.6, 0.9B specialists (96.34) beat Qwen3-VL-235B (89.78) and GPT-5.2 (86.59). PP-OCRv5's paper argues a 5M two-stage pipeline stays competitive with billion-parameter VLMs.

### Metrics: OCR and documents
- **CER / WER:** `jiwer` 4.0.0 (Apache) or torchmetrics `CharErrorRate` / `WordErrorRate`. Fix Unicode normalisation (NFC/NFKC), case, whitespace and punctuation handling **before** scoring, and report them. Reading-order errors inflate WER on multi-column pages; OmniDocBench scores reading order separately.
- **TEDS:** tree-edit-distance similarity for tables (1911.10683). TEDS-S scores structure only.
- **ANLS:** from ST-VQA (1905.13648) and used by DocVQA (2007.00398). The 0.5 similarity threshold is the commonly used setting (not re-verified here).
- **OmniDocBench versions:** v1.0, v1.5 and v1.6 scores are **not comparable**. v1.5 changed the Overall formula and the data (2025-09-25); v1.6 "rectif[ied] element-matching biases" and added a Hard subset (2604.04771).
