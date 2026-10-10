# 1. Object detection: models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

### 1A. Closed-set detectors (real-time and high-accuracy)

All AP values are COCO val2017 box AP50:95 unless stated otherwise. "SAB" means re-measured by Roboflow's harness (pycocotools, full val2017, T4 TensorRT FP16, batch 1; rf-detr README).

| Model / framework | What it is | Best verified number (value, benchmark, split, source) | Code licence | Weights licence | Latest release + date | Install route | CPU feasibility | Fine-tuning support |
|---|---|---|---|---|---|---|---|---|
| **RF-DETR** N/S/M/L (+ XL/2XL "Plus") | DETR with a DINOv2 backbone; sizes found by weight-sharing NAS (arXiv 2511.09554, 2025-11-12) | 2XL 60.1, XL 58.6, L 56.5, M 54.7, S 53.0, N 48.4 at 6.8 ms (L) / 2.3 ms (N) on T4 (SAB). RF100-VL AP50:95: 2XL 63.2, L 62.2. Abstract: "first real-time detector to surpass 60 AP on COCO" | Apache-2.0 | N-L detection, Seg N-2XL, Keypoint preview: Apache-2.0. **XL/2XL detection: PML 1.0** (`rfdetr-plus` 1.1.0, 2026-09-22; licence text not fetched) | rfdetr 1.11.0 (2026-09-24); first release 1.0.0 (2025-03-20); active (commit 2026-09-27) | `pip install rfdetr`; transformers `RfDetrForObjectDetection` / `RfDetrForInstanceSegmentation` (HF `Roboflow/rf-detr-{nano..large}`, `rf-detr-seg-*`, created 2026-05-11, Apache-2.0 tags) | 30.5M (N, 384 px) to 33.9M (L); no sub-10M size; latency published only for GPU | Yes: `model.train()` on COCO- or YOLO-format data; detection, segmentation, keypoint preview. NAS for custom datasets is on the paid Roboflow platform |
| **D-FINE** N/S/M/L/X | RT-DETR-style DETR with fine-grained distribution refinement (2410.13842) | X 55.8, L 54.0, N 42.8; Objects365 then COCO: X 59.3, L 57.3 (README). SAB re-measure: X 59.3, L 57.2, N 42.7 | Apache-2.0 | HF `ustc-community/dfine-*` tagged Apache-2.0. **README note (2026-08-19): Objects365-trained checkpoints "may be subject to the Objects365 dataset terms"** | No GitHub releases; last commit 2026-08-19 | transformers `DFineForObjectDetection`; repo `Peterande/D-FINE` | N: 4M params, 7 GFLOPs, 2.12 ms T4 | Repo training configs; transformers Trainer |
| **DEIM** (v1) | Training recipe (Dense O2O, MAL) applied to D-FINE and RT-DETRv2 (2412.04234) | DEIM-D-FINE-X 56.5, L 54.7 (abstract and README); DEIM-RT-DETRv2-X 55.5 (README) | Apache-2.0 (unchanged; last licence commit 2025-07-21) | Apache-2.0 per repo (checkpoints on Google Drive) | No releases; last commit 2026-03-24 | Repo `ShihuaHuang95/DEIM` | Same sizes as D-FINE (N 4M) | Repo; halves training time vs D-FINE (abstract) |
| **DEIMv2** Atto-X | DEIM plus DINOv3 features and a Spatial Tuning Adapter; HGNetv2 for the tiny sizes (2509.20787) | X 57.8 (50.3M), L 56.0, M 53.0, S 50.9 (9.7M), N 43.0, Pico 38.5 (1.5M), Atto 23.8 (0.5M) (README) | **DEIMv2 License, NC since 2026-08-24** (Apache-2.0 before) | **NC** (licence section 5). S-X also carry the **DINOv3 License** through the backbone. HF `Intellindust/DEIMv2_*` repos have no licence tag | No releases; licence commit 2026-08-24 | Repo; transformers `Deimv2ForObjectDetection`; Intel Geti integration (README, 2026-08-13) | Atto 0.5M / 0.8 GFLOPs; N 3.6M / 6.8 GFLOPs | Repo; research only for commercial users |
| **EdgeCrafter** ECDet / ECInsSeg / ECPose | Compact distilled ViTs for edge detection, segmentation and pose (2603.18739; TMLR 2026 per DEIMv2 README) | ECDet-S 51.7 box AP with under 10M params; with Objects365, the X variants reach 59.9 box AP, 49.8 mask AP and 75.9 keypoint AP (abstract) | **EdgeCrafter License, NC since 2026-08-24** | NC (HF `Intellindust/ECDet_L` still tagged Apache-2.0, a stale tag) | No releases | Repo `Intellindust-AI-Lab/EdgeCrafter` | S under 10M | Repo |
| **RT-DETR / RT-DETRv2** | First real-time DETR (2304.08069); v2 adds a bag of freebies (2407.17140) | v1: R50 53.1, R101 54.3; R101 with Objects365 56.2 (abstract). v2: X 54.3, L 53.4, S 48.1 (README) | Apache-2.0 | Apache-2.0 (HF `PekingU/rtdetr_*`, `rtdetr_v2_*`) | Tag v1.0 (2024-04-18); last commit 2026-09-07 | transformers `RTDetrForObjectDetection`, `RTDetrV2ForObjectDetection`. Ultralytics `RTDETR` class is AGPL. The PyPI package `rtdetr` 0.6.5 is a **third-party** project (leeyunjai82), not the official one | v2-S: 20M params, 60 GFLOPs | Official repo configs; transformers |
| **RT-DETRv3** | Hierarchical dense positive supervision (2409.08475) | R18 48.1, R101 54.6 (abstract) | Apache-2.0 | Not verified | Last commit 2025-03-18 | Repo `clxia12/RT-DETRv3` | R18 class | Repo |
| **RT-DETRv4** S/M/L/X | RT-DETR distilled from a DINOv3 ViT-B/16 teacher during training only (2510.25257) | X 57.0, L 55.4, M 53.7, S 49.8 (README; the abstract says S 49.7) at 78-273 FPS on T4 | Apache-2.0 | Repo licence (Google Drive checkpoints). Whether a student distilled from DINOv3 is a DINOv3 "derivative work" is untested | No releases; last commit 2026-07-06 | Repo `RT-DETRs/RT-DETRv4` | Same backbones as D-FINE (HGNetv2) | Repo; training needs the gated DINOv3 teacher |
| **LW-DETR** T-X | ViT-encoder real-time DETR; RF-DETR's base (2406.03459) | X 58.3, L 56.1, T 42.9 (SAB) | Apache-2.0 | Apache-2.0 | Last commit 2025-02-18 (stale) | transformers `LwDetrForObjectDetection`; repo | T: 12.1M | Repo (stale); prefer RF-DETR |
| **Ultralytics YOLO26** n-x | NMS-free dual head, DFL removed, MuSGD optimiser, STAL small-object assignment (2606.03748) | x 57.5 (NMS-free head 56.9), l 55.0, n 40.9 (Ultralytics docs). SAB: X 56.9, N 40.3 | **AGPL-3.0** | **AGPL-3.0** (HF `Ultralytics/YOLO26`); Enterprise licence sold separately | ultralytics 8.4.163 (2026-09-25); YOLO26 shipped in 8.4.0 (2026-01-14) | `pip install ultralytics` | n: 2.4M params, 5.5 GFLOPs, CPU ONNX 38.9 ms (docs; hardware not restated there). Paper: "up to 43% faster CPU ONNX" than YOLO11n | Best tooling: train/val/export to ONNX, OpenVINO, TensorRT, CoreML and LiteRT; detect, instance seg, pose, OBB, classify; docs also list semantic segmentation and depth tasks |
| **YOLO11** | Previous Ultralytics line, released 2024-09-27 (docs) | x 54.7, n 39.5 (Ultralytics docs); SAB X 50.9 | AGPL-3.0 | AGPL-3.0 | Same package (8.3.0 on 2024-09-30) | ultralytics | n: 2.6M, CPU ONNX 56.1 ms (docs) | Same as YOLO26 |
| **YOLO12** | Attention-centric YOLO (2502.12524) | N 40.6 (abstract); x 55.2 (Ultralytics docs) | AGPL-3.0 (`sunsmarterjie/yolov12` and Ultralytics) | AGPL-3.0 | Repo commit 2026-09-15 | ultralytics or repo | n 2.6M | Ultralytics docs: "maintained primarily for benchmarking and research"; only detection weights are published |
| **YOLOv13** | Hypergraph-enhanced YOLO (2506.17733) | Not verified | AGPL-3.0 | AGPL-3.0 | Weights release 2025-06-22; last commit 2025-11-18 | Repo | n-size exists | Repo |
| **YOLOv10** | NMS-free YOLO (2405.14458) | x 54.4 (Ultralytics docs) | AGPL-3.0 | AGPL-3.0 | Last commit 2025-03-14 | ultralytics or repo | n-size | Superseded by YOLO26 |
| **YOLOv9 / YOLOv7** | PGI and GELAN (2402.13616) | Not quoted | Original repo GPL-3.0 (`WongKinYiu/yolov9`, last commit 2024-06-10). **MIT re-implementation `WongKinYiu/YOLO` v1.0 (2025-12-30)**, "official ... with an MIT License" | Weights licence of the MIT repo not verified | See code licence cell | `pip install git+https://github.com/WongKinYiu/YOLO.git` | t/s sizes | Yes (MIT repo) |
| **YOLOX** | Anchor-free YOLO | Not quoted | Apache-2.0 | Apache-2.0 | Release 0.3.0 (2022-04-22); last commit 2025-06-08 | Repo | nano/tiny sizes | Legacy permissive option only |
| **torchvision detectors** | Faster R-CNN, RetinaNet, FCOS, Mask R-CNN, Keypoint R-CNN, SSDLite | Faster R-CNN R50-FPN v2 46.7; Mask R-CNN v2 47.4 box / 41.8 mask; RetinaNet v2 41.5; FCOS 39.2; Faster R-CNN MobileNetV3-FPN 32.8; SSDLite320 21.3 (torchvision 0.29.0 weight metadata) | BSD-3-Clause | torchvision-hosted weights, trained on COCO (no separate licence file checked) | torchvision 0.29.0 (2026-09-02) | `torchvision.models.detection.*` | SSDLite 3.4M / 0.58 GFLOPs; FRCNN-MobileNetV3-320 0.72 GFLOPs; R50 variants 134-333 GFLOPs | Stable, simple API; well below DETR/YOLO accuracy |
| **Co-DETR** | Collaborative hybrid assignment training (2211.12860) | 66.0 COCO **test-dev** and 67.9 LVIS val with ViT-L (abstract) | MIT | Not verified | Last commit 2024-12-29 | mmdet-based repo (frozen stack) | Not feasible (ViT-L, large inputs) | Hard: needs the legacy mmcv stack |
| **DINO (DETR)** | DeNoising anchor DETR (2203.03605) | 63.3 test-dev / 63.2 val2017 with Swin-L and Objects365 (abstract) | Apache-2.0 | Not verified | Last commit 2024-07-31 | detrex or mmdet | No | Legacy stacks |

**Frameworks and maintenance status**

| Framework | Licence | Status (verified) | Use today? |
|---|---|---|---|
| detectron2 | Apache-2.0 | Last release v0.6 (2021-11-15); **not on PyPI** (404); commits continue (2026-08-19, lint-level) | Only to run legacy research code; building against torch 2.14 not verified |
| MMDetection / MMYOLO / mmcv | Apache-2.0 (mmdet, mmcv); **mmyolo GPL-3.0** | mmdet 3.3.0 (2024-01-05), last commit 2024-02-05. mmcv 2.2.0 (2024-04-24); prebuilt mmcv wheels only up to torch 2.4 (Python 3.8-3.12). mmdet 3.3.0 asserts `mmcv < 2.2.0`. mmyolo 0.6.0 (2023-08-15) | Avoid for new work; pin torch 2.1-2.4 if you must reproduce |
| detrex | Apache-2.0 | v0.5.0 (2023-07-21); last commit 2025-09-11 | Research reproduction only |
| PaddleDetection | Apache-2.0 | v2.9.0 (2026-03-19) | Only on PaddlePaddle stacks |
| Roboflow `inference` | Apache-2.0 (classifier) | 1.7.2 (2026-09-25) | Serving RF-DETR and others |

### 1B. Small objects: tiling and slicing

| Tool / trick | What it is | Evidence | Licence | Latest | Notes |
|---|---|---|---|---|---|
| **SAHI** | Slicing Aided Hyper Inference and slice-aware fine-tuning (2202.06934) | Sliced inference added +6.8, +5.1 and +5.3 AP for FCOS, VFNet and TOOD on VisDrone and xView; adding slicing-aided fine-tuning gave +12.7, +13.4 and +14.5 AP cumulative (abstract) | MIT | 0.12.7 (2026-09-25) | Wraps Ultralytics and Hugging Face detectors. README: "torch 2.1.2 is required for mmdet support" |
| supervision `InferenceSlicer` | SAHI-style slicer for any `sv.Detections` producer | None published | MIT | supervision 0.30.5 | Framework-agnostic; works with RF-DETR |
| YOLO26 STAL + P2 head | Small-Target-Aware label assignment is built in; `yolo26-p2.yaml` adds a stride-4 head (docs) | Paper: STAL "guarantees positive coverage for small objects". Independent VisDrone study: YOLO26-x 0.224 vs YOLOv8-x 0.214 mAP50:95 (2605.24831) | AGPL-3.0 | 8.4.x | Gains on dense tiny objects are small; tiling still matters |

### 1C. Rotated / oriented boxes (OBB)

| Model / framework | Best verified number | Code licence | Weights licence | Status | Notes |
|---|---|---|---|---|---|
| **Ultralytics YOLO26-obb** | x: DOTA-v1.0 test mAP50-95 56.7, mAP50 81.7 at 1024 px (docs); paper per docs: "up to +3.4 mAP over YOLO11" | AGPL-3.0 | AGPL-3.0; **trained on DOTA, which is "academic purposes only"** (DOTA site) | Active | Easiest OBB training path |
| MMRotate | n/a | Apache-2.0 | Model zoo | **0.3.4 (2023-02-01); last commit 2024-09-28** | Frozen mmcv stack; avoid for new work |
| LSKNet | DOTA-v1.0 81.85 mAP (abstract, 2303.09030) | **CC BY-NC 4.0** | NC | Repo commit 2026-09-14 | Research only |
| ai4rs (O2-DETR family) | O2-DEIM-R50 80.15%, O2-RTDETR-R50 78.45% and O2-DFINE-L 77.73% AP50 on DOTA 1.0, at 119-132 FPS on a 2080 Ti (2603.15497) | Apache-2.0 | Not verified | Active (commit 2026-09-26) | Oriented real-time DETRs; young repo |

### 1D. Open-vocabulary and zero-shot detection

| Model | What it is | Best verified number | Code licence | Weights licence | Latest / status | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|---|
| **Grounding DINO** (1.0, T/B) | Text-grounded DETR (2303.05499) | 52.5 AP COCO zero-shot; 26.1 mean AP on ODinW zero-shot (abstract) | Apache-2.0 | Apache-2.0 (HF `IDEA-Research/grounding-dino-{tiny,base}`) | Last commit 2024-08-12 | transformers `GroundingDinoForObjectDetection` | Tiny is slow on CPU (Swin-T plus BERT) | Community only; use MM-Grounding-DINO or LLMDet |
| **Grounding DINO 1.5 / 1.6 Pro and Edge** | Proprietary successors (2405.10300) | 1.6 Pro: 55.4 AP COCO zero-shot, 57.7 LVIS-minival (README) | Apache-2.0 (API client only) | **API only**: needs a DeepDataSpace API token, paid quota | Last commit 2025-01-21 | REST API | n/a | No |
| **DINO-X Pro** | Unified open-world model (2411.14347) | 56.0 AP COCO zero-shot; 59.8 LVIS-minival (abstract) | n/a | **API only** (via the same IDEA DeepDataSpace platform) | n/a | API | n/a | No |
| MM-Grounding-DINO | Open reproduction of Grounding DINO (2401.02361) | Tiny "outperforms the Grounding-DINO-Tiny baseline" (abstract) | Apache-2.0 | Apache-2.0 (mmdet release) | In mmdet (frozen) | transformers `MMGroundingDinoForObjectDetection` | as GDINO | Via mmdet (legacy) |
| **LLMDet** T/B/L | Grounding-DINO-style detector co-trained with LLM captions (2501.18954) | Qualitative in abstract only | Apache-2.0 | Apache-2.0 (HF `iSEE-Laboratory/llmdet_*`) | HF 2025-05-30 | transformers zero-shot detection (HF library tag `transformers`) | as GDINO | Repo |
| **OWLv2** | CLIP-based open-vocabulary detector with self-training (2306.09683) | OWL-ST L/14: LVIS rare 44.6 AP (abstract). SAM 3 README reports OWLv2* at 46.1 COCO box AP and 45.5 LVIS box AP | Apache-2.0 | Apache-2.0 (HF `google/owlv2-*`) | HF 2023-10-13 | transformers `Owlv2ForObjectDetection` | B/16 feasible for offline labelling | Image-conditioned (one-shot) queries supported |
| **OmDet-Turbo** | Real-time open-vocabulary DETR (2403.06892) | 30.1 AP on ODinW; 100.2 FPS with TensorRT (abstract) | Apache-2.0 | Apache-2.0 | HF 2024-09-24 | transformers `OmDetTurboForObjectDetection` | Swin-T | Repo |
| YOLO-World | Real-time open-vocabulary YOLO (2401.17270) | 35.4 AP on LVIS at 52.0 FPS on V100 (abstract) | **GPL-3.0** (AILab-CVC) | GPL / AGPL (Ultralytics port) | Last commit 2025-02-26 | ultralytics `YOLOWorld` | Yes (s size) | Yes |
| **YOLOE / YOLOE-26** | "Seeing anything": text, visual or prompt-free (2503.07465) | YOLOE-26x: 40.6 AP LVIS minival with text prompts, 38.5 visual, 31.1 prompt-free (Ultralytics docs, non-e2e figures) | AGPL-3.0 | AGPL-3.0 | Via ultralytics | ultralytics | Yes | Yes |
| **Florence-2** B/L | Seq2seq unified vision model (2311.06242): OD, dense captions, OCR, grounding | No numbers in abstract | MIT | MIT (HF `microsoft/Florence-2-*`) | HF 2024-06-15 | transformers `Florence2ForConditionalGeneration` (native in 5.x) | **0.23B / 0.77B (card): CPU feasible** | Yes (seq2seq fine-tune) |
| **SAM 3 / SAM 3.1** | Concept-prompt detect, segment and track (2511.16719). SAM 3.1 "Object Multiplex" (2026-03-27) makes multi-object tracking faster | Zero-shot box AP: COCO 56.4, LVIS 53.6; LVIS mask AP 48.5 (SAM 3 README). Fine-tuned on RF100-VL: 61.6 AP50:95 (paper Table 36 via rf-detr README) | **SAM License** (2025-11-19): custom-commercial with ITAR, military and nuclear bans; derivatives bound | Same; HF `facebook/sam3`, `facebook/sam3.1` gated (manual) | sam3 0.1.4 on PyPI (2026-05-26, Meta); repo commit 2026-09-18 | transformers `Sam3Model`, `Sam3VideoModel`, `Sam3TrackerModel`; `pip install sam3` (Python 3.12+ per README) | About 850M params (paper figure quoted in rf-detr README): **GPU** | Fine-tuning code in repo |
| Rex-Omni | 3B MLLM detecting via next-point prediction (2510.12798) | Not verified | IDEA License 1.0 | **NC** ("solely for your non-commercial research purposes") | HF 2025-10-09 | transformers | No | n/a |
| Moondream 3 (preview) / 3.1 | 9B-total, 2B-active MoE VLM with `detect()` | Not verified | n/a | **BUSL-1.1 with an additional use grant** ("No Third-Party Service"); 3.1 uses "moondream-model-license-1.0" | 3.1 HF 2026-06-30 | HF | No | n/a |

### Decision guidance: detection
- **Try first (permissive, commercial):**
  1. **RF-DETR-M or -L** (`pip install rfdetr`; Apache). It has the best maintained training API and leads on the fine-tuning benchmark: RF100-VL AP50:95 62.2 for L vs 59.3 for YOLO26-L and 61.6 for D-FINE-L (SAB).
  2. If you need transformers-native code or a sub-5M model: **D-FINE-N/S** (4M/10M; `DFineForObjectDetection`) or **RT-DETRv2** (`RTDetrV2ForObjectDetection`).
- **If AGPL is acceptable** (internal tools, or open-sourcing your service): **YOLO26**. It has the widest export surface, CPU ONNX figures, and OBB, pose and segmentation in one API.
- **Then try:** RT-DETRv4 for maximum Apache accuracy, if your legal team accepts DINOv3-teacher distillation. DEIM (v1) if you want the D-FINE architecture trained in half the time.
- **Avoid:**
  - DEIMv2 and EdgeCrafter for commercial work (NC since 2026-08-24).
  - RF-DETR XL/2XL unless PML 1.0 is cleared.
  - YOLO12 for production (Ultralytics' own advice).
  - New projects on MMDetection, MMRotate or detectron2.
  - The PyPI `rtdetr` package as if it were the official one.
- **Labelling bootstrap:** run SAM 3 (custom licence) or an open-vocabulary detector (OWLv2, LLMDet, Grounding DINO; Apache) to pre-label, then fine-tune a real-time detector. RF-DETR's abstract makes this argument: open-vocabulary detectors "often fail to generalize to real-world datasets with out-of-distribution classes".
- **Trust re-measured numbers over vendor tables.** SAB re-measurement puts YOLO11-X at 50.9 vs 54.7 in Ultralytics docs, and YOLO26-X at 56.9 vs 57.5 (SAB note: "may differ from vendor-reported figures").

### Licence traps: detection
- **DEIMv2 / EdgeCrafter:** NC since 2026-08-24; mirrors still say Apache.
  - The new licence says rights granted under an earlier version "remain governed by the License applicable to that release". Whether copies obtained before 2026-08-24 stay Apache is a legal question.
- **Objects365-pretrained weights:** the D-FINE README warns they are not commercially cleared.
- **DOTA-trained OBB weights:** the DOTA dataset is "academic purposes only" and its images are subject to Google Earth terms.
- **Ultralytics (YOLOv8/11/12/26, YOLOE, YOLO-World port, RT-DETR wrapper):** AGPL-3.0 covers the code and the weights you train with it.
- **YOLO-World upstream:** GPL-3.0.
- **YOLOv9 original repo:** GPL-3.0; the MIT re-implementation is `WongKinYiu/YOLO`.
- **RF-DETR XL/2XL:** PML 1.0.
- **Rex-Omni:** NC.
- **Moondream:** BUSL.

### Metrics: detection (libraries and pitfalls)
- **Libraries:**
  - `pycocotools` 2.0.11 (FreeBSD; PyPI now points to the maintained `ppwwyyxx/cocoapi` fork).
  - `faster-coco-eval` 1.8.0 (Apache-2.0; drop-in `init_as_pycocotools()`; adds LVIS and CrowdPose support).
  - `torchmetrics` 1.9.0 `MeanAveragePrecision(backend="pycocotools" | "faster_coco_eval")`; the default is pycocotools, and torchmetrics' own tests find the two "identical".
- **Pitfalls (verified in `cocoeval.py`):**
  1. `COCOeval(..., iouType='segm')` is the **default**; pass `'bbox'` explicitly.
  2. `maxDets=[1,10,100]`, and AP is reported at 100. Dense scenes (crowds, cells, aerial) are capped silently; raise it (torchmetrics accepts exactly 3 values).
  3. Area buckets are small < 32^2, medium 32^2-96^2, large > 96^2, and they are computed on each ground-truth annotation's **`area` field**. COCO's field is segment area, so datasets converted from YOLO format get box-area buckets.
  4. Every ground-truth annotation needs an `iscrowd` key (read directly in `computeIoU`).
  5. Report AP50:95 and AP50 separately. A "mAP50" headline, common in YOLO tooling, is not COCO AP.
  6. LVIS uses federated evaluation (use the LVIS API or faster-coco-eval's LVIS mode, not plain COCOeval).
  7. TIDE (2008.08115) decomposes errors into classification, localisation, duplicate, background and missed.
