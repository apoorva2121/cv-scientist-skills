# R4 - Task-specific models and frameworks, per task family

Verified inventory for the Computer Vision Scientist skill, area 4: which task model or framework to use for each vision task, what it scores, what licence its code and weights carry, and whether it is still maintained. Backbones and general-purpose encoders (DINOv2/v3, SigLIP 2, CLIP, ConvNeXt, EVA, Perception Encoder, pathology encoders, EO encoders) are in `R3-backbones-foundation-models.md`; here they appear only as the backbone a task model uses.

**Checked 2026-09-27.** How each fact was checked:
- **Packages:** PyPI JSON API (`info.version`, `license`/`license_expression`, upload time of the first file of that release).
- **Repos:** licence text read from `raw.githubusercontent.com/<repo>/<default-branch>/LICENSE*`; archive flag, latest release and latest commit read from the repo page and its `releases.atom` / `commits/<branch>.atom` feeds (no GitHub API quota used). Licence *change dates* come from the `commits/<branch>/LICENSE.atom` history, with the old text fetched at the older commit SHA.
- **Weights:** Hugging Face API (`cardData.license`, `license_name`, tags, `gated`, `createdAt`), plus the card text or the upstream README where they disagree. **Where HF metadata and upstream disagree, this report follows the more restrictive one** and lists the conflict under licence traps.
- **Papers:** arXiv API (ID, title, v1 date); every quoted claim comes from the abstract unless another source is named.
- **Benchmarks:** only from primary sources: the paper abstract, the official README or model zoo, or the official docs. The source is named in each cell. "SAB" means Roboflow's single-artifact benchmark harness, which re-measures every model with pycocotools on the full COCO val2017 split and T4 TensorRT FP16 at batch 1 (rf-detr README).
- **Library classes:** read from an installed `transformers` 5.17.0 (`dir(transformers)`), plus torchvision 0.29.0 weight metadata in the source files at tag v0.29.0.

**Environment baseline (PyPI, 2026-09-27):** torch 2.14.0 and torchvision 0.29.0 (2026-09-02); transformers 5.17.0 (2026-09-09); timm 1.0.30 (2026-09-22); ultralytics 8.4.163 (2026-09-25); rfdetr 1.11.0 (2026-09-24); supervision 0.30.5 (2026-09-22); trackers 2.6.1 (2026-09-25); sahi 0.12.7 (2026-09-25); pycocotools 2.0.11 (2025-12-15, the ppwwyyxx fork); faster-coco-eval 1.8.0 (2026-08-24); torchmetrics 1.9.0 (2026-03-09); onnxruntime 1.30.0 (2026-09-10); openvino 2026.4.0 (2026-09-16).

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

---

## Most decision-changing findings (2026-09-27)

1. **DEIMv2 and EdgeCrafter became non-commercial on 2026-08-24.**
   - DEIMv2's `LICENSE` (Apache-2.0 since 2025-09-19) was replaced by a custom "DEIMv2 License" in commit "docs: update license information". It grants rights "solely for Non-Commercial Purposes", and section 5 explicitly covers model weights. The EdgeCrafter repo (ECDet/ECInsSeg/ECPose) changed the same day.
   - Stale mirrors still say Apache-2.0: the HF `Intellindust/ECDet_L` tag and the transformers-format mirrors `harshaljanjani/DEIMv2_*_Transformers` (created 2026-04-16, when upstream was still Apache).
   - DEIM (v1) stays Apache-2.0 (last licence commit 2025-07-21).
   - Even before the change, DEIMv2 S/M/L/X embed DINOv3-derived backbones, which carry the DINOv3 License.
2. **YOLO26 exists and is the current Ultralytics line** (ultralytics 8.4.0, 2026-01-14; paper arXiv 2606.03748, 2026-06-02).
   - It is NMS-free (optional one-to-one head), drops DFL and scores 40.9-57.5 COCO mAP50-95 across n to x (Ultralytics docs).
   - It is AGPL-3.0 like all Ultralytics weights. YOLO27 is announced but "not yet available" (docs).
   - Independent re-measurement matters. SAB measures YOLO26-X at 56.9, which matches the vendor's NMS-free e2e column rather than the headline 57.5. An independent study (2605.24831) found YOLO26's gain over YOLOv8 shrinks to 0.224 vs 0.214 mAP50:95 on VisDrone small objects, and that YOLOv8-s was faster on GPU (6.92 vs 8.38 ms).
3. **RF-DETR is the strongest permissive real-time detector with a maintained training package.**
   - RF-DETR-L scores 56.5 COCO AP50:95 at 6.8 ms on T4 (SAB).
   - N-L detection, all segmentation sizes and the keypoint preview are Apache-2.0. **XL and 2XL detection (58.6 / 60.1 AP) are under "PML 1.0"** via `rfdetr[plus]`.
   - RF-DETR is now in transformers 5.17 (`RfDetrForObjectDetection`, `RfDetrForInstanceSegmentation`).
4. **D-FINE added a dataset-licence caveat (2026-08-19).** Objects365-pretrained checkpoints (`*_obj365`, `*_obj2coco`) "should not be assumed to be commercially cleared under the D-FINE license". The same logic applies to every Objects365-pretrained detector (RT-DETR O365 weights, DINO-DETR, LW-DETR, EdgeCrafter X).
5. **OpenMMLab is effectively frozen.**
   - MMDetection's last release was 3.3.0 (2024-01-05) and its last commit 2024-02-05. MMSegmentation's last release was 1.2.2 (2023-12-14) and its last commit 2024-03-22. MMRotate's last release was 0.3.4 (2023-02-01).
   - mmdet 3.3.0 asserts `mmcv < 2.2.0`, but prebuilt mmcv wheels on download.openmmlab.com exist only as mmcv 2.2.0, and only up to torch 2.4 (no index page for torch 2.5 or 2.6). The current torch is 2.14.
   - Use transformers, Ultralytics or model-specific repos instead. For pose, use `rtmlib` (no mmcv).
6. **Several popular weights are NC, and some mirrors say otherwise:**
   - SegFormer (NVIDIA Source Code License: "research or evaluation purposes only").
   - Depth Anything V2 Base/Large and DA3 Large/Giant (CC-BY-NC-4.0). Upstream lists DA3-LARGE-1.1 as CC BY-NC 4.0, but HF tags it Apache-2.0.
   - VideoMAE / VideoMAE V2 (CC-BY-NC-4.0).
   - Sapiens v1 (CC-BY-NC-4.0), UniDepth (CC BY-NC 4.0), DUSt3R/MASt3R (CC BY-NC-SA 4.0), Nougat (CC-BY-NC-4.0), LayoutLMv3 (CC-BY-NC-SA-4.0).
   - Sonata point-cloud weights (upstream CC-BY-NC 4.0; HF mirror `Pointcept/Sonata` tagged MIT).
   - InsightFace face models ("non-commercial research purposes only").
   - Qwen2.5-VL-3B (Qwen Research licence).
   - VISTA3D weights (NVIDIA NCLS v1).
7. **Permissive replacements now exist for most of those:**
   - Detection and segmentation: RF-DETR, D-FINE, RT-DETRv2/v4.
   - Universal segmentation: EoMT (MIT).
   - Depth: Depth Anything V2-Small, DA3-SMALL/BASE/METRIC-LARGE/MONO-LARGE, MoGe-1/2/3, MapAnything-apache, Pi3.
   - Video: V-JEPA 2 (MIT).
   - VLMs: Qwen3-VL, Qwen3.5/3.6 and Gemma 4, all Apache-2.0. Gemma 4 moved from Gemma terms to Apache-2.0; Qwen3.5 (2026-02) is natively multimodal.
   - Document parsing: PaddleOCR-VL 1.5/1.6 and MinerU2.5-Pro (Apache-2.0).
8. **OCR licence shifts in 2026:**
   - Surya's code moved from GPL-3.0 to Apache-2.0 (2026-05-14), and Marker's did the same (2026-07-17). **Their weights, and Chandra's, remain a modified OpenRAIL-M with revenue caps:** $5M for Surya/Marker, $2M plus a no-competing-API clause for Chandra.
   - MinerU code is "Apache-2.0 plus additional terms": a separate licence is needed above 100M MAU or USD 20M monthly revenue, and online-service attribution is required. MinerU2.5-2509 weights are AGPL-3.0, while MinerU2.5-Pro weights are Apache-2.0.
   - HunyuanOCR's licence excludes the EU, UK and South Korea.
9. **The document-parsing leaderboard moved from pipelines to sub-2B VLMs.** On OmniDocBench v1.6 (end-to-end Overall, README updated 2026-09-11):
   - PaddleOCR-VL-1.6 (0.9B) scores 96.34, MinerU2.5-Pro (1.2B) 95.75 and GLM-OCR (0.9B) 95.22.
   - General VLMs score lower: Qwen3-VL-235B 89.78 and Gemini 3 Pro 92.91. The Marker pipeline scores 78.44.
   - OmniDocBench is maintained by opendatalab, the MinerU team, so re-check on your own documents.
10. **Zero-shot VLM or open-vocabulary detection is not a substitute for fine-tuning off-distribution.**
    - RF100-VL (2505.20612): GroundingDINO and Qwen2.5-VL "achieve less than 2% zero-shot accuracy on challenging medical imaging datasets".
    - VLMs "significantly underperform CLIP" on standard image classification (2405.18415).
    - But zero-shot open-vocabulary detectors beat classic few-shot detectors on common categories: GroundingDINO scores 48 AP zero-shot vs 33 AP for few-shot SOTA on COCO (2312.14494).
11. **Metric-library traps that silently change numbers:**
    - `torchmetrics.segmentation.MeanIoU` (1.9.0) averages per-image IoU over (image, class) pairs where the class is present. It has no `ignore_index` argument and defaults to `input_format="one-hot"`. This is **not** the dataset-level accumulated mIoU that ADE20K and Cityscapes results use.
    - `pycocotools.COCOeval` defaults to `iouType='segm'`.
    - COCO small/medium/large buckets use each ground-truth annotation's `area` field (segment area for COCO), not box area.
    - `sv.ByteTrack` is removed in supervision's `develop` branch (scheduled for 0.31.0); use `trackers`.
    - The PyPI package `trackeval` is a fork (kovalp/TrackEval), not the official JonathonLuiten/TrackEval, which is not on PyPI and was last committed 2022-11-29.

---

## 1. Object detection

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

---

## 2. Segmentation

### 2A. Semantic, instance and panoptic

| Model / framework | What it is | Best verified number | Code licence | Weights licence | Latest release + date | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|---|
| **EoMT** (DINOv2 / DINOv3) | Encoder-only Mask Transformer: a plain ViT does the segmentation, with no adapter or pixel decoder (2503.19108, CVPR 2025) | With DINOv3: "up to 58.9 PQ on COCO with EoMT-L at 1280x1280" and "up to 59.5 mIoU on ADE20K with EoMT-L at 512x512" (README). Abstract: "up to 4x faster with ViT-L" than adapter-plus-decoder designs | MIT | HF `tue-mps/*` tagged MIT. **DINOv3 variants ship delta weights relative to DINOv3 (README), so the DINOv3 License and its gate apply**; the DINOv2 variants are Apache-backbone | No releases; active (commit 2026-09-24); VidEoMT video extension (2602.17807, CVPR 2026) | transformers `EomtForUniversalSegmentation`, `EomtDinov3ForUniversalSegmentation`; video `VideomtForUniversalSegmentation` | ViT-S/B variants exist (`coco_panoptic_eomt_small_640_2x`); L is GPU | Lightning training in repo |
| **Mask2Former** | Masked-attention universal segmenter (2112.01527) | 57.8 PQ COCO panoptic, 50.1 AP COCO instance, 57.7 mIoU ADE20K (abstract) | MIT | HF `facebook/mask2former-*` tagged "other"; card points to the repo licence (MIT) | **Repo archived**; last commit 2022-05-20 | transformers `Mask2FormerForUniversalSegmentation` | Swin-T/R50 variants heavy on CPU | transformers Trainer (no upstream support) |
| **OneFormer** | Task-token universal segmenter trained once for all three tasks (2211.06220) | "Outperforms specialized Mask2Former models across all three segmentation tasks on ADE20k, CityScapes, and COCO" (abstract) | MIT | MIT (HF `shi-labs/oneformer_*`) | Last commit 2023-07-05 | transformers `OneFormerForUniversalSegmentation` | Heavy | transformers |
| **SegFormer** B0-B5 | Hierarchical MiT encoder with an all-MLP decoder (2105.15203) | B4 50.3 mIoU ADE20K with 64M params; B5 84.0 mIoU Cityscapes val (abstract) | **NVIDIA Source Code License: "research or evaluation purposes only"** | **NC** (HF `nvidia/segformer-*`, `nvidia/mit-b*` tagged "other", linking the same licence) | Last commit 2023-06-13 | transformers `SegformerForSemanticSegmentation`; smp `Segformer` | B0 is small and CPU-friendly (params not re-verified) | Easy, but NC |
| **UperNet + backbone** | Multi-level FPN head (1807.10221) on ConvNeXt, Swin or DINOv2/v3 | Depends on backbone (see R3 for DINOv3 frozen ADE20K) | Head: transformers/smp Apache/MIT | Backbone licence governs | transformers 5.17 | transformers `UperNetForSemanticSegmentation`; smp `UPerNet` | Backbone-dependent | Yes |
| **RF-DETR-Seg** N-2XL | Instance segmentation head on RF-DETR | 2XL 49.9, XL 48.8, L 47.1, N 40.3 COCO mask AP50:95, at 21.8 / 13.5 / 8.8 / 3.4 ms on T4 (SAB) | Apache-2.0 | **Apache-2.0 for all Seg sizes, including XL/2XL** (README) | rfdetr 1.11.0 | `rfdetr`; transformers `RfDetrForInstanceSegmentation` | 33.6-38.6M params | Yes |
| **YOLO26-seg** | Ultralytics instance segmentation | x 47.0 mask / 56.5 box mAP50-95 e2e (docs); SAB 46.8 | AGPL-3.0 | AGPL-3.0 | ultralytics 8.4.x | ultralytics | n 2.7M, CPU ONNX 53.3 ms (docs) | Yes |
| **Mask R-CNN** (torchvision) | Classic two-stage instance segmenter | v2 R50-FPN 47.4 box / 41.8 mask AP (torchvision metadata) | BSD-3-Clause | torchvision weights | torchvision 0.29.0 | `torchvision.models.detection.maskrcnn_resnet50_fpn_v2` | 46M params, 334 GFLOPs | Yes, simplest baseline |
| **segmentation-models-pytorch** (smp) | 12 encoder-decoder architectures (Unet, Unet++, FPN, DeepLabV3+, UPerNet, Segformer, DPT...) and "800+ pretrained encoders" including timm `tu-*` (README) | n/a | MIT, **except `encoders/mix_transformer.py` (the `mit_b*` encoders), which carries the NVIDIA Source Code License header** | Per encoder (smp-hub mirrors) | 0.5.0 (2025-04-17); commits continue (2026-09-15, dependency bumps) | `pip install segmentation-models-pytorch` | Pick a MobileNet/EfficientNet/MobileOne encoder | Yes; the default for small-data semantic seg |
| MMSegmentation | OpenMMLab semantic seg | n/a | Apache-2.0 | Model zoo | 1.2.2 (2023-12-14); last commit 2024-03-22; mmcv < 2.2.0 | Frozen | n/a | Legacy only |

### 2B. Promptable and interactive segmentation

| Model | Best verified number | Code licence | Weights licence | Latest | Install route | CPU feasibility | Notes |
|---|---|---|---|---|---|---|---|
| **SAM 2 / 2.1** | Image segmentation "more accurate and 6x faster than SAM"; video with "3x fewer interactions" (2408.00714) | Apache-2.0 | Apache-2.0 (HF `facebook/sam2.1-hiera-{tiny,small,base-plus,large}`) | sam2 1.1.0 (2024-12-21); last commit 2024-12-16 | transformers `Sam2Model`, `Sam2VideoModel`; `pip install sam2` | Hiera-tiny is the CPU option | Permissive default for promptable image and video masks |
| **SAM 3 / 3.1** | See 1D. Concept prompts (noun phrase or exemplar) return all instances; "doubles the accuracy of existing systems" in promptable concept segmentation (abstract) | SAM License | SAM License, gated | 0.1.4 PyPI (2026-05-26); SAM 3.1 checkpoints 2026-03-27 | transformers `Sam3*`; `pip install sam3` | GPU (about 850M) | Custom-commercial; bans ITAR, military and nuclear uses |
| SAM (v1) | Original promptable model (2304.02643) | Apache-2.0 | Apache-2.0 (HF `facebook/sam-vit-*`) | segment-anything 1.0 (2023-04-06) | transformers `SamModel` | ViT-B is slow on CPU | Superseded by SAM 2.1 |
| SAM-HQ | High-quality masks (2306.01567) | Apache-2.0 | Apache-2.0 (repo) | Last commit 2025-09-12 | transformers `SamHQModel` | as SAM | Better thin structures |
| **EfficientSAM** | "~4 AP" gain over other fast SAMs on COCO/LVIS zero-shot instance segmentation (2312.00863) | Apache-2.0 | Apache-2.0 (repo) | Last commit 2024-12-24 (added EfficientTAM checkpoints) | Repo | Yes (ViT-T/S) | Edge promptable |
| **MobileSAM** | "More than 60 times smaller yet performs on par with the original SAM"; about 10 ms per image on GPU (2306.14289) | Apache-2.0 | Apache-2.0 (repo) | Commit 2026-05-05 | Repo; Ultralytics wrapper (AGPL) | **Yes** | Drop-in SAM-v1 decoder |
| **EdgeTAM** | 87.7 / 70.0 / 72.3 / 71.7 J&F on DAVIS 2017 / MOSE / SA-V val / SA-V test (2501.07256) | Apache-2.0 | Apache-2.0 (HF `facebook/EdgeTAM`) | HF 2025-04-30; commit 2026-01-27 | transformers `EdgeTamModel`, `EdgeTamVideoModel` | Designed for on-device | On-device SAM 2 |
| SAM3-LiteText (`vil-uob/sam3-litetext-s0`) | Replaces SAM 3's 353M text encoder with a distilled MobileCLIP-based one; **keeps the SAM 3 ViT-H image encoder** (card) | n/a | HF tag Apache-2.0, **but it redistributes SAM 3 weights (SAM License covers derivatives)**; the MobileCLIP-derived text tower licence was not verified | HF 2026-03-26 | transformers `Sam3LiteTextModel` | GPU | Licence trap |

### 2C. Medical segmentation

| Model / framework | What it is | Best verified number | Code licence | Weights licence | Latest release + date | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|---|
| **nnU-Net v2 + ResEnc presets** | Self-configuring U-Net framework (1809.10486); Residual-Encoder presets M/L/XL ("nnU-Net Revisited", 2404.09556) | 5-fold CV Dice (resenc_presets.md): ResEnc L scores KiTS 88.17 vs 86.04 for original nnU-Net, and AMOS 89.41 vs 88.64. VRAM: M 9.1 GB, L 22.7 GB, XL 36.6 GB | Apache-2.0 | You train your own; OpenMind SSL checkpoints (`MIC-DKFZ/ResEncL-OpenMind-*`) exist; **nnFoundation weights CC-BY-SA-4.0** | nnunetv2 2.8.1 (2026-07-01); active (commit 2026-09-25) | `pip install nnunetv2` | Training is GPU; inference on CPU is slow but possible | Full; fine-tune from nnssl or nnFoundation checkpoints (documented) |
| **nnFoundation** CNN / ViT | 3D radiology foundation models on 2.1M CT/MR/PET volumes from 125 datasets, evaluated on 108 tasks (2609.26924, 2026-09-22) | Not quoted | nnssl / nnU-Net (Apache) | **CC-BY-SA-4.0** (HF `MIC-DKFZ/nnFoundationCNN` 102M ResEnc, `nnFoundationViT` 674M Primus; 2026-09-24) | New | via nnU-Net fine-tuning docs | CNN 102M | Yes (segmentation now; detection and classification "TBA" per card) |
| **TotalSegmentator** | nnU-Net models for whole-body CT/MR (2208.05868) | Original paper: Dice 0.943 on 104 structures (abstract). Current `total` task: 117 classes (CT); `total_mr`: 50 classes | Apache-2.0 | **Open tasks Apache-2.0** (`total`, `total_mr`, lung_vessels, body, vertebrae, liver_segments...). **Licensed tasks** (free NC licence or paid commercial): heartchambers_highres, appendicular_bones, tissue_types, brain_structures, coronary_arteries, face and others. **brain_aneurysm: CC BY-NC 4.0, "no commercial license available"** (README) | PyPI 2.18.0 (2026-08-12); **GitHub v3.0.0 tag and weights 2026-09-07, not yet on PyPI** | `pip install TotalSegmentator` | CPU works with `--fast` or `--roi_subset` (README) | Train with nnU-Net on its dataset |
| **MedSAM** | SAM ViT-B fine-tuned on medical images (2304.12306) | Promptable; no single number | Apache-2.0 | Apache-2.0 (HF `wanglab/medsam-vit-base`) | v1.0.0 (2024-01-02) | transformers `SamModel` | ViT-B | Yes |
| **MedSAM2** | SAM 2 fine-tuned on "over 455,000 3D image-mask pairs and 76,000 frames"; "reduce manual costs by more than 85%" (2504.03600) | As stated | Apache-2.0 | **CC-BY-SA-4.0** (HF `wanglab/MedSAM2`) | Last commit 2025-07-11 | Repo | GPU | Yes |
| **VISTA3D** (MONAI) | 3D automatic (127 classes) plus interactive segmentation (2406.05285) | "State-of-the-art ... in both 3D automatic ... and 3D interactive segmentation" (abstract) | Apache-2.0 (model-zoo code) | **NVIDIA NCLS v1 (non-commercial)** per `MONAI/VISTA3D-HF` card | model-zoo bundle 0.5.11 | MONAI bundle / HF | GPU | Yes (NC) |
| MONAI Auto3DSeg | AutoML ensemble (SegResNet, DiNTS, SwinUNETR) | nnU-Net Revisited (A3DS rows): KiTS 81.11 / 65.28 / 52.82 for SegResNet / DiNTS / SwinUNETR, vs 88.17 for ResEnc L | Apache-2.0 | n/a | monai 1.6.0 (2026-06-22); 1.6.1rc0 tag (2026-09-18) | `pip install monai` | GPU | Yes |

### Decision guidance: segmentation
- **Semantic or panoptic, commercial:**
  1. **EoMT with DINOv2** (MIT + Apache backbone) or **Mask2Former / OneFormer via transformers** (MIT).
  2. For small data, **smp Unet/UPerNet with a timm ConvNeXt or EfficientNet encoder** (MIT; avoid `mit_b*`).
  3. EoMT-DINOv3 is stronger (59.5 mIoU ADE20K) but inherits the DINOv3 licence and gate.
- **Instance segmentation, real-time:**
  1. **RF-DETR-Seg** (Apache for every size; 2XL 49.9 mask AP).
  2. YOLO26-seg if AGPL is acceptable (x 47.0).
  3. torchvision Mask R-CNN v2 as the simplest baseline.
- **Promptable:** **SAM 2.1** (Apache) by default; EdgeTAM or MobileSAM on device; SAM 3 when text or concept prompts are needed and the SAM License is acceptable.
- **Medical 3D:**
  1. **nnU-Net v2 with ResEnc L** first. nnU-Net Revisited found "the recipe for state-of-the-art performance is 1) employing CNN-based U-Net models ... 2) using the nnU-Net framework, and 3) scaling models to modern hardware resources".
  2. TotalSegmentator open tasks for anatomy.
  3. MedSAM2 for interactive labelling (share-alike weights).
  4. Do not default to transformer 3D segmenters (SwinUNETR, nnFormer) or Auto3DSeg; both trail in that benchmark.
- **Avoid:** SegFormer and smp `mit_b*` in products (NC); MMSegmentation for new work (frozen); Mask2Former upstream repo (archived; use transformers).

### Licence traps: segmentation
- **SegFormer / MiT weights and the smp `mix_transformer.py` code:** NVIDIA NC.
- **EoMT-DINOv3:** delta weights on DINOv3, so the DINOv3 License applies.
- **SAM 3 / 3.1 and all derivatives (SAM3-LiteText's Apache tag notwithstanding):** SAM License.
- **MedSAM2 weights:** CC-BY-SA (share-alike).
- **nnFoundation:** CC-BY-SA-4.0.
- **VISTA3D weights:** NCLS v1.
- **TotalSegmentator:** licensed subtasks, and brain_aneurysm is NC with no commercial option.

### Metrics: segmentation
- **mIoU:**
  - Standard ADE20K and Cityscapes results use **dataset-level** accumulated intersection and union per class, with an ignore label (commonly 255). The ADE20K background handling ("reduce_zero_label" in OpenMMLab configs) was not re-verified here, so check how your dataset handles class 0.
  - `torchmetrics.segmentation.MeanIoU` (1.9.0) instead averages **per-sample** IoU over present classes (see `update()`), has no `ignore_index`, defaults to `input_format="one-hot"` and `include_background=True`. Its numbers are not comparable with papers. Compute a confusion matrix yourself or use `torchmetrics.classification.MulticlassJaccardIndex` with `ignore_index` (not re-verified here).
- **PQ:** `torchmetrics.detection.PanopticQuality` (1.9.0) or the COCO panopticapi (not on PyPI).
- **Instance masks:** COCO mask AP via pycocotools (`iouType='segm'`).
- **Boundaries:** Boundary IoU (2103.16562) for thin structures.
- **Medical:** Dice, NSD and HD95 via MONAI metrics or torchmetrics `DiceScore` / `HausdorffDistance` (1.9.0). Follow Metrics Reloaded (2206.01653) for metric choice by task, and report per-structure values, not only the mean.

---

## 3. Pose and keypoints

| Model / framework | What it is | Best verified number | Code licence | Weights licence | Latest release + date | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|---|
| **ViTPose / ViTPose++** | Plain-ViT top-down pose (2204.12484); ViTPose+ adds task-specific FFNs for human, whole-body and animal keypoints (2212.04246) | ViTPose-G (COCO+AIC+MPII, 576x432): 81.0 AP / 85.6 AR on COCO val with a 56-mAP person detector; ViTPose-B 77.1 (README) | Apache-2.0 | HF `usyd-community/vitpose-*` tagged Apache-2.0. Upstream backbones are MAE-pretrained; whether MAE's NC terms carry over was not verified | Repo last commit 2025-12-25 | **transformers `VitPoseForPoseEstimation`** (HF 2025-01) | Sizes S/B/L/H/G (params not re-verified); top-down needs a separate person detector | transformers or repo (mmcv-based) |
| **RTMPose / RTMO / RTMW / DWPose** | Real-time top-down (RTMPose 2303.07399), one-stage (RTMO 2312.07526), whole-body (RTMW 2407.08634), distilled whole-body (DWPose 2307.15880) | RTMPose-m 75.8% AP on COCO at "90+ FPS on an Intel i7-11700 CPU"; RTMO-l 74.8% AP on COCO val2017 at 141 FPS on V100; RTMW-l 70.2 mAP on COCO-WholeBody; DWPose raises RTMPose-l whole-body from 64.8% to 66.5% (abstracts) | Apache-2.0 (MMPose, DWPose) | Apache-2.0 (MMPose model zoo; HF `yzd-v/DWPose` Apache-2.0) | MMPose 1.3.2 (2024-07-12), last commit 2025-08-04 (mmcv-bound). **rtmlib 0.0.16 (2026-08-04)** runs RTMPose, DWPose, RTMO, RTMW and ViTPose "without mmcv, mmpose, mmdet" | **`pip install rtmlib`** (ONNX Runtime or OpenVINO) | **Best CPU option** (RTMPose-m CPU figure above) | Training still needs MMPose (frozen stack) |
| **Sapiens2** 0.4B-5B | Human-centric foundation models at 1K-4K resolution (2604.21681); pose (308 keypoints), body-part segmentation, normals, pointmap, albedo | "+4 mAP" pose and "+24.3 mIoU" body-part segmentation over Sapiens v1 (abstract) | Sapiens2 License (custom Meta; acceptable-use bans on military, ITAR and similar; no non-commercial clause found by keyword search) | Same (HF `facebook/sapiens2-*`, 2026-04-23) | HF 2026-04/05 | **transformers `Sapiens2ForPoseEstimation`, `Sapiens2ForSemanticSegmentation`** | No (0.4B+ at 1K) | Repo |
| Sapiens (v1) | Previous generation (2408.12569) | n/a | **CC-BY-NC-4.0** | **CC-BY-NC-4.0** | Repo commit 2026-05-26 | torchscript checkpoints | No | NC |
| **YOLO26-pose** | Ultralytics one-stage pose with RLE loss | x 71.6 / 91.6 mAP50-95 / mAP50 (OKS) on COCO val; n 57.2 (docs). Paper per docs: up to +7.2 AP over YOLO11-pose | AGPL-3.0 | AGPL-3.0 | ultralytics 8.4.x | ultralytics | n 2.9M, CPU ONNX 40.3 ms (docs) | Yes, custom keypoint schemas |
| **RF-DETR Keypoint (preview)** | DETR keypoint head | 71.8 COCO keypoint AP50:95, 9.7 ms on T4, 40.7M params (SAB) | Apache-2.0 | Apache-2.0 | rfdetr 1.11.0 | `rfdetr` (`RFDETRKeypointPreview`) | GPU-class | Preview |
| MediaPipe (pose, hand and face landmarkers) | On-device landmark models | Not verified | Apache-2.0 | Not verified per model | v1.0.0 (2026-07-28) | `pip install mediapipe` (version not checked) | **Yes (mobile, CPU)** | Limited (Model Maker) |
| HaMeR (hands, 3D) | Transformer hand mesh recovery (2312.05251) | Not quoted | MIT | Needs the **MANO model**, which requires registration on the MANO site (licence terms not fetched) | Commit 2026-02-07 | Repo | No | Repo |
| WiLoR (hands) | 3D hand localisation and reconstruction (2409.12259) | Not quoted | **CC BY-NC-ND 4.0** | **CC-BY-NC-4.0** (HF) | Commit 2026-04-07 | Repo | No | NC |
| DeepLabCut / SuperAnimal (animals) | Animal pose toolbox (SuperAnimal: 2203.07436) | Not quoted | **LGPL-3.0** | SuperAnimal HF repos have **no licence tag** | DeepLabCut v3.0.2 (2026-09-23) | `pip install deeplabcut` (version not checked) | Yes (ResNet variants) | Yes |
| InsightFace (face detection and recognition) | Face analysis library | n/a | MIT | **"non-commercial research purposes only"**, for manual and auto-downloaded models alike (README) | insightface 2.0 on PyPI (2026-09-08) | pip | Yes | Commercial licence by contact |

### Decision guidance: pose
- **CPU or real-time, commercial:** **rtmlib with RTMPose-m or RTMO** (Apache; mmcv-free). RTMPose-m does 75.8 AP at 90+ FPS on a desktop CPU (abstract).
- **Highest accuracy, top-down:** **ViTPose(+) via transformers** (Apache tag) or **Sapiens2** (custom licence, 0.4B-5B, 308 keypoints).
- **Single-stage with an easy custom-keypoint trainer:** YOLO26-pose if AGPL is acceptable; otherwise RF-DETR Keypoint (preview, Apache).
- **Animals:** DeepLabCut (LGPL) or ViTPose++ (AP-10K and APT-36K in the abstract).
- **Hands:** HaMeR (MIT, but MANO access is gated). Avoid WiLoR in products (NC-ND).
- **Faces:** InsightFace pretrained models are NC.
- **Comparing numbers:** top-down numbers depend on the person detector (ViTPose's table uses a 56-mAP detector). Do not compare them directly with bottom-up or one-stage numbers (RTMO, YOLO-pose).

### Metrics: pose
- **COCO OKS AP** via `pycocotools` `COCOeval(iouType='keypoints')`. Verified defaults: `maxDets=[20]`, area ranges all/medium/large (no "small"), and `kpt_oks_sigmas` hard-coded for the 17 COCO keypoints. **For custom skeletons, set `params.kpt_oks_sigmas` yourself**, or the metric is silently wrong or crashes.
- **Other benchmarks:** PCK / PCKh for MPII-style and animal benchmarks.
- **Library gaps:** torchmetrics has no OKS metric.
- **Alternatives:** faster-coco-eval supports keypoints and CrowdPose (README).

---

## 4. Depth and 3D geometry

| Model | What it is | Best verified number / claim | Code licence | Weights licence (per size) | Latest | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|---|
| **Depth Anything V2** S/B/L/(G) | Relative monocular depth trained on synthetic labels plus pseudo-labelled real images (2406.09414); metric fine-tunes available | "More than 10x faster and more accurate" than Stable-Diffusion-based depth (abstract) | Apache-2.0 | **Small (24.8M): Apache-2.0. Base (97.5M), Large (335.3M), Giant (1.3B, "coming soon"): CC-BY-NC-4.0** (README; HF tags agree) | Commit 2026-03-24 | transformers `DepthAnythingForDepthEstimation` (`depth-anything/Depth-Anything-V2-*-hf`) | **Small: yes** | Metric fine-tuning scripts in repo |
| **Depth Anything 3** (DA3) | Any-view geometry (depth plus ray maps) from a plain DINO transformer (2511.10647) | "Surpassing prior SOTA VGGT by an average of 44.3% in camera pose accuracy and 25.1% in geometric accuracy", and beats DA2 on monocular depth (abstract) | Apache-2.0 | README table: **DA3-SMALL (0.08B) and DA3-BASE (0.12B) Apache-2.0; DA3METRIC-LARGE and DA3MONO-LARGE (0.35B) Apache-2.0; DA3-LARGE(-1.1) (0.35B), DA3-GIANT(-1.1) (1.15B) and DA3NESTED-GIANT-LARGE(-1.1) (1.40B) CC BY-NC 4.0.** HF tags `DA3-LARGE-1.1` as Apache-2.0, which conflicts with the README | HF 2025-11-13; -1.1 checkpoints 2025-12-11 ("retrained after fixing a training bug; prefer these") | `depth-anything-3` repo (no transformers class found) | Small/Base feasible | Repo |
| Video Depth Anything | Temporally consistent video depth | Not verified | Apache-2.0 | Not verified | Release 1.3.1 (2025-09-15) | Repo | Small size exists (not verified) | Repo |
| **Depth Pro** | Zero-shot metric depth with sharp boundaries (2410.02073) | "2.25-megapixel depth map in 0.3 seconds on a standard GPU" (abstract) | Apple sample-code licence (permissive text) | **Conflict.** HF metadata says `apple-amlr` (research-only; changed 2025-02-28 per R3), the HF card body says "Apple-ASCL", and the GitHub README says weights are under the repo LICENSE. Treat as research-only until cleared | Repo commit 2026-09-11 | transformers `DepthProForDepthEstimation` | No (ViT-L multi-scale) | Limited |
| Metric3D v2 | Metric depth plus normals via canonical camera space (2404.15506) | Qualitative (abstract) | BSD-2-Clause | Weights licence not stated (README); ONNX exports on HF `onnx-community/metric3d-*` | Commit 2025-03-13 | `torch.hub`, ONNX | ViT-S ONNX | Limited |
| UniDepth V1 / V2 | Universal metric depth with a self-promptable camera module (2403.18913, 2502.20110) | "Superior performance and generalization" on ten datasets zero-shot (abstract) | **CC BY-NC 4.0** | HF `lpiccinelli/unidepth-v2-*` untagged; treat as NC | Commit 2025-05-18 | Repo | ViT-S variant | NC |
| **MoGe / MoGe-2 / MoGe-3** | Monocular geometry (point maps); MoGe-2 adds metric scale (2507.02546); MoGe-3 adds sparse 3D refinement for fine detail (2607.17967, 2026-07-20) | MoGe-3 "significantly outperforms existing approaches in recovering fine detailed 3D geometry" (abstract) | MIT (DINOv2 modules Apache-2.0) | **MIT** (HF `Ruicheng/moge-*`, including `moge-3-vitl` and `moge-3-vitg`, 2026-08-18) | Commit 2026-08-19 ("V3") | Repo `microsoft/MoGe` | **MoGe-2 ViT-S: 35M params** (README); MoGe-3: 370M / 1.25B; MoGe-3 needs FlexGEMM/Triton (no macOS) | Repo |
| **VGGT** | Feed-forward multi-view geometry: cameras, depth, point maps, tracks (2503.11651) | Commercial checkpoint AUC@30 90.37 (README) | VGGT License v1 (custom; commercial allowed, military excluded) | **`VGGT-1B`: CC-BY-NC-4.0. `VGGT-1B-Commercial`: VGGT licence, gated (manual)**, "only this model checkpoint allows commercial usage" (README). `VGGT-Omega` (2026-03-17): FAIR non-commercial, gated | Commit 2026-05-19 | Repo | No (1B) | Repo |
| **MapAnything** | Universal feed-forward metric 3D reconstruction (2509.13414) | "Outperforms or matches specialist feed-forward models" (abstract) | Apache-2.0 | **`facebook/map-anything`: CC-BY-NC-4.0. `facebook/map-anything-apache`: Apache-2.0** | v1.1.3 (2026-07-17) | Repo | No | Repo |
| Pi3 / Pi3X | Permutation-equivariant multi-view geometry (2507.13347) | Qualitative | BSD-2-style | **Pi3: BSD-2-Clause. Pi3X: CC-BY-NC-4.0** (HF) | Commit 2026-05-18 | Repo | No | Repo |
| DUSt3R / MASt3R | Pairwise 3D reconstruction and matching (2312.14132, 2406.09756) | Qualitative | **CC BY-NC-SA 4.0** | HF untagged; the repo licence (NC-SA) governs | Commits 2025-06/07 | Repo | No | NC |
| **Marigold** | Diffusion-based depth, normals and intrinsics (2312.02145) | Qualitative | Apache-2.0 | **depth v1-1: OpenRAIL++. depth v1-0: Apache-2.0** (HF) | Commit 2026-09-06 | **diffusers 0.40.0** Marigold pipelines | Slow (diffusion; LCM variant exists) | Repo |
| Prompt Depth Anything, ZoeDepth, DPT | Other transformers depth heads | n/a | Apache-2.0 (transformers) | Per checkpoint | transformers 5.17 | `PromptDepthAnythingForDepthEstimation`, `ZoeDepthForDepthEstimation`, `DPTForDepthEstimation` | Varies | Yes |

### Decision guidance: depth
- **Relative depth on CPU, commercial:** Depth Anything V2-Small (Apache, 24.8M) or MoGe-2 ViT-S (MIT, 35M).
- **Metric depth, commercial:** DA3METRIC-LARGE (Apache per README), MoGe-2/3 (MIT) or UniDepth (NC only). Depth Pro is licence-unclear.
- **Multi-view or SfM-like:** DA3-BASE/SMALL (Apache), MapAnything-apache, Pi3 (BSD-2) or VGGT-1B-Commercial (custom).
- **Avoid in products:** DA V2 B/L/G, DA3 L/G, VGGT-1B, MapAnything (non-apache), Pi3X, UniDepth and DUSt3R/MASt3R (all NC).
- **Metrics:**
  - AbsRel, δ1 (δ < 1.25) and RMSE. Relative (affine-invariant) predictions must be aligned per image by least-squares scale and shift before scoring; metric models are scored without alignment. Never compare the two regimes in one table.
  - Boundary sharpness needs dedicated metrics (Depth Pro proposes some in its abstract).
  - No torchmetrics depth metric exists; write the formulae explicitly and state the alignment and valid-pixel mask.

---

## 5. Video

### 5A. Action recognition and video encoders

| Model | Best verified number | Code licence | Weights licence | Latest | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|
| **V-JEPA 2** (2506.09985) | "77.3 top-1 accuracy on Something-Something v2"; 39.7 recall@5 on Epic-Kitchens-100 anticipation (abstract) | MIT | **MIT** (HF `facebook/vjepa2-*`, including SSv2 and Diving48 classifiers) | Commit 2026-03-23 | **transformers `VJEPA2ForVideoClassification`** | ViT-L 64-frame is GPU | Yes (probe or fine-tune) |
| VideoMAE V2 (2303.16727) | ViT-g: 90.0% K400, 89.9% K600, 68.7% SSv1, 77.0% SSv2 (abstract) | MIT | **CC-BY-NC-4.0** (HF `OpenGVLab/VideoMAEv2-*`) | Commit 2024-10-08 | Repo | No | NC |
| VideoMAE (2203.12602) | n/a | Code in transformers (Apache) | **CC-BY-NC-4.0** (HF `MCG-NJU/videomae-*`) | n/a | transformers `VideoMAEForVideoClassification` | ViT-B | NC |
| InternVideo2 (2403.15377) | "Superior performance on over 60 video and audio tasks" (abstract) | Apache-2.0 | Apache-2.0, **gated (auto)** (HF `OpenGVLab/InternVideo2-*`) | Commit 2026-07-02 | Repo | No (1B-6B) | Repo |
| torchvision video models | K400 acc@1 (torchvision 0.29 metadata): MViT-V2-S 80.757, Swin3D-B 79.427 (81.643 with `KINETICS400_IMAGENET22K_V1`), Swin3D-S 79.521, MViT-V1-B 78.477, S3D 68.368, R(2+1)D-18 67.463 | BSD-3-Clause | torchvision weights (Kinetics-trained) | torchvision 0.29.0 | `torchvision.models.video.*` | **S3D 8.3M / 18 GFLOPs**; MViT-V2-S 34.5M / 64 GFLOPs | Yes; permissive baseline |
| PyTorchVideo (X3D, MViT, SlowFast) | Not verified | Apache-2.0 | Model zoo | **0.1.5 on PyPI (2022-01-20)**; repo gets only internal lint commits (2026-05-05) | pip or `torch.hub` | X3D-XS/S are CPU-class (numbers not verified) | Effectively unmaintained |
| MMAction2 | n/a | Apache-2.0 | Model zoo | 1.2.0 (2023-10-12); commit 2026-03-18 | mmcv-bound | n/a | Legacy |
| TimeSformer | n/a | Apache (transformers) | **CC-BY-NC-4.0** (HF `facebook/timesformer-*`) | n/a | transformers | No | NC |
| VideoPrism | Not verified | n/a | Not verified | n/a | transformers `VideoPrismForVideoClassification` (present in 5.17) | n/a | n/a |

### 5B. Temporal action localisation
- **OpenTAD** (2502.20361; Apache-2.0; v0.1.0 2024-04-17, commit 2026-07-14): a unified framework with "16 different TAD methods and 9 standard datasets" (abstract). Use it as the default harness.
- **ActionFormer** (2202.07925; MIT; last commit 2024-04-11): a strong single-method baseline.
- **Metric:** mAP averaged over temporal-IoU thresholds, using each dataset's official threshold set. The threshold sets differ between datasets, so always state them; exact sets not re-verified here.

### 5C. Video object segmentation and single-object tracking
- **SAM 2.1** (Apache) is the default promptable video segmenter (transformers `Sam2VideoModel`).
- **EdgeTAM** is the on-device option (Apache; J&F above).
- **SAM 3 tracker** handles concept-prompted video: SA-V test cgF1 30.3 and pHOTA 58.0 vs human 53.1 and 70.5 (SAM 3 README); SAM License.
- **SAMURAI** (2411.11922; Apache; uses SAM 2 weights): "7.1% AUC gain on LaSOT_ext and a 3.5% AO gain on GOT-10k" zero-shot (abstract).
- **VidEoMT** (2602.17807): encoder-only video segmentation; HF weights untagged, repo MIT.

### 5D. Multi-object tracking (MOT)

| Tracker / library | Verified number | Code licence | Latest / status | Notes |
|---|---|---|---|---|
| **`trackers`** (Roboflow) | Clean-room SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU and McByte. MOT17 HOTA with default parameters and YOLOX detections: BoT-SORT 63.7, OC-SORT 61.9, ByteTrack 60.1, SORT 58.4 (README) | Apache-2.0 | **2.6.1 (2026-09-25)**, active | Recommended permissive default; `trackers eval` computes CLEAR, HOTA and Identity metrics |
| supervision | `sv.ByteTrack` deprecated; **removed in `develop` ("as scheduled for supervision-0.31.0")**, use `trackers.ByteTrackTracker` (changelog) | MIT | 0.30.5 (2026-09-22) | Keep for annotation and slicing, not tracking |
| ByteTrack (2110.06864) | 80.3 MOTA, 77.3 IDF1, 63.1 HOTA on MOT17 test at 30 FPS on V100 (abstract) | MIT | Last commit 2022-12-11 | Use the `trackers` re-implementation |
| BoT-SORT (2206.14651) | 80.5 MOTA, 80.2 IDF1, 65.0 HOTA on MOT17 (abstract) | MIT | Last commit 2022-10-30 | Camera-motion compensation |
| OC-SORT (2203.14360) | "700+ FPS on a single CPU" given detections (abstract) | MIT | Commit 2026-04-21 (C++ overhaul) | Non-linear motion (DanceTrack) |
| Deep OC-SORT (2302.11813) | 64.9 HOTA on MOT17, 63.9 on MOT20, 61.3 on DanceTrack (abstract) | MIT | Last commit 2023-05-06 | Adds appearance re-ID |
| BoxMOT | Many trackers plus re-ID models | **AGPL-3.0** | 25.0.0 (2026-09-09) | Copyleft |
| Ultralytics `track` | BoT-SORT and ByteTrack configs | AGPL-3.0 | 8.4.x | Copyleft |

### Metrics: tracking
- **Official TrackEval:** `JonathonLuiten/TrackEval` (MIT) is **not on PyPI**; its last commit was 2022-11-29.
- **PyPI `trackeval` 1.3.0** (2026-03-03) is a **fork** (`kovalp/TrackEval`).
- **py-motmetrics** (MIT): PyPI 1.4.0 (2022-12-26); the repo is active (2026-07-17, "one-call MOTChallenge evaluation API"). It covers CLEAR MOT and IDF1, not HOTA.
- **HOTA** (2009.07736) "explicitly balances ... detection, association and localization". Report HOTA, IDF1 and MOTA together. MOTA over-weights detection.
- Detector quality dominates tracker rankings (the `trackers` README says so), so fix the detector when comparing trackers.

---

## 6. OCR and document understanding

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

---

## 7. Anomaly detection (industrial)

| Model / framework | What it is | Best verified number | Code licence | Weights licence | Latest release + date | Install route | CPU feasibility | Training data needed |
|---|---|---|---|---|---|---|---|---|
| **anomalib** | Library with 30+ models: Patchcore, EfficientAd, Dinomaly, AnomalyDINO, UniNet, SuperADD, Supersimplenet, Glass, InpFormer, WinClip, VlmAd, AnomalyVFM, Fastflow, Padim, Cflow, Draem, ReverseDistillation and more (`models/__init__.py`) | n/a | Apache-2.0 | Backbone licences apply (for example DINOv2 is Apache-2.0) | **2.6.2 (2026-09-11)**; active (commit 2026-09-23) | `pip install anomalib` | OpenVINO export (Intel-backed) | Normal images only (plus a few anomalous for validation) |
| **PatchCore** (2106.08265) | kNN over a coreset memory bank of ImageNet CNN patch features | "Image-level anomaly detection AUROC score of up to 99.6%" on MVTec AD (abstract) | Apache-2.0 (amazon-science; last commit 2023-03-06) | ImageNet backbone (torchvision) | Stale upstream; use anomalib | anomalib `Patchcore` | **Yes** (no training; memory bank grows with data) | Normal only |
| **EfficientAD** (2303.14535) | Student-teacher plus autoencoder for logical anomalies | "Latency of two milliseconds and a throughput of six hundred images per second", evaluated on 32 datasets (abstract) | Apache-2.0 (unofficial `nelson1425/EfficientAD`, last commit 2023-05-28) | n/a | anomalib | anomalib `EfficientAd` | Yes | Normal only |
| **Dinomaly** (2405.14325) | Multi-class reconstruction with a DINOv2 ViT | Image AUROC 99.6% MVTec AD, 98.7% VisA, 89.3% Real-IAD in the **multi-class** setting (abstract) | Apache-2.0 (active, 2026-09-17) | DINOv2 backbone (Apache-2.0) | Repo / anomalib | anomalib `Dinomaly` | GPU for training | Normal only, one model for all classes |
| **AnomalyDINO** (2405.14529) | Training-free patch kNN on DINOv2 | One-shot MVTec AD AUROC raised "from 93.1% to 96.6%" (abstract) | n/a | DINOv2 | anomalib | anomalib `AnomalyDINO` | Yes | **1 to few normal shots** |
| FastFlow (2111.07677) | 2D normalising flows | Not quoted | via anomalib | n/a | anomalib | anomalib `Fastflow` | Moderate | Normal only |
| UniAD (2206.03687) | Unified multi-class reconstruction transformer | Unified MVTec AD: detection 96.5%, localisation 96.8% (abstract) | Apache-2.0 (last commit 2022-11-22) | n/a | Stale; not in anomalib | Repo | GPU | Normal only |

**Benchmarks and their licences**

| Dataset | Licence (verified source) | Notes |
|---|---|---|
| MVTec AD | **CC BY-NC-SA 4.0** (anomalib `mvtecad.py` docstring; also R3) | Near-saturated: 99.6% image AUROC (PatchCore; Dinomaly multi-class) |
| **MVTec AD 2** (2503.21622) | **CC BY-NC-SA 4.0; "commercial use is explicitly prohibited"** (MVTec page) | Eight scenarios, 8000+ high-resolution images. SOTA "remains below 60% average AU-PRO" (abstract). The private test split is scored only through the public evaluation server |
| VisA (2207.14315) | **Conflict: the upstream `amazon-science/spot-diff` README says "CC BY 4.0"; anomalib's `visa.py` docstring says CC BY-NC-SA 4.0** | Follow upstream but record the conflict |

### Decision guidance: anomaly detection
- **Few normal images (1 to 16):** AnomalyDINO (training-free).
- **Standard single-class setting, CPU deployment:** PatchCore or EfficientAD via anomalib, exported to OpenVINO.
- **Many product classes, one model:** Dinomaly.
- **Validation:** validate on your own line data. MVTec AD is saturated, and MVTec AD 2 shows methods fall below 60% AU-PRO in harder scenarios (transparent, overlapping or tiny defects; lighting shifts).
- **Licence trap:** MVTec AD, MVTec AD 2 and probably Real-IAD (not verified) are NC. **Do not train or tune a commercial model on them.** Use them for method selection only, then refit on your own data.

### Metrics: anomaly detection
- **Metrics in anomalib 2.6.2:** AUROC, AUPR, AUPRO, PRO, F1Max, F1AdaptiveThreshold, ManualThreshold, PIMO / AUPIMO, PGn/PBn.
- **Pitfalls:**
  1. Image AUROC is threshold-free and saturates near 99%. Report the operating point (F1Max, or recall at a fixed false-positive rate) that the line will actually run at.
  2. Pixel AUROC is dominated by the huge number of normal pixels. Prefer AUPRO, which weights each defect region equally.
  3. **anomalib's `AUPRO` integrates only up to `fpr_limit=0.3` by default** (source), and its docs state that raising the limit raises the value. Always state the limit.
  4. Normalisation and thresholds must be fitted on validation data, not test data.
  5. Multi-class ("unified") and one-model-per-class numbers are different protocols.

---

## 8. Retrieval, re-identification and metric learning

| Component | What it is | Verified facts | Code licence | Weights licence | Latest | Install | CPU feasibility |
|---|---|---|---|---|---|---|---|
| **FAISS** (2401.08281) | Vector index and search (flat, IVF, HNSW, PQ) | n/a | MIT | n/a | faiss-cpu 1.15.1 (2026-09-16); faiss-gpu 1.15.1 on PyPI (2026-09-16; not verified as official); `faiss-gpu-cu12` 1.14.1.post1 (community wheels) | pip / conda | **Yes** |
| **Embedding backbones** | CLIP/SigLIP 2/DINOv2/DINOv3/PE: see R3 | See R3 | | See R3 (SigLIP 2 and DINOv2 are Apache-2.0; DINOv3 is custom) | | timm / open_clip / transformers | ViT-S/B on CPU |
| Qwen3-VL-Embedding 2B/8B | Multimodal (image and text) embedding model | n/a | Apache-2.0 | Apache-2.0 (HF, 2026-01-07) | | sentence-transformers 6.1.0 | 2B is marginal on CPU |
| **pytorch-metric-learning** | Losses (ArcFace, triplet, contrastive...), miners, samplers, AccuracyCalculator | "A Metric Learning Reality Check" (2003.08505, same author): "actual improvements over time have been marginal at best" | MIT | n/a | **2.9.0 (2025-08-17)**; last commit 2025-08-17 | pip | Yes |
| **SuperGlobal** (2308.06954) | Improved GeM pooling plus cheap global-feature reranking for landmark retrieval | Revisited Oxford+1M Hard: single-stage +7.1%, two-stage +3.7% "with a strong 64,865x speedup" (abstract) | Not verified | Not verified | n/a | Repo | Yes |
| revisitop (ROxford / RParis evaluation) | Official evaluation code for the revisited benchmarks (1803.11285) | `compute_map(ranks, gnd, kappas)` handles junk images, as declared in the ground truth | Not verified | n/a | n/a | Copy `evaluate.py` | Yes |
| **FastReID** (2006.02631) | Re-ID toolbox | n/a | Apache-2.0 | Model zoo | **V1.3.0 (2021-05-31)**; last commit 2024-07-15 | Repo | Stale |
| torchreid (deep-person-reid) | Re-ID library (OSNet etc.) | Market-1501 evaluation drops gallery images with the same person ID and camera as the query (`rank.py`, verified) | MIT | Model zoo | PyPI `torchreid` 0.2.5 (2022-10-16); repo commit 2026-01-09 (README) | pip | OSNet on CPU |
| CLIP-ReID (2211.13977) | CLIP image encoder fine-tuned with learned ID text tokens | "Simply fine-tuning the visual model initialized by the image encoder in CLIP, has already obtained competitive performances" (abstract) | MIT | CLIP (OpenAI) weights | Last commit 2023-11-21 | Repo | ViT-B |
| BoxMOT re-ID | OSNet, CLIP-ReID and other re-ID heads for trackers | n/a | **AGPL-3.0** | n/a | 25.0.0 (2026-09-09) | pip | Yes |
| InsightFace | Face recognition | n/a | MIT | **NC** (see Section 3) | 2.0 (2026-09-08) | pip | Yes |

### Decision guidance: retrieval and re-ID
- **Instance or landmark retrieval:**
  1. Use frozen **DINOv2** (Apache) or SigLIP 2 global embeddings with FAISS (see R3 for encoder choice).
  2. Add **query-expansion or reranking** (SuperGlobal-style) before any fine-tuning.
  3. Fine-tune with pytorch-metric-learning only when frozen features plateau.
  4. Follow the Reality Check's protocol: same embedding size, same batch sampler, tuned on a validation split, with no test-set early stopping.
- **Person or vehicle re-ID:** the maintained toolboxes are stale (FastReID 2021, torchreid 2022 on PyPI). The practical route is a CLIP- or DINOv2-initialised ViT with ArcFace or triplet loss from pytorch-metric-learning, evaluated with torchreid's Market-1501 protocol code.
- **Licence traps:** InsightFace models are NC. BoxMOT is AGPL. Re-ID benchmark datasets (Market-1501 and others) carry their own terms, which were not checked here.

### Metrics: retrieval and re-ID
- **Retrieval:**
  - mAP@k and recall@k (torchmetrics 1.9.0 `RetrievalMAP`, `RetrievalRecall`, `RetrievalPrecision`, `RetrievalNormalizedDCG`, all with `top_k`).
  - ROxford/RParis mAP on the **Medium and Hard** protocols with junk handling (revisitop).
  - GLDv2 (2004.01804) uses mAP@100.
- **Re-ID:**
  - CMC rank-1/5/10 plus mAP, with **same-ID-same-camera gallery removal**. Omitting this removal inflates rank-1.
  - Report whether re-ranking (k-reciprocal) was used.
- **Pitfall:** torchmetrics retrieval metrics group scores by `indexes` (query ID). Forgetting to pass per-query indexes silently computes one global ranking.

---

## 9. Vision-language models (captioning, VQA, grounding, OCR, open-vocabulary)

| Model family | Sizes (verified on HF) | Weights licence (read from card / licence file) | Latest | Install route | CPU feasibility | Grounding / detection output |
|---|---|---|---|---|---|---|
| **Qwen3.5** | 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B (HF, 2026-02) | **Apache-2.0** | 2026-02-16 to 2026-02-28 | transformers `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` | 0.8B/2B feasible (quantised) | Card: "Unified Vision-Language Foundation: early fusion training ... outperforms Qwen3-VL models across ... visual understanding" (vendor claim) |
| Qwen3.6 | 27B, 35B-A3B (HF, 2026-04) | Apache-2.0 | 2026-04-15/21 | transformers (image-text-to-text) | No | Not verified |
| **Qwen3-VL** (2511.21631) | 2B, 4B, 8B, 32B (not checked), 235B-A22B | **Apache-2.0** (2B/8B/235B checked) | 2025-09/10 | transformers `Qwen3VLForConditionalGeneration`; vLLM | 2B feasible | Boxes and points (grounding); tech report emphasises spatial-temporal grounding |
| Qwen2.5-VL (2502.13923) | 3B, 7B, 32B, 72B | **3B: Qwen Research licence ("Non-Commercial ... research or evaluation purposes only"). 7B: Apache-2.0. 72B: Qwen licence (separate licence needed above 100M MAU)** | 2025-01 | transformers `Qwen2_5_VLForConditionalGeneration` | 3B (NC) | Boxes (JSON) |
| **InternVL3 / InternVL3.5** (2504.10479, 2508.18265) | 1B-241B | InternVL3-8B card: "released under the MIT License", Qwen2.5 component "licensed under the Apache-2.0 License" (HF metadata has `license_name: qwen`, a conflict). InternVL3.5-8B: Apache-2.0 (Qwen3 component Apache-2.0) | 2025-04 / 2025-08 | transformers `InternVLForConditionalGeneration` (`-HF` repos) | 1B feasible | Boxes |
| **Molmo / Molmo2** (2409.17146, 2601.10611) | Molmo2 4B/8B (8B card: Qwen3-8B + SigLIP so400m), O-7B; MolmoPoint-8B | **Apache-2.0** | 2025-12-14 (Molmo2); MolmoPoint 2026-03-16 | transformers (remote code) | No | **Pointing** and video tracking: Molmo2 "56.2 vs 41.1 J&F on video tracking" against Gemini 3 Pro (abstract) |
| **Gemma 4** (2607.02770) | E2B, E4B, 12B, 26B-A4B, 31B | **Apache-2.0** (HF tag; the Gemma 4 licence page header reads "Apache License 2.0 / Gemma"). Gemma 3 and PaliGemma 2 remain under **Gemma terms**, gated (manual) | 2026-03 to 2026-05 | transformers `Gemma4ForConditionalGeneration` | E2B/E4B designed for on-device | Not verified |
| PaliGemma 2 (2412.03555) | 3B, 10B, 28B at 224/448/896 px | **Gemma terms, gated** | 2024-11-21 | transformers `PaliGemmaForConditionalGeneration` | 3B marginal | Detection and segmentation tokens; built for fine-tuning transfer |
| **Florence-2** | 0.23B / 0.77B | **MIT** | 2024-06 | transformers `Florence2ForConditionalGeneration` | **Yes** | Detection, dense captioning, OCR, grounding task prompts |
| **SmolVLM / SmolVLM2** (2504.05299) | 256M, 500M, 2.2B | **Apache-2.0** | 2025-01/04 | transformers `SmolVLMForConditionalGeneration` | **Yes**: "SmolVLM-256M uses less than 1GB GPU memory" (abstract) | Limited |
| Moondream 3 / 3.1 | 9B total, 2B active MoE | **BUSL-1.1 plus additional use grant** (3 preview); "moondream-model-license-1.0" (3.1) | 2025-09 / 2026-06 | HF | No | `detect()` and `point()` |
| Perception LM (1B/3B/8B) | | **FAIR non-commercial research**, gated | 2025-04 | transformers `PerceptionLMForConditionalGeneration` | 1B | n/a |

Serving and evaluation: **vLLM** 0.30.0 (2026-09-22, Apache-2.0); qwen-vl-utils 0.0.14; **lmms-eval** 0.7.3 (2026-08-28, MIT). VLMEvalKit's PyPI status was not verified.

### When zero-shot VLMs win and when fine-tuned specialists win (evidence)

| Setting | Evidence | Verdict |
|---|---|---|
| Common-category detection with no training data | GroundingDINO zero-shot scores 48 AP vs 33 AP for SOTA few-shot detectors on COCO (2312.14494). SAM 3 zero-shot reaches 56.4 COCO box AP (README) | **Zero-shot or open-vocabulary wins** at 0 to 10 shots on web-common classes |
| Out-of-distribution classes and modalities (medical, aerial, industrial) | RF100-VL: GroundingDINO and Qwen2.5-VL "achieve less than 2% zero-shot accuracy on challenging medical imaging datasets" (2505.20612). Fine-tuned RF-DETR-2XL beats GroundingDINO-tiny by 1.2 AP on RF100-VL at 20x the speed (2511.09554) | **Fine-tuned specialists win**; use VLMs only for pre-labelling |
| Closed-set image classification | VLMs "significantly underperform CLIP on standard image classification benchmarks like ImageNet"; the cause is data exposure (2405.18415) | Use a CLIP/SigLIP linear probe or a fine-tuned classifier, not a chat VLM |
| Fine-grained geometry (counting overlaps, intersections) | Four SOTA VLMs average 58.07% on BlindTest's simple geometric tasks (2407.06581) | Use specialist detectors, segmenters or measurement code |
| Document parsing | 0.9B document specialists (96.34) beat Qwen3-VL-235B (89.78) on OmniDocBench v1.6 | Specialised small VLMs win |
| Video pointing and tracking | Molmo2 surpasses Gemini 3 Pro on video pointing (38.4 vs 20.0 F1) (2601.10611) | Open grounding-trained VLMs are competitive |

### Decision guidance: VLMs
- **Permissive defaults:** Qwen3-VL (2B/4B/8B), Qwen3.5 (0.8B-9B) and Gemma 4 (E2B/E4B/12B), all Apache-2.0. Add Florence-2 (MIT) or SmolVLM2 (Apache) for CPU.
- **Grounding and pointing:** Molmo2 (Apache).
- **Avoid:** Qwen2.5-VL-3B (NC), Perception LM (NC), Moondream (BUSL) and HunyuanOCR (territory) without a licence review.
- **Measure before deploying:** always benchmark a VLM against a fine-tuned specialist on a held-out slice of your own data before choosing it for a production perception task (see the table above).

---

## 10. Remote sensing and Earth observation

| Component | What it is | Verified facts | Code licence | Weights licence | Latest | Install | Notes |
|---|---|---|---|---|---|---|---|
| **TorchGeo** (2111.08872) | EO datasets, samplers, transforms, model registry and Lightning tasks | 0.10.0 **requires Python >= 3.12**. Tasks moved to `torchgeo.tasks.*`; the old `torchgeo.trainers.*Task` names are deprecated aliases. Tasks: Classification, SemanticSegmentation, InstanceSegmentation, ObjectDetection, **ChangeDetection**, Regression, spatio-temporal tasks, MAE/MoCo/SimCLR/BYOL. Models include DOFA, CROMA, Copernicus-FM, Panopticon, OlmoEarth v1, Presto, SatCLIP, Scale-MAE, Tessera, **ChangeStar, ChangeViT, FCSiamConc/Diff** | MIT | Per weight (see R3) | 0.10.0 (2026-08-14); active | `pip install torchgeo` | Default EO toolkit |
| **TerraTorch** | Fine-tuning toolkit for geospatial foundation models (Prithvi, TerraMind) | Tracks TorchGeo 0.10 (commit 2026-09-25) | Apache-2.0 AND MIT | Per model | 1.2.13 (2026-09-16 GitHub; PyPI 2026-08-31) | pip | IBM/NASA/ESA models |
| **Prithvi-EO-2.0** (2412.02732) | Multi-temporal HLS foundation model, 300M/600M | Outperforms six other geospatial FMs (abstract) | MIT (NASA-IMPACT repo) | **Apache-2.0** (HF) | HF 2024-11/12 | TerraTorch | Permissive |
| **Clay** | Multi-sensor EO foundation model | n/a | Apache-2.0 | Apache-2.0 (HF `made-with-clay/Clay`) | v1.0.1 "Pip installable" (2025-07-05) | pip (package name not verified) | Permissive |
| **DOFA** (2403.15356) | Wavelength-conditioned any-sensor foundation model | "Generalizes well to unseen modalities" (abstract) | MIT | **CC-BY-4.0** (HF `earthflow/DOFA`, `torchgeo/dofa`) | Commit 2026-07-22 | TorchGeo `dofa_*` | Attribution required |
| **SSL4EO-S12** (2211.07044) | Sentinel-1/2 SSL dataset and weights | "Downstream performance close to, or surpassing" supervised (abstract) | Apache-2.0 | **CC-BY-4.0** (HF `wangyi111/SSL4EO-S12`) | 2024-04 | TorchGeo ResNet/ViT weights | |
| SatMAE (2207.08051) | Temporal and multispectral MAE | Up to 14% gains on land-cover transfer (abstract) | **CC BY-NC 4.0** | NC | Commit 2025-08-10 | Repo | Research only |
| **DINOv3-SAT** | DINOv3 ViT-L/16 and ViT-7B/16 pretrained on SAT-493M satellite imagery | See R3 | DINOv3 License | **DINOv3 License, gated (manual)** (HF `facebook/dinov3-vit{l16,7b16}-pretrain-sat493m`) | HF 2025-08-06 | transformers | Custom-commercial |
| **Oriented detection on DOTA** | See 1C: YOLO26-obb (AGPL), LSKNet (NC, 81.85 mAP), ai4rs O2-DETRs (Apache), MMRotate (frozen) | DOTA: "academic purposes only" | | | | | Dataset licence limits weights |
| **Change detection** | TorchGeo ChangeDetection task; **open-cd** (Apache-2.0; v1.1.0 2024-02-10, commit 2026-06-15; mmseg-based per its lineage, not re-verified); BIT (2103.00208); ChangeFormer (2201.01293) | Recent FM-based papers: 2509.09572 (PEFT on VFMs), 2602.13780 | varies | varies | | | Start with a Siamese UNet (TorchGeo FCSiam) baseline |

### Decision guidance: remote sensing
- **Always run a supervised baseline.** PANGAEA (2412.04204): geospatial FMs "do not consistently outperform supervised models" such as UNet and vanilla ViT.
- **Model choice:** for multispectral time series, Prithvi-EO-2.0 or TerraMind via TerraTorch (Apache). For arbitrary band sets, DOFA (CC-BY-4.0). For RGB very-high-resolution imagery, DINOv3-SAT (custom licence) or DINOv2 (Apache).
- **Oriented objects:** YOLO26-obb (AGPL) or ai4rs O2-DETR (Apache). DOTA-trained weights inherit DOTA's academic-only terms.
- **TorchGeo 0.10 needs Python 3.12+.** Pin accordingly and use `torchgeo.tasks`.
- **Metrics:** per-class IoU/F1 for segmentation. For change detection, F1 and IoU of the *change* class; overall accuracy is meaningless under heavy imbalance. For OBB, rotated-IoU mAP at 0.5 (DOTA's official evaluation) vs mAP50-95 (Ultralytics). Say which.

---

## 11. 3D medical imaging beyond segmentation

| Component | Task | Verified facts | Code licence | Weights licence | Latest | Install |
|---|---|---|---|---|---|---|
| **MONAI** | 3D classification networks (DenseNet, ResNet, SEResNet, EfficientNet-BN, ViT, SwinUNETR encoder), transforms, metrics, bundles | n/a | Apache-2.0 | Per bundle (VISTA3D weights NCLS v1) | **1.6.0 (2026-06-22)**; 1.6.1rc0 tag 2026-09-18 | `pip install monai` |
| MedicalNet (Med3D, 1904.00625) | 3D ResNet weights pretrained across segmentation datasets | n/a | MIT (Tencent) | MIT | Last substantive work 2019; commit 2025-11-27 (requirements) | Repo |
| **VoxelMorph** (1809.05231) | Unsupervised deformable registration | v0.3 (2025-08-25) adds a `voxelmorph/nn` namespace | **Conflict: repo LICENSE.md (dev) Apache-2.0; PyPI classifier MIT** | You train | voxelmorph 0.3.3 (2026-07-27) | pip |
| **uniGradICON** (2403.05780) | Registration foundation model, zero-shot across 12 datasets | "Zero-shot capabilities for new registration tasks" (abstract) | **Conflict: repo Apache-2.0; PyPI classifier MIT** | Released with the repo | unigradicon 1.0.5 (2026-02-25); icon-registration 1.1.8 | pip |
| **CT-FM** (2501.09001) | CT foundation model for classification, triage and retrieval | "Superior performance against state-of-the-art models" (abstract) | MIT | **Apache-2.0** (HF `project-lighter/ct_fm_feature_extractor`, `ct_fm_segresnet`) | HF 2025-01/02 | `project-lighter` |
| **Merlin** (2406.06512) | CT vision-language model: findings classification, chronic disease prediction, report generation | Validated on "44,098 CT scans from 3 independent sites" externally (abstract) | MIT | **MIT** (HF `stanfordmimi/Merlin`, updated 2026-09-16) | Active (commit 2026-09-24) | Repo |
| RadFM (2308.02463) | Generalist 2D/3D radiology VQA and report generation | Outperforms OpenFlamingo, MedFlamingo, MedVInT and GPT-4V on RadBench (abstract) | MIT (2025) | HF `chaoyi-wu/RadFM` **untagged** | Release 2025-06-12 | Repo |
| **nnFoundation** | 3D radiology foundation models (see 2C) | 108 evaluation tasks (abstract) | Apache (nnssl) | **CC-BY-SA-4.0** | 2026-09-24 | nnU-Net |
| Curia / Curia-2 | See R3 | | | Research-only RAIL-M, gated (R3) | | |

### Decision guidance: 3D medical beyond segmentation
- **3D classification with small data:**
  1. Start with a MONAI 3D ResNet or DenseNet trained from scratch, and compare against CT-FM (Apache weights) as a frozen or fine-tuned encoder.
  2. Try nnFoundation encoders (share-alike weights) when licence-compatible.
  3. R3 cites a 10-encoder frozen CT benchmark (2608.05960) that found no universal SOTA.
- **Registration:** uniGradICON zero-shot first, then fine-tune. VoxelMorph for a learned baseline. Classical ANTs/SimpleITK (SimpleITK 2.5.6, Apache) as the reference.
- **Metrics:** Dice of warped labels, target registration error (TRE) on landmarks, and the percentage of voxels with non-positive Jacobian determinant (folding). Report all three; Dice alone rewards implausible warps.

---

## 12. Point clouds (reference level)

| Component | Status | Code licence | Weights licence |
|---|---|---|---|
| **Pointcept** (PTv3 2312.10035, Sonata 2503.16429, Concerto 2510.23607, Utonia 2603.03283) | **v1.7.0 "Utonia" (2026-04-02)**; active (commit 2026-09-11) | MIT | PTv3: MIT (HF). **Sonata: CC-BY-NC 4.0** per upstream README ("restricted by NC of datasets like HM3D, ArkitScenes"); the HF mirror `Pointcept/Sonata` is tagged MIT, a conflict. **Concerto and Utonia: CC-BY-NC-4.0** (HF) |
| facebookresearch/sonata | Last commit 2025-06-04 | Apache-2.0 | CC-BY-NC 4.0 |
| OpenPCDet (LiDAR 3D detection) | Last release v0.5.2 (2022-01-05); last commit 2025-10-08 | Apache-2.0 | Model zoo |
| MinkowskiEngine | Last release 0.5.4 (2021-05-21); last commit 2023-03-22 | MIT | n/a; **unmaintained**, build against current CUDA and torch not verified |
| spconv | 2.3.8 (2024-12-15) | Apache-2.0 | n/a |
| torchsparse | v2.0.0 (2023-06-18); last commit 2025-02-24; not on PyPI | MIT | n/a |
| Open3D | 0.20.0 (2026-09-16) | MIT | n/a |

Guidance: for segmentation or classification backbones, use **PTv3 in Pointcept** (MIT, from scratch or with your own pretraining). Sonata, Concerto and Utonia pretrained weights are NC. Avoid MinkowskiEngine for new work. For LiDAR detection, OpenPCDet is still the reference codebase but is stale. Metrics: mIoU (dataset-level) for segmentation; mAP/NDS (nuScenes) or 3D AP (KITTI, 40 recall points) for detection, with each benchmark's official devkit (not re-verified here).

---

## 13. Small-data and few-shot: when zero-shot or few-shot beats fine-tuning

| Regime | Evidence (verified) | Recommendation |
|---|---|---|
| Classification with a few labels per class | A linear classifier on a good embedding "outperforms state-of-the-art few-shot learning methods" (2003.11539). LP++ is a "surprisingly strong linear probe" for few-shot CLIP (2404.02285). Tip-Adapter is training-free (2111.03930). A Closer Look at Few-shot Classification (1904.04232) finds simple baselines competitive. R3 adds that CLIP prompt-learning needs per-task tuning to beat a linear probe (2312.12730) | **Frozen DINOv2 or SigLIP 2 features, then logistic regression or kNN, first.** Fine-tune (LP-FT, see R3) only once there are enough labels to beat it on a validation split |
| Detection with 0-10 shots, common categories | Zero-shot GroundingDINO 48 AP vs 33 AP for few-shot SOTA on COCO (2312.14494) | Open-vocabulary detector (OWLv2, LLMDet, Grounding DINO, SAM 3) with prompt tuning, or visual exemplars |
| Detection with 0-10 shots, unusual domains | RF100-VL: zero-shot VLM detectors under 2% on hard medical sets; "the need for few-shot concept alignment". The CVPR 2025 FSOD challenge winner beat the baseline by 17 mAP (2505.20612) | Pre-label with open-vocabulary models, correct by hand, fine-tune RF-DETR or D-FINE. Frustratingly-simple fine-tuning (TFA, 2003.06957) and DE-ViT (2309.12969) are few-shot baselines |
| Anomaly detection with 1-4 normal images | AnomalyDINO one-shot MVTec AUROC 96.6% (2405.14529) | Training-free patch kNN on DINOv2 |
| Promptable segmentation | SAM zero-shot in 2D medical imaging is "comparable to, but in certain cases, surpasses the current state-of-the-art" on six datasets (2305.00109) | SAM 2.1 or MedSAM for interactive labelling; nnU-Net once around 20+ labelled volumes exist (threshold not verified; decide on a validation split) |
| Retrieval with no labels | Global-feature reranking gives large gains at low cost (SuperGlobal, 2308.06954) | Frozen encoder plus reranking before metric learning |

---

## Cross-cutting summary: metric libraries that compute things correctly

| Task | Library (version, licence) | Must-set parameters and pitfalls |
|---|---|---|
| Detection, instance seg | pycocotools 2.0.11 (FreeBSD); faster-coco-eval 1.8.0 (Apache-2.0); torchmetrics 1.9.0 `MeanAveragePrecision` (backends pycocotools or faster_coco_eval) | `iouType` (default `'segm'`), `maxDets` (100 cap), GT `area` field defines size buckets, `iscrowd` key required, LVIS needs federated eval |
| Keypoints | pycocotools `iouType='keypoints'` | `maxDets=[20]`; `kpt_oks_sigmas` fixed to COCO-17; no "small" area range |
| Semantic seg | Your own confusion matrix, or OpenMMLab-style dataset-level mIoU | torchmetrics `MeanIoU` is per-sample and has no ignore index |
| Panoptic | torchmetrics `PanopticQuality`; COCO panopticapi | Thing/stuff split, void label |
| Medical seg | MONAI metrics; torchmetrics `DiceScore`, `HausdorffDistance` | Per-structure reporting (Metrics Reloaded 2206.01653); physical spacing for surface metrics |
| Tracking | TrackEval (official, GitHub only); `trackers eval`; py-motmetrics | HOTA + IDF1 + MOTA; fix the detector |
| OCR | jiwer 4.0.0; torchmetrics CER/WER | Normalisation; reading order |
| Tables / DocVQA | TEDS (1911.10683); ANLS (1905.13648) | TEDS vs TEDS-S; ANLS threshold |
| Anomaly | anomalib metrics (AUROC, AUPRO, PIMO, F1Max) | AUPRO `fpr_limit=0.3` default; threshold fitted on validation |
| Retrieval / re-ID | torchmetrics Retrieval*; revisitop; torchreid `rank.py` | Pass `indexes`; junk handling; same-camera removal |
| VLM benchmarks | lmms-eval 0.7.3 (MIT) | Fixed prompts and decoding; contamination |

---

## Licence traps (consolidated)

1. **Changed in 2026:** DEIMv2 and EdgeCrafter went Apache to NC on 2026-08-24. Surya went GPL to Apache on 2026-05-14 and Marker on 2026-07-17 (code only; weights stay revenue-capped OpenRAIL-M). Gemma moved from Gemma terms (Gemma 3, PaliGemma 2) to Apache-2.0 for Gemma 4.
2. **Mirror or metadata disagrees with upstream (follow the more restrictive):**
   - DA3-LARGE-1.1: HF Apache vs README CC BY-NC.
   - Depth Pro: HF `apple-amlr` vs card "Apple-ASCL" vs GitHub sample-code.
   - Sonata: HF MIT vs README CC-BY-NC.
   - ECDet_L: HF Apache vs licence file NC.
   - `harshaljanjani/DEIMv2_*`: Apache tag, but upstream is now NC and the DINOv3 backbone is custom-licensed.
   - EoMT-DINOv3: MIT tag, but ships delta weights on DINOv3.
   - SAM3-LiteText: Apache tag, but contains the SAM 3 image encoder.
   - DocLayout-YOLO: Apache weights tag, AGPL code.
   - InternVL3-8B: metadata says `qwen`, card says MIT plus Apache.
   - VisA: CC BY 4.0 upstream vs CC BY-NC-SA per the anomalib docstring.
   - VoxelMorph and uniGradICON: repo Apache vs PyPI MIT.
3. **NC weights frequently used by accident:**
   - SegFormer / MiT (including smp `mit_b*`), DA V2 B/L/G, DA3 L/G, VGGT-1B, MapAnything (non-apache), Pi3X, UniDepth, DUSt3R/MASt3R.
   - VideoMAE / V2, TimeSformer, Sapiens v1, WiLoR, Nougat, LayoutLMv3.
   - Qwen2.5-VL-3B (and derivatives such as Nanonets-OCR2-3B), Perception LM, Rex-Omni, InsightFace, SatMAE, LSKNet, Sonata, Concerto, Utonia, VISTA3D.
   - MVTec AD / AD 2 (data), DOTA (data).
4. **Copyleft:**
   - AGPL-3.0: all Ultralytics models (YOLOv8-YOLO26, YOLOE, the YOLO-World port, the RT-DETR wrapper, trackers), BoxMOT, DocLayout-YOLO code, MinerU2.5-2509 weights, albumentationsx (2.4.10, AGPL-3.0-only; see R2).
   - GPL-3.0: YOLO-World upstream, YOLOv9/v7 originals, mmyolo, the PANGAEA benchmark code.
   - Share-alike: MedSAM2 weights and nnFoundation (CC-BY-SA-4.0).
5. **Custom-commercial:**
   - SAM 3 / 3.1, Sapiens2 and VGGT-1B-Commercial (Meta licences with acceptable-use bans).
   - DINOv3 (and everything built on it: DEIMv2 S-X, EoMT-DINOv3, DINOv3-SAT, possibly RT-DETRv4 via distillation).
   - RF-DETR XL/2XL (PML 1.0).
   - MinerU code (MAU and revenue thresholds, attribution).
   - Chandra ($2M, no competing API), Surya and Marker weights ($5M).
   - Moondream (BUSL).
   - HunyuanOCR (no EU/UK/KR).
   - Qwen2.5-VL-72B (100M MAU).
   - TotalSegmentator licensed subtasks.
6. **Dataset-inherited restrictions:** Objects365-pretrained detectors (D-FINE warning), DOTA-trained OBB models, MVTec-trained anomaly models, and Sonata's NC-derived weights.

---

## Could not verify

- **Licence texts not obtained:**
  - RF-DETR "PML 1.0" (the `rfdetr-plus` repo licence file was not found on main/develop/master).
  - MANO (HaMeR dependency).
  - Metric3D weights (not stated in README).
  - SuperAnimal (DeepLabCut) weights (no HF tag).
  - RadFM and TrOCR weights (no HF tag).
  - DUSt3R/MASt3R HF weights (untagged; assumed to follow the repo's CC BY-NC-SA).
  - GOT-OCR2 GitHub code (no LICENSE file).
  - Video Depth Anything weights.
  - VideoPrism.
  - Real-IAD.
  - Re-ID dataset terms (Market-1501 and others).
  - The MobileCLIP-derived text tower in SAM3-LiteText.
- **Licence interpretations (legal questions, not facts):**
  - Whether DEIMv2 copies obtained before 2026-08-24 remain Apache-2.0.
  - Whether detectors distilled from a DINOv3 teacher (RT-DETRv4, EdgeCrafter) are DINOv3 "derivative works".
  - Whether ViTPose weights inherit NC terms from MAE pretraining.
  - Whether Gemma 4 carries a separate prohibited-use policy in addition to Apache-2.0 (the licence page header reads "Apache License 2.0 / Gemma"; its body was not fully readable).
- **Sapiens2 licence:** a keyword search found no non-commercial clause, but the full text was not reviewed.
- **SAM 3.1 licence:** assumed identical to SAM 3 (the HF card is gated and was not readable).
- **Benchmark numbers not verified:**
  - YOLOv9, YOLOv13, LW-DETR paper numbers (SAB numbers used instead); LLMDet; MM-Grounding-DINO; Florence-2; CLIP-ReID; FastFlow; docTR; EasyOCR; Tesseract.
  - X3D / MViT in PyTorchVideo.
  - VLM general benchmarks (MMMU, DocVQA etc.) for Qwen3-VL, Qwen3.5, InternVL3.5 and Gemma 4 (vendor cards only; not extracted).
  - Chandra OCR 2 on OmniDocBench (not in the parsed table).
- **Environment details not verified:**
  - Hardware for the Ultralytics "CPU ONNX" latency column.
  - The `pip` package names and versions for MediaPipe, DeepLabCut and Clay.
  - Whether detectron2 builds against torch 2.14.
  - mmcv wheel availability for torch 2.7 and later (only the absence of torch 2.5 and 2.6 index pages was checked).
- **Maintenance or provenance not confirmed:**
  - Whether the PyPI `faiss-gpu` 1.15.1 is an official Meta release.
  - VLMEvalKit's PyPI status.
  - Whether open-cd is still mmseg-based.
  - Ultralytics YOLO26 semantic-segmentation and depth-estimation task weights and numbers (docs list the tasks, but specific models were not checked).
- **Metric conventions not re-checked:**
  - ANLS threshold 0.5 (standard, not re-read from the paper).
  - The ADE20K `reduce_zero_label` convention.
  - `MulticlassJaccardIndex` `ignore_index` behaviour.
  - Temporal-action-localisation tIoU threshold sets.
  - KITTI and nuScenes metric details.
- **Independent verification of vendor claims:** PP-OCRv6's "+4.6% / +5.1% / 5.2x CPU", Qwen3.5 "outperforms Qwen3-VL", and the TeleOCR and OvisOCR2 leaderboard entries (HF licence tags only; base models not checked).
