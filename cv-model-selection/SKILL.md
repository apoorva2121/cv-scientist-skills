---
name: cv-model-selection
description: Use when choosing a model, architecture, backbone, foundation model or framework for any vision task (classification, detection, segmentation, promptable segmentation, pose, depth, video, tracking, OCR and documents, anomaly detection, retrieval and re-ID, vision-language, remote sensing, 3D medical, point clouds); when deciding between zero-shot, open-vocabulary or VLM approaches and a fine-tuned specialist; when a baseline is needed; when checking whether code or weights may be used commercially; or when the only evidence for a model is a vendor's benchmark table.
---

# Vision Model Selection

## Overview

Stage 6 picks two or three candidate model families, proves the task is learnable with baselines,
clears their licences, and carries the survivors forward. The choice rests on three kinds of
evidence, in this order:

1. **This data**: scores on the fixed Stage 2 folds, compared with paired statistics.
2. **Independent re-measurements** over vendor tables (re-measured COCO AP is often 0.5-4 points
   below the vendor's headline).
3. **The upstream licence**, read at the source. Hugging Face tags and mirrors are wrong in both
   directions; follow the stricter one.

Everything here was verified on 2026-09-27 and changes monthly. Check versions and licences again
at the source before relying on them.

## Step 1: Licence gate, before downloading anything

Stage 0 records whether commercial use, AGPL and non-commercial weights are allowed. For every
candidate, record the code licence, the **weights** licence (the binding one), the source URL and
the date checked. Traps that caught real projects:

| Trap | Detail |
|---|---|
| Relicensed models | DEIMv2 and EdgeCrafter became non-commercial on 2026-08-24; some mirrors still say Apache-2.0 |
| Copyleft frameworks | Ultralytics YOLO (incl. YOLO26): AGPL-3.0 covers the code and the weights you train with it |
| Popular non-commercial weights | SegFormer / smp `mit_b*`, Depth Anything V2 Base/Large, VideoMAE, Sapiens v1, InsightFace, LayoutLMv3, Nougat, ConvNeXt V2, MAE, Apple AIMv2/DFN/MobileCLIP |
| Custom licences needing review | DINOv3 (and models built on it), SAM 3, RF-DETR XL/2XL (PML 1.0), revenue-capped OCR weights (Surya, Marker, Chandra) |
| Training data licences | Objects365-pretrained detector weights are not commercially cleared; MVTec AD and DOTA are academic-only |
| Derivatives | Fine-tunes and students distilled from a restricted teacher may inherit its terms |

Full list with sources: `references/licence-traps.md`. If a candidate fails the gate, log it as
"skipped: licence" and move to the next one. Do not quietly use it.

## Step 2: Baselines that prove learnability

Run these on the Stage 2 folds before any large model:

| Baseline | What it tells you | Tool |
|---|---|---|
| Prior (majority class, mean box, empty mask) | the floor | a few lines |
| Zero-shot (SigLIP 2 / PE text prompts; an open-vocabulary detector) | whether the task is "web-common" | open_clip, transformers |
| Frozen features + linear probe or k-NN | the number every fine-tune must beat; a quick label-quality check | `cv-method-comparison/scripts/probe_screen.py`, `cv-data-audit/scripts/embed_images.py` |
| A small fine-tuned model | the task-specific floor | `cv-training/scripts/train_*.py` |

**Learnability verdict.** If the frozen probe is near the prior, suspect labels, framing or data
before models: return to Stages 1-3. Bigger models rarely fix a task that a good encoder cannot
separate at all.

## Step 3: Shortlist two or three families

Read only the reference file for your task. Permissive first choices as of 2026-09-27:

| Task | File | Try first (permissive) | Then |
|---|---|---|---|
| Classification | `backbones.md` | DINOv2-reg, SigLIP 2 or PE frozen + probe, then LP-FT | ConvNeXt, EVA-02, CAFormer fine-tunes; MobileNetV4 for edge |
| Detection | `detection.md` | RF-DETR M/L (`rfdetr`), D-FINE or RT-DETRv2 (transformers) | torchvision Faster R-CNN (template baseline); YOLO26 only if AGPL is allowed. torchvision `*_320_*` variants shrink every input to 320 px: fast, but small objects vanish |
| Semantic / panoptic segmentation | `segmentation.md` | EoMT (DINOv2), Mask2Former / OneFormer (transformers) | smp Unet/UPerNet with a ConvNeXt encoder for small data (template) |
| Instance segmentation | `segmentation.md` | RF-DETR-Seg | torchvision Mask R-CNN v2 |
| Promptable masks, labelling help | `segmentation.md` | SAM 2.1 | SAM 3 if its licence is accepted |
| Medical 3D segmentation | `segmentation.md` | nnU-Net v2 ResEnc L | TotalSegmentator tasks for anatomy |
| Pose | `pose.md` | rtmlib RTMPose / RTMO (CPU) | ViTPose via transformers |
| Depth | `depth.md` | Depth Anything V2-Small, MoGe-2 (relative); DA3 metric, MoGe (metric) | multi-view: DA3-Base, Pi3 |
| Video, tracking | `video.md` | V-JEPA 2; the `trackers` package (ByteTrack, OC-SORT) | SAM 2 for video object segmentation |
| OCR, documents | `ocr-documents.md` | PP-OCRv5 (lines); PaddleOCR-VL-1.6 or GLM-OCR (pages); Docling (pipeline) | docTR; Tesseract for clean scans |
| Anomaly detection | `anomaly.md` | anomalib PatchCore / EfficientAD; AnomalyDINO for 1-16 normal images | Dinomaly for many products in one model |
| Retrieval, re-ID | `retrieval.md` | DINOv2 or SigLIP 2 embeddings + FAISS + reranking | metric learning (pytorch-metric-learning) once frozen features plateau |
| Vision-language tasks | `vlm.md` | Qwen3-VL, Qwen3.5, Gemma 4; Florence-2 or SmolVLM2 on CPU | compare with a fine-tuned specialist first |
| Remote sensing | `remote-sensing.md` | a supervised UNet baseline, always; Prithvi-EO-2.0 or TerraMind (multispectral) | DINOv2 for RGB very-high-resolution imagery |
| 3D medical (not segmentation) | `medical-3d.md` | MONAI 3D ResNet vs CT-FM encoder; uniGradICON for registration | VoxelMorph |
| Point clouds | `point-clouds.md` | reference only | |
| Few labels | `few-shot.md` | frozen features + probe or k-NN; open-vocabulary detection for common classes | fine-tune only when it beats the probe |

Rules for the shortlist:
- **Make candidates differ** (CNN versus ViT, frozen versus fine-tuned, different pre-training).
  Diverse survivors make better ensembles in Stage 9.
- **Zero-shot and VLM approaches** win on web-common categories and for pre-labelling. They fail
  off-distribution: zero-shot detectors scored under 2% on hard medical sets. Always compare them
  with a fine-tuned specialist on held-out data from this project.
- **Prefer maintained stacks.** MMDetection, MMSegmentation, MMRotate and detectron2 builds are
  frozen or hard to install on current PyTorch; avoid them for new work.
- **Fit the compute.** Estimate on this hardware (`cv-training` Step 1) before choosing a size.

## Step 4: Screen the shortlist on the same folds

- Frozen encoders: `probe_screen.py --model <timm name>` for each, same variants and folds.
- Fine-tuned candidates: the `cv-training` templates, same folds and seeds, compared with
  `cv-method-comparison/scripts/compare.py`.
- **Models with their own trainer** (rfdetr, anomalib, nnU-Net, Ultralytics): train them on the
  same split (export it to their format), save validation predictions in the common format
  (COCO detections, per-image counts, probabilities), and score them with the same evaluation code
  as every other candidate. Framework-printed metrics use different defaults, as the traps below show.
- Carry the best two or three into `search_space.json`, each with its training `args` and licence.

## Metric traps: read before comparing any numbers

- `pycocotools.COCOeval` defaults to `iouType='segm'`; pass `'bbox'`. It caps at 100 detections per
  image; raise `maxDets` for crowded scenes. Size buckets use each annotation's `area` field.
- "mAP50" in YOLO tooling is not COCO AP (AP50:95). Report both, and say which.
- `torchmetrics.segmentation.MeanIoU` averages per image and has no ignore index; papers use
  dataset-level mIoU (as in `cv-training/scripts/train_segmenter.py`).
- Relative depth must be scale-and-shift aligned per image before scoring; metric depth must not.
- Top-down pose numbers depend on the person detector used.
- `sv.ByteTrack` is being removed from supervision (use `trackers`); the PyPI `trackeval` is a fork.
- anomalib's AUPRO integrates only up to a false-positive rate of 0.3 by default.

More: `references/metric-libraries.md` and the metrics section of each task file.

## Gate (all must exist)

- the licence table: every candidate's code and weights licence, source and date checked
- the baseline table on the folds: prior, zero-shot where it applies, frozen probe, small fine-tune
- the learnability verdict
- the screening table for the shortlist; survivors in `search_space.json` with `args` and licence
- decision-log block: considered, applied, skipped with reasons (licence, maintenance, compute,
  evidence), with elapsed clock time

## Red flags

| Thought | Reality |
|---|---|
| "It is top of the leaderboard" | Vendor numbers are often above re-measured ones; your folds decide |
| "The Hugging Face tag says Apache-2.0" | Tags and mirrors disagree with upstream in both directions; read the upstream licence |
| "Ultralytics is the easy choice" | AGPL covers the weights you train; only if Stage 0 allowed it |
| "A VLM can do this zero-shot" | Off-distribution it can fail badly; compare with a fine-tuned specialist on your data |
| "Use the biggest model" | Measure the compute first; the probe baseline often wins at small data |
| "The framework printed mAP 0.6" | Different defaults (mAP50, maxDets, area buckets); re-score with the common evaluation |
| "MMDetection has every model" | Frozen since 2024 and hard to install on current PyTorch |
