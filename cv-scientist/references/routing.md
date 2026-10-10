# Routing: task families and data types

Read the row for your task and your data type at Stage 0. Model names for each family live in
`cv-model-selection` (checked against live leaderboards); this file routes metrics, splits and stages.

## Contents
1. Task families
2. Data types
3. Operating points

## 1. Task families

| Task | Output | Primary metric | Also report | Stages that change |
|---|---|---|---|---|
| Single-label classification | one class per image | top-1 accuracy (balanced), macro-F1 when imbalanced | per-class recall, confusion matrix, calibration (ECE), top-5 for many classes | 6 (linear probe baseline), 8 (temperature scaling) |
| Multi-label classification | set of tags | mAP over labels (macro), micro/macro F1 at chosen thresholds | per-label AP, label cardinality error | 2 (iterative stratification), 8 (per-label thresholds) |
| Fine-grained classification | one of many similar classes | top-1, macro-F1 | confusion between sibling classes | 4 (higher resolution), 5 (careful colour aug), 6 (strong pretrained) |
| Object detection | boxes + classes | COCO mAP@[.5:.95] | AP50, AP75, AP small/medium/large, per-class AP, recall at the operating point | 4 (tiling), 8 (NMS / fusion), 10 (error breakdown) |
| Oriented / rotated detection | rotated boxes | mAP with rotated IoU (DOTA-style) | per-class AP | 1 (angle convention), 5 (rotation-aware aug) |
| Semantic segmentation | class per pixel | mIoU | per-class IoU, Dice, boundary IoU / boundary F1 | 4 (mask resize nearest), 8 (mask refinement) |
| Instance segmentation | mask per object | mask mAP@[.5:.95] | box mAP, AP by size, boundary AP | as detection + segmentation |
| Panoptic segmentation | things + stuff | PQ (SQ x RQ) | PQ things / stuff | as instance + semantic |
| Keypoints / pose | keypoints per instance | OKS-based AP (COCO keypoints) | PCK@alpha, per-joint error | 5 (flip pairs), 8 (heatmap decoding) |
| Monocular depth | depth per pixel | AbsRel, delta<1.25 | RMSE, scale-invariant log error; scale/shift alignment stated | 4 (depth scaling), 5 (geometric aug only) |
| Video classification / action | label per clip | top-1 (clip and video level) | per-class recall | 2 (split by video/source), 4 (frame sampling) |
| Temporal localisation | segments in time | mAP at temporal IoU thresholds | per-class AP | 4 (feature extraction) |
| Multi-object tracking | tracks | HOTA | IDF1, MOTA, ID switches | 8 (tracker association) |
| Anomaly / defect detection | normal vs anomalous (+ map) | image AUROC; pixel AUPRO | F1 at the operating point, false alarms per 1000 | 2 (normal-only training), 6 (memory-bank / reconstruction families) |
| Retrieval / similarity | ranked neighbours | Recall@K, mAP@R | NDCG | 2 (split by identity), 7 (metric-learning losses) |
| OCR / text recognition | strings | CER, WER | exact-match rate, per-field accuracy | 4 (deskew, binarise), 5 (no flips) |
| Document understanding | fields / layout | field-level F1, ANLS | layout mAP | 6 (document models / VLMs) |
| Image regression | number per image | MAE, RMSE | calibration of intervals | 7 (regression losses) |
| Medical segmentation (2D/3D) | masks | Dice | HD95, NSD, per-structure Dice | 2 (patient split), 4 (spacing, windowing), 6 (volumetric models) |
| Vision-language (VQA, captioning, grounding) | text / boxes from prompts | task accuracy (VQA), CIDEr (captions), grounding AP | human spot-check, hallucination rate | 6 (VLM / open-vocabulary models) |
| Point clouds | labels / boxes / masks on points | mIoU (segmentation), mAP (3D detection) | per-class metrics | reference level only |

## 2. Data types

| Data type | Formats | Decode and orientation | Split key (typical) | Normalisation | Notes |
|---|---|---|---|---|---|
| Consumer / natural RGB | JPEG, PNG, WebP, HEIC | EXIF orientation; colour mode to RGB; sRGB | photographer, session, scene, product | backbone's pretraining stats | Watermarks and borders are common shortcuts |
| Industrial / inspection | PNG, TIFF, BMP (often 8/16-bit gray) | bit depth; fixed pose | line, batch, day, part | dataset stats | Anomaly setting (few defects) is common |
| Medical 2D (X-ray, dermoscopy, fundus) | DICOM, PNG | DICOM windowing, photometric interpretation, laterality | patient | modality-specific | No flips if laterality matters; scanner is a shortcut |
| Medical 3D (CT, MRI) | DICOM series, NIfTI | voxel spacing, orientation (RAS/LPS), intensity units (HU) | patient | windowing / z-score per volume | Resample to common spacing; patch-based training |
| Pathology / microscopy | WSI (SVS, NDPI), TIFF | pyramid levels, magnification (microns per pixel) | patient / slide / site | stain normalisation | Tiles from one slide never straddle splits |
| Remote sensing | GeoTIFF, multispectral (4-13+ bands), SAR | band order, reflectance scaling, no-data values | spatial blocks, region, acquisition date | per-band stats | Spatial autocorrelation: random splits leak |
| Documents | PDF, TIFF, scans, photos | DPI, rotation/deskew, colour to gray | document / template / source | none or per-image | No flips; text orientation matters |
| Video | MP4, frames | frame rate, timestamps, codec | video / camera / day | per-frame, backbone stats | Frames from one video never straddle splits |
| Thermal / depth sensors | 16-bit PNG, TIFF, raw | units (Kelvin, millimetres), invalid pixels | device, session | physical-unit scaling | Colour augmentation is meaningless |
| Point clouds | PLY, LAS/LAZ, PCD | coordinate frames, units | scene / sequence | centring and scaling | Reference level only |

## 3. Operating points

Every task needs one: the threshold, recall, precision or false-alarm rate the deployment runs at.
Write it into the problem statement, choose it on validation data (Stage 8), and report the primary
metric **and** the operating-point metrics on test.
