# Metric libraries that compute things correctly

Source: research R4 (verified 2026-09-27). Per-task metric pitfalls are also in each task file of the cv-model-selection skill's references folder.

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
