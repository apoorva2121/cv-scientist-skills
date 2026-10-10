---
name: cv-postprocessing
description: Use when turning a vision model's raw outputs into decisions - choosing score thresholds or operating points (target recall, max F1, cost-based), calibrating probabilities (temperature scaling, isotonic, Platt), correcting for a different class mix in deployment, conformal prediction sets or abstention, NMS / Soft-NMS / class-agnostic NMS, weighted boxes fusion of several detectors or TTA passes, test-time augmentation, mask cleanup, sliding-window inference, or tracking post-processing; also when probabilities look overconfident or a model's confidence is used to make decisions.
---

# Vision Post-processing

## Overview

Post-processing is where metrics become decisions. Three rules:

1. **Every choice is fitted on validation predictions and frozen before the test set.**
   Thresholds, temperatures, fusion weights, minimum object sizes and TTA choices all count as
   training; the test set only confirms them. Measure each choice cross-fitted: fit on part of the
   validation images, score on the rest, so the number you report is what deployment will see.
2. **Each technique is a candidate that must beat "do nothing"** with a paired comparison on the
   same images. Several popular ones often hurt: standard TTA lowered accuracy in 11 of 12 medical
   model-dataset pairs in a 2026 study; temperature scaling can improve log-loss while making
   calibration worse; soft-NMS is not a way to fuse models.
3. **Keep the metric and the decision separate.** Compute threshold-free metrics (AUROC, AP, COCO AP)
   on raw outputs; choose the operating point afterwards. Thresholding before computing mAP
   truncates the precision-recall curve and lowers AP.

Evidence, verified APIs and licences for every method: `references/methods.md`.

## Inputs

Saved validation predictions from `cv-training`: `val_probs.npz` (classification), `val_dets.json`
(detection, COCO format), `val_seg_counts.npz` or saved probability maps (segmentation). Candidate
folders (`runs/<candidate>/<fold_seed>/`) give out-of-fold predictions over the whole train part.

## Step 1: Operating point (the decision the user cares about)

From the problem statement: target recall, maximum F1, or error costs.

```bash
python scripts/postprocess_cls.py threshold --val runs/<candidate> --positive defect --target-recall 0.95
python scripts/postprocess_cls.py threshold --val runs/<candidate> --positive defect --cost-fp 1 --cost-fn 20
python scripts/postprocess_cls.py multilabel --val val_probs.npz
python scripts/postprocess_det.py thresholds --gt annotations.json --images val_ids.txt --dets val_dets.json
```

- The script reports the recall, precision and cost the rule achieves on held-out folds, with
  Wilson intervals, next to the plain argmax rule. With few positives the recall interval is wide:
  say so rather than promise the target.
- F1 and accuracy are improper for cost-sensitive decisions. When costs are known, use the cost rule
  (with calibrated probabilities the Bayes threshold is c_FP / (c_FP + c_FN)).
- Per-class thresholds for rare labels overfit small validation sets; check them on the holdout.
- If deployment class frequencies differ from training (screening versus referral, a new site),
  correct the priors first (EM with calibrated probabilities, `references/methods.md` B4), then set
  thresholds.

## Step 2: Calibration, only if probabilities are used

Calibrate when a probability is shown to a person, compared across models, thresholded by cost, or
fed to conformal prediction. Otherwise skip it and say why.

```bash
python scripts/postprocess_cls.py calibrate --val runs/<candidate>
```

- Temperature scaling suits neural softmax outputs and never changes the predicted class. The script
  fits it cross-fitted and requires the held-out Brier score to improve as well as NLL.
- For k-NN vote fractions or tree scores, a single temperature fails (NLL improves, calibration
  worsens). Refit the k-NN vote kernel `softmax(sim / T)`, or use isotonic regression (about 1,000+
  samples) or histogram binning.
- Report NLL, Brier and an adaptive-bin or debiased ECE with a reliability diagram; equal-width ECE
  is biased on small sets.
- Detector scores used as probabilities: Platt or isotonic on matched detections (netcal).
- Recalibrate after any change to the loss, sampling, label smoothing, EMA, TTA or ensembling.

## Step 3: Uncertainty and abstention (when errors are costly)

- **Conformal prediction sets** (MAPIE `SplitConformalClassifier`, BSD-3): a set that contains the
  true class with a chosen probability, e.g. 90%. The guarantee is marginal; check coverage per
  class and per site, and use data disjoint from calibration fitting and testing. Temperature
  scaling changes set sizes.
- **Abstention**: reject low-confidence images (max-softmax threshold); report the risk-coverage
  curve. Label smoothing harms this; logit normalisation fixes it.

## Step 4: Detection and segmentation outputs

- **NMS**: class-wise NMS by default; class-agnostic when classes exclude each other in space;
  Soft-NMS or DIoU-NMS only when crowding is the documented error. DETR-family models need no NMS.
  Raise `maxDets` above 100 for crowded scenes.
- **Fusing models or TTA passes**:
  `python scripts/postprocess_det.py fuse --gt annotations.json --images val_ids.txt --dets a.json b.json`.
  Weighted boxes fusion needs coordinates normalised to [0, 1], which the script handles (pixel
  boxes are silently dropped otherwise). Then compare the fused file with the best single model with
  `cv-evaluation/scripts/metrics_ci.py det --dets fused.json --dets-b best.json`. Keep fusion only if
  the primary metric's interval excludes zero and the extra inference cost fits the budget.
- **Segmentation**: average probabilities (not argmax masks) across models or views. Remove small
  components and fill holes with sizes chosen on validation, per class; that step can delete true
  small lesions. Dense CRF rarely helps modern networks. Large images: sliding-window inference with
  overlap and Gaussian weighting (MONAI `sliding_window_inference(..., mode="gaussian")`; MONAI and
  nnU-Net defaults differ, so set them explicitly).
- **Tracking**: low-score box association (BYTE), interpolation of short gaps, re-ID only when
  identity switches dominate; the `trackers` package (Apache) rather than AGPL `boxmot`.

## Step 5: Test-time augmentation, as a candidate

Include the identity view, try intensity-only views before geometric ones, use only operations
seen in training, and compare with the single pass on the same validation images (paired, with
intervals). For detectors trained with the template, TTA costs only inference:

```bash
python scripts/det_tta.py --model runs/final/model.pt --image-dir IMAGE_DIR --coco annotations.json     --images val_ids.txt --views orig,hflip,scale1.5 --out tta/
python scripts/postprocess_det.py fuse --gt annotations.json --images val_ids.txt     --dets tta/dets_orig.json tta/dets_hflip.json tta/dets_scale1.5.json --methods wbf
```

Then test the fused file against `tta/dets_orig.json` with `metrics_ci.py det` (paired). Larger
scales help small objects at the square of the compute. BatchNorm models on shifted data are where TTA most often hurts. Recalibrate after TTA.
Test-time adaptation (TENT and similar) can collapse on long streams; only with a reset policy and
a measured gain.

## Gate (all must exist)

- the operating-point table: rule, frozen threshold(s), cross-fitted recall/precision/cost with intervals
- calibration before/after (NLL, Brier, ECE) with the verdict, or the reason it was skipped
- each fusion, TTA, NMS or refinement candidate versus "do nothing": paired result and the decision
- a frozen post-processing config (thresholds, temperature, fusion settings) saved with the model
- decision-log block with elapsed clock time

## Red flags

| Thought | Reality |
|---|---|
| "Pick the threshold on the test set" | Then the test number is optimistic; thresholds come from validation |
| "TTA is a free gain" | It often hurts, especially BatchNorm models under shift; measure it |
| "Temperature scaling always calibrates" | It can improve NLL while worsening ECE and Brier; check both |
| "The recall target is met" | With few positives the interval is wide; report it |
| "Soft-NMS to merge two models" | Soft-NMS keeps duplicates at lower scores; use weighted boxes fusion |
| "Threshold the detections, then compute mAP" | Truncates the curve; compute AP on raw outputs |
| "Remove every small blob" | Can delete true small lesions or objects; size limits come from validation |
