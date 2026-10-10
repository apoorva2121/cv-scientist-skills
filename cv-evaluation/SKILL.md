---
name: cv-evaluation
description: Use when scoring a finished vision model on its test set or a deployment sample; when reporting accuracy, F1, AUROC, COCO AP, mIoU, Dice, HD95, depth or retrieval metrics with confidence intervals; when comparing two models on the same test images; when analysing errors by slice (camera, site, object size, subgroup), robustness to shift, or out-of-distribution inputs; when producing explanations (Grad-CAM) or a model card; or when exporting to ONNX, checking parity and measuring latency before hand-off.
---

# Vision Evaluation

## Overview

Stage 10 answers one question honestly: **how will this model perform where it will be used?**

1. **The test set is scored once**, with the frozen model, transform and post-processing. Confirm
   the split is untouched first (`cv-validation-design/scripts/cv_folds.py` `verify_untouched`). A
   labelled deployment sample or external site, when it exists, is a second test, also scored once.
2. **Every headline number carries a 95% interval** at the unit of independence: images, or whole
   patients, sites or videos when images are nested in them.
3. **Comparisons are paired**: both models on the same images, with an interval on the difference.
   A difference whose interval includes zero is reported as "no evidence of a difference".
4. **State how each metric is aggregated.** Library defaults silently change numbers:
   torchmetrics `MulticlassAccuracy` defaults to macro averaging, `MeanIoU` averages per image,
   `HausdorffDistance` cannot give HD95, and `RetrievalRecall` is not hit rate.

Evidence and APIs: `references/methods.md` (metrics, intervals, slices, robustness, explanations,
fairness, reporting), `references/metric-libraries.md`, `references/deployment.md`.

## Step 1: Choose the metrics before looking at test results

From the problem statement and `cv-scientist/references/routing.md`: the primary metric, the
operating point, and secondary metrics. For medical images follow Metrics Reloaded (Dice or IoU plus
a boundary metric such as NSD or HD95 for segmentation; aggregate per image). For natural-image
segmentation benchmarks, dataset-level mIoU. For detection, COCO AP and AP50 plus the operating-point
precision and recall. Always add calibration (NLL, Brier, ECE) when probabilities are used.

## Step 2: Score with intervals, slices and paired comparisons

```bash
python scripts/metrics_ci.py cls --pred test_probs.npz --meta meta.csv --group split_group --slice camera_id
python scripts/metrics_ci.py det --gt test_annotations.json --dets test_dets.json --slice site --n-boot 500
python scripts/metrics_ci.py seg --counts test_seg_counts.npz --meta meta.csv --slice scanner
# paired comparison with the baseline or runner-up, same images
python scripts/metrics_ci.py cls --pred final.npz --pred-b baseline.npz --group split_group
```

- `--group` resamples whole groups. Use it whenever images share a patient, session, video or scene.
- Slices: every metric per value of a metadata column (camera, site, size bin, subgroup), each with
  its interval and n. Slices under 30 images are flagged as too small to act on.
- Per-class precision, recall and F1 with Wilson intervals, and the most-confused class pairs, come
  with every classification report.
- Plan test size before the run: at 90% accuracy the Wilson half-width is about 6 points at n=100,
  3.4 at 300 and 1.9 at 1,000. If the interval is wider than the decision needs, say so.

## Step 3: Deployment gap, robustness and out-of-distribution inputs

- **Real shift first**: the deployment sample or external site, scored once, next to the internal
  test. A large gap is a data problem (`cv-data-audit` shift report), not something to tune away.
- **Synthetic corruptions** (blur, noise, JPEG, lighting) only as a supplement, and only for the
  degradations deployment will really have. ImageNet-C is no longer out-of-distribution for
  web-scale models.
- **Out-of-distribution flagging**, if inputs may be off-task: start with max-softmax probability and
  energy (free), then Mahalanobis on L2-normalised features. Choose on a validation OOD set that
  matches the expected shift.

## Step 4: Error analysis

- Look at the worst errors (most confident mistakes, missed objects) as images, with predictions
  drawn (`cv-data-audit/scripts/render_annotations.py`). Sort them into causes: label errors, true
  ambiguity, small or occluded objects, shift, a shortcut cue.
- Detection: separate classification, localisation, duplicate, background and missed errors
  (TIDE-style), and per-size results (COCO buckets use the annotation `area` field).
- Label errors found in the test set: report them. Do not quietly fix test labels after seeing
  predictions; if labels are corrected, re-score both models, and report before and after.

## Step 5: Explanations, only with checks

Saliency maps (grad-cam 1.5.7, MIT: `GradCAM`, `HiResCAM`; ViTs need a `reshape_transform`) generate
hypotheses. They are not evidence. Before showing a map, run the model-randomisation check: randomise
the weights layer by layer, and if the map barely changes, it is not explaining the model. Shortcut
claims are proven by intervention (the Stage 3 masking and cue-present tests), not by maps.

## Step 6: People and patients

When images depict people or patients, report per-subgroup metrics (true- and false-positive rates,
AUROC, calibration) with intervals and n (fairlearn `MetricFrame`, MIT). State the groups that could
not be evaluated because data was missing.

## Step 7: Export, parity and speed (hand-off)

```bash
python scripts/export_check.py --model runs/final/model.pt --image-dir IMAGE_DIR --files test_files.txt --int8
python scripts/export_check.py --module my_model.py:build --image-dir IMAGE_DIR --files test_files.txt
```

`--model` takes the classifier template's `model.pt`; `--module` takes any classifier, for example a
frozen backbone with the fitted linear head copied into an `nn.Linear` (the script header shows the
builder contract). Detectors: the legacy exporter fails on torchvision Faster R-CNN, so keep detectors
in PyTorch, or use the dynamo exporter after installing `onnxscript` (ask before installing).

- Exports to ONNX. In torch 2.14 the default exporter needs `onnxscript`, so the script uses the
  legacy exporter. It then checks parity on real images: the predicted class must agree on every
  image, within a probability tolerance.
- Latency at batch 1 after warm-up: p50, p90 and p99, with the hardware and thread count.
- INT8 is a candidate, not a default: on this project's CPU, static INT8 was slower than fp32 ONNX
  and changed 6% of predictions. Keep it only if agreement and the task metric hold on held-out data.
- Serving code must reproduce the evaluation transform: `cv-preprocessing/scripts/serve_parity.py`.
- On CPU, train and serve in fp32: bf16 autocast was about 36x slower here. `torch.compile` needs a
  C++ compiler on Windows.

## Gate (all must exist)

- the test-set hash check, and a note that the test (and deployment sample) was scored once
- the metrics report: primary and secondary metrics with 95% intervals, aggregation stated, per-class results
- paired comparison with the baseline and the runner-up
- slice table (camera, site, size, subgroup) with n and intervals; the deployment or external result
- error analysis with example images and causes
- export report: parity verdict, latency percentiles and hardware; INT8 decision
- report.md filled per `cv-scientist/references/templates.md`, including what the model must not be used for

## Red flags

| Thought | Reality |
|---|---|
| "Accuracy 0.94" | Without an interval, n and the aggregation it is not a result |
| "B is better: 0.912 vs 0.905" | Paired interval on the difference, or it is a tie |
| "Score the test set again after this small fix" | The test is used once; later changes go to the holdout |
| "torchmetrics gives the standard number" | Check the averaging default; several differ from papers |
| "Grad-CAM shows it looks at the lesion" | Maps can ignore the model; run the randomisation check and prove shortcuts by intervention |
| "INT8 is always faster" | Measured slower here, with changed predictions; measure on the target device |
| "The internal test says 95%, so production will too" | Only the deployment sample or external site says that |
