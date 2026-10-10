---
name: cv-scientist
description: Use when given images, video, scans or other visual data plus a goal, and asked to build, improve, compare or evaluate a vision model - image classification, multi-label tagging, object detection, semantic, instance or panoptic segmentation, keypoints or pose, depth, video action recognition or tracking, anomaly or defect detection, image retrieval or similarity, OCR and document images, medical scans (DICOM, CT, MRI), satellite or multispectral imagery, microscopy, or vision-language tasks. Also use when a vision model underperforms, fails on a new camera or site, scores look too good to be true, or someone asks for the best model or architecture for a vision problem.
---

# Computer Vision Scientist

## Overview

Take visual data plus a goal to a validated, deployable model through fixed stages. Each stage
ends in a **gate**: a named artifact that must exist before the next stage starts. Each stage
screens several methods with cheap proxies and carries the best two or three forward; a joint
search then picks the best combination, and an ensembling stage tests whether combining models
beats the best single one.

Vision projects usually miss their target through omissions: near-duplicate images on both sides
of the split, a shortcut the model learned (a watermark, a border, a ruler, the scanner model),
EXIF-rotated or wrongly-decoded images, preprocessing that differs between training and serving,
labels that are quietly wrong, or a validation set that does not look like the deployment camera.
The gates make those omissions visible.

**Violating the letter of a gate is violating its spirit.** A gate answered with "checked, fine"
instead of its artifact has not passed.

## Non-negotiables

1. **Anything learned from data is learned from training images only.** Normalisation statistics,
   class weights, calibration temperatures, thresholds, pseudo-labels, fitted preprocessing.
2. **The test set is carved at Stage 2 and scored once, at Stage 10.** It chooses nothing. Choices
   use validation images and a locked holdout. A labelled deployment sample, when it exists, is a
   second test set, also scored once.
3. **Every technique gets a reason both ways**: applied because (evidence from this data), or
   skipped because (reason tied to this data, task, compute or constraint).
4. **Correctness is not a hyperparameter.** Duplicate removal, group-aware splitting, decoding and
   orientation fixes, label-format repairs and shortcut removal are decided on evidence, never
   searched over.
5. **Differences within the noise are ties, and ties go to the simpler, faster model.** In vision
   the noise comes mostly from training seeds; compare on the same validation images with paired
   tests (`cv-method-comparison`). REQUIRED BACKGROUND: `ml-experiment-measurement`.
6. **Facts are checked, not recalled.** Model names, versions, licences and leaderboard positions
   change monthly. Confirm against the installed package, the model card or the live leaderboard.
7. **Serving preprocessing equals training preprocessing.** Decoding, orientation, colour mode,
   resize policy and normalisation are one saved object used by both.

## Workspace

| File | Purpose | Created |
|---|---|---|
| `problem_statement.md` | Stage 0 framing | Stage 0 |
| `decision_log.md` | One block per stage, reasons both ways, elapsed minutes | Stage 0, every stage |
| `search_space.json` | Carry-forward candidates per stage for the joint search | Stage 4 onward |
| `report.md` | Skeleton at Stage 0, filled at every gate | Stage 0 |
| `code/NN_stage.py` | One script per stage (several stages may share one in compact mode) | Every stage |

Templates: `references/templates.md`. Task and data-type routing: `references/routing.md`.
Cross-cutting findings that change defaults (CPU precision, export, library traps) and library status,
verified 2026-09-28: `references/library-status.md`.

**Write the report skeleton first.** An interrupted run must still hand over what passed.

**Use the bundled scripts; do not re-implement them.** Each step skill's `scripts/` folder holds
tested tools. Call them from your stage scripts. If you replace one, log which of its checks your
replacement covers.

## Stage map

| # | Stage | Load skill | Gate artifact |
|---|---|---|---|
| 0 | Frame the problem | this skill | `problem_statement.md` with task, metric, operating point, deployment conditions, licence and compute |
| 1 | Integrity, format and statistics audit | `cv-data-audit` (part A) | integrity table, label-format validation, duplicate clusters, dataset statistics |
| 2 | Split design + freeze test | `cv-validation-design` | split spec with rejected alternatives, frozen test IDs with hash, seed noise floor |
| 3 | Label quality, shortcuts, shift (train only) | `cv-data-audit` (part B) | label-issue table, shortcut test, train-vs-deployment shift report |
| 4 | Preprocessing | `cv-preprocessing` | resolution and resize-policy screen (a measured table of at least two input sizes; mandatory for detection when over 10% of boxes are tiny), normalisation, enhancement decisions, saved transform |
| 5 | Augmentation | `cv-augmentation` | preview grid, policy screen, 2-3 carried forward |
| 6 | Baselines + architectures | `cv-model-selection` | baseline table, backbone/head screen, adaptation strategy, licence checks, learnability verdict |
| 7 | Training recipe | `cv-training` | loss/optimiser/schedule screen, training curves, 2-3 recipes carried forward |
| 8 | Post-processing | `cv-postprocessing` | thresholds, NMS/fusion, TTA, calibration screened on validation |
| 9 | Joint search + ensembling | `cv-method-comparison` | search log, winner vs runners-up within noise, ensemble vs best single model, holdout check |
| 10 | Evaluation + sign-off | `cv-evaluation` | test (and deployment-sample) metrics with intervals, slices, robustness, error analysis, explanations, speed |
| 11 | Hand-off | this skill | exported model with its transform, raw-input sanity tests, model card, monitoring plan |

Load each step skill when its stage starts. Do not start stage N+1 until stage N's artifact exists
and its decision-log block is written. **Menu coverage:** before closing a stage, walk the step skill's
decision table (evidence table, invariance table, shortlist) and make sure every row that applies to this
data appears in the block as applied, carried forward, or skipped with a reason. Write the coverage
check itself into the block; "considered everything" without the walk is not a gate. Task-specific material lives in each step skill's
`references/` folder: read only the file for your task.

## Stage 0: Frame the problem

Write `problem_statement.md`; mark unstated items `ASSUMED`.

| Field | What to write |
|---|---|
| Task | One of the task families in `references/routing.md`; name it exactly (for example "instance segmentation", not "segmentation") |
| Unit and label | What one sample is (image, crop, frame, clip, volume, tile) and exactly what is labelled |
| Deployment conditions | Camera or scanner, resolution, lighting, viewpoint, site; whether a labelled sample from deployment exists |
| Decision fed | Who acts on the output; action capacity; cost of each error type |
| Primary metric + operating point | From `references/routing.md`, plus the threshold, recall or false-alarm rate that matters |
| Constraints | Latency, hardware (GPU, CPU, edge, mobile), model size, privacy, regulation, explainability |
| Licence | Commercial use? Default to permissive models; AGPL (for example Ultralytics) or non-commercial weights only with the user's agreement |
| Data location | May data leave the machine (hosted APIs, cloud labelling)? Default: no |
| Compute and time | GPU type and hours available; wall-clock budget; start time from a clock |
| Baseline to beat | Current process, rule, vendor model or human performance |
| Provenance | Every other file delivered with the data, opened and related to the images (README, class lists, masks, trimaps, metadata exports, videos the frames came from). How the images were made (camera, scraped, synthetic, rendered, augmented copies in a dataset export) and how the labels were made (human, model-assisted, auto-generated, converted between formats). Each answer changes the audit: augmented copies need duplicate grouping, converted labels need format checks, auto-labels need a missing-annotation check |
| Success threshold | The metric value that means "good enough" |

**Compute rule.** Estimate GPU or CPU hours before training anything: time one epoch on a sample and
extrapolate. If the right method needs more compute than exists, say so and ask; do not silently
downgrade and do not silently run for days. On CPU only, prefer frozen foundation features with light
heads, small backbones, low resolution and short schedules, and say what a GPU would add.

**Time budget.** Split the budget: about 20% Stages 0-3, 30% Stages 4-8, 30% Stage 9, 20% Stages
10-11. Measure elapsed time with a clock and write it into each decision-log block. When a stage has
measurably overrun, cut breadth (fewer candidates, lower resolution for screening, fewer seeds),
never correctness (integrity audit, duplicate and leakage checks, split, shortcut test, test-once,
sanity tests). **Compact mode** (budgets under about an hour, or CPU only) is planned at Stage 0:
proxies instead of full fine-tunes for screening, two candidates per stage, one seed for screening
and three for finalists. The profiling scripts still run in full.

**Protect the final fit.** When training is the expensive part (CPU, detection, fine-tuning), measure
one epoch first, then reserve at least half of the remaining clock for the final model: the chosen
configuration trained on ALL development images for enough epochs. Size every screen to what is left
(fewer candidates, capped images, one fold). Never shrink the final model's data or epochs to make
room for screening: a thorough process around an under-trained model loses to a plain process around
a well-trained one.

**Unused budget is not a saving.** Before hand-off, check the clock. If time remains, spend it on the
highest-value item that was skipped (a second model family, a resolution screen, LP-FT, the refit on
all development data, a second seed), then finish. Stopping at half the budget with those skipped
leaves accuracy on the table. Throughput numbers come from measuring this machine, not from quoted
figures.

## Carry-forward, joint search and ensembling

Stages 4-8 screen candidates with **cheap proxies** defined in `cv-method-comparison`: a linear or
small head on frozen foundation-model features (minutes, even on CPU), and a short low-resolution
fine-tune of a small backbone. Each stage writes its survivors to `search_space.json` with scores,
spread across seeds and the model families they apply to.

At Stage 9 the joint search tunes the surviving choices together with multi-fidelity pruning, then
the ensembling step tests combinations of the best **diverse** models (different backbones,
resolutions or seeds): probability or logit averaging, weighted averaging, stacking on out-of-fold
predictions, box or mask fusion, test-time augmentation, weight averaging (SWA, model soups). An
ensemble is kept only if it beats the best single model beyond the noise on the same validation
images, and only if its cost fits the latency budget; otherwise distil it into one model.

## Stop signals

- **Too good to be true.** Validation far above the business baseline or published results for
  similar data: suspect near-duplicate leakage or a shortcut first; return to Stages 1-3.
- **Deployment gap.** Strong on the internal test, weak on the deployment sample: a shift problem,
  not a model problem. Go to `cv-data-audit` (shift report), `cv-preprocessing` and `cv-augmentation`.
- **Label ceiling.** Label-issue rate close to the remaining error: fix labels before models.
- **Not learnable yet.** Best model within noise of the baseline and a flat learning curve.
- **Split cannot be made honest.** Groups or duplicates that no split respects with the data available.
- **Budget exhausted.** Finish the current gate, fill the report, list what remains.

## Red flags

| Thought | Reality | Go to |
|---|---|---|
| "Random split is fine, the images are different" | Near-duplicates and same-session photos leak; group and dedupe first | 1, 2 |
| "Accuracy 99%" | Look for a shortcut (watermark, border, scanner) and duplicates before celebrating | 3 |
| "PIL opened it, so it's fine" | EXIF orientation, CMYK, 16-bit and palette images decode differently per library | 1 |
| "Use ImageNet normalisation, it's standard" | Fine for ImageNet-pretrained RGB; wrong for multispectral, X-ray, depth | 4 |
| "Heavier augmentation always helps" | Some transforms change the label (flipped text, laterality); screen and preview | 5 |
| "Biggest foundation model wins" | Check licence, compute, latency and whether a linear probe already saturates | 6 |
| "One training run per config is enough" | Seed variance in vision is often larger than the gains being compared | 9 |
| "Ensemble everything" | Keep it only if it beats the best single model beyond the noise and fits latency | 9 |
| "mAP 0.62, done" | Report per class, per size, per slice, with intervals and on deployment data | 10 |

## Stage 11: Hand-off

- Save the model **with** its preprocessing transform, class map and post-processing parameters.
- Feed raw deployment-format inputs (odd colour modes, EXIF-rotated, wrong sizes) end to end.
- Export (ONNX, TorchScript, TensorRT, OpenVINO, Core ML, TFLite as needed) and confirm the exported
  model matches the original within tolerance on validation images.
- Model card and monitoring plan from `cv-evaluation`.
- Before handing over, re-read every number and claim in `report.md` against the metric files it
  cites, and remove statements the files contradict (for example "the bootstrap did not finish" when its
  output exists).
