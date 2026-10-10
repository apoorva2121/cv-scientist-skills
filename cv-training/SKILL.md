---
name: cv-training
description: Use when training or fine-tuning any vision model (classifier, detector, segmenter, pose, depth, video, anomaly or retrieval model); when choosing between zero-shot, k-NN, linear probe, attentive probe, LP-FT, LoRA/PEFT, full fine-tuning, continued self-supervised pre-training or distillation; when setting learning rate, schedule, epochs, EMA, layer-wise decay, label smoothing, Mixup, freezing or class weighting; when training on CPU or with little GPU; or when a fine-tune overfits, diverges, or fails to beat a frozen-feature baseline.
---

# Vision Training

## Overview

Training has two jobs.

1. **Decide how much of the pretrained model to change.** Changing less is often better: a frozen
   linear probe is strong, and LP-FT (probe first, then fine-tune) beats plain fine-tuning under
   distribution shift. Climb the adaptation ladder only when the rung below is measurably worse.
2. **Run a fixed, logged recipe** that the comparison rules can trust: the Stage 2 folds, fixed
   epochs, the last-epoch score, two or more seeds for finalists, and the config saved with the model.

Evidence, verified APIs and licences for every adaptation method: `references/adaptation.md`.
Optimisers, schedules, regularisation, losses, imbalance, noisy labels, semi-supervised learning,
efficiency and checkpoint selection: `references/training-methods.md`.
Encoder choice and weight licences: `cv-model-selection`.

## Step 1: Measure the compute before training

Time one epoch on a small sample with a clock, then extrapolate to the full plan (candidates x folds x
seeds x epochs). If the right method needs more than the budget or the hardware, say so and ask. Do
not silently downgrade, and do not silently run for days.

- On CPU, frozen features plus scikit-learn heads are the practical path. Measure the throughput on
  this machine before planning: published figures come from other CPUs (timm's table gives about 27
  images per second for ViT-S at 224 px on a desktop i7), while this project's 8-core laptop CPU embedded
  DINOv2 ViT-S at 224 px at about 5-7 images per second with 4 threads (about 25-35 minutes per 10,000
  images). Extract once and cache.
- On CPU, fine-tuning is realistic only for small CNNs (MobileNetV4-S/M, EfficientNet-B0, ResNet-18)
  at low resolution, or with a capped image count for proxies. Say what a GPU would add.
- Measured on this project's CPU (2026-09-28): bf16 autocast made a training step about 36x slower,
  so train in fp32; `channels_last` cut a CNN step by about 20%; `torch.compile` needs a C++ compiler
  on Windows and was unavailable. Measure before assuming a speed-up.

## Step 2: Climb the adaptation ladder

| Rung | What trains | Use when | Tool |
|---|---|---|---|
| Zero-shot (SigLIP 2, PE-Core) | nothing | classes can be named in text; the baseline for everything else | open_clip, transformers pipeline |
| k-NN on frozen features | nothing | tiny data, retrieval, open-set, classes added often | `cv-data-audit/scripts/embed_images.py` + scikit-learn |
| Linear probe | the head | **always run first**, at every data size; on CPU often the final model | screen: `cv-method-comparison/scripts/probe_screen.py`; ship: `scripts/train_probe.py` |
| Attentive probe | head plus pooling queries | masked-image-model or patch-level encoders (MAE, SigLIP, PE, Hiera) | efficient-probing |
| LP-FT | head, then everything at a small LR | about 20 to 1,000 labels per class, and whenever shift is expected | `scripts/train_classifier.py --lp-init` |
| LoRA / PEFT | 0.1-2% of weights | large encoders on limited hardware; many tasks sharing one backbone | peft (`LoraConfig`, module names differ by library) |
| Full fine-tune + layer-wise LR decay + EMA | everything | over about 1,000 labels per class, or a large domain gap | `scripts/train_classifier.py --layer-decay 0.65 --ema 0.999` |
| Continued self-supervised pre-training | everything | tens of thousands of unlabelled in-domain images and a clear domain gap | DINOv2 training code (Apache-2.0), lightly (MIT) |
| Distillation into the deployment model | the student | the winner is too slow for the latency budget | timm `--kd-*` flags |

Rules that come with the ladder:
- **Report the linear probe next to every fine-tune.** `--lp-init` logs it automatically. If the
  fine-tune does not beat the probe on the folds and on any shifted slice, ship the probe.
- For CLIP-like encoders, finish with WiSE-FT (interpolate pretrained and fine-tuned weights) or a
  soup of fine-tunes from the same start: robustness gains at no inference cost.
- Weight licences travel: fine-tunes, and students distilled from a non-commercial teacher, may
  inherit the teacher's terms.

## Step 3: Start from a verified recipe, then screen

**When the probe wins or ties, ship it** instead of a weaker fine-tune:

```bash
python scripts/train_probe.py IMAGE_DIR --meta metadata_grouped.csv --label label --split split/     --model vit_small_patch14_dinov2.lvd142m --img-size 224 [--preprocess prep.py:func] --C 0.1,0.3,1,3
```

It picks C on the folds, reports the holdout, refits on all development images and saves a model that
loads with `weights_only=True`; `load_probe()` gives the model, the exact preprocessing and the classes
for prediction and `cv-evaluation/scripts/export_check.py --module ... --module-arg`. Per-fold
`val_probs.npz` let `ensemble.py` combine it with other candidates.

Classification (`scripts/train_classifier.py`, timm backbones):

```bash
python scripts/train_classifier.py IMAGE_DIR --meta metadata_grouped.csv --label label --split split/ \
    --model convnext_tiny.in12k_ft_in1k --img-size 224 --epochs 15 --lr 2e-4 --wd 0.05 \
    --aug trivialaugment --ema 0.999 --lp-init --fold 0 --seed 0 --out runs/train/lpft/f0_s0
```

- AdamW (no decay on norms and biases), cosine schedule with one warm-up epoch, EMA with warm-up,
  label smoothing 0.1. Keep AdamW (or SGD for CNNs) for fine-tuning: Muon degrades full fine-tuning of
  Adam-pretrained models. For ViTs, layer-wise LR decay 0.65-0.75 (the MAE fine-tuning recipe uses
  base LR 5e-4 x batch/256 and decay 0.65 for ViT-B). These are starting points: the learning rate is
  searched on a log scale in the joint search.
- Class imbalance: compare `--class-weights` with the unweighted loss on macro-F1 and per-class
  recall. Operating-point thresholds in `cv-postprocessing` often fix more than loss weights.
- Mixup and CutMix only with long schedules; see `cv-augmentation`.
- Outputs: `metrics.json` (per-epoch history, probe score), `val_probs.npz` (for ensembling,
  calibration and thresholds) and `model.pt` (weights, classes, config).

Detection (`scripts/train_detector.py`, torchvision detectors, BSD):

```bash
python scripts/train_detector.py IMAGE_DIR --coco annotations.json --split split/ \
    --model fasterrcnn_resnet50_fpn_v2 --size 640 --epochs 24 --aug det_safe --strong-off-last 2 \
    --ema 0.9998 --fold 0 --out runs/det/frcnn/f0_s0
```

After the first fold run writes `val_dets.json`, run the missing-annotation check on it
(`cv-data-audit/scripts/det_missing_annotations.py`, seconds); do not defer it to the end.
It is the licence-safe baseline, not the ceiling. Compare it with the current detectors listed in
`cv-model-selection`, trained on the same split and scored with the same COCO evaluation.

Semantic segmentation (`scripts/train_segmenter.py`, segmentation-models-pytorch, MIT):

```bash
python scripts/train_segmenter.py IMAGE_DIR --masks MASK_DIR --meta metadata_grouped.csv --split split/ \
    --class-map "1:1,2:0,3:255" --arch Unet --encoder resnet34 --size 256 --epochs 20 --aug seg_flip \
    --loss ce_dice --fold 0 --out runs/seg/unet_r34/f0_s0
```

`--class-map` maps raw mask values to training ids, with 255 for ignored pixels; an unmapped value
stops the run instead of becoming a silent class. Metrics are computed at the original mask size.
Per-image intersection and union counts are saved for bootstrap intervals and paired comparisons.

Any training code joins the comparison if it prints one JSON line per epoch with the metric and writes
`metrics.json`, which is the contract `compare.py` and `joint_search.py` read. Instance segmentation,
pose and other tasks follow the same contract.

## Step 4: The short fine-tune proxy

Other stages screen with a cheap version of the real recipe: a small backbone, low resolution
(128-160 px), 5-8 epochs, `--max-train` to cap images, the same folds. Before trusting it, check that
it orders the top two or three candidates the same way as the full recipe
(`cv-method-comparison`).

## Step 5: Test the hand-off path early

As soon as the first model exists (a screening run is enough), load it the way serving will (fresh
process, saved checkpoint, raw image file through the saved transform) and predict on a few images.
Compare with the training script's own predictions for the same images. Serving bugs (a missing
normalisation, a wrong input size, a class-order mismatch) are cheap to fix then and expensive at
hand-off.

## Step 6: Watch the run

- Read the per-epoch lines: training loss falling, validation metric rising then flat. Divergence
  (loss rising or NaN) means the learning rate is too high or the warm-up too short.
- A large gap between training and validation with falling validation means overfitting: use a
  lower rung of the ladder, stronger augmentation, fewer epochs or more data.
- Score the **last epoch of a fixed schedule**. If early stopping is used, its stopping split must be
  carved from the training images, never the fold being reported.
- Fix seeds, and log the config, library versions and hardware with every run.

## Gate (all must exist)

- the compute estimate with the measured epoch time
- the ladder table: rungs tried, fold scores, and the linear-probe baseline next to every fine-tune
- the chosen recipe (config and library versions), the seeds, and last-epoch scores per fold
- per-epoch curves reviewed, with divergence or overfitting explained
- saved validation predictions for Stages 8-9
- decision-log block: applied, carried forward, skipped with reasons (for example "no full
  fine-tune: LP-FT matched it on folds and was 4 points better on the shifted slice")

## Red flags

| Thought | Reality |
|---|---|
| "Fine-tuning everything is always best" | LP-FT beat full fine-tuning by about 10% out of distribution in the LP-FT study; measure it |
| "Small data, so train from scratch" | Pretrained encoders win at small data; start with the probe |
| "Keep the epoch with the best validation score" | That selects on the reported fold; use a fixed schedule and the last epoch |
| "ImageNet mean and std, as usual" | Constants are checkpoint-specific; read them from the checkpoint config |
| "CPU is fine for fine-tuning ViT-B" | Measure an epoch first; cached features are the CPU path |
| "The loss went down, so it works" | Compare with the probe baseline and on the shifted slice |
| "These weights are on Hugging Face, so they are free to use" | Mirror licence tags can be wrong; follow the upstream licence |
