---
name: cv-augmentation
description: Use when choosing data augmentation for any vision model (classification, multi-label, detection, segmentation, pose, depth, video, OCR, medical, pathology, remote sensing, thermal); when deciding on flips, crops, colour jitter, RandAugment, TrivialAugment, AugMix, Mixup, CutMix, Mosaic, Copy-Paste, large-scale jitter, stain augmentation, blur/noise/JPEG/weather simulation, synthetic or generated training images, or test-time augmentation; when a model overfits a small dataset; when boxes, masks or keypoints are wrong after augmentation; or when a model must survive a new camera, lighting, site or weather.
---

# Vision Augmentation

## Overview

Every augmentation is a claim: "the label does not change when this changes". A true claim is free
data. A false one deletes the signal the label depends on: flipping a left-turn sign, hue-jittering a
ripeness grader, channel-swapping a colour-coded part, shrinking a small part to three pixels. So the policy
comes from two facts, not from habit:

1. **What really varies at deployment** (evidence: `cv-data-audit` shift report, deployment notes).
2. **What the label depends on** (evidence: the problem statement and the Stage 3 shortcut tests).

**Library policy (checked 2026-09-27):** `torchvision.transforms.v2` (BSD; boxes, rotated boxes,
masks, keypoints, video) by default; Kornia for GPU batches; timm policies for classification;
MONAI or TorchIO for medical volumes; TorchGeo for remote sensing. Keep `albumentations==2.0.8`
(MIT, archived) only for existing pipelines. Do not install `albumentationsx` (AGPL or commercial,
telemetry, same import name) or copy Ultralytics code unless Stage 0 allowed AGPL.

Full inventory, recipes read from official configs, and citations: `references/methods.md`.

## Step 1: Write the invariance table before choosing any operation

| Evidence | Candidate augmentation |
|---|---|
| Shift report: brightness, contrast or colour differ between sources | brightness, contrast, gamma, white balance, mild colour jitter |
| Shift report: sharpness drops, compression, small sensor | blur, noise, JPEG, downscale-then-upscale (degradation pipeline) |
| New camera, scanner, site or stain protocol | photometric jitter plus noise; stain (HED/HSV) augmentation for pathology; and an external-site test |
| Objects appear at many scales | scale jitter, large-scale jitter (LSJ), multi-scale training |
| No natural "up" (aerial, microscopy, pathology) | D4: 90-degree rotations plus flips |
| Occlusion expected | random erasing, CutOut, Copy-Paste occluders |
| Weather, night, glare in deployment | fog, rain, low-light simulation, glare; evaluate on real bad-weather images |
| Stage 3 found a shortcut cue (watermark, border, burnt-in text, ruler) | remove or randomise the cue: crop or mask the region, or paste the cue onto all classes; then re-run the shortcut test |
| Rare class | Copy-Paste of real instances (instance tasks) plus class-aware sampling in `cv-training`; augmentation alone does not fix imbalance |

| The label depends on | Forbidden or restricted |
|---|---|
| Colour (ripeness, stain intensity, colour-coded parts or uniforms, traffic lights, flags, coat colour) | no hue jitter, channel permutation, solarize, invert, grayscale; mild brightness only |
| Left versus right, reading order, digits (6/9), laterality | no horizontal flip, or flip with an explicit label swap |
| Absolute size, count or distance | scale jitter only within the deployment range; adjust depth or intrinsics with resizing |
| Small objects | no crops or zoom-out that shrink objects below about 8 px; check the kept-box rate |
| Physical intensities (CT HU, thermal, reflectance) | intensity changes in physical units; no per-image contrast stretch |
| Fine texture (defects, pathology, fine-grained species) | no heavy blur or strong downscale beyond what deployment produces |

## Step 2: Start from the published recipe for the model family

Copy the recipe from `references/methods.md` section D, then change one thing at a time.

- **Short fine-tune of a pretrained backbone** (under about 30 epochs): RandomResizedCrop (raise the
  minimum scale when objects are small) plus flip plus TrivialAugment or light RandAugment. Heavy
  regularisation hurts short runs (AugReg). Mixup and CutMix only if the proxy shows a gain.
- **Training from scratch or long schedules:** DeiT III 3-Augment or the DeiT and ConvNeXt
  RandAugment recipe, with Mixup and CutMix.
- **Detection:** the detector's own recipe (photometric distortion, zoom-out, IoU crop, flip,
  multi-scale; Mosaic and MixUp where the recipe uses them). Switch Mosaic and other strong
  operations off for the last 8 to 15 epochs.
- **Segmentation:** LSJ plus fixed-size crop (COCO-style) or multi-scale plus crop plus
  photometric distortion (ADE20K-style).
- **Medical volumes:** nnU-Net's own augmentation, with mirroring off when laterality matters.

## Step 3: Preview before any training

```bash
python scripts/aug_preview.py IMAGE_DIR --list metadata.csv --col file --policy trivialaugment --size 224
python scripts/aug_preview.py IMAGE_DIR --coco annotations.json --policy det_ssd --size 640 --n 30
python scripts/aug_preview.py IMAGE_DIR --coco annotations.json --custom my_policy.py
```

A custom policy file defines `POLICY(size)` returning a `v2.Compose`. Look at the grid, and save it in
the workspace. Check that boxes, masks and keypoints sit on the objects. Check that the label is still
true in every copy, and that small objects are still readable. For detection the script also prints
the kept-box rate per class and the share of kept boxes under 8 px. Crop policies drop boxes outside
the crop by design. Act when a rare class loses clearly more than the others, or when zoom-out leaves
many tiny boxes: cap `side_range`, raise the minimum crop scale, or switch to large-scale jitter.
Built-in policies are named in the script header. For masks, keypoints and video, pass them through
the same `v2` call in a custom policy and draw them.

## Step 4: Screen candidates with the short fine-tune proxy

A frozen-feature probe cannot rank augmentations, because nothing is trained on the augmented
images. Use the short fine-tune proxy from `cv-training` on the fixed Stage 2 folds, with the
Stage 4 preprocessing held fixed:

```bash
# one directory per candidate, one run per fold and seed
python ../cv-training/scripts/train_classifier.py IMAGE_DIR --meta metadata_grouped.csv --label label \
    --split split/ --fold F --seed S --aug trivialaugment --epochs 8 --out runs/aug/trivialaugment/fF_sS
python ../cv-method-comparison/scripts/compare.py runs/aug --metric macro_f1
```

- Compare `none`, the recipe baseline and two or three stronger options. Add `--mix 0.5` as a
  separate candidate, not stacked onto another change.
- Use at least two seeds. The comparison script scores the last epoch, averages seeds within a fold,
  and pairs candidates by fold with overlap-corrected errors.
- Validation images are never augmented. If a labelled shift or deployment sample exists, score it
  as well. The benefit of augmentation often appears only under shift, so report both scores.
- Detection and segmentation use the same run layout with their own training template and metric
  (mAP, mIoU).
- If the screen will not fit the time budget, reduce it in this order: fewer folds (three), a
  training-image cap (`--max-train`), smaller images, fewer candidates. Say which, and tell the user
  when the honest screen needs a GPU.

Carry the best two or three into `search_space.json` with their proxy scores. Augmentation strength
interacts with model size, schedule length and Mixup, so the final choice is made in the joint search.

## Rules that are never cut

- **Boxes:** clamp and sanitise after every geometric operation; declare the box format explicitly;
  rotated objects need oriented-box operations.
- **Masks:** nearest-neighbour interpolation, padding with the ignore index, the same parameters as
  the image.
- **Keypoints:** flips must remap left and right identities with a flip-pair list (torchvision and
  Kornia do not); update visibility after crops.
- **Video:** one parameter set for all frames of a clip.
- **Stereo and depth:** flipping swaps the views; resizing changes metric depth or intrinsics.
- **Multi-label:** crops can remove a labelled object; use a high minimum scale or update the labels.
- **Documents and text:** no flips, no large rotations, no crops through text.
- **Anything fitted from data** (stain targets, band statistics, copy-paste banks) comes from
  training images only.

## Synthetic and generated images

- Training only. Validation and test stay real, from the deployment distribution.
- Mix with real data, never replace it. Report the score with and without synthetic data.
- Gains are largest when real data is scarce or the target condition is rare.
- Check generated images for near-duplicates of test images (`find_duplicates.py`), and check the
  generator's licence.

## Test-time augmentation

TTA belongs to `cv-postprocessing`. It is used only after a measured gain on validation data, and
only with operations the model was trained with. It is not a default: standard TTA hurt most
pairs in a 2026 medical study.

## Gate (all must exist)

- the invariance table, each row linked to an audit finding or to the problem statement
- the saved preview grid, plus the per-class kept-box rate and tiny-box share for detection
- the screening table from `compare.py` (two or more seeds), with the noise floor
- the chosen policy and runners-up in `search_space.json`; the exact config and library versions logged
- decision-log block: applied, carried forward, and skipped with reasons (for example "no hue jitter:
  colour is the label")

## Red flags

| Thought | Reality |
|---|---|
| "More augmentation is always better" | Heavy policies hurt short fine-tunes of large pretrained models; screen it |
| "RandAugment everywhere" | Some of its operations (solarize, invert, colour) change colour-defined labels |
| "The SSD-style photometric distortion is standard" | torchvision's `RandomPhotometricDistort` permutes channels with no separate switch; use `ColorJitter` (policy `det_safe`) when colour is a cue |
| "Flip keypoints like the image" | Left wrist becomes right wrist unless flip pairs are remapped |
| "Augmentation will fix the shortcut" | Only if it removes the cue; re-run the shortcut test to prove it |
| "Augmentation fixes class imbalance" | It adds variety, not frequency; sampling and loss weighting live in `cv-training` |
| "Validate on augmented images too" | Validation and test are real, unaugmented images |
| "Generated images can fill the test set" | Synthetic images never enter validation or test |
| "Mosaic until the end" | Detectors switch it off for the last 8 to 15 epochs |
