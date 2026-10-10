---
name: cv-preprocessing
description: Use when deciding how images, video frames, scans or multispectral tiles are decoded, resized, cropped, padded, tiled, normalised or enhanced before a vision model; when choosing input resolution, letterbox versus crop versus stretch, mean and std, colour space, CLAHE, gamma, stain normalisation, DICOM windowing, voxel spacing, band scaling, denoising, deblurring, low-light enhancement or super-resolution; when objects are small in high-resolution images; or when a model behaves differently in serving than in validation.
---

# Vision Preprocessing

## Overview

Preprocessing is whatever turns a stored file into the tensor the model sees. Two rules dominate:

1. **Serving runs exactly the evaluation path**: same decoder, orientation handling, colour
   conversion, resize library, interpolation, antialias flag, crop or padding, and normalisation.
   A different resize library alone is a distribution shift. Save the transform with the model and
   test parity (`scripts/serve_parity.py`).
2. **Copy the pretrained checkpoint's own preprocessing first**, then change one thing at a time with
   evidence. Exception: a detector's training scale (for example the 320 px of torchvision's `*_320_*`
   models) is a starting point, not a decision. When objects are small, the input size is screened
   (Step 1), and the checkpoint's own scale is only one of the candidates. Normalisation constants are checkpoint-specific (ImageNet, OpenAI CLIP, SigLIP 0.5/0.5,
   satellite-specific DINOv3); read them from the checkpoint config (`timm.data.resolve_model_data_config`,
   `transformers.AutoImageProcessor`), never from memory.

Full inventory with versions, licences and citations (checked 2026-09-27): `references/methods.md`.

## Step 1: What does the data need? (from Stage 1 and 3 evidence)

Read `image_audit.csv` and the shift report from `cv-data-audit`:

| Evidence | Candidate preprocessing |
|---|---|
| Mixed aspect ratios | letterbox (pad, keeps shape) vs centre crop vs stretch: screen all three |
| Objects small relative to the image | **screen at least two input sizes, mandatory when the audit reports many tiny objects** (e.g. over 10% under 12 px): for images smaller than the detector's usual input this means UPSCALING (a detector input larger than the image, set with `--size`); for large images, higher resolution or **tiling** for training and sliced inference (SAHI or supervision `InferenceSlicer`, both MIT); oversample images with small objects. Upscaling and tiling are different levers: do not rule out one by arguing against the other |
| Wide or tall inputs (documents, panoramas) | aspect-ratio bucketing |
| Physical-unit intensities (CT, thermal, reflectance, depth) | **fixed** scaling (HU window, sensor range); never per-image min-max |
| Multispectral / non-RGB bands | per-band mean and std from training images after nodata masking; sensor offsets (e.g. Sentinel-2 processing baseline) |
| 3D medical | resample to common voxel spacing, canonical orientation (nnU-Net conventions; masks nearest-neighbour) |
| Pathology slides | patches at a fixed microns-per-pixel; stain **augmentation** first, normalisation second |
| Low contrast, dark, hazy, blurry, noisy | first: train on matching degradations (`cv-augmentation`). Restoration front ends only if a held-out real test shows a gain |
| Genuinely low-resolution inputs | super-resolution is a candidate; it helps only when the input truly lacks resolution |
| Wide-gamut or scanned colour where colour matters | convert the ICC profile to sRGB (Pillow does not apply it on its own) |

## Step 2: Screen candidates with the probe proxy

```bash
python ../cv-method-comparison/scripts/probe_screen.py IMAGE_DIR --meta metadata_grouped.csv --label label \
    --split split/ --variants crop224,letterbox224,resize224,crop320,clahe224 [--custom my_variants.py]
```

A linear probe on frozen foundation-model features, on the fixed Stage 2 folds, compared with paired,
overlap-corrected statistics. It ranks what changes what the backbone can see: resolution, crop
policy, colour handling, enhancement. Custom variants (deskew, stain normalisation, band selection,
tiling) go in `--custom`. Carry the best two or three into `search_space.json`. Effects that only
appear under fine-tuning (for example the FixRes train-low/test-high trick) are screened with the
short fine-tune proxy in `cv-training`.

**Resolution is usually the biggest preprocessing lever** and the biggest cost lever: screen at least
two resolutions, and note that attention cost grows with the square of the token count.

## Step 3: Decide, save, and prove parity

- Write the chosen transform as one function or `torchvision.transforms.v2` pipeline, saved with the
  model config: decode, orientation, colour mode, resize (library + interpolation + antialias), crop or
  pad, normalisation.
- Run `scripts/serve_parity.py` against the serving implementation (for example the ONNX or
  production loader) on 50 raw files: maximum absolute pixel difference after normalisation must be
  within tolerance.
- Anything fitted (per-band statistics, stain targets, CLAHE parameters chosen on data) is fitted on
  training images only.

## Gate (all must exist)

- evidence table (Step 1) linking each candidate to a finding
- probe screen table; the chosen resolution and crop policy with the runner-up
- the saved transform and a passing parity check
- decision-log block: applied, carried forward, skipped with reasons (for example "no dehazing: no
  held-out gain; train on hazy augmentations instead")

## Red flags

| Thought | Reality |
|---|---|
| "ImageNet mean/std, as always" | SigLIP uses 0.5/0.5, CLIP its own, satellite weights their own; read the config |
| "Resize is resize" | Library, interpolation and antialias differences shift features; one path end to end |
| "Enhance the images first, then classify" | Dehazing and enhancement often do not help and sometimes hurt; train on degradations |
| "Downscale the 4000 px image to 640" | Small objects vanish; tile at the scale you infer at |
| "Normalise each image to 0-1" | Per-image min-max destroys physical meaning (CT, thermal, reflectance) |
| "Bilinear resize for the masks too" | Label maps are resized nearest-neighbour, padded with the ignore index |
