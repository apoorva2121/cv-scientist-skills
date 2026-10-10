---
name: cv-data-audit
description: Use when first receiving an image, video, scan or annotation dataset, before any split or training; when images may be corrupt, truncated, EXIF-rotated, CMYK, 16-bit, grayscale or mislabelled by extension; when checking COCO, YOLO, VOC, mask or keypoint annotations; when hunting exact or near-duplicate images, label errors, outliers or out-of-distribution images; when a vision model may rely on a shortcut such as a watermark, border, ruler, text, scanner or camera; or when training data may differ from deployment data.
---

# Vision Data Audit

## Overview

Two parts, at two moments:
- **Part A (Stage 1), all files:** integrity, format and annotation validity, statistics, and
  duplicate clusters. Runs before the split, because duplicates and sources decide how to split.
- **Part B (Stage 3), training images only:** label quality, shortcuts, outliers, and the shift
  between training and deployment. Runs after the test set is frozen.

The key questions: **does every file decode the way training and serving will decode it**, and
**could anything other than the object itself predict the label?**

All scripts are in `scripts/`; full method inventory with versions, licences and citations
(checked 2026-09-27) in `references/methods.md`.

## Part A: Integrity, format, statistics, duplicates (Stage 1)

**0. Open every other file in the data directory.** List everything delivered with the images and
relate each file to them: extra annotation files (masks, trimaps, keypoints), class lists, metadata
exports, README or licence files, source videos. They often answer questions the images cannot (how
the data was made, which images are augmented copies, which labels are auto-generated) and sometimes
carry the most useful features. Record what each file is and whether it is used.

**Labels in another format?** Class folders, YOLO text files (including Roboflow exports with a
`data.yaml`) and Pascal VOC XML are converted first; the converter counts everything it drops
(images without labels, out-of-range or zero-size boxes, unknown classes) and treats XML as untrusted:

```bash
python scripts/convert_labels.py folders ROOT --out metadata.csv
python scripts/convert_labels.py yolo --images IMAGES --labels LABELS --names data.yaml --out annotations.json
python scripts/convert_labels.py voc --images IMAGES --xml XML_DIR --out annotations.json
```

Split folders shipped with the data (train/valid/test) are recorded, not trusted: exports often put
augmented copies of one photo into different splits.

**1. Decode every file the way the training loader will.**

```bash
python scripts/image_audit.py IMAGE_DIR [--list metadata.csv --col file] --workers 8 --out image_audit.csv
```

It forces a full decode (truncated files fail only at `load()`, not at `open()`), reads the real
format from the bytes, and reports colour mode, bit depth, EXIF orientation, frames, transparency,
size after orientation, brightness, contrast, blur, clipped pixels, near-constant borders, and
exact and perceptual hashes.

| Finding | Action (stateless, logged with counts) |
|---|---|
| Unreadable / truncated | Exclude, or re-export from source; never set `LOAD_TRUNCATED_IMAGES` globally |
| Extension does not match content | Decode by content; fix names at export |
| EXIF orientation != 1 | Apply `ImageOps.exif_transpose` in the **shared loader**; boxes and masks must refer to the same orientation. OpenCV `imread` auto-rotates, Pillow does not: never mix them silently |
| Non-RGB modes (L, RGBA, CMYK, P) | Convert inside the pipeline (alpha composited on a fixed background); keep 16-bit where it carries information |
| Size outliers (under half or over twice the median side) | Inspect; tiny images may be thumbnails or crops, huge ones may need tiling (Stage 4) |
| Near-constant border | Possible frame, scanner edge or letterbox: a shortcut candidate for Part B |
| Medical, geo, video, 3D | Apply the checks in `references/methods.md` section A (DICOM LUTs and PHI, CRS and nodata, frame-by-frame decode, image/mask geometry) |

**2. Validate annotations, then look at them.**

```bash
python scripts/coco_validate.py annotations.json --images IMAGE_DIR --out coco_report.json
python scripts/render_annotations.py annotations.json --images IMAGE_DIR --from-report coco_report.json --out flagged.png
```

The validator checks files, sizes (JSON vs real), categories (undefined, unused, duplicate names),
boxes (zero size, beyond the image, negative origin, area mismatch), corner-format boxes written as
`[x, y, w, h]`, duplicate boxes (IoU >= 0.9 or all four numbers within about a pixel), masks and
statistics (per-class counts, tiny and COCO size shares). Other formats: convert with `supervision`
or Datumaro first (`references/methods.md` A2). **Render about 50 random images per class plus
every flagged one**: coordinate-convention bugs are obvious in a picture and invisible in a table.

**3. Find duplicate clusters before any split.**

```bash
python scripts/find_duplicates.py image_audit.csv --meta metadata.csv --label label --group <source col> --sheet 24
```

Exact duplicates by file hash, near-duplicates by perceptual hash (default Hamming 8 on both phash
and dhash), clusters by connected components. Reports clusters that **span several groups** (a
group split alone leaks them) and clusters with **conflicting labels**. Add `--embed <timm model>`
for crops, colour edits and re-photographed copies that hashes miss. For grouping, err toward
merging; before deleting copies, confirm on the contact sheet.

**Gate A** (all must exist):
- the delivered-files table (every file in the data directory, what it is, used or not)
- `image_audit.csv`, `coco_report.json` (or the format's equivalent) and the rendered overlays
- duplicate clusters with the cross-group and conflicting-label counts
- a cleaning log: every exclusion or repair with its count, and the shared loader that applies it
- group candidates (source, session, patient, video, near-duplicate cluster) handed to Stage 2

## Part B: Labels, shortcuts, outliers, shift (Stage 3, training images only)

**4. Embeddings once, reused everywhere.**

```bash
python scripts/embed_images.py IMAGE_DIR --list train.csv --img-size 224 --out train_emb.npz
```

Default model: a small DINOv2 through timm (Apache-2.0), at 224 px for CPU speed. The same file
feeds label issues, the shortcut probe, OOD, shift and the Stage 6 linear-probe baseline.

**5. Label quality.**

```bash
python scripts/label_issues.py train_emb.npz train.csv --label label --group <group col>
```

Out-of-fold probe probabilities (group-aware, so near-copies cannot vouch for a wrong label), then
confident learning. **Send the top of the list to review**: in the published test-set study only about
half of the flagged items were real errors. Estimated issue rate close to the remaining model error is a
stop signal.

For detection, check for missing and wrong-class boxes once a detector exists (the Stage 6-7 fold
runs provide out-of-fold `val_dets.json` files):

```bash
python scripts/det_missing_annotations.py --gt annotations.json --dets runs/det/<cand>/f*_s0/val_dets.json --images IMAGE_DIR
```

It lists confident predictions that overlap no annotated box ("missing") or only a box of another
class ("class"), with a contact sheet. The check applies to generated and auto-labelled data too:
generators and converters drop objects. Judge its hit rate from the contact sheet before acting on
the list. `cleanlab.object_detection` and
`cleanlab.segmentation` are alternatives (`references/methods.md` C1).

**6. Shortcut screen.**

```bash
python scripts/shortcut_screen.py attributes --audit image_audit.csv --meta train.csv --label label --extra camera_id,site --group <group col>
python scripts/shortcut_screen.py masking --images IMAGE_DIR --meta train.csv --label label --group <group col> --cue-only --per-class 20 --out-dir audit/
```

Run it here, in Stage 3, before any modelling: it is one of the checks that are never cut. `--cue-only`
runs the "cue present" test alone (the part that finds watermarks, stamps and burnt-in text) with
region crops embedded at 112 px; on this project's CPU it took about a minute for a few classes. Drop
`--cue-only` to add the masking half (several times slower) when the budget allows.

Part 1: does any non-content attribute (format, size, file size, EXIF, border, brightness, blur,
camera, site, scanner) predict the label alone? Part 2: blank each corner, the border and the centre
(control) and measure per-class recall. **A class whose recall drops when a small corner is blanked
is recognised by something in that corner.** For segmentation or detection data with masks, also
test foreground-only and background-only variants. The "cue present" part saves a contact sheet of
every flagged region (`--out-dir`): **look at it before deciding**, and decide from what the crops
show, not from a plausible story. A cue in one corner only cannot come from the object's geometry,
which is symmetric; it points to a watermark, stamp, text or frame. Write down, per finding, whether
the cue is legitimate or a shortcut, and how Stage 4-5 remove it (masking, cropping, augmentation that
breaks the correlation, rebalancing, or DFR on a balanced set).

**7. Outliers and OOD.** On the embeddings: L2-normalised Mahalanobis or k-nearest-neighbour
distance (`references/methods.md` D). Review the most unusual images visually; an outlier is not
automatically an error.

**8. Train versus deployment shift.**

```bash
python scripts/shift_report.py --train-emb train_emb.npz --deploy-emb deploy_emb.npz --train-audit train_audit.csv --deploy-audit deploy_audit.csv
```

Image statistics (what changed: darker, blurrier, smaller), adversarial validation AUC on embeddings
(0.5 means indistinguishable) and the share of deployment images unlike anything in training. A
strong shift means internal validation will overstate deployment performance: use it to choose
Stage 4-5 methods, and insist on a labelled deployment sample for the final evaluation.

**Gate B** (all must exist):
- `label_issues.csv` with the reviewed top of the list and the decision taken; for detection,
  `missing_candidates.csv` and its contact sheet (or the reason no detector existed yet)
- for every shortcut-cue flag, its contact sheet and what the crops showed
- shortcut table (attributes and masking) with a verdict per finding
- outlier list reviewed; shift report with its verdict
- decision-log block, including every check skipped and why

## Red flags

| Thought | Reality |
|---|---|
| "It opened fine" | Truncated files open fine; only a full decode proves them |
| "Hashes found no duplicates" | Hashes miss crops and re-photographs; add embeddings for web or scraped data |
| "Sessions are separate, so the split is safe" | Re-uploaded copies carry new session ids; cluster by content too |
| "cleanlab flagged 300 images, drop them" | About half of flags are real errors; review before dropping |
| "The model is 99% accurate on this class" | Mask the corners and borders before believing it |
| "Deployment is the same camera model" | Measure the shift; firmware, lighting and site change the data |
