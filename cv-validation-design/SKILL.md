---
name: cv-validation-design
description: Use when splitting images, video, scans or annotated datasets into train, validation and test sets; when choosing between random, grouped, stratified, multi-label, detection-aware, time-based, spatial-block or patient-level splits or cross-validation; when photos repeat per object, session, patient, camera, scene, video or site; when near-duplicates might cross splits; when a vision model validated well but failed in deployment; or when deciding how big a test set must be or whether two models really differ.
---

# Vision Validation Design

## Overview

The split must mirror deployment. In vision, the usual way it fails is not the random split itself
but what random splitting ignores: several photos of the same object, frames of the same video,
slices of the same patient, tiles of the same slide or neighbouring satellite tiles, and copies
re-uploaded under new names. Each of these puts near-identical images on both sides and inflates
every metric.

This stage produces, once, before any model: a **frozen test set** (plus an **external test** when
a whole site or camera can be held out), a **locked holdout** for confirming the Stage 9 search, and
**CV folds** that every later stage screens on. It also measures the **noise floor**.

## 1. Build the split key

```bash
python scripts/build_groups.py metadata.csv --key file --groups capture_session,patient_id --dups duplicate_clusters.csv --out metadata_grouped.csv
```

Rows are joined when they share any group value or a duplicate cluster from `cv-data-audit`. The
output column `split_group` is the only key splitters may use. If one group holds over about 10% of
rows, inspect it before splitting.

| Data type | Group key (never split within) | Stratify on | Scheme |
|---|---|---|---|
| Web / scraped / product photos | source, session, product, near-duplicate cluster | class | `single` |
| Multi-label | as above | label vector | `multilabel` |
| Detection / instance segmentation | video, scene, session, near-duplicate cluster | classes present + count bin + size bin | `detection` |
| Semantic segmentation / depth | scene, sequence, sensor rig | class-pixel presence | `multilabel` on presence tags |
| Pose / keypoints | subject, session | pose or viewpoint bin | `single` on the bin |
| Video | source video, clip group, recording session | action class | `single` |
| OCR / documents | document, template, scanner | document type, script | `single` |
| Medical 2D / 3D, pathology | **patient** (and site) | diagnosis | `single`, plus `--external site=...` |
| Remote sensing | spatial block at least the autocorrelation range, region, date | land-cover class | spatial blocks as groups (verde `BlockKFold`, torchgeo grid cells; `references/methods.md`) |
| Anything drifting over time | time period | - | `time` |

## 2. Split and freeze

```bash
python scripts/make_split.py metadata_grouped.csv --key file --group split_group --scheme single --label label \
    [--coco annotations.json] [--time date_col] [--external site=SITE_B] --test-size 0.2 --holdout-size 0.1 --n-splits 5 --out split/
```

- Groups never straddle parts or folds; the spec records the overlap check (must be 0).
- Multi-label and detection use iterative stratification over group-aggregated tag vectors.
- With few groups a single draw can be badly unbalanced, so the script tries `--tries` seeds (default
  20) and keeps the most label-balanced partition. This looks only at labels, never at a model.
- `--external COL=VALUE` holds out a whole site, camera or scanner: the most honest estimate of
  performance on a new site. Scored once, next to the internal test.
- The test and external IDs are hashed. Before Stage 10 scores them, `verify_untouched("split")` must
  return True.

Load folds everywhere with `from cv_folds import load_split` (`scripts/cv_folds.py`).

**When a labelled deployment sample exists** (a new camera, a new hospital), it is a test set of its
own: never train or tune on it, never pool it with the internal test; report both.

**When the user supplies a fixed evaluation set** (a "benchmark" to be used once), that set is the
test. Do not carve a second internal test from the development pool: it only removes training data.
Split the pool into CV folds plus, if needed, a small holdout for confirming the final choice:
`make_split.py ... --test-size 0 --holdout-size 0.1`. Parts are then `train` (with folds) and
`holdout`, which is what `--eval-part holdout` in the training templates reads.

## 3. Post-split leak audit

- Re-run `find_duplicates.py` with the split: count duplicate pairs across parts (target 0 apart from
  what hashing cannot see); for web or scraped data add the embedding mode.
- For foundation-model backbones, check the test set against any accessible pretraining data:
  benchmark images often appear in web-scale pretraining (`references/methods.md` F).

## 4. How big must the test set be?

For accuracy-like metrics, the 95% Wilson half-width at 90% accuracy is roughly: 100 images, plus
or minus 6.0 points; 300, 3.4; 1,000, 1.9; 3,000, 1.1; 10,000, 0.6. For mAP, mIoU and Dice, bootstrap
over images (or over groups when images are grouped) and recompute the dataset metric. If the
interval is wider than the difference that matters, enlarge the test set or report cross-validated
estimates alongside it.

## 5. Cross-validation or holdout?

Deep models make k-fold expensive. Rules:
- **Small data** (a few hundred to a few thousand images): repeated grouped k-fold for screening
  with cheap proxies (frozen features, short fine-tunes); a single split is noise at this size.
- **Medium and large data**: one grouped train/validation split for full training, the fixed folds
  for proxy screening, the holdout for confirmation.
- Never tune on the test set; the holdout exists because the search overfits validation.

## 6. Noise floor

Train the Stage 6 reference proxy with at least 3 seeds on the same split and record the standard
deviation of the validation metric as `noise_floor_sd` in `search_space.json`, **in the primary
metric's units**. In vision, seed variance often exceeds the gains being compared. Comparisons of two
methods on the same validation images use paired tests (`cv-method-comparison`).

## Gate (all must exist)

- `metadata_grouped.csv` with `split_group`, and the group-key reasoning per data type
- `split/split.csv` and `split/split_spec.json`: zero group overlap, tag balance per part, hashes
- the scheme chosen, at least one rejected scheme and why, and the external or deployment test plan
- post-split duplicate check; test-size precision estimate; noise floor

## Red flags

| Thought | Reality |
|---|---|
| "Random 80/20 like everyone does" | Photos of one object, frames of one video, slices of one patient then sit on both sides |
| "Group by session and we're safe" | Re-uploaded copies carry new session ids; add duplicate clusters to the key |
| "The holdout is only 100 images" | Its interval is about plus or minus 6 points; do not choose models on it |
| "Tiles are independent" | Neighbouring tiles and tiles of one slide are not; block spatially or by slide |
| "Add the deployment sample to training" | Then nothing honest remains to measure deployment performance |
