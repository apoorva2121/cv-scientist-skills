---
name: cv-method-comparison
description: Use when deciding whether one vision method, model, preprocessing, augmentation, loss or training recipe really beats another; when screening many candidates cheaply with linear probes on frozen features or short fine-tunes; when tuning several choices together (joint search, Optuna, Hyperband, successive halving, early stopping of weak trials); when testing ensembles, weighted averaging, Caruana selection, stacking, weight averaging or distillation; or when a validation gain might be noise, seed luck or selection bias.
---

# Vision Method Comparison

## Overview

Every "A is better than B" in the project is decided here, with the same rules:

1. **Same images, same pipeline.** All candidates use the fixed Stage 2 folds; nothing is compared across
   different validation sets. When a correctness change alters the input (a shortcut crop, a new
   decoder, cleaned labels), re-score every surviving candidate on the changed pipeline before choosing:
   a number measured on the old pipeline is not a candidate score any more.
2. **Paired, corrected statistics.** Compare fold by fold. Seeds are averaged within a fold (seeds
   reduce training noise; folds are the independent evidence). Errors use the Nadeau-Bengio correction
   because cross-validation training sets overlap.
3. **Keep rule.** A candidate is kept if it is not worse than the best by more than one corrected
   standard error. Among kept candidates, prefer the simpler or cheaper one. A tie needs a measured
   error: with one fold and one seed there is no error estimate, so "within noise" cannot be claimed;
   prefer the better-scoring candidate or measure more.
4. **Equal training.** Candidates get the same epochs (or full learning curves), not the same
   wall-clock minutes; a model that saw fewer epochs has not lost a comparison, it has not finished it.
5. **Honest scoring.** Score the last epoch of a fixed schedule, not the best epoch on the fold being
   scored. The holdout confirms the final choice once. The test set is scored once, in Stage 10.
6. **Refit before the test.** After the holdout confirms the choice, retrain the chosen configuration
   (same recipe and epochs) on ALL development images (train folds + validation + holdout), then score
   the test once. A final model trained on part of the pool throws away data the test result depends on.
7. **Noise floor.** The standard deviation of the reference proxy across three seeds, in metric units.
   It sizes gains from single, unpaired runs; paired fold comparisons use the corrected error instead.

Evidence for search, ensembling and distillation: `references/ensembling-and-search.md`.

## The proxy ladder

| Proxy | Cost | Ranks well | Cannot rank |
|---|---|---|---|
| Linear probe on frozen foundation features (`scripts/probe_screen.py`) | minutes, CPU is fine | resolution, crop, colour handling, enhancement, which frozen backbone sees the task best | augmentation, loss, optimiser, anything learned during fine-tuning |
| Short fine-tune (`cv-training`, few epochs, small backbone, capped images) | tens of minutes to hours | augmentation, loss, class weighting, EMA, learning rate range, freezing | effects that only appear in long schedules (heavy Mixup, from-scratch recipes) |
| Joint search (`scripts/joint_search.py`) | hours | interactions between the survivors of every stage | nothing new: it only combines what the stages carried forward |

**Check the proxy once.** Before a proxy ranks many candidates, run the top two or three at the full
budget. If the full-budget order disagrees with the proxy, the proxy is misleading for this data:
raise its budget (more epochs, higher resolution) until the order holds.

## Step 1: Screen within a stage

```bash
# preprocessing and backbone features: frozen-feature probe
python scripts/probe_screen.py IMAGE_DIR --meta metadata_grouped.csv --label label --split split/ \
    --variants crop224,letterbox224,resize224 [--custom my_variants.py] [--C 0.3,1,3] [--max-train 1500] [--threads 4]

# augmentation, loss, training knobs: short fine-tunes, one run per fold and seed
#   layout: runs/<stage>/<candidate>/f<fold>_s<seed>/metrics.json
python scripts/compare.py runs/aug --metric macro_f1
```

Both print the same table: mean, fold spread, seed spread, difference from the best, corrected
error, and keep. Fewer than five folds gives a warning; treat decisions as provisional or add seeds.
Write the kept candidates, each with its training `args`, into `search_space.json`.

## Step 2: Joint search over the survivors

```bash
python scripts/joint_search.py --space search_space.json --folds 0,1,2 --epochs 8 --trials 30 \
    --timeout-min 120 --metric macro_f1 --out runs/joint -- \
    python ../cv-training/scripts/train_classifier.py IMAGE_DIR --meta metadata_grouped.csv --label label --split split/
```

- Trial 0 is the reference combination, so the search reports what it gained over the reference.
- Optuna TPE picks combinations. On the first fold, a Hyperband pruner stops weak trials after one or
  a few epochs. Surviving trials run the remaining folds.
- The end table re-compares the top trials, paired over folds, and recommends the cheapest trial
  among those kept.
- Any training script works if it prints one JSON line per epoch with the metric and writes
  `metrics.json` (detection and segmentation templates follow the same contract).
- Set `--timeout-min` from the Stage 9 share of the time budget, measured by the clock. In compact
  mode, use two candidates per stage, three folds and one seed, and say so in the decision log.

Overtuning is common (in about 10% of tuning runs the chosen configuration generalises worse than
the default), so the reference trial stays in every table. Then confirm the recommended combination
against the reference once on the holdout
(`--eval-part holdout`). Choosing among many trials on the same folds flatters the winner. The holdout
gives the unflattered number. Finally refit the confirmed configuration on all development images
(rule 6) and use that model for Stage 10.

## Step 3: Ensembles

```bash
python scripts/ensemble.py runs/joint/trial_012 runs/joint/trial_007 runs/joint/trial_003 \
    --meta metadata_grouped.csv --group split_group --metric macro_f1
```

- Give candidate folders: their folds are joined into out-of-fold predictions over the whole train
  part. Diverse members (different backbones, resolutions, augmentation) matter more than more
  members of the same kind.
- Methods: mean, Caruana greedy selection, fitted weights, logistic stacking, and a recalibrated best
  single model. Everything fitted is cross-fitted with group-aware folds. Every method is compared
  with the best single model by a paired bootstrap over whole groups.
- **Stacking must beat the recalibrated single model, not just the raw one.** A stacker also fixes one
  model's class bias and calibration, which the single model can get from post-processing at no extra
  inference cost.
- Keep an ensemble only if its interval against the best single model excludes zero **and** its
  inference cost fits the latency budget. If it wins but is too slow, distil it into one model and
  compare the student with the same test.
- Weight-space averaging (SWA, EMA, model soups of fine-tunes from one initialisation) adds no
  inference cost. Use a greedy soup (add a run only if validation improves), not a uniform one: in
  the soups paper the uniform soup fell below the best single model while the greedy soup beat it.
- Members fine-tuned from one checkpoint differ little; different pre-training or methods give the
  diverse errors that make ensembles pay.
- Detection ensembles fuse boxes (weighted boxes fusion); segmentation ensembles average per-pixel
  probabilities. The same paired rule applies, with images as the bootstrap unit.

## Gate (all must exist)

- one screening table per stage (Step 1), with the kept candidates and their `args` in `search_space.json`
- the proxy check: full-budget order of the top candidates versus the proxy order
- `runs/joint/trials.csv` and `top_trials.csv`; the recommended combination versus the reference
- the holdout confirmation of the recommendation, run once, and the refit on all development images
- the ensemble report, including stacking versus the recalibrated single model; kept or not, and why
- the decision-log block with elapsed clock time

## Red flags

| Thought | Reality |
|---|---|
| "B beat A by 0.4 points on one run" | Inside the seed noise until paired fold statistics say otherwise |
| "Take the best epoch of each run" | Selecting the epoch on the scored fold flatters every run; score the last epoch |
| "The joint search winner's CV score is the expected performance" | Selected among many trials; the holdout gives the honest number |
| "Stacking gained 5 points" | Check against the recalibrated single model; the gain may be calibration, not combining |
| "Five seeds of one model make a strong ensemble" | Diversity matters; different backbones or resolutions usually add more |
| "The ensemble wins, ship it" | Only if the latency budget allows; otherwise distil |
| "Different folds for the new candidate, it's fine" | Unpaired comparisons across different images are not evidence |
