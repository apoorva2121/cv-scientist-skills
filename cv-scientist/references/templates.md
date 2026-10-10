# Templates

Copy into the project workspace at Stage 0 and replace every `<...>`. Examples use a neutral domain
(crop leaf disease photos) that matches no evaluation set.

## Contents
1. problem_statement.md
2. decision_log.md
3. search_space.json
4. report.md

## 1. problem_statement.md

```markdown
# Problem statement: <project>
Date: <YYYY-MM-DD>   Start time (clock): <HH:MM>   Open ASSUMED lines: <n>

| Field | Value | Source |
|---|---|---|
| Task | <task family from routing.md> | |
| Unit and label | <one sample = ...; label = ...> | |
| Data type | <data type from routing.md>; formats <...> | |
| Deployment conditions | <camera/scanner, resolution, lighting, site>; labelled deployment sample: <yes/no> | |
| Decision fed | <who acts, capacity, cost of errors> | |
| Primary metric + operating point | <metric>; operate at <threshold / recall / false-alarm rate> | |
| Constraints | <latency, hardware, model size, privacy, explainability> | |
| Licence | <commercial?>; AGPL / non-commercial weights allowed: <yes/no> | |
| Data location | <may data leave the machine?> | |
| Compute and time | <GPU/CPU, hours>; wall-clock budget <...>; compact mode <yes/no> | |
| Baseline to beat | <current process / vendor / human> | |
| Provenance | <files delivered and what each is; how images and labels were made> | |
| Success threshold | <number> | |
```

## 2. decision_log.md

```markdown
## Stage 5 - Augmentation - GATE PASSED - elapsed 34 min
Considered: flips, rotation, random resized crop, colour jitter, RandAugment, TrivialAugment,
            Mixup, CutMix, blur/noise/JPEG degradation, Copy-Paste (n/a: classification)
Applied (fixed): no vertical flips (leaves photographed stem-down; orientation carries meaning)
Carried forward: trivialaugment + cutmix (proxy macro-F1 0.842 +/- 0.006, 3 seeds);
                 randaugment(n=2,m=9) + degradation (0.838 +/- 0.007; best on the field-camera slice)
Skipped: AutoAugment - searched on ImageNet, no gain over TrivialAugment beyond noise (0.835 +/- 0.008);
         Mixup alone - below CutMix by 0.011 (paired SE 0.004)
Evidence: code/05_augmentation.py, aug_preview.png, aug_screen.csv
Open risks: field images blurrier than studio images; degradation strength revisited at Stage 9
```

## 3. search_space.json

```json
{
  "task": "single-label classification",
  "primary_metric": "macro_f1",
  "noise_floor_sd": 0.004,
  "noise_floor_definition": "sd of the validation metric across 3 training seeds of the reference proxy",
  "fixed_steps": ["exif_transpose + convert RGB", "drop 41 near-duplicate clusters across groups",
                  "mask 4-px border (scanner frame shortcut)"],
  "stages": {
    "preprocessing": [{"name": "crop384", "args": "--eval-transform crop --img-size 384", "proxy": [0.861, 0.005]},
                      {"name": "letterbox384", "args": "--eval-transform letterbox --img-size 384", "proxy": [0.858, 0.006]}],
    "augmentation": [{"name": "trivialaugment_mix", "args": "--aug trivialaugment --mix 0.5", "proxy": [0.842, 0.006]},
                     {"name": "randaugment", "args": "--aug randaugment", "proxy": [0.838, 0.007]}],
    "models": [{"name": "convnext_small", "args": "--model convnext_small.fb_in22k_ft_in1k", "family": "cnn"},
               {"name": "vit_small_dinov2", "args": "--model vit_small_patch14_dinov2.lvd142m --layer-decay 0.75", "family": "vit"}],
    "training": [{"name": "ema", "args": "--ema 0.999"}, {"name": "plain", "args": ""}],
    "postprocessing": []
  },
  "knobs": {"lr": {"low": 0.0001, "high": 0.003, "log": true},
            "label-smoothing": {"choices": [0.1, 0.0]}}
}
```

Every candidate carries `args`: the training-script arguments it stands for, so
`cv-method-comparison/scripts/joint_search.py` can run any combination. The first candidate of each
stage is the reference; `proxy` is [mean, corrected SE] from that stage's screen. `knobs` are numeric
ranges searched jointly. Post-processing candidates (thresholds, TTA, fusion) are applied after
training and screened on saved predictions, not inside the search.

## 4. report.md

```markdown
# <Project>: vision model report
Status: <gates passed>

## 1. Answer first
<What the model does, test metric with interval, deployment-sample metric with interval, versus
baseline and success threshold.>

## 2. Problem and data
<Framing, ASSUMED lines, data found: integrity, duplicates, label issues, shortcuts, shift.>

## 3. Validation design
<Split scheme and why, rejected schemes, test sizes, noise floor.>

## 4. Decisions
<One line per stage: applied, carried forward, skipped with reasons.>

## 5. Results
| Model / ensemble | Validation (mean +/- sd, seeds) | Holdout | Test (95% CI) | Deployment sample (95% CI) | Latency |
|---|---|---|---|---|---|

## 6. Where it fails
<Per class, per size, per slice, robustness, error analysis with examples.>

## 7. What the model looks at
<Saliency / attention checks, shortcut test results.>

## 8. Limitations, risks, next steps
<Ranked by expected gain: labels, data, resolution, models.>
```
