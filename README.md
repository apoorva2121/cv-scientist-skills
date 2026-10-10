# Computer Vision Scientist skill set

Ten Claude Code skills that take visual data plus a goal to a validated, deployable model through gated
stages: every stage leaves an artefact, screens several methods, carries the best two or three forward,
and the final choice comes from a joint search with honest statistics. Built and tested 2026-09-27/28.

## The skills

| Stage | Skill | What it decides |
|---|---|---|
| 0, 11 | `cv-scientist` | orchestrator: problem framing (incl. provenance, licence, compute), stage gates, time budget, hand-off |
| 1, 3 | `cv-data-audit` | decoding/format, label-format conversion, duplicates and leaks, label errors, missing boxes, shortcut cues, shift |
| 2 | `cv-validation-design` | grouped / duplicate-aware / multi-label / detection / time splits, test size, holdout |
| 4 | `cv-preprocessing` | resolution, crop/letterbox, normalisation, enhancement, serving parity |
| 5 | `cv-augmentation` | invariance table, recipes, previews, screening |
| 6 | `cv-model-selection` | licence gate, learnability baselines, per-task shortlist (16 task/backbone reference files) |
| 7 | `cv-training` | adaptation ladder (probe, LP-FT, LoRA, full fine-tune), recipes, shippable probe, classifier/detector/segmenter templates |
| 8 | `cv-postprocessing` | operating points, calibration, conformal sets, NMS, box fusion, TTA, mask cleanup |
| 9 | `cv-method-comparison` | paired statistics, proxy screens, joint search (Optuna), ensembles, refit |
| 10 | `cv-evaluation` | metrics with intervals, slices, deployment gap, error analysis, explanations, ONNX export and latency |

28 bundled scripts, each tested on real or synthetic data (see `STATUS.md` for every result). Method
inventories with verified versions, licences and citations live in each skill's `references/` (sources:
`research/R1-R5`, checked 2026-09-27/28).

## Install

```bash
python tools/install_skills.py --dry-run
python tools/install_skills.py
```

Copies the ten `cv-*` folders into `~/.claude/skills/`. Rerun after improving a skill; the repository is
the source of truth. The orchestrator also names `ml-experiment-measurement` as background; it is in the
user skills folder already.

## Python environment

```bash
conda create -n cv-skills python=3.12 -y
conda activate cv-skills
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r tools/requirements-cv.txt
```

On a GPU machine use the CUDA index for torch/torchvision. Two traps: keep `albumentations==2.0.8`
(its successor `albumentationsx` is AGPL/commercial under the same import name), and do not install
Ultralytics unless the client accepts AGPL.

## Starting a client project

Give Claude the data location and the goal, for example: "Images of welds are in D:/client/welds with
labels.csv; we need to flag defective welds; commercial use; a laptop CPU; two hours." The orchestrator
asks for anything it cannot infer (operating point, licence, deployment camera, budget) and writes
`problem_statement.md`, `decision_log.md`, `search_space.json` and `report.md` as it goes. Client
deliverables come out of Stage 10-11: metrics with intervals, slices, error analysis, the exported model
with its transform, a load-and-predict script and a model card.

## How well it works (evals, Sonnet agents, CPU only)

| Eval | Without skills | With skills (best iteration) |
|---|---|---|
| Pets, 37 breeds, real images | 5/10 checks, 95.07% | 10/10, 95.15% [94.5, 95.8] |
| Production-line classification with planted defects and camera shift | 9/17, 98.8% | 14/17, 99.8% [99.1, 100] |
| Detection with planted annotation defects | 5/13, AP 0.735 | 10/13, AP 0.731 (tie on common held-out images, iteration 2; iteration 3 skipped the size screen and lost 0.07, now enforced by the validator) |
| Construction-site detection, real images | 6/11, AP 0.325 | 9/11, AP 0.340 [0.298, 0.404]: better, paired diff +0.015 [0.001, 0.027] |

Iteration 3 on production-line classification found and removed the planted watermark shortcut for
the first time but shipped a weaker ResNet-18 (91.5% on the new camera); the causes (DINOv2 stuck at
518 px in the classifier template, no shippable probe) are fixed, and the fixed probe path scores 98.0%
on the same sample with the shortcut removed.

The skills reliably find what plain runs miss (duplicate leaks between pool and benchmark, label errors,
corrupted boxes, format defects, serving bugs) and report honest intervals. Model quality matched the
plain runs once three rules were added: refit on all development data, protect the final training run's
share of the clock, and screen input resolution when objects are small. Benchmarks per iteration:
`cv-workspace/iteration-*/benchmark.md`.

## Limits

- CPU-only tuning: seeds, ensembles and large backbones are expensive; the skills say what a GPU adds.
- Licence data (models, weights, datasets) was verified on 2026-09-27; re-check before shipping.
- `evals/` holds the test data and answer keys; never point a run at it.

### Install on macOS or Linux

```bash
git clone https://github.com/apoorva2121/cv-scientist-skills.git
cd cv-scientist-skills
python tools/install_skills.py        # copies cv-* into ~/.claude/skills/
```

Re-run that script after editing any skill in the clone.

## Licence

[MIT](LICENSE). Fork these and adapt them; no attribution needed in your own
skills. Model weights and datasets referenced by the skills keep their own
licences, which is exactly what the `cv-model-selection` licence gate is for.
