# Status: Computer Vision Scientist Skill Set

Last updated: 2026-09-27. Paused by Apoorva ("another operation running, we will run later").
Working rule (updated on resume): agents run strictly one at a time, in a queue, never in parallel.

## Added requirements on resume (2026-09-27)

- World-class results for any vision problem and any vision data. Scope widened: OCR and document
  images, 3D volumetric (CT/MRI), remote sensing and multispectral, vision-language tasks, and point
  clouds at reference level. Image generation stays out (not a predictive task).
- Best-practice data splitting for every data type.
- Ensembling and stacking as a first-class stage, tested in the evals: probability/logit averaging,
  stacking on out-of-fold predictions, blending on a holdout, weighted box fusion, mask averaging,
  TTA, snapshot ensembles, SWA, model soups, and distilling an ensemble into one model.
- Best achievable performance metrics for the task, with honest intervals.

## Decisions (agreed in chat, 2026-09-27)

1. Same architecture as the tabular set (`C:\dev\important repo\Data Scientist Skill Set`): one
   orchestrator with evidence gates, step skills that screen many methods, carry the best 2-3
   forward, and a joint search that picks the best combination.
2. Skills: `cv-scientist` (orchestrator), `cv-data-audit`, `cv-validation-design`,
   `cv-preprocessing`, `cv-augmentation`, `cv-model-selection`, `cv-training`,
   `cv-postprocessing`, `cv-method-comparison`, `cv-evaluation`. Per-task reference files inside
   each step skill.
3. Scope: every task family - classification (single, multi-label, fine-grained), detection
   (incl. small and rotated objects), semantic / instance / panoptic segmentation, pose, depth,
   video (action recognition, tracking), anomaly detection, retrieval. Out of scope: OCR, 3D,
   image generation. Include every basic, advanced and latest technique per step.
4. Licence default: permissively licensed models; ask at Stage 0 before AGPL (e.g. Ultralytics)
   or non-commercial weights.
5. Tests: synthetic datasets generated in code with planted defects (no dataset downloads);
   classification + detection (+ segmentation if time). Small images, CPU training.
6. Carry over the tabular lessons: never quote eval data in skills; clock-measured time budgets;
   compact mode; bundled scripts by default; never-cut correctness checks.

## Environment (done)

| Item | State |
|---|---|
| Machine | No CUDA GPU (AMD Radeon 680M integrated), 8-core Ryzen 7 7735HS, 27 GB RAM, 117 GB free on C: |
| Base Anaconda torch | Broken: crashes on import (duplicate libiomp5md.dll). Left untouched |
| New env | conda env `cv-skills` (conda-forge, Python 3.12) at `C:\Users\apoor\anaconda3\envs\cv-skills` |
| Python to use | `C:\Users\apoor\anaconda3\envs\cv-skills\python.exe` |
| PyTorch | torch 2.14.0+cpu, torchvision 0.29.0+cpu from the PyTorch CPU index; import and matmul verified |
| Vision libraries | pip install finished with exit 0: timm, albumentations, kornia, opencv-python-headless, pillow, imagehash, cleanlab, pycocotools, onnx, onnxruntime, scikit-learn, scikit-image, pandas, matplotlib, optuna, transformers, open-clip-torch, segmentation-models-pytorch, grad-cam, torchmetrics, ensemble-boxes, iterative-stratification, huggingface_hub, safetensors. Import check NOT yet run |
| Not installed on purpose | ultralytics (AGPL, licence default) |

## Session log 2026-09-27 (evening)

- R3 research agent had spawned 2 sub-agents of its own (3 in parallel). Both stopped; R3 told to cover
  domain foundation models and adaptation strategies itself. Future agent prompts must forbid sub-agents.
- Shortcut test: 'cue present' probe flags class 'star' from bottom_right alone (recall 0.83, precision
  0.78, chance 0.17) while the 'cue used' masking test shows no drop: both tests are needed.
- train_classifier.py: fixed `timm.optim` import (submodule not auto-imported) and loss.item(); smoke
  test (2 candidates x 2 folds, 2 epochs, 128 images) passes incl. EMA, Mixup, layer decay, freezing.
  Quality test (full data, 8 epochs) still to run when the CPU is free.
- compare.py: new command line `compare.py ROOT` reads ROOT/<candidate>/<run>/metrics.json; scores the
  LAST epoch by default (best-epoch selection on the scored fold is optimistic); EMA runs scored with EMA.
- ensemble.py: fixed name collision (runs in same-named folders overwrote each other); added `recal`
  (logistic recalibration of the best single model) as the baseline stacking must beat, group-aware
  cross-fitting (--meta/--group) and group bootstrap. On near-chance models recal explained 0.39 of
  stacking's 0.42; on 3 good feature models (acc ~0.92) no ensemble beat the best single (correct call).
- probe_screen.py: added --max-train (fixed subset, folds keep groups), --threads, cache key includes
  model and subset. Test running.
- The user's other jobs (ds-rerun 07_models.py, 08_joint_search.py) were using ~85% CPU: keep CV tests
  small (2-3 threads) until they finish.

## Queue (resume at item 1)

| # | Task | State |
|---|---|---|
| 1 | Import-check every installed library in `cv-skills`; record versions | Done: all 26 import. Notable: torch 2.14.0+cpu, torchvision 0.29, timm 1.0.30, albumentations 2.0.8, kornia 0.8.3, opencv 5.0.0, pandas 3.0.6, transformers 5.17, open-clip 3.3, cleanlab 2.9, optuna 5.0, smp 0.5 |
| 2 | Synthetic eval datasets with planted defects, answer keys, prompts, assertions | Done: classification 3086 + 600 production; detection 1200 images / 4177 anns; defects verified by an independent script; evals.json, evals/README.md, cv-workspace/iteration-1/*/eval_metadata.json |
| 2b | Deep research queue (one agent at a time): R1 data quality + splitting, R2 preprocessing + augmentation, R3 backbones + foundation models, R4 per-task SOTA (det/seg/pose/depth/video/OCR/3D/anomaly/retrieval/VLM), R5 training + post-processing + ensembling + evaluation + deployment. Output: research/R*.md | R1 done (research/R1-data-quality-and-splitting.md; key: OpenCV imread auto-applies EXIF while PIL does not; torchvision image decoders deprecated in 0.29, video removed in 0.26 -> TorchCodec; fastdup CC BY-NC-ND; DINOv2 Apache-2.0, DINOv3 custom gated; cleanlab 2.9 APIs confirmed; FiftyOne compute_leaky_splits; iterstrat MultilabelStratifiedKFold). R2 done (research/R2-preprocessing-and-augmentation.md; key: MIT albumentations archived 2025-07-10, last 2.0.8 = what cv-skills has; successor albumentationsx is AGPL/commercial + telemetry and shadows the same import name -> pin albumentations==2.0.8 or use torchvision v2 / Kornia; DEIMv2 relicensed non-commercial 2026-08-24 while DEIM/D-FINE/RT-DETR/RF-DETR stay Apache-2.0; imgaug broken on NumPy 2; standard TTA hurt 11/12 medical pairs (arXiv 2604.09697); stain augmentation beats normalisation; restoration before recognition often hurts; normalisation stats are checkpoint-specific; detectors turn Mosaic off for the last 8-15 epochs). R3 done (research/R3-backbones-foundation-models.md, 310 lines; split into cv-model-selection/references/backbones.md and cv-training/references/adaptation.md; key: mirror licence tags wrong both ways (Apple AMLR, ConvNeXt V2/MAE NC), DINOv3 custom-commercial, permissive defaults DINOv2-reg / SigLIP 2 / PE / TIPSv2, LP-FT +10% OOD, WiSE-FT +4-6 pp under shift, tuned PEFT ~ full FT, CPU ViT-S@224 ~27 img/s). R4 done (research/R4-per-task-models.md, 754 lines; split into cv-model-selection/references/<task>.md x13 + licence-traps.md + metric-libraries.md (also copied to cv-evaluation/references); key: DEIMv2 NC since 2026-08-24 confirmed, YOLO26 AGPL, RF-DETR best permissive real-time detector (XL/2XL PML), OpenMMLab frozen, SegFormer/mit_b* NC (segmenter now warns), torchmetrics MeanIoU per-image with no ignore index, COCOeval default iouType segm). R5 done (research/R5-training-postprocessing-evaluation.md, 1,112 lines, 389 arXiv citations machine-checked; split into cv-training/references/training-methods.md, cv-postprocessing/references/methods.md, cv-method-comparison/references/ensembling-and-search.md, cv-evaluation/references/{methods,deployment}.md, cv-scientist/references/library-status.md; key: bf16 autocast ~36x slower on this CPU, channels_last -20%, torch.compile needs a C++ compiler (absent), torch 2.14 default ONNX exporter needs onnxscript (legacy dynamo=False works), torchmetrics defaults (MulticlassAccuracy macro, BootStrapper 10 resamples + fails on mAP, no HD95), WBF needs normalised boxes, Muon not for fine-tuning, greedy not uniform soups, TTA/test-time adaptation can hurt, TS on non-logit scores explained). All research done; prompts for R4/R5 in the session scratchpad forbid sub-agents and write to research/R4-*.md, R5-*.md |
| 2c | Real-world eval datasets (Apoorva: synthetic images differ from real ones; grade metrics on real data too). Proposed: EuroSAT RGB 94.3 MB (MIT), Oxford-IIIT Pet 791.9 + 19.2 MB (CC BY-SA 4.0, confirmed on VGG page), Roboflow 100 construction-safety 150.6 MB (HF tag license:cc). Penn-Fudan rejected (server SSL certificate fails) | Downloaded to evals/real/; converted by evals/prepare_real.py into evals/real_prepared/ with FIXED benchmark sets: construction_safety (pool 1087 imgs / 7009 boxes COCO; benchmark = official test 119), oxford_pets (pool = official trainval 3680; benchmark = official test 3669), eurosat (pool 21600; benchmark fixed stratified 20% = 5400). Real evals defined: eval-real-pets-finegrained, eval-real-det-construction (evals.json + eval_metadata.json). Reference floor for Pets computed 2026-09-28 by evals/compute_real_references.py: DINOv2 ViT-S/14 frozen CLS at 224 + logistic regression (C=1 by 5-fold CV on pool, CV acc 0.964) -> benchmark accuracy 0.9518 (Wilson 95% 0.9443-0.9582), macro-F1 0.9509, n=3669; stored in evals/answer_keys/real_references.json. Eval data copied (2026-09-28) to C:\dev\cv-eval-data\{shapes_classification, shapes_detection, oxford_pets, construction_safety} so test agents never browse evals/ (answer keys); counts verified. pyarrow added to the env. Apoorva also offered knee images on the GCP server "rsna-knee" if needed: NOT used yet (medical, privacy); ask before accessing |
| 3 | No-skill baseline runs (synthetic + real evals), Sonnet, one at a time | Queued after research |
| 4 | `cv-scientist` orchestrator | Draft written (SKILL.md, references/routing.md, references/templates.md); validator passes; revisit after research |
| 5 | `cv-data-audit` (+ scripts: integrity/format/stats audit, near-duplicate and cross-split leakage finder, label-noise, shortcut check) | SKILL.md + references/methods.md (from R1) written. Scripts tested on eval data: image_audit.py (all integrity defects), find_duplicates.py (Hamming 8: 90% recall / 81% precision; contact sheet), coco_validate.py (exact on files/sizes/categories/invalid boxes; 82/83 duplicate boxes; 17/21 corner-format, no false alarm), render_annotations.py, embed_images.py (DINOv2-S at 224, ~10 img/s CPU), label_issues.py (97% recall / 93% precision on planted noise; OOF probe acc 0.945), shortcut_screen.py attributes (no false shortcut), shift_report.py (camera shift found: AUC 0.999). Masking test running |
| 6 | `cv-validation-design` (+ grouped / near-duplicate-aware split script) | SKILL.md + references (R1 section F) written. Scripts tested: build_groups.py (870 sessions -> 712 groups with duplicate links), make_split.py (single / multilabel / detection / time / external; balanced-partition search: detection gap 0.31 -> 0.09), cv_folds.py (0 groups shared; hash check). Grouped split leaves 8 of 190 planted duplicate pairs straddling vs 85 for random |
| 7 | `cv-preprocessing` (+ image statistics script) | SKILL.md, references/methods.md (R2 A2/A3/B), scripts/serve_parity.py written and tested: identical paths diff 0; an OpenCV bilinear serving path fails (max diff 3.7 in normalised units); files unreadable in both paths are reported separately |
| 8 | `cv-augmentation` (+ augmentation preview renderer) | SKILL.md, references/methods.md (R2 A1/C/D) and scripts/aug_preview.py written; validator passes. aug_preview tested on real construction-safety (det_flip / det_ssd / det_safe / det_lsj: per-class kept-box rate, tiny-box share) and Oxford Pets. Findings built into the skill: torchvision RandomPhotometricDistort permutes channels with no switch (turns orange vests blue), zoom-out 4x makes helmets unreadable (det_safe caps at 2x: 0 tiny boxes vs 6), TrivialAugment can shift pets out of frame and invert coat colour |
| 9 | `cv-model-selection` (verify current leaderboards and licences) | SKILL.md written (licence gate, learnability baselines, per-task shortlist of permissive first choices, screening on shared folds incl. external trainers re-scored with common evaluation, metric traps, gate, red flags) + 16 reference files from R3/R4; validator passes |
| 10 | `cv-training` (+ training template) | SKILL.md (adaptation ladder, recipes, proxy, gate) + references/adaptation.md (R3 E) written; refine with R5. scripts/train_classifier.py tested (smoke) + new --lp-init (LP-FT: sklearn probe on frozen features loaded into the head; probe score logged); scripts/train_detector.py (torchvision Faster R-CNN / RetinaNet / FCOS, shared aug policies, strong-off-last, EMA with warm-up, pycocotools COCO metrics, val_dets.json) tested on real construction-safety: mAP 0.05 -> 0.11 in 2 tiny epochs. EMA fixed in both scripts (timm ModelEmaV3 use_warmup): EMA was 0.03 vs 0.11 raw, now 0.114 vs 0.111. scripts/train_segmenter.py written (smp, class-map with 255 ignore, metrics at original mask size, per-image counts saved); test on Oxford Pets trimaps queued (the Pets download also contains 7,390 macOS '._' metadata files named like images: a real audit trap) |
| 11 | `cv-postprocessing` | SKILL.md written (operating point, calibration, conformal/abstention, detection/segmentation outputs, TTA as candidate, gate, red flags) + references/methods.md (R5 B). scripts/postprocess_cls.py written and tested (temperature scaling cross-fitted with a Brier+NLL verdict: on k-NN probabilities NLL improved 1.16 -> 0.65 while ECE worsened 0.07 -> 0.26, so NLL alone would have misled; operating-point thresholds for target recall / max F1 / min cost, cross-fitted with Wilson intervals: 80% recall on a 10-positive rare class cost precision 0.06; multilabel per-class thresholds; apply mode). scripts/postprocess_det.py written and tested: fuse (WBF / NMW / NMS / soft-NMS via ensemble-boxes 1.0.9 MIT; self-fusion sanity check: WBF/NMW/NMS keep AP 0.1145, soft-NMS halves it -> not a fusion method) and per-class score thresholds (max-F1 or min precision at IoU 0.5; construction-safety smoke model: no-helmet recall 0 on 15 boxes). Real two-model fusion test done: WBF of two small Faster R-CNNs (AP 0.1145 and 0.1353) gave AP 0.1358; paired image bootstrap: AP tie, AP50 -0.015 (B better), AP75 +0.013 (A better) -> fusion trades loose-box recall for tighter boxes; keep the single model on the primary metric. |
| 12 | `cv-method-comparison` (+ proxy screening and joint search templates) | SKILL.md written (rules, proxy ladder, screen, joint search, ensembles, gate). compare.py CLI, probe_screen.py (tested: 5 variants x 596 images, 5 folds), ensemble.py (OOF folders, recal baseline, group cross-fitting; tested), joint_search.py (Optuna TPE + pruning, reference trial 0, paired top-trial table; tested; pruning made less eager: auto median/Hyperband, no pruning in first third of epochs). search_space.json template now carries per-candidate `args` and `knobs` |
| 13 | `cv-evaluation` (+ metrics with intervals, slices, sanity tests) | SKILL.md written (score once, intervals, slices, deployment gap, error analysis, explanations with randomisation check, subgroups, export) + references (R5 D/E, metric libraries). scripts/export_check.py tested: ONNX (legacy exporter) parity max prob diff 4e-6, argmax 100%; ORT 4.8 ms vs PyTorch 25 ms p50 at batch 1; static INT8 SLOWER (7.0 ms) and changed 6% of predictions -> INT8 is a measured candidate only. scripts/metrics_ci.py written and tested: cls (accuracy/macro-F1/balanced acc/AUROC/top-5/ECE + per-class Wilson + most-confused pairs; group bootstrap; slices; paired A-vs-B; lower-is-better handled for ECE) and det (COCO AP family, images re-identified so bootstrap duplicates count; reproduces the trainer's mAP 0.1145 exactly). seg mode written, untested until train_segmenter runs. |
| 14 | With-skill test runs in pairs, grade, fix, verify | NEXT. All 10 skills validate; every script/reference named in a SKILL.md exists (24 scripts, all tested). Runs one agent at a time (Sonnet): no-skill baseline + with-skill for each of 4 evals; data in C:\dev\cv-eval-data; runner template in session scratchpad (eval_runner_template.md): CPU <=4 threads, clock budget 60 min synthetic / 90 min real, no sub-agents, no access to evals/ or other runs |
| 15 | Benchmark, README, install only on request | Queued |

### Iteration 3 (approved by Apoorva 2026-09-28: 'run iteration 3 and install too, make the skill best')

Version 3 changes before launch: shortcut_screen --cue-only (1 min, found the watermark); provenance row in
Stage 0 + 'open every other file' step 0 + delivered-files table in Gate A; menu-coverage rule for every
decision-log block; early hand-off test (cv-training Step 5); NEW det_tta.py (orig/hflip/scale views ->
postprocess_det fuse; tested: TTA +0.029 AP [0.023, 0.034] on a weak model); NEW convert_labels.py
(class folders / YOLO incl. data.yaml and polygons / VOC with DOCTYPE/ENTITY rejection; round trips exact);
fusion/ensemble name collisions for files in one folder fixed. Tabular skills NOT touched: another session
is applying ITERATION-4-FIXES.md there (its provenance, menu-coverage and early-serving lessons were ported).

| Order | Eval | Budget | State |
|---|---|---|---|
| 1 | eval-real-det-construction | 90 min | done: 10/11; benchmark AP 0.340 [0.298, 0.404]; PAIRED vs no-skill: AP +0.015 [0.0006, 0.027] = with-skill BETTER (AP50 tie). Final-fit rule worked (half the clock reserved, 978 images). Platform outage paused it 12:06-15:33; resumed via SendMessage |
| 2 | eval-cls-shortcut-shift | 60 min | done: 48.7 min, 251k tokens; 13/17. WATERMARK FOUND for the first time (cue-only test, contact sheets, 8% border crop, production star recall 0.991) but shipped a ResNet-18 LP-FT at 160 px: production acc 0.915 (iter-2 DINOv2 probe 0.998). Causes: train_classifier never passed img_size to timm (DINOv2 stuck at 518 px) and no bundled way to ship a frozen probe; backbones compared on different pipelines. FIXED: img_size passed; NEW train_probe.py (C on folds, holdout, refit, weights_only model, load_probe, export via --module-arg; tested: border-cropped DINOv2 probe = production 0.980, star recall 0.991, export parity 1.4e-6); same-pipeline rule in cv-method-comparison |
| 3 | eval-det-annotation-defects | 60 min | done: ~50 min, 278k tokens; 9/13; missing-annotation check ran (real unlabelled objects confirmed); FAIR vs no-skill on 124 common images: AP 0.677 vs 0.747, diff -0.070 [-0.105, -0.045] (320 px inherited, no size screen). FIXED: coco_validate ACTION line at >10% tiny, preprocessing exception, Stage 4 gate requires the size table |
| 4 | install the 10 CV skills into ~/.claude/skills | - | DONE 2026-09-28 (tools/install_skills.py; installed copies validated, scripts and cross-skill imports run from ~/.claude/skills; harness lists all 10). Summary: cv-workspace/iteration-3/benchmark.md |

### Iteration 2 queue (with_skill only; baselines unchanged; one agent at a time)

Before launch: shortcut contact sheet verified (flags 'star' bottom_right, one-corner warning, sheet shows
the 'C' watermark on 10/12 star crops); eval-specific numbers/examples removed from skills (tabular
lesson: never quote eval data in skills).

| Order | Eval | Budget | State |
|---|---|---|---|
| A | eval-det-annotation-defects | 60 min | done: 51.9 min, 245k tokens; 10/13 (iter-1 8/13); own test AP 0.678 [0.655, 0.710]; resolution screen ran (320 vs 640 -> 512). FAIR vs iter-1 no-skill on 223 common images: AP 0.731 vs 0.735, diff -0.004 [-0.019, 0.011] = TIE (iter-1 gap was -0.127). Still missed: missing-annotation check (deferred), refit, seeds, ensemble |
| B | eval-real-det-construction | 90 min | done: 73.8 min, 320k tokens; 10/11; benchmark AP 0.192 [0.167, 0.227] = WORSE than iter-1 with-skill 0.300 and no-skill 0.325: after measuring 10.4 min/epoch it capped the final model to 250 images (process stages ate the clock); also hit a split-role bug (make_split refused --test-size 0 -> empty holdout validation, caught mid-run) |
| C | eval-cls-shortcut-shift | 60 min | done (version 2b): 38.5 min, 346k tokens; 14/17 (iter-1 13/17); production 0.9983 [0.991, 1.000]; refit on all dev data done; 18/21 internal-test errors verified as label errors; watermark MISSED again (shortcut test deferred to the end and killed, ~17 CPU-min). Fixed after: --cue-only mode (1 min vs ~25), run in Stage 3 |

Iteration 2 summary: cv-workspace/iteration-2/benchmark.md. Detection quality gap closed on synthetic data
(tie); construction got worse under v2 (fixed in 2b, untested); shortcut test made cheap after run C.
Possible iteration 3 (user decision, ~2.5 h / ~0.7M tokens): construction and shapes-cls with version 2c.

Version 2b fixes (after run B): make_split.py accepts --test-size 0 (tested: construction 923 train+folds /
164 holdout; Pets 3311/369) and cv-validation-design shows the fixed-benchmark recipe; cv-scientist 'Protect
the final fit' (measure an epoch, reserve >= half the remaining clock for the final model on ALL development
images, size screens to the rest); cv-training: run det_missing_annotations.py right after the first fold run.

### Iteration 1 result (cv-workspace/iteration-1/benchmark.md, benchmark.json)

Assertion pass rate 0.49 without skill -> 0.82 with skill (+0.33); 54 vs 44 min, 355k vs 197k tokens per run.
Model quality: ties on classification; no-skill better on both detection evals (resolution, refit).

### Iteration-2 fixes: APPLIED 2026-09-28 (validator passes, all references resolve)

Applied: refit on all development data before the test (cv-method-comparison rule 6 + gate; no extra
internal test when a fixed benchmark exists, cv-validation-design); equal-epoch comparisons and 'tie' only
with a measured error (rules 3-4); unused-budget rule and report/metric reconciliation (cv-scientist);
mandatory input-size screen incl. upscaling for tiny objects (cv-preprocessing, cv-model-selection,
train_detector docstring: *_320_* variants shrink inputs); CPU throughput measured not quoted (cv-training);
shortcut_screen.py saves cue contact sheets + one-corner warning, audit skill says look before judging;
NEW det_missing_annotations.py (tested: recovered 68% of objects lacking a correct box at ~92% precision on
the synthetic detection set) wired into cv-data-audit; probe_screen.py --C grid (tested); export_check.py
--module for any classifier (tested on a frozen DINOv2 probe: parity 5e-7) + detector export guidance.

### Iteration-2 fixes collected (history)

- probe_screen.py: add a --C grid (the Pets run wrote its own head screen because C=1.0 is hardcoded).
- cv-scientist hand-off: re-check every number and claim in report.md against the metric files (Pets
  run's log said the bootstrap "did not finish" although the file with n_boot=1000 exists).
- Budget: with-skill runs spend about 2x time/tokens on process; the compact-mode section should say
  which stages to shrink first when a strong frozen probe already sits at the reference level.

### Eval-design notes from iteration 1

- Shapes classification: the planted star watermark is detectable (cue-present probe) but barely hurts a
  fine-tuned ResNet18 (no-skill run reached 0.988 on production without noticing it). The eval can grade
  whether the shortcut is FOUND, not whether finding it saves accuracy. Production > validation in that
  run is largely explained by the 4% planted label noise in the pool (production labels are clean).

- cv-data-audit / shortcut_screen.py: when the cue-present test flags a region, save and LOOK at a contact
  sheet of that region's crops (flagged class vs others) before judging; a cue in one corner only cannot
  be explained by object geometry. (Shapes with-skill run dismissed the planted watermark as 'star points'.)
- Budget rule: unused clock budget is spent on the highest-value skipped item (second family, LP-FT,
  shift-matched augmentation) before hand-off; stopping at 31/60 min with those skipped is a failure.
- cv-training Step 1: the '27 img/s ViT-S' figure is an i7 batch-1 number; on this project's CPU DINOv2-S@224
  embeds at ~5 img/s with 4 threads. Tell agents to measure, not to compare against the quoted figure.
- export_check.py only accepts train_classifier model.pt; add a generic path (timm backbone + sklearn
  linear head, or --module file:func) so frozen-probe models can use it (Shapes run wrote its own).

- Detection, small objects: when coco_validate reports a large tiny share, screening at least two detector
  input sizes INCLUDING UPSCALING (e.g. torchvision min_size 320 vs 800) is mandatory; tiling is a different
  lever. The with-skill run skipped it and lost 0.13 AP to a no-skill run that used the 800-px default.
- Screening fairness: compare architectures at equal epochs (or full learning curves), not equal wall-clock;
  'ties go to the simpler model' only against a MEASURED noise floor (1 seed = no tie can be declared).
- cv-data-audit: add a bundled missing-annotation check for detection (confident predictions with no
  matching box, out-of-fold); the with-skill run skipped the check, arguing generated labels cannot be missing.
- The '320-px' variants in torchvision (fasterrcnn_mobilenet_v3_large_320_fpn) downscale inputs; say so in
  cv-model-selection/cv-training so they are not chosen for small objects by accident.

### Eval run queue (one agent at a time, Sonnet; timing saved per run as timing.json)

| Order | Eval | Config | Budget | State |
|---|---|---|---|---|
| 1 | eval-real-pets-finegrained | without_skill | 90 min | done: 43.6 min, 162.7k tokens; benchmark top-1 0.9507 (ref 0.9518); graded 5/10 (no audit/duplicate check, no seed spread, no CI, confusion pairs asserted not computed). Note: it bypassed a harness block on the file name report.md via Bash; brief now says save as results.md instead |
| 2 | eval-real-pets-finegrained | with_skill | 90 min | done: 81.9 min, 310.5k tokens; benchmark top-1 0.9515, CI [0.9452, 0.9583] (ref 0.9518); graded 10/10. Found 23 pool/benchmark near-duplicate clusters (leak in the fixed split; -0.03 pp when excluded), group split with holdout confirmation, TTA measured and rejected |
| 3 | eval-cls-shortcut-shift | without_skill | 60 min | done: 33.2 min, 244.5k tokens; production acc 0.9883 (CI 0.976-0.994), val 0.921; graded 9/17 (missed EXIF, duplicates, label noise, watermark shortcut, second family, ensemble, internal test) |
| 4 | eval-cls-shortcut-shift | with_skill | 60 min | done: 33.4 min, 314.3k tokens; production 0.995 (CI 0.988-1.0), internal test 0.957; graded 13/17 (missed: watermark dismissed as 'star points', no shift-matched augmentation screen, one family, no ensemble). Stopped at 31/60 min citing CPU throughput |
| 5 | eval-det-annotation-defects | without_skill | 60 min | done: 40.3 min, 226.8k tokens; Faster R-CNN AP 0.65 / AP50 0.79 on 956 held-out images (trained on 191); graded 5/13 (corner-format clipped, no duplicate boxes, no missing-annotation check, no intervals, no post-processing/ensemble/spread) |
| 6 | eval-det-annotation-defects | with_skill | 60 min | done: 39.5 min, 384.5k tokens; own test AP 0.563; graded 8/13 (all integrity/format/duplicate defects found incl. corner format; missed missing-annotation check, resolution screen, intervals, ensemble, spread). FAIR COMPARISON on 157 common held-out images vs clean key (cv-workspace/iteration-1/eval-det-annotation-defects/fair_comparison.json): no-skill Faster R-CNN AP 0.725 vs with-skill SSDlite 0.598 |
| 7 | eval-real-det-construction | without_skill | 90 min | done: 58.7 min, 155.5k tokens; benchmark AP 0.325 / AP50 0.664, AP_small 0.004 (320-px variant downscales 640-px photos), no-helmet AP50 0.188; graded 6/11 |
| 8 | eval-real-det-construction | with_skill | 90 min | done: 61 min, 410.3k tokens; benchmark AP 0.300 [0.268, 0.345], AP50 0.588, no-helmet 0; graded 10/11. Paired vs no-skill on the benchmark: no-skill better AP +0.025 [0.001, 0.074], AP50 +0.077 (with-skill final model trained on 692/1,087 images, never refit) |

