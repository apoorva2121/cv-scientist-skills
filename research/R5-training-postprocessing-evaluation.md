# R5 - Training, post-processing, ensembling, evaluation and deployment

Verified method inventory for the Computer Vision Scientist skill, area 5: everything after the model family is chosen. It covers how to train the model, how to turn raw outputs into decisions, how to combine models, how to measure them honestly, and how to ship them. Backbones and adaptation (linear probe, LP-FT, LoRA, layer decay, WiSE-FT, distillation basics) are in R3. Per-task models and their metric libraries are in R4. Data audit and splitting are in R1, and preprocessing and augmentation are in R2. This file points to those instead of repeating them.

**Checked 2026-09-28.** How each kind of fact was verified:
- **Papers.** Every arXiv ID was resolved through the arXiv API (`export.arxiv.org/api/query?id_list=...`), which gave its title and v1 date. Citations are written as `arXiv ID (v1 YYYY-MM-DD)`. Claims are paraphrased or quoted from the abstract unless marked *(paper body)*; paper-body facts were read from arXiv HTML or ar5iv. The reference appendix at the end was generated from the API output, and a script checked every dated citation in this file against it.
- **Packages.** The PyPI JSON API gave the latest version, the upload date of the first file of that release, and the licence metadata.
- **Repositories.** The GitHub REST API (15 calls) gave `license.spdx_id`, `archived` and `pushed_at`. None of the 15 repositories is archived.
- **APIs.** Checked in the local `cv-skills` environment with `inspect.signature`, docstrings and small in-memory tests that write no files. Versions: torch 2.14.0+cpu, torchvision 0.29.0, timm 1.0.30, torchmetrics 1.9.0, optuna 5.0.0, onnx 1.23.0 (IR 14, max opset 28), onnxruntime 1.30.0, grad-cam 1.5.7, ensemble-boxes 1.0.9, scikit-learn 1.9.1, scipy 1.18.1, cleanlab 2.9.0, segmentation-models-pytorch 0.5.0, kornia 0.8.3, scikit-image 0.26.0 and opencv 5.0.0. Libraries that are not installed were checked against their source on GitHub (raw files) or their official docs: MONAI, nnU-Net, NNCF, MAPIE, TorchCP, Captum, fairlearn, FiftyOne, ExecuTorch, LiteRT Torch, Ray, schedule_free, Muon, pytorch-metric-learning, detectron2, DINOv2 and torch-lr-finder.
- **Local measurements.** Taken on the project machine described in STATUS.md (8-core Ryzen 7 7735HS, no CUDA). Each is a single run, and the user's other jobs may have been using the CPU, so treat them as indicative.
- **Tags.** "(reasoning)" marks a mechanism argued here rather than taken from a source. "(practice)" marks common practice with no verified source.

**Licence vocabulary** (as in R3). *Permissive* means Apache-2.0, MIT or BSD. *Custom-commercial* means a bespoke licence that allows commercial use. *NC* means non-commercial. *Copyleft* means GPL, LGPL or AGPL. *Source-available* means BSL and similar licences that are not open source until a change date.

---

## Most decision-changing findings (2026-09-28)

1. **bf16 autocast is a trap on the project CPU.** In a local ResNet-18 training step (batch 16, 128 px, 4 threads, median of 4):
   - fp32: 845 ms
   - fp32 + `channels_last`: 672 ms
   - `torch.autocast("cpu", dtype=torch.bfloat16)`: 30,490 ms, about 36x slower
   - bf16 + `channels_last`: 25,038 ms

   On this machine, train in fp32 and use `channels_last` for CNNs. Mixed precision only pays off on GPUs, or on CPUs with native bf16 support (not verified here).
2. **`torch.compile` will not work here as the machine is set up.** Windows Inductor needs a C++ compiler: MSVC `cl`, `clang-cl` or `icx-cl` (PyTorch tutorial "How to use torch.compile on Windows CPU/XPU"). `where cl` and `where clang-cl` find nothing on PATH.
3. **Export defaults have moved.** In torch 2.14, `torch.onnx.export(..., dynamo=True)` is the default ("Changed in version 2.9"), and it needs `onnxscript`, which `cv-skills` lacks, so the default export fails with `ModuleNotFoundError: No module named 'onnxscript'`.
   - `dynamo=False` still works but emits a DeprecationWarning ("legacy TorchScript-based ONNX export").
   - `torch.jit.script` and `torch.jit.trace` emit a FutureWarning: "deprecated. Please switch to `torch.compile` or `torch.export`".
   - `torch.ao.quantization` emits a DeprecationWarning ("will be removed in 2.10", yet it is still present in 2.14) that points to torchao.
   - Conflict: the Core ML Tools docs still call `torch.jit.trace` the stable, recommended conversion path, and call `torch.export` "beta".
4. **TensorRT 11 removed INT8 calibration.** TensorRT 11.x removes `IInt8Calibrator` and all its subclasses, removes the `BuilderFlag.FP16` and `BuilderFlag.INT8` precision flags, and makes every network strongly typed (NVIDIA migration guide). Quantise explicitly with Q/DQ nodes through NVIDIA Model Optimizer (`nvidia-modelopt` 0.47.0). Older "calibrate INT8 in TensorRT" recipes are obsolete.
5. **DirectML is in "sustained engineering"** (ONNX Runtime docs), and new Windows work has moved to WinML. Its PyPI build lags: `onnxruntime-directml` 1.24.4 (2026-03-17) against `onnxruntime` 1.30.0 (2026-09-10). Do not plan GPU work on the AMD Radeon 680M around DirectML without testing it.
6. **torchmetrics defaults change numbers silently** (all verified locally):
   - `MulticlassAccuracy` defaults to `average="macro"`. On the same predictions it gave 0.500, against 0.833 with `average="micro"`.
   - `BootStrapper` defaults to 10 resamples and raises `TypeError` when wrapped around `MeanAveragePrecision`, so torchmetrics offers no image-level bootstrap for mAP.
   - `HausdorffDistance` has no percentile argument, so it cannot give HD95. Use MONAI `HausdorffDistanceMetric(percentile=95)`, which returns NaN when both masks are empty.
   - `DiceScore` defaults to per-sample aggregation and `include_background=True`.
7. **ensemble-boxes WBF silently drops boxes in pixel coordinates.** Boxes must be normalised to [0, 1]. Given pixel boxes, it clips them to 1, skips them as zero-area, and returns empty arrays, with only warnings (tested locally).
8. **Muon has arrived in torch and timm, but it is not a fine-tuning default.**
   - Availability: `torch.optim.Muon` exists in torch 2.14 and accepts only 2-D parameters (anything else raises `ValueError`). timm 1.0.30 offers `muon`, `adamuon` and `nmuon`; they flatten conv kernels and fall back to AdamW for 1-D parameters.
   - Against it for fine-tuning: fully fine-tuning Adam-pretrained models with Muon "leads to degraded performance due to an optimizer mismatch", and LoRA narrows the gap (arXiv 2605.10468 (v1 2026-05-11)).
   - For it from scratch: Muon beat AdamW when training ViTs from scratch, with gains that depend on the recipe (arXiv 2605.24770 (v1 2026-05-23)).
   - Keep AdamW (or SGD for CNNs) for fine-tuning pretrained backbones.
9. **Temperature scaling (TS) is not a safe default for non-logit scores.** The project's own result (TS on k-NN probabilities improved NLL but worsened ECE and Brier) fits the literature:
   - TS lowers the confidence of every prediction, "irrespective of whether the classification of a given input is correct or incorrect" (arXiv 2207.06211 (v1 2022-07-13)).
   - No existing calibrator is at once accuracy-preserving, data-efficient and expressive (arXiv 2003.07329 (v1 2020-03-16)).
   - Most calibration methods are "useless or harmful for failure prediction" (arXiv 2303.02970 (v1 2023-03-06)), and TS "negatively affects" the set sizes of adaptive conformal prediction (arXiv 2402.05806 (v1 2024-02-08)).
   - Binned ECE estimators are biased (arXiv 2012.08668 (v1 2020-12-15)).
   - For k-NN, fit the vote-kernel temperature instead (DINOv2's k-NN uses `softmax(sim / 0.07)`), or use binning or isotonic calibration on out-of-fold scores. Choose the calibrator by the metric you will report (section B3).
10. **Test-time augmentation is not a default.** Standard TTA lowered accuracy on 3 MedMNIST datasets across 4 architectures, by up to 31.6 percentage points, and got worse with more views. It helped in only one case (arXiv 2604.09697 (v1 2026-04-06)). Validate it for each model and dataset. nnU-Net still turns mirroring TTA on by default.
11. **Test-time adaptation can collapse.** On a long continual benchmark, "all but one" state-of-the-art method ended up worse than a non-adapting model, and periodic resetting did as well or better (arXiv 2306.05401 (v1 2023-06-08)).
12. **Use greedy soups, not uniform soups.** For CLIP ViT-B/32 on ImageNet: best individual model 80.38%, uniform soup 79.97%, greedy soup 81.03% (arXiv 2203.05482 (v1 2022-03-10), Table 3, paper body).
13. **Ensembles need diversity in pre-training or method, not just seeds.** Models fine-tuned from one checkpoint stay in the same basin and have "limited diversity" (arXiv 2303.03374 (v1 2023-03-06)). Two models trained with different methods, each about 76.5% on ImageNet, ensemble to 83.4% (arXiv 2110.12899 (v1 2021-10-20)). A larger single model can reproduce the benefits of an ensemble (arXiv 2202.06985 (v1 2022-02-14)).
14. **Overtuning is common.** In about 10% of HPO cases, the selected configuration generalises worse than the default or the first configuration tried (arXiv 2506.19540 (v1 2025-06-24)). A tuned cross-entropy baseline beat all but one of eight published data-efficient methods (arXiv 2108.13122 (v1 2021-08-30)).
15. **Confidence intervals are missing where they matter.**
    - Of 221 MICCAI 2023 segmentation papers, one reported confidence intervals. The median CI width, 0.03 Dice, is three times the median gap between the first- and second-ranked methods (arXiv 2409.17763 (v1 2024-09-26)).
    - A 1%-wide CI needs about 100-200 test cases when the SD is around 3%, and over 1,000 for harder tasks (arXiv 2307.10926 (v1 2023-07-20)).
16. **New licence traps in this area:**
    - Source-available or copyleft: `alibi-detect` (Business Source License 1.1), `deepchecks` (AGPL), `boxmot` (AGPL-3.0), TorchCP (LGPL-3.0), `quantus` (GPLv3 classifier on PyPI; GitHub says NOASSERTION), `zennit` (GPL/LGPL), `medpy` (GPL).
    - Custom or proprietary: `relplot` (Apple licence text), TensorRT (proprietary).
    - Not official: PyPI `trackeval` is a third-party fork (kovalp/TrackEval); the official JonathonLuiten/TrackEval is GitHub only.
17. **Stale or restricted tools:**
    - Stale: `tidecv` 1.0.1 (2020; repo last push 2023-03-31), `ttach` 0.0.3 (2020), `pydensecrf` 1.0rc3 (2018), `ffcv` 1.0.2 (2023; repo last push 2024-06-16), `semilearn` 0.3.2 (2023-11), `lvis` 0.5.3 (2020), `model-card-toolkit` 2.0.0 (2023), `hpbandster` 0.7.4 (2018), `coral-pytorch` 1.4.0 (2022), `torch-ema` 0.3 (2021), `abstention` 0.1.3.1 (2020).
    - Platform-limited: LiteRT Torch (the renamed `ai-edge-torch`) lists Linux as its only OS. DALI needs Linux x64 and an NVIDIA GPU.

---

## A. Training

### A1. Optimisers and schedules for fine-tuning vision models

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **AdamW** (decoupled weight decay) | Default for ViT and transformer fine-tuning, and a safe default for CNNs | Plain Adam with L2 `weight_decay` is not the same thing. A too-high LR destroys pretrained features (see LP-FT in R3) | arXiv 1711.05101 (v1 2017-11-14). Optimizer benchmark over 50,000 runs: "Adam remains a strong contender, with newer methods failing to significantly and consistently outperform it", and trying several optimizers at defaults works about as well as tuning one: arXiv 2007.01547 (v1 2020-07-03) | `torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, *, foreach=None, fused=None)`. `timm.optim.create_optimizer_v2(model, opt="adamw", lr=..., weight_decay=..., layer_decay=None, filter_bias_and_bn=True)` | torch permissive (BSD-style); timm Apache-2.0 |
| **SGD + momentum / Nesterov** | CNN fine-tuning. Fine-tuning momentum matters, and the best effective LR depends on how similar source and target domains are | torch's default `momentum=0` gives plain SGD. ResNet Strikes Back found SGD combined with BCE "difficult" to converge *(paper body, ar5iv)* | arXiv 2002.11770 (v1 2020-02-19). Nesterov and Adam "match or exceed" LARS/LAMB at large batch: arXiv 2102.06356 (v1 2021-02-12) | `torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False)`. timm `opt="sgd"` / `"momentum"` / `"nesterov"` (`create_optimizer_v2` default `momentum=0.9`) | as above |
| **LAMB / LARS** | Batches in the thousands. The DeiT III and ResNet Strikes Back A1 recipes use LAMB (DeiT III: "LAMB ... includes gradient clipping by default", *paper body*) | No benefit at the small batches typical of fine-tuning; traditional optimizers suffice across batch sizes | arXiv 1904.00962 (v1 2019-04-01); arXiv 1708.03888 (v1 2017-08-13); reality check arXiv 2102.06356 (v1 2021-02-12) | timm `lamb`, `lambc`, `lambw`, `lars`, `larc`, `nlars`, `nlarc` (from `timm.optim.list_optimizers()`) | Apache-2.0 |
| **Lion** | Memory (tracks momentum only). ViT on ImageNet up to +2%. Its gain "grows with the training batch size" | Needs "a smaller learning rate than Adam", so AdamW settings do not transfer. The abstract notes cases where "improvements are small or not statistically significant" | arXiv 2302.06675 (v1 2023-02-13) | timm `lion` (also cautious `clion`); `lion-pytorch` 0.2.5 (2026-07-09) | MIT (lion-pytorch) |
| **Muon** (orthogonalised momentum for 2-D weights) | Training ViTs from scratch: beat AdamW on ImageNet-100 and Pl@ntNet-300K, most on long-tailed macro top-1, and benefits more than AdamW from heavy augmentation. Muon-pretrained features are more robust and transfer better | Full fine-tuning of Adam-pretrained models degrades; LoRA narrows the gap. Without heavy augmentation Muon showed "spectral concentration and mode collapse" in deep MLP blocks | arXiv 2605.24770 (v1 2026-05-23); arXiv 2605.10468 (v1 2026-05-11); arXiv 2606.09658 (v1 2026-06-08); LLM scaling arXiv 2502.16982 (v1 2025-02-24) | `torch.optim.Muon(params, lr=0.001, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5, adjust_lr_fn=None)`, where `adjust_lr_fn` is one of `"original"` (default), `"match_rms_adamw"` ("can directly reuse the learning rate and weight decay tuned for AdamW", per the docstring) or `"spectral_unclamped"`. It raises `ValueError` for non-2-D parameters, so give conv, bias, norm, embedding and head parameters to AdamW. timm `muon`, `adamuon`, `nmuon` flatten conv weights and fall back to AdamW for 1-D parameters. Reference README (KellerJordan/Muon, MIT, pushed 2026-05-24): hidden weights only, "all the convolutional filters except the first one"; example groups use `lr=0.02` (Muon) and `3e-4` (AdamW) | torch permissive; MIT |
| **Schedule-Free AdamW / SGD** | No schedule and no need to fix the training length. It won the AlgoPerf self-tuning ruleset (fully hyperparameter-free) | Evaluating without calling `optimizer.eval()` uses the wrong weights. BatchNorm models need extra steps before evaluation. No verified evidence for vision fine-tuning | arXiv 2405.15682 (v1 2024-05-24); AlgoPerf analysis arXiv 2502.15015 (v1 2025-02-20) | `schedulefree` 1.4.1 (2025-03-24; repo facebookresearch/schedule_free Apache-2.0, pushed 2026-07-28). Classes: `AdamWScheduleFree`, `SGDScheduleFree`, `RAdamScheduleFree`, `ScheduleFreeWrapper` and closure variants. Call `optimizer.train()` / `optimizer.eval()` wherever `model.train()` / `model.eval()` is called. README: AdamW LR "1x-10x larger than with schedule-based approaches", SGD 10x-50x, `warmup_steps` recommended, beta 0.9 (0.95-0.98 for very long runs). Not in timm 1.0.30 | Apache-2.0 |
| **SAM / ASAM / LookSAM** | Sharp minima, as in ViTs and MLP-Mixers trained from scratch: +5.3% ImageNet top-1 for ViT-B/16 with Inception-style preprocessing. SAM also improves fine-tuning and gives label-noise robustness "on par with" dedicated methods | Two sequential gradient computations per step, which "can double the computational overhead", a poor trade on CPU | arXiv 2010.01412 (v1 2020-10-03); arXiv 2106.01548 (v1 2021-06-03); ASAM arXiv 2102.11600 (v1 2021-02-23); LookSAM arXiv 2203.02714 (v1 2022-03-05) | Not in torch 2.14 or the timm 1.0.30 optimizer list. `davda54/sam` (GitHub, MIT, pushed 2024-02-21) is the common implementation (constructor not re-verified). PyPI `sam-pytorch` 0.0.1 (2021) is stale | MIT |
| **Cosine decay** (SGDR, without restarts in practice) | The default schedule for fine-tuning | Needs the total length in advance, so training cannot be extended cleanly | arXiv 1608.03983 (v1 2016-08-13); cosine versus constant-plus-cooldown arXiv 2405.18392 (v1 2024-05-28) | `torch.optim.lr_scheduler.CosineAnnealingLR`. `timm.scheduler.create_scheduler_v2(optimizer, sched="cosine", num_epochs=300, warmup_epochs=0, warmup_lr=1e-05, min_lr=0.0, cooldown_epochs=0, step_on_epochs=True, updates_per_epoch=0)`; timm's scheduler names are `cosine`, `tanh`, `step`, `multistep`, `plateau` and `poly` | permissive |
| **Step / multistep** | Reproducing legacy CNN recipes | Needs hand-picked milestones | (baseline; no specific study) | `MultiStepLR`, `StepLR`; timm `step`, `multistep` | permissive |
| **One-cycle** (super-convergence) | Short runs at a high peak LR | Adds peak-LR and `pct_start` hyperparameters; momentum is cycled too (`cycle_momentum=True` by default) | arXiv 1708.07120 (v1 2017-08-23) | `OneCycleLR(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy="cos", cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=False)` | permissive |
| **WSD (warmup-stable-decay) / constant LR plus cooldown** | Training without a fixed length. Several run lengths can branch from one run, and scaling is "predictable and reliably similar to cosine". SWA along the trajectory helps | The evidence comes from LLMs; not verified for vision fine-tuning | arXiv 2405.18392 (v1 2024-05-28); MiniCPM arXiv 2404.06395 (v1 2024-04-09) | No timm scheduler exists (verified list). Build one: `SequentialLR(opt, [LinearLR(warmup), ConstantLR(factor=1.0, total_iters=...), LinearLR(decay to ~0)], milestones=[...])` (all classes exist in torch 2.14) | permissive |
| **Warmup** | Lets the network tolerate a larger target LR, which makes tuning more robust. A must with linear LR scaling | Too long a warmup wastes a short fine-tune (reasoning) | arXiv 2406.09405 (v1 2024-06-13); arXiv 1706.02677 (v1 2017-06-08) | timm `warmup_epochs`, `warmup_lr`, `warmup_prefix`; torch `LinearLR(start_factor=...)` | permissive |
| **LR range test** | Finding the scale of a workable LR quickly on a new dataset | Few iterations with an exponential sweep is noisy; smoothing and the divergence threshold change the answer | arXiv 1506.01186 (v1 2015-06-03) | `torch-lr-finder` 0.2.2 (2024-09-21): `LRFinder(model, optimizer, criterion, device=None, memory_cache=True, ...)`, then `.range_test(train_loader, val_loader=None, start_lr=None, end_lr=10, num_iter=100, step_mode="exp", smooth_f=0.05, diverge_th=5, accumulation_steps=1)` | MIT |
| **Batch size and LR scaling** | SGD: scale the LR linearly with batch size, plus warmup (no accuracy loss up to 8,192 images). Adam/RMSprop: square-root scaling. Increasing the batch can replace LR decay | EMA dynamics also change with batch size, so rescale EMA momentum as well | arXiv 1706.02677 (v1 2017-06-08); arXiv 2205.10287 (v1 2022-05-20); arXiv 1711.00489 (v1 2017-11-01); EMA scaling arXiv 2307.13813 (v1 2023-07-25) | (rules; no API) | - |
| **Gradient accumulation** | Emulates a larger batch when memory (or CPU time per step) limits it | BatchNorm still sees micro-batch statistics, so accumulation does not fix tiny-batch BN; prefer GroupNorm/LayerNorm or freeze BN (reasoning). Divide the loss by the number of accumulation steps (practice) | (no specific study) | Manual loop; `LRFinder.range_test(accumulation_steps=...)` supports it | - |
| **Gradient clipping** | Stability with high LRs, LAMB, detectors and AMP | Clipping scaled gradients under AMP. Call `scaler.unscale_(optimizer)` before clipping (GradScaler docstring example) | Theory: arXiv 1905.11881 (v1 2019-05-28); adaptive gradient clipping (normaliser-free nets) arXiv 2102.06171 (v1 2021-02-11) | `torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None)`. `timm.utils.dispatch_clip_grad(parameters, value, mode="norm")` (modes `norm`, `value`, `agc`); `timm.utils.adaptive_clip_grad(parameters, clip_factor=0.01, eps=0.001)` | permissive |

**Also in timm 1.0.30** (`list_optimizers()`): `adopt` (ADOPT, arXiv 2411.02853 (v1 2024-11-05)), `kron`, `mars`, `laprop`, `adan`, `madgrad`, `adabelief`, `adafactorbv`, `nadamw` and more. The `c`-prefixed names (such as `cadamw`) are "cautious" variants, and `bnb*` names need bitsandbytes. None of these has verified vision fine-tuning evidence that beats AdamW.

**Decision guidance (A1)**
- **Pretrained ViT or other transformer:** AdamW plus cosine decay plus linear warmup (about 5-10% of steps, practice), with layer-wise LR decay from R3. Keep weight decay off 1-D parameters and ViT tokens (A2).
- **Pretrained CNN:** SGD with Nesterov momentum 0.9, or AdamW. Tune the LR first; momentum matters too (arXiv 2002.11770).
- **Unknown training length, or a need to extend runs:** constant LR with a linear cooldown (WSD) or Schedule-Free. Save and evaluate in the right mode (`optimizer.eval()`).
- **Training from scratch (small models on CPU, or synthetic tests):** consider Muon (timm `muon` handles conv and 1-D parameters) as a second candidate beside AdamW. Do not use Muon for full fine-tuning of Adam-pretrained backbones.
- **Screen before tuning.** "Evaluating multiple optimizers with default parameters works approximately as well as tuning the hyperparameters of a single, fixed optimizer" (arXiv 2007.01547). Tune LR, weight decay and batch size before switching optimizer (arXiv 2108.13122).

**Pitfalls (A1)**
- Defaults differ between libraries: torch `AdamW` has `weight_decay=0.01`, torch `SGD` has `momentum=0`, and timm `create_optimizer_v2` has `momentum=0.9` and `filter_bias_and_bn=True`. Record the resolved optimizer in the run config.
- `torch.optim.Muon` given `model.parameters()` raises `ValueError` on the first bias or norm tensor. Split the parameter groups, or use timm `muon`.
- Schedule-Free: checkpoints and evaluation must happen after `optimizer.eval()`, and BatchNorm nets need the README's BN workaround.
- timm schedulers step per epoch unless `step_on_epochs=False` with `updates_per_epoch` set. Calling `.step(epoch)` with a step counter (or the reverse) gives a schedule that is wrong by the number of steps per epoch.
- A step scheduler resumed from a checkpoint must have its state restored too, or warmup runs again (practice).

### A2. Regularisation

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Weight decay with exclusions** (no decay on biases, norms, ViT tokens and position embeddings) | Always with AdamW. Tuning weight decay alone "significantly improves" long-tailed recognition | Decaying norm gains and position embeddings (practice; "no bias decay" is in Bag of Tricks) | arXiv 1812.01187 (v1 2018-12-04); arXiv 1810.12281 (v1 2018-10-29); long tail arXiv 2203.14197 (v1 2022-03-27) | `timm.optim.param_groups_weight_decay(model, weight_decay=1e-05, weight_decay_exclude_1d=True, no_weight_decay_list=())`. timm ViT `model.no_weight_decay()` returns `{'cls_token', 'dist_token', 'reg_token', 'pos_embed'}` (verified). `create_optimizer_v2(..., filter_bias_and_bn=True)` applies both | Apache-2.0 |
| **Drop path / stochastic depth** | Deep residual nets and ViTs trained or fine-tuned for many epochs. DeiT III uses 0.0 for ViT-S up to 0.5 for ViT-H on ImageNet-1k *(paper body)* | Small models and short fine-tunes (DeiT III uses 0 for ViT-S) | arXiv 1603.09382 (v1 2016-03-30); arXiv 2204.07118 (v1 2022-04-14) | `timm.create_model(name, drop_path_rate=0.1)` gives `DropPath(drop_prob=0.100)` in the last block (verified); `torchvision.ops.StochasticDepth` | permissive |
| **Label smoothing** | Improves generalisation and calibration. Competitive with loss correction under label noise | Teachers trained with it distil worse. It "consistently degrades" selective classification (fix with post-hoc logit normalisation). Under high noise its benefit vanishes and negative smoothing is better | arXiv 1512.00567 (v1 2015-12-02); arXiv 1906.02629 (v1 2019-06-06); arXiv 2403.14715 (v1 2024-03-19); arXiv 2003.02819 (v1 2020-03-05); arXiv 2106.04149 (v1 2021-06-08) | `torch.nn.CrossEntropyLoss(label_smoothing=0.1)`; `timm.loss.LabelSmoothingCrossEntropy`. **timm `BinaryCrossEntropy` defaults to `smoothing=0.1`** | permissive |
| **EMA of weights** | Generalisation, robustness to noisy labels, prediction consistency, calibration and transfer. Needs less LR decay. A stable teacher for semi-supervised learning and self-distillation | Horizon too long for the run: EMA stays near the initial weights and scores below the raw model (see pitfall). BN buffers must be averaged or copied consistently | arXiv 2411.18704 (v1 2024-11-27); batch-size scaling rule for EMA arXiv 2307.13813 (v1 2023-07-25) | `timm.utils.ModelEmaV3(model, decay=0.9999, min_decay=0.0, update_after_step=0, use_warmup=False, warmup_gamma=1.0, warmup_power=0.667, device=None, foreach=True, exclude_buffers=False)`. With `use_warmup=True`, `decay_t = 1 - (1 + t/warmup_gamma)^(-warmup_power)`, capped at `decay` (source read). torch: `AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=0.999), use_buffers=True)` | permissive |
| **SWA** (stochastic weight averaging) | Better generalisation and flatter solutions at almost no cost. Also improves runs along a constant-LR trajectory. SWAG adds uncertainty estimates | Forgetting `update_bn` for BatchNorm nets leaves stale statistics. LayerNorm/ViT does not need it (reasoning) | arXiv 1803.05407 (v1 2018-03-14); SWAG arXiv 1902.02476 (v1 2019-02-07); arXiv 2405.18392 (v1 2024-05-28) | `torch.optim.swa_utils.AveragedModel(model, device=None, avg_fn=None, multi_avg_fn=None, use_buffers=False)`, `SWALR(optimizer, swa_lr, anneal_epochs=10, anneal_strategy="cos")`, `update_bn(loader, model, device=None)` | permissive |
| **Model soups** (average the weights of several fine-tunes from one initialisation) | Free gain after a hyperparameter sweep, with no extra inference cost. Greedy soup: 81.03% against 80.38% for the best single model (CLIP ViT-B/32, ImageNet) | Uniform soup (79.97%) loses to the best single model when some ingredients are poor. There is "an error barrier between fine-tuned models" at high LR. Fine-tunes must share the same pretrained initialisation | arXiv 2203.05482 (v1 2022-03-10), Recipe 1 and Table 3 *(paper body)* | No package. Average `state_dict`s; greedy: sort by held-out accuracy and add an ingredient only if held-out accuracy improves. Reference code mlfoundations/model-soups (MIT, pushed 2024-07-15) | MIT |
| **WiSE-FT** | Robustness under shift for zero-shot models; covered in R3 | - | arXiv 2109.01903 (v1 2021-09-04) (see R3) | Interpolate state_dicts | - |

**Pitfalls (A2)**
- **EMA horizon arithmetic.** The weight left on the initial weights after t steps is decay^t. With `decay=0.9999`, 0.9999^2000 ≈ 0.82, so a 2,000-step fine-tune leaves the EMA about 82% at its starting point (computed). Use `ModelEmaV3(use_warmup=True)`, or a decay near 1 - 1/(0.1 x total steps) (practice), and **select the EMA-or-raw choice on validation, not on test**.
- Weight decay applied through `model.parameters()` in torch `AdamW` decays everything, including norms and tokens. Use timm param groups.
- Label smoothing interacts with later stages: it hurts distillation teachers, selective prediction and abstention (B5), and calibration measurement. Decide it together with those stages.
- `update_bn` runs a full pass over the loader in train mode. Passing an augmented loader changes the statistics (reasoning), so use the training loader without heavy augmentation.

### A3. Losses

**Classification**

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Cross-entropy vs BCE** (one-vs-all targets with Mixup/CutMix) | BCE: with Mixup/CutMix, the mixed classes are treated as all present. ResNet Strikes Back: BCE "slightly outperforms" CE at the best settings. DeiT III: BCE gives "a significant improvement" for larger ViTs on ImageNet-1k *(paper bodies)* | DeiT III kept CE for ImageNet-21k (BCE did not work as well). SGD + BCE is hard to converge (ResNet Strikes Back) | arXiv 2110.00476 (v1 2021-10-01); arXiv 2204.07118 (v1 2022-04-14) | `timm.loss.BinaryCrossEntropy(smoothing=0.1, target_threshold=None, weight=None, reduction="mean", sum_classes=False, pos_weight=None)`; `timm.loss.SoftTargetCrossEntropy` (for Mixup targets); `torch.nn.BCEWithLogitsLoss(weight=None, reduction="mean", pos_weight=None)` | permissive |
| **Focal loss** | Dense detection with extreme foreground-background imbalance. In classification it improves calibration | `torchvision.ops.sigmoid_focal_loss` defaults to `reduction="none"`, so it returns per-element losses. `alpha` weights the positives | arXiv 1708.02002 (v1 2017-08-07); calibration arXiv 2002.09437 (v1 2020-02-21) | `torchvision.ops.sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction="none")`; smp `FocalLoss`; kornia `FocalLoss`, `BinaryFocalLossWithLogits`; MONAI `FocalLoss` | permissive |
| **Class-balanced loss** (effective number) | Long-tailed classification with a from-scratch or weak backbone. Effective number = (1 - β^n)/(1 - β); each class is weighted by its inverse | β is a hyperparameter. Re-weighting early in training can hurt representation learning (see LDAM-DRW's deferred schedule) | arXiv 1901.05555 (v1 2019-01-16) | No package in the environment. Pass weights to `CrossEntropyLoss(weight=...)` | - |
| **LDAM + DRW** (label-distribution-aware margin, deferred re-weighting) | Long tail: larger margins for rare classes, with re-weighting deferred until late in training | Extra schedule and hyperparameters | arXiv 1906.07413 (v1 2019-06-18) | Official repo kaidic/LDAM-DRW (licence not verified); a few lines to implement | - |
| **Logit adjustment / balanced softmax** | Long tail and prior shift. Add τ·log(prior) to the logits in the loss, or subtract it after training. Statistically grounded and simple | Wrong priors (for example the train prior when deployment differs) move errors elsewhere | arXiv 2007.07314 (v1 2020-07-14); arXiv 2007.10740 (v1 2020-07-21) | A one-line change to the logits (no package needed) | - |
| **Asymmetric loss (ASL)** | Multi-label with many negatives per image (positive-negative imbalance) | Its three hyperparameters interact with the decision thresholds | arXiv 2009.14119 (v1 2020-09-29) | `timm.loss.AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-08)`; `AsymmetricLossSingleLabel` | Apache-2.0 |
| **PolyLoss (Poly-1)** | CE + ε(1 - p_t), one extra hyperparameter. "Outperforms the cross-entropy loss and focal loss" on classification, segmentation and detection | The best ε "is indeed dependent on the task and dataset" | arXiv 2204.12511 (v1 2022-04-26) | A few lines (no package) | - |

**Detection** (losses live inside detector codebases; change them only with the codebase's own recipe)

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **GIoU / DIoU / CIoU box losses** | Box regression that stays informative when boxes do not overlap (GIoU). Faster convergence (DIoU and CIoU add centre distance and aspect ratio) | Boxes must be in `xyxy` format. Losses default to `reduction="none"` | arXiv 1902.09630 (v1 2019-02-25); arXiv 1911.08287 (v1 2019-11-19) | `torchvision.ops.generalized_box_iou_loss(boxes1, boxes2, reduction="none", eps=1e-07)`, `distance_box_iou_loss`, `complete_box_iou_loss`; MONAI `BoxGIoULoss` | BSD |
| **Varifocal / Generalized Focal + Distribution Focal Loss (DFL)** | IoU-aware classification scores (VFL). Box edges as distributions (DFL); used by modern one-stage detectors | Tied to their head designs | arXiv 2008.13367 (v1 2020-08-31); arXiv 2006.04388 (v1 2020-06-08) | mmdet 3.3.0 (last release 2024-01-05, stale) and the D-FINE/DEIM repos (R4) | Apache-2.0 |
| **Matching-aware losses in DETRs** (Hungarian matching; denoising; stable matching; IoU-aware query selection; FDR; Dense O2O + MAL) | One-to-one matching removes NMS. DEIM: Dense O2O + MAL "reducing training time by 50%" when added to RT-DETR and D-FINE | Sparse positives slow convergence (the reason DEIM exists). Loss weights are recipe-specific | DETR arXiv 2005.12872 (v1 2020-05-26); DINO arXiv 2203.03605 (v1 2022-03-07); Stable-DINO arXiv 2304.04742 (v1 2023-04-10); Align-DETR arXiv 2304.07527 (v1 2023-04-15); RT-DETR arXiv 2304.08069 (v1 2023-04-17); D-FINE arXiv 2410.13842 (v1 2024-10-17); DEIM arXiv 2412.04234 (v1 2024-12-05) | Inside the detector repos (licences in R4; DEIMv2 is NC) | varies |

**Segmentation**

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Dice / generalised Dice / Dice + CE** | Foreground-background imbalance. Compound CE + Dice is the common robust default (nnU-Net uses Dice + CE; practice) | Soft Dice is unstable on empty masks without smoothing, and per-image Dice on empty images is undefined | V-Net arXiv 1606.04797 (v1 2016-06-15); GDL arXiv 1707.03237 (v1 2017-07-11); taxonomy arXiv 2005.13449 (v1 2020-05-27) | smp `DiceLoss(mode, classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07)`; MONAI `DiceLoss`, `DiceCELoss`, `DiceFocalLoss`, `GeneralizedDiceLoss`, `GeneralizedDiceFocalLoss`; kornia `DiceLoss` | MIT / Apache-2.0 |
| **Lovász-softmax** | Directly optimises IoU (a Jaccard surrogate), which often helps mIoU late in training | Needs per-image versus per-batch care (`per_image`) | arXiv 1705.08790 (v1 2017-05-24) | smp `LovaszLoss(mode, per_image=False, ignore_index=None, from_logits=True)`; kornia `LovaszSoftmaxLoss`, `LovaszHingeLoss` | MIT / Apache-2.0 |
| **Boundary loss / Hausdorff losses** | Highly unbalanced, thin or small structures | Needs distance maps; usually ramped in alongside Dice | arXiv 1812.07032 (v1 2018-12-17) | MONAI `BoundaryLoss`, `HausdorffDTLoss`, `LogHausdorffDTLoss` (MONAI dev branch listing); kornia `HausdorffERLoss` | Apache-2.0 |
| **Tversky / focal Tversky / unified focal** | Recall-precision trade-off for small lesions (α, β) | More hyperparameters | arXiv 1706.05721 (v1 2017-06-18); arXiv 1810.07842 (v1 2018-10-18); arXiv 2102.04525 (v1 2021-02-08) | smp `TverskyLoss(mode, ..., alpha=0.5, beta=0.5, gamma=1.0)`; MONAI `TverskyLoss`, `AsymmetricUnifiedFocalLoss`; kornia `TverskyLoss` | MIT / Apache-2.0 |
| **clDice** | Tubular and topological structures (vessels, roads) | Only meaningful for thin connected structures | arXiv 2003.07311 (v1 2020-03-16) | MONAI `SoftclDiceLoss`, `SoftDiceclDiceLoss` | Apache-2.0 |

**Ordinal regression and metric learning**

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **CORAL / CORN** (rank-consistent ordinal) | Ordered grades (severity, age bins) where plain CE ignores the order | Assumes the grades are truly ordinal | arXiv 1901.07884 (v1 2019-01-20); arXiv 2111.08851 (v1 2021-11-17) | `coral-pytorch` 1.4.0 (2022-07-18, stale) | MIT |
| **ArcFace / CosFace / Sub-center ArcFace** | Retrieval, re-ID and face-like verification with many identities | Margins need tuning; much claimed progress in the field is "marginal at best" once methodology is fixed | arXiv 1801.07698 (v1 2018-01-23); arXiv 1801.09414 (v1 2018-01-29); reality check arXiv 2003.08505 (v1 2020-03-18) | `pytorch-metric-learning` 2.9.0 (2025-08-17): `ArcFaceLoss`, `SubCenterArcFaceLoss`, `CosFaceLoss`, `ProxyAnchorLoss`, `MultiSimilarityLoss` (names from `losses/__init__.py`) | MIT |
| **SupCon / triplet** | Representation learning with labels; re-ID with batch-hard mining | Sensitive to batch composition and mining | arXiv 2004.11362 (v1 2020-04-23); arXiv 1703.07737 (v1 2017-03-22) | `SupConLoss`, `TripletMarginLoss`, plus the library's miners | MIT |

**Pitfalls (A3)**
- Several torchvision losses default to `reduction="none"`; calling `.backward()` on a non-scalar raises an error, and a sum where a mean was intended changes the effective LR.
- CE with `ignore_index=-100` (torch default) against a dataset that uses 255 for "ignore" counts void pixels as class 255 or crashes. Set it explicitly (practice; the torch default was verified).
- Class weights and logit adjustment change calibration and thresholds. Recalibrate (B3) and re-select thresholds (B4) after changing the loss.
- For detection and DETR models, swapping only the loss without the matching and assignment code breaks the recipe (reasoning).

### A4. Class imbalance and long tail

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Oversampling / class-aware sampling** | CNN classification under imbalance. Oversampling "emerged as dominant in almost all analyzed scenarios", should fully remove the imbalance, and did not cause CNN overfitting in that study | Repeats rare images many times, so heavy augmentation is needed (reasoning). Changes the training prior, so recalibrate (B3) | arXiv 1710.05381 (v1 2017-10-15) | `torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)` (verified) | permissive |
| **Repeat factor sampling (RFS)** for detection and instance segmentation | LVIS-style long-tailed detection. Category factor r(c) = max(1, sqrt(t / f(c))), where f(c) is the fraction of images containing c; the image factor is the max over its categories; fractional parts use stochastic rounding (detectron2 source) | Rare classes co-occurring with common ones also repeat the common ones (reasoning). The threshold t is a hyperparameter (LVIS used 0.001; paper body, not re-verified) | LVIS arXiv 1908.03195 (v1 2019-08-08) | detectron2 `RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh, sqrt=True)` (source read; detectron2 is not on PyPI). For other frameworks, reimplement the formula | Apache-2.0 (detectron2 repo; not re-checked by API) |
| **Re-weighting** (inverse frequency, effective number) | From-scratch or weak features | Destabilises early training; defer it (DRW) | arXiv 1901.05555 (v1 2019-01-16); arXiv 1906.07413 (v1 2019-06-18) | `CrossEntropyLoss(weight=...)` | - |
| **Decoupled training: cRT / τ-normalisation** | Learn features with natural (instance-balanced) sampling, then retrain or re-normalise only the classifier. "Data imbalance might not be an issue in learning high-quality representations" | Needs a second stage and balanced sampling for the head | arXiv 1910.09217 (v1 2019-10-21) | Manual: freeze the backbone, retrain the head with class-balanced sampling (cRT), or scale each class weight vector by 1/‖w‖^τ (τ-norm) | - |
| **Weight balancing** (tuned weight decay + MaxNorm, two stages) | Long tail with a simple recipe; weight decay alone "significantly improves over prior work" | Tuning weight decay costs time | arXiv 2203.14197 (v1 2022-03-27) | Optimizer weight decay plus a MaxNorm constraint on the classifier | - |
| **Logit adjustment / balanced softmax** | Post-hoc or in the loss; firmer statistical grounding | Needs the right deployment prior (B4) | arXiv 2007.07314 (v1 2020-07-14); arXiv 2007.10740 (v1 2020-07-21) | A few lines | - |
| **Lightweight fine-tuning of foundation models (LIFT)** | Long tail with a pretrained foundation model: "heavy fine-tuning may even lead to non-negligible performance deterioration on tail classes", while "lightweight fine-tuning is more effective" | Full fine-tuning with a high LR | arXiv 2309.10019 (v1 2023-09-18) | peft LoRA or adapters (R3); official code shijxcs/LIFT (licence not verified) | - |
| **Detection long-tail losses** (Seesaw, federated loss) | Large-vocabulary detection | Framework-specific | arXiv 2008.10032 (v1 2020-08-23); CenterNet2 federated loss arXiv 2103.07461 (v1 2021-03-12); survey arXiv 2110.04596 (v1 2021-10-09) | mmdet and detectron2 projects | Apache-2.0 |
| **Two-branch or cumulative schedules (BBN)** | Research comparisons | More complex than decoupling or logit adjustment for similar goals (reasoning) | arXiv 1912.02413 (v1 2019-12-05) | Official repo | - |

**Decision guidance (A4)**
1. With a pretrained backbone, first try natural sampling with a linear probe or light fine-tuning, then **post-hoc logit adjustment or a re-balanced classifier retrain (cRT)**. These are cheap and need no retraining of the backbone.
2. If the tail is still weak, add class-aware sampling or LDAM-DRW. Keep fine-tuning light (LoRA or partial), because heavy full fine-tuning hurts tail classes (LIFT).
3. For detection, use RFS first. Use Seesaw or federated losses only inside frameworks that already implement them.
4. Always report macro metrics (balanced accuracy, macro-F1, per-class AP) with per-class tables, and fix the operating point on validation (B4).

**Pitfalls (A4)**
- Any re-sampling or re-weighting changes the implicit prior, so predicted probabilities no longer match deployment frequencies. Recalibrate, or apply a prior correction, before using probabilities or thresholds (B3, B4).
- Oversampling the training set **after** splitting is fine. Oversampling before splitting leaks duplicates across folds (R1).

### A5. Noisy labels during training

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Find and fix labels first (confident learning)** | Any dataset where relabelling is possible. Test sets matter most: at least 3.3% errors on average across 10 benchmarks, and at least 6% of ImageNet val. On noisy real-world data, "lower capacity models may be practically more useful" | Flags only candidates: 51% of flagged items were real errors on average. Needs out-of-fold probabilities | arXiv 1911.00068 (v1 2019-10-31); arXiv 2103.14749 (v1 2021-03-26) | cleanlab 2.9.0: `cleanlab.filter.find_label_issues(labels, pred_probs, *, filter_by="prune_by_noise_rate", return_indices_ranked_by=None, ...)`; `cleanlab.classification.CleanLearning(clf=None, *, cv_n_folds=5, ...)`; `cleanlab.object_detection` and `cleanlab.segmentation` modules (`filter`, `rank`, `summary`) | Apache-2.0 (PyPI and installed metadata) |
| **Label smoothing** | Moderate noise (competitive with loss correction) | High noise (use negative smoothing instead); selective prediction | arXiv 2003.02819 (v1 2020-03-05); arXiv 2106.04149 (v1 2021-06-08) | `CrossEntropyLoss(label_smoothing=...)` | permissive |
| **EMA of weights** | More robust to noisy labels than the last iterate | See A2 horizon pitfall | arXiv 2411.18704 (v1 2024-11-27) | timm `ModelEmaV3` | Apache-2.0 |
| **SAM** | Robustness to label noise "on par with" dedicated methods | 2x compute | arXiv 2010.01412 (v1 2020-10-03) | see A1 | MIT |
| **Early stopping / early-learning regularisation (ELR)** | Networks fit clean labels first and memorise noisy ones later. ELR exploits this with SSL-style targets | Needs a clean validation set to stop on | arXiv 1611.03530 (v1 2016-11-10); arXiv 2007.00151 (v1 2020-06-30) | ELR official repo (licence not verified) | - |
| **Robust losses: GCE, SCE** | Synthetic and class-conditional noise. GCE generalises MAE and CE, and MAE alone "can perform poorly with DNNs" | Real-world noise is instance-dependent, and results on synthetic noise overstate the gains | arXiv 1805.07836 (v1 2018-05-20); arXiv 1908.06112 (v1 2019-08-16); real noise arXiv 2110.12088 (v1 2021-10-22) | A few lines (no maintained package verified) | - |
| **Co-teaching / DivideMix** | High noise when relabelling is impossible. DivideMix splits clean and noisy samples with a mixture model and trains semi-supervised | Two networks (2x compute), many hyperparameters; results mostly on CIFAR with synthetic noise | arXiv 1804.06872 (v1 2018-04-18); arXiv 2002.07394 (v1 2020-02-18); survey arXiv 2007.08199 (v1 2020-07-16) | Official repos | - |
| **Vision-language noise detectors (DeFT)** | Fine-tuning with a CLIP-like model whose text-image alignment can sieve out noisy labels | Needs a VLM and class names that text can describe | arXiv 2409.19696 (v1 2024-09-29) | Paper code | - |

**Decision guidance (A5)**
- Label cleaning beats clever losses when labels can be fixed. Run confident learning on out-of-fold probabilities from a frozen-feature probe (R1 and the cv-data-audit scripts), review the flagged items, and **relabel the test and validation sets by hand first**, because noisy evaluation labels change which model wins.
- Then train normally, with EMA and mild label smoothing (unless selective prediction or distillation follows), and stop on a clean validation set.
- Reach for GCE/SCE or DivideMix only if noise stays high and cannot be relabelled, and show the gain on a clean-labelled subset.

### A6. Semi-supervised learning, self-training, domain adaptation and test-time adaptation

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Strong supervised baseline first** | Always. "The performance of simple baselines which do not use unlabeled data is often underreported" | SSL can "degrade substantially when the unlabeled dataset contains out-of-class examples" | arXiv 1804.09170 (v1 2018-04-24) | - | - |
| **Self-supervised pre-training, then fine-tuning, then distillation** (task-agnostic use of unlabelled data) | Few labels plus many in-domain unlabelled images. "The fewer the labels, the more this approach ... benefits from a bigger network" | Pre-training compute (see R3 for continued SSL on domain data) | arXiv 2006.10029 (v1 2020-06-17) | R3 (DINOv2 code, lightly) | varies |
| **FixMatch / FlexMatch / FreeMatch** (confidence-thresholded pseudo-labels plus consistency) | Classification with a large in-distribution unlabelled pool | Out-of-class unlabelled images; confirmation bias; many epochs | arXiv 2001.07685 (v1 2020-01-21); arXiv 2110.08263 (v1 2021-10-15); arXiv 2205.07246 (v1 2022-05-15) | USB / `semilearn` 0.3.2 (2023-11-10, stale on PyPI; repo microsoft/Semi-supervised-learning MIT, pushed 2026-07-26). USB fine-tunes pretrained ViTs rather than training from scratch | MIT |
| **Mean Teacher** | EMA teacher for consistency and pseudo-labels; the basis of detection SSL | Needs the EMA scaling rule when batch size changes | arXiv 1703.01780 (v1 2017-03-06); arXiv 2307.13813 (v1 2023-07-25) | timm `ModelEmaV3` | - |
| **Noisy Student** (iterative self-training, equal-or-larger student, noised) | Abundant unlabelled data. ImageNet 88.4%, with large robustness gains (ImageNet-A 61.0 to 83.7) | 300M unlabelled images and heavy compute in the paper | arXiv 1911.04252 (v1 2019-11-11) | - | - |
| **Detection SSL: Unbiased Teacher, Soft Teacher, Semi-DETR** | Detection with many unlabelled images | Framework-bound (mmdet/detectron2 era), GPU-heavy | arXiv 2102.09480 (v1 2021-02-18); arXiv 2106.09018 (v1 2021-06-16); arXiv 2307.08095 (v1 2023-07-16) | Official repos | varies |
| **AdaBN** (recompute BN statistics on the target domain) | Parameter-free domain adaptation for BN networks | No effect on LayerNorm/ViT; needs enough target images per statistic | arXiv 1603.04779 (v1 2016-03-15) | Run forward passes in train mode on target data without gradients; `torch.optim.swa_utils.update_bn(target_loader, model)` does exactly this | permissive |
| **Test-time adaptation (TENT, EATA, CoTTA, SAR, MEMO)** | Known corruption shift with large, single-domain test batches (TENT: new SOTA on ImageNet-C) | Mixed shifts, small batches and imbalanced online labels (SAR). Collapse over long streams (RDumb: all but one method worse than no adaptation). Hyperparameter and model selection "exceedingly difficult" (TTAB). Under compute constraints, slower methods lose (SHOT from 2020 beats SAR) | arXiv 2006.10726 (v1 2020-06-18); arXiv 2204.02610 (v1 2022-04-06); arXiv 2203.13591 (v1 2022-03-25); arXiv 2302.12400 (v1 2023-02-24); arXiv 2110.09506 (v1 2021-10-18); benchmarks arXiv 2307.03133 (v1 2023-07-06), arXiv 2306.03536 (v1 2023-06-06), arXiv 2306.05401 (v1 2023-06-08), arXiv 2304.04795 (v1 2023-04-10) | Benchmark repos (TTAB, RDumb); no maintained PyPI package verified | - |

**Decision guidance (A6)**
- Order of preference with unlabelled images: (1) a stronger or in-domain pretrained backbone (R3); (2) pseudo-labelling or distillation with an EMA teacher; (3) FixMatch-family SSL only after (1) and (2), with a clean check that the unlabelled pool is in-distribution.
- For deployment shift, try AdaBN (BN nets) or a stronger foundation backbone before online TTA. If TTA is used, add resets (RDumb) and monitor it (E5).

### A7. Efficiency

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Mixed precision (fp16 + GradScaler, bf16)** | NVIDIA GPUs with tensor cores: less memory, faster | **On the project CPU, bf16 autocast was about 36x slower** (local measurement in the findings). fp16 without a scaler underflows | arXiv 1710.03740 (v1 2017-10-10) | `torch.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)`; `torch.amp.GradScaler(device="cuda", init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)` | permissive |
| **`channels_last`** | CNNs on CPU and GPU. Local: 845 to 672 ms per step (fp32, ResNet-18) | Little effect for ViTs (reasoning) | local measurement | `model.to(memory_format=torch.channels_last)`, `x.contiguous(memory_format=torch.channels_last)` | - |
| **`torch.compile`** | Repeated shapes on Linux, or Windows with a C++ compiler | Windows without MSVC `cl` / `clang-cl` / `icx-cl` (the project machine). Recompiles on dynamic shapes | PyTorch tutorial "How to use torch.compile on Windows CPU/XPU" | `torch.compile(model=None, *, fullgraph=False, dynamic=None, backend=None, mode=None, options=None, disable=False, recompile_limit=None, ...)` | permissive |
| **Gradient checkpointing** | Memory-bound training of large ViTs (trades compute for memory) | About one extra forward pass per step (reasoning); slower on CPU | arXiv 1604.06174 (v1 2016-04-21) | timm `model.set_grad_checkpointing(enable=True)` (verified on ViT). `torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=...)`: the docstring says "It is recommended that you use `use_reentrant=False`" and that it "should be passed explicitly" | permissive |
| **Data loading: DataLoader workers, cached features** | I/O- or decode-bound training | Windows uses spawn for workers, so the main guard and picklable datasets are required (practice). `num_workers=0` is the default | - | `DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, *, prefetch_factor=None, persistent_workers=False, in_order=True)` | permissive |
| **NVIDIA DALI** | GPU-side JPEG decode and augmentation on Linux servers | Not on Windows; needs an NVIDIA GPU (install docs: "Linux x64"; NVIDIA driver for CUDA 12.0+) | NVIDIA DALI installation docs | `nvidia-dali-cuda120` / `nvidia-dali-cuda130` 2.3.0 (2026-08-28) | Apache-2.0 |
| **FFCV** | Fast loading from a preprocessed file format | Stale: PyPI 1.0.2 (2023-03-05), repo last push 2024-06-16 | arXiv 2306.12517 (v1 2023-06-21) | `ffcv` 1.0.2 | Apache-2.0 |
| **TorchCodec** | Video decoding (torchvision video I/O was removed; see R1) | - | - | `torchcodec` 0.16.0 (2026-08-13) | not stated on PyPI (pytorch/torchcodec repo; licence not checked) |
| **DDP / FSDP2** | Multi-GPU; FSDP shards parameters, gradients and optimizer state for large models | Irrelevant on a single CPU machine | FSDP arXiv 2304.11277 (v1 2023-04-21) | `torch.nn.parallel.DistributedDataParallel`; FSDP2 `torch.distributed.fsdp.fully_shard(module, *, mesh=None, reshard_after_forward=None, mp_policy=MixedPrecisionPolicy(...), offload_policy=OffloadPolicy(), ignored_params=None)` (verified present in 2.14) | permissive |

**CPU-only training recipe for the cv-skills machine**
- Cache frozen features, then use a linear probe or k-NN (R3 measured throughput: ViT-S at 224 is about 27 img/s).
- Fine-tune only small CNNs, or ViT-S with LoRA.
- Use fp32 with `channels_last`, no bf16 autocast, and no `torch.compile` unless a compiler is installed.
- Set threads explicitly (`torch.set_num_threads`) because other jobs share the CPU.
- Use ONNX Runtime for inference: locally, ResNet-18 at batch 1 and 224 px had a p50 of 31 ms in ONNX Runtime against 62 ms in PyTorch eager (E4).

### A8. Hyperparameter search and seed variance

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **TPE (Optuna default)** | Mixed and conditional spaces with tens to hundreds of trials. In Optuna 5.0, `TPESampler` "now enables multivariate TPE and constant liar strategy by default" (release notes) | Fewer than about `n_startup_trials=10` trials is effectively random search | Optuna arXiv 1907.10902 (v1 2019-07-25); TPE components arXiv 2304.11127 (v1 2023-04-21) | optuna 5.0.0 (2026-09-07): `TPESampler(*, n_startup_trials=10, n_ei_candidates=24, seed=None, multivariate=None, group=False, constant_liar=True, ...)`; `optuna.create_study(*, storage=None, sampler=None, pruner=None, direction=None, ...)`. Samplers: `TPESampler`, `GPSampler` (stable in 5.0), `CmaEsSampler`, `NSGAIISampler`, `NSGAIIISampler`, `QMCSampler`, `RandomSampler`, `GridSampler`, `BruteForceSampler`, `PartialFixedSampler` | MIT |
| **Hyperband / ASHA / successive halving (pruning)** | Many cheap configurations with learning curves (epochs as the resource) | Early rankings flip for fine-tuning with warmup, or when LR schedules differ in length (reasoning). Needs intermediate `trial.report` / `should_prune` | arXiv 1603.06560 (v1 2016-03-21); arXiv 1810.05934 (v1 2018-10-13) | Optuna pruners: `HyperbandPruner(min_resource=1, max_resource="auto", reduction_factor=3, bootstrap_count=0)`, `SuccessiveHalvingPruner`, `MedianPruner`, `PercentilePruner`, `PatientPruner`, `ThresholdPruner`, `WilcoxonPruner`. Ray Tune (ray 2.58.0): `ASHAScheduler`, `HyperBandScheduler`, `HyperBandForBOHB` | MIT / Apache-2.0 |
| **Population-based training (PBT)** | Many parallel workers, and schedules that should adapt mid-run | Needs parallel compute; hard to reproduce | arXiv 1711.09846 (v1 2017-11-27) | Ray Tune `PopulationBasedTraining`, `PB2` (names from `tune/schedulers/__init__.py`) | Apache-2.0 |
| **Tune the plain baseline** (LR, weight decay, batch size on a separate validation split) | Always first. It beat all but one of eight specialised data-efficient methods | - | arXiv 2108.13122 (v1 2021-08-30) | - | - |
| **Seeds and repeated runs** | Deciding whether a change is real | Single-seed comparisons: "it is surprisingly easy to find an outlier that performs much better or much worse than the average" | arXiv 2109.08203 (v1 2021-09-16); test-set versus distribution variance arXiv 2304.01910 (v1 2023-04-04); variance sources and cheaper estimators (randomise more sources: 51x compute reduction) arXiv 2103.03098 (v1 2021-03-01); fine-tuning seeds (NLP): init and data order contribute comparably, and many small-data runs diverge, arXiv 2002.06305 (v1 2020-02-15) | `seed` arguments; `torch.use_deterministic_algorithms(True)` for bit-reproducibility (practice) | - |
| **Guard against overtuning** | Heavy HPO on small validation sets | In about 10% of cases, HPO selects a configuration worse than the default or first one tried | arXiv 2506.19540 (v1 2025-06-24) | - | - |

**How many trials.** Random search with n trials hits the top q fraction of the space with probability 1 - (1 - q)^n. For q = 5%, n = 59 gives 95.2% (computed). Beyond roughly 30-60 trials on a small validation set, gains are mostly selection noise (see overtuning), unless the validation set is large or repeated CV is used (reasoning plus arXiv 2506.19540).

**Decision guidance (A8)**
1. Search LR (log scale), weight decay, epochs and augmentation strength first; the rest later.
2. Use TPE with the Optuna 5 defaults and a pruner only when epochs are the resource and learning curves are comparable.
3. Re-run the top 2-3 configurations with 3 or more seeds, and compare them paired, on the same folds and the same seeds (D2).
4. Never tune on the test set; freeze the search before the final evaluation.

### A9. Early stopping and checkpoint selection

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Select the epoch on an inner validation split; report on a separate fold or test set** | Always when selecting a checkpoint | Selecting the best epoch on the fold you report inflates the score. Low-variance selection criteria matter because "the degradation in performance due to over-fitting the model selection criterion can be surprisingly large" | Cawley & Talbot, JMLR 11(70):2079-2107, 2010 (jmlr.org/papers/v11/cawley10a.html); model selection must be part of the method (DomainBed) arXiv 2007.01434 (v1 2020-07-02) | Project `compare.py` scores the last epoch by default (STATUS.md) | - |
| **Fixed epoch budget (or last checkpoint), EMA weights** | Small validation sets: no selection noise; EMA flattens the late-epoch curve | Under-training if the budget is set badly; pick it with inner CV | arXiv 2411.18704 (v1 2024-11-27) | timm `ModelEmaV3` | - |
| **Flat versus nested CV** | Flat CV picks algorithms of similar quality when there are few hyperparameters (the flat estimate is still biased) | Heavy tuning needs nested CV or a held-out test for an unbiased estimate | arXiv 1809.09446 (v1 2018-09-25) | sklearn `GridSearchCV` inside `cross_val_score` (practice) | BSD |
| **Early stopping for noisy labels** | Stops before noisy labels are memorised | Needs a clean validation set | arXiv 1611.03530 (v1 2016-11-10); arXiv 2007.00151 (v1 2020-06-30) | - | - |

**Pitfalls (A9)**
- Test sets that are reused for years are adaptively overfitted. ImageNet-v2 showed drops of 11-14% on a new test set, attributed mainly to harder images rather than adaptivity (arXiv 1902.10811 (v1 2019-02-13)), so both effects exist.
- Keep one untouched test set per project and log every look at it.
- The EMA-or-raw choice, TTA on or off, the calibration method and the thresholds are all "checkpoint selection". Make each choice on validation or out-of-fold data, never on the reported test.

---

## B. Post-processing

### B1. Non-maximum suppression and output filtering (detection)

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Class-wise NMS** (standard) | Default for anchor-based and anchor-free one-to-many detectors; different classes may overlap (a rider on a horse) | Duplicate boxes with different labels on the same object survive (reasoning) | (standard) | `torchvision.ops.batched_nms(boxes, scores, idxs, iou_threshold)`, where `idxs` holds the class ids | BSD |
| **Class-agnostic NMS** | Classes that are mutually exclusive in space (fine-grained look-alikes) and suppress each other | Removes true overlapping objects of different classes (reasoning) | (standard) | `torchvision.ops.nms(boxes, scores, iou_threshold)` | BSD |
| **Soft-NMS** | Crowded scenes and overlapping same-class objects: decays scores instead of deleting boxes. +1.7% on VOC (R-FCN, Faster R-CNN) and +1.3% / +1.1% on COCO | More low-score boxes survive, which needs a score floor; slower in Python | arXiv 1704.04503 (v1 2017-04-14) | Not in torchvision. `ensemble_boxes.soft_nms(boxes, scores, labels, method=2, iou_thr=0.5, sigma=0.5, thresh=0.001, weights=None)` | MIT |
| **DIoU-NMS** | Occluded neighbours: uses centre distance in the suppression criterion | Needs a custom implementation | arXiv 1911.08287 (v1 2019-11-19) | Not in torchvision 0.29 (op list verified); implement with `torchvision.ops.distance_box_iou` | - |
| **Matrix NMS** | Instance masks: parallel, one-shot NMS over masks | Framework-specific (SOLO) | arXiv 2003.10152 (v1 2020-03-23) | mmdet (Apache-2.0) | Apache-2.0 |
| **Adaptive or learned NMS** | Very crowded pedestrians | Extra network or head | arXiv 1904.03629 (v1 2019-04-07); arXiv 1705.02950 (v1 2017-05-08) | Research code | - |
| **NMS-free outputs (DETR family, YOLOv10 dual assignment)** | One-to-one matching yields one box per object, so no NMS is needed and latency is stable | Adding NMS on top is unnecessary; its effect is not verified here | DETR arXiv 2005.12872 (v1 2020-05-26); RT-DETR arXiv 2304.08069 (v1 2023-04-17); YOLOv10 arXiv 2405.14458 (v1 2024-05-23) | Take the top-k queries by score (framework postprocessors) | see R4 |
| **Score threshold, top-k and max detections** | Deployment operating points (per-class thresholds chosen on validation, B4) | **Thresholding before computing mAP truncates the PR curve and lowers AP** (reasoning). COCOeval caps at `maxDets` 100 (R4) | R4 metric pitfalls | `torchmetrics.detection.MeanAveragePrecision(max_detection_thresholds=[1, 10, 100])` (3 values) | - |

**Decision guidance (B1).** Evaluate mAP on raw top-k outputs, with a very low score threshold and class-wise NMS at the framework default. For deployment, choose the NMS IoU and per-class score thresholds on validation for the operating point that matters (B4). Use Soft-NMS or DIoU-NMS only when crowding is the documented error mode (TIDE, D3).

### B2. Box and mask fusion (ensembles and TTA)

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Weighted Boxes Fusion (WBF)** | Ensembles of different detectors, and multi-scale or flip TTA: averages the coordinates of clustered boxes weighted by confidence; used in top Open Images and COCO challenge results | **Coordinates must be normalised to [0, 1].** Pixel boxes are clipped to 1 and dropped as zero-area, with only warnings (verified locally). On a single model's raw output (many overlapping low-quality boxes), fusion is not NMS; run NMS per model first (practice) | arXiv 1910.13302 (v1 2019-10-29) | `ensemble-boxes` 1.0.9 (2022-04-20; repo ZFTurbo/Weighted-Boxes-Fusion MIT, pushed 2026-07-27): `weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type="avg", allows_overflow=False)`; also `weighted_boxes_fusion_experimental`, `weighted_boxes_fusion_1d`, `weighted_boxes_fusion_3d` | MIT |
| **Non-maximum weighted (NMW) / NMS across models** | Simpler fusion baselines | Usually weaker than WBF for ensembles (WBF paper; not re-verified) | arXiv 1910.13302 (v1 2019-10-29) | `ensemble_boxes.non_maximum_weighted(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0)`; `ensemble_boxes.nms(boxes, scores, labels, iou_thr=0.5, weights=None)` | MIT |
| **Probability or mask averaging (semantic segmentation)** | Averaging softmax maps across models, folds or TTA views before the argmax (the nnU-Net ensembling style; practice) | Averaging hard masks (argmax first) loses information (reasoning); logits from different models are on different scales | (practice) | numpy/torch mean of probabilities; nnU-Net `nnUNetv2_ensemble` (not re-verified) | - |
| **Instance mask fusion** | Match instances across models by mask IoU, then average the soft masks and scores (WBF-style) | No maintained library verified; matching thresholds matter | (reasoning) | Implement with pycocotools `maskUtils.iou` (R4) | - |

### B3. Calibration

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Temperature scaling (TS)** | Overconfident neural classifiers with softmax logits. One parameter fitted on NLL, and accuracy is preserved: "on most datasets, temperature scaling ... is surprisingly effective" | Non-logit or discrete scores (k-NN vote fractions; see below). Lowers every confidence regardless of correctness. Distribution shift. Enlarges adaptive conformal sets. Recent non-convolutional models are already among the best calibrated | arXiv 1706.04599 (v1 2017-06-14); arXiv 2106.07998 (v1 2021-06-15); arXiv 2207.06211 (v1 2022-07-13); arXiv 1906.02530 (v1 2019-06-06); arXiv 2402.05806 (v1 2024-02-08) | scikit-learn 1.9.1 `CalibratedClassifierCV(estimator, *, method="temperature", cv=None, ensemble="auto")` ("Added option 'temperature'" in 1.8; it "optimizes the log loss"); use `sklearn.frozen.FrozenEstimator` for a pre-fitted model. netcal 1.4.0 `TemperatureScaling`. Or 10 lines of LBFGS on validation logits | BSD-3 / Apache-2.0 |
| **Platt (sigmoid) / vector / matrix scaling** | Binary tasks. Sigmoid "fits an additional intercept", which helps "very uncalibrated classifiers on very imbalanced datasets" (sklearn docstring) | Matrix scaling overfits with many classes (Guo et al. 2017, paper body; not re-verified) | arXiv 1706.04599 (v1 2017-06-14) | `CalibratedClassifierCV(method="sigmoid")` (the default; one-vs-rest plus renormalisation for multiclass) | BSD-3 |
| **Isotonic regression / histogram binning** | Enough calibration data; non-parametric. Histogram binning has measurable calibration error | sklearn: isotonic "is not recommended when the number of calibration samples is too low (≪1000)". Histogram binning needs O(B/ε²) samples | arXiv 1909.10155 (v1 2019-09-23) | `CalibratedClassifierCV(method="isotonic")`; netcal `IsotonicRegression`, `HistogramBinning`, `BBQ`, `ENIR` | BSD-3 / Apache-2.0 |
| **Scaling-binning calibrator** | Guarantees plus sample efficiency, O(1/ε² + B). 35% lower calibration error than histogram binning on CIFAR-10/ImageNet | Extra step; the package is stale | arXiv 1909.10155 (v1 2019-09-23) | `uncertainty-calibration` 0.1.4 (2022, stale) | MIT |
| **Dirichlet calibration** | Multiclass calibration beyond a single temperature | More parameters (regularise them) | arXiv 1910.12656 (v1 2019-10-28) | `dirichletcal` 0.5.3 (2025-06-12; requires Python 3.12 or later) | MIT |
| **Beta calibration** | Binary calibration with more flexible shapes than Platt | - | (AISTATS 2017; no arXiv verified) | netcal `BetaCalibration` | Apache-2.0 |
| **Sample-adaptive, parameterised and local TS (segmentation)** | Where a global temperature is too blunt; local TS for per-pixel segmentation calibration | More parameters and more data needed | arXiv 2207.06211 (v1 2022-07-13); arXiv 2102.12182 (v1 2021-02-24); arXiv 2008.05105 (v1 2020-08-12); arXiv 2506.23492 (v1 2025-06-30); decision-focused constrained TS (dermoscopy) arXiv 2406.11456 (v1 2024-06-17) | Paper code | - |
| **Mix-n-Match (ensemble and composition of calibrators)** | Small calibration sets: better data efficiency and expressiveness while preserving accuracy | - | arXiv 2003.07329 (v1 2020-03-16) | Paper code | - |
| **Detection calibration** (position- and scale-dependent; D-ECE) | Detector scores used as probabilities. Properly designed post-hoc Platt or isotonic calibration beats train-time calibration methods (D-DETR + isotonic beats Cal-DETR by more than 7 D-ECE on COCO) | TS for detectors has "notable drawbacks". D-ECE evaluation choices change the conclusions | arXiv 2004.13546 (v1 2020-04-28); arXiv 2207.01242 (v1 2022-07-04); arXiv 2405.20459 (v1 2024-05-30) | netcal 1.4.0: `LogisticCalibration(detection=True)`, `LogisticCalibrationDependent`, `BetaCalibrationDependent`; metric D-ECE (from the PyPI project description) | Apache-2.0 |

**ECE and other calibration metrics: estimator pitfalls**

| Estimator | Why it matters | Evidence | Library + verified API |
|---|---|---|---|
| Equal-width binned top-label ECE (the default everywhere) | Biased, and the ranking of recalibration methods "is drastically impacted by the choice of calibration measure". Histogram ECE "may provide misleading results especially in small-data regime" | arXiv 1904.01685 (v1 2019-04-02); arXiv 2003.07329 (v1 2020-03-16) | torchmetrics 1.9.0 `MulticlassCalibrationError(num_classes, n_bins=15, norm="l1", ignore_index=None)` (`norm` = `"l1"` gives ECE, `"max"` gives MCE, `"l2"` gives RMS). **No equal-mass option** (signature verified) |
| Equal-mass (adaptive) bins, ECE_sweep, debiased ECE | Equal-mass bins "have lower bias than ... equal width"; the reliable estimators are "the debiased estimator ... and ... ECE_sweep" | arXiv 2012.08668 (v1 2020-12-15); arXiv 1909.10155 (v1 2019-09-23) | `uncertainty-calibration` (debiased estimator); or implement with quantile bins |
| Class-wise and marginal calibration | "Conditioning on the class leads to more effective calibration evaluations" | arXiv 1904.01685 (v1 2019-04-02); top-label arXiv 2107.08353 (v1 2021-07-18) | Implement per class |
| Smooth ECE (kernel), consistent Lp estimators, proper calibration errors | Principled reliability diagrams; consistent estimation; upper bounds tied to proper scores | arXiv 2309.12236 (v1 2023-09-21); arXiv 2211.16886 (v1 2022-11-30); arXiv 2210.07810 (v1 2022-10-13); arXiv 2203.07835 (v1 2022-03-15); frameworks arXiv 1902.06977 (v1 2019-02-19), arXiv 1910.11385 (v1 2019-10-24) | `relplot` 1.0.3 (2025-11-04; Apple licence text: **custom**) |
| Proper scores: NLL and Brier | Reward sharpness and calibration together. NLL is unbounded; Brier is bounded | Grouping loss (calibration is "not enough") arXiv 2210.16315 (v1 2022-10-28) | sklearn `log_loss`, `brier_score_loss`, `d2_brier_score`, `d2_log_loss_score`; MONAI `CalibrationErrorMetric` (dev branch listing) |

**Why NLL-fitted TS improved NLL but worsened ECE and Brier on k-NN probabilities (the project's own test)**
- **Mechanism (reasoning).**
  - k-NN vote fractions are discrete (multiples of 1/k), and many misclassified images give the true class probability 0. To apply TS to them, one takes log(p + ε).
  - NLL is unbounded, so those few near-zero true-class probabilities dominate the objective. The NLL-optimal temperature therefore flattens every prediction.
  - The majority of correct, confident predictions become under-confident. Top-label ECE and Brier (both bounded, and dominated by the majority) get worse while NLL improves.
- **Supporting evidence:**
  - TS "reduces the individual confidences of the predictions irrespective of whether the classification ... is correct or incorrect" (arXiv 2207.06211).
  - A single temperature lacks expressive power (arXiv 2003.07329).
  - Minimising a proper loss yields calibration only when the predictor is locally optimal within a rich enough post-processing family (arXiv 2305.18764 (v1 2023-05-30)), and a single power applied to vote fractions is not such a family.
  - ECE is not a proper score, and calibration errors relate to different proper scores differently (arXiv 2203.07835).
- **What to do instead:**
  - Build the probability model into the k-NN: similarity-weighted votes `softmax(topk_sims / T)` over L2-normalised features (DINOv2's k-NN evaluation defaults: `temperature=0.07`, `nb_knn=[10, 20, 100, 200]`, source read). Fit T and k on out-of-fold NLL or Brier.
  - Or apply Laplace smoothing to the vote fractions before scaling (reasoning).
  - Or use a non-parametric calibrator (histogram binning, isotonic regression when there are at least about 1,000 calibration samples, or scaling-binning).
  - Choose the calibrator by the metric used for decisions: Brier or debiased/adaptive ECE for thresholded decisions, NLL for likelihoods. Always report NLL, Brier and an adaptive or debiased ECE with a reliability diagram.

**Decision guidance (B3)**
1. Fit calibrators on a calibration split, or on out-of-fold predictions, that is disjoint from both the threshold selection data and the test set. Use nested splits if the data is small.
2. For softmax networks, try TS first. Compare it against doing nothing (modern ViTs may already be calibrated), and against isotonic regression or Dirichlet when there are 1,000 or more calibration samples. Pick by Brier plus adaptive ECE, not by NLL alone.
3. For detectors, use Platt or isotonic regression on matched detections (netcal `detection=True`), and evaluate with the framework of arXiv 2405.20459.
4. Recalibrate after any change of loss weights, sampling, label smoothing, EMA, TTA or ensembling.

### B4. Thresholds, operating points and prior shift

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **F1-optimal threshold** (binary and per class for multi-label) | Reporting F1. With calibrated probabilities, "the optimal threshold is half the optimal F1 score" | An uninformative classifier's F1-optimal choice is to label everything positive. F1 is an improper measure for decisions (see below). Per-class thresholds for rare labels overfit small validation sets (reasoning) | arXiv 1402.1892 (v1 2014-02-08) | sklearn `TunedThresholdClassifierCV(estimator, *, scoring="balanced_accuracy", response_method="auto", thresholds=100, cv=None, refit=True, ...)` (binary); for multi-label, loop over labels on validation | BSD-3 |
| **Cost-based threshold** | Known misclassification costs. With calibrated probabilities the Bayes threshold is c_FP / (c_FP + c_FN) (decision theory; reasoning) | Needs calibrated probabilities and a correct prior | Van Calster et al.: "All classification measures (such as classification accuracy and F1) are improper for clinically relevant decisions"; F1 has neither of the two desired properties; use decision-analytic measures (net benefit) arXiv 2412.10288 (v1 2024-12-13) | `FixedThresholdClassifier(estimator, *, threshold="auto", pos_label=None, response_method="auto")` | BSD-3 |
| **Fixed-recall or fixed-specificity operating points** | Screening ("at least 95% sensitivity") | The threshold is estimated from few positives, so give a CI (D2) | - | torchmetrics 1.9.0: `BinaryRecallAtFixedPrecision`, `BinaryPrecisionAtFixedRecall`, `BinarySensitivityAtSpecificity`, `BinarySpecificityAtSensitivity`, plus `Multiclass*` / `Multilabel*` variants (names verified) | Apache-2.0 |
| **Prior-shift (label-shift) correction: EM with bias-corrected calibration (BCTS)** | Deployment class frequencies differ from training (screening versus referral populations). "Maximum likelihood with appropriate calibration is a formidable and efficient baseline", beating BBSL and RLLS | Assumes p(x given y) fixed (pure label shift); needs calibrated probabilities | arXiv 1901.06852 (v1 2019-01-21); BBSE arXiv 1802.03916 (v1 2018-02-12); RLLS arXiv 1903.09734 (v1 2019-03-22) | `abstention` 0.1.3.1 (kundajelab; 2020, stale); EM is about 15 lines: iterate the priors and reweight the posteriors by new prior / old prior | - |

**Decision guidance (B4).** Fix the operating point on validation or out-of-fold data, never on test. Report the threshold-free metric (AUROC/AP) and the operating-point metrics at that frozen threshold. When the deployment prevalence is known or estimated, correct the priors (EM-BCTS) before choosing thresholds.

### B5. Conformal prediction and selective prediction (abstention)

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Split conformal classification: LAC/THR, APS, RAPS** | Prediction sets with finite-sample marginal coverage (for example 90%) for any model. RAPS: sets "often factors of 5 to 10 smaller than a stand-alone Platt scaling baseline"; APS improves conditional coverage | Coverage is marginal, not per class or per site. Needs exchangeable calibration data (site or time shift breaks it). TS changes set sizes (arXiv 2402.05806) | arXiv 2006.02544 (v1 2020-06-03); arXiv 2009.14193 (v1 2020-09-29); tutorial arXiv 2107.07511 (v1 2021-07-15) | MAPIE 1.5.0 (2026-08-05): `mapie.classification.SplitConformalClassifier(estimator=LogisticRegression(), confidence_level=0.9, conformity_score="lac", prefit=True, n_jobs=None, verbose=0, random_state=None)` with `fit`, `conformalize`, `predict_set`, `predict`; scores `"lac"`, `"top_k"`, `"aps"`, `"raps"`. TorchCP 1.2.1 (2025-10-14): `SplitPredictor(score_function=LAC()/APS()/RAPS()/THR(), model=model, alpha=0.1)`, `.calibrate(cal_dataloader)`, `.predict(x)`, `.evaluate(test_dataloader)`. crepes 0.9.1 (2026-06-12); puncc 0.9.3 (2026-06-23) | MAPIE BSD-3; **TorchCP LGPL-3.0**; crepes BSD; puncc MIT |
| **TTA plus conformal** | Reduces set sizes by 10-14% on average with any conformal score | Adds TTA cost; TTA itself can hurt accuracy (B6) | arXiv 2505.22764 (v1 2025-05-28) | - | - |
| **Conformal risk control / Learn-then-Test (segmentation)** | Bound the expected false-negative rate of masks (a worked example in the paper) | Controls an expectation, not per-image guarantees | arXiv 2208.02814 (v1 2022-08-04); arXiv 2110.01052 (v1 2021-10-03) | Few lines on a calibration set | - |
| **Conformal sets for segmentation masks and instances** | Spatial guarantees (inner and outer mask sets), instance-level IoU sets, calibration of segmentation thresholds | Research code only | arXiv 2410.03406 (v1 2024-10-04); Kandinsky arXiv 2311.11837 (v1 2023-11-20); instance segmentation arXiv 2602.10045 (v1 2026-02-10) | Paper code | - |
| **Conformal boxes (detection)** | Box-coordinate uncertainty intervals | Research code | arXiv 2403.07263 (v1 2024-03-12) | Paper code | - |
| **Selective classification with max-softmax confidence (SR)** | Reject low-confidence cases with a risk guarantee. Top-5 ImageNet error of 2% guaranteed with probability 99.9% at about 60% coverage. Selecting with "the classification scores" beats dedicated selection heads | Label smoothing "consistently degrades" selective classification (fix with logit normalisation). Most calibration methods harm failure prediction | arXiv 1705.08500 (v1 2017-05-23); arXiv 2206.09034 (v1 2022-06-17); SelectiveNet arXiv 1901.09192 (v1 2019-01-26); arXiv 2403.14715 (v1 2024-03-19); arXiv 2303.02970 (v1 2023-03-06) | Threshold on max softmax probability (MSP), or on logit-normalised MSP | - |
| **Evaluate with AUGRC** (area under the generalised risk-coverage curve) | Multi-threshold evaluation of selective systems; it "substantially changes metric rankings on 5 out of the 6 data sets" | - | arXiv 2407.01032 (v1 2024-07-01) | Implement from the paper | - |

**Pitfalls (B5)**
- Use separate data for fitting calibration or TS, for conformal calibration, and for testing. Reusing data voids the guarantee.
- With grouped data (patients, sites), calibrate at the group level or per site; exchangeability fails across sites.
- Check per-class (Mondrian) coverage when classes are imbalanced.
- Report coverage and set size with CIs.

### B6. Test-time augmentation (TTA)

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Standard TTA (flips, crops, multi-scale; averaged)** | Sometimes: ResNet-18 on dermatology gained +1.6%. Flip TTA in segmentation (nnU-Net default mirroring: disabling it gives "Faster, but less accurate inference. Not recommended.") | **Degraded accuracy "consistently"** on 3 MedMNIST datasets across 4 architectures, by up to 31.6 pp (ResNet-18, pathology), and it "worsens with more augmented views". Mechanism: shift between augmented and training inputs, amplified by BatchNorm statistics mismatch. Intensity-only views preserve more accuracy than geometric views; adding the original image partly mitigates | arXiv 2604.09697 (v1 2026-04-06); nnU-Net predictor docstring | `ttach` 0.0.3 (2020-07-09, stale; repo qubvel/ttach MIT, pushed 2023-07-28), or a manual loop | MIT |
| **Learned TTA aggregation** | Weighting views instead of a simple average. Even when TTA improves net accuracy "it can change many correct predictions into incorrect predictions" | Needs a validation set to learn the weights | arXiv 2011.11156 (v1 2020-11-23); arXiv 2010.11422 (v1 2020-10-22); theory arXiv 2402.06892 (v1 2024-02-10) | Paper code | - |
| **TTA for detection with WBF** | Flip and multi-scale views fused with WBF (un-flip the boxes and normalise to [0, 1] first); small objects gain from up-scaled views (practice; tiling in R4) | Same accuracy risks as above; test on validation | arXiv 1910.13302 (v1 2019-10-29) | `ensemble_boxes.weighted_boxes_fusion` | MIT |
| **TTA for uncertainty** | Aleatoric uncertainty maps for segmentation | Cost scales with the number of views | arXiv 1807.07356 (v1 2018-07-19) | - | - |

**Decision guidance (B6).** Treat TTA as a candidate that must beat the single pass on validation: include the identity view, try intensity-only views first, and use a paired comparison with CIs (D2). Recalibrate after TTA (averaging changes confidence).

### B7. Segmentation refinement

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Dense CRF** | Legacy: coarse CNN outputs with poor boundaries (DeepLab v1/v2) | Modern networks do not need it: DeepLabv3 "significantly improves over our previous DeepLab versions without DenseCRF post-processing". Many hyperparameters; CPU-slow | arXiv 1210.5644 (v1 2012-10-20); arXiv 1606.00915 (v1 2016-06-02); arXiv 1706.05587 (v1 2017-06-17) | `pydensecrf` 1.0rc3 (2018-04-01, stale; repo lucasb-eyer/pydensecrf MIT, pushed 2024-03-05); fork `pydensecrf2` 1.1 (2024-12-23, MIT); `SimpleCRF` 0.2.1.1 (2021, BSD) | MIT / BSD |
| **SegFix / PointRend / CascadePSP** | Boundary quality (SegFix replaces boundary predictions with interior ones and is model-agnostic); high-resolution refinement (CascadePSP) | Needs an extra trained model | arXiv 2007.04269 (v1 2020-07-08); arXiv 1912.08193 (v1 2019-12-17); arXiv 2005.02551 (v1 2020-05-06) | Official repos | varies |
| **Morphological cleanup and connected-component filtering** | Removing specks and holes; keeping the largest component for single-organ tasks (tune on validation) | Deletes true small lesions. The minimum size must be chosen on validation per class | (practice) | scikit-image 0.26.0: `remove_small_objects(ar, min_size=<DEPRECATED>, connectivity=1, *, max_size=64)` removes objects with **max_size pixels or fewer** (`min_size` is deprecated), and `remove_small_holes(..., max_size=64)`; `scipy.ndimage.label(input, structure=None, output=None)`; `cv2.connectedComponentsWithStats(image[, labels[, stats[, centroids[, connectivity[, ltype]]]]])` | BSD / Apache-2.0 |
| **Sliding-window inference with overlap and Gaussian weighting** | Large 2-D or 3-D images at training patch size; Gaussian weights down-weight window edges | More overlap costs more time: nnU-Net: a larger step is "faster but less accurate" | nnU-Net arXiv 1809.10486 (v1 2018-09-27); arXiv 2404.09556 (v1 2024-04-15); MONAI arXiv 2211.02701 (v1 2022-11-04) | MONAI `sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, overlap=0.25, mode="constant", sigma_scale=0.125, padding_mode="constant", cval=0.0, sw_device=None, device=None, progress=False, roi_weight_map=None, process_fn=None, buffer_steps=None, buffer_dim=-1, with_coord=False)`; `mode="gaussian"` "gives less weight to predictions on edges of windows". nnU-Net `nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=True, device=torch.device("cuda"), ...)` | Apache-2.0 |

**Pitfalls (B7)**
- The MONAI defaults (`overlap=0.25`, `mode="constant"`) differ from nnU-Net's (a step of 0.5, which is 50% overlap, with Gaussian weighting). Set them explicitly when reproducing nnU-Net-style results.
- scikit-image 0.26 changed small-object removal from `min_size` to `max_size`, with different boundary semantics. Pin the version and the argument.
- nnU-Net's predictor defaults to `device=cuda`; pass the CPU device on the project machine.

### B8. Tracking post-processing

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Associate low-score boxes (BYTE)** | Occluded objects: +1 to 10 IDF1 when added to 9 trackers | Low-score false positives if the detector is poor | arXiv 2110.06864 (v1 2021-10-13) | `supervision` 0.30.5 (2026-09-22; ByteTrack class not re-verified); `boxmot` 25.0.0 (**AGPL-3.0**) | MIT / AGPL |
| **Offline linking and smoothing: AFLink, Gaussian-smoothed interpolation (GSI)** | Filling gaps from missed detections and fixing broken tracklets, without appearance features | Offline only (needs future frames), so it is unusable for real-time streams (reasoning) | StrongSORT arXiv 2202.13514 (v1 2022-02-28) | StrongSORT repo | - |
| **Observation-centric re-update (OC-SORT), re-ID association (Deep OC-SORT, BoT-SORT)** | Non-linear motion and occlusion: fixes Kalman error accumulated during occlusion; appearance helps re-association | Re-ID models add latency and licence questions | arXiv 2203.14360 (v1 2022-03-27); arXiv 2302.11813 (v1 2023-02-23); arXiv 2206.14651 (v1 2022-06-29) | boxmot (AGPL) or the official repos | varies |
| **Evaluation** | HOTA, IDF1 and MOTA (R4) | PyPI `trackeval` 1.3.0 is a **fork** (kovalp/TrackEval, MIT). The official JonathonLuiten/TrackEval (MIT, pushed 2024-07-03) is GitHub-only; `motmetrics` 1.4.0 (2022-12-26) | R4 | - | MIT |

---

## C. Ensembling and distillation

### C1. Ensembles

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Deep ensembles** (independent runs; average probabilities) | Accuracy and predictive uncertainty. Often matches the performance of an individual larger model | Costs M times the inference. "A single (but larger) neural network can replicate" the uncertainty and robustness benefits, and OOD gains track in-distribution gains | arXiv 1612.01474 (v1 2016-12-05); arXiv 2202.06985 (v1 2022-02-14); loss-landscape diversity arXiv 1912.02757 (v1 2019-12-05) | Average softmax probabilities (not logits of differently calibrated models; reasoning) | - |
| **Snapshot ensembles / FGE** (one training run) | Near-free members from cyclic LR | Many "sophisticated ensembling techniques are equivalent to an ensemble of only few independently trained networks" (deep ensemble equivalent) | arXiv 1704.00109 (v1 2017-04-01); arXiv 1802.10026 (v1 2018-02-27); arXiv 2002.06470 (v1 2020-02-15) | `CosineAnnealingWarmRestarts` plus checkpoint saving | permissive |
| **Ensembles of fine-tunes from one pretrained checkpoint** | Cheap diversity from seeds and hyperparameters; StarSSE explores the basin better | They stay in the same "pre-train basin" with "limited diversity"; leaving the basin loses the benefit of transfer | arXiv 2303.03374 (v1 2023-03-06) | - | - |
| **Multi-backbone / multi-pretraining ensembles** | The largest gains: pre-training "is a performant source of diversity" (VTAB, 19 tasks, rank candidates by nearest-neighbour accuracy, greedy selection on validation cross-entropy). Two models with different training methods, each about 76.5% on ImageNet, ensemble to 83.4% | Heterogeneous preprocessing and licences (R3); latency | arXiv 2010.06866 (v1 2020-10-14); arXiv 2110.12899 (v1 2021-10-20) | timm / open_clip models (R3) | varies |
| **Cascades / committees** (cost-aware) | Same accuracy, lower cost: an EfficientNet cascade is 5.4x faster than B7, and a ViT cascade 2.3x faster than ViT-L-384 | Needs a confidence threshold chosen on validation | arXiv 2012.01988 (v1 2020-12-03) | Threshold on MSP (B5) | - |
| **Caruana ensemble selection** (forward stepwise selection with replacement from a model library, optimising the target metric on validation) | Many candidate models; directly optimises the metric (accuracy, cross-entropy, ROC area) | Overfits a small validation set (reasoning); use out-of-fold predictions | Caruana et al., ICML 2004, DOI 10.1145/1015330.1015432 (verified via ACM and ML Anthology) | About 20 lines of code | - |
| **Stacking** (meta-model on out-of-fold predictions) | Members with complementary errors and enough out-of-fold data | Must beat "recalibrate the best single model" (the project's `ensemble.py` `recal` baseline explained 0.39 of stacking's 0.42 gain on near-chance models: STATUS.md). Leaks if the meta-features are not out-of-fold and group-aware | (no verified vision study; practice) | sklearn `LogisticRegression` on OOF probabilities; group-aware cross-fitting | BSD-3 |
| **Weight averaging of diverse fine-tunes (DiWA, ratatouille)** | OOD generalisation with a single model's inference cost | Needs a shared initialisation (same as soups) | arXiv 2205.09739 (v1 2022-05-19); arXiv 2212.10445 (v1 2022-12-20) | State-dict averaging | - |
| **Detection and segmentation ensembles** | WBF for boxes; probability averaging for masks (B2) | See B2 pitfalls | arXiv 1910.13302 (v1 2019-10-29) | `ensemble-boxes` | MIT |

**Measuring diversity and gain.** Report each member's score, the ensemble score, the pairwise error correlation or disagreement rate, and the **marginal gain of adding each member** under a paired bootstrap (D2) (practice). Add a member only if the paired CI of the gain excludes zero, or the gain is worth its latency. Ensembles change calibration (usually toward under-confidence; reasoning), so recalibrate after ensembling.

### C2. Knowledge distillation into one model

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Logit KD (soft targets)** | Compressing a large model or an ensemble. Patient, consistent distillation with matching augmented views (R3) | Large teacher-student gaps: "larger models do not often make better teachers", which early-stopped teachers mitigate. A sequence of distillation steps (teacher assistants) did not fix this in Cho et al. | arXiv 1503.02531 (v1 2015-03-09); arXiv 1910.01348 (v1 2019-10-03); arXiv 1902.03393 (v1 2019-02-09); arXiv 2106.05237 (v1 2021-06-09) | timm `train.py --kd-model-name ... --kd-distill-type {logit,feature,token}` (R3) | Apache-2.0 |
| **Does the student match the teacher?** | - | Often not: "a surprisingly large discrepancy between the predictive distributions of the teacher and the student, even in cases when the student has the capacity to perfectly match". Closer matching "paradoxically does not always lead to better student generalization" | arXiv 2106.05945 (v1 2021-06-10) | Measure top-1 agreement and KL between student and teacher on held-out data | - |
| **DIST (relational KD) / DKD (decoupled KD)** | Strong teachers where exact KL matching "would disturb the training". DIST works for classification, detection and segmentation | Extra hyperparameters | arXiv 2205.10536 (v1 2022-05-21); arXiv 2203.08679 (v1 2022-03-16) | Official repos | varies |
| **DeiT distillation token** | ViT students learning from a CNN teacher | Architecture change (extra token) | arXiv 2012.12877 (v1 2020-12-23) | timm `deit_*_distilled_*` models; `--kd-distill-type token` (R3) | Apache-2.0 |
| **Ensemble distillation / distribution distillation** | One model that keeps the ensemble's accuracy and (with EnD²) its uncertainty decomposition | A standard KD student loses the ensemble's diversity information | arXiv 1905.00076 (v1 2019-04-30); theory arXiv 2012.09816 (v1 2020-12-17) | - | - |
| **Self-distillation (born-again networks)** | A same-size student can beat its teacher | Extra training round | arXiv 1805.04770 (v1 2018-05-12) | - | - |
| **Detection KD: feature imitation, localisation distillation (LD), FGD** | Distil where objects are (fine-grained imitation), box distributions (LD), focal and global feature relations (FGD) | Detector-specific plumbing | arXiv 1906.03609 (v1 2019-06-09); arXiv 2102.12252 (v1 2021-02-24); arXiv 2111.11837 (v1 2021-11-23) | Official repos; mmdet/mmrazor ecosystem (not verified) | varies |
| **Segmentation KD: structured / channel-wise** | Dense prediction students (pairwise, holistic and channel distributions) | - | arXiv 1903.04197 (v1 2019-03-11); arXiv 2011.13256 (v1 2020-11-26) | Official repos | varies |
| **Use unlabelled images for distillation** | Distilling on extra unlabelled in-domain images; SimCLRv2 distils a big fine-tuned model into a small one "with little loss" | Domain mismatch of the unlabelled pool | arXiv 2006.10029 (v1 2020-06-17) | - | - |

**Decision guidance (C).**
1. Build candidates that are diverse by construction: different pre-training (for example DINOv2 and SigLIP 2) and different architecture families.
2. Average probabilities (classification, segmentation) or use WBF (detection).
3. Select members greedily on out-of-fold or validation predictions, then compare against the best single model and against "recalibrated best single" with a paired bootstrap.
4. If latency matters, distil the ensemble into one student using unlabelled or augmented in-domain images. Report the student-teacher agreement, and expect a gap.
5. Keep label smoothing off the teacher (arXiv 1906.02629), unless labels are noisy (arXiv 2003.02819).

---

## D. Evaluation

### D1. Metric definitions and correct implementations (beyond R4)

**Classification and multi-label**

| Metric | Definition / when to use | Pitfall | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| Accuracy, top-k | Overall correctness; top-k for large label spaces | **torchmetrics `MulticlassAccuracy(num_classes, top_k=1, average="macro")` defaults to macro**, which is balanced accuracy, not accuracy (local test: macro 0.500 against micro 0.833 on the same predictions). Pass `average="micro"` for plain accuracy | - | torchmetrics 1.9.0; sklearn `accuracy_score`, `top_k_accuracy_score` | Apache-2.0 / BSD-3 |
| Balanced accuracy | Mean per-class recall; imbalance-robust | sklearn `adjusted=False` by default (not chance-corrected) | - | sklearn `balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False)` | BSD-3 |
| Macro / micro / weighted F1 | Macro = mean over classes (treats rare classes equally); micro = pooled | torchmetrics `MulticlassF1Score(num_classes, top_k=1, average="macro", ..., zero_division=0)` and `MultilabelF1Score(num_labels, threshold=0.5, average="macro", ..., zero_division=0)`. A default 0.5 threshold for multi-label is arbitrary (B4). F1 is improper for clinical decisions (arXiv 2412.10288) | arXiv 1402.1892 (v1 2014-02-08) | torchmetrics; sklearn `f1_score(average=...)` | Apache-2.0 / BSD-3 |
| AUROC / AUPRC (AP) | Threshold-free ranking. Metrics Reloaded recommends AUROC "as the default multi-threshold metric for classification"; AP "can be suitable when class imbalances should be compensated" | "AUPRC is not generally superior in cases of class imbalance" and "can unduly favor model improvements in subpopulations with more frequent positive labels" | arXiv 2401.06091 (v1 2024-01-11); Metrics Reloaded arXiv 2206.01653 (v1 2022-06-03) (published Nature Methods 21(2):195-212, 2024, DOI 10.1038/s41592-023-02151-z) | torchmetrics `MulticlassAUROC`, `MulticlassAveragePrecision(num_classes, average="macro", thresholds=None, ...)` (options `macro`, `weighted`, `none`; **no micro** for multiclass AP); sklearn `roc_auc_score`, `average_precision_score(y_true, y_score, *, average="macro", pos_label=1, sample_weight=None)` | Apache-2.0 / BSD-3 |
| MCC, Cohen's kappa | Single-number summaries under imbalance; quadratic-weighted kappa for ordinal grades (practice) | Kappa depends on prevalence | - | sklearn `matthews_corrcoef`, `cohen_kappa_score(weights="quadratic")` | BSD-3 |
| Multi-label mAP | Mean over labels of per-label AP | Labels with no positives in the test set have undefined AP; exclude them and report the count (practice; library behaviour not verified) | arXiv 2009.14119 (v1 2020-09-29) (the usual benchmark metric) | torchmetrics `MultilabelAveragePrecision(num_labels, average="macro", thresholds=None, ...)`; sklearn `average_precision_score` on the indicator matrix | Apache-2.0 / BSD-3 |
| Calibration and proper scores | NLL, Brier, adaptive/debiased ECE (B3). Metrics Reloaded: "we recommend the [ECE] or [PSR]" (proper scoring rules) | See B3 estimator pitfalls | arXiv 2206.01653 (v1 2022-06-03) | B3 | - |

**Detection** (pycocotools / faster-coco-eval / torchmetrics backends and the COCOeval pitfalls are in R4)

| Metric | Definition / when to use | Pitfall | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| COCO AP / AP50 / AR | Per-class AP averaged over IoU 0.50:0.95 | torchmetrics `MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=None, rec_thresholds=None, max_detection_thresholds=None, class_metrics=False, extended_summary=False, average="macro", backend="pycocotools")`: `average="micro"` changes the definition. **Do not wrap it in `BootStrapper`** (fails; D2) | R4 | torchmetrics 1.9.0; `faster-coco-eval` 1.8.0 (2026-08-24); `pycocotools` 2.0.11 (2025-12-15) | Apache-2.0; FreeBSD |
| LVIS AP (federated), AP-fixed, AP-pool | Large vocabulary. The default AP "is neither category independent, nor does it directly reward properly calibrated detectors" and is "gameable" by re-ranking. AP-fixed (per-class detection limit) and AP-pool (rewards cross-category calibration) fix this | Using plain COCOeval on LVIS | arXiv 1908.03195 (v1 2019-08-08); arXiv 2102.01066 (v1 2021-02-01) | `lvis` 0.5.3 (2020, stale); faster-coco-eval LVIS mode (R4) | - |
| Metrics Reloaded view | Choose the localisation criterion (Box IoU, Mask IoU, ...) and the assignment strategy first, then "a multi-threshold metric ([AP] or [FROC] Score)" | FROC is common in medical detection; AP is the standard | arXiv 2206.01653 (v1 2022-06-03) (arXiv HTML v3) | MONAI `compute_froc_score`, `compute_fp_tp_probs` (dev listing) | Apache-2.0 |
| FiftyOne evaluation | Per-sample and per-object TP/FP/FN fields for slicing (D3) | Default `iou=0.50` in `evaluate_detections` differs from COCO AP@[.50:.95]; state which is used | source read | `fiftyone.utils.eval.detection.evaluate_detections(samples, pred_field, gt_field="ground_truth", eval_key=None, classes=None, missing=None, method=None, iou=0.50, use_masks=False, use_boxes=False, classwise=True, dynamic=True, custom_metrics=None, progress=None, **kwargs)`; methods `"coco"`, `"open-images"`, `"activitynet"`; with `eval_key` it writes `<eval_key>_tp` / `_fp` / `_fn` per sample and `<eval_key>`, `_id`, `_iou` per object. `fiftyone` 1.22.0 (2026-09-12), `fiftyone-brain` 0.26.0, `fiftyone-db` 1.4.1 | Apache-2.0 (all three per PyPI) |

**Segmentation, depth and retrieval**

| Metric | Definition / when to use | Pitfall | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| Dice / IoU | Metrics Reloaded default for semantic segmentation: "DSC or IoU", plus a boundary metric; "metric values should first be computed per image and then be aggregated over the set of images" | Natural-image benchmarks (ADE20K, Cityscapes) instead accumulate dataset-level intersection and union (R4). State the aggregation. torchmetrics `DiceScore(num_classes, include_background=True, average="macro", aggregation_level="samplewise", input_format="one-hot")`; `GeneralizedDiceScore(..., weight_type="square")` | arXiv 2206.01653 (v1 2022-06-03); arXiv 2302.01790 (v1 2023-02-03) | torchmetrics 1.9.0; MONAI `DiceMetric`, `MeanIoU` | Apache-2.0 |
| HD95 | 95th-percentile Hausdorff distance, in physical units | **torchmetrics `HausdorffDistance(num_classes, include_background=False, distance_metric="euclidean", spacing=None, directed=False, input_format="one-hot")` has no percentile, so it gives the max HD, not HD95.** MONAI `HausdorffDistanceMetric(include_background=False, distance_metric="euclidean", percentile=None, directed=False, reduction=MEAN, get_not_nans=False)`: set `percentile=95` and pass the voxel `spacing`. It returns NaN when both prediction and reference are empty (source read). Decide and report how empty cases are scored | MONAI source (1.6.0 tag) | MONAI 1.6.0 (2026-06-22) | Apache-2.0 |
| NSD (normalised surface Dice) | Metrics Reloaded's default boundary metric "in case annotation imprecisions should be compensated for"; tolerance in mm | The tolerance is a task parameter; report it | arXiv 1809.04430 (v1 2018-09-12); arXiv 2206.01653 (v1 2022-06-03) | MONAI `SurfaceDiceMetric`; `surface-distance` 0.1 (2024-07-30, Apache-2.0); MONAI `MetricsReloadedBinary` / `MetricsReloadedCategorical` wrappers (dev listing); reference implementation Project-MONAI/MetricsReloaded (GitHub page: Apache-2.0; install from source; **not on PyPI**, name returns 404) | Apache-2.0 |
| clDice, Boundary IoU | clDice for tubular structures (Metrics Reloaded); Boundary IoU for boundaries (R4) | - | arXiv 2003.07311 (v1 2020-03-16) | MONAI (losses); boundary-iou code (R4) | - |
| PQ (panoptic quality) | Metrics Reloaded: "[PQ] is especially suited for instance segmentation, as it combines ... detection performance and segmentation quality" | Thing/stuff and void handling (R4) | arXiv 1801.00868 (v1 2018-01-03) | torchmetrics `PanopticQuality`, `ModifiedPanopticQuality`; MONAI `PanopticQualityMetric` | Apache-2.0 |
| GPL tools to avoid | - | `medpy` 0.5.2 is GPL | - | Use MONAI, `surface-distance` or `seg-metrics` 1.2.8 (MIT) | - |
| Depth: AbsRel, RMSE, δ1 | AbsRel = mean(abs(d - d\*)/d\*); δ1 = share of pixels with max(d/d\*, d\*/d) < 1.25 (standard definitions from Eigen et al.; paper body not re-read) | Affine-invariant models (MiDaS-style, trained with a loss "invariant to changes in depth range and scale") must be aligned (scale, or scale and shift, by least squares) before metrics. Report the alignment, the valid mask and the depth cap. torchmetrics has no AbsRel or δ metrics (regression module list verified) | arXiv 1406.2283 (v1 2014-06-09); arXiv 1907.01341 (v1 2019-07-02) | Implement (about 10 lines); torchmetrics `MeanAbsoluteError` / `MeanSquaredError` for the plain parts | - |
| Retrieval: Recall@k, mAP@k, MAP@R | Metric-learning "Recall@k" is **hit rate**: 1 if any relevant item is in the top k. IR recall is the share of all relevant items retrieved | **torchmetrics `RetrievalRecall(top_k=1)` is IR recall** (local test: 3 relevant items, top-1 correct gives 0.333, while `RetrievalHitRate(top_k=1)` gives 1.0). Metric-learning papers usually mean hit rate (practice). MAP@R is recommended by the reality check. Revisited Oxford/Paris protocols (Easy/Medium/Hard) for landmarks | arXiv 2003.08505 (v1 2020-03-18); arXiv 1803.11285 (v1 2018-03-29) | torchmetrics `RetrievalMAP(empty_target_action="neg", ignore_index=None, top_k=None, aggregation="mean")`, `RetrievalRecall`, `RetrievalHitRate`, `RetrievalRPrecision`, `RetrievalMRR`, `RetrievalNormalizedDCG` (need `indexes`) | Apache-2.0 |

### D2. Confidence intervals and model comparison

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Image-level bootstrap for dataset-level metrics** (mAP, dataset-level mIoU, AUROC) | Any metric not decomposable per image: resample images with all their objects and pixels, then recompute the metric | torchmetrics `BootStrapper(base_metric, num_bootstraps=10, mean=True, std=True, quantile=None, raw=False, sampling_strategy="poisson")` resamples **within each `update` batch** along dim 0: 10 resamples are too few, it approximates resampling of the whole dataset only per batch, and it **raises `TypeError` on `MeanAveragePrecision`** (local test). Write the loop: cache per-image predictions, resample image ids, recompute | torchmetrics source and docstring (read locally) | Own loop with pycocotools or faster-coco-eval on resampled image-id lists; 1,000-10,000 resamples (practice) | - |
| **Group-level (cluster) bootstrap** | Images nested in patients, sites, videos or scenes: resample groups. Metrics Reloaded: "Respect the hierarchical data structure when aggregating metric values" | Too few groups (fewer than about 20-30) gives unstable intervals (practice) | arXiv 2206.01653 (v1 2022-06-03) (arXiv HTML v3) | Own loop (the project's `ensemble.py` already has a group bootstrap: STATUS.md); fairlearn `MetricFrame(n_boot=..., ci_quantiles=...)` for grouped metrics | MIT |
| **Paired bootstrap for model comparison** | Two models on the same test images: resample the same ids for both and take the CI of the difference. Much tighter than comparing two separate CIs (reasoning) | Correlated folds or seeds: include seeds as a variance source | arXiv 2103.03098 (v1 2021-03-01) (variance sources; randomising more of them gives better estimates at lower cost); Bayesian comparisons arXiv 1606.04316 (v1 2016-06-14) | `scipy.stats.bootstrap(data, statistic, *, n_resamples=9999, paired=False, axis=0, confidence_level=0.95, alternative="two-sided", method="BCa", ...)` with `paired=True` for per-image metrics | BSD-3 |
| **Wilson / Clopper-Pearson for proportions** (accuracy, sensitivity, specificity) | Small n and extreme proportions. Brown, Cai & DasGupta recommend Wilson or equal-tailed Jeffreys for small n, and Agresti-Coull for larger n; the Wald interval behaves erratically | Treats images as independent (use a group bootstrap for clustered data) | Brown, Cai, DasGupta, *Statistical Science* 16(2):101-133, 2001, DOI 10.1214/ss/1009213286 (verified via Project Euclid) | `scipy.stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")`: default `"exact"` is Clopper-Pearson; also `"wilson"`, `"wilsoncc"`. Local example, 45/50: exact 0.7819-0.9667, Wilson 0.7864-0.9565 | BSD-3 |
| **Minimum test sizes** | Plan the test set before running | Tiny test sets: CIs wider than the differences being claimed | Brain MRI segmentation: a 1%-wide CI needs about 100-200 test cases when the SD is around 3%, and over 1,000 for harder tasks, arXiv 2307.10926 (v1 2023-07-20). Classification: "75 - 100 samples will usually be needed to test a good but not perfect classifier" (biospectroscopy) arXiv 1211.1323 (v1 2012-11-06). MICCAI 2023: only 1 of 221 papers reported CIs; median CI width 0.03 Dice, three times the median first-to-second gap, arXiv 2409.17763 (v1 2024-09-26) | Precision planning: half-width ≈ 1.96·SD/√n (reasoning) | - |
| **Repeated runs and seeds** | Separating method effects from seed luck | Single-seed claims | arXiv 2109.08203 (v1 2021-09-16); arXiv 2304.01910 (v1 2023-04-04) | - | - |

**Decision guidance (D2)**
- Report every headline metric with a 95% CI at the unit of independence (image, or patient/site when nested).
- Compare models with a paired interval on the difference.
- Declare "no evidence of a difference" when the paired CI includes zero.
- Pre-register the metric and the aggregation before looking at test results.
- Following Metrics Reloaded: "Include a visualization of the raw metric values", and "Consider multiple test set runs to address the variability of results from non-determinism".

### D3. Slice and error analysis

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Metadata slices** (object size, occlusion, lighting, site, scanner, device, time) | Finding where the model fails: per-slice metrics with CIs. COCO size bins come from the GT `area` field (R4) | Many slices multiply comparisons; small slices are noisy. Report the CI and n per slice | Hidden stratification: failures on low-prevalence, low-label-quality, subtle or spurious subsets, arXiv 1909.12475 (v1 2019-09-27) | fairlearn `MetricFrame(*, metrics, y_true, y_pred, sensitive_features, control_features=None, sample_params=None, n_boot=None, ci_quantiles=None, random_state=None)` with `.by_group`, `.difference()`, `.ratio()`, `.group_min()`, `.group_max()`, `*_ci` (source read); FiftyOne views over per-sample TP/FP/FN fields | MIT / Apache-2.0 |
| **TIDE error decomposition** (detection) | Splits the AP loss into six error types (classification, localisation, both, duplicate, background, missed) and isolates each type's contribution | Stale tooling: `tidecv` 1.0.1 (2020-08-31); repo dbolya/tide MIT, last push 2023-03-31; compatibility with current NumPy not verified | arXiv 2008.08115 (v1 2020-08-18) | `tidecv` (API not re-verified) | MIT |
| **Automatic slice discovery** (Domino, Spotlight) | Unlabelled failure modes, using embeddings (cross-modal for Domino) | Prior slice-discovery methods "often identify slices that are incoherent"; confirm by hand | arXiv 2203.14960 (v1 2022-03-24); arXiv 2107.00758 (v1 2021-07-01) | Paper code | - |
| **Confusion analysis** | Class pairs, and label-noise suspects (cleanlab on evaluation data) | - | arXiv 1911.00068 (v1 2019-10-31) | sklearn `confusion_matrix`; cleanlab | BSD-3 / Apache-2.0 |
| **Hard-example mining (OHEM)** | Training-time focus on hard RoIs (two-stage detectors) | Amplifies label noise (reasoning) | arXiv 1604.03540 (v1 2016-04-12) | Detector frameworks | - |

### D4. Robustness and OOD evaluation

| Benchmark / score | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| ImageNet-C / -R / -A / -Sketch / V2, ObjectNet | Classic shift suites for ImageNet-class models | ImageNet-C corruptions "are no longer OOD relative to today's large, web-scraped datasets" | arXiv 1903.12261 (v1 2019-03-28); arXiv 2006.16241 (v1 2020-06-29); arXiv 1907.07174 (v1 2019-07-16); arXiv 1905.13549 (v1 2019-05-29); arXiv 1902.10811 (v1 2019-02-13); ObjectNet (NeurIPS 2019; no arXiv verified) | `robustbench` 1.1.1 (2025-04-11) | Apache-2.0 |
| **LAION-C** | OOD robustness for web-scale models: six distortion types designed to be OOD even for LAION-scale training. the paper reports "the best models now matching or outperforming the best human observer" | New; ImageNet-class label space | arXiv 2506.16950 (v1 2025-06-20) | Dataset release (not checked) | - |
| COCO-C / detection corruptions | Detector robustness to corruptions | Synthetic | arXiv 1907.07484 (v1 2019-07-17) | `imagecorruptions` 1.1.2 (2020, Apache-2.0) | Apache-2.0 |
| **WILDS and real-shift test sets (site, time, device)** | Real deployment shifts (hospitals, cameras, regions) | Real shifts are task-specific; build your own external set (R1) | arXiv 2012.07421 (v1 2020-12-14) | `wilds` 2.0.0 (2021-12-13) | MIT |
| Effective robustness / accuracy-on-the-line | Interpreting OOD gains: OOD accuracy usually tracks in-distribution accuracy, so claim robustness only above the line | - | arXiv 2007.00644 (v1 2020-07-01); arXiv 2107.04649 (v1 2021-07-09) | - | - |
| **OOD detection scores** (MSP, ODIN, Mahalanobis, energy, KNN, ReAct, ViM, ASH, Mahalanobis++) | Flagging inputs outside the training distribution. On ImageNet, post-hoc methods on standard CE models are strongest, and "no single method ... consistently outperforms". Linear-probed DINOv2 "consistently and remarkably outperforms" its fully supervised counterpart. L2-normalised Mahalanobis (Mahalanobis++) improved results across 44 models | Under covariate shift, complex detectors do "similarly to, or even worse than" MSP. Near-OOD progress is slow. KNN and ASH cost "orders of magnitude" more at inference | arXiv 1610.02136 (v1 2016-10-07); arXiv 1706.02690 (v1 2017-06-08); arXiv 1807.03888 (v1 2018-07-10); arXiv 2010.03759 (v1 2020-10-08); arXiv 2204.06507 (v1 2022-04-13); arXiv 2111.12797 (v1 2021-11-24); arXiv 2203.10807 (v1 2022-03-21); arXiv 2209.09858 (v1 2022-09-20); arXiv 2505.18032 (v1 2025-05-23); OpenOOD arXiv 2210.07242 (v1 2022-10-13) and v1.5 arXiv 2306.09301 (v1 2023-06-15) (HTML v5); arXiv 2310.01755 (v1 2023-10-03); pretrained transformers for near-OOD arXiv 2106.03004 (v1 2021-06-06) | `pytorch-ood` 0.3.3 (2026-09-07, Apache-2.0); OpenOOD (GitHub Jingkang50/OpenOOD MIT, pushed 2025-12-01; not on PyPI) | Apache-2.0 / MIT |

**Decision guidance (D4).**
- Start with MSP and energy (free) and Mahalanobis++ on L2-normalised penultimate features (cheap). Pick on a validation OOD set that matches the expected deployment shift: semantic, covariate, or both.
- Evaluate on a real-shift external set (another site or camera) before any synthetic corruption suite. For ImageNet-class models, use LAION-C rather than ImageNet-C alone.

### D5. Explainability, with sanity checks

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Grad-CAM family** (Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM, HiResCAM, LayerCAM, ...) | Coarse "where is the evidence" maps for CNNs; hypothesis generation | The HiResCAM paper's title recommends it over Grad-CAM "for faithful explanations" (its argument that Grad-CAM can highlight regions the model did not use is in the paper body, not re-read). ViTs need a `reshape_transform` | arXiv 1610.02391 (v1 2016-10-07); arXiv 1710.11063 (v1 2017-10-30); arXiv 1910.01279 (v1 2019-10-03); arXiv 2008.00299 (v1 2020-08-01); arXiv 2011.08891 (v1 2020-11-17) | `grad-cam` 1.5.7 (2026-08-21; repo jacobgil/pytorch-grad-cam): `GradCAM(model, target_layers, reshape_transform=None)`; `cam(input_tensor, targets=[ClassifierOutputTarget(k)], aug_smooth=False, eigen_smooth=False)`. Also exports `HiResCAM`, `ScoreCAM`, `EigenCAM`, `LayerCAM`, `FullGrad`, `AblationCAM`, `FinerCAM`, `ShapleyCAM`, `KPCA_CAM`. Targets include `SemanticSegmentationTarget`, `FasterRCNNBoxScoreTarget`. Faithfulness metrics `pytorch_grad_cam.metrics.road` (ROAD) and `cam_mult_image` | MIT |
| **Attention rollout / transformer attribution** | ViTs | Raw attention is not an explanation; ViTs without registers show high-norm artefact tokens in attention maps | arXiv 2005.00928 (v1 2020-05-02); arXiv 2012.09838 (v1 2020-12-17); arXiv 2309.16588 (v1 2023-09-28) | Implement, or use the Chefer et al. code | - |
| **Integrated Gradients** | Axiomatic pixel attribution (sensitivity, implementation invariance) | The baseline choice drives the result; noisy for images | arXiv 1703.01365 (v1 2017-03-04) | `captum` 0.9.0 (2026-04-17): `IntegratedGradients(forward_func, multiply_by_inputs=True).attribute(inputs, baselines=None, target=None, additional_forward_args=None, n_steps=50, method="gausslegendre", internal_batch_size=None, return_convergence_delta=False)` | BSD-3 |
| **Sanity checks** (model- and data-randomisation tests) | Mandatory before trusting any map: "some existing saliency methods are independent both of the model and of the data generating process" | Visual plausibility alone "can be misleading" | arXiv 1810.03292 (v1 2018-10-08); arXiv 1711.00867 (v1 2017-11-02); ROAR arXiv 1806.10758 (v1 2018-06-28); sanity checks for saliency metrics arXiv 1912.01451 (v1 2019-11-29); ROAD arXiv 2202.00449 (v1 2022-02-01) | Randomise weights layer by layer and compare the maps (SSIM or rank correlation) | - |
| **When saliency misleads** | - | Post hoc explanations "are ineffective when the spurious artifact is unknown at test-time", and attribution can falsely indicate reliance on spurious signals. In medical imaging, all 8 saliency methods failed at least one trustworthiness criterion; use detection or segmentation models when localisation is the goal | arXiv 2212.04629 (v1 2022-12-09); arXiv 2008.02766 (v1 2020-08-06) | - | - |
| **Concept-based checks** (TCAV, concept bottlenecks) | Testing a named hypothesis ("does the model use the ruler?") | Needs concept example sets | arXiv 1711.11279 (v1 2017-11-30); arXiv 2007.04612 (v1 2020-07-09) | Captum `TCAV` (not re-verified) | BSD-3 |
| **Explanation-metric toolkits** | Quantitative evaluation of explanations | Licences: `quantus` 0.6.0 (PyPI classifier GPLv3; GitHub NOASSERTION); `zennit` 1.0.0 (GPL-3.0-or-later AND LGPL-3.0-or-later) | arXiv 2202.06861 (v1 2022-02-14) | Prefer grad-cam ROAD metrics (MIT) or `torchcam` 0.4.1 (Apache-2.0) | copyleft |

**Decision guidance (D5).** Use saliency only to generate hypotheses. Confirm shortcut hypotheses with interventions: the project's "cue used" masking test and "cue present" probe (STATUS.md) are the evidence; maps are illustrations. Run the model-randomisation check on any map shown in a report.

### D6. Fairness and subgroup performance (people and patients)

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Per-group metrics with CIs** (TPR/FPR gaps, AUROC per group, calibration per group) | Any system where images show people or patients. CheXclusion found TPR disparities "in all datasets, for all clinical tasks, and all subgroups"; multi-source training data had the smallest disparities | Small groups: wide CIs. Report n and the CI | arXiv 2003.00827 (v1 2020-02-14) | fairlearn 0.14.0 (2026-06-07) `MetricFrame` (D3) | MIT |
| **Demographic shortcut awareness** | Models can infer race from medical images. Models that encode less demographic information were more "globally optimal" (fairer on external sites); corrections that are "locally optimal" do not transfer | Algorithmic fixes tuned on one site | arXiv 2107.10356 (v1 2021-07-21); arXiv 2312.10083 (v1 2023-12-11) | - | - |
| **Fairness benchmarks and model selection** | MEDFAIR: 11 algorithms, 9 datasets, 3 model-selection criteria; model selection is an "under-studied issue" that affects fairness conclusions | - | arXiv 2210.01725 (v1 2022-10-04) | - | - |
| **Metric choice** | AUPRC can heighten disparities by favouring subgroups with more positives | - | arXiv 2401.06091 (v1 2024-01-11) | - | - |
| **Balanced face datasets** | Auditing face-attribute models | Licences and consent of face datasets need review | FairFace arXiv 1908.04913 (v1 2019-08-14) | - | - |

### D7. Reporting: model cards, datasheets and what a vision model report must contain

- **Standards and templates:**
  - Model Cards: arXiv 1810.03993 (v1 2018-10-05).
  - Datasheets for Datasets: arXiv 1803.09010 (v1 2018-03-23).
  - Data Cards: arXiv 2204.01075 (v1 2022-04-03).
  - REFORMS (reporting standards for ML-based science): arXiv 2308.07832 (v1 2023-08-15).
  - Leakage taxonomy: arXiv 2207.07048 (v1 2022-07-14).
  - Metrics Reloaded: Nature Methods 21(2):195-212 (2024), DOI 10.1038/s41592-023-02151-z.
  - Medical: CLAIM 2024 Update (Tejani et al., *Radiology: Artificial Intelligence* 6(4), 2024, DOI 10.1148/ryai.240300); TRIPOD+AI (Collins et al., *BMJ* 2024;385:e078378). Performance-measure guidance for medical decisions: arXiv 2412.10288 (v1 2024-12-13).
- **Tools:** `huggingface-hub` 2.0.0 (2026-09-24; model card support; API not re-verified). `model-card-toolkit` 2.0.0 (2023-04-03) is stale.
- **Minimum contents of a vision model report** (synthesised from the sources above; the checklist is this project's own):
  1. Intended use and out-of-scope uses.
  2. Data: provenance, licences, consent, the split protocol with group and near-duplicate handling, and the leakage checks run (R1, REFORMS, arXiv 2207.07048).
  3. Label quality: the audit, the estimated noise rate, and whether test labels were reviewed.
  4. Preprocessing and augmentation, including TTA.
  5. Model: architecture, pretrained weights and their licence (R3), training recipe, compute, HPO budget and search space, number of seeds, and the checkpoint-selection rule.
  6. Metrics chosen per Metrics Reloaded, with aggregation stated, 95% CIs at the unit of independence, and paired comparisons against baselines, including a tuned simple baseline.
  7. Per-slice and per-subgroup results with n and CIs, plus a calibration report (reliability diagram; NLL, Brier, adaptive ECE) and the operating point with how it was chosen.
  8. Robustness: an external or real-shift set, and OOD detection if used.
  9. Error analysis (TIDE or confusion) with failure examples.
  10. Explainability, only with sanity checks.
  11. Deployment: export format, runtime, quantisation, parity-test results, latency p50/p90/p99 with hardware, and the monitoring plan.

---

## E. Deployment

### E1. Export formats

| Path | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **ONNX via the torch.export-based exporter** (default since torch 2.9) | Portable graph for ONNX Runtime, OpenVINO and TensorRT; dynamic shapes through `dynamic_shapes` | Needs `onnxscript`, which is **missing in `cv-skills`**, so the default call fails. Models with data-dependent Python control flow need rewriting for `torch.export` | torch 2.14 docs ("Changed in version 2.9: `dynamo` is now True by default"; `dynamic_shapes` for `dynamo=True`, the deprecated `dynamic_axes` for `dynamo=False`); local test | `torch.onnx.export(model, args=(), f=None, *, kwargs=None, input_names=None, output_names=None, opset_version=None, dynamo=True, external_data=True, dynamic_shapes=None, report=False, optimize=True, verify=False, ...)` returns an `ONNXProgram`. `onnxscript` 0.7.2 (2026-09-09, MIT). `onnx` 1.23.0 (2026-09-18; locally max opset 28, IR 14) | Apache-2.0 / MIT |
| **ONNX via the legacy TorchScript exporter** (`dynamo=False`) | Works today in `cv-skills` (legacy export of ResNet-18 at opset 17 with `dynamic_axes`, verified in memory) | DeprecationWarning ("legacy TorchScript-based ONNX export"); it will be removed | local test | `torch.onnx.export(model, (x,), f, dynamo=False, opset_version=17, input_names=[...], output_names=[...], dynamic_axes={...})` | - |
| **TorchScript** (`torch.jit.script` / `trace`) | Legacy C++ deployment only | **Deprecated in torch 2.14**: FutureWarning "Please switch to `torch.compile` or `torch.export`" | local test | - | - |
| **ExecuTorch** (mobile, embedded and edge; `.pte`) | On-device PyTorch-native deployment. Backends: XNNPACK (CPU), Core ML, Vulkan, Qualcomm, MediaTek, Arm, Samsung, OpenVINO, CUDA, MLX and MCUs. Platforms include Windows | Operator coverage per backend | README | `executorch` 1.5.1 (2026-09-22): `to_edge_transform_and_lower(torch.export.export(model, sample_inputs), partitioner=[XnnpackPartitioner()]).to_executorch()`, then write `.buffer` to a `.pte` file | BSD-3 |
| **Core ML** | Apple devices (ANE/GPU) | The docs recommend the `torch.jit.trace` path (now deprecated in torch) and call `torch.export` conversion "beta" with about 70% operator coverage | coremltools docs ("Converting from PyTorch") | `coremltools` 9.0 (2025-11-10): `ct.convert(traced_model, inputs=[ct.TensorType(shape=...)])` or `ct.convert(exported_program)` | BSD |
| **LiteRT (formerly TFLite) via LiteRT Torch** | Android and embedded | README lists **Linux only**. `ai-edge-torch` 0.7.2 (2026-01-29) is "DEPRECATED: renamed to litert-torch" | README; PyPI summary | `litert-torch` 0.9.4 (2026-08-24): `litert_torch.convert(model.eval(), sample_inputs).export("m.tflite")`; runtime `ai-edge-litert` 2.2.0 | Apache-2.0 |
| **Graph simplifiers** | Removing redundant nodes after export | Validate parity after simplifying | - | `onnxslim` 0.1.96 (MIT); `onnxsim` 0.7.3 (MIT AND (Apache-2.0 OR BSD-2-Clause)); ONNX Runtime `quant_pre_process` (E3) | permissive |

### E2. Runtimes

| Runtime | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **ONNX Runtime CPU** | Default CPU inference: locally, ResNet-18 at batch 1 and 224 px ran at p50 31.2 ms against 62.2 ms in PyTorch eager (4 threads; E4) | - | local measurement | `onnxruntime` 1.30.0 (2026-09-10; requires Python 3.11 or later): `InferenceSession(model_bytes_or_path, sess_options, providers=["CPUExecutionProvider"])`; `SessionOptions().intra_op_num_threads`. Providers available in `cv-skills`: `AzureExecutionProvider`, `CPUExecutionProvider` | MIT |
| **ONNX Runtime GPU (CUDA / TensorRT EP)** | NVIDIA servers | Needs a matching CUDA stack | ORT docs | `onnxruntime-gpu` 1.30.0 | MIT |
| **DirectML EP** (Windows, any DX12 GPU, including AMD) | Windows GPU inference without CUDA | **"DirectML is in sustained engineering"**; new work is in WinML. No concurrent `Run` on one session; memory pattern must be disabled and execution sequential; supports up to opset 20; `onnxruntime-directml` 1.24.4 (2026-03-17) lags ORT 1.30.0 | ORT DirectML EP docs; PyPI | `onnxruntime-directml` 1.24.4; repo microsoft/DirectML (MIT, pushed 2026-04-27) | MIT |
| **OpenVINO** | Intel CPUs, GPUs and NPUs; NNCF quantisation | Performance on AMD CPUs not verified here | - | `openvino` 2026.4.0 (2026-09-16); `onnxruntime-openvino` 1.24.1 | Apache-2.0 |
| **TensorRT** | Lowest-latency NVIDIA inference | **TensorRT 11 removed `IInt8Calibrator` and the FP16/INT8 builder flags; networks are strongly typed** (use ModelOpt Q/DQ). Proprietary licence | NVIDIA TensorRT 11.3 migration guide; TensorRT 10.1 deprecated `IInt8EntropyCalibrator2` (NVIDIA docs, search result) | `tensorrt` 11.3.0.99 (2026-09-09, proprietary); `torch-tensorrt` 2.14.0 (2026-09-23, BSD); `nvidia-modelopt` 0.47.0 (2026-09-23, Apache-2.0) | proprietary / BSD / Apache-2.0 |

### E3. Quantisation

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Post-training static INT8 (ONNX Runtime)** | CNNs on CPU. ORT docs: "static quantization for CNN models" and dynamic for transformers; "S8S8 with QDQ is the default setting and balances performance and accuracy" | U8S8 on x86-64 with AVX2/AVX512 but no VNNI "might suffer from saturation issues"; use `reduce_range` or U8U8. Whether the Ryzen 7735HS has VNNI was not verified | ORT quantization docs; 8-bit "within 1% of the floating-point baseline on all networks studied, including ... MobileNets" arXiv 2004.09602 (v1 2020-04-20); PTQ "sufficient for achieving 8-bit quantization with close to floating-point accuracy" in most cases arXiv 2106.08295 (v1 2021-06-15) | `onnxruntime.quantization.quantize_static(model_input, model_output, calibration_data_reader=None, quant_format=QuantFormat.QDQ, op_types_to_quantize=None, per_channel=False, reduce_range=False, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, nodes_to_quantize=None, nodes_to_exclude=None, use_external_data_format=False, calibrate_method=CalibrationMethod.MinMax, calibration_providers=None, extra_options=None, calibration_cache_path=None)`. `CalibrationMethod`: `MinMax`, `Entropy`, `Percentile`, `Distribution`. Run `quant_pre_process` first ("works best when the tensor's shape is known"). Debug with `qdq_loss_debug` | MIT |
| **NNCF PTQ (OpenVINO, ONNX, PyTorch)** | Accuracy-aware PTQ with a drop budget | `model_type` supports only `transformer`; batch sizes above 1 are unsupported for some models during quantisation | NNCF docs and source | `nncf` 3.4.0 (2026-09-17): `nncf.quantize(model, calibration_dataset, *, mode=None, preset=None, target_device=TargetDevice.ANY, subset_size=300, fast_bias_correction=True, model_type=None, ignored_scope=None, advanced_parameters=None)`; `nncf.quantize_with_accuracy_control(model, calibration_dataset, validation_dataset, validation_fn, *, max_drop=0.01, drop_type=DropType.ABSOLUTE, ..., subset_size=300, ...)`; `nncf.Dataset(data_source, transform_fn)` | Apache-2.0 |
| **PyTorch-native quantisation (torchao)** | PT2E flows (`prepare_pt2e` / `convert_pt2e`) and QAT | `torch.ao.quantization` is deprecated (warning seen locally): migrate | local DeprecationWarning text (links pytorch/ao issue 2259) | `torchao` 0.18.0 (2026-08-03; PyPI licence field empty) | not verified |
| **QAT** | Below 8 bits, or when PTQ loses too much | Needs training data and time | arXiv 2106.08295 (v1 2021-06-15) | NNCF, torchao, ModelOpt; `brevitas` 0.13.4 (Xilinx; licence field empty) | varies |
| **ViT-specific PTQ** | ViT activations (post-softmax, post-GELU) are hard to quantise uniformly | Plain min-max PTQ can lose accuracy on ViTs | arXiv 2111.12293 (v1 2021-11-24); arXiv 2212.08254 (v1 2022-12-16) | Research code; NNCF `model_type=TRANSFORMER` | - |
| **FP8** | Training or inference on hardware with FP8 support | For on-device inference, FP formats are "somewhere between 50-180% less efficient in terms of compute in dedicated hardware than the INT" formats | arXiv 2209.05433 (v1 2022-09-12); arXiv 2303.17951 (v1 2023-03-31) | TensorRT via ModelOpt | - |

**Accuracy-drop checks and calibration set (E3)**
- Compare the quantised model on the full validation set with a paired bootstrap against fp32. Check the task metric, per-class metrics and calibration: INT8 changes the score distributions, so re-fit TS and thresholds if they drift (reasoning).
- **Calibration data:** use unaugmented images representative of deployment (reasoning). NNCF's default `subset_size=300` is the only verified default. The ORT docs give no size guidance, and TensorRT 11 has no runtime calibrators.
- Exclude sensitive layers (first conv, final classifier) through `nodes_to_exclude` / `ignored_scope` when the drop is too large (practice).

### E4. Latency measurement and parity tests

**Methodology**
- Warm up first; this run used 10 warm-up iterations.
- Measure batch-1 latency percentiles (p50, p90, p99) separately from throughput at the deployment batch size.
- Pin the number of threads, and report the hardware, runtime version and input size.
- Exclude preprocessing, or measure it separately.
- Use enough runs for stable percentiles; p99 needs hundreds of runs (practice).
- MLPerf Inference defines scenario rules for comparable measurements: arXiv 1911.02549 (v1 2019-11-06); the scenario details were not re-read.
- **`torch.utils.benchmark.Timer(stmt, setup, globals=None, ..., num_threads=1)` defaults to a single thread.** Set `num_threads` to match deployment, and use `.blocked_autorange(min_run_time=0.2)`.

**Local reference (not a benchmark)**

| Runtime (ResNet-18, 224 px, 4 threads, batch 1) | p50 | p90 | max (of 50) |
|---|---|---|---|
| ONNX Runtime CPU | 31.2 ms | 35.0 ms | 35.6 ms |
| PyTorch eager | 62.2 ms | 66.3 ms | 77.0 ms |

**Parity tests**
- `torch.testing.assert_close` default tolerances (docstring): float32 rtol 1.3e-6 / atol 1e-5; float16 rtol 1e-3 / atol 1e-5; bfloat16 rtol 1.6e-2 / atol 1e-5; float64 1e-7 / 1e-7.
- Local ORT-vs-PyTorch fp32 on ResNet-18 (batch 8): max absolute difference 2.15e-06, max relative difference 1.15e-05, argmax agreement 1.000. It passed at the default fp32 tolerances.
- Recommended gates:
  1. Compare tensors on a fixed batch at fp32 tolerances (or rtol 1e-3 / atol 1e-4 when graph optimisations reorder operations).
  2. Measure top-1 agreement at or above 99.9% (practice) and the task-metric difference with a CI on the full validation set.
  3. For INT8 or FP16 models, compare task metrics, not tensors.
  4. Test dynamic batch sizes and edge input sizes.
  5. `torch.onnx.export(..., verify=True)` checks the export with ONNX Runtime (docs).

### E5. Monitoring after deployment

| Method | When it helps | When it hurts | Evidence | Library + verified API | Licence |
|---|---|---|---|---|---|
| **Embedding / output drift with two-sample tests** | Detecting input shift without labels. "A two-sample-testing-based approach, using pre-trained classifiers for dimensionality reduction, performs best"; domain classifiers help characterise shifts | Alarms on harmless shift. Drift does not equal an accuracy drop | arXiv 1810.11953 (v1 2018-10-29); MMD arXiv 0805.2368 (v1 2008-05-15) | `evidently` 0.7.23 (2026-09-11): `EmbeddingsDriftMetric` / `TestEmbeddingsDrift` (**legacy API only**; "coming soon to the new Evidently API") with `drift_method` = `model()` (default: domain classifier; drift when ROC AUC > 0.55 above 1,000 observations, or above the random classifier's 95th percentile at or below 1,000), `ratio()`, `distance()` or `mmd()`. MONAI `MMDMetric` (dev listing) | Apache-2.0 |
| **Label-shift monitoring (BBSE)** | Estimating how class frequencies changed, from the black-box predictions' confusion matrix | Assumes pure label shift | arXiv 1802.03916 (v1 2018-02-12); arXiv 1901.06852 (v1 2019-01-21) | About 20 lines | - |
| **Label-free performance estimation (label delay)** | Estimating accuracy before labels arrive: ATC is 2-4x more accurate than earlier methods; agreement-on-the-line; DoC; AutoEval | "Identifying the accuracy is just as hard as identifying the optimal predictor" in general, so the methods rest on assumptions | arXiv 2201.04234 (v1 2022-01-11); arXiv 2206.13089 (v1 2022-06-27); arXiv 2107.03315 (v1 2021-07-07); arXiv 2007.02915 (v1 2020-07-06) | `nannyml` 0.13.1 (2025-07-12; requires Python below 3.13; CBPE API not re-verified) | Apache-2.0 |
| **Other drift libraries** | - | **`alibi-detect` 0.13.0 is Business Source License 1.1**; `deepchecks` 0.19.1 is AGPL; `frouros` 0.9.0 (BSD-3, requires Python below 3.13, last release 2024-10); `whylogs` 1.6.4 (Apache-2.0, last release 2024-12) | PyPI | - | mixed |

**Re-training triggers (decision guidance, reasoning built on the sources above).**
1. Monitor input drift on penultimate embeddings (domain classifier, or MMD on a fixed reference window), prediction-distribution shift (BBSE), confidence summaries (ATC), and the OOD-score rate (D4).
2. Trigger a **labelled audit sample**, not an automatic retrain, when drift persists over consecutive windows. Size the audit to give a CI narrow enough to detect the tolerated drop (D2).
3. Retrain, or re-fit calibration and priors (B3, B4), when the audited metric's CI falls below the acceptance threshold.
4. Keep the original test set and the audit sets fixed for regression tests. Avoid online test-time adaptation in production unless it has a reset policy (A6).

---

## Consolidated library status (verified 2026-09-28)

| Package | Version (upload date) | Licence | Status note |
|---|---|---|---|
| torch / torchvision | 2.14.0 / 0.29.0 (2026-09-02) | BSD-style | `torch.optim.Muon`, FSDP2 `fully_shard`; ONNX dynamo default; TorchScript deprecated |
| timm | 1.0.30 (2026-09-22) | Apache-2.0 | Muon/AdaMuon/Lion/LAMB/ADOPT/Kron/MARS; no SAM, no Schedule-Free, no WSD scheduler |
| torchmetrics | 1.9.0 (2026-03-09) | Apache-2.0 | Watch the defaults: macro accuracy, BootStrapper, no HD95 |
| optuna | 5.0.0 (2026-09-07) | MIT | Multivariate TPE and constant liar on by default |
| ray | 2.58.0 (2026-08-23) | Apache-2.0 | ASHA, PBT, PB2 |
| schedulefree | 1.4.1 (2025-03-24) | Apache-2.0 | Repo active (2026-07-28) |
| lion-pytorch | 0.2.5 (2026-07-09) | MIT | - |
| torch-lr-finder | 0.2.2 (2024-09-21) | MIT | - |
| cleanlab | 2.9.0 (2026-01-13) | Apache-2.0 | - |
| semilearn (USB) | 0.3.2 (2023-11-10) | MIT (repo) | PyPI stale; repo active |
| segmentation-models-pytorch | 0.5.0 (2025-04-17) | MIT | Loss suite |
| kornia | 0.8.3 (2026-05-19) | Apache-2.0 | Loss suite |
| monai | 1.6.0 (2026-06-22) | Apache-2.0 | Losses, HD95, NSD, PQ, sliding window, MetricsReloaded wrappers (dev listing) |
| nnunetv2 | 2.8.1 (2026-07-01) | Apache-2.0 | Sliding window 0.5 step + Gaussian + mirroring |
| pytorch-metric-learning | 2.9.0 (2025-08-17) | MIT | - |
| coral-pytorch | 1.4.0 (2022-07-18) | MIT | Stale |
| ensemble-boxes | 1.0.9 (2022-04-20) | MIT | Repo active (2026-07-27); normalised coordinates only |
| ttach | 0.0.3 (2020-07-09) | MIT | Stale |
| netcal | 1.4.0 (2026-04-16) | Apache-2.0 | Detection calibration, D-ECE |
| dirichletcal | 0.5.3 (2025-06-12) | MIT | Requires Python 3.12 or later |
| uncertainty-calibration | 0.1.4 (2022-11-11) | MIT | Stale |
| relplot | 1.0.3 (2025-11-04) | Apple custom | Custom licence |
| torch-uncertainty | 0.13.0 (2026-07-22) | Apache-2.0 | Not API-verified |
| lightning-uq-box | 0.3.0 (2026-08-23) | Apache-2.0 | Not API-verified |
| mapie | 1.5.0 (2026-08-05) | BSD-3 | v1 API `SplitConformalClassifier` |
| torchcp | 1.2.1 (2025-10-14) | **LGPL-3.0** (GitHub) | - |
| crepes / puncc | 0.9.1 / 0.9.3 (2026-06) | BSD / MIT | - |
| abstention | 0.1.3.1 (2020-01-22) | not stated | Stale |
| pydensecrf / pydensecrf2 / SimpleCRF | 1.0rc3 (2018) / 1.1 (2024-12-23) / 0.2.1.1 (2021) | MIT / MIT / BSD | Stale |
| supervision | 0.30.5 (2026-09-22) | MIT | Tracking utilities |
| boxmot | 25.0.0 (2026-09-09) | **AGPL-3.0** | Avoid under the permissive default |
| trackeval (PyPI) | 1.3.0 (2026-03-03) | MIT | **Third-party fork**, not official |
| motmetrics | 1.4.0 (2022-12-26) | MIT | Stale |
| pycocotools / faster-coco-eval / lvis | 2.0.11 / 1.8.0 / 0.5.3 | FreeBSD / (R4: Apache-2.0) / not stated | lvis stale |
| tidecv | 1.0.1 (2020-08-31) | MIT | Stale |
| fiftyone (+ brain, db) | 1.22.0 / 0.26.0 / 1.4.1 | Apache-2.0 | - |
| fairlearn | 0.14.0 (2026-06-07) | MIT | `MetricFrame` bootstrap CIs |
| captum | 0.9.0 (2026-04-17) | BSD-3 | - |
| grad-cam | 1.5.7 (2026-08-21) | MIT | ROAD metrics included |
| torchcam | 0.4.1 (2025-10-27) | Apache-2.0 | - |
| quantus / zennit | 0.6.0 / 1.0.0 | **GPL** / **GPL+LGPL** | Copyleft |
| medpy | 0.5.2 (2024-07-23) | **GPL** | Copyleft |
| surface-distance / seg-metrics | 0.1 / 1.2.8 | Apache-2.0 / MIT | - |
| robustbench / imagecorruptions / wilds | 1.1.1 / 1.1.2 / 2.0.0 | Apache-2.0 / Apache-2.0 / MIT | imagecorruptions and wilds stale |
| pytorch-ood | 0.3.3 (2026-09-07) | Apache-2.0 | - |
| onnx / onnxscript | 1.23.0 / 0.7.2 | Apache-2.0 / MIT | `onnxscript` needed by the default exporter |
| onnxruntime (-gpu / -directml / -openvino) | 1.30.0 / 1.30.0 / 1.24.4 / 1.24.1 | MIT | DirectML in sustained engineering |
| openvino / nncf | 2026.4.0 / 3.4.0 | Apache-2.0 | - |
| tensorrt / torch-tensorrt / nvidia-modelopt | 11.3.0.99 / 2.14.0 / 0.47.0 | proprietary / BSD / Apache-2.0 | TRT 11: no INT8 calibrators |
| executorch | 1.5.1 (2026-09-22) | BSD-3 | - |
| coremltools | 9.0 (2025-11-10) | BSD | Recommends jit.trace (deprecated in torch) |
| litert-torch / ai-edge-torch / ai-edge-litert | 0.9.4 / 0.7.2 (deprecated) / 2.2.0 | Apache-2.0 | litert-torch is Linux only |
| torchao / brevitas / neural-compressor | 0.18.0 / 0.13.4 / 3.9 | not stated / not stated / Apache-2.0 | - |
| nvidia-dali-cuda120/130 | 2.3.0 (2026-08-28) | Apache-2.0 | Linux x64 + NVIDIA GPU |
| ffcv | 1.0.2 (2023-03-05) | Apache-2.0 (repo) | Stale |
| torchcodec | 0.16.0 (2026-08-13) | not stated on PyPI | - |
| evidently | 0.7.23 (2026-09-11) | Apache-2.0 | Embedding drift in the legacy API only |
| nannyml / frouros / whylogs | 0.13.1 / 0.9.0 / 1.6.4 | Apache-2.0 / BSD-3 / Apache-2.0 | nannyml and frouros require Python below 3.13 |
| alibi-detect | 0.13.0 (2025-12-11) | **BSL 1.1** | Source-available |
| deepchecks | 0.19.1 (2024-12-15) | **AGPL** | Copyleft |
| scikit-learn / scipy / statsmodels | 1.9.1 / 1.18.1 / 0.15.0 | BSD-3 | sklearn TS calibrator since 1.8 |

---

## Could not verify

1. **The default opset of the dynamo ONNX exporter in torch 2.14.** The docs say "Leave as default (None) to use the recommended version" without a number, and `onnxscript` is not installed, so it could not be tested.
2. **Muon and Schedule-Free for fine-tuning pretrained vision backbones on small data.** No study found beyond the Muon-mismatch paper (arXiv 2605.10468, which covers full FT versus LoRA) and ViT from-scratch results. The same applies to WSD schedules for vision fine-tuning.
3. **Lion's exact LR and weight-decay rescaling rule** (the commonly quoted "3-10x smaller LR, larger WD"). Only "requires a smaller learning rate than Adam" was verified, from the abstract.
4. **The `davda54/sam` constructor signature** (`SAM(params, base_optimizer, rho, adaptive, ...)`) was not re-read.
5. **Metrics Reloaded's default image-level counting metric** (for example Expected Cost versus Balanced Accuracy versus MCC). Only the multi-threshold (AUROC; AP under imbalance) and calibration (ECE or proper scoring rules) recommendations were verified.
6. **Formal definitions of AbsRel and δ1** were not re-read in Eigen et al. (arXiv 1406.2283, paper body); the standard definitions are given.
7. **Seed-variance magnitudes** for fine-tuned vision models (numbers in arXiv 2109.08203 and 2304.01910 are in the paper bodies, not the abstracts).
8. **USB's quantitative findings** on SSL with pretrained ViTs (paper body).
9. **Detection-specific claims:**
   - The effect of adding NMS on top of DETR outputs.
   - Evidence for instance-mask voting or averaging.
   - Whether WBF beats NMW, soft-NMS and NMS in general (the WBF abstract only claims top challenge results).
   - FiftyOne's `compute_mAP` keyword.
10. **Library behaviour and licences not confirmed:**
    - `tidecv` compatibility with NumPy 2.
    - `pydensecrf` / `pydensecrf2` wheels for Python 3.12 on Windows.
    - FFCV on Windows.
    - Licences for `torchao`, `brevitas` and `torchcodec` (empty PyPI licence fields).
    - The LDAM-DRW, ELR and LIFT repository licences.
    - The detectron2 repository licence (not queried through the API; widely known as Apache-2.0).
11. **Deployment details:**
    - Whether the Ryzen 7 7735HS supports AVX-VNNI or AVX512-VNNI (it determines the ORT U8S8 saturation risk).
    - OpenVINO performance on AMD CPUs.
    - DirectML on the Radeon 680M in this environment.
    - WinML Python availability.
    - A TensorRT calibration-set size recommendation (moot in TRT 11).
12. **Tracking library APIs:** the class names of the `supervision` ByteTrack implementation.
13. **Other APIs not re-verified:** nannyml CBPE, Captum `TCAV`, the huggingface-hub model card API, and `nnUNetv2_ensemble`.
14. **Metric and threshold details:**
    - Whether torchmetrics multi-label AP skips labels without positives.
    - Whether `TunedThresholdClassifierCV` supports only binary targets (the docstring was not read for this).
15. **The MLPerf Inference scenario rules** (for example percentile targets per scenario) were not re-read from the paper body.
16. **MONAI features read from the `dev` branch listing** (`MetricsReloadedBinary`/`Categorical`, `CalibrationErrorMetric`, `MMDMetric`, loss names) may be newer than the 1.6.0 release. `HausdorffDistanceMetric` was read at the 1.6.0 tag.
17. **Why the local bf16 CPU run was so slow** was not investigated (no native bf16 hardware is suspected; not verified). The measurement is a single run on a shared machine.

---

## Appendix: arXiv references verified through the arXiv API (2026-09-28)

All 319 arXiv IDs cited above, ordered by v1 date. The title and v1 date come directly from `export.arxiv.org/api/query?id_list=...`.

| arXiv ID | v1 date | Title | First author |
|---|---|---|---|
| 0805.2368 | 2008-05-15 | A Kernel Method for the Two-Sample Problem | Arthur Gretton |
| 1210.5644 | 2012-10-20 | Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials | Philipp Krähenbühl |
| 1211.1323 | 2012-11-06 | Sample Size Planning for Classification Models | Claudia Beleites |
| 1402.1892 | 2014-02-08 | Thresholding Classifiers to Maximize F1 Score | Zachary Chase Lipton |
| 1406.2283 | 2014-06-09 | Depth Map Prediction from a Single Image using a Multi-Scale Deep Network | David Eigen |
| 1503.02531 | 2015-03-09 | Distilling the Knowledge in a Neural Network | Geoffrey Hinton |
| 1506.01186 | 2015-06-03 | Cyclical Learning Rates for Training Neural Networks | Leslie N. Smith |
| 1512.00567 | 2015-12-02 | Rethinking the Inception Architecture for Computer Vision | Christian Szegedy |
| 1603.04779 | 2016-03-15 | Revisiting Batch Normalization For Practical Domain Adaptation | Yanghao Li |
| 1603.06560 | 2016-03-21 | Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization | Lisha Li |
| 1603.09382 | 2016-03-30 | Deep Networks with Stochastic Depth | Gao Huang |
| 1604.03540 | 2016-04-12 | Training Region-based Object Detectors with Online Hard Example Mining | Abhinav Shrivastava |
| 1604.06174 | 2016-04-21 | Training Deep Nets with Sublinear Memory Cost | Tianqi Chen |
| 1606.00915 | 2016-06-02 | DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs | Liang-Chieh Chen |
| 1606.04316 | 2016-06-14 | Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis | Alessio Benavoli |
| 1606.04797 | 2016-06-15 | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation | Fausto Milletari |
| 1608.03983 | 2016-08-13 | SGDR: Stochastic Gradient Descent with Warm Restarts | Ilya Loshchilov |
| 1610.02136 | 2016-10-07 | A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks | Dan Hendrycks |
| 1610.02391 | 2016-10-07 | Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization | Ramprasaath R. Selvaraju |
| 1611.03530 | 2016-11-10 | Understanding deep learning requires rethinking generalization | Chiyuan Zhang |
| 1612.01474 | 2016-12-05 | Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles | Balaji Lakshminarayanan |
| 1703.01365 | 2017-03-04 | Axiomatic Attribution for Deep Networks | Mukund Sundararajan |
| 1703.01780 | 2017-03-06 | Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results | Antti Tarvainen |
| 1703.07737 | 2017-03-22 | In Defense of the Triplet Loss for Person Re-Identification | Alexander Hermans |
| 1704.00109 | 2017-04-01 | Snapshot Ensembles: Train 1, get M for free | Gao Huang |
| 1704.04503 | 2017-04-14 | Soft-NMS -- Improving Object Detection With One Line of Code | Navaneeth Bodla |
| 1705.02950 | 2017-05-08 | Learning non-maximum suppression | Jan Hosang |
| 1705.08500 | 2017-05-23 | Selective Classification for Deep Neural Networks | Yonatan Geifman |
| 1705.08790 | 2017-05-24 | The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks | Maxim Berman |
| 1706.02677 | 2017-06-08 | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour | Priya Goyal |
| 1706.02690 | 2017-06-08 | Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks | Shiyu Liang |
| 1706.04599 | 2017-06-14 | On Calibration of Modern Neural Networks | Chuan Guo |
| 1706.05587 | 2017-06-17 | Rethinking Atrous Convolution for Semantic Image Segmentation | Liang-Chieh Chen |
| 1706.05721 | 2017-06-18 | Tversky loss function for image segmentation using 3D fully convolutional deep networks | Seyed Sadegh Mohseni Salehi |
| 1707.03237 | 2017-07-11 | Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations | Carole H Sudre |
| 1708.02002 | 2017-08-07 | Focal Loss for Dense Object Detection | Tsung-Yi Lin |
| 1708.03888 | 2017-08-13 | Large Batch Training of Convolutional Networks | Yang You |
| 1708.07120 | 2017-08-23 | Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates | Leslie N. Smith |
| 1710.03740 | 2017-10-10 | Mixed Precision Training | Paulius Micikevicius |
| 1710.05381 | 2017-10-15 | A systematic study of the class imbalance problem in convolutional neural networks | Mateusz Buda |
| 1710.11063 | 2017-10-30 | Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks | Aditya Chattopadhyay |
| 1711.00489 | 2017-11-01 | Don't Decay the Learning Rate, Increase the Batch Size | Samuel L. Smith |
| 1711.00867 | 2017-11-02 | The (Un)reliability of saliency methods | Pieter-Jan Kindermans |
| 1711.05101 | 2017-11-14 | Decoupled Weight Decay Regularization | Ilya Loshchilov |
| 1711.09846 | 2017-11-27 | Population Based Training of Neural Networks | Max Jaderberg |
| 1711.11279 | 2017-11-30 | Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) | Been Kim |
| 1801.00868 | 2018-01-03 | Panoptic Segmentation | Alexander Kirillov |
| 1801.07698 | 2018-01-23 | ArcFace: Additive Angular Margin Loss for Deep Face Recognition | Jiankang Deng |
| 1801.09414 | 2018-01-29 | CosFace: Large Margin Cosine Loss for Deep Face Recognition | Hao Wang |
| 1802.03916 | 2018-02-12 | Detecting and Correcting for Label Shift with Black Box Predictors | Zachary C. Lipton |
| 1802.10026 | 2018-02-27 | Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs | Timur Garipov |
| 1803.05407 | 2018-03-14 | Averaging Weights Leads to Wider Optima and Better Generalization | Pavel Izmailov |
| 1803.09010 | 2018-03-23 | Datasheets for Datasets | Timnit Gebru |
| 1803.11285 | 2018-03-29 | Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking | Filip Radenović |
| 1804.06872 | 2018-04-18 | Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels | Bo Han |
| 1804.09170 | 2018-04-24 | Realistic Evaluation of Deep Semi-Supervised Learning Algorithms | Avital Oliver |
| 1805.04770 | 2018-05-12 | Born Again Neural Networks | Tommaso Furlanello |
| 1805.07836 | 2018-05-20 | Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels | Zhilu Zhang |
| 1806.10758 | 2018-06-28 | A Benchmark for Interpretability Methods in Deep Neural Networks | Sara Hooker |
| 1807.03888 | 2018-07-10 | A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks | Kimin Lee |
| 1807.07356 | 2018-07-19 | Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks | Guotai Wang |
| 1809.04430 | 2018-09-12 | Deep learning to achieve clinically applicable segmentation of head and neck anatomy for radiotherapy | Stanislav Nikolov |
| 1809.09446 | 2018-09-25 | Nested cross-validation when selecting classifiers is overzealous for most practical applications | Jacques Wainer |
| 1809.10486 | 2018-09-27 | nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation | Fabian Isensee |
| 1810.03993 | 2018-10-05 | Model Cards for Model Reporting | Margaret Mitchell |
| 1810.03292 | 2018-10-08 | Sanity Checks for Saliency Maps | Julius Adebayo |
| 1810.05934 | 2018-10-13 | A System for Massively Parallel Hyperparameter Tuning | Liam Li |
| 1810.07842 | 2018-10-18 | A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation | Nabila Abraham |
| 1810.11953 | 2018-10-29 | Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift | Stephan Rabanser |
| 1810.12281 | 2018-10-29 | Three Mechanisms of Weight Decay Regularization | Guodong Zhang |
| 1812.01187 | 2018-12-04 | Bag of Tricks for Image Classification with Convolutional Neural Networks | Tong He |
| 1812.07032 | 2018-12-17 | Boundary loss for highly unbalanced segmentation | Hoel Kervadec |
| 1901.05555 | 2019-01-16 | Class-Balanced Loss Based on Effective Number of Samples | Yin Cui |
| 1901.07884 | 2019-01-20 | Rank consistent ordinal regression for neural networks with application to age estimation | Wenzhi Cao |
| 1901.06852 | 2019-01-21 | Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation | Amr Alexandari |
| 1901.09192 | 2019-01-26 | SelectiveNet: A Deep Neural Network with an Integrated Reject Option | Yonatan Geifman |
| 1902.02476 | 2019-02-07 | A Simple Baseline for Bayesian Uncertainty in Deep Learning | Wesley Maddox |
| 1902.03393 | 2019-02-09 | Improved Knowledge Distillation via Teacher Assistant | Seyed-Iman Mirzadeh |
| 1902.10811 | 2019-02-13 | Do ImageNet Classifiers Generalize to ImageNet? | Benjamin Recht |
| 1902.06977 | 2019-02-19 | Evaluating model calibration in classification | Juozas Vaicenavicius |
| 1902.09630 | 2019-02-25 | Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression | Hamid Rezatofighi |
| 1903.04197 | 2019-03-11 | Structured Knowledge Distillation for Dense Prediction | Yifan Liu |
| 1903.09734 | 2019-03-22 | Regularized Learning for Domain Adaptation under Label Shifts | Kamyar Azizzadenesheli |
| 1903.12261 | 2019-03-28 | Benchmarking Neural Network Robustness to Common Corruptions and Perturbations | Dan Hendrycks |
| 1904.00962 | 2019-04-01 | Large Batch Optimization for Deep Learning: Training BERT in 76 minutes | Yang You |
| 1904.01685 | 2019-04-02 | Measuring Calibration in Deep Learning | Jeremy Nixon |
| 1904.03629 | 2019-04-07 | Adaptive NMS: Refining Pedestrian Detection in a Crowd | Songtao Liu |
| 1905.00076 | 2019-04-30 | Ensemble Distribution Distillation | Andrey Malinin |
| 1905.11881 | 2019-05-28 | Why gradient clipping accelerates training: A theoretical justification for adaptivity | Jingzhao Zhang |
| 1905.13549 | 2019-05-29 | Learning Robust Global Representations by Penalizing Local Predictive Power | Haohan Wang |
| 1906.02530 | 2019-06-06 | Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift | Yaniv Ovadia |
| 1906.02629 | 2019-06-06 | When Does Label Smoothing Help? | Rafael Müller |
| 1906.03609 | 2019-06-09 | Distilling Object Detectors with Fine-grained Feature Imitation | Tao Wang |
| 1906.07413 | 2019-06-18 | Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss | Kaidi Cao |
| 1907.01341 | 2019-07-02 | Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer | René Ranftl |
| 1907.07174 | 2019-07-16 | Natural Adversarial Examples | Dan Hendrycks |
| 1907.07484 | 2019-07-17 | Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming | Claudio Michaelis |
| 1907.10902 | 2019-07-25 | Optuna: A Next-generation Hyperparameter Optimization Framework | Takuya Akiba |
| 1908.03195 | 2019-08-08 | LVIS: A Dataset for Large Vocabulary Instance Segmentation | Agrim Gupta |
| 1908.04913 | 2019-08-14 | FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age | Kimmo Kärkkäinen |
| 1908.06112 | 2019-08-16 | Symmetric Cross Entropy for Robust Learning with Noisy Labels | Yisen Wang |
| 1909.10155 | 2019-09-23 | Verified Uncertainty Calibration | Ananya Kumar |
| 1909.12475 | 2019-09-27 | Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning for Medical Imaging | Luke Oakden-Rayner |
| 1910.01279 | 2019-10-03 | Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks | Haofan Wang |
| 1910.01348 | 2019-10-03 | On the Efficacy of Knowledge Distillation | Jang Hyun Cho |
| 1910.09217 | 2019-10-21 | Decoupling Representation and Classifier for Long-Tailed Recognition | Bingyi Kang |
| 1910.11385 | 2019-10-24 | Calibration tests in multi-class classification: A unifying framework | David Widmann |
| 1910.12656 | 2019-10-28 | Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration | Meelis Kull |
| 1910.13302 | 2019-10-29 | Weighted boxes fusion: Ensembling boxes from different object detection models | Roman Solovyev |
| 1911.00068 | 2019-10-31 | Confident Learning: Estimating Uncertainty in Dataset Labels | Curtis G. Northcutt |
| 1911.02549 | 2019-11-06 | MLPerf Inference Benchmark | Vijay Janapa Reddi |
| 1911.04252 | 2019-11-11 | Self-training with Noisy Student improves ImageNet classification | Qizhe Xie |
| 1911.08287 | 2019-11-19 | Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression | Zhaohui Zheng |
| 1912.01451 | 2019-11-29 | Sanity Checks for Saliency Metrics | Richard Tomsett |
| 1912.02413 | 2019-12-05 | BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition | Boyan Zhou |
| 1912.02757 | 2019-12-05 | Deep Ensembles: A Loss Landscape Perspective | Stanislav Fort |
| 1912.08193 | 2019-12-17 | PointRend: Image Segmentation as Rendering | Alexander Kirillov |
| 2001.07685 | 2020-01-21 | FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence | Kihyuk Sohn |
| 2003.00827 | 2020-02-14 | CheXclusion: Fairness gaps in deep chest X-ray classifiers | Laleh Seyyed-Kalantari |
| 2002.06305 | 2020-02-15 | Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping | Jesse Dodge |
| 2002.06470 | 2020-02-15 | Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning | Arsenii Ashukha |
| 2002.07394 | 2020-02-18 | DivideMix: Learning with Noisy Labels as Semi-supervised Learning | Junnan Li |
| 2002.11770 | 2020-02-19 | Rethinking the Hyperparameters for Fine-tuning | Hao Li |
| 2002.09437 | 2020-02-21 | Calibrating Deep Neural Networks using Focal Loss | Jishnu Mukhoti |
| 2003.02819 | 2020-03-05 | Does label smoothing mitigate label noise? | Michal Lukasik |
| 2003.07311 | 2020-03-16 | clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation | Suprosanna Shit |
| 2003.07329 | 2020-03-16 | Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning | Jize Zhang |
| 2003.08505 | 2020-03-18 | A Metric Learning Reality Check | Kevin Musgrave |
| 2003.10152 | 2020-03-23 | SOLOv2: Dynamic and Fast Instance Segmentation | Xinlong Wang |
| 2004.09602 | 2020-04-20 | Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation | Hao Wu |
| 2004.11362 | 2020-04-23 | Supervised Contrastive Learning | Prannay Khosla |
| 2004.13546 | 2020-04-28 | Multivariate Confidence Calibration for Object Detection | Fabian Küppers |
| 2005.00928 | 2020-05-02 | Quantifying Attention Flow in Transformers | Samira Abnar |
| 2005.02551 | 2020-05-06 | CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement | Ho Kei Cheng |
| 2005.12872 | 2020-05-26 | End-to-End Object Detection with Transformers | Nicolas Carion |
| 2005.13449 | 2020-05-27 | Segmentation Loss Odyssey | Jun Ma |
| 2006.02544 | 2020-06-03 | Classification with Valid and Adaptive Coverage | Yaniv Romano |
| 2006.04388 | 2020-06-08 | Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection | Xiang Li |
| 2006.10029 | 2020-06-17 | Big Self-Supervised Models are Strong Semi-Supervised Learners | Ting Chen |
| 2006.10726 | 2020-06-18 | Tent: Fully Test-time Adaptation by Entropy Minimization | Dequan Wang |
| 2006.16241 | 2020-06-29 | The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization | Dan Hendrycks |
| 2007.00151 | 2020-06-30 | Early-Learning Regularization Prevents Memorization of Noisy Labels | Sheng Liu |
| 2007.00644 | 2020-07-01 | Measuring Robustness to Natural Distribution Shifts in Image Classification | Rohan Taori |
| 2007.01434 | 2020-07-02 | In Search of Lost Domain Generalization | Ishaan Gulrajani |
| 2007.01547 | 2020-07-03 | Descending through a Crowded Valley - Benchmarking Deep Learning Optimizers | Robin M. Schmidt |
| 2007.02915 | 2020-07-06 | Are Labels Always Necessary for Classifier Accuracy Evaluation? | Weijian Deng |
| 2007.04269 | 2020-07-08 | SegFix: Model-Agnostic Boundary Refinement for Segmentation | Yuhui Yuan |
| 2007.04612 | 2020-07-09 | Concept Bottleneck Models | Pang Wei Koh |
| 2007.07314 | 2020-07-14 | Long-tail learning via logit adjustment | Aditya Krishna Menon |
| 2007.08199 | 2020-07-16 | Learning from Noisy Labels with Deep Neural Networks: A Survey | Hwanjun Song |
| 2007.10740 | 2020-07-21 | Balanced Meta-Softmax for Long-Tailed Visual Recognition | Jiawei Ren |
| 2008.00299 | 2020-08-01 | Eigen-CAM: Class Activation Map using Principal Components | Mohammed Bany Muhammad |
| 2008.02766 | 2020-08-06 | Assessing the (Un)Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging | Nishanth Arun |
| 2008.05105 | 2020-08-12 | Local Temperature Scaling for Probability Calibration | Zhipeng Ding |
| 2008.08115 | 2020-08-18 | TIDE: A General Toolbox for Identifying Object Detection Errors | Daniel Bolya |
| 2008.10032 | 2020-08-23 | Seesaw Loss for Long-Tailed Instance Segmentation | Jiaqi Wang |
| 2008.13367 | 2020-08-31 | VarifocalNet: An IoU-aware Dense Object Detector | Haoyang Zhang |
| 2009.14119 | 2020-09-29 | Asymmetric Loss For Multi-Label Classification | Emanuel Ben-Baruch |
| 2009.14193 | 2020-09-29 | Uncertainty Sets for Image Classifiers using Conformal Prediction | Anastasios Angelopoulos |
| 2010.01412 | 2020-10-03 | Sharpness-Aware Minimization for Efficiently Improving Generalization | Pierre Foret |
| 2010.03759 | 2020-10-08 | Energy-based Out-of-distribution Detection | Weitang Liu |
| 2010.06866 | 2020-10-14 | Deep Ensembles for Low-Data Transfer Learning | Basil Mustafa |
| 2010.11422 | 2020-10-22 | Learning Loss for Test-Time Augmentation | Ildoo Kim |
| 2011.08891 | 2020-11-17 | Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks | Rachel Lea Draelos |
| 2011.11156 | 2020-11-23 | Better Aggregation in Test-Time Augmentation | Divya Shanmugam |
| 2011.13256 | 2020-11-26 | Channel-wise Knowledge Distillation for Dense Prediction | Changyong Shu |
| 2012.01988 | 2020-12-03 | Wisdom of Committees: An Overlooked Approach To Faster and More Accurate Models | Xiaofang Wang |
| 2012.07421 | 2020-12-14 | WILDS: A Benchmark of in-the-Wild Distribution Shifts | Pang Wei Koh |
| 2012.08668 | 2020-12-15 | Mitigating Bias in Calibration Error Estimation | Rebecca Roelofs |
| 2012.09816 | 2020-12-17 | Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning | Zeyuan Allen-Zhu |
| 2012.09838 | 2020-12-17 | Transformer Interpretability Beyond Attention Visualization | Hila Chefer |
| 2012.12877 | 2020-12-23 | Training data-efficient image transformers & distillation through attention | Hugo Touvron |
| 2102.01066 | 2021-02-01 | Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details | Achal Dave |
| 2102.04525 | 2021-02-08 | Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation | Michael Yeung |
| 2102.06171 | 2021-02-11 | High-Performance Large-Scale Image Recognition Without Normalization | Andrew Brock |
| 2102.06356 | 2021-02-12 | A Large Batch Optimizer Reality Check: Traditional, Generic Optimizers Suffice Across Batch Sizes | Zachary Nado |
| 2102.09480 | 2021-02-18 | Unbiased Teacher for Semi-Supervised Object Detection | Yen-Cheng Liu |
| 2102.11600 | 2021-02-23 | ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks | Jungmin Kwon |
| 2102.12182 | 2021-02-24 | Parameterized Temperature Scaling for Boosting the Expressive Power in Post-Hoc Uncertainty Calibration | Christian Tomani |
| 2102.12252 | 2021-02-24 | Localization Distillation for Dense Object Detection | Zhaohui Zheng |
| 2103.03098 | 2021-03-01 | Accounting for Variance in Machine Learning Benchmarks | Xavier Bouthillier |
| 2103.07461 | 2021-03-12 | Probabilistic two-stage detection | Xingyi Zhou |
| 2103.14749 | 2021-03-26 | Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks | Curtis G. Northcutt |
| 2106.01548 | 2021-06-03 | When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations | Xiangning Chen |
| 2106.03004 | 2021-06-06 | Exploring the Limits of Out-of-Distribution Detection | Stanislav Fort |
| 2106.04149 | 2021-06-08 | To Smooth or Not? When Label Smoothing Meets Noisy Labels | Jiaheng Wei |
| 2106.05237 | 2021-06-09 | Knowledge distillation: A good teacher is patient and consistent | Lucas Beyer |
| 2106.05945 | 2021-06-10 | Does Knowledge Distillation Really Work? | Samuel Stanton |
| 2106.07998 | 2021-06-15 | Revisiting the Calibration of Modern Neural Networks | Matthias Minderer |
| 2106.08295 | 2021-06-15 | A White Paper on Neural Network Quantization | Markus Nagel |
| 2106.09018 | 2021-06-16 | End-to-End Semi-Supervised Object Detection with Soft Teacher | Mengde Xu |
| 2107.00758 | 2021-07-01 | The Spotlight: A General Method for Discovering Systematic Errors in Deep Learning Models | Greg d'Eon |
| 2107.03315 | 2021-07-07 | Predicting with Confidence on Unseen Distributions | Devin Guillory |
| 2107.04649 | 2021-07-09 | Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization | John Miller |
| 2107.07511 | 2021-07-15 | A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification | Anastasios N. Angelopoulos |
| 2107.08353 | 2021-07-18 | Top-label calibration and multiclass-to-binary reductions | Chirag Gupta |
| 2107.10356 | 2021-07-21 | Reading Race: AI Recognises Patient's Racial Identity In Medical Images | Imon Banerjee |
| 2108.13122 | 2021-08-30 | Tune It or Don't Use It: Benchmarking Data-Efficient Image Classification | Lorenzo Brigato |
| 2109.01903 | 2021-09-04 | Robust fine-tuning of zero-shot models | Mitchell Wortsman |
| 2109.08203 | 2021-09-16 | Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision | David Picard |
| 2110.00476 | 2021-10-01 | ResNet strikes back: An improved training procedure in timm | Ross Wightman |
| 2110.01052 | 2021-10-03 | Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control | Anastasios N. Angelopoulos |
| 2110.04596 | 2021-10-09 | Deep Long-Tailed Learning: A Survey | Yifan Zhang |
| 2110.06864 | 2021-10-13 | ByteTrack: Multi-Object Tracking by Associating Every Detection Box | Yifu Zhang |
| 2110.08263 | 2021-10-15 | FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling | Bowen Zhang |
| 2110.09506 | 2021-10-18 | MEMO: Test Time Robustness via Adaptation and Augmentation | Marvin Zhang |
| 2110.12899 | 2021-10-20 | No One Representation to Rule Them All: Overlapping Features of Training Methods | Raphael Gontijo-Lopes |
| 2110.12088 | 2021-10-22 | Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations | Jiaheng Wei |
| 2111.08851 | 2021-11-17 | Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities | Xintong Shi |
| 2111.11837 | 2021-11-23 | Focal and Global Knowledge Distillation for Detectors | Zhendong Yang |
| 2111.12293 | 2021-11-24 | PTQ4ViT: Post-training quantization for vision transformers with twin uniform quantization | Zhihang Yuan |
| 2111.12797 | 2021-11-24 | ReAct: Out-of-distribution Detection With Rectified Activations | Yiyou Sun |
| 2201.04234 | 2022-01-11 | Leveraging Unlabeled Data to Predict Out-of-Distribution Performance | Saurabh Garg |
| 2202.00449 | 2022-02-01 | A Consistent and Efficient Evaluation Strategy for Attribution Methods | Yao Rong |
| 2202.06861 | 2022-02-14 | Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond | Anna Hedström |
| 2202.06985 | 2022-02-14 | Deep Ensembles Work, But Are They Necessary? | Taiga Abe |
| 2202.13514 | 2022-02-28 | StrongSORT: Make DeepSORT Great Again | Yunhao Du |
| 2203.02714 | 2022-03-05 | Towards Efficient and Scalable Sharpness-Aware Minimization | Yong Liu |
| 2203.03605 | 2022-03-07 | DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection | Hao Zhang |
| 2203.05482 | 2022-03-10 | Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time | Mitchell Wortsman |
| 2203.07835 | 2022-03-15 | Better Uncertainty Calibration via Proper Scores for Classification and Beyond | Sebastian G. Gruber |
| 2203.08679 | 2022-03-16 | Decoupled Knowledge Distillation | Borui Zhao |
| 2203.10807 | 2022-03-21 | ViM: Out-Of-Distribution with Virtual-logit Matching | Haoqi Wang |
| 2203.14960 | 2022-03-24 | Domino: Discovering Systematic Errors with Cross-Modal Embeddings | Sabri Eyuboglu |
| 2203.13591 | 2022-03-25 | Continual Test-Time Domain Adaptation | Qin Wang |
| 2203.14197 | 2022-03-27 | Long-Tailed Recognition via Weight Balancing | Shaden Alshammari |
| 2203.14360 | 2022-03-27 | Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking | Jinkun Cao |
| 2204.01075 | 2022-04-03 | Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI | Mahima Pushkarna |
| 2204.02610 | 2022-04-06 | Efficient Test-Time Model Adaptation without Forgetting | Shuaicheng Niu |
| 2204.06507 | 2022-04-13 | Out-of-Distribution Detection with Deep Nearest Neighbors | Yiyou Sun |
| 2204.07118 | 2022-04-14 | DeiT III: Revenge of the ViT | Hugo Touvron |
| 2204.12511 | 2022-04-26 | PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions | Zhaoqi Leng |
| 2205.07246 | 2022-05-15 | FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning | Yidong Wang |
| 2205.09739 | 2022-05-19 | Diverse Weight Averaging for Out-of-Distribution Generalization | Alexandre Ramé |
| 2205.10287 | 2022-05-20 | On the SDEs and Scaling Rules for Adaptive Gradient Algorithms | Sadhika Malladi |
| 2205.10536 | 2022-05-21 | Knowledge Distillation from A Stronger Teacher | Tao Huang |
| 2206.01653 | 2022-06-03 | Metrics reloaded: Recommendations for image analysis validation | Lena Maier-Hein |
| 2206.09034 | 2022-06-17 | Towards Better Selective Classification | Leo Feng |
| 2206.13089 | 2022-06-27 | Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift | Christina Baek |
| 2206.14651 | 2022-06-29 | BoT-SORT: Robust Associations Multi-Pedestrian Tracking | Nir Aharon |
| 2207.01242 | 2022-07-04 | Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection | Fabian Küppers |
| 2207.06211 | 2022-07-13 | Sample-dependent Adaptive Temperature Scaling for Improved Calibration | Tom Joy |
| 2207.07048 | 2022-07-14 | Leakage and the Reproducibility Crisis in ML-based Science | Sayash Kapoor |
| 2208.02814 | 2022-08-04 | Conformal Risk Control | Anastasios N. Angelopoulos |
| 2209.05433 | 2022-09-12 | FP8 Formats for Deep Learning | Paulius Micikevicius |
| 2209.09858 | 2022-09-20 | Extremely Simple Activation Shaping for Out-of-Distribution Detection | Andrija Djurisic |
| 2210.01725 | 2022-10-04 | MEDFAIR: Benchmarking Fairness for Medical Imaging | Yongshuo Zong |
| 2210.07242 | 2022-10-13 | OpenOOD: Benchmarking Generalized Out-of-Distribution Detection | Jingkang Yang |
| 2210.07810 | 2022-10-13 | A Consistent and Differentiable Lp Canonical Calibration Error Estimator | Teodora Popordanoska |
| 2210.16315 | 2022-10-28 | Beyond calibration: estimating the grouping loss of modern neural networks | Alexandre Perez-Lebel |
| 2211.02701 | 2022-11-04 | MONAI: An open-source framework for deep learning in healthcare | M. Jorge Cardoso |
| 2211.16886 | 2022-11-30 | A Unifying Theory of Distance from Calibration | Jarosław Błasiok |
| 2212.04629 | 2022-12-09 | Post hoc Explanations may be Ineffective for Detecting Unknown Spurious Correlation | Julius Adebayo |
| 2212.08254 | 2022-12-16 | RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers | Zhikai Li |
| 2212.10445 | 2022-12-20 | Model Ratatouille: Recycling Diverse Models for Out-of-Distribution Generalization | Alexandre Ramé |
| 2302.01790 | 2023-02-03 | Understanding metric-related pitfalls in image analysis validation | Annika Reinke |
| 2302.06675 | 2023-02-13 | Symbolic Discovery of Optimization Algorithms | Xiangning Chen |
| 2302.11813 | 2023-02-23 | Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification | Gerard Maggiolino |
| 2302.12400 | 2023-02-24 | Towards Stable Test-Time Adaptation in Dynamic Wild World | Shuaicheng Niu |
| 2303.02970 | 2023-03-06 | Rethinking Confidence Calibration for Failure Prediction | Fei Zhu |
| 2303.03374 | 2023-03-06 | To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning | Ildus Sadrtdinov |
| 2303.17951 | 2023-03-31 | FP8 versus INT8 for efficient deep learning inference | Mart van Baalen |
| 2304.01910 | 2023-04-04 | On the Variance of Neural Network Training with respect to Test Sets and Distributions | Keller Jordan |
| 2304.04742 | 2023-04-10 | Detection Transformer with Stable Matching | Shilong Liu |
| 2304.04795 | 2023-04-10 | Evaluation of Test-Time Adaptation Under Computational Time Constraints | Motasem Alfarra |
| 2304.07527 | 2023-04-15 | Align-DETR: Enhancing End-to-end Object Detection with Aligned Loss | Zhi Cai |
| 2304.08069 | 2023-04-17 | DETRs Beat YOLOs on Real-time Object Detection | Yian Zhao |
| 2304.11127 | 2023-04-21 | Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for Better Empirical Performance | Shuhei Watanabe |
| 2304.11277 | 2023-04-21 | PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel | Yanli Zhao |
| 2305.18764 | 2023-05-30 | When Does Optimizing a Proper Loss Yield Calibration? | Jarosław Błasiok |
| 2306.03536 | 2023-06-06 | On Pitfalls of Test-Time Adaptation | Hao Zhao |
| 2306.05401 | 2023-06-08 | RDumb: A simple approach that questions our progress in continual test-time adaptation | Ori Press |
| 2306.09301 | 2023-06-15 | OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection | Jingyang Zhang |
| 2306.12517 | 2023-06-21 | FFCV: Accelerating Training by Removing Data Bottlenecks | Guillaume Leclerc |
| 2307.03133 | 2023-07-06 | Benchmarking Test-Time Adaptation against Distribution Shifts in Image Classification | Yongcan Yu |
| 2307.08095 | 2023-07-16 | Semi-DETR: Semi-Supervised Object Detection with Detection Transformers | Jiacheng Zhang |
| 2307.10926 | 2023-07-20 | Confidence Intervals for Performance Estimates in Brain MRI Segmentation | R. El Jurdi |
| 2307.13813 | 2023-07-25 | How to Scale Your EMA | Dan Busbridge |
| 2308.07832 | 2023-08-15 | REFORMS: Reporting Standards for Machine Learning Based Science | Sayash Kapoor |
| 2309.10019 | 2023-09-18 | Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts | Jiang-Xin Shi |
| 2309.12236 | 2023-09-21 | Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing | Jarosław Błasiok |
| 2309.16588 | 2023-09-28 | Vision Transformers Need Registers | Timothée Darcet |
| 2310.01755 | 2023-10-03 | ImageNet-OOD: Deciphering Modern Out-of-Distribution Detection Algorithms | William Yang |
| 2311.11837 | 2023-11-20 | Kandinsky Conformal Prediction: Efficient Calibration of Image Segmentation Algorithms | Joren Brunekreef |
| 2312.10083 | 2023-12-11 | The Limits of Fair Medical Imaging AI In The Wild | Yuzhe Yang |
| 2401.06091 | 2024-01-11 | A Closer Look at AUROC and AUPRC under Class Imbalance | Matthew B. A. McDermott |
| 2402.05806 | 2024-02-08 | On Temperature Scaling and Conformal Prediction of Deep Classifiers | Lahav Dabah |
| 2402.06892 | 2024-02-10 | Understanding Test-Time Augmentation | Masanari Kimura |
| 2403.07263 | 2024-03-12 | Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction | Alexander Timans |
| 2403.14715 | 2024-03-19 | Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It | Guoxuan Xia |
| 2404.06395 | 2024-04-09 | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies | Shengding Hu |
| 2404.09556 | 2024-04-15 | nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation | Fabian Isensee |
| 2405.14458 | 2024-05-23 | YOLOv10: Real-Time End-to-End Object Detection | Ao Wang |
| 2405.15682 | 2024-05-24 | The Road Less Scheduled | Aaron Defazio |
| 2405.18392 | 2024-05-28 | Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations | Alexander Hägele |
| 2405.20459 | 2024-05-30 | On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines | Selim Kuzucu |
| 2406.09405 | 2024-06-13 | Why Warmup the Learning Rate? Underlying Mechanisms and Improvements | Dayal Singh Kalra |
| 2406.11456 | 2024-06-17 | Calibrating Where It Matters: Constrained Temperature Scaling | Stephen McKenna |
| 2407.01032 | 2024-07-01 | Overcoming Common Flaws in the Evaluation of Selective Classification Systems | Jeremias Traub |
| 2409.17763 | 2024-09-26 | Confidence intervals uncovered: Are we ready for real-world medical imaging AI? | Evangelia Christodoulou |
| 2409.19696 | 2024-09-29 | Vision-Language Models are Strong Noisy Label Detectors | Tong Wei |
| 2410.03406 | 2024-10-04 | Conformal confidence sets for biomedical image segmentation | Samuel Davenport |
| 2410.13842 | 2024-10-17 | D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement | Yansong Peng |
| 2411.02853 | 2024-11-05 | ADOPT: Modified Adam Can Converge with Any $β_2$ with the Optimal Rate | Shohei Taniguchi |
| 2411.18704 | 2024-11-27 | Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits | Daniel Morales-Brotons |
| 2412.04234 | 2024-12-05 | DEIM: DETR with Improved Matching for Fast Convergence | Shihua Huang |
| 2412.10288 | 2024-12-13 | Performance evaluation of predictive AI models to support medical decisions: Overview and guidance | Ben Van Calster |
| 2502.15015 | 2025-02-20 | Accelerating Neural Network Training: An Analysis of the AlgoPerf Competition | Priya Kasimbeg |
| 2502.16982 | 2025-02-24 | Muon is Scalable for LLM Training | Jingyuan Liu |
| 2505.18032 | 2025-05-23 | Mahalanobis++: Improving OOD Detection via Feature Normalization | Maximilian Mueller |
| 2505.22764 | 2025-05-28 | Test-time augmentation improves efficiency in conformal prediction | Divya Shanmugam |
| 2506.16950 | 2025-06-20 | LAION-C: An Out-of-Distribution Benchmark for Web-Scale Vision Models | Fanfei Li |
| 2506.19540 | 2025-06-24 | Overtuning in Hyperparameter Optimization | Lennart Schneider |
| 2506.23492 | 2025-06-30 | Sample Margin-Aware Recalibration of Temperature Scaling | Haolan Guo |
| 2602.10045 | 2026-02-10 | Conformal Prediction Sets for Instance Segmentation | Kerri Lu |
| 2604.09697 | 2026-04-06 | I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts Medical Image Classification | Daniel Nobrega Medeiros |
| 2605.10468 | 2026-05-11 | Can Muon Fine-tune Adam-Pretrained Models? | Xingyu Qu |
| 2605.24770 | 2026-05-23 | Muon in Vision Transformers: Optimizer-Recipe Interactions and Gradient Spectra | Ben S. Southworth |
| 2606.09658 | 2026-06-08 | Muon Learns More Robust and Transferable Features than Adam | Tianyu Ruan |
