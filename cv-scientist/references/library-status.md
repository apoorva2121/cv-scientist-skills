# Cross-cutting findings, library status and unverified items (training to deployment)

Source: research R5. How facts were checked:

**Checked 2026-09-28.** How each kind of fact was verified:
- **Papers.** Every arXiv ID was resolved through the arXiv API (`export.arxiv.org/api/query?id_list=...`), which gave its title and v1 date. Citations are written as `arXiv ID (v1 YYYY-MM-DD)`. Claims are paraphrased or quoted from the abstract unless marked *(paper body)*; paper-body facts were read from arXiv HTML or ar5iv. The reference appendix at the end was generated from the API output, and a script checked every dated citation in this file against it.
- **Packages.** The PyPI JSON API gave the latest version, the upload date of the first file of that release, and the licence metadata.
- **Repositories.** The GitHub REST API (15 calls) gave `license.spdx_id`, `archived` and `pushed_at`. None of the 15 repositories is archived.
- **APIs.** Checked in the local `cv-skills` environment with `inspect.signature`, docstrings and small in-memory tests that write no files. Versions: torch 2.14.0+cpu, torchvision 0.29.0, timm 1.0.30, torchmetrics 1.9.0, optuna 5.0.0, onnx 1.23.0 (IR 14, max opset 28), onnxruntime 1.30.0, grad-cam 1.5.7, ensemble-boxes 1.0.9, scikit-learn 1.9.1, scipy 1.18.1, cleanlab 2.9.0, segmentation-models-pytorch 0.5.0, kornia 0.8.3, scikit-image 0.26.0 and opencv 5.0.0. Libraries that are not installed were checked against their source on GitHub (raw files) or their official docs: MONAI, nnU-Net, NNCF, MAPIE, TorchCP, Captum, fairlearn, FiftyOne, ExecuTorch, LiteRT Torch, Ray, schedule_free, Muon, pytorch-metric-learning, detectron2, DINOv2 and torch-lr-finder.
- **Local measurements.** Taken on the project machine described in STATUS.md (8-core Ryzen 7 7735HS, no CUDA). Each is a single run, and the user's other jobs may have been using the CPU, so treat them as indicative.
- **Tags.** "(reasoning)" marks a mechanism argued here rather than taken from a source. "(practice)" marks common practice with no verified source.

**Licence vocabulary** (as in R3). *Permissive* means Apache-2.0, MIT or BSD. *Custom-commercial* means a bespoke licence that allows commercial use. *NC* means non-commercial. *Copyleft* means GPL, LGPL or AGPL. *Source-available* means BSL and similar licences that are not open source until a change date.

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
