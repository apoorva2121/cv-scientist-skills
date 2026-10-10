# Deployment: export, runtimes, quantisation, latency, monitoring

Source: research R5 (verified 2026-09-28; arXiv IDs checked through the arXiv API, APIs checked in the cv-skills environment). Cross-cutting findings, library status and unverified items: the cv-scientist skill's `references/library-status.md`. 

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
