# Post-processing methods: inventory with evidence

Source: research R5 (verified 2026-09-28; arXiv IDs checked through the arXiv API, APIs checked in the cv-skills environment). Cross-cutting findings, library status and unverified items: the cv-scientist skill's `references/library-status.md`. 

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
