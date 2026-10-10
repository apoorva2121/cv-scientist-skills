<!-- Source: research/R1-data-quality-and-splitting.md, section F and decision guidance (verified 2026-09-27). -->
# Split design: method inventory

## F. Splitting best practice

| Method / tool | pip or source | Latest version (release date) | Licence | Maintained? | Use when | Caveats | Source URL |
|---|---|---|---|---|---|---|---|
| Group-aware splitters | `sklearn.model_selection.GroupKFold`, `StratifiedGroupKFold`, `GroupShuffleSplit`, `LeaveOneGroupOut`, `LeavePGroupsOut`, `TimeSeriesSplit`, `PredefinedSplit` | scikit-learn 1.9.1 (2026-09-10) | BSD-3-Clause | Active | **Default for every vision split**: the group is the patient, camera, site, scene, video, capture session or duplicate cluster | `StratifiedGroupKFold` stratifies a single label only | https://github.com/scikit-learn/scikit-learn |
| Multi-label iterative stratification | `iterstrat.ml_stratifiers.MultilabelStratifiedKFold`, `RepeatedMultilabelStratifiedKFold`, `MultilabelStratifiedShuffleSplit` (pip `iterative-stratification`) | 0.1.9 (2024-10-12) | BSD-3-Clause | Stale, but worked in a 2026-09-27 test (sklearn 1.7.2, numpy 2.3.5, Py 3.13) | Multi-label and detection-as-multi-label stratification | **No group support**; aggregate labels per group first, then split groups | https://github.com/trent-b/iterative-stratification ; doi:10.1007/978-3-642-23808-6_10 |
| scikit-multilearn | `skmultilearn.model_selection.IterativeStratification(n_splits, order)`, `iterative_train_test_split` | 0.2.0 (**2018-12-10**) | BSD | Stale (push 2024-02) | Second-order (label-pair) stratification (`order=2`) | The PyPI release is from 2018; it worked in the same test | https://github.com/scikit-multilearn/scikit-multilearn ; https://proceedings.mlr.press/v74/szyma%C5%84ski17a.html |
| Detection stratification recipe | no dedicated maintained library found | n/a | n/a | Method (synthesis) | Per image (or per group): multi-hot of {classes present, object-count bin, COCO size bins small/medium/large, rare-attribute flags}, then iterstrat over groups | Verify after splitting: per-class instance counts and size histograms per split | (see iterstrat) |
| Near-duplicate clustering before splitting | FiftyOne `compute_near_duplicates` / `compute_leaky_splits`; embeddings + FAISS, then connected components become group ids | see B | see B | Active | Web-scraped, video-frame, burst-photo and augmented-copy datasets | CIFAR test sets contained near-duplicates of training images (ciFAIR) | arXiv 1902.00423 |
| Spatial block CV (Python) | `verde.BlockKFold`, `verde.BlockShuffleSplit`, `verde.train_test_split` | verde 1.9.0 (2026-03-19) | BSD-3-Clause | Active (push 2026-09-21) | Remote sensing / geo tiles: block size at least the autocorrelation range | Coordinates must be projected (metres) | https://github.com/fatiando/verde |
| TorchGeo splitters | `torchgeo.datasets.random_bbox_assignment`, `random_bbox_splitting`, `random_grid_cell_assignment`, `roi_split`, `time_series_split` | torchgeo 0.10.0 (2026-08-14) | MIT | Active | Splitting GeoDatasets by grid cell, ROI or time | Random bbox assignment is *not* spatially blocked; prefer grid cells or ROI | https://docs.torchgeo.org/en/stable/api/datasets.html |
| Spatial CV (R, reference implementations) | CRAN `blockCV`, `CAST` (NNDM/kNNDM), `spatialsample` | 4.0-0 (2026-08-20) / 1.1.2 (2026-09-09) / 0.6.1 (2025-12-02) | GPL>=3 / GPL>=2 / MIT | Active | Choosing block size (autocorrelation range) and matching the CV distance distribution to the prediction distances | R only | doi:10.1111/2041-210X.13107 ; doi:10.1111/2041-210X.13851 ; doi:10.5194/gmd-17-5897-2024 |
| Unmaintained Python spatial CV | `spacv` 0.0.22 (2020), `museotoolbox` 0.13.6 (2020, GPLv3), `spatial-kfold` 0.0.4 (2025-06, Alpha, conflicting licence metadata GPL-3.0 vs MIT) | see left | see left | Stale / Stale / Slow | Avoid; use verde or torchgeo | - | PyPI |
| Spatial autocorrelation evidence | papers | n/a | n/a | Method | Justifying spatial splits | Random holdouts overestimated CNN performance by **up to 28%** versus spatial blocks (Kattenborn 2022). Counterpoint: for design-based map accuracy with a probability sample, random validation is appropriate (Wadoux 2021) | doi:10.1111/ecog.02881 ; doi:10.1038/s41467-020-18321-y ; doi:10.1016/j.ophoto.2022.100018 ; doi:10.1007/s10994-021-05972-1 ; doi:10.1016/j.ecolmodel.2021.109692 |
| Patient-level splits (medical) | `GroupKFold` on patient ID (never on slice, tile or image) | n/a | n/a | Method | MRI slices, pathology tiles, repeated studies | Slice-level splits inflate accuracy (Yagis 2021); tile leakage (Bussola 2019) | doi:10.1038/s41598-021-01681-w ; arXiv 1909.06539 ; doi:10.1038/s41746-022-00592-y |
| Video splits | group by source video, clip group or recording session | n/a | n/a | Method | Action recognition, tracking, frame-sampled detection | UCF101: videos in the same group share background and viewpoint, so split by group | https://www.crcv.ucf.edu/data/UCF101.php ; arXiv 1212.0402 |
| Temporal splits | `TimeSeriesSplit`; torchgeo `time_series_split`; train on the past, test on the future | n/a | n/a | Method | Deployment will see future data (sensor drift, seasons) | Wild-Time and WILDS (FMoW) show drops under time shift | arXiv 2211.14238 ; arXiv 2012.07421 |
| k-fold vs holdout | repeated grouped k-fold for small data; a single frozen holdout for large data | n/a | n/a | Method | With n in the low hundreds, a single split is noise: **about Â±10% error bars at n=100**, and the SE across folds underestimates them | Variance also comes from seeds and hyperparameters, so report several | doi:10.1016/j.neuroimage.2017.06.061 ; arXiv 2103.03098 |
| Test-set size and CIs | `statsmodels.stats.proportion.proportion_confint(method="wilson")`; `scipy.stats.bootstrap` (scipy 1.18.1); `torchmetrics.wrappers.BootStrapper` (torchmetrics 1.9.0, 2026-03-09); `confidenceinterval` 1.0.5 (2024-08, MIT, stale) | see left | BSD / BSD / Apache-2.0 / MIT | Active / Active / Slow / Stale | Always report 95% CIs. **Wilson half-width at 90% accuracy: n=100 Â±6.0pp, 300 Â±3.4, 1000 Â±1.9, 3000 Â±1.1, 10000 Â±0.6** (computed 2026-09-27) | For mAP/mIoU/Dice: bootstrap over **images** (cluster bootstrap over patients/scenes when grouped) and recompute the dataset-level metric; compare models with a *paired* bootstrap on the same resamples | arXiv 2409.17763 ; doi:10.1038/s41592-023-02151-z ; doi:10.1038/s41592-023-02150-0 |
| Pretraining-data contamination | dedup the test set against accessible pretraining data with SSCD/DINOv2 kNN; prefer fresh or private test sets | n/a | n/a | Method | Any foundation-model backbone or benchmark claim | DINOv2 removed near-duplicates of benchmark test/val images from LVD-142M. For CLIP, pruning near-test samples left performance high (2023), but in-domain samples explain much of the "OOD" generalisation (2024). Multimodal LLM benchmarks show image contamination. Test sets are refreshable (ImageNetV2) | arXiv 2304.07193 ; arXiv 2310.09562 ; arXiv 2410.08258 ; arXiv 2411.03823 ; arXiv 1902.10811 ; arXiv 2506.16950 |

---

## Decision guidance with citations

**What to do first (in order):**
1. **Integrity pass before anything else.** Decode every file with the *same decoder the training loader uses*.
   - Log failures, modes, bit depth and EXIF orientation.
   - Normalise orientation explicitly: `ImageOps.exif_transpose`, or document that OpenCV auto-rotates.
   - Medical: apply the rescale/VOI LUTs and check image/mask geometry (SimpleITK, nibabel).
   - Geo: check CRS, nodata and band order (rasterio).
   - Video: decode every frame (TorchCodec/PyAV). torchvision video I/O no longer exists (removed in 0.26).
2. **Validate annotations and look at them.**
   - Load the annotations with pycocotools, Datumaro or FiftyOne.
   - Assert the checks from the A2 conventions list (boxes in bounds after EXIF, positive areas, consistent class maps, mask/image size and geometry).
   - Render about 50 random overlays per class; this is the cheapest way to catch coordinate-convention bugs.
3. **Profile the pool.** Run CleanVision or Datalab(image_key) for blur, dark and odd-size images. Add FiftyOne uniqueness and representativeness, or embedding-density outliers.
   - fastdup is fast but **non-commercial (CC BY-NC-ND 4.0) and Python <3.13**.
4. **Deduplicate, then define groups, before any split.**
   - Hashes first (ImageHash/imagededup), then embeddings (DINOv2/SigLIP 2 + FAISS, or SSCD for copy-style edits).
   - Take connected components of the near-duplicate graph and use them as group ids, together with patient, site, camera, video and session ids.
   - Evidence: Barz & Denzler 2020 (arXiv 1902.00423); Kapoor & Narayanan 2023 (doi:10.1016/j.patter.2023.100804).
5. **Split with a group-aware splitter, stratifying within the groups.**
   - Choose the splitter from the "which split for which data" table below.
   - Freeze the test set: store the ids and file hashes.
   - Run FiftyOne `compute_leaky_splits` (threshold 0.1-0.25) as a post-split audit.
6. **Clean labels using held-out predictions.**
   - K-fold out-of-sample predictions, then the cleanlab API for the task: `filter.find_label_issues` (multi-class), `multilabel_classification.filter.find_label_issues`, `object_detection.filter.find_label_issues`, `segmentation.filter.find_label_issues`.
   - **Send the flagged items to human review.** Only about 51% of algorithm-flagged test labels were true errors (arXiv 2103.14749).
   - Relabel in CVAT or Label Studio. Measure agreement with kappa or Krippendorff's alpha; use STAPLE for masks. Clean the **test set** with extra care: at least 6% of ImageNet val labels are wrong.
7. **Test for shortcuts before trusting a metric.**
   - (a) Metadata-only baseline and the Datalab `spurious_correlations` report.
   - (b) Evaluate on foreground-masked and background-only images (arXiv 2006.09994); crop borders and text overlays.
   - (c) Report per-site, per-scanner and per-camera metrics, since hidden stratification is common (arXiv 1909.12475).
   - (d) Grad-CAM spot checks, only alongside sanity checks (arXiv 1810.03292).
   - (e) Slice discovery with CLIP/SigLIP embeddings (Domino-style) and LLM/VLM hypotheses (LADDER, GH-ESD).
   - Mitigate in this order: class/group balancing (arXiv 2110.14503), then DFR (arXiv 2204.02937), then Group DRO if groups are labelled, or JTT if not. Re-test for other shortcuts afterwards (Whac-A-Mole, arXiv 2212.04825).
8. **OOD and outlier gate.**
   - Pretrained-backbone features: L2-normalised Mahalanobis (Mahalanobis++, arXiv 2505.18032) or KNN (arXiv 2204.06507).
   - Trained classifier logits: Energy, reported next to MSP.
   - Evaluate on both near-OOD and far-OOD sets from *your* domain (pytorch-ood). Treat OpenOOD as a reference benchmark, not a dependency.
9. **Report uncertainty.**
   - Give n and 95% CIs for every headline metric.
   - Use a bootstrap over images, or a cluster bootstrap over patients or scenes, for mAP, mIoU and Dice; use paired bootstraps for model deltas.
   - Only 1 of 221 MICCAI 2023 segmentation papers reported CIs (arXiv 2409.17763). With n<1000, differences below about 2pp in accuracy are usually not resolvable (Wilson numbers in F).
10. **Contamination check for foundation models.**
    - Search your test set against any accessible pretraining corpus, and prefer private or new test data.
    - Treat ImageNet-era robustness sets (e.g. ImageNet-C) as possibly in-distribution for web-scale models (arXiv 2506.16950, 2410.08258).

**Which split for which data:**

| Data type | Group key (never split within) | Stratify on | Tool |
|---|---|---|---|
| Classification (web or scraped) | near-dup cluster, source URL/domain | class | `StratifiedGroupKFold` |
| Multi-label | near-dup cluster, source | label vector | iterstrat over group-aggregated labels |
| Detection / instance segmentation | video, scene, session, near-dup cluster | classes present + count bin + size bin | iterstrat recipe (F) |
| Semantic segmentation / depth | scene, sequence, sensor rig | class-pixel presence, scene type | `GroupKFold`, or iterstrat on presence vectors |
| Pose / keypoints | subject/person id, session | pose/viewpoint bins | `StratifiedGroupKFold` |
| Video | source video, clip group, recording session | action class | `StratifiedGroupKFold` |
| OCR / documents | document, template/layout, author/scanner | script/language, doc type | `GroupKFold` |
| Medical 2D/3D | **patient** (and site for external tests) | diagnosis, scanner | `StratifiedGroupKFold`; a held-out site as external test |
| Remote sensing | spatial block â‰¥ autocorrelation range; acquisition/date | land-cover class | verde `BlockKFold`, torchgeo `random_grid_cell_assignment` / `roi_split`; a held-out region or time |
| Point clouds / driving | drive/sequence, city | classes, range bins | `GroupKFold` by sequence; hold out a city |
| Any data under temporal drift | time period | - | forward-chaining (`TimeSeriesSplit`), held-out future window |

---


