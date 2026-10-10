<!-- Source: research/R1-data-quality-and-splitting.md (verified 2026-09-27). Section F (splitting) is also used by cv-validation-design. -->
# Data audit: method inventory

Verified method inventory for the Computer Vision Scientist skill, area 1. It covers classification, multi-label, detection, segmentation, pose, depth, video, OCR/documents, 3D volumes, remote sensing and point clouds.

**Checked 2026-09-27.** Versions and dates come from the PyPI JSON API (the upload time of the first file of the latest release). Licence, archive status and last push come from the GitHub REST API. Licences are cross-checked against PyPI metadata; where PyPI leaves the field empty, the GitHub licence is used. R packages were checked on the CRAN DB, model licences on the Hugging Face API, and papers through the arXiv API or Crossref DOI lookup. API names were confirmed against the current source on GitHub (raw files).

**"Maintained?" legend** (relative to 2026-09-27):
- **Active**: a release or a push within the last 6 months.
- **Slow**: the newest release or push is 6 to 18 months old.
- **Stale**: nothing for more than 18 months.
- **Archived**: the GitHub repository is archived (read-only).
- **Method**: a paper, not a package.

---

## A. Integrity and format

### A1. Decoding, colour modes, medical, geo, video and point-cloud I/O

| Method / tool | pip or source | Latest version (release date) | Licence | Maintained? | Use when | Caveats | Source URL |
|---|---|---|---|---|---|---|---|
| Pillow integrity pass | `pillow` | 12.3.0 (2026-07-01) | MIT-CMU | Active | Detecting corrupt or truncated files: call `Image.open(p).verify()`, then re-open and call `.load()` to force a full decode. Also: `ImageOps.exif_transpose`; mode normalisation (P to RGBA/RGB, CMYK to RGB, LA/RGBA alpha handling); keep `I;16` for 16-bit data | `verify()` does not decode the pixels, and the file must be re-opened afterwards. `ImageFile.LOAD_TRUNCATED_IMAGES` defaults to False; setting it to True globally silently pads broken files. Blindly calling `.convert("RGB")` destroys 16-bit depth and alpha | https://pillow.readthedocs.io |
| pillow-heif | `pillow-heif` | 1.8.0 (2026-09-22) | BSD-3-Clause | Active | HEIC/HEIF images from phones, opened through Pillow | Needs an explicit `register_heif_opener()` | https://github.com/bigcat88/pillow_heif |
| OpenCV imread | `opencv-python` | 5.0.0.93 (2026-07-02) | Apache-2.0 | Active | Fast decoding; `IMREAD_ANYDEPTH` for 16/32-bit; `IMREAD_UNCHANGED` to keep alpha | Returns an **empty matrix (None in Python) rather than raising**, so always check for it. It **applies EXIF orientation by default** unless `IMREAD_IGNORE_ORIENTATION` or `IMREAD_UNCHANGED` is set, so Pillow and OpenCV pipelines can disagree and misalign boxes. Output is in BGR order | https://github.com/opencv/opencv/blob/4.x/modules/imgcodecs/include/opencv2/imgcodecs.hpp |
| torchvision.io | `torchvision` | 0.29.0 (2026-09-02) | BSD | Active | Legacy code paths only | **Video decoding/encoding was removed in 0.26.0 (2026-03-23)**. **The image decoders/encoders are deprecated as of 0.29** in favour of TorchCodec >= 0.16. `decode_image(..., apply_exif_orientation=False)` is the default | https://github.com/pytorch/vision/releases |
| TorchCodec | `torchcodec` | 0.16.0 (2026-08-13) | BSD-3-Clause | Active | PyTorch's decoder for video, audio and now images; the repo moved to `meta-pytorch/torchcodec` | Requires FFmpeg; pin FFmpeg and TorchCodec versions together | https://github.com/meta-pytorch/torchcodec |
| PyAV | `av` | 18.1.0 (2026-08-12) | BSD-3-Clause | Active | Frame-exact decoding, and detecting corrupt streams by decoding every frame | Reading only the container header does not prove the frames decode | https://github.com/PyAV-Org/PyAV |
| decord | `decord` | 0.6.0 (2021-06-14) | Apache-2.0 | Stale (last push 2024-07) | Only for reproducing old video pipelines | No release since 2021; prefer TorchCodec or PyAV | https://github.com/dmlc/decord |
| tifffile | `tifffile` | 2026.9.20 (2026-09-21) | BSD-3-Clause | Active | Multi-page, 16/32-bit float, OME-TIFF and microscopy stacks | Not geo-aware; use rasterio for CRS and transforms | https://github.com/cgohlke/tifffile |
| rasterio | `rasterio` | 1.5.1 (2026-08-08) | BSD-3-Clause | Active | GeoTIFF/COG and multispectral bands: CRS, affine transform, nodata masks | Check band order, nodata and dtype (uint16 reflectance) per sensor | https://github.com/rasterio/rasterio |
| pydicom | `pydicom` | 3.0.2 (2026-03-19) | MIT | Active (push 2026-09-19) | Reading DICOM; apply the Modality LUT (RescaleSlope/Intercept) and VOI LUT; check PhotometricInterpretation (MONOCHROME1 is inverted) | Headers contain PHI, so de-identify. Pixel data may need extra codec plugins | https://github.com/pydicom/pydicom |
| highdicom | `highdicom` | 0.28.1 (2026-07-28) | MIT | Active | Reading and writing DICOM SEG/SR annotations | Beta status | https://github.com/imagingdatacommons/highdicom |
| dicom-validator | `dicom-validator` | 0.9.0 (2026-09-09) | MIT | Active | Validating DICOM files against the standard's IOD definitions | Beta | https://github.com/pydicom/dicom-validator |
| dcm2niix | `dcm2niix` | 1.0.20260724 (2026-07-24) | Mixed (see repo) | Active | Converting DICOM to NIfTI with a BIDS JSON sidecar | Check the slice ordering and orientation of the output | https://github.com/rordenlab/dcm2niix |
| nibabel | `nibabel` | 5.4.2 (2026-03-11) | MIT | Active (push 2026-09-19) | NIfTI I/O and affine/orientation checks (`as_closest_canonical`) | An inconsistent affine between image and mask is a silent label corruption | https://github.com/nipy/nibabel |
| SimpleITK | `SimpleITK` | 2.5.6 (2026-07-30) | Apache-2.0 | Active | Resampling to common spacing, checking that image and mask geometry match, STAPLE consensus | Spacing and origin mismatches between image and mask are common | https://github.com/SimpleITK/SimpleITK |
| MONAI / TorchIO | `monai` / `torchio` | 1.6.0 (2026-06-22) / 1.2.1 (2026-06-02) | Apache-2.0 / Apache-2.0 | Active / Active | Medical-volume loading, orientation and spacing transforms | Transform order affects the label/image alignment | https://github.com/Project-MONAI/MONAI , https://github.com/TorchIO-project/torchio |
| Open3D / laspy | `open3d` / `laspy` | 0.20.0 (2026-09-16) / 2.7.0 (2026-01-14) | MIT / BSD-2-Clause | Active / Active | Point clouds: PLY/PCD (Open3D) and LAS/LAZ (laspy); check NaNs, duplicate points, coordinate frame | Open3D is classed "Alpha" on PyPI | https://github.com/isl-org/Open3D , https://github.com/laspy/laspy |

### A2. Annotation formats: validators and converters

| Method / tool | pip or source | Latest version (release date) | Licence | Maintained? | Use when | Caveats | Source URL |
|---|---|---|---|---|---|---|---|
| pycocotools | `pycocotools` | 2.0.11 (2025-12-15) | BSD-2 (FreeBSD) | Active (PyPI now built from the ppwwyyxx fork; cocodataset/cocoapi last push 2024-04) | Loading COCO JSON, RLE `mask.encode/decode/area/toBbox`, reference COCOeval | Loading the JSON is not a schema validation. RLE is column-major (Fortran order) | https://github.com/ppwwyyxx/cocoapi |
| faster-coco-eval | `faster-coco-eval` | 1.8.0 (2026-08-24) | Apache-2.0 (GitHub; the PyPI field is empty) | Active | A faster drop-in COCOeval, useful for bootstrapped mAP | Check parity with pycocotools on your data | https://github.com/MiXaiLL76/faster_coco_eval |
| supervision | `supervision` | 0.30.5 (2026-09-22) | MIT | Active | `DetectionDataset.from_coco/from_yolo/from_pascal_voc/from_labelme/from_createml` and the matching `as_*` for conversion; overlay rendering for visual QA | The built-in split is random only, not group-aware | https://github.com/roboflow/supervision |
| Datumaro | `datumaro` | 1.13.11 (2026-09-07) | MIT (GitHub) | Active (repo moved to open-edge-platform) | Widest format coverage: COCO, VOC, YOLO, CVAT, KITTI and KITTI-3D, DOTA, LabelMe, MOT/MOTS, Cityscapes, ICDAR, Kinetics, Segment Anything, Supervisely point cloud, MVTec. The CLI has `convert`, `merge`, `filter`, `stats`, `validate`, `detect_format` | The PyPI licence field is empty; no 2.x release on PyPI | https://github.com/open-edge-platform/datumaro |
| FiftyOne (import/export + app) | `fiftyone` | 1.22.0 (2026-09-12) | Apache-2.0 | Active | Loading COCO/VOC/YOLO/KITTI/CVAT and more, then *looking* at labels; also video, 3D and geo | Needs a local MongoDB (bundled via `fiftyone-db`) | https://github.com/voxel51/fiftyone |
| globox | `globox` | 2.9.0 (2026-02-09) | MIT | Slow | Lightweight box conversion (COCO, YOLO, Pascal VOC, OpenImages, CVAT, LabelMe, ImageNet) and COCO-style evaluation | Boxes only; no masks or keypoints | https://github.com/laclouis5/globox |
| CVAT + SDK | `cvat-sdk` | 2.76.0 (2026-09-16) | MIT | Active | Annotation and review, exporting to many formats | Self-hosting is effort | https://github.com/cvat-ai/cvat |
| labelme | `labelme` | 7.7.0 (2026-09-18) | GPL-3.0-only | Active | Polygon/point annotation (JSON) | GPL licence | https://github.com/wkentaro/labelme |
| Ultralytics converters | `ultralytics` | 8.4.163 (2026-09-25) | AGPL-3.0 | Active | `convert_coco`, `convert_dota_to_yolo_obb`, `convert_segment_masks_to_yolo_seg` | **AGPL-3.0**: importing it into a product has licence consequences | https://github.com/ultralytics/ultralytics |
| pylabel | `pylabel` | 0.1.55 (2023-11-26) | MIT | Stale (push 2024-08) | Legacy notebooks only | Unmaintained | https://github.com/pylabel-project/pylabel |

**Format conventions that commonly break silently:**
- **Box encodings.** COCO bbox is `[x, y, w, h]` in absolute pixels. YOLO is `class cx cy w h`, normalised to 0-1. Pascal VOC is `xmin ymin xmax ymax`, 1-based. KITTI is 15 fields (truncation, occlusion, alpha, 2D box as left/top/right/bottom, 3D dims/location/rotation_y). DOTA is 8 corner coordinates plus class plus difficult.
- **Keypoints.** COCO keypoints are `[x, y, v]`, where v=0 is not labelled, 1 is labelled but not visible, and 2 is visible.
- **Checks to run.** Boxes stay inside the image after EXIF transpose; no zero or negative area; class-id maps are identical across splits; mask and image have the same size and geometry; 16-bit depth PNG scale factors match the dataset devkit.

---

## B. Dataset profiling and curation tools

| Method / tool | pip or source | Latest version (release date) | Licence | Maintained? | Use when | Caveats | Source URL |
|---|---|---|---|---|---|---|---|
| CleanVision | `cleanvision` | 0.3.7 (2026-01-06) | Apache-2.0 | Slow (push 2026-01-08) | A first-pass image audit that detects 9 issue types: `exact_duplicates`, `near_duplicates`, `blurry`, `low_information`, `dark`, `light`, `grayscale`, `odd_aspect_ratio`, `odd_size`. Inputs: folder, HF datasets, torchvision | Heuristic thresholds; works at image level (knows nothing about labels) | https://github.com/cleanlab/cleanvision |
| cleanlab Datalab | `cleanlab` | 2.9.0 (2026-01-13) | Apache-2.0 | Slow (2.7.1 2025-02, then 2.8.0 and 2.9.0 in Jan 2026) | One call to find issues: `outlier`, `label`, `near_duplicate`, `non_iid`, `class_imbalance`, `underperforming_group`, `data_valuation`, `null`. With `image_key` it also runs CleanVision checks and a `spurious_correlations` report (image properties correlated with labels) | Needs out-of-sample `pred_probs` and/or features. 2.9.0 **removed TF/Keras support**; 2.8.0 moved to Python 3.10-3.14 | https://github.com/cleanlab/cleanlab |
| FiftyOne Brain | `fiftyone-brain` | 0.26.0 (2026-09-22) | Apache-2.0 | Active | `compute_similarity` (backends: sklearn, pinecone, qdrant, milvus, lancedb, redis, mongodb, elasticsearch, pgvector, mosaic), `compute_uniqueness`, `compute_representativeness`, `compute_mistakenness`, `compute_hardness`, `compute_near_duplicates`, `compute_exact_duplicates`, **`compute_leaky_splits`** (default threshold 0.2, doc suggests 0.1-0.25), `compute_visualization` (umap/tsne/pca) | Embedding-model choice drives the results | https://github.com/voxel51/fiftyone-brain |
| fastdup | `fastdup` | 2.54 (2026-08-23) | **CC BY-NC-ND 4.0 (non-commercial)** | Active releases | Very fast duplicate, outlier and cluster galleries on large folders | **Commercial use is not permitted** without a separate licence from Visual Layer. `requires_python <3.13` (wheels cp39-cp312 only) | https://github.com/visual-layer/fastdup |
| imagededup | `imagededup` | 0.3.3.post2 (2025-08-15) | Apache-2.0 | Slow (push 2025-08-15) | Hash dedup (PHash, DHash, WHash, AHash) plus CNN embeddings | Wheels only up to cp312 | https://github.com/idealo/imagededup |
| ImageHash | `imagehash` | 4.3.2 (2025-02-01) | BSD-2-Clause | Repo active (push 2026-09-26); last release 2025-02 | Cheap exact and near-exact duplicates (resize/recompress) | Misses crops, flips and re-renders; use embeddings for those | https://github.com/JohannesBuchner/imagehash |
| SSCD copy-detection descriptors | GitHub only (TorchScript weights, e.g. `sscd_disc_mixup`, 512-d) | no package release | MIT (code) | **Archived (2022-08)** | Copy and near-duplicate detection that survives crops and edits. The DINOv2 pipeline used it to dedup its data *and* remove near-duplicates of benchmark test/val images | Archived; ResNet-50 era; weights licence not stated separately from the code | https://github.com/facebookresearch/sscd-copy-detection ; arXiv 2202.10261 |
| Embedding near-dup search: DINOv2 / SigLIP 2 / DINOv3 + FAISS | `transformers`, `open-clip-torch` 3.3.0 (2026-02-27, MIT), `faiss-cpu` 1.15.1 (2026-09-16, MIT), `usearch` 2.26.2 (Apache-2.0) | DINOv2 (HF, Apache-2.0); SigLIP 2 (HF 2025-02, Apache-2.0); DINOv3 (2025-08, gated) | DINOv2, SigLIP 2: Apache-2.0; **DINOv3: custom "DINOv3 License"** (royalty-free, gated, trade-control/ITAR clauses, attribution in publications) | Active | Semantic near-duplicates, cross-split leak search, clustering into groups before splitting | Cosine thresholds are data-specific, so calibrate them on a hand-labelled sample of pairs. OpenCLIP weight licences vary by checkpoint | https://github.com/facebookresearch/dinov2 ; arXiv 2502.14786 ; https://github.com/facebookresearch/dinov3 |
| SemDeDup (method) | GitHub | n/a | GitHub reports NOASSERTION | **Archived (2023)** | Web-scale semantic dedup: k-means on embeddings, then a cosine threshold within each cluster | Use the method (or NeMo Curator), not the repo | arXiv 2303.09540 |
| Clustering-based curation (method) | paper | n/a | n/a | Method (TMLR 2024) | Balancing a large unlabelled pool (hierarchical k-means plus balanced sampling) before SSL or labelling | Designed for pretraining pools | arXiv 2405.15613 |
| NeMo Curator | `nemo-curator` | 1.3.0 (2026-07-27) | Apache-2.0 | Active | GPU/Ray pipelines over images and video: embeddings, dedup, aesthetic/NSFW filters, video scene/clip dedup | Heavy; suited to 10^6+ images | https://github.com/NVIDIA-NeMo/Curator |
| Lightly family | `lightly` / `lightly-studio` / `lightly-train` | 1.5.26 (2026-07-27) / 1.1.2 (2026-09-21) / 0.17.0 (2026-07-28) | MIT / Apache-2.0 / **AGPL-3.0** | Active | LightlyStudio: local curation, annotation, embedding plots and near-dup removal (COCO, YOLO, folders, video). `lightly`: SSL library | LightlyTrain is AGPL | https://github.com/lightly-ai/lightly-studio |
| Renumics Spotlight | `renumics-spotlight` | 1.8.2 (2026-08-03) | MIT | Active | Interactive browsing of embeddings, issues and predictions | UI tool, not a pipeline | https://github.com/Renumics/spotlight |
| Encord Active | (PyPI 404) | last push 2025-05-23 | Apache-2.0 | **Archived** | Do not adopt | Repo archived, and the PyPI project no longer resolves | https://github.com/encord-team/encord-active |
| Deepchecks (vision) | `deepchecks[vision]` | 0.19.1 (2024-12-15) | AGPL-3.0 | Slow (push 2025-12; no release for 21 months) | Legacy suites only | AGPL; release cadence stopped | https://github.com/deepchecks/deepchecks |

---

## C. Label quality, agreement, relabelling and active learning

### C1. Label-error detection (all APIs confirmed in cleanlab master source, 2026-09-27)

| Method / tool | pip or source | Latest version (release date) | Licence | Maintained? | Use when | Caveats | Source URL |
|---|---|---|---|---|---|---|---|
| Confident learning (multi-class) | `cleanlab`: `cleanlab.filter.find_label_issues(labels, pred_probs)`, `cleanlab.rank.get_label_quality_scores` | 2.9.0 (2026-01-13) | Apache-2.0 | Slow | Single-label classification | Requires **out-of-sample** pred_probs (K-fold cross-val). Flags are candidates: in the test-set study only 51% of flagged items were confirmed errors | arXiv 1911.00068 ; arXiv 2103.14749 |
| Multi-label | `cleanlab.multilabel_classification.filter.find_label_issues`, `.find_multilabel_issues_per_class`; `...rank.get_label_quality_scores(_per_class)` | same | Apache-2.0 | Slow | Multi-label tagging | Labels are passed as a list of label lists | arXiv 2211.13895 |
| Object detection (ObjectLab) | `cleanlab.object_detection.filter.find_label_issues(labels, predictions)`; `...rank.get_label_quality_scores`, `compute_overlooked_box_scores`, `compute_badloc_box_scores`, `compute_swap_box_scores`, `issues_from_scores` | same | Apache-2.0 | Slow | Missing boxes, badly localised boxes and swapped classes | Needs held-out detector predictions for every training image | arXiv 2309.00832 |
| Semantic segmentation | `cleanlab.segmentation.filter.find_label_issues(labels, pred_probs)`; `...rank.get_label_quality_scores`, `issues_from_scores` | same | Apache-2.0 | Slow | Pixel-label errors and per-image quality scores | Memory-heavy at full resolution; downsample the probabilities | arXiv 2307.05080 |
| Multi-annotator consensus (CROWDLAB) | `cleanlab.multiannotator.get_label_quality_multiannotator`, `get_majority_vote_label`, `get_active_learning_scores` | same | Apache-2.0 | Slow | Several labels per image; consensus plus per-annotator quality | Classification only | arXiv 2210.06812 |
| FiftyOne mistakenness / hardness | `fiftyone.brain.compute_mistakenness`, `compute_hardness` | 0.26.0 (2026-09-22) | Apache-2.0 | Active | Ranking suspect classifications and detections for review inside the app | Heuristic scores, not calibrated probabilities | https://github.com/voxel51/fiftyone-brain |
| Loss inspection (detection) | method | n/a | n/a | Method (WACV 2024) | Detection label errors via per-box loss | Benchmark uses simulated errors | arXiv 2303.06999 |
| "Rechecked" detection-to-correction | method + benchmark | n/a | n/a | Method (2025-08, v2 2026-01) | A semi-automated workflow: detector proposals followed by crowd micro-task verification. Reports 18% missing or inaccurate labels in KITTI pedestrian ground truth | Evidence covers a single class | arXiv 2508.06556 |
| Learning to detect label errors by making them | method | n/a | n/a | Method (2025-08) | One method for detection, semantic and instance segmentation; releases 459 real Cityscapes label errors as a benchmark | New; no maintained package verified | arXiv 2508.17930 |
| Video annotation errors ("Loss Knows Best") | method | n/a | n/a | Method (2026-02) | Mislabelled segments in temporal/phase annotations, found through loss trajectories | New; not independently replicated | arXiv 2602.15154 |

### C2. Inter-annotator agreement and consensus

| Method / tool | pip or source | Latest version (release date) | Licence | Maintained? | Use when | Caveats | Source URL |
|---|---|---|---|---|---|---|---|
| Cohen's kappa | `sklearn.metrics.cohen_kappa_score`; `statsmodels.stats.inter_rater.cohens_kappa` | sklearn 1.9.1 (2026-09-10); statsmodels 0.15.0 (2026-08-27) | BSD-3-Clause | Active | Two raters, categorical labels (use weighted kappa for ordinal grades) | Depends on prevalence | https://github.com/statsmodels/statsmodels |
| Fleiss' kappa | `statsmodels.stats.inter_rater.fleiss_kappa` (after `aggregate_raters`) | 0.15.0 | BSD-3-Clause | Active | More than 2 raters, every item rated by the same number of raters | Cannot handle missing ratings | same |
| Krippendorff's alpha | `krippendorff` (`krippendorff.alpha`) | 0.8.2 (2025-11-03) | **GPL-3.0-or-later** | Active (push 2026-08) | Any number of raters, missing ratings, nominal/ordinal/interval data | GPL licence | https://github.com/pln-fing-udelar/fast-krippendorff |
| STAPLE consensus masks | `SimpleITK.STAPLEImageFilter`, `MultiLabelSTAPLEImageFilter` | 2.5.6 (2026-07-30) | Apache-2.0 | Active | Fusing several segmentation annotations and estimating each rater's sensitivity/specificity | Assumes raters are independent | doi:10.1109/TMI.2004.828354 |
| Detection and segmentation agreement (practice) | compute pairwise: box matching F1 or mAP at IoU 0.5 with one rater as the reference; mask Dice/IoU plus a boundary metric | n/a | n/a | Method | Setting a realistic ceiling for model metrics and finding ambiguous classes | Choose metrics with Metrics Reloaded rather than ad hoc | doi:10.1038/s41592-023-02151-z |

### C3. Relabelling tools and active-learning libraries

| Method / tool | pip or source | Latest version (release date) | Licence | Maintained? | Use when | Caveats | Source URL |
|---|---|---|---|---|---|---|---|
| CVAT / Label Studio | `cvat-sdk` / `label-studio` | 2.76.0 (2026-09-16) / 1.23.1 (2026-09-25) | MIT / Apache-2.0 | Active | Queues for review and relabelling. FiftyOne's annotation API has backends `cvat`, `labelstudio` and `labelbox` | Send a *prioritised* queue (top-k by label-quality score), not the whole set | https://github.com/cvat-ai/cvat ; https://github.com/HumanSignal/label-studio |
| scikit-activeml | `scikit-activeml` | 1.0.0 (2025-12-11) | BSD-3-Clause | Active (push 2026-09-18) | A sklearn-style pool-based active-learning (AL) toolbox with many query strategies | Deep-learning integration is manual | https://github.com/scikit-activeml/scikit-activeml |
| baal | `baal` | 2.1.0 (2025-06-24) | Apache-2.0 | Slow (push 2025-12) | Bayesian AL (MC-dropout BALD) with PyTorch | Low activity | https://github.com/baal-org/baal |
| modAL | **`modAL-python`** | 0.4.2.1 (2023-06-01) | MIT | Stale (push 2024-02) | Legacy code | **`pip install modAL` installs Modal Labs' unrelated client** (modAL 1.5.5) | https://github.com/modAL-python/modAL |
| ALiPy | `alipy` | 1.2.5 (2021-07-28) | BSD-3-Clause | Stale | Do not adopt | No release since 2021 | https://github.com/NUAA-AL/ALiPy |

### C4. Active-learning methods

| Method / tool | pip or source | Latest version (release date) | Licence | Maintained? | Use when | Caveats | Source URL |
|---|---|---|---|---|---|---|---|
| BALD / MC-dropout | paper (baal) | n/a | n/a | Method | Uncertainty-based querying | Batch redundancy | arXiv 1703.02910 |
| Core-set | paper | n/a | n/a | Method | Diversity-based batch selection | Cost of computing distances | arXiv 1708.00489 |
| BADGE | paper | n/a | n/a | Method | Mixing uncertainty with diversity (gradient embeddings) | Classification only | arXiv 1906.03671 |
| TypiClust | paper | n/a | n/a | Method | **Low budget / cold start**: pick typical cluster representatives | For high budgets, uncertainty methods win | arXiv 2202.02794 |
| AL with foundation models | paper (TMLR) | n/a | n/a | Method | Selecting the initial pool and balancing diversity against uncertainty on DINOv2/OpenCLIP features | Gains are smaller with strong features | arXiv 2401.14555 |
| AL evaluation pitfalls | paper | n/a | n/a | Method | Always compare against random sampling plus self/semi-supervised baselines with tuned training | Many reported AL gains vanish under realistic evaluation | arXiv 2301.10625 |

---

## D. Outliers and out-of-distribution (OOD) detection

| Method / tool | pip or source | Latest version (release date) | Licence | Maintained? | Use when | Caveats | Source URL |
|---|---|---|---|---|---|---|---|
| MSP (max softmax) | `pytorch_ood.detector.MaxSoftmax` | pytorch-ood 0.3.3 (2026-09-07) | Apache-2.0 | Active | Always report it as the baseline | Overconfident on far-OOD inputs | arXiv 1610.02136 |
| ODIN | `...ODIN` | same | same | Active | Temperature plus input perturbation | Needs gradient passes and tuning (risk of OOD leakage) | arXiv 1706.02690 |
| Energy score | `...EnergyBased` | same | same | Active | A cheap logit-based improvement over MSP | Classifier-specific | arXiv 2010.03759 |
| Mahalanobis | `...Mahalanobis`, `MultiMahalanobis` | same | same | Active | Class-conditional Gaussians on features | Sensitive to feature-norm variation | arXiv 1807.03888 |
| **Mahalanobis++** | method (L2-normalise features, then Mahalanobis) | n/a | n/a | Method (ICML 2025) | **Default feature-based detector for pretrained backbones**: across 44 models it improved on standard Mahalanobis consistently and outperformed other recent methods | Not verified as a named class in pytorch-ood; the change is trivial to implement | arXiv 2505.18032 |
| KNN (deep nearest neighbours) | `...KNN` | same | same | Active | Non-parametric scoring on L2-normalised embeddings (DINOv2/SigLIP) | k and the size of the feature bank | arXiv 2204.06507 |
| ViM | `...ViM` | same | same | Active | Combines the feature residual with logits | Needs the classifier head | arXiv 2203.10807 |
| ReAct / ASH | `...ReAct`, `...ASH` | same | same | Active | Activation-shaping add-ons to logit scores | Architecture-dependent | arXiv 2111.12797 ; arXiv 2209.09858 |
| MCM (CLIP zero-shot OOD) | `...MCM` | same | same | Active | Zero-shot OOD with vision-language models | Only as good as the prompts and label names | arXiv 2211.13445 |
| Embedding-density outliers | `fiftyone.brain.compute_uniqueness` / `compute_representativeness`; cleanlab Datalab `outlier`; sklearn GMM/LOF on embeddings | see B | see B | Active | Dataset-level outlier triage (junk, wrong modality, corrupt renders) *before* training | Outlier is not the same as an error, so review visually | see B |
| pytorch-ood | `pytorch-ood` | 0.3.3 (2026-09-07) | Apache-2.0 | Active | Detector zoo, including MaxSoftmax, EnergyBased, Mahalanobis, KNN, ViM, ReAct, ASH, DICE, GEN, NNGuide, SCALE, fDBD, MCM, ODIN, OpenMax, GradNorm | Check each detector's needs (fit on ID train features) | https://github.com/kkirchheim/pytorch-ood |
| OpenOOD v1.5 | `pip install git+https://github.com/Jingkang50/OpenOOD` (not on PyPI) | v1.5 (2023-06; journal version accepted Nov 2024) | MIT | Slow (push 2025-12-01; last README news 2024-11) | Standardised OOD benchmark (ImageNet; CLIP/DINOv2 support; full-spectrum shift) | Not packaged; benchmark only | arXiv 2306.09301 |
| torch-uncertainty | `torch-uncertainty` | 0.13.0 (2026-07-22) | Apache-2.0 | Active | Uncertainty and calibration baselines alongside OOD | - | https://torch-uncertainty.github.io |
| anomalib | `anomalib` | 2.6.2 (2026-09-11) | Apache-2.0 | Active | Industrial visual anomaly detection (defects), which is a different problem from semantic OOD | Trained on normal-only data | https://github.com/open-edge-platform/anomalib |
| LAION-C benchmark | dataset | 2025-06 | see paper | Method (ICML 2025) | Robustness evaluation for web-scale models | **ImageNet-C corruptions are no longer OOD for web-scale pretraining** | arXiv 2506.16950 |

---

## E. Shortcut and spurious-correlation detection

| Method / tool | pip or source | Latest version (release date) | Licence | Maintained? | Use when | Caveats | Source URL |
|---|---|---|---|---|---|---|---|
| Shortcut-learning framing | paper | n/a | n/a | Method | Why i.i.d. test accuracy is not enough; test out of distribution | - | arXiv 2004.07780 (Nat Mach Intell 2020) |
| Documented shortcut cases | papers | n/a | n/a | Method | Evidence for mandatory checks: a source-tag/watermark "Clever Hans" in PASCAL VOC; hospital-identifying features in chest X-rays; COVID models using non-lung shortcuts; hidden stratification | Shortcuts come in multiples; removing one can amplify others ("Whac-A-Mole") | arXiv 1902.10178 ; doi:10.1371/journal.pmed.1002683 ; doi:10.1038/s42256-021-00338-7 ; arXiv 1909.12475 ; arXiv 2212.04825 |
| Counterfactual masking / background tests | method (ImageNet-9 "Backgrounds Challenge") | n/a | n/a | Method | Evaluating on foreground-only, background-only and mixed-background variants to measure background reliance | Needs masks or boxes | arXiv 2006.09994 |
| Metadata / image-property correlation | cleanlab Datalab `spurious_correlations` (image properties vs label); a metadata-only baseline (EXIF, size, scanner, site, JPEG quality) | cleanlab 2.9.0 | Apache-2.0 | Slow | Quick leakage screen: if a model trained on metadata alone beats chance, a shortcut is available | Correlation is not proof of use; confirm with counterfactual tests | https://github.com/cleanlab/cleanlab |
| Grad-CAM and variants | `grad-cam` (pytorch-grad-cam) | 1.5.7 (2026-08-21) | MIT | Active | Spot-checking where the model looks (borders, text, rulers, markers) | Saliency can fail sanity checks; never the only evidence | arXiv 1610.02391 ; arXiv 1810.03292 |
| Captum | `captum` | 0.9.0 (2026-04-17) | BSD-3-Clause | Active | Integrated gradients, occlusion and other attributions | Same caveat as Grad-CAM | https://github.com/pytorch/captum |
| LRP + SpRAy (Clever-Hans hunting at scale) | `zennit` / `corelay` | 1.0.0 (2025-07-31) / 1.0.0 (2025-07-21) | GPL-3.0-or-later AND LGPL-3.0-or-later | Slow | Clustering explanation heatmaps to find groups of images with the same strategy | GPL/LGPL | https://github.com/chr5tphr/zennit ; https://github.com/virelay/corelay |
| Domino (slice discovery) | `domino` | 0.1.5 (2022-04-29) | Apache-2.0 | **Stale** (push 2023-10) | Cross-modal (CLIP) embedding mixture model that finds coherent underperforming slices | Unmaintained; reimplement with current CLIP/SigLIP | arXiv 2203.14960 |
| Spotlight / failure directions / B2T | papers (MadryLab/failure-directions repo MIT, stale 2023) | n/a | n/a | Method | Finding error slices from embeddings; B2T names biases via caption keywords | Research code | arXiv 2107.00758 ; arXiv 2206.14754 ; arXiv 2301.11104 |
| LADDER / GH-ESD (2024-2025) | papers | n/a | n/a | Method | LLM/VLM-generated slice hypotheses (LADDER: classifiers). GH-ESD: instance-level slices for detection-type tasks plus the GESD benchmark (2025-12) | Very new; verify hypotheses with held-out statistics | arXiv 2408.07832 ; arXiv 2512.24592 |
| Group DRO | paper + `kohpangwei/group_DRO` (MIT, stale 2023) | n/a | MIT | Method | Group labels available; optimise worst-group loss | Needs strong regularisation or early stopping | arXiv 1911.08731 |
| JTT | paper + `anniesch/jtt` (**no licence**) | n/a | none | Method | No group labels: upweight the errors of a first model | Sensitive to hyperparameters; tune on a group-labelled validation set | arXiv 2107.09044 |
| DFR (last-layer retraining) | paper + repo (BSD-2-Clause, stale 2023) | n/a | BSD-2-Clause | Method | **Cheapest first mitigation**: retrain only the head on a group-balanced held-out set | Needs a small group-balanced set | arXiv 2204.02937 |
| Group/class balancing (SUBG) | paper | n/a | n/a | Method | A strong simple baseline to beat before using fancier methods | - | arXiv 2110.14503 |
| WILDS | `wilds` | 2.0.0 (2021-12-13) | MIT | Stale | Real distribution-shift benchmarks (camera traps, satellite, histopathology) with domain/time splits | Package frozen since 2021 | arXiv 2012.07421 |

---

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

## Could not verify

- **SSCD weights licence**: the repo states MIT for the code only; no separate weight licence was found.
- **SemDeDup licence**: GitHub reports NOASSERTION, and the repo is archived. The licence text was not read.
- **DINOv3 licence**: only the header and sections 1-6 of LICENSE.md were read. That text shows a royalty-free grant with redistribution, attribution and trade-control conditions. Confirm the commercial terms with the full text before use.
- **Kattenborn 2022 "up to 28%"**: taken from the authors' lab abstract page via search; the ScienceDirect page returned 403.
- **The 2025-2026 papers** (Rechecked, Penquitt 2508.17930, Loss Knows Best, LADDER, GH-ESD, Mahalanobis++): title, authors and date were confirmed through the arXiv API. Claims beyond the abstracts were not checked, and no maintained pip package was confirmed for any of them.
- **Mahalanobis++ in pytorch-ood**: the docs detector list did not show it by that name. The absence was not confirmed in the source.
- **Stratified splitting for detection**: no maintained, dedicated library was found. The recipe in section F is a synthesis, not a cited standard.
- **Encord Active**: the PyPI JSON returned 404. Whether it was renamed rather than removed was not checked.
- **Deepchecks vision**: whether it is still officially supported was not checked; only the release and push dates were.
- **LightlyOne**: its current status (the commercial product) and its relationship to LightlyStudio were not checked.
- **fastdup licence history**: when the licence changed to CC BY-NC-ND was not checked; only the current LICENSE file was read.
- **iterstrat and scikit-multilearn**: tested only against scikit-learn 1.7.2 and NumPy 2.3.5, not the latest scikit-learn 1.9.1.
- **Datumaro `validate` command**: the file exists in the CLI; its exact checks were not reviewed.
- **Availability of a public LAION kNN index** for contamination search (clip-retrieval 2.45.0, 2025-08-15) was not checked.
- **UCF101 official splits**: the dataset page confirms that groups share background and viewpoint. Whether the official splits keep groups separate was not re-read.

