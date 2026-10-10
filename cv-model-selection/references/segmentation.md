# 2. Segmentation: models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

### 2A. Semantic, instance and panoptic

| Model / framework | What it is | Best verified number | Code licence | Weights licence | Latest release + date | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|---|
| **EoMT** (DINOv2 / DINOv3) | Encoder-only Mask Transformer: a plain ViT does the segmentation, with no adapter or pixel decoder (2503.19108, CVPR 2025) | With DINOv3: "up to 58.9 PQ on COCO with EoMT-L at 1280x1280" and "up to 59.5 mIoU on ADE20K with EoMT-L at 512x512" (README). Abstract: "up to 4x faster with ViT-L" than adapter-plus-decoder designs | MIT | HF `tue-mps/*` tagged MIT. **DINOv3 variants ship delta weights relative to DINOv3 (README), so the DINOv3 License and its gate apply**; the DINOv2 variants are Apache-backbone | No releases; active (commit 2026-09-24); VidEoMT video extension (2602.17807, CVPR 2026) | transformers `EomtForUniversalSegmentation`, `EomtDinov3ForUniversalSegmentation`; video `VideomtForUniversalSegmentation` | ViT-S/B variants exist (`coco_panoptic_eomt_small_640_2x`); L is GPU | Lightning training in repo |
| **Mask2Former** | Masked-attention universal segmenter (2112.01527) | 57.8 PQ COCO panoptic, 50.1 AP COCO instance, 57.7 mIoU ADE20K (abstract) | MIT | HF `facebook/mask2former-*` tagged "other"; card points to the repo licence (MIT) | **Repo archived**; last commit 2022-05-20 | transformers `Mask2FormerForUniversalSegmentation` | Swin-T/R50 variants heavy on CPU | transformers Trainer (no upstream support) |
| **OneFormer** | Task-token universal segmenter trained once for all three tasks (2211.06220) | "Outperforms specialized Mask2Former models across all three segmentation tasks on ADE20k, CityScapes, and COCO" (abstract) | MIT | MIT (HF `shi-labs/oneformer_*`) | Last commit 2023-07-05 | transformers `OneFormerForUniversalSegmentation` | Heavy | transformers |
| **SegFormer** B0-B5 | Hierarchical MiT encoder with an all-MLP decoder (2105.15203) | B4 50.3 mIoU ADE20K with 64M params; B5 84.0 mIoU Cityscapes val (abstract) | **NVIDIA Source Code License: "research or evaluation purposes only"** | **NC** (HF `nvidia/segformer-*`, `nvidia/mit-b*` tagged "other", linking the same licence) | Last commit 2023-06-13 | transformers `SegformerForSemanticSegmentation`; smp `Segformer` | B0 is small and CPU-friendly (params not re-verified) | Easy, but NC |
| **UperNet + backbone** | Multi-level FPN head (1807.10221) on ConvNeXt, Swin or DINOv2/v3 | Depends on backbone (see R3 for DINOv3 frozen ADE20K) | Head: transformers/smp Apache/MIT | Backbone licence governs | transformers 5.17 | transformers `UperNetForSemanticSegmentation`; smp `UPerNet` | Backbone-dependent | Yes |
| **RF-DETR-Seg** N-2XL | Instance segmentation head on RF-DETR | 2XL 49.9, XL 48.8, L 47.1, N 40.3 COCO mask AP50:95, at 21.8 / 13.5 / 8.8 / 3.4 ms on T4 (SAB) | Apache-2.0 | **Apache-2.0 for all Seg sizes, including XL/2XL** (README) | rfdetr 1.11.0 | `rfdetr`; transformers `RfDetrForInstanceSegmentation` | 33.6-38.6M params | Yes |
| **YOLO26-seg** | Ultralytics instance segmentation | x 47.0 mask / 56.5 box mAP50-95 e2e (docs); SAB 46.8 | AGPL-3.0 | AGPL-3.0 | ultralytics 8.4.x | ultralytics | n 2.7M, CPU ONNX 53.3 ms (docs) | Yes |
| **Mask R-CNN** (torchvision) | Classic two-stage instance segmenter | v2 R50-FPN 47.4 box / 41.8 mask AP (torchvision metadata) | BSD-3-Clause | torchvision weights | torchvision 0.29.0 | `torchvision.models.detection.maskrcnn_resnet50_fpn_v2` | 46M params, 334 GFLOPs | Yes, simplest baseline |
| **segmentation-models-pytorch** (smp) | 12 encoder-decoder architectures (Unet, Unet++, FPN, DeepLabV3+, UPerNet, Segformer, DPT...) and "800+ pretrained encoders" including timm `tu-*` (README) | n/a | MIT, **except `encoders/mix_transformer.py` (the `mit_b*` encoders), which carries the NVIDIA Source Code License header** | Per encoder (smp-hub mirrors) | 0.5.0 (2025-04-17); commits continue (2026-09-15, dependency bumps) | `pip install segmentation-models-pytorch` | Pick a MobileNet/EfficientNet/MobileOne encoder | Yes; the default for small-data semantic seg |
| MMSegmentation | OpenMMLab semantic seg | n/a | Apache-2.0 | Model zoo | 1.2.2 (2023-12-14); last commit 2024-03-22; mmcv < 2.2.0 | Frozen | n/a | Legacy only |

### 2B. Promptable and interactive segmentation

| Model | Best verified number | Code licence | Weights licence | Latest | Install route | CPU feasibility | Notes |
|---|---|---|---|---|---|---|---|
| **SAM 2 / 2.1** | Image segmentation "more accurate and 6x faster than SAM"; video with "3x fewer interactions" (2408.00714) | Apache-2.0 | Apache-2.0 (HF `facebook/sam2.1-hiera-{tiny,small,base-plus,large}`) | sam2 1.1.0 (2024-12-21); last commit 2024-12-16 | transformers `Sam2Model`, `Sam2VideoModel`; `pip install sam2` | Hiera-tiny is the CPU option | Permissive default for promptable image and video masks |
| **SAM 3 / 3.1** | See 1D. Concept prompts (noun phrase or exemplar) return all instances; "doubles the accuracy of existing systems" in promptable concept segmentation (abstract) | SAM License | SAM License, gated | 0.1.4 PyPI (2026-05-26); SAM 3.1 checkpoints 2026-03-27 | transformers `Sam3*`; `pip install sam3` | GPU (about 850M) | Custom-commercial; bans ITAR, military and nuclear uses |
| SAM (v1) | Original promptable model (2304.02643) | Apache-2.0 | Apache-2.0 (HF `facebook/sam-vit-*`) | segment-anything 1.0 (2023-04-06) | transformers `SamModel` | ViT-B is slow on CPU | Superseded by SAM 2.1 |
| SAM-HQ | High-quality masks (2306.01567) | Apache-2.0 | Apache-2.0 (repo) | Last commit 2025-09-12 | transformers `SamHQModel` | as SAM | Better thin structures |
| **EfficientSAM** | "~4 AP" gain over other fast SAMs on COCO/LVIS zero-shot instance segmentation (2312.00863) | Apache-2.0 | Apache-2.0 (repo) | Last commit 2024-12-24 (added EfficientTAM checkpoints) | Repo | Yes (ViT-T/S) | Edge promptable |
| **MobileSAM** | "More than 60 times smaller yet performs on par with the original SAM"; about 10 ms per image on GPU (2306.14289) | Apache-2.0 | Apache-2.0 (repo) | Commit 2026-05-05 | Repo; Ultralytics wrapper (AGPL) | **Yes** | Drop-in SAM-v1 decoder |
| **EdgeTAM** | 87.7 / 70.0 / 72.3 / 71.7 J&F on DAVIS 2017 / MOSE / SA-V val / SA-V test (2501.07256) | Apache-2.0 | Apache-2.0 (HF `facebook/EdgeTAM`) | HF 2025-04-30; commit 2026-01-27 | transformers `EdgeTamModel`, `EdgeTamVideoModel` | Designed for on-device | On-device SAM 2 |
| SAM3-LiteText (`vil-uob/sam3-litetext-s0`) | Replaces SAM 3's 353M text encoder with a distilled MobileCLIP-based one; **keeps the SAM 3 ViT-H image encoder** (card) | n/a | HF tag Apache-2.0, **but it redistributes SAM 3 weights (SAM License covers derivatives)**; the MobileCLIP-derived text tower licence was not verified | HF 2026-03-26 | transformers `Sam3LiteTextModel` | GPU | Licence trap |

### 2C. Medical segmentation

| Model / framework | What it is | Best verified number | Code licence | Weights licence | Latest release + date | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|---|
| **nnU-Net v2 + ResEnc presets** | Self-configuring U-Net framework (1809.10486); Residual-Encoder presets M/L/XL ("nnU-Net Revisited", 2404.09556) | 5-fold CV Dice (resenc_presets.md): ResEnc L scores KiTS 88.17 vs 86.04 for original nnU-Net, and AMOS 89.41 vs 88.64. VRAM: M 9.1 GB, L 22.7 GB, XL 36.6 GB | Apache-2.0 | You train your own; OpenMind SSL checkpoints (`MIC-DKFZ/ResEncL-OpenMind-*`) exist; **nnFoundation weights CC-BY-SA-4.0** | nnunetv2 2.8.1 (2026-07-01); active (commit 2026-09-25) | `pip install nnunetv2` | Training is GPU; inference on CPU is slow but possible | Full; fine-tune from nnssl or nnFoundation checkpoints (documented) |
| **nnFoundation** CNN / ViT | 3D radiology foundation models on 2.1M CT/MR/PET volumes from 125 datasets, evaluated on 108 tasks (2609.26924, 2026-09-22) | Not quoted | nnssl / nnU-Net (Apache) | **CC-BY-SA-4.0** (HF `MIC-DKFZ/nnFoundationCNN` 102M ResEnc, `nnFoundationViT` 674M Primus; 2026-09-24) | New | via nnU-Net fine-tuning docs | CNN 102M | Yes (segmentation now; detection and classification "TBA" per card) |
| **TotalSegmentator** | nnU-Net models for whole-body CT/MR (2208.05868) | Original paper: Dice 0.943 on 104 structures (abstract). Current `total` task: 117 classes (CT); `total_mr`: 50 classes | Apache-2.0 | **Open tasks Apache-2.0** (`total`, `total_mr`, lung_vessels, body, vertebrae, liver_segments...). **Licensed tasks** (free NC licence or paid commercial): heartchambers_highres, appendicular_bones, tissue_types, brain_structures, coronary_arteries, face and others. **brain_aneurysm: CC BY-NC 4.0, "no commercial license available"** (README) | PyPI 2.18.0 (2026-08-12); **GitHub v3.0.0 tag and weights 2026-09-07, not yet on PyPI** | `pip install TotalSegmentator` | CPU works with `--fast` or `--roi_subset` (README) | Train with nnU-Net on its dataset |
| **MedSAM** | SAM ViT-B fine-tuned on medical images (2304.12306) | Promptable; no single number | Apache-2.0 | Apache-2.0 (HF `wanglab/medsam-vit-base`) | v1.0.0 (2024-01-02) | transformers `SamModel` | ViT-B | Yes |
| **MedSAM2** | SAM 2 fine-tuned on "over 455,000 3D image-mask pairs and 76,000 frames"; "reduce manual costs by more than 85%" (2504.03600) | As stated | Apache-2.0 | **CC-BY-SA-4.0** (HF `wanglab/MedSAM2`) | Last commit 2025-07-11 | Repo | GPU | Yes |
| **VISTA3D** (MONAI) | 3D automatic (127 classes) plus interactive segmentation (2406.05285) | "State-of-the-art ... in both 3D automatic ... and 3D interactive segmentation" (abstract) | Apache-2.0 (model-zoo code) | **NVIDIA NCLS v1 (non-commercial)** per `MONAI/VISTA3D-HF` card | model-zoo bundle 0.5.11 | MONAI bundle / HF | GPU | Yes (NC) |
| MONAI Auto3DSeg | AutoML ensemble (SegResNet, DiNTS, SwinUNETR) | nnU-Net Revisited (A3DS rows): KiTS 81.11 / 65.28 / 52.82 for SegResNet / DiNTS / SwinUNETR, vs 88.17 for ResEnc L | Apache-2.0 | n/a | monai 1.6.0 (2026-06-22); 1.6.1rc0 tag (2026-09-18) | `pip install monai` | GPU | Yes |

### Decision guidance: segmentation
- **Semantic or panoptic, commercial:**
  1. **EoMT with DINOv2** (MIT + Apache backbone) or **Mask2Former / OneFormer via transformers** (MIT).
  2. For small data, **smp Unet/UPerNet with a timm ConvNeXt or EfficientNet encoder** (MIT; avoid `mit_b*`).
  3. EoMT-DINOv3 is stronger (59.5 mIoU ADE20K) but inherits the DINOv3 licence and gate.
- **Instance segmentation, real-time:**
  1. **RF-DETR-Seg** (Apache for every size; 2XL 49.9 mask AP).
  2. YOLO26-seg if AGPL is acceptable (x 47.0).
  3. torchvision Mask R-CNN v2 as the simplest baseline.
- **Promptable:** **SAM 2.1** (Apache) by default; EdgeTAM or MobileSAM on device; SAM 3 when text or concept prompts are needed and the SAM License is acceptable.
- **Medical 3D:**
  1. **nnU-Net v2 with ResEnc L** first. nnU-Net Revisited found "the recipe for state-of-the-art performance is 1) employing CNN-based U-Net models ... 2) using the nnU-Net framework, and 3) scaling models to modern hardware resources".
  2. TotalSegmentator open tasks for anatomy.
  3. MedSAM2 for interactive labelling (share-alike weights).
  4. Do not default to transformer 3D segmenters (SwinUNETR, nnFormer) or Auto3DSeg; both trail in that benchmark.
- **Avoid:** SegFormer and smp `mit_b*` in products (NC); MMSegmentation for new work (frozen); Mask2Former upstream repo (archived; use transformers).

### Licence traps: segmentation
- **SegFormer / MiT weights and the smp `mix_transformer.py` code:** NVIDIA NC.
- **EoMT-DINOv3:** delta weights on DINOv3, so the DINOv3 License applies.
- **SAM 3 / 3.1 and all derivatives (SAM3-LiteText's Apache tag notwithstanding):** SAM License.
- **MedSAM2 weights:** CC-BY-SA (share-alike).
- **nnFoundation:** CC-BY-SA-4.0.
- **VISTA3D weights:** NCLS v1.
- **TotalSegmentator:** licensed subtasks, and brain_aneurysm is NC with no commercial option.

### Metrics: segmentation
- **mIoU:**
  - Standard ADE20K and Cityscapes results use **dataset-level** accumulated intersection and union per class, with an ignore label (commonly 255). The ADE20K background handling ("reduce_zero_label" in OpenMMLab configs) was not re-verified here, so check how your dataset handles class 0.
  - `torchmetrics.segmentation.MeanIoU` (1.9.0) instead averages **per-sample** IoU over present classes (see `update()`), has no `ignore_index`, defaults to `input_format="one-hot"` and `include_background=True`. Its numbers are not comparable with papers. Compute a confusion matrix yourself or use `torchmetrics.classification.MulticlassJaccardIndex` with `ignore_index` (not re-verified here).
- **PQ:** `torchmetrics.detection.PanopticQuality` (1.9.0) or the COCO panopticapi (not on PyPI).
- **Instance masks:** COCO mask AP via pycocotools (`iouType='segm'`).
- **Boundaries:** Boundary IoU (2103.16562) for thin structures.
- **Medical:** Dice, NSD and HD95 via MONAI metrics or torchmetrics `DiceScore` / `HausdorffDistance` (1.9.0). Follow Metrics Reloaded (2206.01653) for metric choice by task, and report per-structure values, not only the mean.
