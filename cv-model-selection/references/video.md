# 5. Video: models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

### 5A. Action recognition and video encoders

| Model | Best verified number | Code licence | Weights licence | Latest | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|
| **V-JEPA 2** (2506.09985) | "77.3 top-1 accuracy on Something-Something v2"; 39.7 recall@5 on Epic-Kitchens-100 anticipation (abstract) | MIT | **MIT** (HF `facebook/vjepa2-*`, including SSv2 and Diving48 classifiers) | Commit 2026-03-23 | **transformers `VJEPA2ForVideoClassification`** | ViT-L 64-frame is GPU | Yes (probe or fine-tune) |
| VideoMAE V2 (2303.16727) | ViT-g: 90.0% K400, 89.9% K600, 68.7% SSv1, 77.0% SSv2 (abstract) | MIT | **CC-BY-NC-4.0** (HF `OpenGVLab/VideoMAEv2-*`) | Commit 2024-10-08 | Repo | No | NC |
| VideoMAE (2203.12602) | n/a | Code in transformers (Apache) | **CC-BY-NC-4.0** (HF `MCG-NJU/videomae-*`) | n/a | transformers `VideoMAEForVideoClassification` | ViT-B | NC |
| InternVideo2 (2403.15377) | "Superior performance on over 60 video and audio tasks" (abstract) | Apache-2.0 | Apache-2.0, **gated (auto)** (HF `OpenGVLab/InternVideo2-*`) | Commit 2026-07-02 | Repo | No (1B-6B) | Repo |
| torchvision video models | K400 acc@1 (torchvision 0.29 metadata): MViT-V2-S 80.757, Swin3D-B 79.427 (81.643 with `KINETICS400_IMAGENET22K_V1`), Swin3D-S 79.521, MViT-V1-B 78.477, S3D 68.368, R(2+1)D-18 67.463 | BSD-3-Clause | torchvision weights (Kinetics-trained) | torchvision 0.29.0 | `torchvision.models.video.*` | **S3D 8.3M / 18 GFLOPs**; MViT-V2-S 34.5M / 64 GFLOPs | Yes; permissive baseline |
| PyTorchVideo (X3D, MViT, SlowFast) | Not verified | Apache-2.0 | Model zoo | **0.1.5 on PyPI (2022-01-20)**; repo gets only internal lint commits (2026-05-05) | pip or `torch.hub` | X3D-XS/S are CPU-class (numbers not verified) | Effectively unmaintained |
| MMAction2 | n/a | Apache-2.0 | Model zoo | 1.2.0 (2023-10-12); commit 2026-03-18 | mmcv-bound | n/a | Legacy |
| TimeSformer | n/a | Apache (transformers) | **CC-BY-NC-4.0** (HF `facebook/timesformer-*`) | n/a | transformers | No | NC |
| VideoPrism | Not verified | n/a | Not verified | n/a | transformers `VideoPrismForVideoClassification` (present in 5.17) | n/a | n/a |

### 5B. Temporal action localisation
- **OpenTAD** (2502.20361; Apache-2.0; v0.1.0 2024-04-17, commit 2026-07-14): a unified framework with "16 different TAD methods and 9 standard datasets" (abstract). Use it as the default harness.
- **ActionFormer** (2202.07925; MIT; last commit 2024-04-11): a strong single-method baseline.
- **Metric:** mAP averaged over temporal-IoU thresholds, using each dataset's official threshold set. The threshold sets differ between datasets, so always state them; exact sets not re-verified here.

### 5C. Video object segmentation and single-object tracking
- **SAM 2.1** (Apache) is the default promptable video segmenter (transformers `Sam2VideoModel`).
- **EdgeTAM** is the on-device option (Apache; J&F above).
- **SAM 3 tracker** handles concept-prompted video: SA-V test cgF1 30.3 and pHOTA 58.0 vs human 53.1 and 70.5 (SAM 3 README); SAM License.
- **SAMURAI** (2411.11922; Apache; uses SAM 2 weights): "7.1% AUC gain on LaSOT_ext and a 3.5% AO gain on GOT-10k" zero-shot (abstract).
- **VidEoMT** (2602.17807): encoder-only video segmentation; HF weights untagged, repo MIT.

### 5D. Multi-object tracking (MOT)

| Tracker / library | Verified number | Code licence | Latest / status | Notes |
|---|---|---|---|---|
| **`trackers`** (Roboflow) | Clean-room SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU and McByte. MOT17 HOTA with default parameters and YOLOX detections: BoT-SORT 63.7, OC-SORT 61.9, ByteTrack 60.1, SORT 58.4 (README) | Apache-2.0 | **2.6.1 (2026-09-25)**, active | Recommended permissive default; `trackers eval` computes CLEAR, HOTA and Identity metrics |
| supervision | `sv.ByteTrack` deprecated; **removed in `develop` ("as scheduled for supervision-0.31.0")**, use `trackers.ByteTrackTracker` (changelog) | MIT | 0.30.5 (2026-09-22) | Keep for annotation and slicing, not tracking |
| ByteTrack (2110.06864) | 80.3 MOTA, 77.3 IDF1, 63.1 HOTA on MOT17 test at 30 FPS on V100 (abstract) | MIT | Last commit 2022-12-11 | Use the `trackers` re-implementation |
| BoT-SORT (2206.14651) | 80.5 MOTA, 80.2 IDF1, 65.0 HOTA on MOT17 (abstract) | MIT | Last commit 2022-10-30 | Camera-motion compensation |
| OC-SORT (2203.14360) | "700+ FPS on a single CPU" given detections (abstract) | MIT | Commit 2026-04-21 (C++ overhaul) | Non-linear motion (DanceTrack) |
| Deep OC-SORT (2302.11813) | 64.9 HOTA on MOT17, 63.9 on MOT20, 61.3 on DanceTrack (abstract) | MIT | Last commit 2023-05-06 | Adds appearance re-ID |
| BoxMOT | Many trackers plus re-ID models | **AGPL-3.0** | 25.0.0 (2026-09-09) | Copyleft |
| Ultralytics `track` | BoT-SORT and ByteTrack configs | AGPL-3.0 | 8.4.x | Copyleft |

### Metrics: tracking
- **Official TrackEval:** `JonathonLuiten/TrackEval` (MIT) is **not on PyPI**; its last commit was 2022-11-29.
- **PyPI `trackeval` 1.3.0** (2026-03-03) is a **fork** (`kovalp/TrackEval`).
- **py-motmetrics** (MIT): PyPI 1.4.0 (2022-12-26); the repo is active (2026-07-17, "one-call MOTChallenge evaluation API"). It covers CLEAR MOT and IDF1, not HOTA.
- **HOTA** (2009.07736) "explicitly balances ... detection, association and localization". Report HOTA, IDF1 and MOTA together. MOTA over-weights detection.
- Detector quality dominates tracker rankings (the `trackers` README says so), so fix the detector when comparing trackers.
