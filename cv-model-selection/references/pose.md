# 3. Pose and keypoints: models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

| Model / framework | What it is | Best verified number | Code licence | Weights licence | Latest release + date | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|---|
| **ViTPose / ViTPose++** | Plain-ViT top-down pose (2204.12484); ViTPose+ adds task-specific FFNs for human, whole-body and animal keypoints (2212.04246) | ViTPose-G (COCO+AIC+MPII, 576x432): 81.0 AP / 85.6 AR on COCO val with a 56-mAP person detector; ViTPose-B 77.1 (README) | Apache-2.0 | HF `usyd-community/vitpose-*` tagged Apache-2.0. Upstream backbones are MAE-pretrained; whether MAE's NC terms carry over was not verified | Repo last commit 2025-12-25 | **transformers `VitPoseForPoseEstimation`** (HF 2025-01) | Sizes S/B/L/H/G (params not re-verified); top-down needs a separate person detector | transformers or repo (mmcv-based) |
| **RTMPose / RTMO / RTMW / DWPose** | Real-time top-down (RTMPose 2303.07399), one-stage (RTMO 2312.07526), whole-body (RTMW 2407.08634), distilled whole-body (DWPose 2307.15880) | RTMPose-m 75.8% AP on COCO at "90+ FPS on an Intel i7-11700 CPU"; RTMO-l 74.8% AP on COCO val2017 at 141 FPS on V100; RTMW-l 70.2 mAP on COCO-WholeBody; DWPose raises RTMPose-l whole-body from 64.8% to 66.5% (abstracts) | Apache-2.0 (MMPose, DWPose) | Apache-2.0 (MMPose model zoo; HF `yzd-v/DWPose` Apache-2.0) | MMPose 1.3.2 (2024-07-12), last commit 2025-08-04 (mmcv-bound). **rtmlib 0.0.16 (2026-08-04)** runs RTMPose, DWPose, RTMO, RTMW and ViTPose "without mmcv, mmpose, mmdet" | **`pip install rtmlib`** (ONNX Runtime or OpenVINO) | **Best CPU option** (RTMPose-m CPU figure above) | Training still needs MMPose (frozen stack) |
| **Sapiens2** 0.4B-5B | Human-centric foundation models at 1K-4K resolution (2604.21681); pose (308 keypoints), body-part segmentation, normals, pointmap, albedo | "+4 mAP" pose and "+24.3 mIoU" body-part segmentation over Sapiens v1 (abstract) | Sapiens2 License (custom Meta; acceptable-use bans on military, ITAR and similar; no non-commercial clause found by keyword search) | Same (HF `facebook/sapiens2-*`, 2026-04-23) | HF 2026-04/05 | **transformers `Sapiens2ForPoseEstimation`, `Sapiens2ForSemanticSegmentation`** | No (0.4B+ at 1K) | Repo |
| Sapiens (v1) | Previous generation (2408.12569) | n/a | **CC-BY-NC-4.0** | **CC-BY-NC-4.0** | Repo commit 2026-05-26 | torchscript checkpoints | No | NC |
| **YOLO26-pose** | Ultralytics one-stage pose with RLE loss | x 71.6 / 91.6 mAP50-95 / mAP50 (OKS) on COCO val; n 57.2 (docs). Paper per docs: up to +7.2 AP over YOLO11-pose | AGPL-3.0 | AGPL-3.0 | ultralytics 8.4.x | ultralytics | n 2.9M, CPU ONNX 40.3 ms (docs) | Yes, custom keypoint schemas |
| **RF-DETR Keypoint (preview)** | DETR keypoint head | 71.8 COCO keypoint AP50:95, 9.7 ms on T4, 40.7M params (SAB) | Apache-2.0 | Apache-2.0 | rfdetr 1.11.0 | `rfdetr` (`RFDETRKeypointPreview`) | GPU-class | Preview |
| MediaPipe (pose, hand and face landmarkers) | On-device landmark models | Not verified | Apache-2.0 | Not verified per model | v1.0.0 (2026-07-28) | `pip install mediapipe` (version not checked) | **Yes (mobile, CPU)** | Limited (Model Maker) |
| HaMeR (hands, 3D) | Transformer hand mesh recovery (2312.05251) | Not quoted | MIT | Needs the **MANO model**, which requires registration on the MANO site (licence terms not fetched) | Commit 2026-02-07 | Repo | No | Repo |
| WiLoR (hands) | 3D hand localisation and reconstruction (2409.12259) | Not quoted | **CC BY-NC-ND 4.0** | **CC-BY-NC-4.0** (HF) | Commit 2026-04-07 | Repo | No | NC |
| DeepLabCut / SuperAnimal (animals) | Animal pose toolbox (SuperAnimal: 2203.07436) | Not quoted | **LGPL-3.0** | SuperAnimal HF repos have **no licence tag** | DeepLabCut v3.0.2 (2026-09-23) | `pip install deeplabcut` (version not checked) | Yes (ResNet variants) | Yes |
| InsightFace (face detection and recognition) | Face analysis library | n/a | MIT | **"non-commercial research purposes only"**, for manual and auto-downloaded models alike (README) | insightface 2.0 on PyPI (2026-09-08) | pip | Yes | Commercial licence by contact |

### Decision guidance: pose
- **CPU or real-time, commercial:** **rtmlib with RTMPose-m or RTMO** (Apache; mmcv-free). RTMPose-m does 75.8 AP at 90+ FPS on a desktop CPU (abstract).
- **Highest accuracy, top-down:** **ViTPose(+) via transformers** (Apache tag) or **Sapiens2** (custom licence, 0.4B-5B, 308 keypoints).
- **Single-stage with an easy custom-keypoint trainer:** YOLO26-pose if AGPL is acceptable; otherwise RF-DETR Keypoint (preview, Apache).
- **Animals:** DeepLabCut (LGPL) or ViTPose++ (AP-10K and APT-36K in the abstract).
- **Hands:** HaMeR (MIT, but MANO access is gated). Avoid WiLoR in products (NC-ND).
- **Faces:** InsightFace pretrained models are NC.
- **Comparing numbers:** top-down numbers depend on the person detector (ViTPose's table uses a 56-mAP detector). Do not compare them directly with bottom-up or one-stage numbers (RTMO, YOLO-pose).

### Metrics: pose
- **COCO OKS AP** via `pycocotools` `COCOeval(iouType='keypoints')`. Verified defaults: `maxDets=[20]`, area ranges all/medium/large (no "small"), and `kpt_oks_sigmas` hard-coded for the 17 COCO keypoints. **For custom skeletons, set `params.kpt_oks_sigmas` yourself**, or the metric is silently wrong or crashes.
- **Other benchmarks:** PCK / PCKh for MPII-style and animal benchmarks.
- **Library gaps:** torchmetrics has no OKS metric.
- **Alternatives:** faster-coco-eval supports keypoints and CrowdPose (README).
