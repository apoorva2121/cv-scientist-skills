# Licence traps and unverified items (task models)

Source: research R4 (verified 2026-09-27). How facts were checked:

**Checked 2026-09-27.** How each fact was checked:
- **Packages:** PyPI JSON API (`info.version`, `license`/`license_expression`, upload time of the first file of that release).
- **Repos:** licence text read from `raw.githubusercontent.com/<repo>/<default-branch>/LICENSE*`; archive flag, latest release and latest commit read from the repo page and its `releases.atom` / `commits/<branch>.atom` feeds (no GitHub API quota used). Licence *change dates* come from the `commits/<branch>/LICENSE.atom` history, with the old text fetched at the older commit SHA.
- **Weights:** Hugging Face API (`cardData.license`, `license_name`, tags, `gated`, `createdAt`), plus the card text or the upstream README where they disagree. **Where HF metadata and upstream disagree, this report follows the more restrictive one** and lists the conflict under licence traps.
- **Papers:** arXiv API (ID, title, v1 date); every quoted claim comes from the abstract unless another source is named.
- **Benchmarks:** only from primary sources: the paper abstract, the official README or model zoo, or the official docs. The source is named in each cell. "SAB" means Roboflow's single-artifact benchmark harness, which re-measures every model with pycocotools on the full COCO val2017 split and T4 TensorRT FP16 at batch 1 (rf-detr README).
- **Library classes:** read from an installed `transformers` 5.17.0 (`dir(transformers)`), plus torchvision 0.29.0 weight metadata in the source files at tag v0.29.0.

**Environment baseline (PyPI, 2026-09-27):** torch 2.14.0 and torchvision 0.29.0 (2026-09-02); transformers 5.17.0 (2026-09-09); timm 1.0.30 (2026-09-22); ultralytics 8.4.163 (2026-09-25); rfdetr 1.11.0 (2026-09-24); supervision 0.30.5 (2026-09-22); trackers 2.6.1 (2026-09-25); sahi 0.12.7 (2026-09-25); pycocotools 2.0.11 (2025-12-15, the ppwwyyxx fork); faster-coco-eval 1.8.0 (2026-08-24); torchmetrics 1.9.0 (2026-03-09); onnxruntime 1.30.0 (2026-09-10); openvino 2026.4.0 (2026-09-16).

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

## Most decision-changing findings (2026-09-27)

1. **DEIMv2 and EdgeCrafter became non-commercial on 2026-08-24.**
   - DEIMv2's `LICENSE` (Apache-2.0 since 2025-09-19) was replaced by a custom "DEIMv2 License" in commit "docs: update license information". It grants rights "solely for Non-Commercial Purposes", and section 5 explicitly covers model weights. The EdgeCrafter repo (ECDet/ECInsSeg/ECPose) changed the same day.
   - Stale mirrors still say Apache-2.0: the HF `Intellindust/ECDet_L` tag and the transformers-format mirrors `harshaljanjani/DEIMv2_*_Transformers` (created 2026-04-16, when upstream was still Apache).
   - DEIM (v1) stays Apache-2.0 (last licence commit 2025-07-21).
   - Even before the change, DEIMv2 S/M/L/X embed DINOv3-derived backbones, which carry the DINOv3 License.
2. **YOLO26 exists and is the current Ultralytics line** (ultralytics 8.4.0, 2026-01-14; paper arXiv 2606.03748, 2026-06-02).
   - It is NMS-free (optional one-to-one head), drops DFL and scores 40.9-57.5 COCO mAP50-95 across n to x (Ultralytics docs).
   - It is AGPL-3.0 like all Ultralytics weights. YOLO27 is announced but "not yet available" (docs).
   - Independent re-measurement matters. SAB measures YOLO26-X at 56.9, which matches the vendor's NMS-free e2e column rather than the headline 57.5. An independent study (2605.24831) found YOLO26's gain over YOLOv8 shrinks to 0.224 vs 0.214 mAP50:95 on VisDrone small objects, and that YOLOv8-s was faster on GPU (6.92 vs 8.38 ms).
3. **RF-DETR is the strongest permissive real-time detector with a maintained training package.**
   - RF-DETR-L scores 56.5 COCO AP50:95 at 6.8 ms on T4 (SAB).
   - N-L detection, all segmentation sizes and the keypoint preview are Apache-2.0. **XL and 2XL detection (58.6 / 60.1 AP) are under "PML 1.0"** via `rfdetr[plus]`.
   - RF-DETR is now in transformers 5.17 (`RfDetrForObjectDetection`, `RfDetrForInstanceSegmentation`).
4. **D-FINE added a dataset-licence caveat (2026-08-19).** Objects365-pretrained checkpoints (`*_obj365`, `*_obj2coco`) "should not be assumed to be commercially cleared under the D-FINE license". The same logic applies to every Objects365-pretrained detector (RT-DETR O365 weights, DINO-DETR, LW-DETR, EdgeCrafter X).
5. **OpenMMLab is effectively frozen.**
   - MMDetection's last release was 3.3.0 (2024-01-05) and its last commit 2024-02-05. MMSegmentation's last release was 1.2.2 (2023-12-14) and its last commit 2024-03-22. MMRotate's last release was 0.3.4 (2023-02-01).
   - mmdet 3.3.0 asserts `mmcv < 2.2.0`, but prebuilt mmcv wheels on download.openmmlab.com exist only as mmcv 2.2.0, and only up to torch 2.4 (no index page for torch 2.5 or 2.6). The current torch is 2.14.
   - Use transformers, Ultralytics or model-specific repos instead. For pose, use `rtmlib` (no mmcv).
6. **Several popular weights are NC, and some mirrors say otherwise:**
   - SegFormer (NVIDIA Source Code License: "research or evaluation purposes only").
   - Depth Anything V2 Base/Large and DA3 Large/Giant (CC-BY-NC-4.0). Upstream lists DA3-LARGE-1.1 as CC BY-NC 4.0, but HF tags it Apache-2.0.
   - VideoMAE / VideoMAE V2 (CC-BY-NC-4.0).
   - Sapiens v1 (CC-BY-NC-4.0), UniDepth (CC BY-NC 4.0), DUSt3R/MASt3R (CC BY-NC-SA 4.0), Nougat (CC-BY-NC-4.0), LayoutLMv3 (CC-BY-NC-SA-4.0).
   - Sonata point-cloud weights (upstream CC-BY-NC 4.0; HF mirror `Pointcept/Sonata` tagged MIT).
   - InsightFace face models ("non-commercial research purposes only").
   - Qwen2.5-VL-3B (Qwen Research licence).
   - VISTA3D weights (NVIDIA NCLS v1).
7. **Permissive replacements now exist for most of those:**
   - Detection and segmentation: RF-DETR, D-FINE, RT-DETRv2/v4.
   - Universal segmentation: EoMT (MIT).
   - Depth: Depth Anything V2-Small, DA3-SMALL/BASE/METRIC-LARGE/MONO-LARGE, MoGe-1/2/3, MapAnything-apache, Pi3.
   - Video: V-JEPA 2 (MIT).
   - VLMs: Qwen3-VL, Qwen3.5/3.6 and Gemma 4, all Apache-2.0. Gemma 4 moved from Gemma terms to Apache-2.0; Qwen3.5 (2026-02) is natively multimodal.
   - Document parsing: PaddleOCR-VL 1.5/1.6 and MinerU2.5-Pro (Apache-2.0).
8. **OCR licence shifts in 2026:**
   - Surya's code moved from GPL-3.0 to Apache-2.0 (2026-05-14), and Marker's did the same (2026-07-17). **Their weights, and Chandra's, remain a modified OpenRAIL-M with revenue caps:** $5M for Surya/Marker, $2M plus a no-competing-API clause for Chandra.
   - MinerU code is "Apache-2.0 plus additional terms": a separate licence is needed above 100M MAU or USD 20M monthly revenue, and online-service attribution is required. MinerU2.5-2509 weights are AGPL-3.0, while MinerU2.5-Pro weights are Apache-2.0.
   - HunyuanOCR's licence excludes the EU, UK and South Korea.
9. **The document-parsing leaderboard moved from pipelines to sub-2B VLMs.** On OmniDocBench v1.6 (end-to-end Overall, README updated 2026-09-11):
   - PaddleOCR-VL-1.6 (0.9B) scores 96.34, MinerU2.5-Pro (1.2B) 95.75 and GLM-OCR (0.9B) 95.22.
   - General VLMs score lower: Qwen3-VL-235B 89.78 and Gemini 3 Pro 92.91. The Marker pipeline scores 78.44.
   - OmniDocBench is maintained by opendatalab, the MinerU team, so re-check on your own documents.
10. **Zero-shot VLM or open-vocabulary detection is not a substitute for fine-tuning off-distribution.**
    - RF100-VL (2505.20612): GroundingDINO and Qwen2.5-VL "achieve less than 2% zero-shot accuracy on challenging medical imaging datasets".
    - VLMs "significantly underperform CLIP" on standard image classification (2405.18415).
    - But zero-shot open-vocabulary detectors beat classic few-shot detectors on common categories: GroundingDINO scores 48 AP zero-shot vs 33 AP for few-shot SOTA on COCO (2312.14494).
11. **Metric-library traps that silently change numbers:**
    - `torchmetrics.segmentation.MeanIoU` (1.9.0) averages per-image IoU over (image, class) pairs where the class is present. It has no `ignore_index` argument and defaults to `input_format="one-hot"`. This is **not** the dataset-level accumulated mIoU that ADE20K and Cityscapes results use.
    - `pycocotools.COCOeval` defaults to `iouType='segm'`.
    - COCO small/medium/large buckets use each ground-truth annotation's `area` field (segment area for COCO), not box area.
    - `sv.ByteTrack` is removed in supervision's `develop` branch (scheduled for 0.31.0); use `trackers`.
    - The PyPI package `trackeval` is a fork (kovalp/TrackEval), not the official JonathonLuiten/TrackEval, which is not on PyPI and was last committed 2022-11-29.

## Licence traps (consolidated)

1. **Changed in 2026:** DEIMv2 and EdgeCrafter went Apache to NC on 2026-08-24. Surya went GPL to Apache on 2026-05-14 and Marker on 2026-07-17 (code only; weights stay revenue-capped OpenRAIL-M). Gemma moved from Gemma terms (Gemma 3, PaliGemma 2) to Apache-2.0 for Gemma 4.
2. **Mirror or metadata disagrees with upstream (follow the more restrictive):**
   - DA3-LARGE-1.1: HF Apache vs README CC BY-NC.
   - Depth Pro: HF `apple-amlr` vs card "Apple-ASCL" vs GitHub sample-code.
   - Sonata: HF MIT vs README CC-BY-NC.
   - ECDet_L: HF Apache vs licence file NC.
   - `harshaljanjani/DEIMv2_*`: Apache tag, but upstream is now NC and the DINOv3 backbone is custom-licensed.
   - EoMT-DINOv3: MIT tag, but ships delta weights on DINOv3.
   - SAM3-LiteText: Apache tag, but contains the SAM 3 image encoder.
   - DocLayout-YOLO: Apache weights tag, AGPL code.
   - InternVL3-8B: metadata says `qwen`, card says MIT plus Apache.
   - VisA: CC BY 4.0 upstream vs CC BY-NC-SA per the anomalib docstring.
   - VoxelMorph and uniGradICON: repo Apache vs PyPI MIT.
3. **NC weights frequently used by accident:**
   - SegFormer / MiT (including smp `mit_b*`), DA V2 B/L/G, DA3 L/G, VGGT-1B, MapAnything (non-apache), Pi3X, UniDepth, DUSt3R/MASt3R.
   - VideoMAE / V2, TimeSformer, Sapiens v1, WiLoR, Nougat, LayoutLMv3.
   - Qwen2.5-VL-3B (and derivatives such as Nanonets-OCR2-3B), Perception LM, Rex-Omni, InsightFace, SatMAE, LSKNet, Sonata, Concerto, Utonia, VISTA3D.
   - MVTec AD / AD 2 (data), DOTA (data).
4. **Copyleft:**
   - AGPL-3.0: all Ultralytics models (YOLOv8-YOLO26, YOLOE, the YOLO-World port, the RT-DETR wrapper, trackers), BoxMOT, DocLayout-YOLO code, MinerU2.5-2509 weights, albumentationsx (2.4.10, AGPL-3.0-only; see R2).
   - GPL-3.0: YOLO-World upstream, YOLOv9/v7 originals, mmyolo, the PANGAEA benchmark code.
   - Share-alike: MedSAM2 weights and nnFoundation (CC-BY-SA-4.0).
5. **Custom-commercial:**
   - SAM 3 / 3.1, Sapiens2 and VGGT-1B-Commercial (Meta licences with acceptable-use bans).
   - DINOv3 (and everything built on it: DEIMv2 S-X, EoMT-DINOv3, DINOv3-SAT, possibly RT-DETRv4 via distillation).
   - RF-DETR XL/2XL (PML 1.0).
   - MinerU code (MAU and revenue thresholds, attribution).
   - Chandra ($2M, no competing API), Surya and Marker weights ($5M).
   - Moondream (BUSL).
   - HunyuanOCR (no EU/UK/KR).
   - Qwen2.5-VL-72B (100M MAU).
   - TotalSegmentator licensed subtasks.
6. **Dataset-inherited restrictions:** Objects365-pretrained detectors (D-FINE warning), DOTA-trained OBB models, MVTec-trained anomaly models, and Sonata's NC-derived weights.

## Could not verify

- **Licence texts not obtained:**
  - RF-DETR "PML 1.0" (the `rfdetr-plus` repo licence file was not found on main/develop/master).
  - MANO (HaMeR dependency).
  - Metric3D weights (not stated in README).
  - SuperAnimal (DeepLabCut) weights (no HF tag).
  - RadFM and TrOCR weights (no HF tag).
  - DUSt3R/MASt3R HF weights (untagged; assumed to follow the repo's CC BY-NC-SA).
  - GOT-OCR2 GitHub code (no LICENSE file).
  - Video Depth Anything weights.
  - VideoPrism.
  - Real-IAD.
  - Re-ID dataset terms (Market-1501 and others).
  - The MobileCLIP-derived text tower in SAM3-LiteText.
- **Licence interpretations (legal questions, not facts):**
  - Whether DEIMv2 copies obtained before 2026-08-24 remain Apache-2.0.
  - Whether detectors distilled from a DINOv3 teacher (RT-DETRv4, EdgeCrafter) are DINOv3 "derivative works".
  - Whether ViTPose weights inherit NC terms from MAE pretraining.
  - Whether Gemma 4 carries a separate prohibited-use policy in addition to Apache-2.0 (the licence page header reads "Apache License 2.0 / Gemma"; its body was not fully readable).
- **Sapiens2 licence:** a keyword search found no non-commercial clause, but the full text was not reviewed.
- **SAM 3.1 licence:** assumed identical to SAM 3 (the HF card is gated and was not readable).
- **Benchmark numbers not verified:**
  - YOLOv9, YOLOv13, LW-DETR paper numbers (SAB numbers used instead); LLMDet; MM-Grounding-DINO; Florence-2; CLIP-ReID; FastFlow; docTR; EasyOCR; Tesseract.
  - X3D / MViT in PyTorchVideo.
  - VLM general benchmarks (MMMU, DocVQA etc.) for Qwen3-VL, Qwen3.5, InternVL3.5 and Gemma 4 (vendor cards only; not extracted).
  - Chandra OCR 2 on OmniDocBench (not in the parsed table).
- **Environment details not verified:**
  - Hardware for the Ultralytics "CPU ONNX" latency column.
  - The `pip` package names and versions for MediaPipe, DeepLabCut and Clay.
  - Whether detectron2 builds against torch 2.14.
  - mmcv wheel availability for torch 2.7 and later (only the absence of torch 2.5 and 2.6 index pages was checked).
- **Maintenance or provenance not confirmed:**
  - Whether the PyPI `faiss-gpu` 1.15.1 is an official Meta release.
  - VLMEvalKit's PyPI status.
  - Whether open-cd is still mmseg-based.
  - Ultralytics YOLO26 semantic-segmentation and depth-estimation task weights and numbers (docs list the tasks, but specific models were not checked).
- **Metric conventions not re-checked:**
  - ANLS threshold 0.5 (standard, not re-read from the paper).
  - The ADE20K `reduce_zero_label` convention.
  - `MulticlassJaccardIndex` `ignore_index` behaviour.
  - Temporal-action-localisation tIoU threshold sets.
  - KITTI and nuScenes metric details.
- **Independent verification of vendor claims:** PP-OCRv6's "+4.6% / +5.1% / 5.2x CPU", Qwen3.5 "outperforms Qwen3-VL", and the TeleOCR and OvisOCR2 leaderboard entries (HF licence tags only; base models not checked).
