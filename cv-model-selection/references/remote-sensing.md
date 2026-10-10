# 10. Remote sensing and Earth observation: models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

| Component | What it is | Verified facts | Code licence | Weights licence | Latest | Install | Notes |
|---|---|---|---|---|---|---|---|
| **TorchGeo** (2111.08872) | EO datasets, samplers, transforms, model registry and Lightning tasks | 0.10.0 **requires Python >= 3.12**. Tasks moved to `torchgeo.tasks.*`; the old `torchgeo.trainers.*Task` names are deprecated aliases. Tasks: Classification, SemanticSegmentation, InstanceSegmentation, ObjectDetection, **ChangeDetection**, Regression, spatio-temporal tasks, MAE/MoCo/SimCLR/BYOL. Models include DOFA, CROMA, Copernicus-FM, Panopticon, OlmoEarth v1, Presto, SatCLIP, Scale-MAE, Tessera, **ChangeStar, ChangeViT, FCSiamConc/Diff** | MIT | Per weight (see R3) | 0.10.0 (2026-08-14); active | `pip install torchgeo` | Default EO toolkit |
| **TerraTorch** | Fine-tuning toolkit for geospatial foundation models (Prithvi, TerraMind) | Tracks TorchGeo 0.10 (commit 2026-09-25) | Apache-2.0 AND MIT | Per model | 1.2.13 (2026-09-16 GitHub; PyPI 2026-08-31) | pip | IBM/NASA/ESA models |
| **Prithvi-EO-2.0** (2412.02732) | Multi-temporal HLS foundation model, 300M/600M | Outperforms six other geospatial FMs (abstract) | MIT (NASA-IMPACT repo) | **Apache-2.0** (HF) | HF 2024-11/12 | TerraTorch | Permissive |
| **Clay** | Multi-sensor EO foundation model | n/a | Apache-2.0 | Apache-2.0 (HF `made-with-clay/Clay`) | v1.0.1 "Pip installable" (2025-07-05) | pip (package name not verified) | Permissive |
| **DOFA** (2403.15356) | Wavelength-conditioned any-sensor foundation model | "Generalizes well to unseen modalities" (abstract) | MIT | **CC-BY-4.0** (HF `earthflow/DOFA`, `torchgeo/dofa`) | Commit 2026-07-22 | TorchGeo `dofa_*` | Attribution required |
| **SSL4EO-S12** (2211.07044) | Sentinel-1/2 SSL dataset and weights | "Downstream performance close to, or surpassing" supervised (abstract) | Apache-2.0 | **CC-BY-4.0** (HF `wangyi111/SSL4EO-S12`) | 2024-04 | TorchGeo ResNet/ViT weights | |
| SatMAE (2207.08051) | Temporal and multispectral MAE | Up to 14% gains on land-cover transfer (abstract) | **CC BY-NC 4.0** | NC | Commit 2025-08-10 | Repo | Research only |
| **DINOv3-SAT** | DINOv3 ViT-L/16 and ViT-7B/16 pretrained on SAT-493M satellite imagery | See R3 | DINOv3 License | **DINOv3 License, gated (manual)** (HF `facebook/dinov3-vit{l16,7b16}-pretrain-sat493m`) | HF 2025-08-06 | transformers | Custom-commercial |
| **Oriented detection on DOTA** | See 1C: YOLO26-obb (AGPL), LSKNet (NC, 81.85 mAP), ai4rs O2-DETRs (Apache), MMRotate (frozen) | DOTA: "academic purposes only" | | | | | Dataset licence limits weights |
| **Change detection** | TorchGeo ChangeDetection task; **open-cd** (Apache-2.0; v1.1.0 2024-02-10, commit 2026-06-15; mmseg-based per its lineage, not re-verified); BIT (2103.00208); ChangeFormer (2201.01293) | Recent FM-based papers: 2509.09572 (PEFT on VFMs), 2602.13780 | varies | varies | | | Start with a Siamese UNet (TorchGeo FCSiam) baseline |

### Decision guidance: remote sensing
- **Always run a supervised baseline.** PANGAEA (2412.04204): geospatial FMs "do not consistently outperform supervised models" such as UNet and vanilla ViT.
- **Model choice:** for multispectral time series, Prithvi-EO-2.0 or TerraMind via TerraTorch (Apache). For arbitrary band sets, DOFA (CC-BY-4.0). For RGB very-high-resolution imagery, DINOv3-SAT (custom licence) or DINOv2 (Apache).
- **Oriented objects:** YOLO26-obb (AGPL) or ai4rs O2-DETR (Apache). DOTA-trained weights inherit DOTA's academic-only terms.
- **TorchGeo 0.10 needs Python 3.12+.** Pin accordingly and use `torchgeo.tasks`.
- **Metrics:** per-class IoU/F1 for segmentation. For change detection, F1 and IoU of the *change* class; overall accuracy is meaningless under heavy imbalance. For OBB, rotated-IoU mAP at 0.5 (DOTA's official evaluation) vs mAP50-95 (Ultralytics). Say which.
