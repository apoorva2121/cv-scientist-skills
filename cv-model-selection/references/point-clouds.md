# 12. Point clouds (reference level): models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

| Component | Status | Code licence | Weights licence |
|---|---|---|---|
| **Pointcept** (PTv3 2312.10035, Sonata 2503.16429, Concerto 2510.23607, Utonia 2603.03283) | **v1.7.0 "Utonia" (2026-04-02)**; active (commit 2026-09-11) | MIT | PTv3: MIT (HF). **Sonata: CC-BY-NC 4.0** per upstream README ("restricted by NC of datasets like HM3D, ArkitScenes"); the HF mirror `Pointcept/Sonata` is tagged MIT, a conflict. **Concerto and Utonia: CC-BY-NC-4.0** (HF) |
| facebookresearch/sonata | Last commit 2025-06-04 | Apache-2.0 | CC-BY-NC 4.0 |
| OpenPCDet (LiDAR 3D detection) | Last release v0.5.2 (2022-01-05); last commit 2025-10-08 | Apache-2.0 | Model zoo |
| MinkowskiEngine | Last release 0.5.4 (2021-05-21); last commit 2023-03-22 | MIT | n/a; **unmaintained**, build against current CUDA and torch not verified |
| spconv | 2.3.8 (2024-12-15) | Apache-2.0 | n/a |
| torchsparse | v2.0.0 (2023-06-18); last commit 2025-02-24; not on PyPI | MIT | n/a |
| Open3D | 0.20.0 (2026-09-16) | MIT | n/a |

Guidance: for segmentation or classification backbones, use **PTv3 in Pointcept** (MIT, from scratch or with your own pretraining). Sonata, Concerto and Utonia pretrained weights are NC. Avoid MinkowskiEngine for new work. For LiDAR detection, OpenPCDet is still the reference codebase but is stale. Metrics: mIoU (dataset-level) for segmentation; mAP/NDS (nuScenes) or 3D AP (KITTI, 40 recall points) for detection, with each benchmark's official devkit (not re-verified here).
