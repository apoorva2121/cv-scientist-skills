# 11. 3D medical imaging beyond segmentation: models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

| Component | Task | Verified facts | Code licence | Weights licence | Latest | Install |
|---|---|---|---|---|---|---|
| **MONAI** | 3D classification networks (DenseNet, ResNet, SEResNet, EfficientNet-BN, ViT, SwinUNETR encoder), transforms, metrics, bundles | n/a | Apache-2.0 | Per bundle (VISTA3D weights NCLS v1) | **1.6.0 (2026-06-22)**; 1.6.1rc0 tag 2026-09-18 | `pip install monai` |
| MedicalNet (Med3D, 1904.00625) | 3D ResNet weights pretrained across segmentation datasets | n/a | MIT (Tencent) | MIT | Last substantive work 2019; commit 2025-11-27 (requirements) | Repo |
| **VoxelMorph** (1809.05231) | Unsupervised deformable registration | v0.3 (2025-08-25) adds a `voxelmorph/nn` namespace | **Conflict: repo LICENSE.md (dev) Apache-2.0; PyPI classifier MIT** | You train | voxelmorph 0.3.3 (2026-07-27) | pip |
| **uniGradICON** (2403.05780) | Registration foundation model, zero-shot across 12 datasets | "Zero-shot capabilities for new registration tasks" (abstract) | **Conflict: repo Apache-2.0; PyPI classifier MIT** | Released with the repo | unigradicon 1.0.5 (2026-02-25); icon-registration 1.1.8 | pip |
| **CT-FM** (2501.09001) | CT foundation model for classification, triage and retrieval | "Superior performance against state-of-the-art models" (abstract) | MIT | **Apache-2.0** (HF `project-lighter/ct_fm_feature_extractor`, `ct_fm_segresnet`) | HF 2025-01/02 | `project-lighter` |
| **Merlin** (2406.06512) | CT vision-language model: findings classification, chronic disease prediction, report generation | Validated on "44,098 CT scans from 3 independent sites" externally (abstract) | MIT | **MIT** (HF `stanfordmimi/Merlin`, updated 2026-09-16) | Active (commit 2026-09-24) | Repo |
| RadFM (2308.02463) | Generalist 2D/3D radiology VQA and report generation | Outperforms OpenFlamingo, MedFlamingo, MedVInT and GPT-4V on RadBench (abstract) | MIT (2025) | HF `chaoyi-wu/RadFM` **untagged** | Release 2025-06-12 | Repo |
| **nnFoundation** | 3D radiology foundation models (see 2C) | 108 evaluation tasks (abstract) | Apache (nnssl) | **CC-BY-SA-4.0** | 2026-09-24 | nnU-Net |
| Curia / Curia-2 | See R3 | | | Research-only RAIL-M, gated (R3) | | |

### Decision guidance: 3D medical beyond segmentation
- **3D classification with small data:**
  1. Start with a MONAI 3D ResNet or DenseNet trained from scratch, and compare against CT-FM (Apache weights) as a frozen or fine-tuned encoder.
  2. Try nnFoundation encoders (share-alike weights) when licence-compatible.
  3. R3 cites a 10-encoder frozen CT benchmark (2608.05960) that found no universal SOTA.
- **Registration:** uniGradICON zero-shot first, then fine-tune. VoxelMorph for a learned baseline. Classical ANTs/SimpleITK (SimpleITK 2.5.6, Apache) as the reference.
- **Metrics:** Dice of warped labels, target registration error (TRE) on landmarks, and the percentage of voxels with non-positive Jacobian determinant (folding). Report all three; Dice alone rewards implausible warps.
