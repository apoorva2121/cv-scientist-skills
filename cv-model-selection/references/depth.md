# 4. Depth and 3D geometry: models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

| Model | What it is | Best verified number / claim | Code licence | Weights licence (per size) | Latest | Install route | CPU feasibility | Fine-tuning |
|---|---|---|---|---|---|---|---|---|
| **Depth Anything V2** S/B/L/(G) | Relative monocular depth trained on synthetic labels plus pseudo-labelled real images (2406.09414); metric fine-tunes available | "More than 10x faster and more accurate" than Stable-Diffusion-based depth (abstract) | Apache-2.0 | **Small (24.8M): Apache-2.0. Base (97.5M), Large (335.3M), Giant (1.3B, "coming soon"): CC-BY-NC-4.0** (README; HF tags agree) | Commit 2026-03-24 | transformers `DepthAnythingForDepthEstimation` (`depth-anything/Depth-Anything-V2-*-hf`) | **Small: yes** | Metric fine-tuning scripts in repo |
| **Depth Anything 3** (DA3) | Any-view geometry (depth plus ray maps) from a plain DINO transformer (2511.10647) | "Surpassing prior SOTA VGGT by an average of 44.3% in camera pose accuracy and 25.1% in geometric accuracy", and beats DA2 on monocular depth (abstract) | Apache-2.0 | README table: **DA3-SMALL (0.08B) and DA3-BASE (0.12B) Apache-2.0; DA3METRIC-LARGE and DA3MONO-LARGE (0.35B) Apache-2.0; DA3-LARGE(-1.1) (0.35B), DA3-GIANT(-1.1) (1.15B) and DA3NESTED-GIANT-LARGE(-1.1) (1.40B) CC BY-NC 4.0.** HF tags `DA3-LARGE-1.1` as Apache-2.0, which conflicts with the README | HF 2025-11-13; -1.1 checkpoints 2025-12-11 ("retrained after fixing a training bug; prefer these") | `depth-anything-3` repo (no transformers class found) | Small/Base feasible | Repo |
| Video Depth Anything | Temporally consistent video depth | Not verified | Apache-2.0 | Not verified | Release 1.3.1 (2025-09-15) | Repo | Small size exists (not verified) | Repo |
| **Depth Pro** | Zero-shot metric depth with sharp boundaries (2410.02073) | "2.25-megapixel depth map in 0.3 seconds on a standard GPU" (abstract) | Apple sample-code licence (permissive text) | **Conflict.** HF metadata says `apple-amlr` (research-only; changed 2025-02-28 per R3), the HF card body says "Apple-ASCL", and the GitHub README says weights are under the repo LICENSE. Treat as research-only until cleared | Repo commit 2026-09-11 | transformers `DepthProForDepthEstimation` | No (ViT-L multi-scale) | Limited |
| Metric3D v2 | Metric depth plus normals via canonical camera space (2404.15506) | Qualitative (abstract) | BSD-2-Clause | Weights licence not stated (README); ONNX exports on HF `onnx-community/metric3d-*` | Commit 2025-03-13 | `torch.hub`, ONNX | ViT-S ONNX | Limited |
| UniDepth V1 / V2 | Universal metric depth with a self-promptable camera module (2403.18913, 2502.20110) | "Superior performance and generalization" on ten datasets zero-shot (abstract) | **CC BY-NC 4.0** | HF `lpiccinelli/unidepth-v2-*` untagged; treat as NC | Commit 2025-05-18 | Repo | ViT-S variant | NC |
| **MoGe / MoGe-2 / MoGe-3** | Monocular geometry (point maps); MoGe-2 adds metric scale (2507.02546); MoGe-3 adds sparse 3D refinement for fine detail (2607.17967, 2026-07-20) | MoGe-3 "significantly outperforms existing approaches in recovering fine detailed 3D geometry" (abstract) | MIT (DINOv2 modules Apache-2.0) | **MIT** (HF `Ruicheng/moge-*`, including `moge-3-vitl` and `moge-3-vitg`, 2026-08-18) | Commit 2026-08-19 ("V3") | Repo `microsoft/MoGe` | **MoGe-2 ViT-S: 35M params** (README); MoGe-3: 370M / 1.25B; MoGe-3 needs FlexGEMM/Triton (no macOS) | Repo |
| **VGGT** | Feed-forward multi-view geometry: cameras, depth, point maps, tracks (2503.11651) | Commercial checkpoint AUC@30 90.37 (README) | VGGT License v1 (custom; commercial allowed, military excluded) | **`VGGT-1B`: CC-BY-NC-4.0. `VGGT-1B-Commercial`: VGGT licence, gated (manual)**, "only this model checkpoint allows commercial usage" (README). `VGGT-Omega` (2026-03-17): FAIR non-commercial, gated | Commit 2026-05-19 | Repo | No (1B) | Repo |
| **MapAnything** | Universal feed-forward metric 3D reconstruction (2509.13414) | "Outperforms or matches specialist feed-forward models" (abstract) | Apache-2.0 | **`facebook/map-anything`: CC-BY-NC-4.0. `facebook/map-anything-apache`: Apache-2.0** | v1.1.3 (2026-07-17) | Repo | No | Repo |
| Pi3 / Pi3X | Permutation-equivariant multi-view geometry (2507.13347) | Qualitative | BSD-2-style | **Pi3: BSD-2-Clause. Pi3X: CC-BY-NC-4.0** (HF) | Commit 2026-05-18 | Repo | No | Repo |
| DUSt3R / MASt3R | Pairwise 3D reconstruction and matching (2312.14132, 2406.09756) | Qualitative | **CC BY-NC-SA 4.0** | HF untagged; the repo licence (NC-SA) governs | Commits 2025-06/07 | Repo | No | NC |
| **Marigold** | Diffusion-based depth, normals and intrinsics (2312.02145) | Qualitative | Apache-2.0 | **depth v1-1: OpenRAIL++. depth v1-0: Apache-2.0** (HF) | Commit 2026-09-06 | **diffusers 0.40.0** Marigold pipelines | Slow (diffusion; LCM variant exists) | Repo |
| Prompt Depth Anything, ZoeDepth, DPT | Other transformers depth heads | n/a | Apache-2.0 (transformers) | Per checkpoint | transformers 5.17 | `PromptDepthAnythingForDepthEstimation`, `ZoeDepthForDepthEstimation`, `DPTForDepthEstimation` | Varies | Yes |

### Decision guidance: depth
- **Relative depth on CPU, commercial:** Depth Anything V2-Small (Apache, 24.8M) or MoGe-2 ViT-S (MIT, 35M).
- **Metric depth, commercial:** DA3METRIC-LARGE (Apache per README), MoGe-2/3 (MIT) or UniDepth (NC only). Depth Pro is licence-unclear.
- **Multi-view or SfM-like:** DA3-BASE/SMALL (Apache), MapAnything-apache, Pi3 (BSD-2) or VGGT-1B-Commercial (custom).
- **Avoid in products:** DA V2 B/L/G, DA3 L/G, VGGT-1B, MapAnything (non-apache), Pi3X, UniDepth and DUSt3R/MASt3R (all NC).
- **Metrics:**
  - AbsRel, δ1 (δ < 1.25) and RMSE. Relative (affine-invariant) predictions must be aligned per image by least-squares scale and shift before scoring; metric models are scored without alignment. Never compare the two regimes in one table.
  - Boundary sharpness needs dedicated metrics (Depth Pro proposes some in its abstract).
  - No torchmetrics depth metric exists; write the formulae explicitly and state the alignment and valid-pixel mask.
