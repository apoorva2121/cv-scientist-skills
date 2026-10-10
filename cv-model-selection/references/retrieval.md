# 8. Retrieval, re-identification and metric learning: models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

| Component | What it is | Verified facts | Code licence | Weights licence | Latest | Install | CPU feasibility |
|---|---|---|---|---|---|---|---|
| **FAISS** (2401.08281) | Vector index and search (flat, IVF, HNSW, PQ) | n/a | MIT | n/a | faiss-cpu 1.15.1 (2026-09-16); faiss-gpu 1.15.1 on PyPI (2026-09-16; not verified as official); `faiss-gpu-cu12` 1.14.1.post1 (community wheels) | pip / conda | **Yes** |
| **Embedding backbones** | CLIP/SigLIP 2/DINOv2/DINOv3/PE: see R3 | See R3 | | See R3 (SigLIP 2 and DINOv2 are Apache-2.0; DINOv3 is custom) | | timm / open_clip / transformers | ViT-S/B on CPU |
| Qwen3-VL-Embedding 2B/8B | Multimodal (image and text) embedding model | n/a | Apache-2.0 | Apache-2.0 (HF, 2026-01-07) | | sentence-transformers 6.1.0 | 2B is marginal on CPU |
| **pytorch-metric-learning** | Losses (ArcFace, triplet, contrastive...), miners, samplers, AccuracyCalculator | "A Metric Learning Reality Check" (2003.08505, same author): "actual improvements over time have been marginal at best" | MIT | n/a | **2.9.0 (2025-08-17)**; last commit 2025-08-17 | pip | Yes |
| **SuperGlobal** (2308.06954) | Improved GeM pooling plus cheap global-feature reranking for landmark retrieval | Revisited Oxford+1M Hard: single-stage +7.1%, two-stage +3.7% "with a strong 64,865x speedup" (abstract) | Not verified | Not verified | n/a | Repo | Yes |
| revisitop (ROxford / RParis evaluation) | Official evaluation code for the revisited benchmarks (1803.11285) | `compute_map(ranks, gnd, kappas)` handles junk images, as declared in the ground truth | Not verified | n/a | n/a | Copy `evaluate.py` | Yes |
| **FastReID** (2006.02631) | Re-ID toolbox | n/a | Apache-2.0 | Model zoo | **V1.3.0 (2021-05-31)**; last commit 2024-07-15 | Repo | Stale |
| torchreid (deep-person-reid) | Re-ID library (OSNet etc.) | Market-1501 evaluation drops gallery images with the same person ID and camera as the query (`rank.py`, verified) | MIT | Model zoo | PyPI `torchreid` 0.2.5 (2022-10-16); repo commit 2026-01-09 (README) | pip | OSNet on CPU |
| CLIP-ReID (2211.13977) | CLIP image encoder fine-tuned with learned ID text tokens | "Simply fine-tuning the visual model initialized by the image encoder in CLIP, has already obtained competitive performances" (abstract) | MIT | CLIP (OpenAI) weights | Last commit 2023-11-21 | Repo | ViT-B |
| BoxMOT re-ID | OSNet, CLIP-ReID and other re-ID heads for trackers | n/a | **AGPL-3.0** | n/a | 25.0.0 (2026-09-09) | pip | Yes |
| InsightFace | Face recognition | n/a | MIT | **NC** (see Section 3) | 2.0 (2026-09-08) | pip | Yes |

### Decision guidance: retrieval and re-ID
- **Instance or landmark retrieval:**
  1. Use frozen **DINOv2** (Apache) or SigLIP 2 global embeddings with FAISS (see R3 for encoder choice).
  2. Add **query-expansion or reranking** (SuperGlobal-style) before any fine-tuning.
  3. Fine-tune with pytorch-metric-learning only when frozen features plateau.
  4. Follow the Reality Check's protocol: same embedding size, same batch sampler, tuned on a validation split, with no test-set early stopping.
- **Person or vehicle re-ID:** the maintained toolboxes are stale (FastReID 2021, torchreid 2022 on PyPI). The practical route is a CLIP- or DINOv2-initialised ViT with ArcFace or triplet loss from pytorch-metric-learning, evaluated with torchreid's Market-1501 protocol code.
- **Licence traps:** InsightFace models are NC. BoxMOT is AGPL. Re-ID benchmark datasets (Market-1501 and others) carry their own terms, which were not checked here.

### Metrics: retrieval and re-ID
- **Retrieval:**
  - mAP@k and recall@k (torchmetrics 1.9.0 `RetrievalMAP`, `RetrievalRecall`, `RetrievalPrecision`, `RetrievalNormalizedDCG`, all with `top_k`).
  - ROxford/RParis mAP on the **Medium and Hard** protocols with junk handling (revisitop).
  - GLDv2 (2004.01804) uses mAP@100.
- **Re-ID:**
  - CMC rank-1/5/10 plus mAP, with **same-ID-same-camera gallery removal**. Omitting this removal inflates rank-1.
  - Report whether re-ranking (k-reciprocal) was used.
- **Pitfall:** torchmetrics retrieval metrics group scores by `indexes` (query ID). Forgetting to pass per-query indexes silently computes one global ranking.
