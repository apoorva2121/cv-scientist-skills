# 7. Anomaly detection (industrial): models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

| Model / framework | What it is | Best verified number | Code licence | Weights licence | Latest release + date | Install route | CPU feasibility | Training data needed |
|---|---|---|---|---|---|---|---|---|
| **anomalib** | Library with 30+ models: Patchcore, EfficientAd, Dinomaly, AnomalyDINO, UniNet, SuperADD, Supersimplenet, Glass, InpFormer, WinClip, VlmAd, AnomalyVFM, Fastflow, Padim, Cflow, Draem, ReverseDistillation and more (`models/__init__.py`) | n/a | Apache-2.0 | Backbone licences apply (for example DINOv2 is Apache-2.0) | **2.6.2 (2026-09-11)**; active (commit 2026-09-23) | `pip install anomalib` | OpenVINO export (Intel-backed) | Normal images only (plus a few anomalous for validation) |
| **PatchCore** (2106.08265) | kNN over a coreset memory bank of ImageNet CNN patch features | "Image-level anomaly detection AUROC score of up to 99.6%" on MVTec AD (abstract) | Apache-2.0 (amazon-science; last commit 2023-03-06) | ImageNet backbone (torchvision) | Stale upstream; use anomalib | anomalib `Patchcore` | **Yes** (no training; memory bank grows with data) | Normal only |
| **EfficientAD** (2303.14535) | Student-teacher plus autoencoder for logical anomalies | "Latency of two milliseconds and a throughput of six hundred images per second", evaluated on 32 datasets (abstract) | Apache-2.0 (unofficial `nelson1425/EfficientAD`, last commit 2023-05-28) | n/a | anomalib | anomalib `EfficientAd` | Yes | Normal only |
| **Dinomaly** (2405.14325) | Multi-class reconstruction with a DINOv2 ViT | Image AUROC 99.6% MVTec AD, 98.7% VisA, 89.3% Real-IAD in the **multi-class** setting (abstract) | Apache-2.0 (active, 2026-09-17) | DINOv2 backbone (Apache-2.0) | Repo / anomalib | anomalib `Dinomaly` | GPU for training | Normal only, one model for all classes |
| **AnomalyDINO** (2405.14529) | Training-free patch kNN on DINOv2 | One-shot MVTec AD AUROC raised "from 93.1% to 96.6%" (abstract) | n/a | DINOv2 | anomalib | anomalib `AnomalyDINO` | Yes | **1 to few normal shots** |
| FastFlow (2111.07677) | 2D normalising flows | Not quoted | via anomalib | n/a | anomalib | anomalib `Fastflow` | Moderate | Normal only |
| UniAD (2206.03687) | Unified multi-class reconstruction transformer | Unified MVTec AD: detection 96.5%, localisation 96.8% (abstract) | Apache-2.0 (last commit 2022-11-22) | n/a | Stale; not in anomalib | Repo | GPU | Normal only |

**Benchmarks and their licences**

| Dataset | Licence (verified source) | Notes |
|---|---|---|
| MVTec AD | **CC BY-NC-SA 4.0** (anomalib `mvtecad.py` docstring; also R3) | Near-saturated: 99.6% image AUROC (PatchCore; Dinomaly multi-class) |
| **MVTec AD 2** (2503.21622) | **CC BY-NC-SA 4.0; "commercial use is explicitly prohibited"** (MVTec page) | Eight scenarios, 8000+ high-resolution images. SOTA "remains below 60% average AU-PRO" (abstract). The private test split is scored only through the public evaluation server |
| VisA (2207.14315) | **Conflict: the upstream `amazon-science/spot-diff` README says "CC BY 4.0"; anomalib's `visa.py` docstring says CC BY-NC-SA 4.0** | Follow upstream but record the conflict |

### Decision guidance: anomaly detection
- **Few normal images (1 to 16):** AnomalyDINO (training-free).
- **Standard single-class setting, CPU deployment:** PatchCore or EfficientAD via anomalib, exported to OpenVINO.
- **Many product classes, one model:** Dinomaly.
- **Validation:** validate on your own line data. MVTec AD is saturated, and MVTec AD 2 shows methods fall below 60% AU-PRO in harder scenarios (transparent, overlapping or tiny defects; lighting shifts).
- **Licence trap:** MVTec AD, MVTec AD 2 and probably Real-IAD (not verified) are NC. **Do not train or tune a commercial model on them.** Use them for method selection only, then refit on your own data.

### Metrics: anomaly detection
- **Metrics in anomalib 2.6.2:** AUROC, AUPR, AUPRO, PRO, F1Max, F1AdaptiveThreshold, ManualThreshold, PIMO / AUPIMO, PGn/PBn.
- **Pitfalls:**
  1. Image AUROC is threshold-free and saturates near 99%. Report the operating point (F1Max, or recall at a fixed false-positive rate) that the line will actually run at.
  2. Pixel AUROC is dominated by the huge number of normal pixels. Prefer AUPRO, which weights each defect region equally.
  3. **anomalib's `AUPRO` integrates only up to `fpr_limit=0.3` by default** (source), and its docs state that raising the limit raises the value. Always state the limit.
  4. Normalisation and thresholds must be fitted on validation data, not test data.
  5. Multi-class ("unified") and one-model-per-class numbers are different protocols.
