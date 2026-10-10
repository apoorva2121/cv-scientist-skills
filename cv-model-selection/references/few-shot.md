# 13. Small-data and few-shot: when zero-shot or few-shot beats fine-tuning: models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

| Regime | Evidence (verified) | Recommendation |
|---|---|---|
| Classification with a few labels per class | A linear classifier on a good embedding "outperforms state-of-the-art few-shot learning methods" (2003.11539). LP++ is a "surprisingly strong linear probe" for few-shot CLIP (2404.02285). Tip-Adapter is training-free (2111.03930). A Closer Look at Few-shot Classification (1904.04232) finds simple baselines competitive. R3 adds that CLIP prompt-learning needs per-task tuning to beat a linear probe (2312.12730) | **Frozen DINOv2 or SigLIP 2 features, then logistic regression or kNN, first.** Fine-tune (LP-FT, see R3) only once there are enough labels to beat it on a validation split |
| Detection with 0-10 shots, common categories | Zero-shot GroundingDINO 48 AP vs 33 AP for few-shot SOTA on COCO (2312.14494) | Open-vocabulary detector (OWLv2, LLMDet, Grounding DINO, SAM 3) with prompt tuning, or visual exemplars |
| Detection with 0-10 shots, unusual domains | RF100-VL: zero-shot VLM detectors under 2% on hard medical sets; "the need for few-shot concept alignment". The CVPR 2025 FSOD challenge winner beat the baseline by 17 mAP (2505.20612) | Pre-label with open-vocabulary models, correct by hand, fine-tune RF-DETR or D-FINE. Frustratingly-simple fine-tuning (TFA, 2003.06957) and DE-ViT (2309.12969) are few-shot baselines |
| Anomaly detection with 1-4 normal images | AnomalyDINO one-shot MVTec AUROC 96.6% (2405.14529) | Training-free patch kNN on DINOv2 |
| Promptable segmentation | SAM zero-shot in 2D medical imaging is "comparable to, but in certain cases, surpasses the current state-of-the-art" on six datasets (2305.00109) | SAM 2.1 or MedSAM for interactive labelling; nnU-Net once around 20+ labelled volumes exist (threshold not verified; decide on a validation split) |
| Retrieval with no labels | Global-feature reranking gives large gains at low cost (SuperGlobal, 2308.06954) | Frozen encoder plus reranking before metric learning |
