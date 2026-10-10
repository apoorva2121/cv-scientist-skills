# 9. Vision-language models (captioning, VQA, grounding, OCR, open-vocabulary): models and frameworks

Source: research R4 (verified 2026-09-27; primary sources only, see method note below). Backbones and encoders: `backbones.md`. Consolidated licence traps and unverified items: `licence-traps.md`.

**Licence vocabulary** (same as R3, plus two terms):
- *Permissive:* Apache-2.0, MIT or BSD.
- *Copyleft:* GPL-3.0 or AGPL-3.0. Under AGPL, serving the model over a network triggers the obligation to release source.
- *Custom-commercial:* commercial use is allowed under a bespoke licence that has field-of-use bans, pass-through terms, revenue caps or gating. Needs legal review under a permissive-by-default policy.
- *NC:* non-commercial or research only.
- *Share-alike:* CC-BY-SA (derivatives must carry the same licence).
- *Source-available:* for example BUSL.

Code and weights licences are listed separately because they often differ. The weights licence is the binding one for deployment.

| Model family | Sizes (verified on HF) | Weights licence (read from card / licence file) | Latest | Install route | CPU feasibility | Grounding / detection output |
|---|---|---|---|---|---|---|
| **Qwen3.5** | 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B (HF, 2026-02) | **Apache-2.0** | 2026-02-16 to 2026-02-28 | transformers `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` | 0.8B/2B feasible (quantised) | Card: "Unified Vision-Language Foundation: early fusion training ... outperforms Qwen3-VL models across ... visual understanding" (vendor claim) |
| Qwen3.6 | 27B, 35B-A3B (HF, 2026-04) | Apache-2.0 | 2026-04-15/21 | transformers (image-text-to-text) | No | Not verified |
| **Qwen3-VL** (2511.21631) | 2B, 4B, 8B, 32B (not checked), 235B-A22B | **Apache-2.0** (2B/8B/235B checked) | 2025-09/10 | transformers `Qwen3VLForConditionalGeneration`; vLLM | 2B feasible | Boxes and points (grounding); tech report emphasises spatial-temporal grounding |
| Qwen2.5-VL (2502.13923) | 3B, 7B, 32B, 72B | **3B: Qwen Research licence ("Non-Commercial ... research or evaluation purposes only"). 7B: Apache-2.0. 72B: Qwen licence (separate licence needed above 100M MAU)** | 2025-01 | transformers `Qwen2_5_VLForConditionalGeneration` | 3B (NC) | Boxes (JSON) |
| **InternVL3 / InternVL3.5** (2504.10479, 2508.18265) | 1B-241B | InternVL3-8B card: "released under the MIT License", Qwen2.5 component "licensed under the Apache-2.0 License" (HF metadata has `license_name: qwen`, a conflict). InternVL3.5-8B: Apache-2.0 (Qwen3 component Apache-2.0) | 2025-04 / 2025-08 | transformers `InternVLForConditionalGeneration` (`-HF` repos) | 1B feasible | Boxes |
| **Molmo / Molmo2** (2409.17146, 2601.10611) | Molmo2 4B/8B (8B card: Qwen3-8B + SigLIP so400m), O-7B; MolmoPoint-8B | **Apache-2.0** | 2025-12-14 (Molmo2); MolmoPoint 2026-03-16 | transformers (remote code) | No | **Pointing** and video tracking: Molmo2 "56.2 vs 41.1 J&F on video tracking" against Gemini 3 Pro (abstract) |
| **Gemma 4** (2607.02770) | E2B, E4B, 12B, 26B-A4B, 31B | **Apache-2.0** (HF tag; the Gemma 4 licence page header reads "Apache License 2.0 / Gemma"). Gemma 3 and PaliGemma 2 remain under **Gemma terms**, gated (manual) | 2026-03 to 2026-05 | transformers `Gemma4ForConditionalGeneration` | E2B/E4B designed for on-device | Not verified |
| PaliGemma 2 (2412.03555) | 3B, 10B, 28B at 224/448/896 px | **Gemma terms, gated** | 2024-11-21 | transformers `PaliGemmaForConditionalGeneration` | 3B marginal | Detection and segmentation tokens; built for fine-tuning transfer |
| **Florence-2** | 0.23B / 0.77B | **MIT** | 2024-06 | transformers `Florence2ForConditionalGeneration` | **Yes** | Detection, dense captioning, OCR, grounding task prompts |
| **SmolVLM / SmolVLM2** (2504.05299) | 256M, 500M, 2.2B | **Apache-2.0** | 2025-01/04 | transformers `SmolVLMForConditionalGeneration` | **Yes**: "SmolVLM-256M uses less than 1GB GPU memory" (abstract) | Limited |
| Moondream 3 / 3.1 | 9B total, 2B active MoE | **BUSL-1.1 plus additional use grant** (3 preview); "moondream-model-license-1.0" (3.1) | 2025-09 / 2026-06 | HF | No | `detect()` and `point()` |
| Perception LM (1B/3B/8B) | | **FAIR non-commercial research**, gated | 2025-04 | transformers `PerceptionLMForConditionalGeneration` | 1B | n/a |

Serving and evaluation: **vLLM** 0.30.0 (2026-09-22, Apache-2.0); qwen-vl-utils 0.0.14; **lmms-eval** 0.7.3 (2026-08-28, MIT). VLMEvalKit's PyPI status was not verified.

### When zero-shot VLMs win and when fine-tuned specialists win (evidence)

| Setting | Evidence | Verdict |
|---|---|---|
| Common-category detection with no training data | GroundingDINO zero-shot scores 48 AP vs 33 AP for SOTA few-shot detectors on COCO (2312.14494). SAM 3 zero-shot reaches 56.4 COCO box AP (README) | **Zero-shot or open-vocabulary wins** at 0 to 10 shots on web-common classes |
| Out-of-distribution classes and modalities (medical, aerial, industrial) | RF100-VL: GroundingDINO and Qwen2.5-VL "achieve less than 2% zero-shot accuracy on challenging medical imaging datasets" (2505.20612). Fine-tuned RF-DETR-2XL beats GroundingDINO-tiny by 1.2 AP on RF100-VL at 20x the speed (2511.09554) | **Fine-tuned specialists win**; use VLMs only for pre-labelling |
| Closed-set image classification | VLMs "significantly underperform CLIP on standard image classification benchmarks like ImageNet"; the cause is data exposure (2405.18415) | Use a CLIP/SigLIP linear probe or a fine-tuned classifier, not a chat VLM |
| Fine-grained geometry (counting overlaps, intersections) | Four SOTA VLMs average 58.07% on BlindTest's simple geometric tasks (2407.06581) | Use specialist detectors, segmenters or measurement code |
| Document parsing | 0.9B document specialists (96.34) beat Qwen3-VL-235B (89.78) on OmniDocBench v1.6 | Specialised small VLMs win |
| Video pointing and tracking | Molmo2 surpasses Gemini 3 Pro on video pointing (38.4 vs 20.0 F1) (2601.10611) | Open grounding-trained VLMs are competitive |

### Decision guidance: VLMs
- **Permissive defaults:** Qwen3-VL (2B/4B/8B), Qwen3.5 (0.8B-9B) and Gemma 4 (E2B/E4B/12B), all Apache-2.0. Add Florence-2 (MIT) or SmolVLM2 (Apache) for CPU.
- **Grounding and pointing:** Molmo2 (Apache).
- **Avoid:** Qwen2.5-VL-3B (NC), Perception LM (NC), Moondream (BUSL) and HunyuanOCR (territory) without a licence review.
- **Measure before deploying:** always benchmark a VLM against a fine-tuned specialist on a held-out slice of your own data before choosing it for a production perception task (see the table above).
