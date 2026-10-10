# Evaluation datasets (do not show to agents under test)

`make_datasets.py` (run with the `cv-skills` env) generates two synthetic datasets with planted
defects and writes hidden answer keys under `answer_keys/`. Prompts are in `evals.json`; grading
assertions in `cv-workspace/iteration-N/<eval>/eval_metadata.json`.

- `shapes_classification`: 6-class images from ~870 capture sessions (3-5 photos per object), with
  near-duplicates re-uploaded under new session ids, exact duplicates (some with conflicting labels),
  4% label noise, truncated files, EXIF-rotated JPEGs, grayscale / RGBA / CMYK images, PNG bytes behind
  .jpg names, 32 px and 1024 px outliers, a rare class, a watermark shortcut on one class, and a
  labelled production batch from a new camera (darker, blurred, busier, watermark decorrelated).
- `shapes_detection`: 150 sequences x 8 near-identical frames, COCO JSON with out-of-bounds and
  zero-size boxes, corner-format boxes on 20 images, duplicate boxes, missing annotations, undefined
  and unused categories, JSON images missing on disk, wrong image sizes, tiny objects, a rare class and
  12% high-resolution frames.

Verified 2026-09-27 by an independent script: every planted classification defect is present and
detectable (near-duplicate pHash distance at most 10 vs at least 18 for random pairs); detection
defects detectable in the final JSON are recorded in the answer key under `final_detectable`.
