"""Test-time augmentation for detectors trained with cv-training/scripts/train_detector.py: one detection file
per view, ready for fusion and a paired comparison with the single pass. No retraining: only extra inference.

Usage:
    python det_tta.py --model runs/final/model.pt --image-dir IMAGE_DIR --coco annotations.json \
        --images val_ids.txt [--views orig,hflip,scale1.5] [--threads 4] [--out tta/]
    python postprocess_det.py fuse --gt annotations.json --images val_ids.txt \
        --dets tta/dets_orig.json tta/dets_hflip.json tta/dets_scale1.5.json --methods wbf

Views
  orig        the single pass (the reference every TTA result must beat)
  hflip       horizontally flipped input; boxes mapped back to the original frame. Only when a flip does not
              change the label (not for text, laterality or left/right classes)
  scaleS      the detector's input size multiplied by S (e.g. scale1.5, scale0.75): helps small objects when
              S > 1, at S^2 the compute
Detections are written in COCO format with the image ids of --coco, restricted to --images (ids or file names).
TTA is a candidate, not a default: keep it only if the fused result beats the single pass on the primary metric
with an interval that excludes zero (cv-evaluation/scripts/metrics_ci.py det --dets fused --dets-b orig) and the
extra inference time fits the latency budget.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "cv-data-audit" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "cv-training" / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images", default="", help="file with image ids or file names (default: all images in --coco)")
    ap.add_argument("--views", default="orig,hflip")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="tta")
    a = ap.parse_args()

    import torch
    import torchvision.transforms.functional as TF

    from embed_images import load_rgb
    from train_detector import build_model

    if a.threads:
        torch.set_num_threads(a.threads)
    ck = torch.load(a.model, map_location="cpu", weights_only=True)
    cfg, cats = ck["config"], ck["categories"]
    model = build_model(cfg["model"], len(cats) + 1, cfg["size"], False, cfg.get("trainable_layers", 3))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    lab2cat = {i + 1: c for i, c in enumerate(cats)}
    coco = json.loads(Path(a.coco).read_text())
    ims = coco["images"]
    if a.images:
        keep = set(Path(a.images).read_text().split())
        ims = [im for im in ims if str(im["id"]) in keep or im["file_name"] in keep]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = cfg["size"]
    for view in [v.strip() for v in a.views.split(",") if v.strip()]:
        size = base
        if view.startswith("scale"):
            size = int(round(base * float(view[len("scale"):])))
        elif view not in ("orig", "hflip"):
            raise SystemExit(f"unknown view {view}")
        model.transform.min_size, model.transform.max_size = (size,), size
        dets, t0 = [], time.time()
        with torch.inference_mode():
            for im in ims:
                try:
                    img = load_rgb(Path(a.image_dir) / im["file_name"])
                except Exception:  # noqa: BLE001
                    continue
                x = TF.to_tensor(img)
                if view == "hflip":
                    x = torch.flip(x, dims=[2])
                o = model([x])[0]
                W = x.shape[-1]
                for b, l, s in zip(o["boxes"].tolist(), o["labels"].tolist(), o["scores"].tolist()):
                    x0, y0, x1, y1 = b
                    if view == "hflip":
                        x0, x1 = W - x1, W - x0
                    dets.append({"image_id": im["id"], "category_id": lab2cat[int(l)],
                                 "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(s)})
        name = out / f"dets_{view}.json"
        name.write_text(json.dumps(dets))
        print(f"{view:10s} input {size:4d} px: {len(dets)} detections on {len(ims)} images, "
              f"{(time.time() - t0) / max(len(ims), 1) * 1000:.0f} ms/image -> {name}")
    model.transform.min_size, model.transform.max_size = (base,), base
    print("next: python postprocess_det.py fuse --gt <coco> --images <ids> --dets " +
          " ".join(str(out / f"dets_{v.strip()}.json") for v in a.views.split(",")) + " --methods wbf")


if __name__ == "__main__":
    main()
