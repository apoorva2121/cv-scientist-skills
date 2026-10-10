"""Render augmented samples (with boxes) and measure what a policy destroys, before training on it.

Usage:
  classification:  python aug_preview.py IMAGE_DIR --list metadata.csv --col file --policy trivialaugment --size 224
  detection:       python aug_preview.py IMAGE_DIR --coco annotations.json --policy det_ssd --size 640
  options:         [--n 8] [--k 4] [--seed 0] [--out aug_preview.png] [--custom my_policy.py]

Policies (torchvision.transforms.v2, BSD):
  classification: light, randaugment, trivialaugment, augmix, rrc_strong (RandomResizedCrop scale 0.08-1)
  detection:      det_none, det_flip, det_ssd (photometric distort + zoom-out + IoU crop + flip),
                  det_safe (det_ssd without channel permutation or hue, zoom-out capped at 2x),
                  det_lsj (large-scale jitter 0.1-2.0 + fixed-size crop + flip)
--custom FILE may define POLICY(size) -> v2.Compose for anything else.

Output: a grid (n images x k augmented copies) with boxes drawn after augmentation. For detection also:
the share of boxes kept, per class (rare classes must not lose more than the others), the share of
kept boxes under 8 px at the model input size (small objects destroyed by zoom-out or downscaling), and
samples left empty.
Look at the grid: flipped text, cropped-away objects, unreadable tiny objects and misaligned boxes are
obvious in a picture and invisible in a metric.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cv-data-audit" / "scripts"))


def policy(name: str, size: int):
    import torch
    import torchvision.transforms.v2 as T
    to = [T.ToImage(), T.ToDtype(torch.uint8, scale=True)]
    cls = {
        "light": [T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True), T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2)],
        "randaugment": [T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True), T.RandomHorizontalFlip(), T.RandAugment()],
        "trivialaugment": [T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True), T.RandomHorizontalFlip(), T.TrivialAugmentWide()],
        "augmix": [T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True), T.RandomHorizontalFlip(), T.AugMix()],
        "rrc_strong": [T.RandomResizedCrop(size, scale=(0.08, 1.0), antialias=True), T.RandomHorizontalFlip()],
    }
    # Detection policies are shared with cv-training/scripts/train_detector.py (what you preview is what you
    # train). They do not resize: the detector resizes so the longer side fits --size. det_lsj ends at size x size.
    det = {
        "det_none": [],
        "det_flip": [T.RandomHorizontalFlip()],
        "det_ssd": [T.RandomPhotometricDistort(p=0.5), T.RandomZoomOut(fill=0), T.RandomIoUCrop(), T.RandomHorizontalFlip()],
        # det_ssd without channel permutation or hue change, and zoom-out capped at 2x (small objects)
        "det_safe": [T.RandomApply([T.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5)], p=0.5),
                     T.RandomZoomOut(fill=0, side_range=(1.0, 2.0)), T.RandomIoUCrop(), T.RandomHorizontalFlip()],
        "det_lsj": [T.ScaleJitter(target_size=(size, size), scale_range=(0.1, 2.0), antialias=True),
                    T.RandomCrop((size, size), pad_if_needed=True), T.RandomHorizontalFlip()],
    }
    if name in cls:
        return T.Compose(to + cls[name]), False
    if name in det:
        return T.Compose(to + det[name] + [T.ClampBoundingBoxes(), T.SanitizeBoundingBoxes(min_size=1)]), True
    raise ValueError(f"unknown policy {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--list", default="")
    ap.add_argument("--col", default="file")
    ap.add_argument("--coco", default="")
    ap.add_argument("--policy", default="trivialaugment")
    ap.add_argument("--custom", default="")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="aug_preview.png")
    a = ap.parse_args()

    import pandas as pd
    import torch
    from PIL import Image, ImageDraw
    from torchvision import tv_tensors

    from embed_images import load_rgb

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    if a.custom:
        s = importlib.util.spec_from_file_location("custom_policy", a.custom)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        tf, is_det = m.POLICY(a.size), bool(a.coco)
    else:
        tf, is_det = policy(a.policy, a.size)

    root = Path(a.image_dir)
    if is_det:
        coco = json.loads(Path(a.coco).read_text())
        anns = {}
        for x in coco["annotations"]:
            if x["bbox"][2] > 0 and x["bbox"][3] > 0:
                anns.setdefault(x["image_id"], []).append(x)
        imgs = [im for im in coco["images"] if im["id"] in anns and (root / im["file_name"]).exists()]
        pick = [imgs[i] for i in rng.choice(len(imgs), size=min(a.n, len(imgs)), replace=False)]
    else:
        names = list(pd.read_csv(a.list)[a.col]) if a.list else sorted(p.name for p in root.iterdir() if p.is_file() and not p.name.startswith("._"))
        pick = [names[i] for i in rng.choice(len(names), size=min(a.n, len(names)), replace=False)]

    tile = 160
    grid = Image.new("RGB", ((a.k + 1) * tile, len(pick) * tile), "white")
    before = after = empty = total = tiny = 0
    cls_before, cls_after = {}, {}
    names = {c["id"]: c["name"] for c in coco.get("categories", [])} if is_det else {}
    for r, item in enumerate(pick):
        try:
            img = load_rgb(root / (item["file_name"] if is_det else item))
        except Exception:  # noqa: BLE001
            continue
        grid.paste(img.resize((tile, tile)), (0, r * tile))
        for c in range(a.k):
            if is_det:
                boxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in (x["bbox"] for x in anns[item["id"]])]
                bb = tv_tensors.BoundingBoxes(torch.tensor(boxes, dtype=torch.float32), format="XYXY", canvas_size=(img.height, img.width))
                labels = torch.tensor([x["category_id"] for x in anns[item["id"]]])
                out_img, out = tf(img, {"boxes": bb, "labels": labels})
                before += len(boxes)
                after += len(out["boxes"])
                for c_ in labels.tolist():
                    cls_before[c_] = cls_before.get(c_, 0) + 1
                for c_ in out["labels"].tolist():
                    cls_after[c_] = cls_after.get(c_, 0) + 1
                if len(out["boxes"]):
                    # object size at the model input: the detector scales the longer side to --size
                    scale = a.size / max(out_img.shape[-2:])
                    wh = (out["boxes"][:, 2:] - out["boxes"][:, :2]) * scale
                    tiny += int((wh.min(dim=1).values < 8).sum())
                empty += len(out["boxes"]) == 0
                total += 1
                pil = Image.fromarray(out_img.permute(1, 2, 0).numpy())
                d = ImageDraw.Draw(pil)
                for b in out["boxes"].tolist():
                    d.rectangle(b, outline=(255, 0, 0), width=2)
            else:
                out_img = tf(img)
                pil = Image.fromarray(out_img.permute(1, 2, 0).numpy())
            grid.paste(pil.resize((tile, tile)), ((c + 1) * tile, r * tile))
    grid.save(a.out)
    print(f"preview: {len(pick)} images x {a.k} augmentations (first column = original) -> {a.out}")
    if is_det and total:
        print(f"boxes kept after augmentation: {after}/{before} ({after / max(before, 1):.1%}); "
              f"samples left with no boxes: {empty}/{total} ({empty / total:.1%})")
        print(f"kept boxes smaller than 8 px at the model input size ({a.size}): {tiny}/{max(after, 1)} ({tiny / max(after, 1):.1%})")
        print("kept rate per class (a rare class losing more than the others is a problem):")
        for c_ in sorted(cls_before, key=lambda k: cls_before[k]):
            print(f"  {names.get(c_, c_)!s:<20} {cls_after.get(c_, 0)}/{cls_before[c_]} ({cls_after.get(c_, 0) / cls_before[c_]:.1%})")
        if after / max(before, 1) < 0.8:
            print("NOTE: over 20% of boxes are gone. Crop policies remove boxes outside the crop by design; "
                  "act if the rare classes lose more than the others, or if many kept boxes are tiny")


if __name__ == "__main__":
    main()
