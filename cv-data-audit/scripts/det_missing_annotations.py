"""Find likely MISSING or WRONG-CLASS box annotations from out-of-fold detector predictions.

Usage:
    python det_missing_annotations.py --gt annotations.json --dets oof_dets_fold*.json [--images IMAGE_DIR]
        [--score 0.5] [--iou 0.3] [--sheet 24] [--out missing_candidates.csv]

--dets are COCO-format detections on images the detector did NOT train on: the val_dets.json of every
cross-validation fold (cv-training/scripts/train_detector.py --fold k), which together cover the training
pool out of fold. Detections are matched to the ground truth by file name, so the files may come from runs
with different image ids.

A confident prediction (score >= --score) is flagged as
  missing   : it overlaps NO annotated box of any class (IoU < --iou): an object may be unlabelled
  class     : it overlaps an annotated box of ANOTHER class (IoU >= 0.5) but none of its own class:
              a possible class error (or a model confusion)
Both are CANDIDATES. Look at the contact sheet (--images, --sheet): red = flagged prediction, green =
annotations. Generated labels are not exempt: synthetic or auto-labelled data can drop objects too.
Weak models flag noise; run it with the best available detector and treat the top of the list first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def iou_xywh(a, b) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax1, bx1) - max(a[0], b[0])), max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dets", nargs="+", required=True, help="COCO detections; each file may carry its own image-id space")
    ap.add_argument("--dets-gt", nargs="*", default=[], help="COCO json whose image ids the matching --dets file uses "
                                                           "(same order); omit when --dets already use --gt ids")
    ap.add_argument("--images", default="")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--sheet", type=int, default=24)
    ap.add_argument("--out", default="missing_candidates.csv")
    a = ap.parse_args()

    gt = json.loads(Path(a.gt).read_text())
    cats = {c["id"]: c["name"] for c in gt["categories"]}
    name_of = {im["id"]: im["file_name"] for im in gt["images"]}
    boxes = {}
    for x in gt["annotations"]:
        if x["bbox"][2] > 0 and x["bbox"][3] > 0:
            boxes.setdefault(name_of.get(x["image_id"]), []).append((x["category_id"], x["bbox"]))

    rows, covered = [], set()
    for k, dpath in enumerate(a.dets):
        id2file = name_of
        if k < len(a.dets_gt):
            id2file = {im["id"]: im["file_name"] for im in json.loads(Path(a.dets_gt[k]).read_text())["images"]}
        dets = json.loads(Path(dpath).read_text())
        covered |= {id2file.get(d["image_id"]) for d in dets}
        for d in dets:
            if d["score"] < a.score:
                continue
            f = id2file.get(d["image_id"])
            if f is None:
                continue
            gts = boxes.get(f, [])
            same = max((iou_xywh(d["bbox"], b) for c, b in gts if c == d["category_id"]), default=0.0)
            other = max((iou_xywh(d["bbox"], b) for c, b in gts if c != d["category_id"]), default=0.0)
            anyb = max(same, other)
            if anyb < a.iou:
                kind = "missing"
            elif same < 0.5 and other >= 0.5:
                kind = "class"
            else:
                continue
            rows.append({"file": f, "kind": kind, "category": cats.get(d["category_id"], d["category_id"]),
                         "score": round(d["score"], 4), "bbox": [round(v, 1) for v in d["bbox"]],
                         "best_iou_same_class": round(same, 3), "best_iou_other_class": round(other, 3)})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"no candidates at score >= {a.score} over {len(covered)} images")
        df.to_csv(a.out, index=False)
        return
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.to_csv(a.out, index=False)
    n_img = len(covered)
    print(f"{len(df)} candidates over {n_img} images with out-of-fold predictions (score >= {a.score}):")
    print(df.groupby(["kind", "category"]).size().to_string())
    print(f"images with at least one 'missing' candidate: {df[df.kind == 'missing'].file.nunique()} of {n_img}")
    print(df.head(10).to_string(index=False))
    if a.images and a.sheet:
        from PIL import Image, ImageDraw
        tile, top = 192, df.head(a.sheet)
        cols = 6
        sheet = Image.new("RGB", (cols * tile, ((len(top) + cols - 1) // cols) * tile), "white")
        for i, r in enumerate(top.itertuples()):
            try:
                im = Image.open(Path(a.images) / r.file).convert("RGB")
            except OSError:
                continue
            d = ImageDraw.Draw(im)
            for _, b in boxes.get(r.file, []):
                d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], outline=(0, 200, 0), width=2)
            b = r.bbox
            d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], outline=(255, 0, 0), width=3)
            s = tile / max(im.size)
            sheet.paste(im.resize((max(1, int(im.width * s)), max(1, int(im.height * s)))), ((i % cols) * tile, (i // cols) * tile))
        sp = Path(a.out).with_suffix(".png")
        sheet.save(sp)
        print(f"contact sheet: {sp} (red = flagged prediction, green = annotations). Look before relabelling.")


if __name__ == "__main__":
    main()
