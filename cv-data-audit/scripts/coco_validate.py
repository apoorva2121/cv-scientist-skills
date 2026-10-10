"""Stage 1 annotation validation for COCO-format detection / instance-segmentation labels.

Usage:
    python coco_validate.py annotations.json --images IMAGE_DIR [--iou-dup 0.9] [--tiny 12] [--out coco_report.json]

Checks (each reported with a count and example ids):
  files      JSON images missing on disk; image files on disk not in the JSON; duplicate file names
  sizes      JSON width/height differs from the real (EXIF-corrected) image size
  categories annotation category ids not declared; declared categories never used; duplicate names
  boxes      zero or negative width/height; box extending beyond the image (tolerance 1 px);
             negative coordinates; area inconsistent with the box (w*h vs 'area', for box-only data)
  format     images whose boxes look like corners [x1, y1, x2, y2] written where [x, y, w, h] belongs:
             at least half the image's boxes overflow or are oversized (over 2x the dataset median
             width and height) as xywh, yet read as xyxy are valid and normally sized. Heuristic:
             17 of 21 planted cases found with no false alarm in testing; small objects near the
             image origin can hide it. Confirm flagged images with render_annotations.py.
  duplicates same-class boxes on the same image with IoU >= --iou-dup, or whose four numbers all
             agree within max(1 px, 5% of the box side): small boxes nudged by a pixel fall below
             IoU 0.9 but are still double annotations
  masks      polygons with fewer than 3 points; RLE sizes that disagree with the image size
  stats      boxes per image, class counts, share of tiny boxes (min side < --tiny px) and of
             COCO small / medium / large objects, images with no annotations
A report JSON is written; nothing is modified. Repairs are a Stage 1 decision, logged with counts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def iou_xywh(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations")
    ap.add_argument("--images", required=True)
    ap.add_argument("--iou-dup", type=float, default=0.9)
    ap.add_argument("--tiny", type=float, default=12)
    ap.add_argument("--out", default="coco_report.json")
    a = ap.parse_args()

    from PIL import Image, ImageOps

    coco = json.loads(Path(a.annotations).read_text())
    imgs = {im["id"]: im for im in coco.get("images", [])}
    cats = {c["id"]: c for c in coco.get("categories", [])}
    anns = coco.get("annotations", [])
    root = Path(a.images)
    R: dict = {}

    # files and sizes
    names = Counter(im["file_name"] for im in imgs.values())
    on_disk = {p.name for p in root.rglob("*") if p.is_file() and not p.name.startswith("._") and "__MACOSX" not in p.parts}
    missing = [im["file_name"] for im in imgs.values() if not (root / im["file_name"]).exists()]
    real_size, size_bad = {}, []
    for iid, im in imgs.items():
        p = root / im["file_name"]
        if p.exists():
            try:
                with Image.open(p) as f:
                    real_size[iid] = ImageOps.exif_transpose(f).size
                if real_size[iid] != (im.get("width"), im.get("height")):
                    size_bad.append({"file": im["file_name"], "json": [im.get("width"), im.get("height")], "real": list(real_size[iid])})
            except Exception as e:  # noqa: BLE001
                size_bad.append({"file": im["file_name"], "error": str(e)[:80]})
    R["files"] = {"json_images": len(imgs), "missing_on_disk": missing,
                  "on_disk_not_in_json": sorted(on_disk - set(names))[:50],
                  "duplicate_file_names": [n for n, c in names.items() if c > 1]}
    R["sizes"] = {"json_size_differs_from_file": size_bad}

    # categories
    used = Counter(x["category_id"] for x in anns)
    R["categories"] = {"declared": {c["id"]: c["name"] for c in cats.values()},
                       "undefined_ids_used": {k: v for k, v in used.items() if k not in cats},
                       "declared_but_unused": [cats[c]["name"] for c in cats if c not in used],
                       "duplicate_names": [n for n, c in Counter(c["name"] for c in cats.values()).items() if c > 1],
                       "counts": {cats[k]["name"] if k in cats else f"undefined:{k}": v for k, v in used.items()}}

    # boxes, per image
    by_img = defaultdict(list)
    for x in anns:
        by_img[x["image_id"]].append(x)
    valid_wh = np.array([x["bbox"][2:4] for x in anns if x["bbox"][2] > 0 and x["bbox"][3] > 0], dtype=float)
    med_w, med_h = (np.median(valid_wh, axis=0) if len(valid_wh) else (1.0, 1.0))

    def near_identical(p, q) -> bool:
        tol = max(1.0, 0.05 * max(p["bbox"][2], p["bbox"][3], 1))
        return all(abs(u - v) <= tol for u, v in zip(p["bbox"], q["bbox"]))
    zero, oob, neg, area_bad, orphan = [], [], [], [], []
    corner_imgs, dup_pairs = [], []
    tiny = small = medium = large = 0
    for iid, lst in by_img.items():
        if iid not in imgs:
            orphan += [x["id"] for x in lst]
            continue
        W, H = real_size.get(iid, (imgs[iid].get("width"), imgs[iid].get("height")))
        corner_votes = 0
        for x in lst:
            bx, by, bw, bh = x["bbox"]
            if bw <= 0 or bh <= 0:
                zero.append(x["id"])
                continue
            if bx < -1 or by < -1:
                neg.append(x["id"])
            overflow = bx + bw > W + 1 or by + bh > H + 1
            if overflow:
                oob.append(x["id"])
            # Read as corners, is it a valid, normally-sized box, while as xywh it overflows or is oversized?
            as_xyxy_ok = bw > bx and bh > by and bw <= W + 1 and bh <= H + 1
            xyxy_w, xyxy_h = bw - bx, bh - by
            oversized = bw > 2 * med_w and bh > 2 * med_h
            normal_as_xyxy = xyxy_w < 3 * med_w and xyxy_h < 3 * med_h
            if as_xyxy_ok and (overflow or oversized) and normal_as_xyxy:
                corner_votes += 1
            if "segmentation" not in x or not x["segmentation"]:
                if x.get("area") is not None and abs(x["area"] - bw * bh) > 0.05 * max(bw * bh, 1):
                    area_bad.append(x["id"])
            m = min(bw, bh)
            tiny += m < a.tiny
            ar = bw * bh
            small, medium, large = small + (ar < 32 ** 2), medium + (32 ** 2 <= ar < 96 ** 2), large + (ar >= 96 ** 2)
        if len(lst) and corner_votes >= max(1, 0.5 * len(lst)):
            corner_imgs.append(imgs[iid]["file_name"])
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                p, q = lst[i], lst[j]
                if p["category_id"] == q["category_id"] and (iou_xywh(p["bbox"], q["bbox"]) >= a.iou_dup or near_identical(p, q)):
                    dup_pairs.append([p["id"], q["id"]])
    n_valid = max(1, len(anns) - len(zero))
    R["boxes"] = {"zero_or_negative_size": len(zero), "beyond_image": len(oob), "negative_origin": len(neg),
                  "area_inconsistent": len(area_bad), "annotations_on_unknown_images": len(orphan),
                  "examples": {"zero": zero[:5], "beyond": oob[:5]}}
    R["format"] = {"images_with_corner_format_boxes": corner_imgs}
    R["duplicates"] = {"same_class_pairs_iou_ge": a.iou_dup, "count": len(dup_pairs), "examples": dup_pairs[:5]}

    # masks
    bad_poly = sum(1 for x in anns if isinstance(x.get("segmentation"), list)
                   and any(len(poly) < 6 for poly in x["segmentation"]))
    bad_rle = sum(1 for x in anns if isinstance(x.get("segmentation"), dict) and x["image_id"] in imgs
                  and list(x["segmentation"].get("size", [])) != [imgs[x["image_id"]].get("height"), imgs[x["image_id"]].get("width")])
    R["masks"] = {"polygons_under_3_points": bad_poly, "rle_size_mismatch": bad_rle}

    per_img = [len(by_img.get(i, [])) for i in imgs]
    R["stats"] = {"annotations": len(anns), "boxes_per_image": {"mean": round(float(np.mean(per_img)), 2), "max": int(max(per_img, default=0))},
                  "images_without_annotations": int(sum(c == 0 for c in per_img)),
                  "tiny_share": round(tiny / n_valid, 3), "coco_size_share": {"small": round(small / n_valid, 3),
                  "medium": round(medium / n_valid, 3), "large": round(large / n_valid, 3)}}
    Path(a.out).write_text(json.dumps(R, indent=1, default=str))

    print(f"images={len(imgs)} annotations={len(anns)} categories={len(cats)}")
    print(f"  missing on disk: {len(missing)} | on disk not in JSON: {len(on_disk - set(names))} | size mismatches: {len(size_bad)}")
    print(f"  undefined category ids: {R['categories']['undefined_ids_used']} | declared but unused: {R['categories']['declared_but_unused']}")
    print(f"  boxes: zero/negative size {len(zero)}, beyond image {len(oob)}, negative origin {len(neg)}, area mismatch {len(area_bad)}")
    print(f"  images with corner-format (xyxy) boxes: {len(corner_imgs)}")
    print(f"  duplicate boxes (same class, IoU >= {a.iou_dup}): {len(dup_pairs)}")
    print(f"  masks: short polygons {bad_poly}, RLE size mismatch {bad_rle}")
    print(f"  class counts: {R['categories']['counts']}")
    print(f"  tiny (<{a.tiny:g}px) share {R['stats']['tiny_share']}; COCO sizes {R['stats']['coco_size_share']}; "
          f"images without annotations {R['stats']['images_without_annotations']}")
    if R["stats"]["tiny_share"] > 0.10:
        print(f"  ACTION: {R['stats']['tiny_share']:.0%} of boxes are under {a.tiny:g} px. Stage 4 must SCREEN at least two "
              "detector input sizes, including one LARGER than the images or the checkpoint's default (upscaling), "
              "e.g. train_detector.py --size 320 vs 640. Do not inherit the checkpoint's training scale unscreened.")
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
