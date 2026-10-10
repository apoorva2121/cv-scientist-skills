"""Stage 8 for detection: fuse detections from several models or TTA passes, and set per-class score
thresholds for the deployment operating point. Everything is scored with COCO AP on the same validation images.

Usage:
  fuse        python postprocess_det.py fuse --gt annotations.json --images val_ids.txt \
                  --dets runs/det/a/f0_s0/val_dets.json runs/det/b/f0_s0/val_dets.json [--weights 2,1] \
                  [--methods wbf,nmw,nms,softnms] [--iou 0.55] [--out fused/]
  thresholds  python postprocess_det.py thresholds --gt annotations.json --images val_ids.txt \
                  --dets val_dets.json [--iou-match 0.5] [--rule max-f1 | --min-precision 0.9] [--out thresholds.json]

Inputs are COCO-format detections (the val_dets.json that cv-training/scripts/train_detector.py writes, or any
detector's COCO results) for the SAME validation images, listed in --images (ids or file names).
fuse        Runs each method from ensemble-boxes (MIT): weighted boxes fusion (WBF), non-maximum weighted
            (NMW), NMS and soft-NMS over the pooled detections; boxes are normalised by image size first.
            Prints COCO AP for every input and every fused result and writes fused_<method>.json. Compare the
            best fused result with the best single input using cv-evaluation/scripts/metrics_ci.py det
            --dets ... --dets-b ... (paired image bootstrap): keep fusion only if the interval excludes zero
            and the extra inference cost fits the latency budget. TTA passes are fused the same way.
            Sanity check (run on 2026-09-27): fusing a model with ITSELF must leave AP unchanged. WBF, NMW and
            NMS pass; soft-NMS halves AP because it keeps every duplicate at a decayed score, and the extras
            count as false positives. Soft-NMS is for one model's crowded scenes, not for fusing models.
thresholds  COCO AP is threshold-free; deployment is not. Per class, the score threshold that maximises F1
            (or reaches --min-precision) at IoU >= --iou-match, with greedy matching by score. Reports
            precision/recall per class at the chosen threshold. Fit on validation, freeze, apply at test.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np


def load_gt(path: str, images: str):
    gt = json.loads(Path(path).read_text())
    ims = gt["images"]
    if images:
        keep = set(Path(images).read_text().split())
        ims = [im for im in ims if str(im["id"]) in keep or im["file_name"] in keep]
    return gt, {im["id"]: im for im in ims}


def coco_ap(gt: dict, imgs: dict, dets: list) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    dets = [d for d in dets if d["image_id"] in imgs]
    if not dets:
        return dict.fromkeys(keys, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO()
        g = dict(gt)
        g["annotations"] = [dict(x, iscrowd=x.get("iscrowd", 0), area=x.get("area", x["bbox"][2] * x["bbox"][3]))
                            for x in gt["annotations"] if x["bbox"][2] > 0 and x["bbox"][3] > 0]
        G.dataset = g
        G.createIndex()
        ev = COCOeval(G, G.loadRes(dets), "bbox")
        ev.params.imgIds = list(imgs)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: round(float(v), 4) for k, v in zip(keys, ev.stats[:6])}


def by_image(dets: list) -> dict:
    out = {}
    for d in dets:
        out.setdefault(d["image_id"], []).append(d)
    return out


def fuse(a) -> None:
    import ensemble_boxes as eb
    gt, imgs = load_gt(a.gt, a.images)
    runs = [json.loads(Path(p).read_text()) for p in a.dets]
    weights = [float(w) for w in a.weights.split(",")] if a.weights else [1.0] * len(runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(imgs)} validation images; {len(runs)} inputs")
    parts = [Path(p).resolve().with_suffix("").parts for p in a.dets]  # file stem included: files may share a folder
    for depth in range(1, max(len(x) for x in parts) + 1):
        names = ["/".join(x[-depth:]) for x in parts]
        if len(set(names)) == len(names):
            break
    for n, r in zip(names, runs):
        print(f"  input {n:30s} {coco_ap(gt, imgs, r)}")
    per = [by_image(r) for r in runs]
    for method in a.methods.split(","):
        fused = []
        for iid, im in imgs.items():
            W, H = im["width"], im["height"]
            bl, sl, ll = [], [], []
            for pr in per:
                ds = pr.get(iid, [])
                bl.append([[max(0, d["bbox"][0] / W), max(0, d["bbox"][1] / H), min(1, (d["bbox"][0] + d["bbox"][2]) / W),
                            min(1, (d["bbox"][1] + d["bbox"][3]) / H)] for d in ds])
                sl.append([d["score"] for d in ds])
                ll.append([d["category_id"] for d in ds])
            if not any(len(b) for b in bl):
                continue
            if method == "wbf":
                b, s, l_ = eb.weighted_boxes_fusion(bl, sl, ll, weights=weights, iou_thr=a.iou, skip_box_thr=0.0)
            elif method == "nmw":
                b, s, l_ = eb.non_maximum_weighted(bl, sl, ll, weights=weights, iou_thr=a.iou, skip_box_thr=0.0)
            elif method == "nms":
                b, s, l_ = eb.nms([np.array(x).reshape(-1, 4) for x in bl], sl, ll, weights=weights, iou_thr=a.iou)
            elif method == "softnms":
                b, s, l_ = eb.soft_nms([np.array(x).reshape(-1, 4) for x in bl], sl, ll, weights=weights, iou_thr=a.iou,
                                       sigma=0.5, thresh=0.001)
            else:
                raise SystemExit(f"unknown method {method}")
            for bb, sc, lb in zip(b, s, l_):
                x0, y0, x1, y1 = float(bb[0]) * W, float(bb[1]) * H, float(bb[2]) * W, float(bb[3]) * H
                fused.append({"image_id": iid, "category_id": int(lb), "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(sc)})
        (out / f"fused_{method}.json").write_text(json.dumps(fused))
        print(f"  fused {method:28s} {coco_ap(gt, imgs, fused)}")
    print("next: paired test of the best fused file vs the best input with cv-evaluation/scripts/metrics_ci.py det")


def iou(a, b) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax1, bx1) - max(a[0], b[0])), max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def thresholds(a) -> None:
    gt, imgs = load_gt(a.gt, a.images)
    dets = [d for d in json.loads(Path(a.dets).read_text()) if d["image_id"] in imgs]
    cats = {c["id"]: c["name"] for c in gt["categories"]}
    gts = {}
    for x in gt["annotations"]:
        if x["image_id"] in imgs and x["bbox"][2] > 0 and x["bbox"][3] > 0:
            gts.setdefault((x["image_id"], x["category_id"]), []).append(x["bbox"])
    report = {"rule": "max-f1" if not a.min_precision else f"precision >= {a.min_precision}", "iou_match": a.iou_match,
              "per_class": {}}
    for cid, name in cats.items():
        cd = sorted([d for d in dets if d["category_id"] == cid], key=lambda d: -d["score"])
        n_gt = sum(len(v) for (i, c), v in gts.items() if c == cid)
        if n_gt == 0 and not cd:
            continue
        used, tp_flags = {}, []
        for d in cd:
            cand = gts.get((d["image_id"], cid), [])
            flags = used.setdefault((d["image_id"], cid), [False] * len(cand))
            best, bj = 0.0, -1
            for j, g in enumerate(cand):
                if not flags[j]:
                    v = iou(d["bbox"], g)
                    if v > best:
                        best, bj = v, j
            if bj >= 0 and best >= a.iou_match:
                flags[bj] = True
                tp_flags.append(1)
            else:
                tp_flags.append(0)
        tp = np.cumsum(tp_flags)
        k = np.arange(1, len(cd) + 1)
        prec, rec = tp / k, tp / max(n_gt, 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        if not len(cd):
            report["per_class"][name] = {"n_gt": n_gt, "note": "no detections"}
            continue
        if a.min_precision:
            ok = np.flatnonzero(prec >= a.min_precision)
            i = int(ok[-1]) if len(ok) else int(np.argmax(f1))
        else:
            i = int(np.argmax(f1))
        report["per_class"][name] = {"threshold": round(float(cd[i]["score"]), 4), "precision": round(float(prec[i]), 4),
                                     "recall": round(float(rec[i]), 4), "f1": round(float(f1[i]), 4), "n_gt": n_gt,
                                     "detections_kept": i + 1}
    Path(a.out).write_text(json.dumps(report, indent=1))
    for n, v in report["per_class"].items():
        print(f"  {n:20s} {v}")
    print(f"written {a.out}; fit on validation images only, frozen for test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fuse", "thresholds"])
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", default="")
    ap.add_argument("--dets", nargs="+", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--methods", default="wbf,nmw,nms,softnms")
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--min-precision", type=float, default=0.0)
    ap.add_argument("--rule", default="max-f1")
    ap.add_argument("--out", default="post_det")
    a = ap.parse_args()
    if a.mode == "fuse":
        fuse(a)
    else:
        a.dets = a.dets[0]
        if a.out == "post_det":
            a.out = "thresholds.json"
        thresholds(a)


if __name__ == "__main__":
    main()
