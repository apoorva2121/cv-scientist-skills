"""Stage 10 metrics with honest intervals: image- (or group-) bootstrap CIs, slices, and paired comparisons.

Usage:
  classification  python metrics_ci.py cls  --pred test_probs.npz [--pred-b other.npz] [--meta meta.csv --key file
                                            --slice camera_id --group split_group] [--n-boot 2000] [--out report.json]
  detection       python metrics_ci.py det  --gt annotations.json --dets test_dets.json [--dets-b other.json]
                                            [--images test_ids.txt] [--meta ... --slice ... --group ...] [--n-boot 200]
  segmentation    python metrics_ci.py seg  --counts val_seg_counts.npz [--counts-b other.npz] [--meta ... --slice ...]

Inputs are the files the cv-training scripts write: *.npz with files/labels/probs/classes (classification),
COCO-format detections (detection), per-image intersection/union counts (segmentation).

What it reports
  - Point estimates: classification accuracy, macro-F1, balanced accuracy, macro one-vs-rest AUROC, top-5
    accuracy (if more than 5 classes), ECE (15 equal-width bins; biased upward on small sets, so read it with its
    interval), per-class precision/recall/F1 and the most-confused class pairs. Detection: COCO AP, AP50, AP75,
    AP small/medium/large, AR100 (pycocotools, maxDets 100). Segmentation: mIoU, mDice, per-class IoU.
  - 95% percentile intervals from resampling IMAGES with replacement, or whole GROUPS with --group (use it
    whenever images share a patient, session, video or scene: image resampling would understate the width).
  - --slice COL: every metric per value of a metadata column (camera, site, size bin), each with its interval.
    Slices under 30 images are flagged: their intervals are too wide to act on.
  - --pred-b / --dets-b / --counts-b: a second model on the SAME images, compared by a paired bootstrap (both
    models scored on each resample), giving the difference, its interval and the share of resamples where A wins.
Accuracy also gets a Wilson interval (exact enough for a single proportion; ignores grouping).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def resample_index(n: int, groups, rng) -> np.ndarray:
    if groups is None:
        return rng.integers(0, n, n)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([members[g] for g in pick])


LOWER_IS_BETTER = {"ece_15bin"}


def boot(metric_fn, n: int, n_boot: int, groups=None, seed: int = 0, metric_b=None) -> dict:
    """metric_fn(idx) -> dict of scalars. With metric_b, also the paired difference A - B."""
    rng = np.random.default_rng(seed)
    point = metric_fn(np.arange(n))
    draws = {k: [] for k in point}
    diffs = {k: [] for k in point} if metric_b is not None else None
    point_b = metric_b(np.arange(n)) if metric_b is not None else None
    for _ in range(n_boot):
        idx = resample_index(n, groups, rng)
        r = metric_fn(idx)
        for k, v in r.items():
            draws[k].append(v)
        if metric_b is not None:
            rb = metric_b(idx)
            for k in point:
                diffs[k].append(r[k] - rb[k])
    out = {}
    for k, v in point.items():
        arr = np.array(draws[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        e = {"value": round(float(v), 4)}
        if len(arr):
            e["ci95"] = [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]
        if metric_b is not None:
            d = np.array(diffs[k], dtype=float)
            d = d[np.isfinite(d)]
            e["b_value"] = round(float(point_b[k]), 4)
            e["diff_a_minus_b"] = round(float(v - point_b[k]), 4)
            if len(d):
                lo, hi = np.percentile(d, [2.5, 97.5])
                e["diff_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
                sign = -1 if k in LOWER_IS_BETTER else 1
                e["p_a_better"] = round(float((sign * d > 0).mean()), 3)
                a_wins, b_wins = (lo > 0, hi < 0) if sign == 1 else (hi < 0, lo > 0)
                e["verdict"] = "A better" if a_wins else "B better" if b_wins else "tie (interval includes 0)"
        out[k] = e
    return out


# ---------------------------------------------------------------- classification
def cls_metrics(y, P, classes):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    def fn(idx):
        yy, pp = y[idx], P[idx]
        pred = pp.argmax(1)
        r = {"accuracy": float((pred == yy).mean()), "macro_f1": float(f1_score(yy, pred, average="macro",
                                                                                 labels=np.arange(len(classes)), zero_division=0)),
             "balanced_accuracy": float(balanced_accuracy_score(yy, pred))}
        present = np.unique(yy)
        if len(classes) > 2 and len(present) == len(classes):
            r["auroc_macro_ovr"] = float(roc_auc_score(yy, pp, multi_class="ovr", average="macro", labels=np.arange(len(classes))))
        elif len(classes) == 2 and len(present) == 2:
            r["auroc"] = float(roc_auc_score(yy, pp[:, 1]))
        if len(classes) > 5:
            top5 = np.argsort(-pp, 1)[:, :5]
            r["top5_accuracy"] = float((top5 == yy[:, None]).any(1).mean())
        conf = pp.max(1)
        bins = np.minimum((conf * 15).astype(int), 14)
        ece = 0.0
        for b in range(15):
            m = bins == b
            if m.any():
                ece += m.mean() * abs((pred[m] == yy[m]).mean() - conf[m].mean())
        r["ece_15bin"] = float(ece)
        return r
    return fn


def load_cls(path):
    z = np.load(path)
    return z["files"].astype(str), z["labels"].astype(int), z["probs"].astype(np.float64), list(z["classes"].astype(str))


def run_cls(a, meta):
    files, y, P, classes = load_cls(a.pred)
    Pb = None
    if a.pred_b:
        fb, yb, Pb_, cb = load_cls(a.pred_b)
        assert cb == classes, "models must share the class order"
        pos = {f: i for i, f in enumerate(fb)}
        common = [i for i, f in enumerate(files) if f in pos]
        if len(common) != len(files):
            print(f"WARNING: comparing on the {len(common)} images both models scored")
        files, y, P = files[common], y[common], P[common]
        Pb = Pb_[[pos[f] for f in files]]
    groups = group_array(meta, a, files)
    rep = {"task": "classification", "n_images": int(len(y)), "resampling": "groups" if groups is not None else "images"}
    rep["overall"] = boot(cls_metrics(y, P, classes), len(y), a.n_boot, groups,
                          metric_b=cls_metrics(y, Pb, classes) if Pb is not None else None)
    k = int((P.argmax(1) == y).sum())
    rep["overall"]["accuracy"]["wilson95"] = wilson(k, len(y))
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    pred = P.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=np.arange(len(classes)), zero_division=0)
    rep["per_class"] = {c: {"precision": round(float(pr[i]), 4), "recall": round(float(rc[i]), 4), "f1": round(float(f1[i]), 4),
                            "support": int(sup[i]), "recall_wilson95": wilson(int(((pred == i) & (y == i)).sum()), int(sup[i]))}
                        for i, c in enumerate(classes)}
    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    off = [(int(cm[i, j]), classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j]]
    rep["most_confused"] = [{"true": t, "pred": p_, "count": n} for n, t, p_ in sorted(off, reverse=True)[:10]]
    if a.slice:
        rep["slices"] = slices(meta, a, files, cls_metrics(y, P, classes), groups)
    return rep


# ---------------------------------------------------------------- detection
def run_det(a, meta):
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_raw = json.loads(Path(a.gt).read_text())
    ids = [im["id"] for im in gt_raw["images"]]
    if a.images:
        keep = set(Path(a.images).read_text().split())
        ids = [im["id"] for im in gt_raw["images"] if str(im["id"]) in keep or im["file_name"] in keep]
    id2file = {im["id"]: im["file_name"] for im in gt_raw["images"]}
    anns_by = {}
    for x in gt_raw["annotations"]:
        anns_by.setdefault(x["image_id"], []).append(x)

    def load_dets(p):
        d = json.loads(Path(p).read_text())
        by = {}
        for x in d:
            by.setdefault(x["image_id"], []).append(x)
        return by

    dets_a = load_dets(a.dets)
    dets_b = load_dets(a.dets_b) if a.dets_b else None

    def coco_ap(img_ids, dets_by):
        """Rebuild GT/DT with fresh ids so that images drawn twice count twice."""
        g = {"images": [], "annotations": [], "categories": gt_raw["categories"]}
        dt, aid = [], 1
        for new_id, old in enumerate(img_ids, start=1):
            g["images"].append({"id": new_id, "file_name": id2file[old]})
            for x in anns_by.get(old, []):
                y = dict(x, id=aid, image_id=new_id, iscrowd=x.get("iscrowd", 0), area=x.get("area", x["bbox"][2] * x["bbox"][3]))
                g["annotations"].append(y)
                aid += 1
            for x in dets_by.get(old, []):
                dt.append(dict(x, image_id=new_id))
        with contextlib.redirect_stdout(io.StringIO()):
            G = COCO()
            G.dataset = g
            G.createIndex()
            if not dt:
                return dict.fromkeys(["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR100"], 0.0)
            D = G.loadRes(dt)
            ev = COCOeval(G, D, "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = [float(v) for v in ev.stats]
        keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
        r = {k: v for k, v in zip(keys, s[:6])}
        r["AR100"] = s[8]
        return {k: (v if v >= 0 else float("nan")) for k, v in r.items()}  # -1 = no objects of that size

    ids = np.array(ids)
    files = np.array([id2file[i] for i in ids])
    groups = group_array(meta, a, files)
    rep = {"task": "detection", "n_images": int(len(ids)), "resampling": "groups" if groups is not None else "images",
           "note": "COCO AP via pycocotools on the listed images; empty images count"}
    rep["overall"] = boot(lambda idx: coco_ap(ids[idx], dets_a), len(ids), a.n_boot, groups,
                          metric_b=(lambda idx: coco_ap(ids[idx], dets_b)) if dets_b is not None else None)
    if a.slice:
        rep["slices"] = slices(meta, a, files, lambda idx: coco_ap(ids[idx], dets_a), groups)
    return rep


# ---------------------------------------------------------------- segmentation
def seg_metrics(I, U, P, G):
    def fn(idx):
        i, u, p, g = I[idx].sum(0), U[idx].sum(0), P[idx].sum(0), G[idx].sum(0)
        present = g > 0
        iou = i / np.maximum(u, 1)
        dsc = 2 * i / np.maximum(p + g, 1)
        r = {"miou": float(iou[present].mean()) if present.any() else float("nan"),
             "mdice": float(dsc[present].mean()) if present.any() else float("nan")}
        for c in range(I.shape[1]):
            r[f"iou_class{c}"] = float(iou[c]) if present[c] else float("nan")
        return r
    return fn


def run_seg(a, meta):
    z = np.load(a.counts)
    files, I, U, P, G = z["files"].astype(str), z["inter"], z["union"], z["pred"], z["gt"]
    fb = None
    if a.counts_b:
        zb = np.load(a.counts_b)
        pos = {f: i for i, f in enumerate(zb["files"].astype(str))}
        order = [pos[f] for f in files]
        fb = seg_metrics(zb["inter"][order], zb["union"][order], zb["pred"][order], zb["gt"][order])
    groups = group_array(meta, a, files)
    rep = {"task": "segmentation", "n_images": int(len(files)), "resampling": "groups" if groups is not None else "images"}
    rep["overall"] = boot(seg_metrics(I, U, P, G), len(files), a.n_boot, groups, metric_b=fb)
    if a.slice:
        rep["slices"] = slices(meta, a, files, seg_metrics(I, U, P, G), groups)
    return rep


# ---------------------------------------------------------------- shared
def group_array(meta, a, files):
    if meta is None or not a.group:
        return None
    m = dict(zip(meta[a.key].astype(str), meta[a.group].astype(str)))
    return np.array([m.get(f, f) for f in files])


def slices(meta, a, files, metric_on_idx, groups):
    m = dict(zip(meta[a.key].astype(str), meta[a.slice].astype(str)))
    vals = np.array([str(m.get(f, "<missing>")) for f in files])
    out = {}
    for v in sorted(set(vals)):
        idx = np.flatnonzero(vals == v)
        sub_groups = groups[idx] if groups is not None else None
        res = boot(lambda j: metric_on_idx(idx[j]), len(idx), max(200, a.n_boot // 5), sub_groups)
        res["n_images"] = int(len(idx))
        if len(idx) < 30:
            res["warning"] = "under 30 images: interval too wide to act on"
        out[str(v)] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["cls", "det", "seg"])
    ap.add_argument("--pred")
    ap.add_argument("--pred-b", default="")
    ap.add_argument("--gt")
    ap.add_argument("--dets")
    ap.add_argument("--dets-b", default="")
    ap.add_argument("--images", default="", help="file with image ids or file names to evaluate (detection)")
    ap.add_argument("--counts")
    ap.add_argument("--counts-b", default="")
    ap.add_argument("--meta", default="")
    ap.add_argument("--key", default="file")
    ap.add_argument("--group", default="")
    ap.add_argument("--slice", default="")
    ap.add_argument("--n-boot", type=int, default=0)
    ap.add_argument("--out", default="metrics_ci.json")
    a = ap.parse_args()
    if not a.n_boot:
        a.n_boot = 200 if a.task == "det" else 2000
    meta = None
    if a.meta:
        import pandas as pd
        meta = pd.read_csv(a.meta)
    rep = {"cls": run_cls, "det": run_det, "seg": run_seg}[a.task](a, meta)
    rep["n_boot"] = a.n_boot
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in rep if k not in ("per_class", "slices")}, indent=1)[:4000])
    if "slices" in rep:
        print("slices:", {k: {m: v[m]["value"] for m in list(v)[:2] if isinstance(v.get(m), dict)} | {"n": v["n_images"]}
                          for k, v in rep["slices"].items()})
    print(f"full report: {a.out}")


if __name__ == "__main__":
    main()
