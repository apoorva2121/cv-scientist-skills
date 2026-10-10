"""Train or fine-tune a torchvision detector on a COCO dataset with a fixed split; save COCO metrics and predictions.

Usage (short proxy on one fold, or full training with --eval-part holdout):
    python train_detector.py IMAGE_DIR --coco annotations.json --split split/ \
        --model fasterrcnn_mobilenet_v3_large_fpn --size 640 --epochs 12 --batch 4 --lr 0.01 \
        [--fold 0 | --eval-part holdout] [--aug det_flip|det_safe|det_ssd|det_lsj|det_none] [--strong-off-last 2]
        [--opt sgd|adamw] [--wd 1e-4] [--ema 0.0] [--trainable-layers 3] [--max-train 0] [--seed 0] [--threads 0]
        [--out run_dir]

Why torchvision detectors: BSD-3 code, COCO-pretrained weights, CPU-runnable small variants, and the same
tv_tensors augmentation API. They are the licence-safe BASELINE, not the accuracy ceiling: for the best
accuracy compare against current real-time DETRs (see cv-model-selection) with the same split, the same
metric and this script's output contract.

Models (num_classes = categories + background, labels 1..K):
  fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2,
  retinanet_resnet50_fpn_v2, fcos_resnet50_fpn
--size sets the input size for every model (the detector resizes so the longer side equals it), so the
*_320_fpn name is only the pretraining size. Small objects need a larger --size (upscale images that are
smaller than the detector's usual input); screen at least two sizes when objects are tiny.
Augmentation policies are shared with cv-augmentation/scripts/aug_preview.py: preview exactly what trains.
--strong-off-last N switches to plain flips for the last N epochs (Mosaic-off practice for strong policies).
The detector resizes so the longer side equals --size (min_size = max_size = size).

Contract (used by cv-method-comparison/scripts/joint_search.py and compare.py):
  - one JSON line per epoch: {"epoch", "train_loss", "map", "map50", "map75", "map_small", "map_medium",
    "map_large", "ar100", "epoch_s"} (+ "ema_map" ... when --ema > 0)
  - --out/metrics.json: {"best", "history", "train_images", "val_images", "minutes", "config"}
  - --out/val_dets.json: COCO-format detections on the validation images from the last epoch (for box
    fusion, per-class thresholds and error analysis); --out/model.pt
COCO metrics use pycocotools on the validation images only (maxDets 100). Empty images count.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "cv-data-audit" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "cv-validation-design" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "cv-augmentation" / "scripts"))

MODELS = ["fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_resnet50_fpn_v2",
          "retinanet_resnet50_fpn_v2", "fcos_resnet50_fpn"]


def build_model(name: str, num_classes: int, size: int, pretrained: bool, trainable: int):
    import functools

    import torch
    import torchvision.models.detection as D
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.fcos import FCOSClassificationHead
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    kw = {"min_size": size, "max_size": size}
    m = getattr(D, name)(weights="DEFAULT" if pretrained else None, weights_backbone=None,
                         trainable_backbone_layers=trainable, **kw)
    if name.startswith("fasterrcnn"):
        inf = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(inf, num_classes)
    elif name.startswith("retinanet"):
        head = m.head.classification_head
        in_ch = head.conv[0][0].in_channels
        m.head.classification_head = RetinaNetClassificationHead(in_ch, head.num_anchors, num_classes,
                                                                 norm_layer=functools.partial(torch.nn.GroupNorm, 32))
    elif name.startswith("fcos"):
        head = m.head.classification_head
        in_ch = head.conv[0].in_channels
        m.head.classification_head = FCOSClassificationHead(in_ch, head.num_anchors, num_classes)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--coco", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--eval-part", default="")
    ap.add_argument("--model", default="fasterrcnn_mobilenet_v3_large_fpn", choices=MODELS)
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--trainable-layers", type=int, default=3)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--opt", default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--aug", default="det_flip")
    ap.add_argument("--strong-off-last", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="run_det")
    a = ap.parse_args()

    import pandas as pd
    import torch
    import torchvision.transforms.v2 as T
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from torchvision import tv_tensors

    from aug_preview import policy
    from cv_folds import load_split
    from embed_images import load_rgb

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    coco = json.loads(Path(a.coco).read_text())
    cats = sorted(c["id"] for c in coco["categories"])
    used = {x["category_id"] for x in coco["annotations"]}
    cats = [c for c in cats if c in used]  # declared-but-unused categories (e.g. a parent class) get no output
    cat2lab = {c: i + 1 for i, c in enumerate(cats)}
    lab2cat = {v: k for k, v in cat2lab.items()}
    images = {im["file_name"]: im for im in coco["images"]}
    anns = {}
    for x in coco["annotations"]:
        if x["category_id"] in cat2lab:
            anns.setdefault(x["image_id"], []).append(x)

    files = pd.Series(sorted(f for f in images if (Path(a.image_dir) / f).exists()))
    s = load_split(Path(a.split) / "split.csv", ids=files)
    tr_files = files[s.train_mask].reset_index(drop=True)
    if a.eval_part == "holdout":
        fit, val = tr_files, files[s.holdout_mask].reset_index(drop=True)
    else:
        ti, vi = s.folds[a.fold]
        fit, val = tr_files.iloc[ti].reset_index(drop=True), tr_files.iloc[vi].reset_index(drop=True)
    if a.max_train and a.max_train < len(fit):
        fit = fit.sample(a.max_train, random_state=a.seed).reset_index(drop=True)

    to_img = [T.ToImage(), T.ToDtype(torch.uint8, scale=True)]
    strong, _ = policy(a.aug, a.size)
    plain = T.Compose(to_img + [T.RandomHorizontalFlip(), T.ClampBoundingBoxes(), T.SanitizeBoundingBoxes(min_size=1)])
    evalt = T.Compose(to_img)

    class DS(torch.utils.data.Dataset):
        def __init__(self, names, train):
            self.names, self.train, self.tf = list(names), train, strong if train else evalt

        def __len__(self):
            return len(self.names)

        def __getitem__(self, i):
            im = images[self.names[i]]
            img = load_rgb(Path(a.image_dir) / self.names[i])
            b = []
            lab = []
            for x in anns.get(im["id"], []):
                x0, y0, w, h = x["bbox"]
                if w > 0 and h > 0:
                    b.append([x0, y0, x0 + w, y0 + h])
                    lab.append(cat2lab[x["category_id"]])
            boxes = tv_tensors.BoundingBoxes(torch.tensor(b, dtype=torch.float32).reshape(-1, 4), format="XYXY",
                                             canvas_size=(img.height, img.width))
            target = {"boxes": boxes, "labels": torch.tensor(lab, dtype=torch.int64)}
            if self.train:
                img_t, target = self.tf(img, target)
            else:
                img_t = self.tf(img)  # evaluation reads ground truth from the COCO file, not from here
            return img_t.float() / 255.0, {"boxes": target["boxes"].as_subclass(torch.Tensor), "labels": target["labels"],
                                           "image_id": im["id"]}

    def collate(b):
        return tuple(zip(*b))

    tr_ds, va_ds = DS(fit, True), DS(val, False)
    dl = torch.utils.data.DataLoader(tr_ds, batch_size=a.batch, shuffle=True, num_workers=0, collate_fn=collate, drop_last=True)
    vl = torch.utils.data.DataLoader(va_ds, batch_size=a.batch, shuffle=False, num_workers=0, collate_fn=collate)

    model = build_model(a.model, len(cats) + 1, a.size, bool(a.pretrained), a.trainable_layers)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = (torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=a.wd) if a.opt == "sgd"
           else torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd))
    steps = max(1, len(dl))
    warm = min(steps, 500)
    total = a.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: (k + 1) / warm if k < warm else 0.5 * (1 + np.cos(np.pi * (k - warm) / max(1, total - warm))))
    from timm.utils import ModelEmaV3  # EMA with warmup: decay ramps up with the step count, so short runs work
    ema = ModelEmaV3(model, decay=a.ema, use_warmup=True) if a.ema else None

    import contextlib
    import io
    gt = COCO()
    gt.dataset = copy.deepcopy(coco)
    gt.dataset["annotations"] = [dict(x, iscrowd=x.get("iscrowd", 0), area=x.get("area", x["bbox"][2] * x["bbox"][3]))
                                 for x in coco["annotations"] if x["category_id"] in cat2lab and x["bbox"][2] > 0 and x["bbox"][3] > 0]
    for i, x in enumerate(gt.dataset["annotations"]):
        x.setdefault("id", i + 1)
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    val_ids = [images[f]["id"] for f in val]

    def evaluate(m):
        m.eval()
        dets = []
        with torch.inference_mode():
            for xs, ts in vl:
                for o, t in zip(m(list(xs)), ts):
                    for bx, sc, lb in zip(o["boxes"].tolist(), o["scores"].tolist(), o["labels"].tolist()):
                        dets.append({"image_id": t["image_id"], "category_id": lab2cat[lb],
                                     "bbox": [bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]], "score": float(sc)})
        keys = ["map", "map50", "map75", "map_small", "map_medium", "map_large"]
        if not dets:
            return dict.fromkeys(keys + ["ar100"], 0.0), dets
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(dets)
            ev = COCOeval(gt, dt, "bbox")
            ev.params.imgIds = val_ids
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        st = [max(0.0, float(v)) for v in ev.stats]
        return dict(zip(keys, st[:6])) | {"ar100": st[8]}, dets

    hist, best, t_start = [], None, time.time()
    step = 0
    for ep in range(a.epochs):
        if a.strong_off_last and ep == a.epochs - a.strong_off_last:
            tr_ds.tf = plain
        model.train()
        t0, losses = time.time(), []
        for xs, ts in dl:
            ts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in ts]
            loss = sum(model(list(xs), ts).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            step += 1
            if ema is not None:
                ema.update(model, step=step)
            losses.append(loss.item())
        res, dets = evaluate(model)
        rec = {"epoch": ep + 1, "train_loss": round(float(np.mean(losses)), 4), **{k: round(v, 4) for k, v in res.items()},
               "epoch_s": round(time.time() - t0, 1)}
        final_dets, final_model = dets, model
        if ema is not None:
            res_e, dets_e = evaluate(ema.module)
            rec.update({f"ema_{k}": round(v, 4) for k, v in res_e.items()})
            final_dets, final_model = dets_e, ema.module
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        score = rec.get("ema_map", rec["map"])
        if best is None or score > best["map"]:
            best = {"map": score, "epoch": ep + 1}
    (out / "val_dets.json").write_text(json.dumps(final_dets))
    torch.save({"state_dict": final_model.state_dict(), "categories": cats, "config": vars(a)}, out / "model.pt")
    summary = {"best": best, "history": hist, "train_images": len(tr_ds), "val_images": len(va_ds),
               "minutes": round((time.time() - t_start) / 60, 2), "config": vars(a)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(f"last epoch map {hist[-1].get('ema_map', hist[-1]['map'])} | best {best} | train {len(tr_ds)} / val {len(va_ds)} "
          f"images | {summary['minutes']} min")


if __name__ == "__main__":
    main()
