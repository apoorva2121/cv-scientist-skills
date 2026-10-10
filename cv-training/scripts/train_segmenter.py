"""Train or fine-tune a semantic segmentation model (segmentation-models-pytorch, MIT) on a fixed split.

Usage (short proxy on one fold, or full training with --eval-part holdout):
    python train_segmenter.py IMAGE_DIR --masks MASK_DIR --meta metadata_grouped.csv --key file --split split/ \
        --class-map "1:1,2:0,3:255" --arch Unet --encoder resnet34 --size 256 --epochs 20 --batch 8 --lr 3e-4 \
        [--fold 0 | --eval-part holdout] [--aug seg_flip|seg_ms|seg_none] [--loss ce|ce_dice] [--ema 0.0]
        [--mask-ext .png] [--max-train 0] [--seed 0] [--threads 0] [--out run_dir]

Masks: one single-channel PNG per image in MASK_DIR, same stem as the image (plus --mask-ext). --class-map maps
raw mask values to training ids; 255 = ignore (boundaries, unlabelled pixels, padding). Unmapped raw values
are an error, never silently a class. Label maps are resized with nearest-neighbour and padded with 255.

Models: smp.create_model(arch, encoder, encoder_weights="imagenet"): Unet, UnetPlusPlus, FPN, DeepLabV3Plus,
UPerNet, DPT, Segformer; timm encoders as "tu-<timm name>" (e.g. tu-convnext_tiny). Licence (checked
2026-09-27): the SegFormer/MiT weights and smp's mix_transformer code (encoders mit_b0..b5) are NVIDIA
research-only; the script warns when one is chosen. ResNet and timm ConvNeXt/EfficientNet encoders are
permissive. Check every encoder's weights licence in cv-model-selection.
Metric note: mIoU here is DATASET-level (intersection and union summed over images, then averaged over
classes), as in ADE20K/Cityscapes papers. torchmetrics.segmentation.MeanIoU (1.9.0) averages per image and
has no ignore index, so its numbers are not comparable.

Evaluation follows the standard protocol: the image is resized to --size, logits are upsampled back to the
ORIGINAL mask size, and metrics are computed there, ignoring 255.
Contract (used by compare.py / joint_search.py):
  - one JSON line per epoch: {"epoch", "train_loss", "miou", "mdice", "pixel_acc", "epoch_s"} (+ "ema_*")
  - --out/metrics.json {"best", "history", "train_images", "val_images", "minutes", "config", "per_class_iou"}
  - --out/val_seg_counts.npz: per-image intersection, union, predicted and true pixel counts per class, for
    image-bootstrap intervals and paired comparisons of mIoU/Dice; --out/model.pt
mIoU = mean over classes (present in the ground truth of the evaluated images) of total intersection /
total union, summed over images; mDice analogously.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "cv-data-audit" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "cv-validation-design" / "scripts"))

IGNORE = 255


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--masks", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--key", default="file")
    ap.add_argument("--split", required=True)
    ap.add_argument("--class-map", required=True, help='raw:train pairs, e.g. "1:1,2:0,3:255"')
    ap.add_argument("--mask-ext", default=".png")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--eval-part", default="")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--aug", default="seg_flip", choices=["seg_none", "seg_flip", "seg_ms"])
    ap.add_argument("--loss", default="ce", choices=["ce", "ce_dice"])
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="run_seg")
    a = ap.parse_args()

    import pandas as pd
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.v2 as T
    from PIL import Image
    from timm.utils import ModelEmaV3
    from torchvision import tv_tensors

    from cv_folds import load_split
    from embed_images import load_rgb

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cmap = {int(k): int(v) for k, v in (p.split(":") for p in a.class_map.split(","))}
    n_cls = max(v for v in cmap.values() if v != IGNORE) + 1
    lut = np.full(256, -1, dtype=np.int64)
    for k, v in cmap.items():
        lut[k] = v

    def load_mask(f: str) -> np.ndarray:
        raw = np.asarray(Image.open(Path(a.masks) / (Path(f).stem + a.mask_ext)))
        if raw.ndim == 3:
            raise SystemExit(f"mask {f} has {raw.shape[2]} channels; expected a single-channel label map")
        m = lut[raw]
        if (m < 0).any():
            bad = sorted(set(np.unique(raw[m < 0]).tolist()))
            raise SystemExit(f"mask {f} has raw values {bad} missing from --class-map")
        return m.astype(np.uint8)

    meta = pd.read_csv(a.meta)
    meta[a.key] = meta[a.key].astype(str)
    s = load_split(Path(a.split) / "split.csv", ids=meta[a.key])
    if a.eval_part == "holdout":
        fit, val = meta[s.train_mask], meta[s.holdout_mask]
    else:
        pool = meta[s.train_mask].reset_index(drop=True)
        ti, vi = s.folds[a.fold]
        fit, val = pool.iloc[ti], pool.iloc[vi]
    if a.max_train and len(fit) > a.max_train:
        fit = fit.sample(a.max_train, random_state=a.seed)
    fit_files, val_files = list(fit[a.key]), list(val[a.key])

    fill = {tv_tensors.Image: 0, tv_tensors.Mask: IGNORE}
    geo = {
        "seg_none": [T.Resize((a.size, a.size), antialias=True)],
        "seg_flip": [T.RandomResizedCrop(a.size, scale=(0.5, 1.0), antialias=True), T.RandomHorizontalFlip()],
        "seg_ms": [T.ScaleJitter((a.size, a.size), scale_range=(0.5, 2.0), antialias=True),
                   T.RandomCrop((a.size, a.size), pad_if_needed=True, fill=fill), T.RandomHorizontalFlip(),
                   T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3)], p=0.5)],
    }[a.aug]
    if a.encoder.startswith("mit_"):
        print("LICENCE WARNING: mit_b* (SegFormer/MiT) weights and smp's mix_transformer code are NVIDIA research-only; "
              "do not ship without a commercial licence", flush=True)
    model = smp.create_model(a.arch, encoder_name=a.encoder, encoder_weights="imagenet" if a.pretrained else None,
                             classes=n_cls)
    pp = smp.encoders.get_preprocessing_params(a.encoder) if a.pretrained else {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    norm = T.Normalize(pp["mean"], pp["std"])
    train_tf = T.Compose([T.ToImage(), *geo, T.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64,
                                                         "others": None}, scale=True), norm])

    class DS(torch.utils.data.Dataset):
        def __init__(self, files, train):
            self.files, self.train = files, train

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):
            f = self.files[i]
            img = load_rgb(Path(a.image_dir) / f)
            m = load_mask(f)
            if m.shape != (img.height, img.width):
                raise SystemExit(f"mask {f} is {m.shape}, image is {(img.height, img.width)}")
            if self.train:
                x, y = train_tf(tv_tensors.Image(T.functional.pil_to_tensor(img)), tv_tensors.Mask(torch.from_numpy(m)))
                return x, y.long()
            x = T.functional.to_dtype(T.functional.pil_to_tensor(img.resize((a.size, a.size), Image.BILINEAR)), torch.float32, scale=True)
            return norm(x), torch.from_numpy(m.astype(np.int64))

    tr_ds, va_ds = DS(fit_files, True), DS(val_files, False)
    dl = torch.utils.data.DataLoader(tr_ds, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    vl = torch.utils.data.DataLoader(va_ds, batch_size=1, shuffle=False, num_workers=0)

    ce = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE) if a.loss == "ce_dice" else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = max(1, len(dl))
    total = a.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: (k + 1) / steps if k < steps else 0.5 * (1 + np.cos(np.pi * (k - steps) / max(1, total - steps))))
    ema = ModelEmaV3(model, decay=a.ema, use_warmup=True) if a.ema else None

    def evaluate(m):
        m.eval()
        I, U, P, G = (np.zeros((len(va_ds), n_cls), dtype=np.int64) for _ in range(4))
        correct = total_px = 0
        with torch.inference_mode():
            for i, (x, y) in enumerate(vl):
                logits = F.interpolate(m(x), size=y.shape[-2:], mode="bilinear", align_corners=False)
                pred = logits.argmax(1)[0].numpy()
                gt = y[0].numpy()
                valid = gt != IGNORE
                correct += int((pred[valid] == gt[valid]).sum())
                total_px += int(valid.sum())
                for c in range(n_cls):
                    pc, gc = (pred == c) & valid, gt == c
                    I[i, c], U[i, c] = int((pc & gc).sum()), int((pc | gc).sum())
                    P[i, c], G[i, c] = int(pc.sum()), int(gc.sum())
        present = G.sum(0) > 0
        iou = I.sum(0) / np.maximum(U.sum(0), 1)
        dsc = 2 * I.sum(0) / np.maximum(P.sum(0) + G.sum(0), 1)
        res = {"miou": float(iou[present].mean()), "mdice": float(dsc[present].mean()),
               "pixel_acc": correct / max(total_px, 1)}
        return res, (I, U, P, G), iou

    hist, best, t_start, step = [], None, time.time(), 0
    for ep in range(a.epochs):
        model.train()
        t0, losses = time.time(), []
        for x, y in dl:
            logits = model(x)
            loss = ce(logits, y) + (dice(logits, y) if dice is not None else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if ema is not None:
                ema.update(model, step=step)
            losses.append(loss.item())
        res, counts, iou = evaluate(model)
        rec = {"epoch": ep + 1, "train_loss": round(float(np.mean(losses)), 4), **{k: round(v, 4) for k, v in res.items()},
               "epoch_s": round(time.time() - t0, 1)}
        final = (counts, iou, model)
        if ema is not None:
            res_e, counts_e, iou_e = evaluate(ema.module)
            rec.update({f"ema_{k}": round(v, 4) for k, v in res_e.items()})
            final = (counts_e, iou_e, ema.module)
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        score = rec.get("ema_miou", rec["miou"])
        if best is None or score > best["miou"]:
            best = {"miou": score, "epoch": ep + 1}
    (I, U, P, G), iou, m_final = final
    np.savez(out / "val_seg_counts.npz", files=np.array(val_files), inter=I, union=U, pred=P, gt=G)
    torch.save({"state_dict": m_final.state_dict(), "n_classes": n_cls, "class_map": cmap, "config": vars(a)}, out / "model.pt")
    summary = {"best": best, "history": hist, "train_images": len(tr_ds), "val_images": len(va_ds),
               "minutes": round((time.time() - t_start) / 60, 2), "config": vars(a),
               "per_class_iou": [round(float(v), 4) for v in iou]}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(f"last epoch miou {hist[-1].get('ema_miou', hist[-1]['miou'])} | per-class IoU {summary['per_class_iou']} | "
          f"train {len(tr_ds)} / val {len(va_ds)} images | {summary['minutes']} min")


if __name__ == "__main__":
    main()
