"""Train or fine-tune an image classifier (timm backbone) on a fixed split; save per-image validation probabilities.

Usage (short proxy on one fold, or full training with --eval-part holdout):
    python train_classifier.py IMAGE_DIR --meta metadata.csv --key file --label label --split split/
        --model resnet18.a1_in1k --img-size 160 --epochs 8 --batch 32 --lr 1e-3 --wd 0.05
        [--fold 0 | --eval-part holdout] [--aug light|randaugment|trivialaugment|augmix] [--mix 0.0]
        [--label-smoothing 0.1] [--ema 0.999] [--layer-decay 0.75] [--freeze-epochs 0] [--lp-init] [--class-weights]
        [--eval-transform crop|letterbox|resize] [--seed 0] [--threads 8] [--out run_dir]

Design choices (each is a Stage 7 screening knob, not a fixed truth):
  - AdamW via timm.optim.create_optimizer_v2 (no weight decay on norms/biases; optional layer-wise decay),
    cosine schedule with 1 warmup epoch (timm CosineLRScheduler), optional EMA of weights (ModelEmaV3).
  - Augmentation from torchvision.transforms.v2 (BSD): RandomResizedCrop + flip, plus one policy;
    Mixup/CutMix through timm.data.Mixup with soft-target loss when --mix > 0.
  - Evaluation transform matches serving: crop (shorter side to S/0.875 then centre crop), letterbox or resize.
  - --lp-init (LP-FT, arXiv 2202.10054): fit a logistic-regression probe on frozen features of the training
    images, load it into the head, then fine-tune everything (use a small --lr). The probe's own validation
    score is logged as "lp_probe" and stored in metrics.json: every fine-tune must beat it.
  - Every epoch scores the validation part; the best epoch (by macro-F1) is kept, with and without EMA.
Outputs in --out: metrics.json (per-epoch history, best epoch, timings), val_probs.npz (files, labels,
probs, classes) for ensembling/stacking/calibration, and model.pt (state dict, classes, config).
Decoding uses cv-data-audit's load_rgb (EXIF orientation, colour modes), so training equals serving.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "cv-data-audit" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "cv-validation-design" / "scripts"))


def eval_resize(kind: str, size: int):
    from PIL import Image, ImageOps

    def crop(img):
        s = int(round(size / 0.875))
        scale = s / min(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC)
        return ImageOps.fit(img, (size, size), Image.BICUBIC)

    def letterbox(img):
        r = size / max(img.size)
        im = img.resize((max(1, round(img.width * r)), max(1, round(img.height * r))), Image.BICUBIC)
        canvas = Image.new("RGB", (size, size), tuple(int(v) for v in np.asarray(im).reshape(-1, 3).mean(0)))
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        return canvas

    return {"crop": crop, "letterbox": letterbox, "resize": lambda im: im.resize((size, size), Image.BICUBIC)}[kind]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--key", default="file")
    ap.add_argument("--label", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--eval-part", default="", help="'holdout' to train on the whole train part and validate on the holdout")
    ap.add_argument("--model", default="resnet18.a1_in1k")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--opt", default="adamw")
    ap.add_argument("--aug", default="light", choices=["none", "light", "randaugment", "trivialaugment", "augmix"])
    ap.add_argument("--mix", type=float, default=0.0, help="Mixup/CutMix probability (0 = off)")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--layer-decay", type=float, default=0.0)
    ap.add_argument("--drop-path", type=float, default=0.0)
    ap.add_argument("--freeze-epochs", type=int, default=0)
    ap.add_argument("--lp-init", action="store_true", help="LP-FT: initialise the head from a logistic-regression probe")
    ap.add_argument("--lp-c", type=float, default=1.0, help="inverse regularisation of the --lp-init probe")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--eval-transform", default="crop", choices=["crop", "letterbox", "resize"])
    ap.add_argument("--max-train", type=int, default=0, help="subsample the training images (screening on a budget)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="run")
    a = ap.parse_args()

    import timm
    import torch
    import torchvision.transforms.v2 as T
    from sklearn.metrics import f1_score
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy
    from timm.scheduler import CosineLRScheduler
    from timm.optim import create_optimizer_v2
    from timm.utils import ModelEmaV3

    from cv_folds import load_split
    from embed_images import load_rgb

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(a.meta)
    meta[a.key] = meta[a.key].astype(str)
    s = load_split(Path(a.split) / "split.csv", ids=meta[a.key])
    if a.eval_part == "holdout":
        tr_df, va_df = meta[s.train_mask], meta[s.holdout_mask]
    else:
        pool = meta[s.train_mask].reset_index(drop=True)
        ti, vi = s.folds[a.fold]
        tr_df, va_df = pool.iloc[ti], pool.iloc[vi]
    if a.max_train and len(tr_df) > a.max_train:
        tr_df = tr_df.groupby(a.label, group_keys=False).sample(frac=a.max_train / len(tr_df), random_state=a.seed)
    classes = sorted(meta[a.label].astype(str).unique())
    cidx = {c: i for i, c in enumerate(classes)}

    try:  # ViTs (e.g. DINOv2 defaults to 518 px) need the input size at creation; CNNs do not accept it
        model = timm.create_model(a.model, pretrained=bool(a.pretrained), num_classes=len(classes),
                                  drop_path_rate=a.drop_path, img_size=a.img_size)
    except TypeError:
        model = timm.create_model(a.model, pretrained=bool(a.pretrained), num_classes=len(classes), drop_path_rate=a.drop_path)
    cfg = timm.data.resolve_data_config({}, model=model)
    norm = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(cfg["mean"], cfg["std"])]
    policy = {"none": [], "light": [T.ColorJitter(0.2, 0.2, 0.2)], "randaugment": [T.RandAugment()],
              "trivialaugment": [T.TrivialAugmentWide()], "augmix": [T.AugMix()]}[a.aug]
    train_tf = T.Compose([T.RandomResizedCrop(a.img_size, scale=(0.6, 1.0), antialias=True), T.RandomHorizontalFlip(), *policy, *norm])
    eval_fn = eval_resize(a.eval_transform, a.img_size)
    eval_tf = T.Compose(norm)

    class DS(torch.utils.data.Dataset):
        def __init__(self, df, train):
            self.df, self.train = df.reset_index(drop=True), train
            self.ok = []
            for i, f in enumerate(self.df[a.key]):
                try:
                    load_rgb(Path(a.image_dir) / f)
                    self.ok.append(i)
                except Exception:  # noqa: BLE001 - unreadable files were excluded at Stage 1
                    pass

        def __len__(self):
            return len(self.ok)

        def __getitem__(self, j):
            r = self.df.iloc[self.ok[j]]
            img = load_rgb(Path(a.image_dir) / r[a.key])
            x = train_tf(img) if self.train else eval_tf(eval_fn(img))
            return x, cidx[str(r[a.label])]

    tr_ds, va_ds = DS(tr_df, True), DS(va_df, False)
    dl = torch.utils.data.DataLoader(tr_ds, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    vl = torch.utils.data.DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0)

    opt = create_optimizer_v2(model, a.opt, lr=a.lr, weight_decay=a.wd,
                                         layer_decay=a.layer_decay if a.layer_decay else None)
    steps = max(1, len(dl))
    sched = CosineLRScheduler(opt, t_initial=a.epochs * steps, warmup_t=steps, warmup_lr_init=a.lr * 0.01, t_in_epochs=False)
    ema = ModelEmaV3(model, decay=a.ema, use_warmup=True) if a.ema else None  # decay ramps up: short runs work
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=a.mix, label_smoothing=a.label_smoothing,
                  num_classes=len(classes)) if a.mix > 0 else None
    weights = None
    if a.class_weights:
        counts = np.bincount([cidx[str(v)] for v in tr_df[a.label]], minlength=len(classes)).astype(float)
        weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)
    hard_loss = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=a.label_smoothing)
    soft_loss = SoftTargetCrossEntropy()

    def evaluate(m):
        m.eval()
        probs, ys = [], []
        with torch.inference_mode():
            for x, y in vl:
                probs.append(torch.softmax(m(x), 1).numpy())
                ys.append(y.numpy())
        p, yy = np.concatenate(probs), np.concatenate(ys)
        pred = p.argmax(1)
        return {"acc": float((pred == yy).mean()), "macro_f1": float(f1_score(yy, pred, average="macro"))}, p, yy

    lp_record = None
    if a.lp_init:
        from sklearn.linear_model import LogisticRegression
        head = model.get_classifier()
        if not isinstance(head, torch.nn.Linear):
            raise SystemExit(f"--lp-init needs a linear head; {a.model} has {type(head).__name__}")
        t0 = time.time()
        model.eval()
        feats, ys = [], []
        fl = torch.utils.data.DataLoader(DS(tr_df, False), batch_size=64, shuffle=False, num_workers=0)
        with torch.inference_mode():
            for x, y in fl:
                feats.append(model.forward_head(model.forward_features(x), pre_logits=True).numpy())
                ys.append(y.numpy())
        X, Y = np.concatenate(feats), np.concatenate(ys)
        clf = LogisticRegression(C=a.lp_c, max_iter=3000).fit(X, Y)
        W = np.zeros((len(classes), X.shape[1]), dtype=np.float32)
        b = np.zeros(len(classes), dtype=np.float32)
        if clf.coef_.shape[0] == 1:  # binary: sklearn keeps one row for the positive class
            W[clf.classes_[1]], b[clf.classes_[1]] = clf.coef_[0], clf.intercept_[0]
        else:
            W[clf.classes_], b[clf.classes_] = clf.coef_, clf.intercept_
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(W))
            head.bias.copy_(torch.from_numpy(b))
        res0, _, _ = evaluate(model)
        lp_record = {"lp_probe": {k: round(v, 4) for k, v in res0.items()}, "lp_s": round(time.time() - t0, 1)}
        print(json.dumps(lp_record), flush=True)

    hist, best, t_start = [], None, time.time()
    step = 0
    for ep in range(a.epochs):
        freeze = ep < a.freeze_epochs
        for n_, p_ in model.named_parameters():
            p_.requires_grad = (not freeze) or n_.startswith(model.default_cfg.get("classifier", "head"))
        model.train()
        t0, losses = time.time(), []
        for x, y in dl:
            if mixup is not None:
                x, yt = mixup(x, y)
                loss = soft_loss(model(x), yt)
            else:
                loss = hard_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            sched.step_update(step)
            if ema is not None:
                ema.update(model, step=step)
            losses.append(loss.item())
        res, p, yy = evaluate(model)
        rec = {"epoch": ep + 1, "train_loss": round(float(np.mean(losses)), 4), **{k: round(v, 4) for k, v in res.items()},
               "epoch_s": round(time.time() - t0, 1)}
        cand = [("model", res, p, model)]
        if ema is not None:
            res_e, p_e, _ = evaluate(ema.module)
            rec.update({f"ema_{k}": round(v, 4) for k, v in res_e.items()})
            cand.append(("ema", res_e, p_e, ema.module))
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        for which, r_, p_, m_ in cand:
            if best is None or r_["macro_f1"] > best["macro_f1"]:
                best = {"macro_f1": r_["macro_f1"], "acc": r_["acc"], "epoch": ep + 1, "weights": which}
                files = va_ds.df.iloc[va_ds.ok][a.key].to_numpy()
                np.savez(out / "val_probs.npz", files=files.astype(str), labels=yy, probs=p_.astype(np.float32),
                         classes=np.array(classes))
                torch.save({"state_dict": m_.state_dict(), "classes": classes, "config": vars(a)}, out / "model.pt")
    summary = {"best": best, "history": hist, "train_images": len(tr_ds), "val_images": len(va_ds),
               "minutes": round((time.time() - t_start) / 60, 2), "config": vars(a), "lp_init": lp_record}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(f"best: {best} | train {len(tr_ds)} / val {len(va_ds)} images | {summary['minutes']} min")


if __name__ == "__main__":
    main()
