"""Frozen-backbone linear probe as a SHIPPABLE model: the first rung of the adaptation ladder, and on CPU often the
best one. Chooses C on the Stage 2 folds, refits on all development images, saves a model that loads safely.

Usage:
    python train_probe.py IMAGE_DIR --meta metadata_grouped.csv --key file --label label --split split/ \
        [--model vit_small_patch14_dinov2.lvd142m] [--img-size 224] [--variant crop|resize|letterbox]
        [--preprocess my_prep.py:func] [--C 0.1,0.3,1,3] [--class-weight balanced] [--threads 4] [--out runs/probe]

What it does
  1. Embeds the train part (and holdout) once with the frozen timm backbone, L2-normalised, cached in --out.
     --preprocess FILE:FUNC applies your own PIL -> PIL step first (e.g. a fixed border crop that removes a
     shortcut region); the SAME function is recorded and used at prediction time.
  2. For each C: logistic regression per fold, macro-F1 on the fold's validation images. Writes
     --out/C<c>/f<k>_s0/metrics.json and val_probs.npz per fold, so cv-method-comparison/scripts/compare.py
     can rank the C values and ensemble.py can combine this probe with other candidates out of fold.
  3. Picks the C with the best mean fold score (ties within one fold-SD go to the smaller C), reports the
     holdout score of a model fitted on the train part, then REFITS on train + holdout (all development
     images) and saves --out/model.pt: backbone name, image size, variant, preprocess spec, classes, the linear
     head as tensors. torch.load(..., weights_only=True) reads it; nothing is pickled.
Prediction / export
    from train_probe import load_probe
    model, preprocess, classes = load_probe("runs/probe/model.pt")   # model: (N,3,H,W) -> logits
    python ../../cv-evaluation/scripts/export_check.py --module train_probe.py:load_probe --module-arg runs/probe/model.pt ...
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "cv-data-audit" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "cv-validation-design" / "scripts"))
sys.path.insert(0, str(HERE))


def _load_func(spec: str):
    path, name = spec.rsplit(":", 1)
    s = importlib.util.spec_from_file_location("probe_preprocess", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return getattr(m, name)


def _backbone(name: str, size: int):
    import timm
    try:
        return timm.create_model(name, pretrained=True, num_classes=0, img_size=size).eval()
    except TypeError:
        return timm.create_model(name, pretrained=True, num_classes=0).eval()


def _pipeline(backbone, size: int, variant: str, prep_spec: str):
    """Returns f(path) -> (3,S,S) tensor: decode (EXIF, RGB) -> optional custom step -> resize policy -> normalise."""
    import timm
    import torch
    import torchvision.transforms.v2 as T

    from embed_images import load_rgb
    from train_classifier import eval_resize
    cfg = timm.data.resolve_data_config({}, model=backbone)
    norm = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(cfg["mean"], cfg["std"])])
    resize = eval_resize(variant, size)
    custom = _load_func(prep_spec) if prep_spec else None

    def f(path):
        img = load_rgb(Path(path))
        if custom is not None:
            img = custom(img)
        return norm(resize(img))
    return f


def load_probe(model_path: str):
    """(model, preprocess, classes) for predict / export_check --module ... --module-arg <model.pt>."""
    import torch

    ck = torch.load(model_path, map_location="cpu", weights_only=True)
    backbone = _backbone(ck["model"], ck["img_size"])

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = torch.nn.Linear(ck["W"].shape[1], ck["W"].shape[0])
            with torch.no_grad():
                self.head.weight.copy_(ck["W"])
                self.head.bias.copy_(ck["b"])

        def forward(self, x):
            return self.head(torch.nn.functional.normalize(self.backbone(x), dim=1) * ck["scale"])

    m = Probe().eval()
    return m, _pipeline(backbone, ck["img_size"], ck["variant"], ck["preprocess"]), list(ck["classes"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--key", default="file")
    ap.add_argument("--label", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--variant", default="crop", choices=["crop", "resize", "letterbox"])
    ap.add_argument("--preprocess", default="", help="FILE:FUNC, a PIL -> PIL step applied before resizing (train and serve)")
    ap.add_argument("--C", default="0.1,0.3,1,3")
    ap.add_argument("--class-weight", default="", choices=["", "balanced"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="runs/probe")
    a = ap.parse_args()

    import pandas as pd
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    from cv_folds import load_split
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(a.meta)
    meta[a.key] = meta[a.key].astype(str)
    s = load_split(Path(a.split) / "split.csv", ids=meta[a.key])
    classes = sorted(meta[a.label].astype(str).unique())
    cidx = {c: i for i, c in enumerate(classes)}
    backbone = _backbone(a.model, a.img_size)
    prep = _pipeline(backbone, a.img_size, a.variant, a.preprocess)

    def embed(frame, tag):
        cache = out / f"emb_{tag}.npz"
        files = frame[a.key].tolist()
        if cache.exists():
            z = np.load(cache)
            if list(z["files"].astype(str)) == files:
                return z["X"], z["ok"]
        X, ok, batch, t0 = [], [], [], time.time()
        with torch.inference_mode():
            for i, f in enumerate(files):
                try:
                    batch.append(prep(Path(a.image_dir) / f))
                    ok.append(i)
                except Exception:  # noqa: BLE001 - unreadable files were excluded at Stage 1
                    pass
                if len(batch) == a.batch or (i == len(files) - 1 and batch):
                    X.append(torch.nn.functional.normalize(backbone(torch.stack(batch)), dim=1).numpy())
                    batch = []
        X, ok = np.concatenate(X), np.array(ok)
        np.savez(cache, X=X, ok=ok, files=np.array(files))
        print(f"embedded {len(ok)}/{len(files)} {tag} images in {time.time() - t0:.0f}s", flush=True)
        return X, ok

    tr = meta[s.train_mask].reset_index(drop=True)
    X, ok = embed(tr, "train")
    y = tr[a.label].astype(str).map(cidx).to_numpy()
    pos = {i: j for j, i in enumerate(ok)}
    scale = 10.0
    cs = [float(c) for c in a.C.split(",")]
    cw = a.class_weight or None
    table = {}
    for C in cs:
        scores = []
        for k, (ti, vi) in enumerate(s.folds):
            ti_ = [pos[i] for i in ti if i in pos]
            vi_ = [i for i in vi if i in pos]
            clf = LogisticRegression(C=C, max_iter=3000, class_weight=cw).fit(X[ti_] * scale, y[ok][ti_])
            P = np.zeros((len(vi_), len(classes)), dtype=np.float32)
            P[:, clf.classes_] = clf.predict_proba(X[[pos[i] for i in vi_]] * scale)
            f1 = float(f1_score(y[vi_], P.argmax(1), average="macro"))
            acc = float((P.argmax(1) == y[vi_]).mean())
            scores.append(f1)
            rd = out / f"C{C:g}" / f"f{k}_s0"
            rd.mkdir(parents=True, exist_ok=True)
            np.savez(rd / "val_probs.npz", files=tr[a.key].to_numpy()[vi_].astype(str), labels=y[vi_], probs=P,
                     classes=np.array(classes))
            (rd / "metrics.json").write_text(json.dumps({"history": [{"epoch": 1, "macro_f1": f1, "acc": acc}],
                                                         "best": {"macro_f1": f1, "acc": acc, "epoch": 1},
                                                         "train_images": len(ti_), "val_images": len(vi_), "minutes": 0,
                                                         "config": {"fold": k, "seed": 0, "C": C, "model": a.model}}))
        table[C] = (float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0)
        print(f"C={C:g}: macro-F1 {table[C][0]:.4f} +/- {table[C][1]:.4f} over {len(scores)} folds", flush=True)
    best = max(table, key=lambda c: table[c][0])
    tied = [c for c in cs if table[c][0] >= table[best][0] - table[best][1]]
    chosen = min(tied)
    print(f"chosen C={chosen:g} (best {best:g}; smallest C within one fold-SD)")
    report = {"model": a.model, "img_size": a.img_size, "variant": a.variant, "preprocess": a.preprocess,
              "C_table": {f"{c:g}": table[c] for c in cs}, "chosen_C": chosen}
    fit_idx = [pos[i] for i in range(len(tr)) if i in pos]
    Xall, yall = X[fit_idx], y[ok]
    if s.holdout_mask.any():
        ho = meta[s.holdout_mask].reset_index(drop=True)
        Xh, okh = embed(ho, "holdout")
        yh = ho[a.label].astype(str).map(cidx).to_numpy()[okh]
        clf = LogisticRegression(C=chosen, max_iter=3000, class_weight=cw).fit(Xall * scale, yall)
        ph = clf.predict(Xh * scale)
        report["holdout"] = {"macro_f1": round(float(f1_score(yh, ph, average="macro")), 4),
                             "accuracy": round(float((ph == yh).mean()), 4), "n": int(len(yh))}
        print(f"holdout (model fitted on the train part): {report['holdout']}")
        Xall, yall = np.concatenate([Xall, Xh]), np.concatenate([yall, yh])
    clf = LogisticRegression(C=chosen, max_iter=3000, class_weight=cw).fit(Xall * scale, yall)
    W = np.zeros((len(classes), X.shape[1]), dtype=np.float32)
    b = np.zeros(len(classes), dtype=np.float32)
    if clf.coef_.shape[0] == 1:
        W[clf.classes_[1]], b[clf.classes_[1]] = clf.coef_[0], clf.intercept_[0]
    else:
        W[clf.classes_], b[clf.classes_] = clf.coef_, clf.intercept_
    torch.save({"model": a.model, "img_size": a.img_size, "variant": a.variant, "preprocess": a.preprocess,
                "classes": classes, "W": torch.from_numpy(W), "b": torch.from_numpy(b), "scale": scale, "C": chosen,
                "n_fit": int(len(yall))}, out / "model.pt")
    report["refit_images"] = int(len(yall))
    (out / "probe_report.json").write_text(json.dumps(report, indent=1))
    print(f"refit on {len(yall)} development images -> {out / 'model.pt'}")


if __name__ == "__main__":
    main()
