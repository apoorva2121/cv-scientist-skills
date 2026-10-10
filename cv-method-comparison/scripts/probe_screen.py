"""Cheap proxy screen: compare preprocessing variants with a linear probe on frozen features, same folds.

Usage:
    python probe_screen.py IMAGE_DIR --meta metadata.csv --key file --label label --split split/
        --variants resize224,crop224,letterbox224,resize320,clahe224 [--model vit_small_patch14_dinov2.lvd142m]
        [--custom my_variants.py] [--cache cache_dir] [--metric macro_f1|accuracy] [--out probe_screen.csv]
        [--max-train 1500] [--threads 4] [--C 0.3,1,3]

Each variant is a function PIL image -> PIL image of the final input size. Built-ins (S = size in px):
  resizeS       stretch to S x S (aspect ratio not kept)
  cropS         resize the shorter side to S/0.875, centre-crop S x S (the ImageNet evaluation recipe)
  letterboxS    resize the longer side to S, pad to S x S with the image mean (aspect ratio kept)
  claheS        CLAHE on the L channel of Lab (OpenCV), then cropS
  grayS         grayscale replicated to 3 channels, then cropS (tests whether colour matters)
  maskborderS   cropS with an 8% border filled with the mean (tests reliance on borders)
--custom FILE may define VARIANTS = {"name": callable} for anything else (deskew, stain normalisation,
band selection, denoising...). Embeddings are cached per variant, model and subset.
On CPU, embedding costs roughly 0.1 s per image per variant at 224 px (ViT-S/14); --max-train screens on
a fixed random subset of the train part (folds keep their groups), and --threads caps CPU use.

For every variant: embed the train part, fit a logistic-regression probe on each fold's training images,
score the fold's validation images, then compare variants with paired, overlap-corrected fold statistics
(compare.summarize_folds). A frozen-feature probe is a PROXY: it ranks preprocessing that changes what
the backbone can see (resolution, crop, colour); effects that only appear under fine-tuning need the
short fine-tune proxy in cv-training.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "cv-validation-design" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "cv-data-audit" / "scripts"))


def builtin(name: str):
    from PIL import Image, ImageOps

    kind = "".join(c for c in name if not c.isdigit())
    S = int("".join(c for c in name if c.isdigit()) or 224)

    def crop(img):
        s = int(round(S / 0.875))
        return ImageOps.fit(img.resize((max(1, round(img.width * s / min(img.size))), max(1, round(img.height * s / min(img.size)))),
                                       Image.BICUBIC), (S, S), Image.BICUBIC)

    if kind == "resize":
        return lambda img: img.resize((S, S), Image.BICUBIC)
    if kind == "crop":
        return crop
    if kind == "letterbox":
        def lb(img):
            r = S / max(img.size)
            im = img.resize((max(1, round(img.width * r)), max(1, round(img.height * r))), Image.BICUBIC)
            fill = tuple(int(v) for v in np.asarray(im).reshape(-1, 3).mean(0))
            canvas = Image.new("RGB", (S, S), fill)
            canvas.paste(im, ((S - im.width) // 2, (S - im.height) // 2))
            return canvas
        return lb
    if kind == "clahe":
        def cl(img):
            import cv2
            lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
            lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
            return crop(Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)))
        return cl
    if kind == "gray":
        return lambda img: crop(ImageOps.grayscale(img).convert("RGB"))
    if kind == "maskborder":
        def mb(img):
            a = np.asarray(crop(img)).copy()
            b = max(1, int(0.08 * S))
            fill = a.reshape(-1, 3).mean(0).astype(a.dtype)
            a[:b], a[-b:], a[:, :b], a[:, -b:] = fill, fill, fill, fill
            return Image.fromarray(a)
        return mb
    raise ValueError(f"unknown variant {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--key", default="file")
    ap.add_argument("--label", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--variants", default="crop224,resize224,letterbox224")
    ap.add_argument("--custom", default="")
    ap.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--metric", default="macro_f1", choices=["macro_f1", "accuracy"])
    ap.add_argument("--cache", default="probe_cache")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="probe_screen.csv")
    ap.add_argument("--max-train", type=int, default=0, help="embed a fixed random subset of the train part")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--C", default="1", help="comma-separated logistic-regression C values; each (variant, C) is a candidate")
    a = ap.parse_args()

    import timm
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from torchvision.transforms import v2 as T

    from compare import summarize_folds
    from cv_folds import load_split
    from embed_images import load_rgb

    meta = pd.read_csv(a.meta)
    meta[a.key] = meta[a.key].astype(str)
    s = load_split(Path(a.split) / "split.csv", ids=meta[a.key])
    tr = meta[s.train_mask].reset_index(drop=True)
    if a.threads:
        torch.set_num_threads(a.threads)
    use = np.arange(len(tr))
    if a.max_train and a.max_train < len(tr):
        use = np.sort(np.random.default_rng(0).choice(len(tr), a.max_train, replace=False))
    tag = f"{a.model.replace('/', '_')}_{len(use)}"
    custom = {}
    if a.custom:
        spec = importlib.util.spec_from_file_location("custom_variants", a.custom)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        custom = getattr(mod, "VARIANTS", {})
    names = [v.strip() for v in a.variants.split(",") if v.strip()] + list(custom)
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    model = None
    rows = []
    for name in names:
        fn = custom.get(name) or builtin(name)
        cache = Path(a.cache) / f"{name}__{tag}.npz"
        if cache.exists():
            z = np.load(cache)  # numeric arrays only; pickled objects are never loaded
            X, ok = z["X"], z["ok"]
        else:
            size = int("".join(c for c in name if c.isdigit()) or 224)
            if model is None or getattr(model, "_size", None) != size:
                try:
                    model = timm.create_model(a.model, pretrained=True, num_classes=0, img_size=size).eval()
                except TypeError:
                    model = timm.create_model(a.model, pretrained=True, num_classes=0).eval()
                model._size = size
                cfg = timm.data.resolve_data_config({}, model=model)
                norm = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(cfg["mean"], cfg["std"])])
            t0 = time.time()
            feats, ok = [], []
            batch = []
            with torch.inference_mode():
                for j, i in enumerate(use):
                    try:
                        batch.append(norm(fn(load_rgb(Path(a.image_dir) / tr[a.key].iloc[i]))))
                        ok.append(int(i))
                    except Exception:  # noqa: BLE001 - unreadable files were logged at Stage 1
                        pass
                    if len(batch) == a.batch or (j == len(use) - 1 and batch):
                        feats.append(torch.nn.functional.normalize(model(torch.stack(batch)), dim=1).numpy())
                        batch = []
            X, ok = np.concatenate(feats), np.array(ok)
            np.savez(cache, X=X, ok=ok)
            print(f"{name}: embedded {len(ok)} images in {time.time() - t0:.0f}s", flush=True)
        y = tr[a.label].astype(str).to_numpy()
        pos = {i: j for j, i in enumerate(ok)}
        cs = [float(c) for c in str(a.C).split(",")]
        for C in cs:
            cand = name if len(cs) == 1 else f"{name}_C{C:g}"
            for k, (ti, vi) in enumerate(s.folds):
                ti_ = [pos[i] for i in ti if i in pos]
                vi_ok = [i for i in vi if i in pos]
                clf = LogisticRegression(C=C, max_iter=3000).fit(X[ti_] * 10, y[ok][ti_])
                pred = clf.predict(X[[pos[i] for i in vi_ok]] * 10)
                truth = y[vi_ok]
                score = f1_score(truth, pred, average="macro") if a.metric == "macro_f1" else float((truth == pred).mean())
                rows.append({"name": cand, "fold": k, "seed": 0, "score": score, "n_train": len(ti_), "n_val": len(vi_ok)})
    long = pd.DataFrame(rows)
    long.to_csv(Path(a.out).with_name(Path(a.out).stem + "_folds.csv"), index=False)
    table = summarize_folds(long)
    table.to_csv(a.out, index=False)
    print(f"\nproxy: linear probe on {a.model} features; metric {a.metric}; {len(s.folds)} folds")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
