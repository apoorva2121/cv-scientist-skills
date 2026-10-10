"""Grader-side reference floors for the real-image evals (never shown to the agents under test).

Oxford-IIIT Pet: DINOv2 ViT-S/14 (Apache-2.0) frozen CLS features at 224 px (shorter side 256, centre crop),
logistic regression with C chosen by 5-fold CV on the pool, trained on the whole pool, scored ONCE on the
fixed benchmark. Writes evals/answer_keys/real_references.json with accuracy, macro-F1 and a Wilson 95%
interval for accuracy.

    python evals/compute_real_references.py [--threads 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cv-data-audit" / "scripts"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    a = ap.parse_args()

    import pandas as pd
    import timm
    import torch
    from PIL import Image, ImageOps
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from torchvision.transforms import v2 as T

    from embed_images import load_rgb

    if a.threads:
        torch.set_num_threads(a.threads)
    model = timm.create_model(a.model, pretrained=True, num_classes=0, img_size=224).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    norm = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(cfg["mean"], cfg["std"])])

    def prep(img):
        s = 256 / min(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
        return norm(ImageOps.fit(img, (224, 224), Image.BICUBIC))

    def embed(img_dir: Path, files: list[str]) -> np.ndarray:
        out, batch, t0 = [], [], time.time()
        with torch.inference_mode():
            for i, f in enumerate(files):
                batch.append(prep(load_rgb(img_dir / f)))
                if len(batch) == 64 or i == len(files) - 1:
                    out.append(torch.nn.functional.normalize(model(torch.stack(batch)), dim=1).numpy())
                    batch = []
                if (i + 1) % 640 == 0:
                    print(f"  {i + 1}/{len(files)} images, {time.time() - t0:.0f}s", flush=True)
        return np.concatenate(out)

    base = ROOT / "evals" / "real_prepared" / "oxford_pets"
    pool = pd.read_csv(base / "pool" / "labels.csv")
    bench = pd.read_csv(base / "benchmark" / "labels.csv")
    print(f"pool {len(pool)} / benchmark {len(bench)} images; model {a.model}", flush=True)
    Xp = embed(base / "pool" / "images", list(pool.file)) * 10
    Xb = embed(base / "benchmark" / "images", list(bench.file)) * 10
    yp, yb = pool.breed.astype(str).to_numpy(), bench.breed.astype(str).to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = {c: float(cross_val_score(LogisticRegression(C=c, max_iter=3000), Xp, yp, cv=cv).mean()) for c in (0.1, 0.3, 1.0, 3.0)}
    c_best = max(scores, key=scores.get)
    clf = LogisticRegression(C=c_best, max_iter=3000).fit(Xp, yp)
    pred = clf.predict(Xb)
    k, n = int((pred == yb).sum()), len(yb)
    lo, hi = wilson(k, n)
    ref = {"oxford_pets": {
        "reference": f"{a.model} frozen CLS features, 224 px centre crop, logistic regression (C={c_best} by 5-fold CV on pool)",
        "pool_cv_accuracy_by_C": {str(c): round(v, 4) for c, v in scores.items()},
        "benchmark_accuracy": round(k / n, 4), "benchmark_accuracy_wilson95": [round(lo, 4), round(hi, 4)],
        "benchmark_macro_f1": round(float(f1_score(yb, pred, average="macro")), 4), "n_benchmark": n,
        "computed": time.strftime("%Y-%m-%d")}}
    outp = ROOT / "evals" / "answer_keys" / "real_references.json"
    old = json.loads(outp.read_text()) if outp.exists() else {}
    old.update(ref)
    outp.write_text(json.dumps(old, indent=1))
    print(json.dumps(ref, indent=1))


if __name__ == "__main__":
    main()
