"""Prove that the serving path produces the same input tensors as the evaluation path.

Usage:
    python serve_parity.py IMAGE_DIR --eval-module my_pipeline.py:eval_transform --serve-module serve.py:preprocess
        [--n 50] [--atol 1e-3]

Both callables take a file path and return a CHW float array (after normalisation) or a tensor.
The script runs both on N raw files (including awkward ones if present: EXIF-rotated, CMYK, RGBA,
grayscale, 16-bit) and reports the maximum absolute difference per file. Any file above --atol fails:
typical causes are a different resize library or interpolation, a missing antialias flag, EXIF handled
on one side only, RGB/BGR order, or different mean/std.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_callable(spec: str):
    path, name = spec.rsplit(":", 1)
    s = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(s)
    sys.path.insert(0, str(Path(path).resolve().parent))
    s.loader.exec_module(mod)
    return getattr(mod, name)


def to_np(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--eval-module", required=True)
    ap.add_argument("--serve-module", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--atol", type=float, default=1e-3)
    a = ap.parse_args()

    ev, sv = load_callable(a.eval_module), load_callable(a.serve_module)
    files = sorted(p for p in Path(a.image_dir).rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("._") and "__MACOSX" not in p.parts)[: a.n]
    worst, fails, both_bad = 0.0, [], []
    for f in files:
        res = []
        for fn in (ev, sv):
            try:
                res.append(to_np(fn(str(f))))
            except Exception as e:  # noqa: BLE001
                res.append(f"{type(e).__name__}: {str(e)[:80]}")
        if isinstance(res[0], str) and isinstance(res[1], str):
            both_bad.append(f.name)  # unreadable for both paths: a data problem (Stage 1), not a parity failure
            continue
        if isinstance(res[0], str) or isinstance(res[1], str):
            side = "evaluation" if isinstance(res[0], str) else "serving"
            fails.append((f.name, f"only the {side} path failed: {res[0] if side == 'evaluation' else res[1]}"))
            continue
        x, y = res
        if x.shape != y.shape:
            fails.append((f.name, f"shape {x.shape} vs {y.shape}"))
            continue
        d = float(np.abs(x - y).max())
        worst = max(worst, d)
        if d > a.atol:
            fails.append((f.name, f"max abs diff {d:.4g}"))
    print(f"files {len(files)}; worst max-abs difference {worst:.4g}; failures {len(fails)} (atol {a.atol})")
    if both_bad:
        print(f"  {len(both_bad)} file(s) unreadable in both paths (excluded; remove them at Stage 1): {both_bad[:5]}")
    for name, why in fails[:20]:
        print(f"  FAIL {name}: {why}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
