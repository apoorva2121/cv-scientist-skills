"""Compute foundation-model embeddings for a set of images (shared by audit, split, shift and proxy steps).

Usage:
    python embed_images.py IMAGE_DIR [--list files.csv --col file] [--model vit_small_patch14_dinov2.lvd142m]
        [--batch 64] [--threads 0] [--out embeddings.npz]

Writes an .npz with `files` (names, in order), `emb` (float32, L2-normalised, one row per readable image)
and `failed` (names that could not be decoded). Images are decoded exactly as training should:
EXIF orientation applied, converted to RGB (CMYK, palette, gray and RGBA handled; alpha composited on
white). The model's own resize and normalisation are used (timm data config).

The default model downloads its weights on first use (about 90 MB for ViT-S/14 DINOv2). Check the
model's licence in cv-model-selection before using embeddings in a product.
Also importable: load_rgb(path) -> PIL RGB image with orientation fixed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(bg, rgba).convert("RGB")
        return im.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--list", default="")
    ap.add_argument("--col", default="file")
    ap.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--img-size", type=int, default=224,
                    help="input size; DINOv2 models default to 518 px in timm, about 25x slower on CPU than 224")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="embeddings.npz")
    a = ap.parse_args()

    import pandas as pd
    import timm
    import torch

    if a.threads:
        torch.set_num_threads(a.threads)
    root = Path(a.image_dir)
    names = list(pd.read_csv(a.list)[a.col]) if a.list else sorted(p.name for p in root.iterdir() if p.suffix.lower() in EXTS and not p.name.startswith("._"))
    kwargs = {"img_size": a.img_size} if a.img_size else {}
    try:
        model = timm.create_model(a.model, pretrained=True, num_classes=0, **kwargs).eval()
    except TypeError:  # CNNs take any input size and have no img_size argument
        model = timm.create_model(a.model, pretrained=True, num_classes=0).eval()
    cfg = timm.data.resolve_data_config({"input_size": (3, a.img_size, a.img_size)} if a.img_size else {}, model=model)
    tf = timm.data.create_transform(**cfg)
    print(f"model {a.model}: input {cfg['input_size']}, mean {cfg['mean']}, std {cfg['std']}", flush=True)

    kept, failed, chunks = [], [], []
    t0 = time.time()
    batch_x, batch_n = [], []
    with torch.inference_mode():
        for k, n in enumerate(names):
            try:
                batch_x.append(tf(load_rgb(root / n)))
                batch_n.append(n)
            except Exception:  # noqa: BLE001 - unreadable files are reported, not fatal
                failed.append(n)
            if len(batch_x) == a.batch or (k == len(names) - 1 and batch_x):
                f = model(torch.stack(batch_x))
                chunks.append(torch.nn.functional.normalize(f, dim=1).numpy().astype(np.float32))
                kept += batch_n
                batch_x, batch_n = [], []
                if len(chunks) % 10 == 0:
                    print(f"  {k + 1}/{len(names)} ({len(kept) / (time.time() - t0):.1f} img/s)", flush=True)
    emb = np.concatenate(chunks) if chunks else np.zeros((0, 0), np.float32)
    np.savez(a.out, files=np.array(kept), emb=emb, failed=np.array(failed), model=a.model)
    print(f"embedded {len(kept)} images ({emb.shape[1] if emb.size else 0}-d) in {time.time() - t0:.0f}s; "
          f"failed {len(failed)}; written {a.out}")


if __name__ == "__main__":
    main()
