"""Stage 1 image audit: integrity, format, colour mode, orientation, size, quality statistics, hashes.

Usage:
    python image_audit.py IMAGE_DIR [--list files.csv --col file] [--workers 8] [--out image_audit.csv]

One row per file. Nothing is modified. Columns:
  readable, error            full decode succeeded (truncated files fail here, not at open())
  format, ext_mismatch       real format from the file's content vs its extension
  mode, bands, bit_depth     PIL mode (L, RGB, RGBA, CMYK, P, I;16, F ...)
  exif_orientation           EXIF tag 274; any value other than 1 means pixels are stored rotated or mirrored
  width, height, aspect      after applying EXIF orientation
  n_frames                   >1 for animated GIF/WebP or multi-page TIFF
  has_alpha_transparency     alpha channel with any pixel below 255
  mean, std                  brightness and contrast on 0-255 grayscale
  blur_var                   variance of a Laplacian filter (low = blurry); compare within a dataset only
  clipped_dark, clipped_bright  share of pixels at 0 or 255
  border_uniform             the outer 3% of the image is near-constant (frames, scanner borders, letterboxing)
  md5, phash, dhash          exact and perceptual hashes for find_duplicates.py
Summary flags are printed at the end.
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
OS_JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}


def is_os_junk(p: Path) -> bool:
    """macOS AppleDouble '._name' files and __MACOSX folders (archive side-files named like images),
    Finder/Explorer metadata. They are not images and not corrupt data: exclude and report them."""
    return p.name.startswith("._") or p.name.lower() in OS_JUNK_NAMES or "__MACOSX" in p.parts
EXT_FOR = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}, "BMP": {".bmp"}, "TIFF": {".tif", ".tiff"},
           "WEBP": {".webp"}, "GIF": {".gif"}, "MPO": {".jpg", ".jpeg"}}


def audit_one(path: str) -> dict:
    from PIL import Image, ImageOps
    import imagehash

    p = Path(path).resolve()
    row = {"file": p.name, "path": str(p), "bytes": p.stat().st_size, "readable": False, "error": ""}
    data = p.read_bytes()
    row["md5"] = hashlib.md5(data).hexdigest()
    try:
        with Image.open(p) as im:
            row["format"] = im.format
            row["ext_mismatch"] = p.suffix.lower() not in EXT_FOR.get(im.format, {p.suffix.lower()})
            row["mode"] = im.mode
            row["bands"] = len(im.getbands())
            row["bit_depth"] = {"I;16": 16, "I;16B": 16, "I": 32, "F": 32, "1": 1}.get(im.mode, 8)
            row["exif_orientation"] = int(im.getexif().get(0x0112, 1) or 1)
            row["n_frames"] = int(getattr(im, "n_frames", 1))
            im.load()  # forces full decode: truncated files raise here
            upright = ImageOps.exif_transpose(im)
            row["width"], row["height"] = upright.size
            row["aspect"] = round(upright.size[0] / upright.size[1], 3)
            alpha = None
            if upright.mode in ("RGBA", "LA") or (upright.mode == "P" and "transparency" in upright.info):
                alpha = np.asarray(upright.convert("RGBA"))[..., 3]
            row["has_alpha_transparency"] = bool(alpha is not None and (alpha < 255).any())
            if upright.mode in ("I;16", "I;16B", "I", "F"):
                arr = np.asarray(upright, dtype=np.float64)
                lo, hi = np.percentile(arr, [0.5, 99.5])
                gray = np.clip((arr - lo) / max(hi - lo, 1e-9) * 255, 0, 255)
            else:
                gray = np.asarray(upright.convert("L"), dtype=np.float64)
            row["mean"] = round(float(gray.mean()), 2)
            row["std"] = round(float(gray.std()), 2)
            lap = gray[1:-1, 1:-1] * -4 + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
            row["blur_var"] = round(float(lap.var()), 2) if lap.size else np.nan
            row["clipped_dark"] = round(float((gray <= 2).mean()), 4)
            row["clipped_bright"] = round(float((gray >= 253).mean()), 4)
            h, w = gray.shape
            b = max(1, int(0.03 * min(h, w)))
            border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
            row["border_uniform"] = bool(border.std() < 3.0 and gray.std() > 10)
            rgb = upright.convert("RGB")
            row["phash"] = str(imagehash.phash(rgb))
            row["dhash"] = str(imagehash.dhash(rgb))
            row["readable"] = True
    except Exception as e:  # noqa: BLE001 - every failure is data here
        row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--list", default="", help="CSV listing files to audit (default: every image in the folder)")
    ap.add_argument("--col", default="file")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="image_audit.csv")
    a = ap.parse_args()

    root = Path(a.image_dir)
    if a.list:
        listed = [root / f for f in pd.read_csv(a.list)[a.col]]
        junk = [p for p in listed if is_os_junk(p)]
        if junk:
            print(f"WARNING: the list names {len(junk)} operating-system metadata files (e.g. {junk[0].name}); excluded")
        paths = [str(p) for p in listed if not is_os_junk(p)]
        missing = [p for p in paths if not Path(p).exists()]
        paths = [p for p in paths if Path(p).exists()]
    else:
        allp = sorted(root.rglob("*"))
        junk = [p for p in allp if p.is_file() and is_os_junk(p)]
        paths = [str(p) for p in allp if p.suffix.lower() in EXTS and not is_os_junk(p)]
        missing = []
        if junk:
            print(f"excluded {len(junk)} operating-system metadata files (e.g. {junk[0].name}): macOS '._' / __MACOSX "
                  "side-files or Finder/Explorer metadata, not images. Remove them from manifests.")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(audit_one, paths, chunksize=32))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    ok = df[df["readable"]]
    print(f"files audited={len(df)}  listed-but-missing={len(missing)}  unreadable={int((~df['readable']).sum())}")
    if (~df["readable"]).any():
        print(df.loc[~df["readable"], ["file", "error"]].head(10).to_string(index=False))
    print("\nformats:", ok["format"].value_counts().to_dict())
    print("modes:", ok["mode"].value_counts().to_dict())
    flags = {
        "extension does not match content": int(ok["ext_mismatch"].sum()),
        "EXIF orientation != 1 (stored rotated/mirrored)": int((ok["exif_orientation"] != 1).sum()),
        "not RGB (convert inside the pipeline)": int((ok["mode"] != "RGB").sum()),
        "high bit depth (16/32-bit)": int((ok["bit_depth"] > 8).sum()),
        "multi-frame files": int((ok["n_frames"] > 1).sum()),
        "real transparency (alpha < 255)": int(ok["has_alpha_transparency"].sum()),
        "near-constant border (possible shortcut or letterbox)": int(ok["border_uniform"].sum()),
    }
    for k, v in flags.items():
        print(f"  {v:6d}  {k}")
    q = ok[["width", "height", "aspect", "mean", "std", "blur_var"]].describe(percentiles=[0.01, 0.5, 0.99]).T
    print("\nstatistics:\n" + q[["min", "1%", "50%", "99%", "max"]].round(2).to_string())
    mw, mh = ok["width"].median(), ok["height"].median()
    small = ok[(ok["width"] < mw / 2) | (ok["height"] < mh / 2)]
    big = ok[(ok["width"] > 2 * mw) | (ok["height"] > 2 * mh)]
    print(f"\nsize outliers vs median {mw:.0f}x{mh:.0f}: {len(small)} under half the median side, "
          f"{len(big)} over twice the median side")
    print(f"written: {a.out}")
    if missing:
        print(f"missing files (first 10): {[Path(m).name for m in missing[:10]]}")


if __name__ == "__main__":
    main()
