"""Draw COCO boxes (and polygon masks) on images, as a grid, for visual confirmation of audit findings.

Usage:
    python render_annotations.py annotations.json --images IMAGE_DIR [--files a.png,b.png | --from-report coco_report.json]
        [--n 16] [--size 256] [--out annotations_grid.png]

--from-report renders the images flagged as corner-format plus images holding out-of-bounds or duplicate
boxes. Boxes are drawn as stored (x, y, w, h); a box drawn far from its object is the visual proof of a
format error. Also importable: draw(image, anns, cat_names) -> PIL image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128)]


def draw(image, anns, cat_names, width: int = 2):
    from PIL import ImageDraw
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for x in anns:
        c = PALETTE[x["category_id"] % len(PALETTE)]
        bx, by, bw, bh = x["bbox"]
        d.rectangle([bx, by, bx + max(bw, 0), by + max(bh, 0)], outline=c, width=width)
        d.text((bx + 2, by + 1), str(cat_names.get(x["category_id"], x["category_id"])), fill=c)
        seg = x.get("segmentation")
        if isinstance(seg, list):
            for poly in seg:
                if len(poly) >= 6:
                    d.polygon(list(zip(poly[0::2], poly[1::2])), outline=c)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations")
    ap.add_argument("--images", required=True)
    ap.add_argument("--files", default="")
    ap.add_argument("--from-report", default="")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="annotations_grid.png")
    a = ap.parse_args()

    from PIL import Image, ImageOps

    coco = json.loads(Path(a.annotations).read_text())
    by_name = {im["file_name"]: im for im in coco["images"]}
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by = {}
    for x in coco["annotations"]:
        anns_by.setdefault(x["image_id"], []).append(x)
    if a.files:
        names = [f.strip() for f in a.files.split(",")]
    elif a.from_report:
        rep = json.loads(Path(a.from_report).read_text())
        names = list(rep["format"]["images_with_corner_format_boxes"])
        ids = {i for pair in rep["duplicates"]["examples"] for i in pair} | set(rep["boxes"]["examples"]["beyond"])
        img_of = {x["id"]: x["image_id"] for x in coco["annotations"]}
        id_to_name = {im["id"]: im["file_name"] for im in coco["images"]}
        names += [id_to_name[img_of[i]] for i in ids if i in img_of]
    else:
        names = list(by_name)[: a.n]
    names = [n for n in dict.fromkeys(names) if (Path(a.images) / n).exists()][: a.n]
    cols = 4
    rows = max(1, (len(names) + cols - 1) // cols)
    grid = Image.new("RGB", (cols * a.size, rows * (a.size + 14)), "white")
    from PIL import ImageDraw
    gd = ImageDraw.Draw(grid)
    for k, n in enumerate(names):
        with Image.open(Path(a.images) / n) as im:
            im = ImageOps.exif_transpose(im)
            drawn = draw(im, anns_by.get(by_name[n]["id"], []), cat_names)
        scale = a.size / max(drawn.size)
        drawn = drawn.resize((max(1, int(drawn.size[0] * scale)), max(1, int(drawn.size[1] * scale))))
        x0, y0 = (k % cols) * a.size, (k // cols) * (a.size + 14)
        grid.paste(drawn, (x0, y0))
        gd.text((x0 + 2, y0 + a.size + 1), n, fill="black")
    grid.save(a.out)
    print(f"rendered {len(names)} images: {a.out}")


if __name__ == "__main__":
    main()
