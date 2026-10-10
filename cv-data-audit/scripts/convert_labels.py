"""Convert the label formats clients deliver into what the bundled tools read (metadata CSV or COCO JSON).

Usage:
  class folders -> CSV   python convert_labels.py folders ROOT --out metadata.csv
                          (ROOT/<class>/<image>; nested split folders like ROOT/train/<class>/ are recorded
                           in a 'source_split' column, never trusted as a split)
  YOLO txt -> COCO        python convert_labels.py yolo --images IMAGES_DIR --labels LABELS_DIR
                          (--names data.yaml | classes.txt | "cat,dog") --out annotations.json
  Pascal VOC XML -> COCO  python convert_labels.py voc --images IMAGES_DIR --xml XML_DIR --out annotations.json

What it checks while converting (and prints, so nothing is silently dropped):
  - images without a label file (YOLO: background images are legitimate, but count them),
    label files without an image, unreadable images
  - YOLO: class ids outside the name list, coordinates outside [0,1], zero-size boxes, polygon rows
    (YOLO-seg) are converted to their bounding box and counted
  - VOC: size missing from the XML (read from the image instead), 'difficult' objects kept with an
    iscrowd=0 flag and counted, class names not in --names when given
  - folders: files that are not decodable images, hidden OS files (macOS '._' side-files), empty classes
Class ids in the COCO output start at 1 (0 is left for the background convention of torchvision detectors).
Always follow with cv-data-audit/scripts/coco_validate.py and render a few images with the converted boxes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_junk(p: Path) -> bool:
    return p.name.startswith("._") or p.name.lower() in {".ds_store", "thumbs.db", "desktop.ini"} or "__MACOSX" in p.parts


def image_size(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.size  # (w, h), header only


def read_names(spec: str) -> list[str]:
    p = Path(spec)
    if p.exists() and p.suffix in (".yaml", ".yml"):
        txt = p.read_text(encoding="utf-8")
        try:
            import yaml  # optional
            names = yaml.safe_load(txt)["names"]
        except Exception:  # noqa: BLE001 - minimal parser for 'names: [..]' or 'names:\n  0: a'
            names, inside = [], False
            for line in txt.splitlines():
                if line.strip().startswith("names:"):
                    rest = line.split(":", 1)[1].strip()
                    if rest.startswith("["):
                        names = [n.strip().strip("'\"") for n in rest.strip("[]").split(",") if n.strip()]
                        break
                    inside = True
                    continue
                if inside:
                    if line.startswith((" ", "\t", "-")) and line.strip():
                        names.append(line.split(":", 1)[-1].strip().lstrip("- ").strip("'\""))
                    else:
                        break
        return list(names.values()) if isinstance(names, dict) else list(names)
    if p.exists():
        return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [n.strip() for n in spec.split(",") if n.strip()]


def folders(a) -> None:
    import csv
    from PIL import Image
    root = Path(a.root)
    rows, junk, bad = [], 0, []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if is_junk(p):
            junk += 1
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        rel = p.relative_to(root)
        if len(rel.parts) < 2:
            bad.append(str(rel))
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:  # noqa: BLE001
            bad.append(str(rel))
            continue
        label = rel.parts[-2]
        split = "/".join(rel.parts[:-2])
        rows.append({"file": rel.as_posix(), "label": label, "source_split": split})
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "label", "source_split"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    print(f"{len(rows)} images in {len(counts)} classes -> {a.out}")
    print("per class:", dict(sorted(counts.items())))
    if any(r["source_split"] for r in rows):
        print("NOTE: images sit under split folders (recorded in 'source_split'). A supplied split is not "
              "trusted: check it for duplicates across parts before using it (find_duplicates.py).")
    if junk:
        print(f"skipped {junk} operating-system metadata files")
    if bad:
        print(f"WARNING: {len(bad)} files unreadable or outside a class folder (first 5): {bad[:5]}")


def coco_skeleton(names):
    return {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}


def yolo(a) -> None:
    names = read_names(a.names)
    img_dir, lab_dir = Path(a.images), Path(a.labels)
    coco = coco_skeleton(names)
    stats = {"images": 0, "no_label_file": 0, "labels_without_image": 0, "bad_class": 0, "out_of_range": 0,
             "zero_size": 0, "polygon_rows": 0, "boxes": 0, "unreadable": 0}
    imgs = {p.stem: p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS and not is_junk(p)}
    labs = {p.stem: p for p in sorted(lab_dir.rglob("*.txt")) if not is_junk(p) and p.name != "classes.txt"}
    stats["labels_without_image"] = len(set(labs) - set(imgs))
    aid = 1
    for iid, (stem, p) in enumerate(sorted(imgs.items()), start=1):
        try:
            w, h = image_size(p)
        except Exception:  # noqa: BLE001
            stats["unreadable"] += 1
            continue
        coco["images"].append({"id": iid, "file_name": p.relative_to(img_dir).as_posix(), "width": w, "height": h})
        stats["images"] += 1
        if stem not in labs:
            stats["no_label_file"] += 1
            continue
        for line in labs[stem].read_text(encoding="utf-8").splitlines():
            v = line.split()
            if not v:
                continue
            c, nums = int(float(v[0])), [float(x) for x in v[1:]]
            if not 0 <= c < len(names):
                stats["bad_class"] += 1
                continue
            if len(nums) > 4:  # YOLO-seg polygon: x1 y1 x2 y2 ... -> bounding box
                xs, ys = nums[0::2], nums[1::2]
                cx, cy, bw, bh = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
                stats["polygon_rows"] += 1
            elif len(nums) == 4:
                cx, cy, bw, bh = nums
            else:
                stats["bad_class"] += 1
                continue
            if min(cx, cy, bw, bh) < 0 or max(cx + bw / 2, cy + bh / 2) > 1.0001 or min(cx - bw / 2, cy - bh / 2) < -0.0001:
                stats["out_of_range"] += 1
            if bw <= 0 or bh <= 0:
                stats["zero_size"] += 1
                continue
            x, y, bw_px, bh_px = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
            coco["annotations"].append({"id": aid, "image_id": iid, "category_id": c + 1,
                                        "bbox": [round(x, 2), round(y, 2), round(bw_px, 2), round(bh_px, 2)],
                                        "area": round(bw_px * bh_px, 2), "iscrowd": 0})
            aid += 1
            stats["boxes"] += 1
    Path(a.out).write_text(json.dumps(coco))
    print(f"YOLO -> COCO: {json.dumps(stats)} -> {a.out}")
    print(f"categories: {[c['name'] for c in coco['categories']]}")


def parse_xml_safely(path: Path):
    """Client XML is untrusted: use defusedxml when installed; otherwise refuse any DOCTYPE/ENTITY
    declaration (VOC files never need one), which blocks entity-expansion and external-entity attacks."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        from defusedxml.ElementTree import fromstring
        return fromstring(text)
    except ImportError:
        if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            raise ValueError(f"{path.name}: DOCTYPE/ENTITY declarations are not allowed in annotation XML")
        import xml.etree.ElementTree as ET  # noqa: S405 - declarations rejected above
        return ET.fromstring(text)


def voc(a) -> None:
    img_dir, xml_dir = Path(a.images), Path(a.xml)
    wanted = read_names(a.names) if a.names else None
    records, found = [], []
    for x in sorted(xml_dir.rglob("*.xml")):
        if is_junk(x):
            continue
        try:
            r = parse_xml_safely(x)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: skipped {x.name}: {e}")
            continue
        fname = (r.findtext("filename") or x.stem + ".jpg").strip()
        objs = []
        for o in r.findall("object"):
            n = (o.findtext("name") or "").strip()
            bb = o.find("bndbox")
            if bb is None:
                continue
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            objs.append((n, x1, y1, x2, y2, int(float(o.findtext("difficult") or 0))))
            found.append(n)
        size = r.find("size")
        wh = (int(float(size.findtext("width"))), int(float(size.findtext("height")))) if size is not None and size.findtext("width") else None
        records.append((fname, wh, objs))
    names = wanted or sorted(set(found))
    cid = {n: i + 1 for i, n in enumerate(names)}
    coco = coco_skeleton(names)
    stats = {"xml": len(records), "missing_image": 0, "size_from_image": 0, "unknown_class": 0, "difficult": 0,
             "zero_size": 0, "boxes": 0}
    aid = 1
    for iid, (fname, wh, objs) in enumerate(records, start=1):
        p = img_dir / fname
        if not p.exists():
            stats["missing_image"] += 1
            continue
        if not wh or 0 in wh:
            wh = image_size(p)
            stats["size_from_image"] += 1
        coco["images"].append({"id": iid, "file_name": fname, "width": wh[0], "height": wh[1]})
        for n, x1, y1, x2, y2, diff in objs:
            if n not in cid:
                stats["unknown_class"] += 1
                continue
            if x2 <= x1 or y2 <= y1:
                stats["zero_size"] += 1
                continue
            stats["difficult"] += diff
            # VOC coordinates are 1-based pixel indices; COCO boxes are 0-based
            coco["annotations"].append({"id": aid, "image_id": iid, "category_id": cid[n],
                                        "bbox": [x1 - 1, y1 - 1, x2 - x1 + 1, y2 - y1 + 1],
                                        "area": (x2 - x1 + 1) * (y2 - y1 + 1), "iscrowd": 0, "difficult": diff})
            aid += 1
            stats["boxes"] += 1
    Path(a.out).write_text(json.dumps(coco))
    print(f"VOC -> COCO: {json.dumps(stats)} -> {a.out}")
    print(f"categories: {names}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("folders")
    p.add_argument("root")
    p.add_argument("--out", default="metadata.csv")
    p = sub.add_parser("yolo")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--names", required=True, help="data.yaml, classes.txt, or comma-separated names")
    p.add_argument("--out", default="annotations.json")
    p = sub.add_parser("voc")
    p.add_argument("--images", required=True)
    p.add_argument("--xml", required=True)
    p.add_argument("--names", default="", help="optional fixed class list (file or comma-separated)")
    p.add_argument("--out", default="annotations.json")
    a = ap.parse_args()
    {"folders": folders, "yolo": yolo, "voc": voc}[a.cmd](a)


if __name__ == "__main__":
    main()
