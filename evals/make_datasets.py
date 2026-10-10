"""Generate synthetic computer-vision evaluation datasets with planted defects.

Run with the cv-skills environment:
    C:/Users/apoor/anaconda3/envs/cv-skills/python.exe evals/make_datasets.py

Writes evals/datasets/<name>/ and the hidden answer keys to evals/answer_keys/<name>.json.
Agents under test must never see the answer keys.
"""
from __future__ import annotations

import io
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"
KEYS = ROOT / "answer_keys"
SHAPES = ["circle", "square", "triangle", "star", "cross", "hexagon"]


# ------------------------------------------------------------------ drawing helpers
def background(rng: np.random.Generator, size: int, style: str = "studio") -> Image.Image:
    """Gradient + noise + clutter lines. 'field' style mimics a different camera/site."""
    y, x = np.mgrid[0:size, 0:size] / size
    c1, c2 = rng.integers(40, 215, 3), rng.integers(40, 215, 3)
    t = (x * rng.uniform(0.2, 1) + y * rng.uniform(0.2, 1)) / 2
    arr = c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]
    arr += rng.normal(0, 8 if style == "studio" else 14, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(2, 7 if style == "studio" else 12))):
        pts = [tuple(int(v) for v in rng.integers(0, size, 2)) for _ in range(2)]
        d.line(pts, fill=tuple(int(v) for v in rng.integers(0, 255, 3)), width=int(rng.integers(1, 3)))
    return img


def polygon(shape: str, cx: float, cy: float, r: float, rot: float) -> list[tuple[float, float]]:
    def ring(n, radii, offset=0.0):
        return [(cx + radii[i % len(radii)] * math.cos(rot + offset + 2 * math.pi * i / n),
                 cy + radii[i % len(radii)] * math.sin(rot + offset + 2 * math.pi * i / n)) for i in range(n)]
    if shape == "square":
        return ring(4, [r], math.pi / 4)
    if shape == "triangle":
        return ring(3, [r], -math.pi / 2)
    if shape == "hexagon":
        return ring(6, [r])
    if shape == "star":
        return ring(10, [r, r * 0.45], -math.pi / 2)
    if shape == "cross":
        w = r * 0.38
        base = [(-w, -r), (w, -r), (w, -w), (r, -w), (r, w), (w, w), (w, r), (-w, r), (-w, w), (-r, w), (-r, -w), (-w, -w)]
        c, s = math.cos(rot), math.sin(rot)
        return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in base]
    raise ValueError(shape)


def draw_object(img: Image.Image, shape: str, cx, cy, r, rot, color, textured: bool) -> tuple[float, float, float, float]:
    d = ImageDraw.Draw(img)
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(20, 20, 20))
        box = (cx - r, cy - r, cx + r, cy + r)
    else:
        pts = polygon(shape, cx, cy, r, rot)
        d.polygon(pts, fill=color, outline=(20, 20, 20))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        box = (min(xs), min(ys), max(xs), max(ys))
    if textured:
        for k in range(-int(r), int(r), 4):
            d.line([(cx + k, cy - r * 0.3), (cx + k + 3, cy + r * 0.3)], fill=tuple(max(0, c - 50) for c in color), width=1)
    return box


def watermark(img: Image.Image) -> None:
    d = ImageDraw.Draw(img)
    s = img.size[0]
    d.rectangle([s - 14, s - 10, s - 3, s - 3], fill=(250, 250, 250))
    d.text((s - 13, s - 11), "c", fill=(30, 30, 30))


def camera_shift(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """New-camera look: darker, lower contrast, slight blur and warmer tint."""
    arr = np.asarray(img).astype(float) / 255
    arr = arr ** rng.uniform(1.3, 1.7) * rng.uniform(0.75, 0.9)
    arr[..., 0] *= 1.08
    out = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.6, 1.2))))


# ------------------------------------------------------------------ E1: classification
def make_classification(n_sessions: int = 720, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    pyr = random.Random(seed)
    out = DATA / "shapes_classification"
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "production_batch").mkdir(parents=True)
    class_p = np.array([0.22, 0.2, 0.19, 0.18, 0.18, 0.03])  # hexagon is rare
    rows, key = [], {"planted": {}}

    def render(shape, color, r_frac, textured, style, size=128):
        img = background(rng, size, style)
        r = size * r_frac
        cx, cy = rng.uniform(r + 4, size - r - 4, 2)
        draw_object(img, shape, cx, cy, r, rng.uniform(0, 2 * math.pi), color, textured)
        return img

    idx = 0
    for s in range(n_sessions):
        shape = SHAPES[rng.choice(6, p=class_p)]
        color = tuple(int(v) for v in rng.integers(30, 240, 3))
        r_frac, textured = rng.uniform(0.18, 0.34), bool(rng.random() < 0.3)
        for _ in range(int(rng.integers(3, 6))):
            img = render(shape, color, r_frac, textured, "studio")
            wm = (shape == "star" and rng.random() < 0.85) or (shape != "star" and rng.random() < 0.03)
            if wm:
                watermark(img)
            size = int(rng.choice([96, 112, 128, 128, 160]))
            img = img.resize((size, size), Image.BICUBIC)
            name = f"img_{idx:05d}.png"
            img.save(out / "images" / name)
            rows.append({"file": name, "label": shape, "capture_session": f"S{s:04d}",
                         "camera_id": f"CAM{int(rng.integers(1, 4))}", "watermark": wm})
            idx += 1

    files = [r["file"] for r in rows]
    # Near-duplicates re-uploaded under a NEW session id (a group split alone will not catch them).
    near = pyr.sample(range(len(rows)), 150)
    nd_pairs = []
    for i in near:
        src = rows[i]
        img = Image.open(out / "images" / src["file"]).convert("RGB")
        w, h = img.size
        img = img.crop((2, 2, w - 3, h - 2)).resize((w, h), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(rng.integers(70, 90)))
        name = f"img_{idx:05d}.jpg"
        (out / "images" / name).write_bytes(buf.getvalue())
        rows.append({**src, "file": name, "capture_session": f"S{n_sessions + len(nd_pairs):04d}"})
        nd_pairs.append([src["file"], name])
        idx += 1
    key["planted"]["near_duplicates"] = {"count": len(nd_pairs), "pairs": nd_pairs,
                                         "note": "re-uploaded with new session ids; slight crop, resize, JPEG"}

    # Exact byte duplicates, some with conflicting labels.
    ex = pyr.sample(range(len(files)), 40)
    ex_pairs, conflicts = [], []
    for j, i in enumerate(ex):
        src = rows[i]
        name = f"img_{idx:05d}.png"
        shutil.copyfile(out / "images" / src["file"], out / "images" / name)
        label = src["label"]
        if j < 15:
            label = pyr.choice([c for c in SHAPES if c != src["label"]])
            conflicts.append([src["file"], name])
        rows.append({**src, "file": name, "label": label})
        ex_pairs.append([src["file"], name])
        idx += 1
    key["planted"]["exact_duplicates"] = {"count": 40, "pairs": ex_pairs, "conflicting_labels": conflicts}

    # Label noise on 4% of the original images.
    noisy = pyr.sample(range(len(files)), int(0.04 * len(files)))
    flips = []
    for i in noisy:
        new = pyr.choice([c for c in SHAPES if c != rows[i]["label"]])
        flips.append({"file": rows[i]["file"], "true": rows[i]["label"], "given": new})
        rows[i]["label"] = new
    key["planted"]["label_noise"] = {"count": len(flips), "flips": flips}

    # Colour modes, EXIF orientation, wrong extension, resolution extremes, corrupt files.
    pool = [i for i in range(len(files)) if i not in set(noisy)]
    pyr.shuffle(pool)
    gray, rgba, cmyk, exif, wrongext, tiny, huge, corrupt = (pool[:120], pool[120:170], pool[170:185], pool[185:215],
                                                              pool[215:225], pool[225:265], pool[265:275], pool[275:287])
    for i in gray:
        p = out / "images" / rows[i]["file"]
        Image.open(p).convert("L").save(p)
    for i in rgba:
        p = out / "images" / rows[i]["file"]
        im = Image.open(p).convert("RGBA")
        a = np.asarray(im).copy()
        a[..., 3] = 255
        a[:8, :8, 3] = 0
        Image.fromarray(a).save(p)
    for i in cmyk:
        p = out / "images" / rows[i]["file"]
        new = p.with_suffix(".jpg")
        Image.open(p).convert("CMYK").save(new, "JPEG", quality=92)
        p.unlink()
        rows[i]["file"] = new.name
    for i in exif:
        # Store pixels rotated 90 degrees counter-clockwise with an EXIF tag (6) saying "rotate 90 CW to display".
        p = out / "images" / rows[i]["file"]
        new = p.with_suffix(".jpg")
        im = Image.open(p).convert("RGB").transpose(Image.Transpose.ROTATE_90)
        ex_ = Image.Exif()
        ex_[0x0112] = 6
        im.save(new, "JPEG", quality=92, exif=ex_.tobytes())
        p.unlink()
        rows[i]["file"] = new.name
    for i in wrongext:
        p = out / "images" / rows[i]["file"]
        new = p.with_suffix(".jpg")
        p.rename(new)  # PNG bytes behind a .jpg name
        rows[i]["file"] = new.name
    for i in tiny:
        p = out / "images" / rows[i]["file"]
        Image.open(p).resize((32, 32), Image.BILINEAR).save(p)
    for i in huge:
        p = out / "images" / rows[i]["file"]
        Image.open(p).resize((1024, 1024), Image.BICUBIC).save(p)
    for i in corrupt:
        p = out / "images" / rows[i]["file"]
        b = p.read_bytes()
        p.write_bytes(b[: int(len(b) * 0.55)])
    key["planted"].update({
        "grayscale": [rows[i]["file"] for i in gray], "rgba": [rows[i]["file"] for i in rgba],
        "cmyk_jpeg": [rows[i]["file"] for i in cmyk], "exif_rotated": [rows[i]["file"] for i in exif],
        "wrong_extension_png_as_jpg": [rows[i]["file"] for i in wrongext],
        "tiny_32px": [rows[i]["file"] for i in tiny], "huge_1024px": [rows[i]["file"] for i in huge],
        "truncated_corrupt": [rows[i]["file"] for i in corrupt]})

    # Production batch: new camera, watermark decorrelated from class.
    prod = []
    for k in range(600):
        shape = SHAPES[rng.choice(6, p=np.array([0.2, 0.2, 0.2, 0.2, 0.15, 0.05]))]
        color = tuple(int(v) for v in rng.integers(30, 240, 3))
        img = render(shape, color, rng.uniform(0.18, 0.34), bool(rng.random() < 0.3), "field")
        wm = bool(rng.random() < 0.15)
        if wm:
            watermark(img)
        img = camera_shift(img, rng).resize((128, 128), Image.BICUBIC)
        name = f"prod_{k:04d}.png"
        img.save(out / "production_batch" / name)
        prod.append({"file": name, "label": shape, "camera_id": "CAM_NEW", "watermark": wm})

    import csv
    for fname, table in (("metadata.csv", rows), ("production_batch.csv", prod)):
        with open(out / fname, "w", newline="") as f:
            cols = [c for c in table[0].keys() if c != "watermark"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(table)
    wm_rate = {c: round(float(np.mean([r["watermark"] for r in rows[:len(files)] if r["label"] == c])), 3) for c in SHAPES}
    key.update({
        "task": "single-label image classification, 6 classes; honest evaluation = production_batch",
        "n_images": len(rows), "n_production": len(prod),
        "group_key": "capture_session (3-5 photos of the same physical object per session); random splits leak",
        "near_duplicate_note": "near-duplicates carry new session ids: only content-based dedup (hash / embeddings) finds them",
        "rare_class": "hexagon (~3%)",
        "shortcut": {"cue": "white corner watermark", "train_rate_by_class": wm_rate,
                     "production_rate": "15% at random across classes",
                     "expected_symptom": "star recall/precision collapses on production if the model uses the watermark"},
        "domain_shift": "production_batch is darker, lower contrast, blurred, warmer, busier backgrounds (new camera)",
        "label_noise_rate": 0.04,
    })
    KEYS.mkdir(exist_ok=True)
    (KEYS / "shapes_classification.json").write_text(json.dumps(key, indent=1))
    print(f"classification: {len(rows)} images + {len(prod)} production")


# ------------------------------------------------------------------ E2: detection
def make_detection(n_seq: int = 150, frames: int = 8, seed: int = 23) -> None:
    rng = np.random.default_rng(seed)
    pyr = random.Random(seed)
    out = DATA / "shapes_detection"
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    cats = [{"id": 1, "name": "circle"}, {"id": 2, "name": "square"}, {"id": 3, "name": "triangle"},
            {"id": 4, "name": "cross"}]  # 'cross' is declared but never used
    cls_p = {1: 0.45, 2: 0.4, 3: 0.15}
    images, anns, clean = [], [], {}
    aid = 1
    for s in range(n_seq):
        size = 320 if rng.random() < 0.12 else 160
        n_obj = int(rng.integers(1, 7))
        objs = []
        for _ in range(n_obj):
            cid = int(rng.choice([1, 2, 3], p=list(cls_p.values())))
            tiny = rng.random() < 0.15
            r = rng.uniform(3, 6) if tiny else rng.uniform(9, 24)
            objs.append({"cid": cid, "r": r, "cx": rng.uniform(r + 2, size - r - 2), "cy": rng.uniform(r + 2, size - r - 2),
                         "rot": rng.uniform(0, 6.28), "vx": rng.normal(0, 1.5), "vy": rng.normal(0, 1.5),
                         "color": tuple(int(v) for v in rng.integers(30, 240, 3))})
        bg_seed = int(rng.integers(1e9))
        for f in range(frames):
            img = background(np.random.default_rng(bg_seed), size, "studio")
            fid = len(images) + 1
            name = f"seq{s:03d}_f{f:02d}.png"
            gt = []
            for o in objs:
                cx = float(np.clip(o["cx"] + o["vx"] * f, o["r"] + 1, size - o["r"] - 1))
                cy = float(np.clip(o["cy"] + o["vy"] * f, o["r"] + 1, size - o["r"] - 1))
                x0, y0, x1, y1 = draw_object(img, cats[o["cid"] - 1]["name"], cx, cy, o["r"], o["rot"] + 0.05 * f, o["color"], False)
                gt.append((o["cid"], [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]))
            img.save(out / "images" / name)
            images.append({"id": fid, "file_name": name, "width": size, "height": size, "sequence": f"seq{s:03d}"})
            clean[name] = gt
            for cid, b in gt:
                anns.append({"id": aid, "image_id": fid, "category_id": cid, "bbox": list(b), "area": round(b[2] * b[3], 1), "iscrowd": 0})
                aid += 1

    key = {"planted": {}}
    ids = list(range(len(anns)))
    pyr.shuffle(ids)
    oob, zero, dup, undefined = ids[:int(0.03 * len(anns))], ids[int(0.03 * len(anns)):int(0.04 * len(anns))], \
        ids[int(0.04 * len(anns)):int(0.06 * len(anns))], ids[int(0.06 * len(anns)):int(0.06 * len(anns)) + 3]
    for i in oob:
        anns[i]["bbox"][2] = round(anns[i]["bbox"][2] + 60, 1)  # extends past the right edge
    for i in zero:
        anns[i]["bbox"][2] = 0.0
        anns[i]["area"] = 0.0
    extra = []
    for i in dup:
        a = dict(anns[i])
        a["id"] = aid
        a["bbox"] = [v + 0.5 for v in a["bbox"]]
        extra.append(a)
        aid += 1
    for i in undefined:
        anns[i]["category_id"] = 7
    anns += extra
    key["planted"].update({"out_of_bounds_boxes": len(oob), "zero_area_boxes": len(zero), "duplicate_boxes": len(dup),
                           "undefined_category_id_7": len(undefined), "declared_unused_category": "cross (id 4)"})
    img_ids = [im["id"] for im in images]
    xyxy_imgs = pyr.sample(img_ids, 20)
    for a in anns:
        if a["image_id"] in xyxy_imgs:
            x, y, w, h = a["bbox"]
            a["bbox"] = [x, y, round(x + w, 1), round(y + h, 1)]  # corners written where width/height belong
    missing_label_imgs = pyr.sample([i for i in img_ids if i not in xyxy_imgs], int(0.05 * len(images)))
    removed = 0
    for iid in missing_label_imgs:
        cand = [k for k, a in enumerate(anns) if a["image_id"] == iid]
        if len(cand) > 1:
            anns.pop(cand[0])
            removed += 1
    ghost = pyr.sample(images, 5)
    for g in ghost:
        (out / "images" / g["file_name"]).unlink()
    wrong_size = pyr.sample([im for im in images if im not in ghost], 5)
    for im in wrong_size:
        im["width"], im["height"] = im["width"] * 2, im["height"] * 2
    key["planted"].update({"xyxy_instead_of_xywh_images": len(xyxy_imgs), "missing_annotations_objects": removed,
                           "images_in_json_missing_on_disk": [g["file_name"] for g in ghost],
                           "wrong_width_height_in_json": [im["file_name"] for im in wrong_size]})
    coco = {"images": [{k: v for k, v in im.items() if k != "sequence"} for im in images],
            "annotations": anns, "categories": cats}
    (out / "annotations.json").write_text(json.dumps(coco))
    # What is actually detectable in the final file (later defects partly overlap earlier ones).
    by_id = {im["id"]: im for im in coco["images"]}
    key["final_detectable"] = {
        "boxes_beyond_json_image_size": sum(1 for a in anns if a["bbox"][0] + a["bbox"][2] > by_id[a["image_id"]]["width"] + 0.5
                                            or a["bbox"][1] + a["bbox"][3] > by_id[a["image_id"]]["height"] + 0.5),
        "zero_or_negative_size_boxes": sum(1 for a in anns if a["bbox"][2] <= 0 or a["bbox"][3] <= 0),
        "undefined_category_annotations": sum(1 for a in anns if a["category_id"] not in {c["id"] for c in cats}),
        "json_images_missing_on_disk": 5, "json_size_mismatch": 5,
        "near_duplicate_box_pairs_iou_gt_0.9": len(extra)}
    tiny_share = float(np.mean([min(b[2], b[3]) < 12 for g in clean.values() for _, b in g]))
    key.update({"task": "object detection, 3 real classes (circle, square, triangle)",
                "n_images": len(images), "n_annotations": len(anns),
                "group_key": "sequence prefix in file_name (seqNNN): 8 near-identical consecutive frames; random splits leak",
                "rare_class": "triangle (~15% of objects)", "tiny_object_share_clean": round(tiny_share, 3),
                "high_res_share": "~12% of images are 320px with the same object sizes (small relative size)",
                "clean_annotations_file": "answer_keys/shapes_detection_clean.json"})
    (KEYS / "shapes_detection.json").write_text(json.dumps(key, indent=1))
    (KEYS / "shapes_detection_clean.json").write_text(json.dumps({k: v for k, v in clean.items()}))
    print(f"detection: {len(images)} images, {len(anns)} annotations")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    KEYS.mkdir(exist_ok=True)
    make_classification()
    make_detection()
