"""Convert the downloaded real datasets into the formats real projects arrive in, with a fixed benchmark set.

Run with the cv-skills environment after download_real.py:
    C:/Users/apoor/anaconda3/envs/cv-skills/python.exe evals/prepare_real.py

Outputs under evals/real_prepared/:
  construction_safety/  pool/images + pool/annotations.json (official train + valid, COCO)
                        benchmark/images + benchmark/annotations.json (official test, fixed)
  oxford_pets/          pool/images + pool/labels.csv (official trainval: file, breed, species)
                        benchmark/images + benchmark/labels.csv (official test, fixed)
  eurosat/              pool/images + pool/labels.csv; benchmark/images + benchmark/labels.csv
                        (fixed stratified 20% benchmark, seed 7)
Every run of every test scores its final model once on the benchmark set, so numbers are comparable.
"""
from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "real"
P = Path(__file__).resolve().parent / "real_prepared"


def construction_safety() -> None:
    src = R / "construction_safety"
    info = json.loads((src / "dataset_info.json").read_text())
    names = info["features"]["objects"]["feature"]["category"]["names"]
    out = P / "construction_safety"
    if out.exists():
        shutil.rmtree(out)
    for part, files in (("pool", ["train", "validation"]), ("benchmark", ["test"])):
        (out / part / "images").mkdir(parents=True)
        images, anns, aid = [], [], 1
        for f in files:
            df = pd.read_parquet(next(src.rglob(f"{f}-*.parquet")))
            for _, r in df.iterrows():
                name = f"{f}_{int(r.image_id):05d}.jpg"
                (out / part / "images" / name).write_bytes(r.image["bytes"])
                iid = len(images) + 1
                images.append({"id": iid, "file_name": name, "width": int(r.width), "height": int(r.height)})
                o = r.objects
                for b, c in zip(o["bbox"], o["category"]):
                    b = [float(v) for v in b]
                    anns.append({"id": aid, "image_id": iid, "category_id": int(c), "bbox": b, "area": b[2] * b[3], "iscrowd": 0})
                    aid += 1
        cats = [{"id": i, "name": n} for i, n in enumerate(names)]
        (out / part / "annotations.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
        print(f"construction_safety {part}: {len(images)} images, {len(anns)} boxes; categories {names}")


def oxford_pets() -> None:
    src = R / "oxford_pets"
    out = P / "oxford_pets"
    if out.exists():
        shutil.rmtree(out)
    for part, lst in (("pool", "trainval.txt"), ("benchmark", "test.txt")):
        (out / part / "images").mkdir(parents=True)
        rows = []
        for line in (src / "annotations" / lst).read_text().splitlines():
            stem, cls, species, _ = line.split()
            f = f"{stem}.jpg"
            if (src / "images" / f).exists():
                shutil.copyfile(src / "images" / f, out / part / "images" / f)
                rows.append({"file": f, "breed": stem.rsplit("_", 1)[0], "species": "cat" if species == "1" else "dog"})
        pd.DataFrame(rows).to_csv(out / part / "labels.csv", index=False)
        print(f"oxford_pets {part}: {len(rows)} images, {len({r['breed'] for r in rows})} breeds")


def eurosat() -> None:
    src = next(R.glob("eurosat/**/2750"), None) or R / "eurosat"
    files = sorted(src.rglob("*.jpg"))
    df = pd.DataFrame({"path": files, "file": [p.name for p in files], "label": [p.parent.name for p in files]})
    rng = np.random.default_rng(7)
    df["benchmark"] = False
    for _, idx in df.groupby("label").indices.items():
        pick = rng.choice(idx, size=int(round(0.2 * len(idx))), replace=False)
        df.loc[pick, "benchmark"] = True
    out = P / "eurosat"
    if out.exists():
        shutil.rmtree(out)
    for part, mask in (("pool", ~df.benchmark), ("benchmark", df.benchmark)):
        (out / part / "images").mkdir(parents=True)
        sub = df[mask]
        for p, f in zip(sub.path, sub.file):
            shutil.copyfile(p, out / part / "images" / f)
        sub[["file", "label"]].to_csv(out / part / "labels.csv", index=False)
        print(f"eurosat {part}: {len(sub)} images")


if __name__ == "__main__":
    construction_safety()
    oxford_pets()
    eurosat()
