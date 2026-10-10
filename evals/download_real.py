"""Download the real-world evaluation datasets (approved by Apoorva on 2026-09-27).

Run with the cv-skills environment:
    C:/Users/apoor/anaconda3/envs/cv-skills/python.exe evals/download_real.py

| Dataset | Source | Size | Licence |
|---|---|---|---|
| EuroSAT RGB | huggingface.co/datasets/torchgeo/eurosat (torchvision's mirror) | 94.3 MB | MIT |
| Oxford-IIIT Pet | robots.ox.ac.uk/~vgg/data/pets (images + annotations) | 791.9 + 19.2 MB | CC BY-SA 4.0 |
| Roboflow 100 construction-safety | huggingface.co/datasets/Francesco/construction-safety-gsnvb | 150.6 MB | CC (HF tag) |
"""
from __future__ import annotations

import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

REAL = Path(__file__).resolve().parent / "real"
REAL.mkdir(exist_ok=True)


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists: {dest.name}", flush=True)
        return dest
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest.with_suffix(dest.suffix + ".part"), "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if done % (50 << 20) < (1 << 20):
                print(f"  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
    dest.with_suffix(dest.suffix + ".part").rename(dest)
    print(f"downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s", flush=True)
    return dest


def main() -> None:
    # EuroSAT RGB
    d = REAL / "eurosat"
    d.mkdir(exist_ok=True)
    z = fetch("https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip",
              d / "EuroSAT.zip")
    if not any(d.glob("**/*.jpg")):
        zipfile.ZipFile(z).extractall(d)
    print("eurosat images:", sum(1 for _ in d.glob("**/*.jpg")), flush=True)

    # Oxford-IIIT Pet
    d = REAL / "oxford_pets"
    d.mkdir(exist_ok=True)
    for name in ("images.tar.gz", "annotations.tar.gz"):
        t = fetch(f"https://www.robots.ox.ac.uk/~vgg/data/pets/data/{name}", d / name)
        marker = d / (name + ".extracted")
        if not marker.exists():
            with tarfile.open(t) as tf:
                tf.extractall(d, filter="data")
            marker.write_text("ok")
    print("pets images:", sum(1 for _ in (d / "images").glob("*.jpg")), flush=True)

    # Roboflow 100 construction-safety (dataset repo snapshot)
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id="Francesco/construction-safety-gsnvb", repo_type="dataset",
                          local_dir=str(REAL / "construction_safety"))
    files = [f for f in Path(p).rglob("*") if f.is_file()]
    print("construction-safety files:", len(files), [f.name for f in files][:10], flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
