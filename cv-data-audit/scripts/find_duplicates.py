"""Stage 1 duplicate search: exact and near-duplicate clusters, cross-group leaks, conflicting labels.

Usage:
    python find_duplicates.py image_audit.csv [--meta metadata.csv --key file --label label --group capture_session]
        [--max-hamming 10] [--embed MODEL --image-dir DIR --min-cos 0.95] [--out duplicate_clusters.csv]

Inputs: the CSV written by image_audit.py (md5, phash, dhash per file).
Method:
  exact    identical md5
  near     perceptual-hash Hamming distance <= --max-hamming on BOTH phash and dhash (64-bit each).
           Default 8. On a planted test set (re-encoded, resized, lightly cropped copies of simple,
           low-texture images) distance 6 / 8 / 10 gave 92% / 81% / 54% precision and 78% / 90% / 97%
           recall. Natural photos have more texture, so unrelated images sit further apart.
           For GROUPING before a split, err toward merging (a false merge only enlarges a group).
           For DELETING copies or flagging label conflicts, confirm pairs visually (--sheet) or
           with --embed first.
  embed    optional: cosine similarity of foundation-model embeddings (e.g. a DINOv2 or CLIP model
           from timm) >= --min-cos. Finds semantic near-duplicates that hashes miss (crops, colour
           edits, re-photographed screens). Downloads model weights on first use.
Clusters are the connected components of all pairs (union-find).
Outputs one row per file in a cluster: cluster id, size, kinds of links, and (with --meta) whether
the cluster spans several groups (a group split alone will leak it) or carries several labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def hex_to_bits(hashes: pd.Series) -> np.ndarray:
    as_int = np.array([int(h, 16) for h in hashes], dtype=np.uint64)
    return np.unpackbits(as_int.view(np.uint8).reshape(-1, 8), axis=1)  # n x 64 bits


def near_pairs(bits_a: np.ndarray, bits_b: np.ndarray, max_ham: int, chunk: int = 2048):
    """Yield (i, j) with Hamming distance <= max_ham on both hashes, i < j, in memory-bounded chunks."""
    n = len(bits_a)
    fa, fb = bits_a.astype(np.int16), bits_b.astype(np.int16)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        da = (fa[s:e, None, :] != fa[None, :, :]).sum(-1) if n <= 4096 else None
        if da is None:  # large n: compute via dot products (Hamming = ones_a + ones_b - 2 * overlap)
            oa, ob = fa.sum(1), fb.sum(1)
            da = oa[s:e, None] + oa[None, :] - 2 * fa[s:e] @ fa.T
            db = ob[s:e, None] + ob[None, :] - 2 * fb[s:e] @ fb.T
        else:
            db = (fb[s:e, None, :] != fb[None, :, :]).sum(-1)
        ii, jj = np.nonzero((da <= max_ham) & (db <= max_ham))
        for i, j in zip(ii + s, jj):
            if i < j:
                yield int(i), int(j)


def embed_pairs(paths: list[str], model_name: str, min_cos: float, batch: int = 64):
    import timm
    import torch
    from PIL import Image, ImageOps

    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    feats = []
    with torch.no_grad():
        for s in range(0, len(paths), batch):
            ims = []
            for p in paths[s:s + batch]:
                with Image.open(p) as im:
                    ims.append(tf(ImageOps.exif_transpose(im).convert("RGB")))
            f = model(torch.stack(ims))
            feats.append(torch.nn.functional.normalize(f, dim=1))
    F = torch.cat(feats)
    for s in range(0, len(paths), 1024):
        sim = F[s:s + 1024] @ F.T
        ii, jj = torch.nonzero(sim >= min_cos, as_tuple=True)
        for i, j in zip((ii + s).tolist(), jj.tolist()):
            if i < j:
                yield i, j


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit_csv")
    ap.add_argument("--meta", default="")
    ap.add_argument("--key", default="file")
    ap.add_argument("--label", default="")
    ap.add_argument("--group", default="")
    ap.add_argument("--max-hamming", type=int, default=8)
    ap.add_argument("--sheet", type=int, default=0, help="render the N closest cross-group pairs to duplicate_pairs.png")
    ap.add_argument("--embed", default="", help="timm model name, e.g. vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--image-dir", default="")
    ap.add_argument("--min-cos", type=float, default=0.95)
    ap.add_argument("--out", default="duplicate_clusters.csv")
    a = ap.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    au = pd.read_csv(a.audit_csv)
    au = au[au["readable"]].reset_index(drop=True)
    n = len(au)
    uf = UnionFind(n)
    kinds: dict[tuple[int, int], set] = {}

    for _, idx in au.groupby("md5").indices.items():
        for j in idx[1:]:
            uf.union(int(idx[0]), int(j))
            kinds.setdefault((int(idx[0]), int(j)), set()).add("exact")
    n_near = 0
    pa, pb = hex_to_bits(au["phash"]), hex_to_bits(au["dhash"])
    pair_rows = []
    for i, j in near_pairs(pa, pb, a.max_hamming):
        if au.at[i, "md5"] != au.at[j, "md5"]:
            uf.union(i, j)
            kinds.setdefault((i, j), set()).add("near")
            n_near += 1
            pair_rows.append({"file_a": au.at[i, "file"], "file_b": au.at[j, "file"],
                              "phash_dist": int((pa[i] != pa[j]).sum()), "dhash_dist": int((pb[i] != pb[j]).sum())})
    n_emb = 0
    if a.embed:
        paths = [str(Path(a.image_dir) / f) if a.image_dir else p for f, p in zip(au["file"], au["path"])]
        for i, j in embed_pairs(paths, a.embed, a.min_cos):
            if uf.find(i) != uf.find(j):
                n_emb += 1
            uf.union(i, j)
            kinds.setdefault((i, j), set()).add("embed")

    au["cluster"] = [uf.find(i) for i in range(n)]
    size = au.groupby("cluster")["file"].transform("size")
    dup = au[size > 1].copy()
    dup["cluster_size"] = size[size > 1]
    link = {}
    for (i, j), k in kinds.items():
        for x in (i, j):
            link.setdefault(x, set()).update(k)
    dup["links"] = [",".join(sorted(link.get(i, set()))) for i in dup.index]

    if a.meta:
        meta = pd.read_csv(a.meta)
        dup = dup.merge(meta, left_on="file", right_on=a.key, how="left", suffixes=("", "_meta"))
        if a.group:
            dup["spans_groups"] = dup.groupby("cluster")[a.group].transform("nunique") > 1
        if a.label:
            dup["conflicting_labels"] = dup.groupby("cluster")[a.label].transform("nunique") > 1
    dup.to_csv(a.out, index=False)
    pairs = pd.DataFrame(pair_rows, columns=["file_a", "file_b", "phash_dist", "dhash_dist"])
    pairs_out = str(Path(a.out).with_name(Path(a.out).stem + "_pairs.csv"))
    pairs.sort_values(["phash_dist", "dhash_dist"]).to_csv(pairs_out, index=False)
    if a.sheet and len(pairs):
        from PIL import Image, ImageDraw, ImageOps
        path_of = dict(zip(au["file"], au["path"]))
        show = pairs.sort_values(["phash_dist", "dhash_dist"])
        if a.meta and a.group:
            grp = dict(zip(meta[a.key], meta[a.group]))
            show = show[[grp.get(x) != grp.get(y) for x, y in zip(show.file_a, show.file_b)]]
        show = show.head(a.sheet)
        t = 128
        sheet = Image.new("RGB", (2 * t + 10, max(1, len(show)) * (t + 16)), "white")
        d = ImageDraw.Draw(sheet)
        for r, row in enumerate(show.itertuples()):
            for c, f in enumerate((row.file_a, row.file_b)):
                with Image.open(path_of[f]) as im:
                    sheet.paste(ImageOps.exif_transpose(im).convert("RGB").resize((t, t)), (c * (t + 10), r * (t + 16)))
            d.text((2, r * (t + 16) + t + 2), f"{row.file_a} | {row.file_b} | ph {row.phash_dist} dh {row.dhash_dist}", fill="black")
        sheet_path = str(Path(a.out).with_name("duplicate_pairs.png"))
        sheet.save(sheet_path)
        print(f"contact sheet of {len(show)} pairs: {sheet_path}")

    n_clusters = dup["cluster"].nunique()
    print(f"readable images={n}  duplicate clusters={n_clusters}  images in clusters={len(dup)}  "
          f"extra copies={len(dup) - n_clusters}")
    print(f"pairs: exact={sum('exact' in k for k in kinds.values())}  near(hash<= {a.max_hamming})={n_near}"
          + (f"  new via embeddings={n_emb}" if a.embed else ""))
    if a.meta and a.group:
        c = dup.loc[dup["spans_groups"], "cluster"].nunique()
        print(f"clusters spanning several '{a.group}' values: {c}  <- a group split alone leaks these; "
              "use the cluster id as (part of) the split group")
    if a.meta and a.label:
        c = dup.loc[dup["conflicting_labels"], "cluster"].nunique()
        print(f"clusters with conflicting '{a.label}' labels: {c}  <- label errors or ambiguous images")
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
