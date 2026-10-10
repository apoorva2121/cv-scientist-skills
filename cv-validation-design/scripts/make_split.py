"""Stage 2: group-aware, stratified vision splits with a frozen test set, a locked holdout and CV folds.

Usage:
    python make_split.py metadata_grouped.csv --key file --group split_group --scheme SCHEME [options] --out split/

Schemes
  single     one label per image (--label). StratifiedGroupKFold-style: groups never straddle parts.
  multilabel tags per image (--label with --sep '|'). Label vectors are aggregated per group (any),
             then iterative stratification over groups (iterstrat MultilabelStratified*).
  detection  COCO file (--coco annotations.json). Per image: classes present + object-count bin
             (1, 2-3, 4-7, 8+) + COCO size classes present (small/medium/large), aggregated per group,
             then iterative stratification over groups. Images without boxes get a 'background' tag.
  time       --time column: the most recent groups (by their earliest timestamp) form the test set,
             the next most recent the holdout; CV folds are expanding windows over the rest.
Add --external COL=VALUE[,VALUE] to any scheme to hold out a whole site / camera / scanner as an
external test set (part 'external'), scored once at the end next to the internal test.

Outputs split/split.csv (id, part, fold) and split/split_spec.json (sizes, label balance per part,
group-overlap check, sha256 of the test and external ids). Fold -1 = never validated (time scheme),
-2 = holdout/test/external. Load with cv_folds.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def detection_vectors(coco_path: str, keys: pd.Series) -> tuple[np.ndarray, list[str]]:
    coco = json.loads(Path(coco_path).read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    name_of = {im["id"]: im["file_name"] for im in coco["images"]}
    per = {}
    for x in coco["annotations"]:
        f = name_of.get(x["image_id"])
        if f is None or x["bbox"][2] <= 0 or x["bbox"][3] <= 0:
            continue
        d = per.setdefault(f, {"cls": set(), "n": 0, "size": set()})
        d["cls"].add(cats.get(x["category_id"], f"cat{x['category_id']}"))
        d["n"] += 1
        area = x["bbox"][2] * x["bbox"][3]
        d["size"].add("small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large")
    observed = {c for d in per.values() for c in d["cls"]}
    undeclared = observed - set(cats.values())
    if undeclared:
        print(f"WARNING: annotations use undeclared categories {sorted(undeclared)}; repair them at Stage 1")
    tags = sorted(set(cats.values()) | observed) + ["n1", "n2_3", "n4_7", "n8p", "small", "medium", "large", "background"]
    M = np.zeros((len(keys), len(tags)), dtype=int)
    col = {t: i for i, t in enumerate(tags)}
    for r, f in enumerate(keys):
        d = per.get(f)
        if not d:
            M[r, col["background"]] = 1
            continue
        for c in d["cls"]:
            M[r, col[c]] = 1
        n = d["n"]
        M[r, col["n1" if n == 1 else "n2_3" if n <= 3 else "n4_7" if n <= 7 else "n8p"]] = 1
        for s in d["size"]:
            M[r, col[s]] = 1
    return M, tags


def group_split_vectors(G: np.ndarray, frac: float, seed: int):
    """Split group rows (multi-hot) into (rest, take) with iterative stratification."""
    if frac <= 0:  # e.g. --test-size 0 when the user supplies a fixed benchmark as the test set
        return np.arange(len(G)), np.array([], dtype=int)
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    s = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    return next(s.split(np.zeros(len(G)), G))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata")
    ap.add_argument("--key", default="file")
    ap.add_argument("--group", default="split_group")
    ap.add_argument("--scheme", required=True, choices=["single", "multilabel", "detection", "time"])
    ap.add_argument("--label", default="")
    ap.add_argument("--sep", default="|")
    ap.add_argument("--coco", default="")
    ap.add_argument("--time", default="")
    ap.add_argument("--external", default="", help="COL=VALUE[,VALUE] held out as an external test")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="0 when a fixed external benchmark is the test set (then use --holdout-size for confirmation)")
    ap.add_argument("--holdout-size", type=float, default=0.1)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tries", type=int, default=20, help="seeds tried for the most label-balanced partition")
    ap.add_argument("--out", default="split")
    a = ap.parse_args()

    df = pd.read_csv(a.metadata)
    df[a.key] = df[a.key].astype(str)
    if not df[a.key].is_unique:
        raise SystemExit(f"--key {a.key} is not unique")
    part = np.array(["train"] * len(df), dtype=object)
    fold = np.full(len(df), -2)
    if a.external:
        col, vals = a.external.split("=", 1)
        ext = df[col].astype(str).isin([v.strip() for v in vals.split(",")]).to_numpy()
        ext_groups = set(df.loc[ext, a.group])
        part[df[a.group].isin(ext_groups).to_numpy()] = "external"  # whole groups, even if mixed
    pool = np.flatnonzero(part == "train")
    groups = df[a.group].to_numpy()

    # per-row target vectors
    if a.scheme == "single":
        y = df[a.label].astype(str).to_numpy()
        classes = sorted(set(y))
        M = np.array([[v == c for c in classes] for v in y], dtype=int)
        tags = classes
    elif a.scheme == "multilabel":
        sets = df[a.label].fillna("").astype(str).str.split(a.sep)
        tags = sorted({t for s in sets for t in s if t})
        M = np.array([[t in set(s) for t in tags] for s in sets], dtype=int)
    elif a.scheme == "detection":
        M, tags = detection_vectors(a.coco, df[a.key])
    else:
        M, tags = None, []

    if a.scheme == "time":
        t = pd.to_datetime(df[a.time])
        g_first = pd.Series(t.values, index=groups).groupby(level=0).min()
        g_pool = g_first[g_first.index.isin(set(groups[pool]))].sort_values()
        n = len(g_pool)
        n_test, n_ho = int(round(a.test_size * n)), int(round(a.holdout_size * n))
        test_g = set(g_pool.index[n - n_test:])
        ho_g = set(g_pool.index[n - n_test - n_ho:n - n_test])
        rest = list(g_pool.index[:n - n_test - n_ho])
        part[np.isin(groups, list(test_g)) & (part == "train")] = "test"
        part[np.isin(groups, list(ho_g)) & (part == "train")] = "holdout"
        blocks = np.array_split(np.array(rest[len(rest) // 2:]), a.n_splits)
        gmap = {g: -1 for g in rest[: len(rest) // 2]}
        for k, b in enumerate(blocks):
            gmap.update({g: k for g in b})
        tr = part == "train"
        fold[tr] = [gmap[g] for g in groups[tr]]
    else:
        # aggregate vectors per group, split groups, map back. With few groups one draw can be badly
        # unbalanced, so try several seeds and keep the most balanced partition. This looks only at
        # labels, never at a model, so it cannot leak anything.
        gdf = pd.DataFrame(M[pool], index=groups[pool]).groupby(level=0).max()
        G, gids = gdf.to_numpy(), gdf.index.to_numpy()
        base_part = part.copy()
        common = M[pool].mean(0) > 0.02  # judge balance on tags that are not vanishingly rare
        best = None
        for s in range(a.seed, a.seed + a.tries):
            trial = base_part.copy()
            keep, take = group_split_vectors(G, a.test_size, s)
            trial[np.isin(groups, list(gids[take])) & (trial == "train")] = "test"
            rem, Grem = gids[keep], G[keep]
            if a.holdout_size > 0:
                k2, t2 = group_split_vectors(G[keep], a.holdout_size / (1 - a.test_size), s + 10_000)
                trial[np.isin(groups, list(rem[t2])) & (trial == "train")] = "holdout"
                rem, Grem = rem[k2], G[keep][k2]
            shares = np.array([M[trial == p][:, common].mean(0) for p in ("train", "holdout", "test") if (trial == p).any()])
            gap = float((shares.max(0) - shares.min(0)).max())
            if best is None or gap < best[0]:
                best = (gap, s, trial, rem, Grem)
        gap, chosen_seed, part, rem, Grem = best
        print(f"balanced-partition search: {a.tries} seeds, chosen seed {chosen_seed}, max tag-share gap {gap:.3f}")
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        gfold = {}
        for k, (_, va) in enumerate(MultilabelStratifiedKFold(n_splits=a.n_splits, shuffle=True, random_state=a.seed).split(np.zeros(len(Grem)), Grem)):
            gfold.update({g: k for g in rem[va]})
        tr = part == "train"
        fold[tr] = [gfold[g] for g in groups[tr]]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame({"id": df[a.key], "part": part, "fold": fold})
    res.to_csv(out / "split.csv", index=False)

    def digest(p):
        return hashlib.sha256("\n".join(sorted(res.loc[res.part == p, "id"])).encode()).hexdigest()

    parts = [p for p in ("train", "holdout", "test", "external") if (part == p).any()]
    overlap = {f"{p}&{q}": len(set(groups[part == p]) & set(groups[part == q]))
               for i, p in enumerate(parts) for q in parts[i + 1:]}
    balance = {}
    if M is not None:
        for p in parts:
            balance[p] = {t: round(float(M[part == p, j].mean()), 4) for j, t in enumerate(tags)}
    spec = {"scheme": a.scheme, "params": {k: v for k, v in vars(a).items() if k not in ("metadata", "out")},
            "sizes": res.part.value_counts().to_dict(),
            "folds": {int(k): int(v) for k, v in res.loc[res.part == "train", "fold"].value_counts().sort_index().items()},
            "group_overlap_between_parts": overlap, "tag_share_by_part": balance,
            "test_ids_sha256": digest("test"), "external_ids_sha256": digest("external") if "external" in parts else None}
    (out / "split_spec.json").write_text(json.dumps(spec, indent=1))
    print(json.dumps({k: spec[k] for k in ("scheme", "sizes", "folds", "group_overlap_between_parts")}, indent=1))
    if balance:
        worst = max(((t, max(balance[p][t] for p in parts) - min(balance[p][t] for p in parts)) for t in tags), key=lambda x: x[1])
        print(f"largest tag-share gap between parts: {worst[0]} ({worst[1]:.3f})")
    print(f"written: {out / 'split.csv'}, {out / 'split_spec.json'}")


if __name__ == "__main__":
    main()
