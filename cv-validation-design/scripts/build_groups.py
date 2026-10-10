"""Stage 2: merge every grouping signal into one split key, so no leak path crosses a split.

Usage:
    python build_groups.py metadata.csv --key file --groups capture_session[,patient_id,video_id]
        [--dups duplicate_clusters.csv] [--out metadata_grouped.csv]

Rows are joined (union-find) when they share a value in ANY listed group column, or sit in the same
duplicate cluster from cv-data-audit/scripts/find_duplicates.py. The result column `split_group` is
the key every splitter must use. Typical leak this closes: a photo re-uploaded under a new session
id, a patient scanned at two sites, a video re-encoded under a new name.
Reports the number of groups and the share of the largest one: a group holding more than about 10%
of rows makes balanced folds impossible; inspect it (a chain of weak near-duplicate links, or one
huge source that deserves its own external test).
"""
from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata")
    ap.add_argument("--key", default="file")
    ap.add_argument("--groups", default="", help="comma-separated group columns")
    ap.add_argument("--dups", default="", help="duplicate_clusters.csv from find_duplicates.py")
    ap.add_argument("--out", default="metadata_grouped.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.metadata)
    n = len(df)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    cols = [c.strip() for c in a.groups.split(",") if c.strip()]
    for c in cols:
        for _, idx in df.groupby(c, dropna=True).indices.items():
            for j in idx[1:]:
                union(int(idx[0]), int(j))
    before = len({find(i) for i in range(n)})
    merged_by_dups = 0
    if a.dups:
        d = pd.read_csv(a.dups)
        pos = {k: i for i, k in enumerate(df[a.key].astype(str))}
        for _, files in d.groupby("cluster")["file"]:
            ids = [pos[f] for f in files.astype(str) if f in pos]
            for j in ids[1:]:
                if find(ids[0]) != find(j):
                    merged_by_dups += 1
                union(ids[0], j)
    roots = [find(i) for i in range(n)]
    df["split_group"] = pd.factorize(pd.Series(roots))[0]
    sizes = df["split_group"].value_counts()
    df.to_csv(a.out, index=False)
    print(f"rows={n}  groups from {cols or 'rows'}: {before}  merges added by duplicate clusters: {merged_by_dups}  "
          f"final groups: {len(sizes)}")
    print(f"largest group: {sizes.iloc[0]} rows ({sizes.iloc[0] / n:.1%}); groups with 1 row: {int((sizes == 1).sum())}")
    if sizes.iloc[0] / n > 0.10:
        print("WARNING: one group holds over 10% of rows; inspect it before splitting")
    print(f"written: {a.out} (column split_group)")


if __name__ == "__main__":
    main()
