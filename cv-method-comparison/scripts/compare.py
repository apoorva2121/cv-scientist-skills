"""Decide whether one vision method really beats another: paired tests on the SAME validation images.

    from compare import summarize_folds, paired_bootstrap
    # (a) scores per fold (and per seed) for several candidates -> keep/drop table
    table = summarize_folds(long_df)        # long_df columns: name, fold, seed, score, n_train, n_val
    # (b) one validation set, a dataset-level metric (mAP, mIoU, macro-F1) computed from per-image data
    res = paired_bootstrap(per_image_a, per_image_b, metric_fn, n_boot=2000)

Command line (screening runs from cv-training/scripts/train_classifier.py or any script writing the same
metrics.json): lay runs out as ROOT/<candidate>/<run>/metrics.json, one run per fold and seed, then
    python compare.py ROOT [--metric macro_f1] [--epoch last|best] [--out table.csv]
--epoch last (default) scores the final epoch: choosing the best epoch on the fold that is also scored
flatters every candidate, and flatters longer or noisier schedules most.

(a) Averages seeds within each fold first (seeds reduce training noise; folds are the independent
    evidence), pairs candidates by fold, and uses the Nadeau-Bengio corrected variance
    (1/n + n_val/n_train) because CV training sets overlap. keep = not worse than the best by more than
    one corrected SE. Ties go to the simpler or cheaper candidate.
(b) Resamples validation images with replacement, recomputes the metric for BOTH candidates on each
    resample (so image difficulty cancels), and returns the mean difference, its 95% interval and the
    share of resamples where A beats B. For grouped data pass `groups` to resample whole groups.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


def summarize_folds(long: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    sign = 1 if higher_is_better else -1
    wide = long.groupby(["fold", "name"])["score"].mean().unstack()
    ratio = float((long["n_val"] / long["n_train"]).mean()) if {"n_val", "n_train"} <= set(long.columns) else 0.25
    means = wide.mean()
    best = (sign * means).idxmax()
    rows = []
    for name in wide.columns:
        d = sign * (wide[name] - wide[best])
        n = int(d.notna().sum())
        se = float(np.sqrt((1 / n + ratio) * d.var(ddof=1))) if n > 1 else float("nan")
        seeds = long[long.name == name].groupby("seed")["score"].mean()
        rows.append({"name": name, "mean": round(float(means[name]), 4), "sd_folds": round(float(wide[name].std(ddof=1)), 4),
                     "sd_seeds": round(float(seeds.std(ddof=1)), 4) if len(seeds) > 1 else np.nan,
                     "diff_vs_best": round(float(d.mean()), 4), "se_corrected": round(se, 4) if np.isfinite(se) else np.nan,
                     "n_folds": n, "keep": bool(name == best or (np.isfinite(se) and d.mean() >= -se))})
    if len(wide) < 5:
        print(f"WARNING: only {len(wide)} folds; treat keep decisions as provisional or add repeats/seeds")
    return pd.DataFrame(rows).sort_values("mean", ascending=not higher_is_better).reset_index(drop=True)


def paired_bootstrap(data_a: Sequence, data_b: Sequence, metric: Callable[[Sequence], float], n_boot: int = 2000,
                     groups: Sequence | None = None, seed: int = 0) -> dict:
    """data_a / data_b: per-image records for the same images in the same order (e.g. (y_true, y_pred) pairs,
    or per-image detection results); metric: computes the dataset-level score from a list of records."""
    rng = np.random.default_rng(seed)
    n = len(data_a)
    if groups is not None:
        g = pd.Series(np.arange(n)).groupby(np.asarray(groups)).apply(list).tolist()
    diffs = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([g[k] for k in rng.integers(0, len(g), len(g))])
        a = metric([data_a[i] for i in idx])
        b = metric([data_b[i] for i in idx])
        diffs.append(a - b)
    diffs = np.asarray(diffs)
    point = metric(list(data_a)) - metric(list(data_b))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi), "p_a_better": float((diffs > 0).mean()),
            "verdict": "A better" if lo > 0 else "B better" if hi < 0 else "tie (interval includes 0)"}


def accuracy(records) -> float:
    """Example metric for paired_bootstrap on classification records (y_true, y_pred)."""
    return float(np.mean([t == p for t, p in records]))


def macro_f1(records) -> float:
    from sklearn.metrics import f1_score
    y, p = zip(*records) if records else ([], [])
    return float(f1_score(y, p, average="macro"))


def collect_runs(root: str, metric: str = "macro_f1", epoch: str = "last"):
    """Build the long table (name, fold, seed, score, n_train, n_val) from ROOT/<candidate>/<run>/metrics.json."""
    import json
    from pathlib import Path
    rows = []
    for f in sorted(Path(root).rglob("metrics.json")):
        m = json.loads(f.read_text())
        cfg = m.get("config", {})
        if cfg.get("eval_part"):
            continue  # holdout runs are not fold evidence
        hist = m.get("history") or []
        if epoch == "last" and hist:
            last = hist[-1]
            score = last.get(f"ema_{metric}", last.get(metric))  # EMA weights are what an EMA run ships
        else:
            score = (m.get("best") or {}).get(metric)
        if score is None or not np.isfinite(score):
            continue
        name = f.parent.parent.relative_to(root).as_posix()
        rows.append({"name": name, "fold": cfg.get("fold", 0), "seed": cfg.get("seed", 0), "score": float(score),
                     "n_train": m.get("train_images", np.nan), "n_val": m.get("val_images", np.nan),
                     "minutes": m.get("minutes", np.nan)})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--metric", default="macro_f1")
    ap.add_argument("--epoch", default="last", choices=["last", "best"])
    ap.add_argument("--lower-is-better", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    long = collect_runs(a.root, a.metric, a.epoch)
    if long.empty:
        raise SystemExit(f"no metrics.json with '{a.metric}' under {a.root}")
    counts = long.groupby("name")["fold"].nunique()
    if counts.nunique() > 1:
        print(f"WARNING: candidates cover different folds {counts.to_dict()}; only shared folds are paired")
    table = summarize_folds(long, higher_is_better=not a.lower_is_better)
    table = table.merge(long.groupby("name")["minutes"].mean().round(2).rename("minutes_per_run").reset_index(), on="name")
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    if a.out:
        table.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
