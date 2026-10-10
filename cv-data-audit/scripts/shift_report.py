"""Stage 3 shift report: how different is deployment data from training data?

Usage:
    python shift_report.py --train-emb train.npz --deploy-emb deploy.npz
        [--train-audit train_audit.csv --deploy-audit deploy_audit.csv] [--out shift_report.csv]

Three views:
  1. Image statistics (from image_audit.py): brightness, contrast, blur, size - medians and the
     Kolmogorov-Smirnov statistic per feature. Tells you WHAT changed (darker? blurrier? smaller?).
  2. Adversarial validation on embeddings: cross-validated ROC-AUC of a classifier separating train
     from deployment images. About 0.5 = indistinguishable; near 1.0 = a different world.
  3. Novelty: each deployment image's cosine distance to its nearest training image, compared with
     train-to-train nearest distances. The share of deployment images beyond the 99th percentile of
     train-to-train distances is the share the model has effectively never seen anything like.
Use the result to choose preprocessing and augmentation (Stages 4-5) that cover the gap, and to
decide whether a labelled deployment sample is needed for honest evaluation.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-emb", required=True)
    ap.add_argument("--deploy-emb", required=True)
    ap.add_argument("--train-audit", default="")
    ap.add_argument("--deploy-audit", default="")
    ap.add_argument("--max-n", type=int, default=5000)
    ap.add_argument("--out", default="shift_report.csv")
    a = ap.parse_args()

    from scipy.stats import ks_2samp
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(0)
    tr = np.load(a.train_emb)["emb"]
    de = np.load(a.deploy_emb)["emb"]
    tr = tr[rng.permutation(len(tr))[: a.max_n]]
    de = de[rng.permutation(len(de))[: a.max_n]]
    rows = []

    if a.train_audit and a.deploy_audit:
        ta, da = pd.read_csv(a.train_audit), pd.read_csv(a.deploy_audit)
        ta, da = ta[ta.readable], da[da.readable]
        print("image statistics (train median -> deployment median, KS statistic):")
        for c in ["mean", "std", "blur_var", "width", "height", "clipped_dark", "clipped_bright"]:
            if c in ta and c in da:
                ks = ks_2samp(ta[c].dropna(), da[c].dropna()).statistic
                rows.append({"view": "stat", "feature": c, "train_median": ta[c].median(), "deploy_median": da[c].median(), "ks": round(ks, 3)})
                print(f"  {c:15s} {ta[c].median():10.2f} -> {da[c].median():10.2f}   KS {ks:.3f}{'   <- shifted' if ks > 0.2 else ''}")

    n = min(len(tr), len(de))
    X = np.vstack([tr[:n], de[:n]]) * 10
    y = np.r_[np.zeros(n), np.ones(n)]
    auc = cross_val_score(LogisticRegression(C=1.0, max_iter=3000), X, y, cv=5, scoring="roc_auc").mean()
    rows.append({"view": "adversarial", "feature": "embedding_auc", "value": round(float(auc), 3)})
    print(f"\nadversarial validation ROC-AUC (train vs deployment, embeddings): {auc:.3f}")

    s = rng.permutation(len(tr))[: min(2000, len(tr))]
    sims_tt = tr[s] @ tr.T
    sims_tt[np.arange(len(s)), s] = -1  # exclude self
    nn_tt = 1 - sims_tt.max(1)
    nn_dt = 1 - (de @ tr.T).max(1)
    thr = np.percentile(nn_tt, 99)
    novel = float((nn_dt > thr).mean())
    rows.append({"view": "novelty", "feature": "deploy_share_beyond_train_p99", "value": round(novel, 3)})
    print(f"nearest-neighbour cosine distance: train->train median {np.median(nn_tt):.3f}, "
          f"deployment->train median {np.median(nn_dt):.3f}; deployment beyond train p99: {novel:.1%}")
    verdict = ("indistinguishable" if auc < 0.6 else "moderate shift" if auc < 0.8 else "strong shift")
    print(f"verdict: {verdict}. Strong shift means internal validation will overstate deployment performance.")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
