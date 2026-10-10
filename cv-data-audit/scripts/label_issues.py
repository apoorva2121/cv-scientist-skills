"""Stage 3 label quality: out-of-fold linear-probe probabilities on embeddings + confident learning.

Usage:
    python label_issues.py embeddings.npz metadata.csv --label label [--group capture_session] [--key file]
        [--multilabel "|"] [--folds 5] [--out label_issues.csv]

1. Joins embeddings (from embed_images.py) with labels.
2. Fits a logistic-regression probe with (group-aware) stratified K-fold and keeps the OUT-OF-FOLD
   probabilities: no image is scored by a model that saw it, or saw its group.
3. Runs cleanlab's confident learning (cleanlab.filter.find_label_issues; for multi-label,
   cleanlab.multilabel_classification.filter.find_label_issues) and ranks suspects by label quality.
The out-of-fold accuracy is also a free linear-probe baseline (Stage 6).
Suspects are candidates for human review, never automatic deletions. Groups matter: without them,
near-copies in other folds vouch for a wrong label and hide it.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("embeddings")
    ap.add_argument("metadata")
    ap.add_argument("--label", required=True)
    ap.add_argument("--key", default="file")
    ap.add_argument("--group", default="")
    ap.add_argument("--multilabel", default="", help="separator for multi-label strings, e.g. '|'")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out", default="label_issues.csv")
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    z = np.load(a.embeddings, allow_pickle=False)
    emb = pd.DataFrame(z["emb"], index=pd.Index(z["files"].astype(str), name=a.key))
    meta = pd.read_csv(a.metadata)
    meta[a.key] = meta[a.key].astype(str)
    df = meta.set_index(a.key).join(emb, how="inner")
    X = df[emb.columns].to_numpy(np.float32) * 10.0  # unit-norm embeddings: scale so C=1 is not over-regularised
    groups = df[a.group].to_numpy() if a.group else None

    if a.multilabel:
        tags = df[a.label].astype(str).str.split(a.multilabel)
        classes = sorted({t for ts in tags for t in ts if t})
        Y = np.array([[c in set(ts) for c in classes] for ts in tags], dtype=int)
        probs = np.zeros(Y.shape, dtype=float)
        splitter = GroupKFold(a.folds) if a.group else KFold(a.folds, shuffle=True, random_state=0)
        for tr, va in splitter.split(X, groups=groups):
            for k in range(len(classes)):
                if Y[tr, k].min() == Y[tr, k].max():
                    probs[va, k] = Y[tr, k].mean()
                    continue
                m = LogisticRegression(C=a.C, max_iter=2000).fit(X[tr], Y[tr, k])
                probs[va, k] = m.predict_proba(X[va])[:, 1]
        from cleanlab.multilabel_classification.filter import find_label_issues
        from cleanlab.multilabel_classification.rank import get_label_quality_scores
        lab = [list(np.flatnonzero(r)) for r in Y]
        issues = find_label_issues(labels=lab, pred_probs=probs, return_indices_ranked_by="self_confidence")
        quality = get_label_quality_scores(lab, probs)
        pred = [a.multilabel.join(classes[k] for k in np.flatnonzero(p >= 0.5)) for p in probs]
        map_score = float(np.mean([(probs[:, k][Y[:, k] == 1].mean() if Y[:, k].any() else np.nan) for k in range(len(classes))]))
        print(f"multi-label: {len(classes)} labels; mean out-of-fold probability on true tags {map_score:.3f}")
    else:
        y_str = df[a.label].astype(str).to_numpy()
        classes = sorted(set(y_str))
        y = np.array([classes.index(v) for v in y_str])
        splitter = (StratifiedGroupKFold(a.folds, shuffle=True, random_state=0) if a.group
                    else StratifiedKFold(a.folds, shuffle=True, random_state=0))
        probs = np.zeros((len(y), len(classes)))
        for tr, va in splitter.split(X, y, groups):
            m = LogisticRegression(C=a.C, max_iter=3000).fit(X[tr], y[tr])
            probs[np.ix_(va, m.classes_)] = m.predict_proba(X[va])
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
        issues = find_label_issues(labels=y, pred_probs=probs, return_indices_ranked_by="self_confidence")
        quality = get_label_quality_scores(y, probs)
        pred = [classes[k] for k in probs.argmax(1)]
        acc = float((probs.argmax(1) == y).mean())
        per_class = pd.Series(probs.argmax(1) == y).groupby(y_str).mean().round(3).to_dict()
        print(f"out-of-fold linear-probe accuracy {acc:.3f} (a Stage 6 baseline); per-class recall {per_class}")

    out = pd.DataFrame({a.key: df.index, "given_label": df[a.label].astype(str).to_numpy(), "predicted": pred,
                        "label_quality": np.round(quality, 4)})
    out["suspected_issue"] = False
    out.loc[out.index[issues], "suspected_issue"] = True
    out["issue_rank"] = np.nan
    out.loc[out.index[issues], "issue_rank"] = np.arange(1, len(issues) + 1)
    out.sort_values(["suspected_issue", "label_quality"], ascending=[False, True]).to_csv(a.out, index=False)
    rate = len(issues) / max(len(out), 1)
    print(f"images={len(out)} suspected label issues={len(issues)} ({rate:.1%}); "
          f"by given label: {out[out.suspected_issue].given_label.value_counts().to_dict()}")
    print(f"written: {a.out} (review the top of the list visually before relabelling or dropping)")


if __name__ == "__main__":
    main()
