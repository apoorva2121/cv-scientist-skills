"""Stage 9 ensembling for classification: does combining models beat the best single model on the SAME images?

Usage:
    python ensemble.py runs/cand_a runs/cand_b runs/cand_c [--metric macro_f1|accuracy]
    python ensemble.py run_a/val_probs.npz run_b/val_probs.npz [...]
        [--folds 5] [--meta metadata.csv --key file --group split_group] [--out ensemble_report.json]

Each input is a val_probs.npz (files, labels, probs, classes) from cv-training/scripts/train_classifier.py,
or a candidate FOLDER laid out as <candidate>/<fold_seed>/val_probs.npz: its folds are joined into
out-of-fold predictions over the whole train part (seeds of the same image are averaged). Prefer folders:
every train image then counts as evidence, not one fold. Methods compared:
  single       every model alone
  mean         average of probabilities
  greedy       Caruana ensemble selection: forward selection with replacement, maximising the metric;
               the weights are selected on K-1 folds of the validation images and scored on the held-out
               fold (cross-fitted), so the selection cannot score the images it was chosen on
  weighted     non-negative weights summing to 1 (softmax parameterisation, Nelder-Mead), cross-fitted
  stacking     multinomial logistic regression on the concatenated log-probabilities, cross-fitted
  recal        the same logistic regression on the best single model's log-probabilities alone
               (vector/matrix-scaling recalibration). Stacking must beat THIS, not just the raw single
               model: a stacker also corrects one model's class bias and calibration, which a single
               model gets from post-processing at no extra inference cost
Cross-fitting folds respect groups when --meta/--group are given (near-duplicates or images of one
subject must not sit on both sides of a stacker fold); the bootstrap then resamples whole groups.
Every method is compared with the best single model by a paired bootstrap over images
(cv-method-comparison/scripts/compare.py). Keep an ensemble only if the interval excludes zero AND the
added inference cost fits the latency budget; otherwise keep the single model (or distil the ensemble).
For detection use weighted box fusion (ensemble-boxes); for segmentation average the per-pixel
probabilities; the same paired-bootstrap rule applies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def metric_fn(name: str):
    from sklearn.metrics import f1_score
    if name == "accuracy":
        return lambda y, p: float((y == p.argmax(1)).mean())
    return lambda y, p: float(f1_score(y, p.argmax(1), average="macro"))


def load_run(path: str) -> dict:
    p = Path(path)
    if p.is_file():
        z = np.load(p)
        return {k: z[k] for k in ("files", "labels", "probs", "classes")}
    parts = sorted(p.rglob("val_probs.npz"))
    if not parts:
        raise SystemExit(f"no val_probs.npz under {p}")
    acc, lab, classes = {}, {}, None
    for q in parts:
        z = np.load(q)
        c = list(z["classes"].astype(str))
        if classes is None:
            classes = c
        assert c == classes, f"class order differs in {q}"
        for f, l, pr in zip(z["files"].astype(str), z["labels"], z["probs"]):
            acc.setdefault(f, []).append(pr)
            lab[f] = l
    files = np.array(sorted(acc))
    return {"files": files, "labels": np.array([lab[f] for f in files]),
            "probs": np.stack([np.mean(acc[f], axis=0) for f in files]), "classes": np.array(classes)}


def greedy_select(P: list[np.ndarray], y: np.ndarray, score, rounds: int = 25) -> np.ndarray:
    counts = np.zeros(len(P))
    current = None
    for _ in range(rounds):
        best_k, best_s = None, -1
        for k in range(len(P)):
            cand = P[k] if current is None else (current * counts.sum() + P[k]) / (counts.sum() + 1)
            s = score(y, cand)
            if s > best_s:
                best_k, best_s = k, s
        counts[best_k] += 1
        current = P[best_k] if current is None else (current * (counts.sum() - 1) + P[best_k]) / counts.sum()
    return counts / counts.sum()


def fit_weights(P: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    def nll(z):
        w = np.exp(z - z.max())
        w = w / w.sum()
        mix = sum(wi * pi for wi, pi in zip(w, P))
        return -float(np.mean(np.log(np.clip(mix[np.arange(len(y)), y], 1e-9, 1))))

    z = minimize(nll, np.zeros(len(P)), method="Nelder-Mead", options={"maxiter": 400}).x
    w = np.exp(z - z.max())
    return w / w.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--metric", default="macro_f1", choices=["macro_f1", "accuracy"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--meta", default="")
    ap.add_argument("--key", default="file")
    ap.add_argument("--group", default="")
    ap.add_argument("--out", default="ensemble_report.json")
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    from compare import paired_bootstrap

    runs = [load_run(p) for p in a.inputs]
    parts = [(Path(p).resolve() if Path(p).is_dir() else Path(p).resolve().with_suffix("")).parts for p in a.inputs]
    for depth in range(1, max(len(x) for x in parts) + 1):
        names = ["/".join(x[-depth:]) for x in parts]
        if len(set(names)) == len(names):
            break
    files0 = runs[0]["files"].astype(str)
    classes = list(runs[0]["classes"].astype(str))
    order = [np.argsort(r["files"].astype(str)) for r in runs]
    common = set(files0)
    for r in runs[1:]:
        common &= set(r["files"].astype(str))
    if len(common) != len(files0):
        print(f"WARNING: runs share {len(common)} of {len(files0)} validation images; using the intersection")
    keep = np.array(sorted(common))
    P, y = [], None
    for r in runs:
        idx = {f: i for i, f in enumerate(r["files"].astype(str))}
        rows = [idx[f] for f in keep]
        P.append(r["probs"][rows].astype(np.float64))
        if y is None:
            y = r["labels"][rows]
        assert list(r["classes"].astype(str)) == classes, "runs must share the class order"
    score = metric_fn(a.metric)

    singles = {n: score(y, p) for n, p in zip(names, P)}
    best_name = max(singles, key=singles.get)
    preds = {f"single:{n}": p for n, p in zip(names, P)}
    preds["mean"] = sum(P) / len(P)

    groups = None
    if a.meta and a.group:
        import pandas as pd
        meta = pd.read_csv(a.meta)
        gmap = dict(zip(meta[a.key].astype(str), meta[a.group].astype(str)))
        groups = np.array([gmap.get(f, f) for f in keep])
        splitter = StratifiedGroupKFold(a.folds, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups)
    else:
        print("NOTE: cross-fitting ignores groups; pass --meta/--group if validation images are grouped")
        splitter = StratifiedKFold(a.folds, shuffle=True, random_state=0).split(np.zeros(len(y)), y)
    recal_p = np.zeros_like(P[0])
    Zb = np.log(np.clip(P[names.index(best_name)], 1e-9, 1))
    greedy_p = np.zeros_like(P[0])
    weighted_p = np.zeros_like(P[0])
    stack_p = np.zeros_like(P[0])
    W_g, W_w = [], []
    Z = np.concatenate([np.log(np.clip(p, 1e-9, 1)) for p in P], axis=1)
    for tr, te in splitter:
        wg = greedy_select([p[tr] for p in P], y[tr], score)
        ww = fit_weights([p[tr] for p in P], y[tr])
        W_g.append(wg)
        W_w.append(ww)
        greedy_p[te] = sum(w * p[te] for w, p in zip(wg, P))
        weighted_p[te] = sum(w * p[te] for w, p in zip(ww, P))
        lr = LogisticRegression(C=0.5, max_iter=3000).fit(Z[tr], y[tr])
        stack_p[np.ix_(te, lr.classes_)] = lr.predict_proba(Z[te])
        lb = LogisticRegression(C=0.5, max_iter=3000).fit(Zb[tr], y[tr])
        recal_p[np.ix_(te, lb.classes_)] = lb.predict_proba(Zb[te])
    preds.update({"greedy": greedy_p, "weighted": weighted_p, "recal": recal_p, "stacking": stack_p})

    def records(p):
        return list(zip(y, p.argmax(1)))

    def m(recs):
        yy = np.array([r[0] for r in recs])
        pp = np.array([r[1] for r in recs])
        from sklearn.metrics import f1_score
        return float((yy == pp).mean()) if a.metric == "accuracy" else float(f1_score(yy, pp, average="macro"))

    report = {"metric": a.metric, "n_images": int(len(y)), "best_single": best_name, "results": {}}
    base = records(preds[f"single:{best_name}"])
    for k, p in preds.items():
        entry = {"score": round(score(y, p), 4)}
        if not k.startswith("single:"):
            bt = paired_bootstrap(records(p), base, m, n_boot=1000, groups=groups)
            entry.update({"vs_best_single": round(bt["diff"], 4), "ci": [round(bt["ci_low"], 4), round(bt["ci_high"], 4)],
                          "verdict": bt["verdict"]})
            if k == "stacking":
                br = paired_bootstrap(records(p), records(preds["recal"]), m, n_boot=1000, groups=groups)
                entry.update({"vs_recal": round(br["diff"], 4), "ci_vs_recal": [round(br["ci_low"], 4), round(br["ci_high"], 4)],
                              "verdict_vs_recal": br["verdict"]})
        report["results"][k] = entry
    report["greedy_mean_weights"] = dict(zip(names, np.round(np.mean(W_g, 0), 3).tolist()))
    report["weighted_mean_weights"] = dict(zip(names, np.round(np.mean(W_w, 0), 3).tolist()))
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"{len(y)} validation images, metric {a.metric}; best single = {best_name}")
    for k, e in report["results"].items():
        extra = f"  vs best single {e['vs_best_single']:+.4f} {e['ci']} -> {e['verdict']}" if "verdict" in e else ""
        print(f"  {k:28s} {e['score']:.4f}{extra}")
        if "verdict_vs_recal" in e:
            print(f"  {'':28s}        vs recal       {e['vs_recal']:+.4f} {e['ci_vs_recal']} -> {e['verdict_vs_recal']}")
    print(f"greedy weights {report['greedy_mean_weights']}\nweighted weights {report['weighted_mean_weights']}")


if __name__ == "__main__":
    main()
