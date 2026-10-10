"""Stage 8 for classification: calibration and operating points, fitted on validation predictions and measured
honestly (cross-fitted), then frozen for the test set.

Usage:
  calibration     python postprocess_cls.py calibrate --val runs/cand_x [--folds 5] [--out post.json]
  operating point python postprocess_cls.py threshold --val runs/cand_x --positive defect
                      (--target-recall 0.95 | --max-f1 | --cost-fp 1 --cost-fn 20) [--folds 5] [--out post.json]
  multi-label     python postprocess_cls.py multilabel --val val_probs.npz [--folds 5] [--out post.json]
  apply to test   python postprocess_cls.py apply --post post.json --pred test_probs.npz --out test_post.npz

--val takes a val_probs.npz or a candidate folder (folds joined into out-of-fold predictions, seeds averaged),
like cv-method-comparison/scripts/ensemble.py.

calibrate   Temperature scaling (one scalar T on log-probabilities, minimising NLL). It never changes the
            predicted class, only confidence. It suits neural-network softmax outputs; the verdict requires the
            held-out Brier score to improve as well as NLL, because NLL alone can be "fixed" by a large T that
            wrecks calibration when some probabilities are exactly zero (k-NN, trees). Reports NLL, Brier and ECE before and after, cross-fitted
            (T fitted on K-1 folds of the validation images, scored on the held-out fold), then refits T on
            all validation images for deployment. Skip it when calibrated probabilities are not used.
threshold   Binary or one-vs-rest decision for a --positive class: the threshold that reaches a target
            recall, maximises F1, or minimises expected cost. Cross-fitted: reports the recall, precision
            and cost the chosen rule actually achieves on held-out folds (with Wilson intervals), which is
            what to expect at deployment; the final threshold is refitted on all validation images.
multilabel  Per-class F1-optimal thresholds for sigmoid outputs (npz with labels as a 0/1 matrix), cross-
            fitted, compared with the global 0.5 threshold on macro-F1.
apply       Applies the frozen choices to new predictions (test or deployment): temperature, threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cv-method-comparison" / "scripts"))


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def softmax_t(logp: np.ndarray, t: float) -> np.ndarray:
    z = logp / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logp: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    def nll(log_t):
        p = softmax_t(logp, float(np.exp(log_t)))
        return -float(np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    conf, pred = p.max(1), p.argmax(1)
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum((bins == b).mean() * abs((pred[bins == b] == y[bins == b]).mean() - conf[bins == b].mean())
              for b in range(15) if (bins == b).any())
    onehot = np.eye(p.shape[1])[y]
    return {"nll": round(-float(np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))), 4),
            "brier": round(float(np.mean(((p - onehot) ** 2).sum(1))), 4), "ece_15bin": round(float(ece), 4),
            "accuracy": round(float((pred == y).mean()), 4)}


def splitter(y, folds, seed=0):
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def choose_threshold(s: np.ndarray, pos: np.ndarray, a) -> float:
    cand = np.unique(np.concatenate([s, [0.0, 1.0]]))
    if a.target_recall:
        n_pos = max(int(pos.sum()), 1)
        ok = [t for t in cand if (s[pos] >= t).sum() / n_pos >= a.target_recall]
        return float(max(ok)) if ok else 0.0
    best_t, best_v = 0.5, None
    for t in cand:
        pr = s >= t
        tp, fp, fn = int((pr & pos).sum()), int((pr & ~pos).sum()), int((~pr & pos).sum())
        v = (2 * tp / max(2 * tp + fp + fn, 1)) if a.max_f1 else -(a.cost_fp * fp + a.cost_fn * fn)
        if best_v is None or v > best_v:
            best_t, best_v = float(t), v
    return best_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "threshold", "multilabel", "apply"])
    ap.add_argument("--val", default="")
    ap.add_argument("--pred", default="")
    ap.add_argument("--post", default="")
    ap.add_argument("--positive", default="")
    ap.add_argument("--target-recall", type=float, default=0.0)
    ap.add_argument("--max-f1", action="store_true")
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--cost-fn", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="post.json")
    a = ap.parse_args()

    if a.mode == "apply":
        post = json.loads(Path(a.post).read_text())
        z = np.load(a.pred)
        p = z["probs"].astype(np.float64)
        if "temperature" in post:
            p = softmax_t(np.log(np.clip(p, 1e-12, 1)), post["temperature"])
        out = {k: z[k] for k in z.files}
        out["probs"] = p.astype(np.float32)
        if "threshold" in post:
            classes = list(z["classes"].astype(str))
            k = classes.index(post["positive"])
            out["decision_positive"] = p[:, k] >= post["threshold"]
        np.savez(a.out, **out)
        print(f"applied {', '.join(x for x in ('temperature', 'threshold') if x in post)} -> {a.out}")
        return

    from ensemble import load_run
    run = load_run(a.val)
    y, P, classes = run["labels"].astype(int), run["probs"].astype(np.float64), list(run["classes"].astype(str))
    report = {"mode": a.mode, "n_images": int(len(y)), "source": a.val}

    if a.mode == "calibrate":
        logp = np.log(np.clip(P, 1e-12, 1))
        cal = np.zeros_like(P)
        ts = []
        for tr, te in splitter(y, a.folds):
            t = fit_temperature(logp[tr], y[tr])
            ts.append(t)
            cal[te] = softmax_t(logp[te], t)
        report["before"], report["after_crossfitted"] = scores(P, y), scores(cal, y)
        report["temperature_per_fold"] = [round(t, 3) for t in ts]
        report["temperature"] = round(fit_temperature(logp, y), 4)
        b, af = report["before"], report["after_crossfitted"]
        if af["brier"] < b["brier"] and af["nll"] <= b["nll"]:
            report["verdict"] = "use the temperature: held-out Brier and NLL both improve"
        elif af["nll"] < b["nll"]:
            report["verdict"] = ("mixed: NLL improves but Brier/ECE do not. Typical of probabilities with exact zeros "
                                 "(k-NN vote fractions, trees): NLL is dominated by the few zero-probability mistakes, so "
                                 "the fitted T flattens every prediction. Instead: for k-NN refit the vote kernel "
                                 "softmax(sim / T) on out-of-fold data; otherwise isotonic regression (1,000+ samples) "
                                 "or histogram binning; or leave uncalibrated")
        else:
            report["verdict"] = "skip: temperature scaling does not improve held-out scores"
    elif a.mode == "threshold":
        if a.positive not in classes:
            raise SystemExit(f"--positive must be one of {classes}")
        if not (a.target_recall or a.max_f1 or a.cost_fp != 1.0 or a.cost_fn != 1.0):
            raise SystemExit("choose a rule: --target-recall R, --max-f1, or --cost-fp/--cost-fn")
        k = classes.index(a.positive)
        s, pos = P[:, k], y == k
        dec = np.zeros(len(y), dtype=bool)
        chosen = []
        for tr, te in splitter(pos.astype(int), a.folds):
            t = choose_threshold(s[tr], pos[tr], a)
            chosen.append(t)
            dec[te] = s[te] >= t
        tp, fp, fn = int((dec & pos).sum()), int((dec & ~pos).sum()), int((~dec & pos).sum())
        report["rule"] = (f"recall >= {a.target_recall}" if a.target_recall else "max F1" if a.max_f1
                          else f"min cost (FP {a.cost_fp}, FN {a.cost_fn})")
        report["crossfitted"] = {"recall": round(tp / max(tp + fn, 1), 4), "recall_wilson95": wilson(tp, tp + fn),
                                 "precision": round(tp / max(tp + fp, 1), 4), "precision_wilson95": wilson(tp, tp + fp),
                                 "false_positive_rate": round(fp / max(int((~pos).sum()), 1), 4),
                                 "cost": a.cost_fp * fp + a.cost_fn * fn, "tp": tp, "fp": fp, "fn": fn}
        argmax = P.argmax(1) == k
        tp0, fp0, fn0 = int((argmax & pos).sum()), int((argmax & ~pos).sum()), int((~argmax & pos).sum())
        report["argmax_rule"] = {"recall": round(tp0 / max(tp0 + fn0, 1), 4), "precision": round(tp0 / max(tp0 + fp0, 1), 4),
                                 "cost": a.cost_fp * fp0 + a.cost_fn * fn0}
        report["threshold_per_fold"] = [round(t, 4) for t in chosen]
        report["threshold"] = round(choose_threshold(s, pos, a), 4)
        report["positive"] = a.positive
        report["note"] = "the cross-fitted numbers are the honest expectation; the refitted threshold is what ships"
    else:  # multilabel
        Y = run["labels"].astype(int)
        if Y.ndim != 2:
            raise SystemExit("multilabel mode needs labels as an (images x classes) 0/1 matrix")
        from sklearn.metrics import f1_score
        from sklearn.model_selection import KFold
        pred = np.zeros_like(Y)
        for tr, te in KFold(a.folds, shuffle=True, random_state=0).split(Y):
            for c in range(Y.shape[1]):
                cand = np.unique(np.concatenate([P[tr, c], [0.5]]))
                f1s = [f1_score(Y[tr, c], P[tr, c] >= t, zero_division=0) for t in cand]
                pred[te, c] = P[te, c] >= cand[int(np.argmax(f1s))]
        report["macro_f1_at_0.5"] = round(float(f1_score(Y, P >= 0.5, average="macro", zero_division=0)), 4)
        report["macro_f1_per_class_thresholds_crossfitted"] = round(float(f1_score(Y, pred, average="macro", zero_division=0)), 4)
        th = []
        for c in range(Y.shape[1]):
            cand = np.unique(np.concatenate([P[:, c], [0.5]]))
            th.append(float(cand[int(np.argmax([f1_score(Y[:, c], P[:, c] >= t, zero_division=0) for t in cand]))]))
        report["thresholds"] = dict(zip(classes, [round(t, 4) for t in th]))
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
