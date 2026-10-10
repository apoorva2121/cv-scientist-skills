"""Stage 9 joint search: combine the 2-3 survivors of every stage and find the best COMBINATION.

Usage (everything after -- is the base training command; the search appends --fold/--epochs/--seed/--out):
    python joint_search.py --space search_space.json --folds 0,1,2 --epochs 8 --trials 30 --timeout-min 120 \
        --metric macro_f1 --out runs/joint -- \
        python ../../cv-training/scripts/train_classifier.py IMAGE_DIR --meta metadata_grouped.csv --label label --split split/

search_space.json: every carried-forward candidate carries the training-script arguments it stands for;
numeric knobs are ranges.
    {"stages": {"preprocessing": [{"name": "crop224", "args": "--eval-transform crop --img-size 224"}, ...],
                "augmentation":  [{"name": "trivialaugment", "args": "--aug trivialaugment"}, ...],
                "models":        [{"name": "convnext_nano", "args": "--model convnext_nano.in12k_ft_in1k"}, ...],
                "training":      [{"name": "ema", "args": "--ema 0.999"}, {"name": "plain", "args": ""}]},
     "knobs": {"lr": {"low": 1e-4, "high": 3e-3, "log": true}, "label-smoothing": {"choices": [0.0, 0.1]}}}

How it works
  - Optuna TPE over the stage choices and knobs. Trial 0 is the reference combination (the first candidate
    of every stage, knob midpoints), so the search always reports what it gained over the reference.
  - Early stopping of weak trials: the first fold reports the validation metric every epoch to a pruner
    (Hyperband successive halving from 10 trials; below that, the median rule after 3 unpruned trials).
    Nothing is pruned during the first third of the epochs. Pruned trials cost a fraction of a full run.
    Surviving trials run the remaining folds. The objective is the mean LAST-epoch score over folds.
  - Any training script works if it prints one JSON line per epoch containing "epoch" and the metric
    (EMA variants as "ema_<metric>" are preferred when present) and writes metrics.json into --out.
  - The end table re-compares the top trials with paired, overlap-corrected fold statistics
    (compare.summarize_folds). Among the trials that are kept, the cheapest is recommended
    (one-standard-error logic). Selecting among many trials on the same folds is optimistic: confirm the
    choice once on the Stage 2 holdout (--eval-part holdout) before Stage 10.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout-min", type=float, default=0, help="wall-clock budget for the whole search")
    ap.add_argument("--metric", default="macro_f1")
    ap.add_argument("--lower-is-better", action="store_true")
    ap.add_argument("--pruner", default="auto", choices=["auto", "hyperband", "median", "none"],
                    help="auto: median rule with 3 unpruned starter trials below 10 trials, Hyperband from 10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default="runs/joint")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    base = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not base:
        raise SystemExit("give the base training command after --")

    import optuna

    from compare import summarize_folds

    space = json.loads(Path(a.space).read_text())
    # post-processing is screened on saved predictions after training, never inside the search
    stages = {k: v for k, v in space.get("stages", {}).items() if v and k != "postprocessing"}
    for k, cands in stages.items():
        missing = [c.get("name") for c in cands if "args" not in c]
        if missing:
            raise SystemExit(f"stage '{k}': candidates {missing} have no 'args'; add the training arguments they stand for")
    knobs = space.get("knobs", {})
    folds = [int(f) for f in a.folds.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sign = -1 if a.lower_is_better else 1

    def run(args: list[str], fold: int, run_dir: Path, trial=None) -> tuple[float, float]:
        cmd = base + args + ["--fold", str(fold), "--epochs", str(a.epochs), "--seed", str(a.seed), "--out", str(run_dir)]
        t0, last = time.time(), None
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        log = open(run_dir.parent / f"f{fold}.log", "w", encoding="utf-8")
        try:
            for line in proc.stdout:
                log.write(line)
                if not line.startswith("{") or '"epoch"' not in line:
                    continue
                rec = json.loads(line)
                val = rec.get(f"ema_{a.metric}", rec.get(a.metric))
                if val is None:
                    continue
                last = float(val)
                if trial is not None:
                    trial.report(last, rec["epoch"])
                    if trial.should_prune():
                        proc.kill()
                        proc.wait()
                        raise optuna.TrialPruned()
            proc.wait()
        finally:
            log.close()
        if proc.returncode != 0 or last is None:
            raise RuntimeError(f"training failed (exit {proc.returncode}); see {run_dir.parent / f'f{fold}.log'}")
        return last, (time.time() - t0) / 60

    def objective(trial):
        args, picked = [], {}
        for stage, cands in stages.items():
            name = trial.suggest_categorical(stage, [c["name"] for c in cands])
            picked[stage] = name
            args += next(c for c in cands if c["name"] == name)["args"].split()
        for k, spec in knobs.items():
            if "choices" in spec:
                v = trial.suggest_categorical(k, spec["choices"])
            else:
                v = trial.suggest_float(k, spec["low"], spec["high"], log=spec.get("log", False))
            args += [f"--{k}", str(v)]
        tdir = out / f"trial_{trial.number:03d}"
        scores, minutes = [], 0.0
        for i, f in enumerate(folds):
            rd = tdir / f"f{f}_s{a.seed}"
            rd.mkdir(parents=True, exist_ok=True)
            s, m = run(args, f, rd, trial if i == 0 else None)
            scores.append(s)
            minutes += m
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("minutes", round(minutes, 2))
        trial.set_user_attr("args", " ".join(args))
        (tdir / "trial.json").write_text(json.dumps({"picked": picked, "params": trial.params, "args": args,
                                                     "fold_scores": scores, "minutes": minutes}, indent=1))
        print(f"trial {trial.number}: {picked} -> mean {np.mean(scores):.4f} over folds {folds} ({minutes:.1f} min)", flush=True)
        return float(np.mean(scores))

    # never prune on the first third of the epochs: early validation scores are noisy and favour fast starters
    warm = max(1, a.epochs // 3)
    kind = a.pruner if a.pruner != "auto" else ("hyperband" if a.trials >= 10 else "median")
    pruner = {"hyperband": optuna.pruners.HyperbandPruner(min_resource=warm, max_resource=a.epochs, reduction_factor=3),
              "median": optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=warm),
              "none": optuna.pruners.NopPruner()}[kind]
    print(f"pruner: {kind} (no pruning before epoch {warm + 1})", flush=True)
    study = optuna.create_study(direction="minimize" if a.lower_is_better else "maximize",
                                sampler=optuna.samplers.TPESampler(seed=a.seed, multivariate=True), pruner=pruner)
    ref = {stage: cands[0]["name"] for stage, cands in stages.items()}
    for k, spec in knobs.items():
        if "choices" in spec:
            ref[k] = spec["choices"][0]
        else:
            ref[k] = float(np.sqrt(spec["low"] * spec["high"])) if spec.get("log") else (spec["low"] + spec["high"]) / 2
    study.enqueue_trial(ref)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout_min * 60 or None, catch=(RuntimeError,))

    rows = []
    for t in study.trials:
        rows.append({"trial": t.number, "state": t.state.name, "value": t.value, **t.params,
                     "minutes": t.user_attrs.get("minutes"), "fold_scores": t.user_attrs.get("fold_scores")})
    trials = pd.DataFrame(rows)
    trials.to_csv(out / "trials.csv", index=False)
    done = trials[trials.state == "COMPLETE"].copy()
    counts = trials.state.value_counts().to_dict()
    print(f"\ntrials: {counts}; results in {out / 'trials.csv'}")
    if done.empty:
        raise SystemExit("no trial completed")
    done["key"] = sign * done["value"]
    top = done.sort_values("key", ascending=False).head(a.top)
    if 0 not in set(top.trial) and 0 in set(done.trial):
        top = pd.concat([top, done[done.trial == 0]])
    long = []
    for _, r in top.iterrows():
        for f, s in zip(folds, r["fold_scores"]):
            mj = out / f"trial_{int(r['trial']):03d}" / f"f{f}_s{a.seed}" / "metrics.json"
            m = json.loads(mj.read_text()) if mj.exists() else {}
            long.append({"name": f"trial_{int(r['trial']):03d}", "fold": f, "seed": a.seed, "score": s,
                         "n_train": m.get("train_images", np.nan), "n_val": m.get("val_images", np.nan)})
    table = summarize_folds(pd.DataFrame(long), higher_is_better=not a.lower_is_better)
    cost = {f"trial_{int(r['trial']):03d}": r["minutes"] for _, r in top.iterrows()}
    table["minutes"] = table["name"].map(cost)
    table.to_csv(out / "top_trials.csv", index=False)
    pd.set_option("display.width", 200)
    print("\ntop trials, paired over folds (trial_000 = reference combination):")
    print(table.to_string(index=False))
    kept = table[table.keep]
    pick = kept.sort_values("minutes").iloc[0]["name"]
    tnum = int(pick.split("_")[1])
    print(f"\nrecommended (cheapest among kept): {pick} {study.trials[tnum].params}")
    print(f"args: {study.trials[tnum].user_attrs.get('args')}")
    print("next: confirm once on the holdout (--eval-part holdout) against the reference; then Stage 10")


if __name__ == "__main__":
    main()
