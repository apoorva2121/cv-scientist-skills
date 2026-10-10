"""Load the Stage 2 split so every stage screens on exactly the same images.

    from cv_folds import load_split, verify_untouched
    s = load_split("split/split.csv", ids=df["file"])
    train_df = df[s.train_mask]                 # screening, CV, the joint search
    for tr, va in s.folds: ...                  # positional indices into train_df
    holdout_df = df[s.holdout_mask]             # Stage 9 confirmation only
    test_df, external_df = df[s.test_mask], df[s.external_mask]   # Stage 10, once each
    assert verify_untouched("split")            # before scoring test / external

Fold semantics: for the 'time' scheme fold k trains on folds {-1, 0..k-1} and validates on fold k
(expanding window); for the other schemes fold k validates on fold k and trains on the other folds.
The result is accepted by scikit-learn as `cv=` and is easy to use with any PyTorch loop.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Split:
    train_mask: np.ndarray
    holdout_mask: np.ndarray
    test_mask: np.ndarray
    external_mask: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    scheme: str

    def __iter__(self):
        return iter(self.folds)

    def __len__(self):
        return len(self.folds)


def load_split(split_csv: str | Path, ids: pd.Series | None = None) -> Split:
    split_csv = Path(split_csv)
    spec = json.loads((split_csv.parent / "split_spec.json").read_text())
    sp = pd.read_csv(split_csv, dtype={"id": str})
    if ids is not None:
        sp = sp.set_index("id").loc[pd.Series(ids).astype(str).to_numpy()].reset_index()
    part = sp["part"].to_numpy()
    train = part == "train"
    fold = sp.loc[train, "fold"].to_numpy()
    expanding = spec["scheme"] == "time"
    folds = []
    for k in sorted(k for k in np.unique(fold) if k >= 0):
        va = np.flatnonzero(fold == k)
        tr = np.flatnonzero(fold < k) if expanding else np.flatnonzero(fold != k)
        folds.append((tr, va))
    return Split(train, part == "holdout", part == "test", part == "external", folds, spec["scheme"])


def verify_untouched(split_dir: str | Path) -> bool:
    """Recompute the test and external id hashes; False means a frozen set changed since Stage 2."""
    split_dir = Path(split_dir)
    spec = json.loads((split_dir / "split_spec.json").read_text())
    sp = pd.read_csv(split_dir / "split.csv", dtype={"id": str})

    def digest(p):
        return hashlib.sha256("\n".join(sorted(sp.loc[sp.part == p, "id"])).encode()).hexdigest()

    ok = digest("test") == spec["test_ids_sha256"]
    if spec.get("external_ids_sha256"):
        ok = ok and digest("external") == spec["external_ids_sha256"]
    return ok
