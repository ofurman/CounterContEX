"""Recompute Exp-4 metrics from saved counterfactual arrays.

Generation is the expensive step (~0.65 s/CF on HELOC), so ``exp4_beam_search.py``
persists the raw generated arrays to ``results/arrays/exp4_<dataset>_<set>_cfs.npz``.
This script scores those arrays without re-running the beam search — use it to add a
new metric, or to salvage the completed cells of an interrupted run.

Usage:
  python experiments/zeroshot_cf/recompute_metrics.py                # all saved cells
  python experiments/zeroshot_cf/recompute_metrics.py --min-n 100    # skip smoke runs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
ARRAYS_DIR = RESULTS_DIR / "arrays"


def score_cell(npz_path: Path) -> Dict[str, float]:
    """Score one saved (dataset, regime) cell. Returns a metrics row."""
    from experiments.zeroshot_cf.data import load_dataset
    from experiments.zeroshot_cf.discriminator import train_discriminator
    from experiments.zeroshot_cf.metrics_harness import compute_metrics

    # exp4_<dataset>_<set>_cfs.npz
    stem = npz_path.stem.replace("exp4_", "").replace("_cfs", "")
    dataset_name, tag = stem.rsplit("_", 1)

    z = np.load(npz_path)
    X_cf = z["X_cf"]  # unclipped, as generated
    X_test, y_test = z["X_test"], z["y_test"]
    y_target = z["y_target"]
    immutable_idx = [int(i) for i in z["immutable_idx"]]
    drift = z["immutable_drift"]

    bundle = load_dataset(dataset_name)
    disc_model = train_discriminator(
        bundle.X_train, bundle.y_train, X_test, y_test, dataset_name
    )

    # frac_oob is measured on the UNCLIPPED array, before metric computation.
    oob_mask = (X_cf < 0.0) | (X_cf > 1.0)
    frac_oob = float(oob_mask.any(axis=1).mean())

    metrics = compute_metrics(
        disc_model=disc_model,
        X_cf=np.clip(X_cf, 0.0, 1.0),
        X_test=X_test,
        X_train=bundle.X_train,
        y_test=y_test,
        y_target=y_target,
        immutable_idx=immutable_idx,
        X_cf_lof=X_cf,  # unclipped preserves true LOF geometry
        categorical_idx=bundle.categorical_features_indices,
        feature_names=bundle.feature_names,
    )
    metrics["frac_oob"] = frac_oob
    metrics["immutable_drift_mean"] = float(np.nanmean(drift)) if immutable_idx else 0.0
    metrics["immutable_drift_max"] = float(np.nanmax(drift)) if immutable_idx else 0.0
    return {"dataset": dataset_name, "set": tag, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-n",
        type=int,
        default=0,
        help="Skip cells with fewer than this many rows (filters out smoke runs).",
    )
    args = parser.parse_args()

    paths = sorted(ARRAYS_DIR.glob("exp4_*_cfs.npz"))
    if not paths:
        print(f"No saved arrays in {ARRAYS_DIR}")
        return

    rows: List[Dict] = []
    for p in paths:
        n = int(np.load(p)["X_cf"].shape[0])
        if n < args.min_n:
            print(f"skip {p.name} (n={n} < --min-n {args.min_n})")
            continue
        print(f"\n=== scoring {p.name} (n={n}) ===")
        row = score_cell(p)
        rows.append(row)
        for k, v in row.items():
            print(f"  {k:28s} {v}" if isinstance(v, str) else f"  {k:28s} {v:.4f}")

    if not rows:
        return
    out = RESULTS_DIR / "exp4_recomputed_metrics.csv"
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
