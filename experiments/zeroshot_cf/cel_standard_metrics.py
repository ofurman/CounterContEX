"""Score saved Exp-4 counterfactual arrays with the **cel library's own metrics**.

Rather than re-implementing formulas, this instantiates the metric classes that the
vendored `cel` (`counterfactuals`) library registers via ``@register_metric`` and calls
them directly on the saved arrays. The numbers are therefore exactly what cel produces.

Density-based cel metrics (``prob_plausibility``, ``log_density_cf/test``) require a
generative model callable ``gen_model(X, y) -> log_probs``; the beam-search pipeline has
no such object, so those are reported as N/A. Everything computable without a gen_model
is included.

Two cel conventions worth flagging (kept as-is so the numbers stay faithful):
  * ``validity`` = mean(y_cf_pred != y_test) — compares to the factual's *true* label,
    not the intended target class. On misclassified factuals this differs from the
    ``== y_target`` definition the project's own harness uses.
  * proximity metrics filter to valid CFs using ``y_cf_pred == y_target``.

Usage:
  python experiments/zeroshot_cf/cel_standard_metrics.py            # all saved cells
  python experiments/zeroshot_cf/cel_standard_metrics.py --min-n 100 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
ARRAYS_DIR = RESULTS_DIR / "arrays"

# cel metrics computable from arrays + disc_model + X_train (no gen_model needed).
CEL_METRICS = [
    "number_of_instances",
    "coverage",
    "validity",
    "actionability",
    "sparsity",
    "proximity_l2_jaccard",
    "proximity_l1_jaccard",
    "proximity_euclidean_hamming",
    "lof_scores_cf",
    "lof_scores_test",
    "isolation_forest_scores_cf",
    "target_distance",
]


def score_cell(npz_path: Path) -> Dict[str, Any]:
    """Compute cel-registered metrics for one saved (dataset, regime) cell."""
    from cel.metrics.utils import _METRIC_REGISTRY  # noqa: PLC0415

    from experiments.zeroshot_cf.data import load_dataset  # noqa: PLC0415
    from experiments.zeroshot_cf.discriminator import train_discriminator  # noqa

    stem = npz_path.stem.replace("exp4_", "").replace("_cfs", "")
    dataset_name, tag = stem.rsplit("_", 1)

    z = np.load(npz_path)
    X_cf_raw = z["X_cf"]
    # cel scores the delivered CF; clip the (rare) out-of-[0,1] cells into range.
    X_cf = np.clip(X_cf_raw, 0.0, 1.0)
    X_test = z["X_test"]
    y_test = z["y_test"].astype(np.int64)
    y_target = z["y_target"].astype(np.int64)

    bundle = load_dataset(dataset_name)
    disc_model = train_discriminator(
        bundle.X_train, bundle.y_train, X_test, y_test, dataset_name
    )
    y_cf_pred = np.asarray(disc_model.predict(X_cf))

    inputs = dict(
        X_cf=X_cf,
        X_test=X_test,
        X_train=bundle.X_train,
        y_test=y_test,
        y_target=y_target,
        y_cf_pred=y_cf_pred,
        disc_model=disc_model,
        continuous_features=list(bundle.numerical_features_indices),
        categorical_features=list(bundle.categorical_features_indices),
    )

    row: Dict[str, Any] = {"dataset": dataset_name, "set": tag}
    for name in CEL_METRICS:
        metric_cls = _METRIC_REGISTRY.get(name)
        if metric_cls is None:
            row[name] = None
            continue
        metric = metric_cls()
        needed = metric.required_inputs()
        if not needed.issubset(inputs.keys()):
            row[name] = None  # requires gen_model or another unavailable input
            continue
        try:
            row[name] = float(metric(**inputs))
        except Exception as exc:  # keep going; record which metric failed
            row[name] = f"ERR: {type(exc).__name__}: {exc}"

    # frac_oob is not a cel metric but records how much clipping was needed.
    oob = (X_cf_raw < 0.0) | (X_cf_raw > 1.0)
    row["frac_oob"] = float(oob.any(axis=1).mean())
    row["n_features"] = int(X_cf.shape[1])
    row["n_continuous"] = len(bundle.numerical_features_indices)
    row["n_categorical"] = len(bundle.categorical_features_indices)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument(
        "--json", type=str, default=str(RESULTS_DIR / "exp4_cel_metrics.json")
    )
    args = parser.parse_args()

    paths = sorted(ARRAYS_DIR.glob("exp4_*_cfs.npz"))
    rows: List[Dict[str, Any]] = []
    for p in paths:
        n = int(np.load(p)["X_cf"].shape[0])
        if n < args.min_n:
            print(f"skip {p.name} (n={n} < {args.min_n})")
            continue
        print(f"\n=== {p.name} (n={n}) ===")
        row = score_cell(p)
        for k, v in row.items():
            if isinstance(v, float):
                print(f"  {k:30s} {v:.4f}")
            else:
                print(f"  {k:30s} {v}")
        rows.append(row)

    if rows:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
