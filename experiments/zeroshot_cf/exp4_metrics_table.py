"""Standard CF metric table for the beam-search + TabPFN counterfactuals.

Computes the metric set used in the `counterfactuals` repo — L1, L2, validity,
LOF, IsolationForest, sparsity, eps-sparsity — on the saved Exp-4 arrays.

Metrics come from the vendored `cel` registry (`cel.metrics.utils._METRIC_REGISTRY`),
so the numbers are produced by the library's own metric classes rather than
reimplementations. The one exception is `eps_sparsity`, which is not in the vendored
registry: it is ported from `cel/metrics/dicoflex_metrics.py` (commit b9715ef,
branch origin/ofurman/CFN_baselines).

Validity note: this pipeline relabels — `y_target = 1 - disc.predict(X_test)`
(`exp4_beam_search.py:144`). The registry's `validity` is `mean(y_cf_pred != y_test)`,
which under relabelling equals the discriminator's accuracy. Both columns are emitted;
`validity` is the `== y_target` definition and is the one to use.

Usage:
    uv run python experiments/zeroshot_cf/exp4_metrics_table.py
    uv run python experiments/zeroshot_cf/exp4_metrics_table.py --min-n 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
ARRAYS_DIR = RESULTS_DIR / "arrays"

# Registry metrics, in table order.
REGISTRY_METRICS = [
    "coverage",
    "proximity_l1_jaccard",
    "proximity_l2_jaccard",
    "sparsity",
    "lof_scores_cf",
    "isolation_forest_scores_cf",
]

EPS_SPARSITY_THRESHOLD = 0.05
EPS = 1e-8


def eps_sparsity(X_test: np.ndarray, X_cf: np.ndarray, cont: List[int]) -> float:
    """mean(|dx| / (|x| + 1e-8) > 0.05) over continuous features.

    Ported from cel/metrics/dicoflex_metrics.py (not in the vendored registry).
    Relative threshold, so it is sensitive to factual values near zero.
    """
    if X_test.size == 0 or len(cont) == 0:
        return float("nan")
    rel = np.abs(X_test[:, cont] - X_cf[:, cont]) / (np.abs(X_test[:, cont]) + EPS)
    return float((rel > EPS_SPARSITY_THRESHOLD).mean())


def score_cell(npz_path: Path) -> Dict[str, Any]:
    from cel.metrics.utils import _METRIC_REGISTRY  # noqa: PLC0415

    from experiments.zeroshot_cf.data import load_dataset  # noqa: PLC0415
    from experiments.zeroshot_cf.discriminator import train_discriminator  # noqa: PLC0415

    stem = npz_path.stem.replace("exp4_", "").replace("_cfs", "")
    dataset_name, tag = stem.rsplit("_", 1)

    z = np.load(npz_path)
    X_cf = np.clip(z["X_cf"], 0.0, 1.0)
    X_test = z["X_test"]
    y_test = z["y_test"].astype(np.int64).squeeze()
    y_target = z["y_target"].astype(np.int64).squeeze()

    bundle = load_dataset(dataset_name)
    disc_model = train_discriminator(
        bundle.X_train, bundle.y_train, X_test, y_test, dataset_name
    )
    y_cf_pred = np.asarray(disc_model.predict(X_cf)).squeeze()
    cont = list(bundle.numerical_features_indices)

    inputs = dict(
        X_cf=X_cf,
        X_test=X_test,
        X_train=bundle.X_train,
        y_test=y_test,
        y_target=y_target,
        y_cf_pred=y_cf_pred,
        disc_model=disc_model,
        continuous_features=cont,
        categorical_features=list(bundle.categorical_features_indices),
    )

    row: Dict[str, Any] = {
        "dataset": dataset_name,
        "set": tag,
        "n": int(X_cf.shape[0]),
        # == y_target: did generation reach the class it aimed at
        "validity": float((y_cf_pred == y_target).mean()),
    }

    for name in REGISTRY_METRICS:
        metric_cls = _METRIC_REGISTRY.get(name)
        if metric_cls is None:
            row[name] = float("nan")
            continue
        metric = metric_cls()
        if not metric.required_inputs().issubset(inputs.keys()):
            row[name] = float("nan")
            continue
        try:
            row[name] = float(metric(**inputs))
        except Exception:
            row[name] = float("nan")

    row["eps_sparsity"] = eps_sparsity(X_test, X_cf, cont)
    # registry convention, = disc accuracy under relabelling; kept for traceability
    row["validity_vs_true"] = float((y_cf_pred != y_test).mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument(
        "--csv", type=str, default=str(RESULTS_DIR / "exp4_metrics_table.csv")
    )
    args = parser.parse_args()

    rows = []
    for p in sorted(ARRAYS_DIR.glob("exp4_*_cfs.npz")):
        n = int(np.load(p)["X_cf"].shape[0])
        if n < args.min_n:
            print(f"skip {p.name} (n={n} < {args.min_n})")
            continue
        rows.append(score_cell(p))

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)

    cols = [
        "dataset",
        "set",
        "n",
        "validity",
        "coverage",
        "proximity_l1_jaccard",
        "proximity_l2_jaccard",
        "sparsity",
        "eps_sparsity",
        "lof_scores_cf",
        "isolation_forest_scores_cf",
    ]
    disp = df[cols].rename(
        columns={
            "proximity_l1_jaccard": "L1",
            "proximity_l2_jaccard": "L2",
            "eps_sparsity": "eps_spars",
            "lof_scores_cf": "LOF",
            "isolation_forest_scores_cf": "IsoForest",
        }
    )
    print(disp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
