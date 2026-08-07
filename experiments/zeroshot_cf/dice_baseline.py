"""DiCE ("random" method) baseline + L2C metrics, for comparison against the
greedy/TabPFN counterfactuals in this experiment.

Uses Microsoft's `dice-ml` (interpretml/DiCE) against this project's own
sklearn discriminator (discriminator.py) — no PyTorch/cel dependency needed,
since DiCE's "random" method only requires a `.predict()`/`.predict_proba()`
sklearn-compatible model.

Metrics (l2c_validity, l2c_sparsity, l2c_diversity_weight_fast,
l2c_hmean_sparsity_diversity) come from l2c_metrics.py — a numpy-only port of
../CETGFN's L2CCounterfactualMetrics, shared with exp2_counterfactuals.py so
both baselines report the same definitions.

Usage:
  uv run python experiments/zeroshot_cf/dice_baseline.py --dataset german
  uv run python experiments/zeroshot_cf/dice_baseline.py --dataset german --total-cfs 10 --max-test 100
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.zeroshot_cf.l2c_metrics import compute_l2c_metrics  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================
# DiCE ("random") baseline
# ===========================


def run_dice_baseline(
    dataset_name: str,
    method: str = "random",
    total_cfs: int = 5,
    disc_type: str = "lr",
    max_test: int = 100,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict]:
    import dice_ml

    from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset
    from experiments.zeroshot_cf.discriminator import train_discriminator
    from experiments.zeroshot_cf.local_data import DISCRETIZED_DATASETS

    bundle = load_dataset(dataset_name)
    feature_names = bundle.feature_names
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test = bundle.X_test[:max_test] if max_test else bundle.X_test

    disc = train_discriminator(X_train, y_train, bundle.X_test, bundle.y_test, dataset_name, disc_type=disc_type)
    y_pred = disc.predict(X_test)

    actionable_idx, _ = get_actionable_immutable(dataset_name, bundle)
    features_to_vary = [feature_names[i] for i in actionable_idx]

    # Datasets discretized via the L2C-style binner have finitely-many, exactly
    # equal-comparable codes for every column (see local_data.py) — treat them
    # all as categorical so DiCE samples from realized values, not interpolated
    # floats. Everything else keeps its numerical/categorical split.
    if dataset_name in DISCRETIZED_DATASETS:
        continuous_features: List[str] = []
    else:
        continuous_features = [feature_names[i] for i in bundle.numerical_features_indices]

    train_df = np.column_stack([X_train, y_train.astype(float)])
    import pandas as pd

    train_df = pd.DataFrame(train_df, columns=feature_names + ["target"])

    data_interface = dice_ml.Data(
        dataframe=train_df, continuous_features=continuous_features, outcome_name="target"
    )
    model_interface = dice_ml.Model(model=disc._clf, backend="sklearn")
    exp = dice_ml.Dice(data_interface, model_interface, method=method)

    query_df = pd.DataFrame(X_test, columns=feature_names)

    print(f"\n=== DiCE ({method}) baseline: {dataset_name.upper()} — "
          f"{len(query_df)} queries, total_CFs={total_cfs} ===")
    t0 = time.time()
    cf_result = exp.generate_counterfactuals(
        query_df,
        total_CFs=total_cfs,
        desired_class="opposite",
        features_to_vary=features_to_vary,
        random_seed=seed,
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"  Generation took {elapsed:.1f}s")

    X_orig_list, X_cf_list, y_orig_list = [], [], []
    n_failed = 0
    for i, cf in enumerate(cf_result.cf_examples_list):
        orig_row = X_test[i]
        orig_y = y_pred[i]
        if cf.final_cfs_df is not None and len(cf.final_cfs_df) > 0:
            cf_rows = cf.final_cfs_df[feature_names].to_numpy(dtype=float)
        else:
            n_failed += 1
            cf_rows = orig_row.reshape(1, -1)  # no-op fallback: counts as an invalid attempt
        for row in cf_rows:
            X_orig_list.append(orig_row)
            X_cf_list.append(row)
            y_orig_list.append(orig_y)

    X_orig_arr = np.array(X_orig_list)
    X_cf_arr = np.clip(np.array(X_cf_list), 0.0, 1.0)
    y_orig_arr = np.array(y_orig_list)
    y_cf_arr = disc.predict(X_cf_arr)

    metrics = compute_l2c_metrics(X_orig_arr, X_cf_arr, y_orig_arr, y_cf_arr)
    metrics["n_queries"] = len(query_df)
    metrics["n_failed"] = n_failed
    metrics["search_time_s"] = elapsed

    print(
        f"  l2c_validity={metrics['l2c_validity']:.2f}  "
        f"l2c_sparsity={metrics['l2c_sparsity']:.2f}  "
        f"l2c_diversity={metrics['l2c_diversity_weight_fast']:.2f}  "
        f"l2c_hmean={metrics['l2c_hmean_sparsity_diversity']:.2f}  "
        f"(n_failed={n_failed}/{len(query_df)})"
    )

    info = {"bundle": bundle, "disc_model": disc, "cf_result": cf_result}
    return metrics, info


def main() -> None:
    from experiments.zeroshot_cf.local_data import LOCAL_DATASET_NAMES

    parser = argparse.ArgumentParser(description="DiCE baseline + L2C metrics")
    parser.add_argument("--dataset", choices=["moons", "heloc", *sorted(LOCAL_DATASET_NAMES)], default="german")
    parser.add_argument("--method", choices=["random", "genetic", "kdtree"], default="random")
    parser.add_argument("--total-cfs", type=int, default=5)
    parser.add_argument("--disc-type", choices=["lr", "mlp"], default="lr")
    parser.add_argument("--max-test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics, _ = run_dice_baseline(
        args.dataset,
        method=args.method,
        total_cfs=args.total_cfs,
        disc_type=args.disc_type,
        max_test=args.max_test,
        seed=args.seed,
    )

    csv_path = RESULTS_DIR / f"dice_baseline_{args.dataset}_metrics.csv"
    row = {
        "dataset": args.dataset,
        "method": args.method,
        "disc_type": args.disc_type,
        "total_cfs": args.total_cfs,
        **metrics,
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n  Wrote {csv_path}")


if __name__ == "__main__":
    main()
