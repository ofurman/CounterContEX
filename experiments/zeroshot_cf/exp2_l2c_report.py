"""Run exp2_counterfactuals.py's TabPFN zero-shot CF generation N times (a
different sampler seed each time) for one dataset, and report the same L2C
metrics (l2c_metrics.py) as dice_baseline.py, for an apples-to-apples
comparison between the two CF-generation methods.

exp2 generates exactly one CF per test point per call (a single MAP-ish
imputation draw, not DiCE's explicit diverse candidate set) — so to get a
non-trivial l2c_diversity_weight_fast this script repeats generation
--n-repeats times with different sampler random_state values (exp2's
`base_seed` param) and pools the results, giving the same (query, repeat)
shape dice_baseline.py's --total-cfs produces. Each repeat re-runs the full
TabPFN autoregressive imputation over --max-test points, so this is
`n_repeats` times slower than a single exp2 run — keep --max-test small.

Usage:
  uv run python experiments/zeroshot_cf/exp2_l2c_report.py --dataset german --n-repeats 5 --max-test 30
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.zeroshot_cf.l2c_metrics import compute_l2c_metrics  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_exp2_l2c_report(
    dataset_name: str,
    n_repeats: int = 5,
    max_test: int = 30,
    temperature: float = 1.0,
    n_permutations: int = 5,
    max_context: int = 256,
    context_type: str = "target_only",
    base_seed: int = 42,
) -> Dict[str, float]:
    from experiments.zeroshot_cf.exp2_counterfactuals import generate_counterfactuals

    print(f"\n=== exp2 (TabPFN) L2C report: {dataset_name.upper()} — "
          f"{n_repeats} repeats x {max_test} queries ===")

    X_orig_list, X_cf_list, y_orig_list = [], [], []
    disc = None
    t0 = time.time()
    for r in range(n_repeats):
        seed = base_seed + r * 1000  # keep the per-target-class (+0/+1) offsets from colliding across repeats
        print(f"\n--- repeat {r + 1}/{n_repeats} (base_seed={seed}) ---")
        X_test, y_test, X_cf, info = generate_counterfactuals(
            dataset_name,
            temperature=temperature,
            n_permutations=n_permutations,
            max_context=max_context,
            context_type=context_type,
            max_test=max_test,
            base_seed=seed,
        )
        disc = info["disc_model"]
        X_orig_list.append(X_test)
        X_cf_list.append(np.clip(X_cf, 0.0, 1.0))
        y_orig_list.append(info["y_pred"])

    elapsed = time.time() - t0
    X_orig_arr = np.concatenate(X_orig_list, axis=0)
    X_cf_arr = np.concatenate(X_cf_list, axis=0)
    y_orig_arr = np.concatenate(y_orig_list, axis=0)
    y_cf_arr = disc.predict(X_cf_arr)

    metrics = compute_l2c_metrics(X_orig_arr, X_cf_arr, y_orig_arr, y_cf_arr)
    metrics["n_queries"] = max_test
    metrics["n_repeats"] = n_repeats
    metrics["search_time_s"] = elapsed

    print(
        f"\n  l2c_validity={metrics['l2c_validity']:.2f}  "
        f"l2c_sparsity={metrics['l2c_sparsity']:.2f}  "
        f"l2c_diversity={metrics['l2c_diversity_weight_fast']:.2f}  "
        f"l2c_hmean={metrics['l2c_hmean_sparsity_diversity']:.2f}  "
        f"({elapsed:.1f}s total, {n_repeats} repeats x {max_test} queries)"
    )
    return metrics


def main() -> None:
    from experiments.zeroshot_cf.local_data import LOCAL_DATASET_NAMES

    parser = argparse.ArgumentParser(description="exp2 (TabPFN) L2C metric report")
    parser.add_argument(
        "--dataset", choices=["moons", "heloc", *sorted(LOCAL_DATASET_NAMES)], default="german"
    )
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--max-test", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument(
        "--context-type", choices=["target_only", "all_classes"], default="target_only"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = run_exp2_l2c_report(
        args.dataset,
        n_repeats=args.n_repeats,
        max_test=args.max_test,
        temperature=args.temperature,
        n_permutations=args.n_permutations,
        max_context=args.max_context,
        context_type=args.context_type,
        base_seed=args.seed,
    )

    csv_path = RESULTS_DIR / f"exp2_l2c_{args.dataset}_metrics.csv"
    row = {"dataset": args.dataset, **metrics}
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n  Wrote {csv_path}")


if __name__ == "__main__":
    main()
