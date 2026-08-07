"""Systematic TabPFN zero-shot CF benchmark across the ported CETGFN datasets,
pairing each dataset with a classifier type and the L2C/DiCoFlex metric suite
CETGFN itself uses for it:

  german         -> lr,  l2c        (L2C-discretized)
  adult          -> mlp, l2c        (L2C-discretized)
  admission      -> lr,  l2c        (L2C-discretized)
  student        -> mlp, l2c        (L2C-discretized)
  adult_dicoflex -> mlp, dicoflex   (continuous, no discretization)
  bank           -> mlp, dicoflex   (continuous, no discretization)
  default        -> mlp, dicoflex   (continuous, no discretization)
  gmc            -> mlp, dicoflex   (continuous, no discretization)
  lending-club   -> mlp, dicoflex   (continuous, no discretization)
  sba            -> mlp, dicoflex   (continuous, no discretization)

Each dataset is generated through exp2_counterfactuals.generate_counterfactuals
--n-repeats times (default 5 — a different sampler seed each time, via the
`base_seed` param), and the repeats are pooled per query point before scoring
— the same pattern exp2_l2c_report.py uses — so l2c_diversity_weight_fast /
dicoflex_pairwise_distance are non-degenerate (5 CFs per point to compare
against each other). This makes the benchmark --n-repeats times more
expensive than a single pass; see slurm/README.md for calibrated timing.

TabPFN inference on --max-test points scales with (points x actionable
columns x n_repeats) and is comparatively expensive — --max-test 1 is meant
as a fast correctness smoke test to run BEFORE the real (default 256-point)
benchmark.

Each dataset's run is also logged to Weights & Biases by default (one run per
dataset, named "<dataset>-<disc_type>-<metric_suite>-mt<max_test>-nr<n_repeats>-seed<seed>"
so it's identifiable at a glance in the W&B UI — e.g. "german-lr-l2c-mt256-nr5-seed42").
Config (dataset, disc_type, metric_suite, max_test, n_repeats, n_permutations,
seed) is logged alongside the metrics. Requires `wandb` (`uv sync --extra
wandb`) and either network access or `WANDB_MODE=offline` (sync later with
`wandb sync` from a machine with network — SLURM compute nodes are usually
offline). Disable with --no-wandb.

Usage:
  # Smoke test everything at 1 point first
  uv run python experiments/zeroshot_cf/run_full_benchmark.py --max-test 1

  # Full run (GPU strongly recommended — see README)
  HF_HUB_OFFLINE=1 TABPFN_DEVICE=cuda uv run python experiments/zeroshot_cf/run_full_benchmark.py --max-test 256

  # Single dataset, no W&B
  uv run python experiments/zeroshot_cf/run_full_benchmark.py --dataset german --max-test 1 --no-wandb
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.zeroshot_cf.dicoflex_metrics import compute_dicoflex_metrics  # noqa: E402
from experiments.zeroshot_cf.l2c_metrics import compute_l2c_metrics  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PROJECT_DEFAULT = "zeroshot-cf-benchmark"


def wandb_run_name(
    dataset_name: str, disc_type: str, suite: str, max_test: int, n_repeats: int, seed: int
) -> str:
    return f"{dataset_name}-{disc_type}-{suite}-mt{max_test}-nr{n_repeats}-seed{seed}"

# dataset -> (disc_type, metric_suite)
BENCHMARK_CONFIG: Dict[str, Tuple[str, str]] = {
    "german": ("lr", "l2c"),
    "adult": ("mlp", "l2c"),
    "admission": ("lr", "l2c"),
    "student": ("mlp", "l2c"),
    "adult_dicoflex": ("mlp", "dicoflex"),
    "bank": ("mlp", "dicoflex"),
    "default": ("mlp", "dicoflex"),
    "gmc": ("mlp", "dicoflex"),
    "lending-club": ("mlp", "dicoflex"),
    "sba": ("mlp", "dicoflex"),
}


def run_one(
    dataset_name: str,
    max_test: int,
    base_seed: int = 42,
    n_permutations: int = 1,
    n_repeats: int = 5,
    use_wandb: bool = True,
    wandb_project: str = WANDB_PROJECT_DEFAULT,
    wandb_entity: Optional[str] = None,
) -> Dict[str, float]:
    from experiments.zeroshot_cf.exp2_counterfactuals import generate_counterfactuals

    disc_type, suite = BENCHMARK_CONFIG[dataset_name]
    run_name = wandb_run_name(dataset_name, disc_type, suite, max_test, n_repeats, base_seed)
    print(
        f"\n{'=' * 70}\n{dataset_name.upper()}  disc={disc_type}  metrics={suite}  "
        f"max_test={max_test}  n_permutations={n_permutations}  n_repeats={n_repeats}"
        f"\n{'=' * 70}"
    )

    wb_run = None
    if use_wandb:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is not installed. Install it with 'uv sync --extra wandb' "
                "(or pip install wandb), or pass --no-wandb to skip logging."
            ) from e
        wb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "dataset": dataset_name,
                "disc_type": disc_type,
                "metric_suite": suite,
                "max_test": max_test,
                "n_repeats": n_repeats,
                "n_permutations": n_permutations,
                "seed": base_seed,
            },
            tags=[dataset_name, disc_type, suite],
        )

    try:
        X_orig_list, X_cf_list, y_orig_list = [], [], []
        disc = bundle = None
        n_failed_total = 0
        t0 = time.time()
        for r in range(n_repeats):
            seed = base_seed + r * 1000  # keep per-target-class (+0/+1) offsets from colliding
            print(f"\n--- repeat {r + 1}/{n_repeats} (base_seed={seed}) ---")
            X_test, y_test, X_cf, info = generate_counterfactuals(
                dataset_name,
                max_test=max_test,
                base_seed=seed,
                disc_type=disc_type,
                n_permutations=n_permutations,
            )
            disc = info["disc_model"]
            bundle = info["bundle"]
            X_orig_list.append(X_test)
            X_cf_list.append(np.clip(X_cf, 0.0, 1.0))
            y_orig_list.append(info["y_pred"])
            n_failed_total += info["n_failed"]
        elapsed = time.time() - t0

        X_orig_arr = np.concatenate(X_orig_list, axis=0)
        X_cf_arr = np.concatenate(X_cf_list, axis=0)
        y_orig_arr = np.concatenate(y_orig_list, axis=0)
        y_cf_arr = disc.predict(X_cf_arr)

        if suite == "l2c":
            metrics = compute_l2c_metrics(X_orig_arr, X_cf_arr, y_orig_arr, y_cf_arr)
        else:
            y_cf_proba = disc.predict_proba(X_cf_arr)
            metrics = compute_dicoflex_metrics(
                X_orig_arr,
                X_cf_arr,
                y_orig_arr,
                y_cf_arr,
                y_cf_proba,
                num_indices=bundle.numerical_features_indices,
                cat_indices=bundle.categorical_features_indices,
                X_train=bundle.X_train,
            )

        metrics["dataset"] = dataset_name
        metrics["disc_type"] = disc_type
        metrics["metric_suite"] = suite
        metrics["n_queries"] = max_test
        metrics["n_repeats"] = n_repeats
        metrics["n_failed"] = n_failed_total
        metrics["search_time_s"] = elapsed

        print(f"\n  [{dataset_name}] {elapsed:.1f}s, n_failed={n_failed_total}/{len(X_orig_arr)}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:35s} {v:.4f}")

        if wb_run is not None:
            wb_run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            wb_run.summary["run_name"] = run_name
        return metrics
    finally:
        if wb_run is not None:
            wb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full TabPFN zero-shot benchmark across ported CETGFN datasets"
    )
    parser.add_argument("--dataset", choices=["all", *BENCHMARK_CONFIG.keys()], default="all")
    parser.add_argument("--max-test", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1,
        help="Imputation permutations (default: 1, vs exp2's own default of 5 — "
             "kept low here so the 256-point x 10-dataset benchmark is tractable).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Independent CFs drawn per query point (default: 5, different sampler "
             "seed each time), needed for a non-degenerate diversity metric. "
             "Multiplies runtime by roughly this factor.",
    )
    parser.add_argument(
        "--wandb",
        dest="use_wandb",
        action="store_true",
        default=True,
        help="Log each dataset's run to Weights & Biases (default: on).",
    )
    parser.add_argument(
        "--no-wandb", dest="use_wandb", action="store_false", help="Disable W&B logging."
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=WANDB_PROJECT_DEFAULT,
        help=f"W&B project name (default: {WANDB_PROJECT_DEFAULT}).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity/team (default: your W&B account default).",
    )
    args = parser.parse_args()

    datasets = list(BENCHMARK_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    l2c_csv = RESULTS_DIR / "benchmark_l2c_metrics.csv"
    dicoflex_csv = RESULTS_DIR / "benchmark_dicoflex_metrics.csv"

    l2c_rows, dicoflex_rows = [], []
    for ds in datasets:
        metrics = run_one(
            ds,
            max_test=args.max_test,
            base_seed=args.seed,
            n_permutations=args.n_permutations,
            n_repeats=args.n_repeats,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )
        suite = BENCHMARK_CONFIG[ds][1]
        (l2c_rows if suite == "l2c" else dicoflex_rows).append(metrics)

    for csv_path, rows in [(l2c_csv, l2c_rows), (dicoflex_csv, dicoflex_rows)]:
        if not rows:
            continue
        write_header = not csv_path.exists()
        fieldnames = list(rows[0].keys())
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Wrote {len(rows)} row(s) to {csv_path}")


if __name__ == "__main__":
    main()
