"""Systematic exp4 (greedy, classifier-in-the-loop) CF benchmark across the
ported CETGFN datasets — the greedy counterpart to run_full_benchmark.py
(which uses exp2's one-shot joint imputation instead). Same dataset ->
classifier + metric-suite pairing (BENCHMARK_CONFIG, unchanged from
run_full_benchmark.py):

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

Unlike exp2 (one batched TabPFN call across the whole test set), exp4's
default "prob_ascent" selector (greedy.py's _select_prob_ascent) calls
sampler.sample_feature once per REMAINING actionable candidate, at EVERY
step, per point — worst case (budget exhausted without an early flip) is
roughly n_actionable*(n_actionable+1)/2 individual, unbatched TabPFN calls
PER POINT. For `default` (23 actionable columns) that's ~276 calls/point;
for `german` (16) ~136 calls/point. This is dramatically more expensive than
exp2's approach, and — going by exp2's near-zero validity on several of
these datasets — greedy plausibly burns its full budget often rather than
flipping early, so the worst case is not a rare edge case here.

*** --max-test therefore defaults to 16, NOT 256. *** Calibrate before
raising it: start at --max-test 1-4, watch the per-point timing exp4 prints,
and multiply out before committing to anything larger.

One-point local calibration (RTX 4070 Ti, prob_ascent, tau=0.5): german
(16 actionable, budget=16) took 25.6s and flipped after only 2 steps
(l0=2, 100% validity on that point); admission (6 actionable, budget=6)
took 25.3s and FAILED after exhausting its full budget (0% validity, all
6 candidates tried at every step). Cost and success are inversely
correlated here — the datasets exp2 already struggled with (gmc,
lending-club, bank, default) are plausibly both the SLOWEST (full budget,
every time) and the least likely to flip even with the classifier steering
every step. Treat any --time budget as a guess until you've calibrated the
actual hard datasets, not just a cheap one.

Because exp4 is expensive, --n-repeats defaults to 1 here (not 5) — and at
the default near-MAP temperature (1e-9) the committed values are already
close to deterministic, so repeats wouldn't buy much diversity signal
anyway. Raise --temperature if you want repeats to mean something.

Logs to Weights & Biases by default, same pattern as run_full_benchmark.py —
run name is "greedy-<dataset>-<disc_type>-<metric_suite>-<selector>-tau<tau>-mt<max_test>-seed<seed>"
(the "greedy-" prefix + selector/tau keep these distinguishable from
run_full_benchmark.py's runs in the same W&B project). Requires `wandb`
(`uv sync --extra wandb`) and either network or `WANDB_MODE=offline`.
Disable with --no-wandb.

Usage:
  # Smoke test first — 1 point, cheap dataset
  uv run python experiments/zeroshot_cf/run_greedy_benchmark.py --dataset admission --max-test 1

  # Calibration run across all datasets
  HF_HUB_OFFLINE=1 TABPFN_DEVICE=cuda uv run python experiments/zeroshot_cf/run_greedy_benchmark.py --max-test 16
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

# dataset -> (disc_type, metric_suite) — identical mapping to run_full_benchmark.py
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


def wandb_run_name(
    dataset_name: str, disc_type: str, suite: str, selector: str, tau: float,
    max_test: int, n_repeats: int, seed: int,
) -> str:
    mt_label = "full" if max_test is None or max_test < 0 else str(max_test)
    return (
        f"greedy-{dataset_name}-{disc_type}-{suite}-{selector}-tau{tau}"
        f"-mt{mt_label}-nr{n_repeats}-seed{seed}"
    )


def run_one(
    dataset_name: str,
    max_test: int,
    base_seed: int = 42,
    selector: str = "prob_ascent",
    tau: float = 0.5,
    budget: Optional[int] = None,
    temperature: float = 1e-9,
    n_permutations: int = 3,
    max_context: int = 256,
    n_repeats: int = 1,
    use_wandb: bool = True,
    wandb_project: str = WANDB_PROJECT_DEFAULT,
    wandb_entity: Optional[str] = None,
) -> Dict[str, float]:
    from experiments.zeroshot_cf.exp4_greedy_cf import generate_counterfactuals

    disc_type, suite = BENCHMARK_CONFIG[dataset_name]
    run_name = wandb_run_name(
        dataset_name, disc_type, suite, selector, tau, max_test, n_repeats, base_seed
    )
    print(
        f"\n{'=' * 70}\n{dataset_name.upper()}  disc={disc_type}  metrics={suite}  "
        f"selector={selector}  tau={tau}  budget={budget}  max_test={max_test}"
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
                "method": "exp4_greedy",
                "selector": selector,
                "tau": tau,
                "budget": budget,
                "temperature": temperature,
                "n_permutations": n_permutations,
                "max_context": max_context,
                "max_test": max_test,
                "n_repeats": n_repeats,
                "seed": base_seed,
            },
            tags=[dataset_name, disc_type, suite, "greedy", selector],
        )

    try:
        X_orig_list, X_cf_list, y_orig_list = [], [], []
        all_changed, all_flipped, all_steps = [], [], []
        disc = bundle = None
        n_failed_total = 0
        n_queries = None
        t0 = time.time()
        for r in range(n_repeats):
            seed = base_seed + r * 1000
            print(f"\n--- repeat {r + 1}/{n_repeats} (base_seed={seed}) ---")
            X_test, y_test, X_cf, info = generate_counterfactuals(
                dataset_name,
                selector=selector,
                tau=tau,
                budget=budget,
                temperature=temperature,
                n_permutations=n_permutations,
                max_context=max_context,
                max_test=max_test,
                base_seed=seed,
                disc_type=disc_type,
            )
            if n_queries is None:
                n_queries = len(X_test)  # actual point count — max_test may be -1/None ("full")
            disc = info["disc_model"]
            bundle = info["bundle"]
            X_orig_list.append(X_test)
            X_cf_list.append(np.clip(X_cf, 0.0, 1.0))
            y_orig_list.append(info["y_pred"])
            all_changed.extend(info["changed_per_point"])
            all_flipped.extend(info["flipped_per_point"])
            all_steps.extend(info["steps_per_point"])
            n_failed_total += int(sum(1 for f in info["flipped_per_point"] if not f))
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

        # Greedy-specific diagnostics (mirrors exp4_greedy_cf.evaluate_and_report).
        flipped = np.asarray(all_flipped, dtype=bool)
        l0_all = np.array([len(c) for c in all_changed], dtype=float)
        steps_all = np.asarray(all_steps, dtype=float)
        if flipped.any():
            metrics["l0_count_mean"] = float(l0_all[flipped].mean())
            metrics["l0_count_median"] = float(np.median(l0_all[flipped]))
            metrics["steps_mean"] = float(steps_all[flipped].mean())
        else:
            metrics["l0_count_mean"] = float("nan")
            metrics["l0_count_median"] = float("nan")
            metrics["steps_mean"] = float("nan")
        metrics["failure_rate"] = float((~flipped).mean())

        metrics["dataset"] = dataset_name
        metrics["disc_type"] = disc_type
        metrics["metric_suite"] = suite
        metrics["selector"] = selector
        metrics["tau"] = tau
        metrics["n_queries"] = n_queries
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
        description="Greedy (exp4) TabPFN CF benchmark across ported CETGFN datasets"
    )
    parser.add_argument("--dataset", choices=["all", *BENCHMARK_CONFIG.keys()], default="all")
    parser.add_argument(
        "--max-test",
        type=int,
        default=16,
        help="Points per dataset (default: 16 — NOT 256; see module docstring "
             "for why exp4 is far more expensive per point than exp2). -1 for "
             "the FULL test split — up to 10,000 points for some datasets; "
             "see slurm/README.md's 'full test set' section before using this, "
             "it is very likely infeasible for the largest datasets with exp4.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selector",
        choices=["prob_ascent", "class_divergence"],
        default="prob_ascent",
        help="Candidate-selection strategy (default: prob_ascent).",
    )
    parser.add_argument("--tau", type=float, default=0.5,
                        help="Flip probability threshold (default: 0.5 = hard flip).")
    parser.add_argument("--budget", type=int, default=None,
                        help="Max features to change (default: |actionable|).")
    parser.add_argument("--temperature", type=float, default=1e-9,
                        help="Committed-value temperature (default: 1e-9 = near-MAP).")
    parser.add_argument("--n-permutations", type=int, default=3,
                        help="Imputation permutations (default: 3, matches exp4's own default).")
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Independent CFs drawn per query point (default: 1 — greedy is "
             "near-deterministic at the default temperature, so repeats add "
             "cost without much diversity signal unless --temperature is raised).",
    )
    parser.add_argument("--wandb", dest="use_wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT)
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()

    datasets = list(BENCHMARK_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    l2c_csv = RESULTS_DIR / "greedy_benchmark_l2c_metrics.csv"
    dicoflex_csv = RESULTS_DIR / "greedy_benchmark_dicoflex_metrics.csv"

    l2c_rows, dicoflex_rows = [], []
    for ds in datasets:
        metrics = run_one(
            ds,
            max_test=args.max_test,
            base_seed=args.seed,
            selector=args.selector,
            tau=args.tau,
            budget=args.budget,
            temperature=args.temperature,
            n_permutations=args.n_permutations,
            max_context=args.max_context,
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
