"""Combine our own W&B benchmark results (run_full_benchmark.py) with the
published baseline numbers from the CounterFlowNet paper
(arXiv:2602.17244 — "CounterFlowNet: From Minimal Changes to Meaningful
Counterfactual Explanations", the paper ../CETGFN implements) into one
comparison table per dataset.

*** IMPORTANT — paper numbers are best-effort, NOT verified ***
PAPER_TABLE_A / PAPER_TABLE_B below were transcribed by visually reading the
paper's rendered PDF pages (Table 1 on page 6, Table 2 on page 7 of the PDF).
Several rows had an inconsistent number of columns when read this way (a
value the extraction likely dropped) — those cells are left as `None` rather
than guessed. SPOT-CHECK THESE AGAINST THE ACTUAL PAPER before relying on
them for anything you'd publish or make a decision from. The paper PDF is at
https://arxiv.org/pdf/2602.17244.

Protocol differences worth knowing before treating this as apples-to-apples:
- The paper draws K=10 CFs per test point; run_full_benchmark.py defaults to
  --n-repeats 5.
- Protocol A (Table 1) states uniform B=4 discretization bins; our ported
  german/adult/admission/student config.json bin counts (local_data.py's
  DISCRETIZED_DATASETS) come from CETGFN's own per-feature bin edges, which
  vary (e.g. admission uses 3 bins/feature, not 4).
- The paper's Table 2 "Adult Income" row and our "adult_dicoflex" dataset are
  NOT the same underlying object — Table 2 reuses the same base Adult dataset
  as Table 1 (just evaluated continuously), while adult_dicoflex is a
  separately-ported CETGFN dataset variant. Mapped here for a rough
  reference only.
- Our "sba" dataset has no counterpart in this paper at all (not one of its
  8 datasets) — it won't appear in the combined table.
- Diversity: the paper's Protocol B Div. formula (Table 6) is mean pairwise
  L1(continuous) + Hamming(categorical) distance; our closest metric,
  dicoflex_pairwise_distance, uses Euclidean (not L1) for the continuous part
  — same idea, not a bit-identical formula.
- We don't compute Coverage or Unary (monotonicity-constraint satisfaction)
  at all, so those two Table 1 columns are always blank for our own rows.

Usage:
  python experiments/zeroshot_cf/compare_with_paper.py --wandb-project zeroshot-cf-benchmark
  python experiments/zeroshot_cf/compare_with_paper.py --wandb-project CounterContEX --wandb-entity my-team
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Paper reference tables (best-effort transcription — see caveats above)
# ===========================================================================

# Table 1 (Protocol A, discretized B=4). Columns: Spars, Div, H.Mean, Val,
# Cov, Unary (all %, higher is better for every column).
PAPER_TABLE_A: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
    "german": {
        "L2C":    {"Spars": 61.35, "Div": 37.61, "H.Mean": 46.39, "Val": 100.00, "Cov": 100.00, "Unary": 99.06},
        "DICE-R": {"Spars": 88.23, "Div": 15.29, "H.Mean": 26.06, "Val": 100.00, "Cov": 100.00, "Unary": 90.81},
        "DICE-G": {"Spars": 43.45, "Div": 37.56, "H.Mean": 40.29, "Val": 62.87,  "Cov": 90.24,  "Unary": 56.66},
        "COPA":   {"Spars": 57.88, "Div": 18.88, "H.Mean": 28.47, "Val": 44.00,  "Cov": 44.00,  "Unary": 84.31},
        "MCCE":   {"Spars": 28.76, "Div": 33.40, "H.Mean": 30.91, "Val": 48.74,  "Cov": 58.76,  "Unary": None},
        "CFN":    {"Spars": 69.09, "Div": 41.26, "H.Mean": 51.65, "Val": 99.48,  "Cov": 100.00, "Unary": 100.00},
    },
    "adult": {
        "L2C":    {"Spars": 45.70, "Div": 28.11, "H.Mean": 34.80, "Val": 100.00, "Cov": 97.62,  "Unary": None},
        "DICE-R": {"Spars": 89.26, "Div": 9.05,  "H.Mean": 16.44, "Val": 100.00, "Cov": 87.15,  "Unary": None},
        "DICE-G": {"Spars": 41.48, "Div": 26.27, "H.Mean": 32.14, "Val": 92.64,  "Cov": 74.76,  "Unary": 45.79},
        "MCCE":   {"Spars": 24.93, "Div": 4.58,  "H.Mean": 7.74,  "Val": 30.63,  "Cov": 74.76,  "Unary": 45.79},
        "CFN":    {"Spars": 62.36, "Div": 38.21, "H.Mean": 47.39, "Val": 99.63,  "Cov": 100.00, "Unary": None},
    },
    "admission": {
        "L2C":    {"Spars": 42.23, "Div": 37.90, "H.Mean": 39.94, "Val": 100.00, "Cov": 100.00, "Unary": None},
        "DICE-R": {"Spars": 66.25, "Div": 30.93, "H.Mean": 42.15, "Val": 100.00, "Cov": 85.30,  "Unary": None},
        "DICE-G": {"Spars": 23.05, "Div": 47.54, "H.Mean": 31.04, "Val": 92.91,  "Cov": 100.00, "Unary": 66.69},
        "MCCE":   {"Spars": 17.39, "Div": 22.98, "H.Mean": 19.51, "Val": 43.79,  "Cov": 84.60,  "Unary": 79.11},
        "CFN":    {"Spars": 55.41, "Div": 46.18, "H.Mean": 50.37, "Val": 99.53,  "Cov": 100.00, "Unary": 100.00},
    },
    "student": {
        "L2C":    {"Spars": 55.32, "Div": 29.54, "H.Mean": 38.51, "Val": 100.00, "Cov": 100.00, "Unary": None},
        "DICE-R": {"Spars": 87.60, "Div": 13.64, "H.Mean": 23.60, "Val": 100.00, "Cov": 98.99,  "Unary": None},
        "DICE-G": {"Spars": 39.20, "Div": 39.88, "H.Mean": 38.54, "Val": 84.83,  "Cov": 100.00, "Unary": 60.77},
        "COPA":   {"Spars": 50.45, "Div": 25.28, "H.Mean": 33.68, "Val": 67.26,  "Cov": 67.26,  "Unary": 95.32},
        "MCCE":   {"Spars": 25.97, "Div": 24.97, "H.Mean": 25.44, "Val": 68.61,  "Cov": 93.10,  "Unary": 67.70},
        "CFN":    {"Spars": 71.18, "Div": 31.44, "H.Mean": 43.61, "Val": 99.63,  "Cov": 100.00, "Unary": 100.00},
    },
}

# Table 2 (Protocol B, continuous, generation-time B=64). Columns: Val (0-1,
# higher better), Prox-Cont (L1 on continuous, lower better), Spars-Cat
# (fraction of categoricals changed, lower better), eps-Spars (lower
# better), LOF (median log-LOF, lower better), Div (lower is NOT better here
# — higher is better, paper marks Div.^ with an up-arrow).
# A couple of cells I could not read confidently from the rendered PDF are
# left as None (flagged inline) rather than guessed.
PAPER_TABLE_B: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
    "adult_dicoflex": {  # paper's "Adult Income" Protocol B row — see caveat above
        "DICE":     {"Val": 1.00, "Prox-Cont": 0.44, "Spars-Cat": 0.05, "eps-Spars": 0.50, "LOF": 0.17, "Div": 0.06},
        "CCHVAE":   {"Val": 1.00, "Prox-Cont": 0.50, "Spars-Cat": 0.06, "eps-Spars": 0.98, "LOF": 0.22, "Div": 0.03},
        "DiCoFlex": {"Val": 1.00, "Prox-Cont": 0.85, "Spars-Cat": 0.49, "eps-Spars": 0.97, "LOF": 0.31, "Div": 0.40},
        "CFN":      {"Val": 1.00, "Prox-Cont": 0.19, "Spars-Cat": 0.27, "eps-Spars": 0.36, "LOF": 0.20, "Div": 0.21},
    },
    "bank": {
        "DICE":     {"Val": 1.00, "Prox-Cont": 0.61, "Spars-Cat": 0.07, "eps-Spars": 0.66, "LOF": 0.11, "Div": 0.12},
        "CCHVAE":   {"Val": 1.00, "Prox-Cont": 0.73, "Spars-Cat": 0.15, "eps-Spars": 0.97, "LOF": 0.15, "Div": 0.08},
        "DiCoFlex": {"Val": 1.00, "Prox-Cont": 0.89, "Spars-Cat": 0.40, "eps-Spars": 0.95, "LOF": 0.16, "Div": 0.39},
        "CFN":      {"Val": 1.00, "Prox-Cont": 0.25, "Spars-Cat": 0.23, "eps-Spars": 0.38, "LOF": 0.21, "Div": 0.21},
    },
    "default": {
        "DICE":     {"Val": 1.00, "Prox-Cont": 0.31, "Spars-Cat": 0.06, "eps-Spars": 0.77, "LOF": 0.24, "Div": 0.05},
        "CCHVAE":   {"Val": 1.00, "Prox-Cont": 0.48, "Spars-Cat": 0.10, "eps-Spars": 0.33, "LOF": 0.04, "Div": None},  # unclear in source render
        "DiCoFlex": {"Val": 1.00, "Prox-Cont": 0.64, "Spars-Cat": 0.56, "eps-Spars": 0.95, "LOF": 0.39, "Div": 0.33},
        "CFN":      {"Val": 0.96, "Prox-Cont": 0.12, "Spars-Cat": 0.19, "eps-Spars": 0.34, "LOF": 0.19, "Div": 0.14},
    },
    "gmc": {
        "DICE":     {"Val": 1.00, "Prox-Cont": 0.26, "Spars-Cat": 0.04, "eps-Spars": 0.68, "LOF": 0.05, "Div": 0.04},
        "CCHVAE":   {"Val": 1.00, "Prox-Cont": 0.43, "Spars-Cat": 0.07, "eps-Spars": 0.96, "LOF": 0.01, "Div": None},  # unclear in source render
        "DiCoFlex": {"Val": 1.00, "Prox-Cont": 0.79, "Spars-Cat": 0.83, "eps-Spars": 0.94, "LOF": 0.53, "Div": 0.51},
        "CFN":      {"Val": 1.00, "Prox-Cont": 0.12, "Spars-Cat": 0.63, "eps-Spars": 0.25, "LOF": 0.24, "Div": 0.18},  # Spars-Cat=0.63 looks high vs. others — re-verify
    },
    "lending-club": {
        "DICE":     {"Val": 1.00, "Prox-Cont": 0.82, "Spars-Cat": 0.19, "eps-Spars": 0.79, "LOF": 0.03, "Div": 0.13},
        "CCHVAE":   {"Val": 1.00, "Prox-Cont": 0.63, "Spars-Cat": 0.65, "eps-Spars": 0.94, "LOF": 0.15, "Div": 0.05},
        "DiCoFlex": {"Val": 1.00, "Prox-Cont": 1.06, "Spars-Cat": 0.76, "eps-Spars": 0.94, "LOF": 0.05, "Div": 0.34},
        "CFN":      {"Val": 0.94, "Prox-Cont": 0.56, "Spars-Cat": 0.41, "eps-Spars": 0.29, "LOF": 0.21, "Div": 0.24},
    },
}

OUR_METHOD_NAME = "TabPFN-ZeroShot (ours)"


# ===========================================================================
# W&B fetching
# ===========================================================================


def fetch_wandb_runs(project: str, entity: Optional[str] = None) -> List[dict]:
    """Pull one row per run from run_full_benchmark.py's W&B project: config
    (dataset, disc_type, metric_suite, max_test, n_repeats, ...) + every
    logged metric, keyed by the run's `dataset` config value."""
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path)

    rows = []
    for run in runs:
        row = {"run_name": run.name, "run_id": run.id, "state": run.state}
        row.update(run.config)
        row.update({k: v for k, v in run.summary.items() if not k.startswith("_")})
        rows.append(row)
    return rows


# ===========================================================================
# Merge + report
# ===========================================================================


def build_comparison(wandb_rows: List[dict]) -> List[dict]:
    """One row per (dataset, method) — paper baselines + our own W&B run(s),
    normalized onto whichever column set (Table A or Table B) applies."""
    by_dataset: Dict[str, List[dict]] = {}

    for ds, methods in PAPER_TABLE_A.items():
        for method, vals in methods.items():
            by_dataset.setdefault(ds, []).append({"dataset": ds, "protocol": "A", "method": method, **vals})

    for ds, methods in PAPER_TABLE_B.items():
        for method, vals in methods.items():
            by_dataset.setdefault(ds, []).append({"dataset": ds, "protocol": "B", "method": method, **vals})

    for run in wandb_rows:
        ds = run.get("dataset")
        suite = run.get("metric_suite")
        if ds is None or suite is None:
            continue

        if suite == "l2c":
            row = {
                "dataset": ds,
                "protocol": "A",
                "method": f"{OUR_METHOD_NAME} [{run.get('run_name', '')}]",
                "Spars": run.get("l2c_sparsity"),
                "Div": run.get("l2c_diversity_weight_fast"),
                "H.Mean": run.get("l2c_hmean_sparsity_diversity"),
                "Val": run.get("l2c_validity"),
                "Cov": None,   # not computed by our metrics
                "Unary": None,  # not computed by our metrics
            }
        else:  # dicoflex
            row = {
                "dataset": ds,
                "protocol": "B",
                "method": f"{OUR_METHOD_NAME} [{run.get('run_name', '')}]",
                "Val": run.get("dicoflex_validity", 0) / 100 if run.get("dicoflex_validity") is not None else None,
                "Prox-Cont": run.get("dicoflex_proximity_l1_num"),
                "Spars-Cat": run.get("dicoflex_sparsity_cat", 0) / 100 if run.get("dicoflex_sparsity_cat") is not None else None,
                "eps-Spars": run.get("dicoflex_eps_sparsity", 0) / 100 if run.get("dicoflex_eps_sparsity") is not None else None,
                "LOF": run.get("dicoflex_lof_score"),
                "Div": run.get("dicoflex_pairwise_distance", 0) / 100 if run.get("dicoflex_pairwise_distance") is not None else None,
            }
        by_dataset.setdefault(ds, []).append(row)

    combined = []
    for ds in sorted(by_dataset):
        combined.extend(by_dataset[ds])
    return combined


def print_comparison(rows: List[dict]) -> None:
    current_ds = None
    for row in rows:
        if row["dataset"] != current_ds:
            current_ds = row["dataset"]
            print(f"\n=== {current_ds} (Protocol {row['protocol']}) ===")
        cols = [c for c in row if c not in ("dataset", "protocol", "method")]
        vals = "  ".join(f"{c}={row[c]:.2f}" if isinstance(row[c], (int, float)) else f"{c}=--" for c in cols)
        print(f"  {row['method']:35s} {vals}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine our W&B benchmark results with the CounterFlowNet paper's published numbers"
    )
    parser.add_argument("--wandb-project", type=str, required=True)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--paper-only",
        action="store_true",
        help="Skip W&B entirely and just print/write the paper's own tables.",
    )
    args = parser.parse_args()

    wandb_rows = [] if args.paper_only else fetch_wandb_runs(args.wandb_project, args.wandb_entity)
    if not args.paper_only:
        print(f"Fetched {len(wandb_rows)} run(s) from W&B project '{args.wandb_project}'.")

    combined = build_comparison(wandb_rows)
    print_comparison(combined)

    all_cols = sorted({k for row in combined for k in row} - {"dataset", "protocol", "method"})
    fieldnames = ["dataset", "protocol", "method", *all_cols]
    csv_path = RESULTS_DIR / "comparison_with_paper.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    print(f"\nWrote {len(combined)} row(s) to {csv_path}")


if __name__ == "__main__":
    main()
