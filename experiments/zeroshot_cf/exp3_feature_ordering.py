"""Experiment 3: Feature-ordering (DAG) ablation for counterfactual generation.

Two-factor ablation:
  Factor 1 — Ordering : 'random' (baseline) vs 'dag' (Y → immutables → actionable chain)
  Factor 2 — Actionable set: 'full' vs 'reduced' (HELOC only; top-6 by |LR coef|)

Grid:
  MOONS  : {random, dag} × {full}          = 2 cells
  HELOC  : {random, dag} × {full, reduced} = 4 cells

Context type: all_classes (both datasets). Required because the DAG places Y as an
explicit parent; with target_only, Y is constant in context → TabPFN validation error.
HELOC: n_permutations reduced to 1 and max_test to 20 for runtime feasibility
(Stage-8's 17 masked cols × 5 perms × 50 test pts × 4 cells ≈ 88 min); identical
settings across all 4 HELOC cells keep the within-exp3 comparison fair.

Outputs:
  results/exp3_ordering_moons.csv   — 2 rows (one per cell)
  results/exp3_ordering_heloc.csv   — 4 rows
  results/exp3_summary.md           — comparison tables + honest verdict

Usage:
  uv run python experiments/zeroshot_cf/exp3_feature_ordering.py --dataset moons
  uv run python experiments/zeroshot_cf/exp3_feature_ordering.py --dataset heloc
  uv run python experiments/zeroshot_cf/exp3_feature_ordering.py --dataset all
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

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Context strategy: use all_classes so Y varies in the training context.
# With target_only, Y is constant (single class) → TabPFN's constant-feature
# validator rejects it as a DAG parent. all_classes is required for the DAG
# path; using it for random too keeps the within-exp3 comparison fair.
# (Stage-8 used target_only as its default; exp3 numbers are not directly
#  comparable to Stage-8 but the random/dag contrast within exp3 is valid.)
BASELINE_TEMPERATURE = 1.0
BASELINE_MAX_CONTEXT = 256
BASELINE_CONTEXT_TYPE = "all_classes"
REDUCED_K = 6

# MOONS: baseline n_permutations and sample size
MOONS_N_PERMUTATIONS = 5
MOONS_MAX_TEST = 100

# HELOC: reduced settings for runtime feasibility.
# HELOC has 17 masked columns × 5 perms × 2 class batches = 170 TabPFN calls/cell
# → ~22 min/cell × 4 cells = ~88 min total. Reduced to n_permutations=1
# (single-pass per column) and max_test=20 → ~4 min/cell × 4 = ~18 min.
# All 4 HELOC cells use identical settings → comparison is fair within exp3.
HELOC_N_PERMUTATIONS = 1
HELOC_MAX_TEST = 20

# Grid definition
_MOONS_GRID: List[Tuple[str, str]] = [
    ("random", "full"),
    ("dag", "full"),
]
_HELOC_GRID: List[Tuple[str, str]] = [
    ("random", "full"),
    ("dag", "full"),
    ("random", "reduced"),
    ("dag", "reduced"),
]


def _dataset_settings(dataset_name: str) -> Dict:
    """Return per-dataset n_permutations and max_test for exp3."""
    if dataset_name == "heloc":
        return {
            "n_permutations": HELOC_N_PERMUTATIONS,
            "max_test": HELOC_MAX_TEST,
        }
    return {
        "n_permutations": MOONS_N_PERMUTATIONS,
        "max_test": MOONS_MAX_TEST,
    }


def run_cell(
    dataset_name: str,
    ordering: str,
    actionable_set: str,
) -> Dict:
    """Run one ablation cell and return a metrics dict with metadata."""
    from experiments.zeroshot_cf.exp2_counterfactuals import (
        generate_counterfactuals,
        evaluate_and_report,
    )

    ds_settings = _dataset_settings(dataset_name)
    t0 = time.perf_counter()
    X_test, y_test, X_cf, info = generate_counterfactuals(
        dataset_name,
        temperature=BASELINE_TEMPERATURE,
        n_permutations=ds_settings["n_permutations"],
        max_context=BASELINE_MAX_CONTEXT,
        context_type=BASELINE_CONTEXT_TYPE,
        ordering=ordering,
        actionable_set=actionable_set,
        reduced_k=REDUCED_K,
        max_test=ds_settings["max_test"],
    )
    metrics = evaluate_and_report(
        dataset_name, X_test, y_test, X_cf, info, write_csv=False
    )
    runtime_s = time.perf_counter() - t0

    n_masked = len(info["mask_cols"])
    return {
        "ordering": ordering,
        "actionable_set": actionable_set,
        "n_masked": n_masked,
        **metrics,
        "runtime_s": runtime_s,
    }


CSV_COLUMNS = [
    "ordering", "actionable_set", "n_masked",
    "validity", "lof_scores_cf", "sparsity",
    "true_actionability", "proximity_l2_jaccard", "frac_oob",
    "runtime_s",
]


def run_dataset_ablation(dataset_name: str) -> List[Dict]:
    """Run the full ablation grid for one dataset, write CSV, return rows."""
    grid = _MOONS_GRID if dataset_name == "moons" else _HELOC_GRID
    rows = []

    for ordering, actionable_set in grid:
        label = f"{ordering}/{actionable_set}"
        print(f"\n{'='*60}")
        print(f"  Exp3 cell: {dataset_name.upper()} | {label}")
        print(f"{'='*60}")
        row = run_cell(dataset_name, ordering, actionable_set)
        rows.append(row)
        print(f"  Cell done in {row['runtime_s']:.1f}s: "
              f"validity={row['validity']:.3f}, frac_oob={row['frac_oob']:.3f}, "
              f"lof={row['lof_scores_cf']:.3f}")

    csv_path = RESULTS_DIR / f"exp3_ordering_{dataset_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, float("nan")) for k in CSV_COLUMNS})
    print(f"\nWrote {csv_path}")
    return rows


def write_exp3_summary(all_rows: Dict[str, List[Dict]]) -> None:
    """Write exp3_summary.md with per-dataset comparison tables and honest verdict."""
    lines = [
        "# Experiment 3: Feature-Ordering (DAG) Ablation",
        "",
        "## Setup",
        "",
        f"- temperature={BASELINE_TEMPERATURE}, max_context={BASELINE_MAX_CONTEXT}, "
        f"context_type={BASELINE_CONTEXT_TYPE} (see note below)",
        f"- MOONS: n_permutations={MOONS_N_PERMUTATIONS}, max_test={MOONS_MAX_TEST}",
        f"- HELOC: n_permutations={HELOC_N_PERMUTATIONS}, max_test={HELOC_MAX_TEST} "
        "(reduced from baseline 5/50 for runtime feasibility; all 4 cells identical)",
        f"- Reduced actionable set: top-{REDUCED_K} features by |LR coef| (HELOC only)",
        "- **Note on context_type**: Exp3 uses `all_classes` for all cells. Reason: the "
        "DAG places Y as an explicit conditioning parent; with `target_only`, Y is "
        "constant in context (single class) and TabPFN raises a constant-feature "
        "validation error. `all_classes` keeps Y informative and makes the random/dag "
        "comparison within Exp3 fair. Stage-8 results (target_only) remain the "
        "reference for the recommended production configuration.",
        "",
        "## Mechanism framing",
        "",
        "In the **random-permutation path** (`dag=None`), every masked cell already "
        "conditions on all observed columns (Y + immutables). Putting Y/immutables "
        '"first" is therefore a **no-op** — only the relative ordering of the masked '
        "actionable columns matters, and that effect is averaged away over multiple "
        "random permutations.",
        "",
        "The **DAG path** imposes `p(A₁|Y,immut) · p(A₂|Y,immut,A₁) · …` — a strict "
        "left-to-right chain where each actionable also conditions on the already-filled "
        "siblings. This differs from the random path in two ways: (1) the parent set for "
        "each actionable is a **subset** (not the full conditioning set), and (2) the "
        "ordering is **deterministic** rather than averaged.",
        "",
        "**Expected behaviour:**",
        "- MOONS: DAG ≈ random (only 2 actionable features; little ordering freedom).",
        "- HELOC full: DAG neutral-to-worse (smaller parent set than random path).",
        "- HELOC reduced: the cell most likely to improve OOB (denser conditioning).",
        "",
    ]

    for dataset_name, rows in all_rows.items():
        lines += [
            f"## {dataset_name.upper()}",
            "",
            "| ordering | actionable_set | n_masked | validity | lof_scores_cf | "
            "sparsity | true_actionability | proximity_l2 | frac_oob | runtime_s |",
            "|----------|---------------|---------|---------|--------------|"
            "---------|-------------------|-------------|---------|-----------|",
        ]
        for row in rows:
            lines.append(
                f"| {row['ordering']} "
                f"| {row['actionable_set']} "
                f"| {row['n_masked']} "
                f"| {row.get('validity', float('nan')):.3f} "
                f"| {row.get('lof_scores_cf', float('nan')):.3f} "
                f"| {row.get('sparsity', float('nan')):.3f} "
                f"| {row.get('true_actionability', float('nan')):.3f} "
                f"| {row.get('proximity_l2_jaccard', float('nan')):.4f} "
                f"| {row.get('frac_oob', float('nan')):.3f} "
                f"| {row.get('runtime_s', float('nan')):.1f} |"
            )

        # Generate a per-dataset verdict
        lines += ["", "### Verdict", ""]
        _append_verdict(lines, dataset_name, rows)
        lines += [""]

    lines += [
        "## Summary",
        "",
        "- `true_actionability` must be 1.0 for every cell — immutable and frozen "
        "columns are preserved by construction.",
        "- `frac_oob` measures extrapolation artefacts; only the `reduced` HELOC cells "
        "are expected to show meaningful improvement (denser conditioning).",
        "- The DAG result is an honest test of structured vs. random-permutation ordering; "
        "no further ablation dimensions are explored here.",
    ]

    summary_path = RESULTS_DIR / "exp3_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {summary_path}")


def _append_verdict(lines: List[str], dataset_name: str, rows: List[Dict]) -> None:
    """Append a short honest verdict for one dataset's ablation rows."""
    row_by_key = {(r["ordering"], r["actionable_set"]): r for r in rows}

    if dataset_name == "moons":
        r_rand = row_by_key.get(("random", "full"), {})
        r_dag = row_by_key.get(("dag", "full"), {})
        v_rand = r_rand.get("validity", float("nan"))
        v_dag = r_dag.get("validity", float("nan"))
        delta = v_dag - v_rand
        direction = "improved" if delta > 0.01 else ("worsened" if delta < -0.01 else "unchanged")
        lines.append(
            f"DAG vs random (MOONS, full): validity {direction} "
            f"({v_rand:.3f} → {v_dag:.3f}, Δ={delta:+.3f}). "
            "With only 2 actionable features, ordering freedom is minimal — "
            "near-zero delta is expected."
        )
        return

    # HELOC
    r_rand_full = row_by_key.get(("random", "full"), {})
    r_dag_full = row_by_key.get(("dag", "full"), {})
    r_rand_red = row_by_key.get(("random", "reduced"), {})
    r_dag_red = row_by_key.get(("dag", "reduced"), {})

    def _fmt(r: Dict) -> str:
        v = r.get("validity", float("nan"))
        oob = r.get("frac_oob", float("nan"))
        return f"validity={v:.3f}, frac_oob={oob:.3f}"

    lines.append(
        f"- random/full   : {_fmt(r_rand_full)}"
    )
    lines.append(
        f"- dag/full      : {_fmt(r_dag_full)}"
    )
    lines.append(
        f"- random/reduced: {_fmt(r_rand_red)}"
    )
    lines.append(
        f"- dag/reduced   : {_fmt(r_dag_red)}"
    )
    lines.append("")

    # DAG-vs-random at full actionable set
    v_rf = r_rand_full.get("validity", float("nan"))
    v_df = r_dag_full.get("validity", float("nan"))
    dag_full_dir = "improved" if v_df - v_rf > 0.01 else ("worsened" if v_rf - v_df > 0.01 else "neutral")
    lines.append(
        f"DAG vs random at full actionable set: validity {dag_full_dir} "
        f"({v_rf:.3f} → {v_df:.3f}). "
        "DAG gives each actionable a *subset* of the full conditioning set, "
        "so neutral-to-worse is expected here."
    )

    # Reduced vs full impact
    v_rr = r_rand_red.get("validity", float("nan"))
    oob_rf = r_rand_full.get("frac_oob", float("nan"))
    oob_rr = r_rand_red.get("frac_oob", float("nan"))
    red_oob_improved = (oob_rf - oob_rr) > 0.05
    lines.append(
        f"Reduced actionable set (random): validity {v_rr:.3f} vs {v_rf:.3f} (full); "
        f"frac_oob {oob_rr:.3f} vs {oob_rf:.3f} (full). "
        + (
            "OOB fraction reduced — denser conditioning helps as expected."
            if red_oob_improved
            else "OOB fraction not substantially reduced — sparse conditioning persists."
        )
    )

    # Best cell recommendation
    best = max(rows, key=lambda r: r.get("validity", -1))
    lines.append(
        f"Best HELOC cell by validity: ordering={best['ordering']}, "
        f"actionable_set={best['actionable_set']} "
        f"(validity={best.get('validity', float('nan')):.3f}, "
        f"frac_oob={best.get('frac_oob', float('nan')):.3f})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3: feature-ordering DAG ablation"
    )
    parser.add_argument(
        "--dataset",
        choices=["moons", "heloc", "all"],
        default="all",
        help="Dataset to run (default: all)",
    )
    args = parser.parse_args()

    datasets = ["moons", "heloc"] if args.dataset == "all" else [args.dataset]
    all_rows: Dict[str, List[Dict]] = {}
    for ds in datasets:
        all_rows[ds] = run_dataset_ablation(ds)

    write_exp3_summary(all_rows)
    print("\nExperiment 3 done.")


if __name__ == "__main__":
    main()
