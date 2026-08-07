"""Experiment 5: selector ablation — Strategy 1 vs Strategy 2.

Compare the two candidate-selection strategies head-to-head at the Stage-1
baseline context, on one dataset at a time:

  - ``prob_ascent``      (Strategy 1) — steepest-ascent on the target-class
                         probability; uses a ``target_only`` context.
  - ``class_divergence`` (Strategy 2) — class-divergence selector; *requires* an
                         ``all_classes`` context (non-constant Y), Decision #6.

Everything else is held at the Stage-1 baseline and **identical across the two
selectors within a dataset** so the comparison is fair: ``max_context=256``,
``temperature=1e-9`` (near-MAP commit), ``n_permutations``, and ``--max-test``.
The only *necessary* difference is the context scope (Strategy 2 needs both
classes) — this is recorded in the summary as the context-scope caveat. The
apples-to-apples contrast lives in Stage 4 (context scope is a controlled axis).

Reuses the Stage-1 Exp4 generation + metric path verbatim (incl. the inline
``frac_oob`` on unclipped CFs), so the metrics are computed exactly as in Exp4.

Outputs (under experiments/zeroshot_cf/results/):
  results/exp5_selector_<dataset>.csv  — one row per selector
  results/exp5_summary.md              — per-dataset tables + verdict + chosen selector

Usage:
  uv run python experiments/zeroshot_cf/exp5_selector_ablation.py --dataset moons
  uv run python experiments/zeroshot_cf/exp5_selector_ablation.py --dataset heloc --max-test 50
  # Keep --max-test identical across the two selectors within a dataset (this driver does).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.zeroshot_cf.exp4_greedy_cf import (  # noqa: E402
    MAX_CONTEXT,
    N_PERMUTATIONS,
    TAU,
    TEMPERATURE,
    evaluate_and_report,
    generate_counterfactuals,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SELECTORS = ["prob_ascent", "class_divergence"]

# Columns written to results/exp5_selector_<dataset>.csv, in order.
CSV_COLUMNS = [
    "selector",
    "context_scope",
    "n_test",
    "validity",
    "l0_count_mean",
    "steps_mean",
    "steps_median",
    "steps_max",
    "failure_rate",
    "lof_scores_cf",
    "true_actionability",
    "proximity_l2_jaccard",
    "frac_oob",
    "runtime_s",
]


def run_dataset_ablation(
    dataset_name: str,
    tau: float = TAU,
    temperature: float = TEMPERATURE,
    n_permutations: int = N_PERMUTATIONS,
    max_context: int = MAX_CONTEXT,
    max_test: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Run both selectors on one dataset and write the per-dataset CSV.

    Returns the list of two rows (one per selector). ``max_test``,
    ``max_context``, ``temperature`` and ``n_permutations`` are held identical
    across selectors within the dataset by passing the same values to both runs.
    """
    rows: List[Dict[str, float]] = []

    print(f"\n########## Exp5 selector ablation: {dataset_name.upper()} ##########")
    for selector in SELECTORS:
        t0 = time.perf_counter()
        X_test, y_test, X_cf, info = generate_counterfactuals(
            dataset_name,
            selector=selector,
            tau=tau,
            budget=None,
            temperature=temperature,
            n_permutations=n_permutations,
            max_context=max_context,
            max_test=max_test,
        )
        metrics = evaluate_and_report(
            dataset_name, X_test, y_test, X_cf, info, write_csv=False
        )
        runtime_s = time.perf_counter() - t0

        row = {
            "selector": selector,
            "context_scope": info["context_type"],
            "n_test": int(len(X_test)),
            "validity": metrics["validity"],
            "l0_count_mean": metrics["l0_count_mean"],
            "steps_mean": metrics["steps_mean"],
            "steps_median": metrics["steps_median"],
            "steps_max": metrics["steps_max"],
            "failure_rate": metrics["failure_rate"],
            "lof_scores_cf": metrics["lof_scores_cf"],
            "true_actionability": metrics["true_actionability"],
            "proximity_l2_jaccard": metrics["proximity_l2_jaccard"],
            "frac_oob": metrics["frac_oob"],
            "runtime_s": round(runtime_s, 2),
        }
        rows.append(row)

    csv_path = RESULTS_DIR / f"exp5_selector_{dataset_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_COLUMNS})
    print(f"\n  Wrote {csv_path}")

    return rows


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "nan"
    if abs(f) >= 1000 or (abs(f) < 1e-3 and f != 0):
        return f"{f:.3e}"
    return f"{f:.4g}"


def _read_dataset_rows(dataset_name: str) -> Optional[List[Dict[str, str]]]:
    csv_path = RESULTS_DIR / f"exp5_selector_{dataset_name}.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _verdict_for_dataset(rows: List[Dict[str, str]]) -> Dict[str, str]:
    """Per-metric winner for one dataset (higher-is-better validity; lower-is-
    better l0/steps/frac_oob/lof). Returns a dict of metric -> winning selector
    (or 'tie')."""
    by_sel = {r["selector"]: r for r in rows}
    pa, cd = by_sel.get("prob_ascent"), by_sel.get("class_divergence")
    out: Dict[str, str] = {}
    if pa is None or cd is None:
        return out

    def num(r, k):
        try:
            return float(r[k])
        except (TypeError, ValueError):
            return float("nan")

    def winner(metric, higher_better, tol):
        a, b = num(pa, metric), num(cd, metric)
        if a != a or b != b:  # NaN involved
            return "n/a"
        if abs(a - b) <= tol:
            return "tie"
        better_is_pa = (a > b) if higher_better else (a < b)
        return "prob_ascent" if better_is_pa else "class_divergence"

    out["validity"] = winner("validity", True, 1e-3)
    out["l0_count_mean"] = winner("l0_count_mean", False, 1e-3)
    out["steps_mean"] = winner("steps_mean", False, 1e-3)
    out["frac_oob"] = winner("frac_oob", False, 1e-3)
    out["lof_scores_cf"] = winner("lof_scores_cf", False, 1e-3)
    return out


def _choose_downstream_selector(dataset_rows: Dict[str, List[Dict[str, str]]]) -> tuple:
    """Pick the downstream selector (Decision #6 default tie-break: prefer
    prob_ascent, which is compatible with all four Stage-4 context strategies).
    Choose class_divergence only if it *clearly wins on plausibility without
    losing validity* on HELOC (the discriminating dataset). Returns
    (selector, rationale)."""
    heloc = dataset_rows.get("heloc")
    if heloc:
        by_sel = {r["selector"]: r for r in heloc}
        pa, cd = by_sel.get("prob_ascent"), by_sel.get("class_divergence")
        if pa is not None and cd is not None:
            def num(r, k):
                try:
                    return float(r[k])
                except (TypeError, ValueError):
                    return float("nan")

            pa_val, cd_val = num(pa, "validity"), num(cd, "validity")
            pa_oob, cd_oob = num(pa, "frac_oob"), num(cd, "frac_oob")
            if (cd_val == cd_val and pa_val == pa_val
                    and cd_oob == cd_oob and pa_oob == pa_oob):
                if cd_val >= pa_val - 1e-3 and cd_oob < pa_oob - 1e-3:
                    return (
                        "class_divergence",
                        "class_divergence wins plausibility on HELOC "
                        f"(frac_oob {cd_oob:.3g} < {pa_oob:.3g}) without losing "
                        f"validity ({cd_val:.3g} ≥ {pa_val:.3g}). NOTE: Stage-4 "
                        "*_target context cells will be skipped (Decision #6).",
                    )
    return (
        "prob_ascent",
        "Default tie-break (Decision #6): prob_ascent is compatible with all "
        "four Stage-4 context strategies and directly optimizes the flip; "
        "class_divergence did not clearly win plausibility on HELOC without a "
        "validity cost.",
    )


def write_summary() -> None:
    """(Re)build results/exp5_summary.md from whatever per-dataset CSVs exist."""
    dataset_rows: Dict[str, List[Dict[str, str]]] = {}
    for ds in ("moons", "heloc"):
        rows = _read_dataset_rows(ds)
        if rows:
            dataset_rows[ds] = rows

    lines = [
        "# Experiment 5: Selector Ablation — prob_ascent (Strategy 1) vs "
        "class_divergence (Strategy 2)",
        "",
        "One factor (the selector), two levels, across MOONS + HELOC at the "
        "Stage-1 baseline context.",
        "Held identical across the two selectors **within a dataset**: "
        "`max_context=256`, `temperature=1e-9` (MAP commit), `n_permutations`, "
        "`n_test` (= `--max-test`).",
        "",
        "> **Context-scope caveat (Decision #6).** Strategy 2 (`class_divergence`) "
        "*requires* a both-classes context pool (`all_classes`) so the Y column is "
        "non-constant; Strategy 1 (`prob_ascent`) uses a target-only context "
        "(`target_only`). The two cells within a dataset therefore *necessarily* "
        "differ in context scope — this is **each selector at its natural/required "
        "context**, not both at an identical context. The apples-to-apples contrast "
        "is deferred to Stage 4, where the context strategy is a controlled axis.",
        "",
    ]

    metric_cols = [
        "context_scope", "n_test", "validity", "l0_count_mean", "steps_mean",
        "steps_median", "steps_max", "failure_rate", "lof_scores_cf",
        "true_actionability", "proximity_l2_jaccard", "frac_oob", "runtime_s",
    ]
    for ds, rows in dataset_rows.items():
        lines += [f"## {ds.upper()}", ""]
        header = "| selector | " + " | ".join(metric_cols) + " |"
        sep = "|" + "---|" * (len(metric_cols) + 1)
        lines += [header, sep]
        for r in rows:
            cells = [r["selector"]] + [_fmt(r.get(c, "")) for c in metric_cols]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        verdict = _verdict_for_dataset(rows)
        if verdict:
            lines += ["**Per-metric winner:**", ""]
            label = {
                "validity": "validity (higher better)",
                "l0_count_mean": "L0 count (lower better)",
                "steps_mean": "steps-to-flip (lower better)",
                "frac_oob": "plausibility frac_oob (lower better)",
                "lof_scores_cf": "plausibility LOF (lower better)",
            }
            for k in ("validity", "l0_count_mean", "steps_mean", "frac_oob", "lof_scores_cf"):
                if k in verdict:
                    lines.append(f"- {label[k]}: **{verdict[k]}**")
            lines.append("")

    selector, rationale = _choose_downstream_selector(dataset_rows)
    lines += [
        "## Chosen downstream selector (used by Stage 4)",
        "",
        f"**`{selector}`**",
        "",
        rationale,
        "",
    ]
    if not dataset_rows:
        lines += ["_No per-dataset CSVs found yet — run the ablation first._", ""]

    out_path = RESULTS_DIR / "exp5_summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main() -> None:
    from experiments.zeroshot_cf.local_data import LOCAL_DATASET_NAMES

    parser = argparse.ArgumentParser(
        description="Experiment 5: selector ablation (prob_ascent vs class_divergence)"
    )
    parser.add_argument(
        "--dataset",
        choices=["moons", "heloc", "all", *sorted(LOCAL_DATASET_NAMES)],
        default="moons",
        help="'all' runs moons+heloc; local (CETGFN-ported) datasets run individually.",
    )
    parser.add_argument("--tau", type=float, default=TAU,
                        help=f"Flip probability threshold (default: {TAU}).")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Committed-value temperature (default: {TEMPERATURE} ≈ MAP).")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS,
                        help=f"Imputation permutations (default: {N_PERMUTATIONS}).")
    parser.add_argument("--max-context", type=int, default=MAX_CONTEXT,
                        help=f"Max context rows (default: {MAX_CONTEXT}).")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Number of test points, identical across both selectors "
                             "(default: moons=100, heloc=50; -1 for full split).")
    args = parser.parse_args()

    datasets = ["moons", "heloc"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_dataset_ablation(
            ds,
            tau=args.tau,
            temperature=args.temperature,
            n_permutations=args.n_permutations,
            max_context=args.max_context,
            max_test=args.max_test,
        )

    write_summary()
    print("\nDone.")


if __name__ == "__main__":
    main()
