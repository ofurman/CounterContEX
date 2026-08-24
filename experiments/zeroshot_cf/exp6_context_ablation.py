"""Experiment 6: context ablation — size × strategy grid (Stage 4).

For the Stage-2 winning selector, sweep the conditioning context across the
two-factor grid:

  - **size** (`max_context`): 256, 512, 1024, 2048 — capped at the available
    pool; the actual rows used (`effective_size`) is recorded per cell.
  - **strategy** (class scope × selection method):
      ``random_target`` ≡ (target_class=<t>, selection="random")  [baseline]
      ``random_both``   ≡ (target_class=None, selection="random")
      ``knn_target``    ≡ (target_class=<t>, selection="knn")
      ``knn_both``      ≡ (target_class=None, selection="knn")

Grid = 4 × 4 = 16 cells per dataset for ``prob_ascent``. ``class_divergence``
needs a both-classes pool (non-constant Y), so the 8 ``*_target`` cells are
**skipped (logged)**, leaving 8 cells (Decision #6).

kNN context is selected **per query point** (Decision #5): for ``knn_*``
strategies, the context is fit per test point from that point's factual row.
``random_*`` strategies fit the context once per target-class batch and reuse it
(mirrors Exp4's batched fit) for speed; correctness is identical because the
random subsample does not depend on the query.

Everything else (selector, temperature, n_permutations, --max-test, tau) is held
**identical across all cells within a dataset**, so size/strategy are the only
things that vary. Metrics — including the inline ``frac_oob`` on the UNCLIPPED
CFs — are computed by the Exp4 ``evaluate_and_report`` path verbatim.

Outputs (under experiments/zeroshot_cf/results/):
  results/exp6_context_<dataset>.csv  — one row per cell
  results/exp6_summary.md             — per-dataset size×strategy tables + verdict

Usage:
  uv run python experiments/zeroshot_cf/exp6_context_ablation.py --dataset moons
  uv run python experiments/zeroshot_cf/exp6_context_ablation.py --dataset heloc --selector prob_ascent --max-test 50
  # The full [256,512,1024,2048] sizes are fixed; --max-test is held identical across cells.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.zeroshot_cf.exp4_greedy_cf import (  # noqa: E402
    MAX_CONTEXT,
    N_ESTIMATORS,
    N_PERMUTATIONS,
    TAU,
    TEMPERATURE,
    _DATASET_PARAMS,
    evaluate_and_report,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIZES = [256, 512, 1024, 2048]
STRATEGIES = ["random_target", "random_both", "knn_target", "knn_both"]

# strategy -> (class_scope, selection). class_scope drives target_class:
#   "target" -> target_class = <per-point target>; "both" -> target_class = None.
STRATEGY_SPEC = {
    "random_target": ("target", "random"),
    "random_both": ("both", "random"),
    "knn_target": ("target", "knn"),
    "knn_both": ("both", "knn"),
}

# Columns written to results/exp6_context_<dataset>.csv, in order (grids.md).
CSV_COLUMNS = [
    "selector",
    "size",
    "effective_size",
    "strategy",
    "class_scope",
    "selection",
    "n_test",
    "validity",
    "l0_count_mean",
    "l0_count_median",
    "l0_count_max",
    "steps_mean",
    "steps_median",
    "steps_max",
    "failure_rate",
    "lof_scores_cf",
    "sparsity",
    "true_actionability",
    "proximity_l2_jaccard",
    "frac_oob",
    "runtime_s",
]


def _strategies_for_selector(selector: str) -> List[str]:
    """Strategies to run for a selector. ``class_divergence`` needs a
    both-classes pool, so the ``*_target`` strategies are skipped (Decision #6)."""
    if selector == "class_divergence":
        kept = [s for s in STRATEGIES if STRATEGY_SPEC[s][0] == "both"]
        skipped = [s for s in STRATEGIES if STRATEGY_SPEC[s][0] != "both"]
        print(
            f"  [Decision #6] selector={selector} requires a both-classes pool; "
            f"SKIPPING {len(SIZES) * len(skipped)} *_target cells "
            f"({skipped}); running {len(SIZES) * len(kept)} cells."
        )
        return kept
    return list(STRATEGIES)


def _resolve_max_test(dataset_name: str, max_test: Optional[int]) -> Optional[int]:
    params = _DATASET_PARAMS.get(dataset_name, {"max_test": 50})
    if max_test is not None and max_test < 0:
        return None
    if max_test is not None:
        return max_test
    return params["max_test"]


def _run_cell(
    dataset_name: str,
    selector: str,
    size: int,
    strategy: str,
    *,
    bundle,
    disc_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_target: np.ndarray,
    actionable_idx: List[int],
    immutable_idx: List[int],
    clf,
    reg,
    tau: float,
    temperature: float,
    n_permutations: int,
) -> Dict[str, float]:
    """Run one (size, strategy) cell on a pre-loaded dataset. Returns a CSV row."""
    from experiments.zeroshot_cf.greedy import greedy_counterfactual
    from experiments.zeroshot_cf.sampler import ConditionalDensitySampler

    class_scope, selection = STRATEGY_SPEC[strategy]
    X_train = bundle.X_train
    y_train = bundle.y_train
    n = len(X_test)
    eff_budget = len(actionable_idx)

    print(
        f"\n  --- cell: size={size} strategy={strategy} "
        f"(scope={class_scope}, selection={selection}) ---"
    )

    X_cf = X_test.copy()
    changed_per_point: List[List[int]] = [[] for _ in range(n)]
    flipped_per_point: List[bool] = [False] * n
    steps_per_point: List[int] = [0] * n
    effective_sizes: List[int] = []

    t0 = time.perf_counter()

    # Batch test points by their per-point target class. Within a batch the
    # class scope is fixed, so for random selection the context is fit ONCE and
    # reused; for knn it is re-fit per test point from that point's factual row.
    for target_cls in np.unique(y_target):
        target_cls = int(target_cls)
        test_idx = np.where(y_target == target_cls)[0]
        if len(test_idx) == 0:
            continue

        ctx_target = target_cls if class_scope == "target" else None
        # Pool the selection draws from (for effective_size / pool_size logging).
        if ctx_target is not None:
            pool_size = int((y_train == ctx_target).sum())
        else:
            pool_size = int(len(X_train))

        sampler = ConditionalDensitySampler(
            clf=clf,
            reg=reg,
            append_target=True,
            n_permutations=n_permutations,
            temperature=temperature,
            random_state=42 + target_cls,
        )

        if selection == "random":
            # One fit per class batch (query-independent), reused for all points.
            sampler.set_context(
                X_train,
                y_context=y_train,
                target_class=ctx_target,
                max_context=size,
                selection="random",
            )
            eff = min(size, pool_size)
            for i in test_idx:
                effective_sizes.append(eff)
                x_cf, changed, gi = greedy_counterfactual(
                    sampler, disc_model, X_test[i], target_cls,
                    actionable_idx, selector,
                    tau=tau, budget=eff_budget, temperature=temperature,
                )
                X_cf[i] = x_cf
                changed_per_point[i] = changed
                flipped_per_point[i] = gi["flipped"]
                steps_per_point[i] = gi["steps"]
        else:  # knn — per-query context fit (Decision #5)
            eff = min(size, pool_size)
            for i in test_idx:
                effective_sizes.append(eff)
                sampler.set_context(
                    X_train,
                    y_context=y_train,
                    target_class=ctx_target,
                    max_context=size,
                    selection="knn",
                    query=X_test[i],
                )
                x_cf, changed, gi = greedy_counterfactual(
                    sampler, disc_model, X_test[i], target_cls,
                    actionable_idx, selector,
                    tau=tau, budget=eff_budget, temperature=temperature,
                )
                X_cf[i] = x_cf
                changed_per_point[i] = changed
                flipped_per_point[i] = gi["flipped"]
                steps_per_point[i] = gi["steps"]

    runtime_s = time.perf_counter() - t0

    # effective_size: actual rows used, capped at size and at the pool. We report
    # the max over points (the cap binds identically for all points of a scope;
    # both-scope is one pool, target-scope pools by class but the cap is per-cell).
    effective_size = int(max(effective_sizes)) if effective_sizes else 0
    assert effective_size <= size, f"effective_size {effective_size} > size {size}"

    info = {
        "bundle": bundle,
        "y_pred": y_pred,
        "y_target": y_target,
        "actionable_idx": actionable_idx,
        "immutable_idx": immutable_idx,
        "disc_model": disc_model,
        "selector": selector,
        "context_type": class_scope,
        "tau": tau,
        "budget": eff_budget,
        "temperature": temperature,
        "n_permutations": n_permutations,
        "max_context": size,
        "changed_per_point": changed_per_point,
        "flipped_per_point": flipped_per_point,
        "steps_per_point": steps_per_point,
    }
    metrics = evaluate_and_report(
        dataset_name, X_test, y_test, X_cf, info, write_csv=False
    )

    return {
        "selector": selector,
        "size": size,
        "effective_size": effective_size,
        "strategy": strategy,
        "class_scope": class_scope,
        "selection": selection,
        "n_test": int(n),
        "validity": metrics["validity"],
        "l0_count_mean": metrics["l0_count_mean"],
        "l0_count_median": metrics["l0_count_median"],
        "l0_count_max": metrics["l0_count_max"],
        "steps_mean": metrics["steps_mean"],
        "steps_median": metrics["steps_median"],
        "steps_max": metrics["steps_max"],
        "failure_rate": metrics["failure_rate"],
        "lof_scores_cf": metrics["lof_scores_cf"],
        "sparsity": metrics["sparsity"],
        "true_actionability": metrics["true_actionability"],
        "proximity_l2_jaccard": metrics["proximity_l2_jaccard"],
        "frac_oob": metrics["frac_oob"],
        "runtime_s": round(runtime_s, 2),
    }


def run_dataset_ablation(
    dataset_name: str,
    selector: str = "prob_ascent",
    tau: float = TAU,
    temperature: float = TEMPERATURE,
    n_permutations: int = N_PERMUTATIONS,
    max_test: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Run the full size×strategy grid for one dataset and write the per-dataset
    CSV. Loads the dataset / discriminator / TabPFN models ONCE and reuses them
    across all cells (only the context fit varies)."""
    from experiments.zeroshot_cf.checkpoints import get_models
    from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset
    from experiments.zeroshot_cf.discriminator import train_discriminator

    MAX_TEST = _resolve_max_test(dataset_name, max_test)

    print(f"\n########## Exp6 context ablation: {dataset_name.upper()} ##########")
    print(
        f"  selector={selector}, tau={tau}, temperature={temperature}, "
        f"n_permutations={n_permutations}, max_test={MAX_TEST}, "
        f"sizes={SIZES}"
    )

    bundle = load_dataset(dataset_name)
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test = bundle.X_test[:MAX_TEST]
    y_test = bundle.y_test[:MAX_TEST]

    actionable_idx, immutable_idx = get_actionable_immutable(dataset_name, bundle)
    print(
        f"  Features: {X_train.shape[1]} total, {len(actionable_idx)} actionable, "
        f"{len(immutable_idx)} immutable; test points: {len(X_test)}; "
        f"train pool: {len(X_train)}"
    )

    disc_model = train_discriminator(X_train, y_train, X_test, y_test, dataset_name)
    y_pred = disc_model.predict(X_test)
    y_target = 1 - y_pred
    print(f"  Target distribution: {np.bincount(y_target)}")

    print("  Loading TabPFN models …")
    clf, reg = get_models(n_estimators=N_ESTIMATORS)

    strategies = _strategies_for_selector(selector)

    rows: List[Dict[str, float]] = []
    for size in SIZES:
        for strategy in strategies:
            row = _run_cell(
                dataset_name, selector, size, strategy,
                bundle=bundle, disc_model=disc_model,
                X_test=X_test, y_test=y_test, y_pred=y_pred, y_target=y_target,
                actionable_idx=actionable_idx, immutable_idx=immutable_idx,
                clf=clf, reg=reg,
                tau=tau, temperature=temperature, n_permutations=n_permutations,
            )
            rows.append(row)

    csv_path = RESULTS_DIR / f"exp6_context_{dataset_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_COLUMNS})
    print(f"\n  Wrote {csv_path}  ({len(rows)} cells)")

    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

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
    csv_path = RESULTS_DIR / f"exp6_context_{dataset_name}.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _num(r: Dict[str, str], k: str) -> float:
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def _grid_table(rows: List[Dict[str, str]], metric: str) -> List[str]:
    """A size (rows) × strategy (cols) table for one metric."""
    strategies = []
    for r in rows:
        if r["strategy"] not in strategies:
            strategies.append(r["strategy"])
    sizes = []
    for r in rows:
        if r["size"] not in sizes:
            sizes.append(r["size"])
    by = {(r["size"], r["strategy"]): r for r in rows}

    header = "| size \\ strategy | " + " | ".join(strategies) + " |"
    sep = "|" + "---|" * (len(strategies) + 1)
    out = [f"**{metric}**", "", header, sep]
    for s in sizes:
        cells = [str(s)]
        for strat in strategies:
            r = by.get((s, strat))
            cells.append(_fmt(r.get(metric, "")) if r else "—")
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def _best_cells(rows: List[Dict[str, str]]) -> List[str]:
    """Auto-derived best-validity (higher better) and best-frac_oob (lower
    better) cells for one dataset."""
    out: List[str] = []
    valid_rows = [r for r in rows if _num(r, "validity") == _num(r, "validity")]
    if valid_rows:
        best_val = max(valid_rows, key=lambda r: (_num(r, "validity"), -_num(r, "frac_oob")))
        out.append(
            f"- **Best validity cell**: size={best_val['size']}, "
            f"strategy={best_val['strategy']} "
            f"(validity={_fmt(best_val['validity'])}, "
            f"frac_oob={_fmt(best_val['frac_oob'])}, "
            f"l0_count_mean={_fmt(best_val['l0_count_mean'])})"
        )
    oob_rows = [r for r in rows if _num(r, "frac_oob") == _num(r, "frac_oob")]
    if oob_rows:
        best_oob = min(oob_rows, key=lambda r: (_num(r, "frac_oob"), -_num(r, "validity")))
        out.append(
            f"- **Best frac_oob cell**: size={best_oob['size']}, "
            f"strategy={best_oob['strategy']} "
            f"(frac_oob={_fmt(best_oob['frac_oob'])}, "
            f"validity={_fmt(best_oob['validity'])}, "
            f"lof_scores_cf={_fmt(best_oob['lof_scores_cf'])})"
        )
    out.append("")
    return out


def write_summary() -> None:
    """(Re)build results/exp6_summary.md from whatever exp6_context_*.csv exist."""
    dataset_rows: Dict[str, List[Dict[str, str]]] = {}
    for ds in ("moons", "heloc"):
        rows = _read_dataset_rows(ds)
        if rows:
            dataset_rows[ds] = rows

    lines = [
        "# Experiment 6: Context Ablation — size × strategy",
        "",
        "Two-factor grid (size `{256, 512, 1024, 2048}` × strategy "
        "`{random_target, random_both, knn_target, knn_both}`) at the Stage-2 "
        "winning selector, on each dataset.",
        "Held identical across all cells **within a dataset**: selector, "
        "`temperature=1e-9` (MAP commit), `n_permutations`, `tau`, "
        "`n_test` (= `--max-test`).",
        "",
        "> **Strategy = class scope × selection.** `*_target` draws context from "
        "the per-point target-class pool; `*_both` from all training rows. "
        "`random_*` uniformly subsamples (one fit per class batch, reused); "
        "`knn_*` keeps the `size` nearest neighbours to each factual point "
        "(re-fit per test point, Decision #5).",
        "> **`effective_size`** is the rows actually used, capped at `size` and at "
        "the available pool (`effective_size <= size`, `<= pool_size`).",
        "> If the selector is `class_divergence`, the 8 `*_target` cells are "
        "skipped (it needs a both-classes pool); only 8 `*_both` cells appear.",
        "",
    ]

    headline = ["validity", "l0_count_mean", "frac_oob", "lof_scores_cf"]
    for ds, rows in dataset_rows.items():
        n_test = rows[0].get("n_test", "?") if rows else "?"
        selector = rows[0].get("selector", "?") if rows else "?"
        lines += [
            f"## {ds.upper()}",
            "",
            f"Selector: `{selector}` · cells: {len(rows)} · n_test: {n_test}",
            "",
        ]
        for metric in headline:
            lines += _grid_table(rows, metric)
        lines += ["**Auto-derived best cells:**", ""]
        lines += _best_cells(rows)

    lines += [
        "## Verdict (recommended context config)",
        "",
        "_Placeholder — to be filled by the orchestrator after the full GPU run._",
        "",
        "Questions the real run should answer (per Stage 4):",
        "- Does **larger context** lift HELOC validity / lower `frac_oob`, and "
        "where does it saturate?",
        "- Does **kNN** beat **random** at equal size? Does **both-classes** vs "
        "**target-only** matter?",
        "- The single recommended **(selector, size, strategy)** for HELOC and for "
        "MOONS, with the metric trade-off stated.",
        "",
    ]
    if not dataset_rows:
        lines += ["_No per-dataset CSVs found yet — run the ablation first._", ""]

    out_path = RESULTS_DIR / "exp6_summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 6: context ablation (size × strategy grid)"
    )
    parser.add_argument("--dataset", choices=["moons", "heloc", "all"], default="moons")
    parser.add_argument(
        "--selector",
        choices=["prob_ascent", "class_divergence"],
        default="prob_ascent",
        help="Stage-2 winning selector (default: prob_ascent → 16 cells; "
             "class_divergence → 8 cells, *_target skipped).",
    )
    parser.add_argument("--tau", type=float, default=TAU,
                        help=f"Flip probability threshold (default: {TAU}).")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Committed-value temperature (default: {TEMPERATURE} ≈ MAP).")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS,
                        help=f"Imputation permutations (default: {N_PERMUTATIONS}).")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Number of test points, identical across all cells "
                             "(default: moons=100, heloc=50; -1 for full split).")
    args = parser.parse_args()

    datasets = ["moons", "heloc"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_dataset_ablation(
            ds,
            selector=args.selector,
            tau=args.tau,
            temperature=args.temperature,
            n_permutations=args.n_permutations,
            max_test=args.max_test,
        )

    write_summary()
    print("\nDone.")


if __name__ == "__main__":
    main()
