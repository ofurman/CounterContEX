"""Experiment 7: multi-pass budget × classifier-label conditioning.

Follow-up to the Exp6 v3 context sweep: the best HELOC cell (``knn_both@512``)
tops out at validity 0.96, and by construction the failed points already
changed **every** actionable column once. This experiment tests two levers on
the residual failures, on the winning ``knn_both`` strategy only:

1. **Budget** (``--max-rounds``): allow the greedy loop to re-edit columns in
   additional passes. Within a round each column is eligible at most once;
   after an unflipped round, eligibility resets and every column's re-draw is
   conditioned on the *current* CF (the other columns moved since it was set)
   — coordinate ascent toward the target-class conditional mode. Rounds >= 2
   commit only strictly p_target-improving edits and the loop stops at a
   fixed point. ``--max-rounds 1`` reproduces the Exp6 single-pass budget.

2. **Conditioning labels** (``--labels``): Exp6 conditioned generation on
   ground-truth ``y_train``, while validity is judged by the discriminator —
   a generator/oracle mismatch for boundary points whose data labels disagree
   with the discriminator. ``--labels disc`` replaces the context labels with
   ``disc.predict(X_train)`` (both for the appended conditioning Y column
   and, for *_target pools, class filtering — irrelevant for ``knn_both``,
   which never filters). ``--labels data`` keeps the Exp6 behaviour.

Per-point kNN context (Decision #5), sizes swept via ``--sizes``. Metrics go
through the Exp4 ``evaluate_and_report`` path verbatim, plus round bookkeeping:
``validity_r1`` (fraction flipped within round 1 — isolates the label effect
from the extra-budget effect in one run) and ``n_rescued`` (points flipped in
rounds >= 2, i.e. the direct payoff of the bigger budget).

Outputs (under experiments/zeroshot_cf/results/ or $ZEROSHOT_CF_RESULTS_DIR):
  results/exp7_multipass_<dataset>.csv  — one row per (size,) cell
  results/exp7_summary.md               — per-dataset tables

Usage:
  uv run python experiments/zeroshot_cf/exp7_multipass_budget.py --dataset heloc --max-rounds 3 --labels disc
  uv run python experiments/zeroshot_cf/exp7_multipass_budget.py --dataset all --sizes 512,1024 --max-rounds 1 --labels disc
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.zeroshot_cf.exp4_greedy_cf import (  # noqa: E402
    N_ESTIMATORS,
    N_PERMUTATIONS,
    TAU,
    TEMPERATURE,
    _DATASET_PARAMS,
    evaluate_and_report,
)
from experiments.zeroshot_cf.exp6_context_ablation import (  # noqa: E402
    _parse_sizes,
    _resolve_max_test,
)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR = Path(os.environ.get("ZEROSHOT_CF_RESULTS_DIR", DEFAULT_RESULTS_DIR))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIZES = [512, 1024]
STRATEGY = "knn_both"  # the Exp6 v3 winner; the only strategy Exp7 runs

CSV_COLUMNS = [
    "selector",
    "strategy",
    "labels",
    "max_rounds",
    "size",
    "effective_size",
    "n_test",
    "validity",
    "validity_r1",
    "n_rescued",
    "failure_rate",
    "rounds_mean",
    "rounds_max",
    "l0_count_mean",
    "l0_count_median",
    "l0_count_max",
    "steps_mean",
    "steps_median",
    "steps_max",
    "lof_scores_cf",
    "sparsity",
    "true_actionability",
    "proximity_l2_jaccard",
    "frac_oob",
    "runtime_s",
]


def _run_cell(
    dataset_name: str,
    selector: str,
    size: int,
    *,
    bundle,
    disc_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_target: np.ndarray,
    y_context: np.ndarray,
    labels_mode: str,
    actionable_idx: List[int],
    immutable_idx: List[int],
    clf,
    reg,
    tau: float,
    temperature: float,
    n_permutations: int,
    max_rounds: int,
) -> Dict[str, float]:
    """Run one knn_both cell with per-query context and multi-pass budget."""
    from experiments.zeroshot_cf.greedy import greedy_counterfactual
    from experiments.zeroshot_cf.sampler import ConditionalDensitySampler

    X_train = bundle.X_train
    n = len(X_test)
    eff_budget = len(actionable_idx)
    pool_size = int(len(X_train))  # knn_both: no class filter
    eff = min(size, pool_size)

    print(
        f"\n  --- cell: size={size} strategy={STRATEGY} labels={labels_mode} "
        f"max_rounds={max_rounds} ---"
    )

    X_cf = X_test.copy()
    changed_per_point: List[List[int]] = [[] for _ in range(n)]
    flipped_per_point: List[bool] = [False] * n
    steps_per_point: List[int] = [0] * n
    rounds_per_point: List[int] = [0] * n

    t0 = time.perf_counter()

    # Batch by target class only to mirror Exp6's per-class sampler seeding
    # (random_state=42+target_cls); the knn context itself is per-query.
    for target_cls in np.unique(y_target):
        target_cls = int(target_cls)
        test_idx = np.where(y_target == target_cls)[0]
        if len(test_idx) == 0:
            continue

        sampler = ConditionalDensitySampler(
            clf=clf,
            reg=reg,
            append_target=True,
            n_permutations=n_permutations,
            temperature=temperature,
            random_state=42 + target_cls,
        )

        for i in test_idx:
            sampler.set_context(
                X_train,
                y_context=y_context,
                target_class=None,  # knn_both: pool from both classes
                max_context=size,
                selection="knn",
                query=X_test[i],
            )
            x_cf, changed, gi = greedy_counterfactual(
                sampler, disc_model, X_test[i], target_cls,
                actionable_idx, selector,
                tau=tau, budget=eff_budget, temperature=temperature,
                max_rounds=max_rounds,
            )
            X_cf[i] = x_cf
            changed_per_point[i] = changed
            flipped_per_point[i] = gi["flipped"]
            steps_per_point[i] = gi["steps"]
            rounds_per_point[i] = gi["rounds"]

    runtime_s = time.perf_counter() - t0

    info = {
        "bundle": bundle,
        "y_pred": y_pred,
        "y_target": y_target,
        "actionable_idx": actionable_idx,
        "immutable_idx": immutable_idx,
        "disc_model": disc_model,
        "selector": selector,
        "context_type": "both",
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

    flipped = np.asarray(flipped_per_point, dtype=bool)
    rounds = np.asarray(rounds_per_point, dtype=float)
    # Flipped within the first pass — the Exp6-comparable number (isolates the
    # label effect); rescued = flipped only thanks to rounds >= 2.
    validity_r1 = float((flipped & (rounds <= 1)).mean())
    n_rescued = int((flipped & (rounds >= 2)).sum())
    print(
        f"  validity={metrics['validity']:.3f} validity_r1={validity_r1:.3f} "
        f"rescued={n_rescued} rounds_max={int(rounds.max()) if n else 0}"
    )

    return {
        "selector": selector,
        "strategy": STRATEGY,
        "labels": labels_mode,
        "max_rounds": max_rounds,
        "size": size,
        "effective_size": eff,
        "n_test": int(n),
        "validity": metrics["validity"],
        "validity_r1": validity_r1,
        "n_rescued": n_rescued,
        "failure_rate": metrics["failure_rate"],
        "rounds_mean": float(rounds.mean()) if n else float("nan"),
        "rounds_max": int(rounds.max()) if n else 0,
        "l0_count_mean": metrics["l0_count_mean"],
        "l0_count_median": metrics["l0_count_median"],
        "l0_count_max": metrics["l0_count_max"],
        "steps_mean": metrics["steps_mean"],
        "steps_median": metrics["steps_median"],
        "steps_max": metrics["steps_max"],
        "lof_scores_cf": metrics["lof_scores_cf"],
        "sparsity": metrics["sparsity"],
        "true_actionability": metrics["true_actionability"],
        "proximity_l2_jaccard": metrics["proximity_l2_jaccard"],
        "frac_oob": metrics["frac_oob"],
        "runtime_s": round(runtime_s, 2),
    }


def _write_csv(dataset_name: str, rows: List[Dict[str, float]]) -> Path:
    """Persist completed cells atomically (same recipe as Exp6: survive Slurm
    wall-time kills mid-run and mid-write)."""
    csv_path = RESULTS_DIR / f"exp7_multipass_{dataset_name}.csv"
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, csv_path)
    return csv_path


def run_dataset(
    dataset_name: str,
    selector: str = "prob_ascent",
    labels_mode: str = "disc",
    max_rounds: int = 3,
    tau: float = TAU,
    temperature: float = TEMPERATURE,
    n_permutations: int = N_PERMUTATIONS,
    max_test: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Run the size sweep for one dataset and write the per-dataset CSV."""
    from experiments.zeroshot_cf.checkpoints import get_models
    from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset
    from experiments.zeroshot_cf.discriminator import train_discriminator

    MAX_TEST = _resolve_max_test(dataset_name, max_test)

    print(f"\n########## Exp7 multipass budget: {dataset_name.upper()} ##########")
    print(
        f"  selector={selector}, strategy={STRATEGY}, labels={labels_mode}, "
        f"max_rounds={max_rounds}, tau={tau}, temperature={temperature}, "
        f"n_permutations={n_permutations}, max_test={MAX_TEST}, sizes={SIZES}"
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

    # Conditioning labels for the context: the discriminator's own predictions
    # align the generator's Y with the validity oracle; ground-truth labels
    # reproduce the Exp6 conditioning.
    if labels_mode == "disc":
        y_context = disc_model.predict(X_train)
        n_relabel = int((y_context != y_train).sum())
        print(
            f"  Context labels: disc.predict(X_train) — {n_relabel}/{len(y_train)} "
            f"rows relabelled vs ground truth"
        )
    else:
        y_context = y_train
        print("  Context labels: ground-truth y_train (Exp6 behaviour)")

    print("  Loading TabPFN models …")
    clf, reg = get_models(n_estimators=N_ESTIMATORS)

    rows: List[Dict[str, float]] = []
    for size in SIZES:
        row = _run_cell(
            dataset_name, selector, size,
            bundle=bundle, disc_model=disc_model,
            X_test=X_test, y_test=y_test, y_pred=y_pred, y_target=y_target,
            y_context=y_context, labels_mode=labels_mode,
            actionable_idx=actionable_idx, immutable_idx=immutable_idx,
            clf=clf, reg=reg,
            tau=tau, temperature=temperature, n_permutations=n_permutations,
            max_rounds=max_rounds,
        )
        rows.append(row)
        csv_path = _write_csv(dataset_name, rows)
        print(f"  Wrote {csv_path}  ({len(rows)} completed cells)")

    print(f"\n  Completed {len(rows)} cells for {dataset_name}")
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


def write_summary() -> None:
    """(Re)build results/exp7_summary.md from whatever exp7_multipass_*.csv exist."""
    headline = [
        "size", "labels", "max_rounds", "validity", "validity_r1", "n_rescued",
        "failure_rate", "rounds_max", "l0_count_mean", "steps_mean",
        "frac_oob", "lof_scores_cf", "runtime_s",
    ]
    lines = [
        "# Experiment 7: Multi-pass budget × classifier-label conditioning",
        "",
        f"Strategy fixed to `{STRATEGY}` (Exp6 v3 winner). Two levers vs Exp6: "
        "`max_rounds` (re-edit columns across greedy passes; 1 = Exp6 budget) "
        "and `labels` (`disc` = condition context on discriminator predictions, "
        "`data` = ground-truth `y_train` as in Exp6).",
        "",
        "> `validity_r1` = fraction flipped within round 1 (Exp6-comparable; "
        "isolates the label effect). `n_rescued` = points flipped only in "
        "rounds >= 2 (the budget effect). L0 counts distinct changed columns; "
        "`steps_*` counts total edits.",
        "",
    ]
    found = False
    for ds in ("moons", "heloc"):
        csv_path = RESULTS_DIR / f"exp7_multipass_{ds}.csv"
        if not csv_path.exists():
            continue
        found = True
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        n_test = rows[0].get("n_test", "?") if rows else "?"
        selector = rows[0].get("selector", "?") if rows else "?"
        lines += [
            f"## {ds.upper()}",
            "",
            f"Selector: `{selector}` · cells: {len(rows)} · n_test: {n_test}",
            "",
            "| " + " | ".join(headline) + " |",
            "|" + "---|" * len(headline),
        ]
        for r in rows:
            lines.append(
                "| " + " | ".join(_fmt(r.get(k, "")) for k in headline) + " |"
            )
        lines.append("")
    if not found:
        lines += ["_No per-dataset CSVs found yet — run the experiment first._", ""]

    out_path = RESULTS_DIR / "exp7_summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main() -> None:
    global RESULTS_DIR, SIZES

    parser = argparse.ArgumentParser(
        description="Experiment 7: multi-pass budget × classifier-label conditioning"
    )
    parser.add_argument("--dataset", choices=["moons", "heloc", "all"], default="heloc")
    parser.add_argument(
        "--selector",
        choices=["prob_ascent", "class_divergence"],
        default="prob_ascent",
        help="Candidate-selection strategy (default: prob_ascent).",
    )
    parser.add_argument(
        "--labels",
        choices=["disc", "data"],
        default="disc",
        help="Context conditioning labels: 'disc' = discriminator predictions "
             "on X_train (default), 'data' = ground-truth y_train (Exp6).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Greedy passes over the actionable columns; 1 = Exp6 single-pass "
             "budget (default: 3).",
    )
    parser.add_argument("--tau", type=float, default=TAU,
                        help=f"Flip probability threshold (default: {TAU}).")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Committed-value temperature (default: {TEMPERATURE} ≈ MAP).")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS,
                        help=f"Imputation permutations (default: {N_PERMUTATIONS}).")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Number of test points (default: moons=100, heloc=50; "
                             "-1 for full split).")
    parser.add_argument(
        "--sizes",
        default=",".join(map(str, SIZES)),
        help=f"Comma-separated context sizes (default: {','.join(map(str, SIZES))}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for exp7 CSV/summary outputs. Useful for Slurm arrays.",
    )
    args = parser.parse_args()

    SIZES = _parse_sizes(args.sizes)
    if args.results_dir is not None:
        RESULTS_DIR = args.results_dir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ["moons", "heloc"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_dataset(
            ds,
            selector=args.selector,
            labels_mode=args.labels,
            max_rounds=args.max_rounds,
            tau=args.tau,
            temperature=args.temperature,
            n_permutations=args.n_permutations,
            max_test=args.max_test,
        )

    write_summary()
    print("\nDone.")


if __name__ == "__main__":
    main()
