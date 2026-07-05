"""Experiment 7: budget sweep for revisit-enabled greedy CFs (Stage 5).

Sweeps the greedy commit budget above |A| at the Stage-4 recommended context
configuration:

  - MOONS: prob_ascent + random_both@512
  - HELOC: prob_ascent + knn_both@256

The key distinction after Stage 5 is that ``steps`` is the ordered commit count
and may exceed |A|, while ``l0_count`` is the number of distinct features changed.
"""

from __future__ import annotations

import argparse
import csv
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
    evaluate_and_report,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SELECTOR = "prob_ascent"

BUDGETS = {
    "moons": [2, 4, 8, 16, 32, 64],
    "heloc": [17, 34, 51, 100, 250, 1000],
}

CONFIGS = {
    "moons": {"strategy": "random_both", "size": 512, "class_scope": "both", "selection": "random"},
    "heloc": {"strategy": "knn_both", "size": 256, "class_scope": "both", "selection": "knn"},
}

DEFAULT_MAX_TEST = {"moons": 100, "heloc": 30}

CSV_COLUMNS = [
    "dataset",
    "selector",
    "strategy",
    "size",
    "effective_size",
    "budget",
    "n_test",
    "validity",
    "failure_rate",
    "l0_count_mean",
    "steps_mean",
    "steps_max",
    "proximity_l2_jaccard",
    "lof_scores_cf",
    "frac_oob",
    "true_actionability",
    "runtime_s",
]


def _resolve_max_test(dataset_name: str, max_test: Optional[int]) -> Optional[int]:
    if max_test is not None and max_test < 0:
        return None
    if max_test is not None:
        return max_test
    return DEFAULT_MAX_TEST[dataset_name]


def _run_budget(
    dataset_name: str,
    budget: int,
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
    stall_eps: float,
    n_permutations: int,
) -> Dict[str, float]:
    from experiments.zeroshot_cf.greedy import greedy_counterfactual
    from experiments.zeroshot_cf.sampler import ConditionalDensitySampler

    config = CONFIGS[dataset_name]
    size = int(config["size"])
    class_scope = str(config["class_scope"])
    selection = str(config["selection"])
    strategy = str(config["strategy"])
    X_train = bundle.X_train
    y_train = bundle.y_train
    n = len(X_test)

    print(
        f"\n  --- budget={budget} strategy={strategy}@{size} "
        f"(scope={class_scope}, selection={selection}) ---"
    )

    X_cf = X_test.copy()
    changed_per_point: List[List[int]] = [[] for _ in range(n)]
    flipped_per_point: List[bool] = [False] * n
    steps_per_point: List[int] = [0] * n
    effective_sizes: List[int] = []
    t0 = time.perf_counter()

    for target_cls in np.unique(y_target):
        target_cls = int(target_cls)
        test_idx = np.where(y_target == target_cls)[0]
        if len(test_idx) == 0:
            continue

        ctx_target = target_cls if class_scope == "target" else None
        pool_size = int((y_train == ctx_target).sum()) if ctx_target is not None else int(len(X_train))

        sampler = ConditionalDensitySampler(
            clf=clf,
            reg=reg,
            append_target=True,
            n_permutations=n_permutations,
            temperature=temperature,
            random_state=42 + target_cls,
        )

        if selection == "random":
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
                    sampler,
                    disc_model,
                    X_test[i],
                    target_cls,
                    actionable_idx,
                    SELECTOR,
                    tau=tau,
                    budget=budget,
                    temperature=temperature,
                    stall_eps=stall_eps,
                )
                X_cf[i] = x_cf
                changed_per_point[i] = changed
                flipped_per_point[i] = gi["flipped"]
                steps_per_point[i] = gi["steps"]
        else:
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
                    sampler,
                    disc_model,
                    X_test[i],
                    target_cls,
                    actionable_idx,
                    SELECTOR,
                    tau=tau,
                    budget=budget,
                    temperature=temperature,
                    stall_eps=stall_eps,
                )
                X_cf[i] = x_cf
                changed_per_point[i] = changed
                flipped_per_point[i] = gi["flipped"]
                steps_per_point[i] = gi["steps"]

    runtime_s = time.perf_counter() - t0
    effective_size = int(max(effective_sizes)) if effective_sizes else 0

    info = {
        "bundle": bundle,
        "y_pred": y_pred,
        "y_target": y_target,
        "actionable_idx": actionable_idx,
        "immutable_idx": immutable_idx,
        "disc_model": disc_model,
        "selector": SELECTOR,
        "context_type": class_scope,
        "tau": tau,
        "budget": budget,
        "temperature": temperature,
        "stall_eps": stall_eps,
        "n_permutations": n_permutations,
        "max_context": size,
        "changed_per_point": changed_per_point,
        "flipped_per_point": flipped_per_point,
        "steps_per_point": steps_per_point,
    }
    metrics = evaluate_and_report(dataset_name, X_test, y_test, X_cf, info, write_csv=False)

    return {
        "dataset": dataset_name,
        "selector": SELECTOR,
        "strategy": strategy,
        "size": size,
        "effective_size": effective_size,
        "budget": budget,
        "n_test": int(n),
        "validity": metrics["validity"],
        "failure_rate": metrics["failure_rate"],
        "l0_count_mean": metrics["l0_count_mean"],
        "steps_mean": metrics["steps_mean"],
        "steps_max": metrics["steps_max"],
        "proximity_l2_jaccard": metrics["proximity_l2_jaccard"],
        "lof_scores_cf": metrics["lof_scores_cf"],
        "frac_oob": metrics["frac_oob"],
        "true_actionability": metrics["true_actionability"],
        "runtime_s": round(runtime_s, 2),
    }


def run_dataset_sweep(
    dataset_name: str,
    *,
    tau: float = TAU,
    temperature: float = TEMPERATURE,
    stall_eps: float = 1e-6,
    n_permutations: int = N_PERMUTATIONS,
    max_test: Optional[int] = None,
) -> List[Dict[str, float]]:
    from experiments.zeroshot_cf.checkpoints import get_models
    from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset
    from experiments.zeroshot_cf.discriminator import train_discriminator

    max_test_resolved = _resolve_max_test(dataset_name, max_test)
    config = CONFIGS[dataset_name]

    print(f"\n########## Exp7 budget sweep: {dataset_name.upper()} ##########")
    print(
        f"  selector={SELECTOR}, strategy={config['strategy']}@{config['size']}, "
        f"tau={tau}, temperature={temperature}, stall_eps={stall_eps}, "
        f"n_permutations={n_permutations}, max_test={max_test_resolved}, "
        f"budgets={BUDGETS[dataset_name]}"
    )

    bundle = load_dataset(dataset_name)
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test = bundle.X_test[:max_test_resolved]
    y_test = bundle.y_test[:max_test_resolved]
    actionable_idx, immutable_idx = get_actionable_immutable(dataset_name, bundle)

    disc_model = train_discriminator(X_train, y_train, X_test, y_test, dataset_name)
    y_pred = disc_model.predict(X_test)
    y_target = 1 - y_pred

    print("  Loading TabPFN models ...")
    clf, reg = get_models(n_estimators=N_ESTIMATORS)

    rows: List[Dict[str, float]] = []
    budgets = BUDGETS[dataset_name]
    budget_idx = 0
    while budget_idx < len(budgets):
        budget = budgets[budget_idx]
        row = _run_budget(
            dataset_name,
            budget,
            bundle=bundle,
            disc_model=disc_model,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_target=y_target,
            actionable_idx=actionable_idx,
            immutable_idx=immutable_idx,
            clf=clf,
            reg=reg,
            tau=tau,
            temperature=temperature,
            stall_eps=stall_eps,
            n_permutations=n_permutations,
        )
        rows.append(row)

        steps_max = float(row["steps_max"])
        next_idx = budget_idx + 1
        if next_idx < len(budgets) and np.isfinite(steps_max) and steps_max < budgets[next_idx]:
            print(
                f"  Saturated at budget={budget}: steps_max={steps_max:g} "
                f"< next budget={budgets[next_idx]}; copying identical rows for "
                "remaining larger budgets."
            )
            for copied_budget in budgets[next_idx:]:
                copied = dict(row)
                copied["budget"] = copied_budget
                copied["runtime_s"] = 0.0
                rows.append(copied)
            break

        budget_idx += 1

    csv_path = RESULTS_DIR / f"exp7_budget_{dataset_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_COLUMNS})
    print(f"\n  Wrote {csv_path}")
    return rows


def _fmt(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:
        return "nan"
    if abs(f) >= 1000 or (abs(f) < 1e-3 and f != 0):
        return f"{f:.3e}"
    return f"{f:.4g}"


def _read_rows(dataset_name: str) -> Optional[List[Dict[str, str]]]:
    csv_path = RESULTS_DIR / f"exp7_budget_{dataset_name}.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _trend_verdict(rows: List[Dict[str, str]]) -> str:
    validities = [float(r["validity"]) for r in rows]
    budgets = [r["budget"] for r in rows]
    steps_max = [float(r["steps_max"]) for r in rows]
    best_idx = int(np.nanargmax(validities))
    first = validities[0]
    best = validities[best_idx]
    saturation = ""
    for idx, row_steps_max in enumerate(steps_max[:-1]):
        if row_steps_max < float(budgets[idx + 1]):
            saturation = (
                f" The curve saturates by budget {budgets[idx]} "
                f"(steps_max={_fmt(row_steps_max)} < next budget {budgets[idx + 1]})."
            )
            break
    if best > first + 1e-9:
        return (
            f"Validity improves from {_fmt(first)} to {_fmt(best)} and first reaches "
            f"that maximum at budget {budgets[best_idx]}.{saturation}"
        )
    return (
        f"Validity does not improve over the baseline budget; best observed value is "
        f"{_fmt(best)} at budget {budgets[best_idx]}.{saturation}"
    )


def write_summary() -> None:
    lines = [
        "# Experiment 7: Budget Sweep with Feature Revisiting",
        "",
        "Budget is swept above `|A|` at the Stage-4 recommended context configs: "
        "`random_both@512` for MOONS and `knn_both@256` for HELOC. "
        "`l0_count_mean` counts distinct changed features; `steps_*` counts commits, "
        "so repeated features show up as extra steps without inflating L0.",
        "",
    ]

    any_rows = False
    for dataset_name in ("moons", "heloc"):
        rows = _read_rows(dataset_name)
        if not rows:
            continue
        any_rows = True
        lines += [
            f"## {dataset_name.upper()}",
            "",
            f"Config: `{rows[0]['strategy']}@{rows[0]['size']}` · n_test: {rows[0]['n_test']}",
            "",
            "| budget | validity | failure_rate | l0_count_mean | steps_mean | steps_max | proximity_l2_jaccard | frac_oob | true_actionability | runtime_s |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["budget"],
                        _fmt(row["validity"]),
                        _fmt(row["failure_rate"]),
                        _fmt(row["l0_count_mean"]),
                        _fmt(row["steps_mean"]),
                        _fmt(row["steps_max"]),
                        _fmt(row["proximity_l2_jaccard"]),
                        _fmt(row["frac_oob"]),
                        _fmt(row["true_actionability"]),
                        _fmt(row["runtime_s"]),
                    ]
                )
                + " |"
            )
        lines += ["", f"Verdict: {_trend_verdict(rows)}", ""]

    if not any_rows:
        lines += ["_No Exp7 CSVs found yet._", ""]

    out_path = RESULTS_DIR / "exp7_summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 7: budget sweep")
    parser.add_argument("--dataset", choices=["moons", "heloc", "all"], default="moons")
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--stall-eps", type=float, default=1e-6)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Number of test points (default: moons=100, heloc=30; -1 for full split).",
    )
    args = parser.parse_args()

    datasets = ["moons", "heloc"] if args.dataset == "all" else [args.dataset]
    for dataset_name in datasets:
        run_dataset_sweep(
            dataset_name,
            tau=args.tau,
            temperature=args.temperature,
            stall_eps=args.stall_eps,
            n_permutations=args.n_permutations,
            max_test=args.max_test,
        )

    write_summary()
    print("\nDone.")


if __name__ == "__main__":
    main()
