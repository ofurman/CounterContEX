"""Experiment 9: low-cardinality integer routing audit for greedy CFs.

HELOC contains scaled integer/ordinal columns that TabPFN's unsupervised wrapper
auto-routes through the classifier head. This runner compares the Stage-4
recommended greedy configuration with current routing versus forcing those
columns through the numerical bar-distribution/regressor path.
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
    parse_force_numeric_cols,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SELECTOR = "prob_ascent"
CONTEXT_SIZE = 256
CONTEXT_STRATEGY = "knn_both"
DEFAULT_MAX_TEST = 30
DEFAULT_BUDGET = 5

CSV_COLUMNS = [
    "dataset",
    "cell",
    "selector",
    "strategy",
    "size",
    "effective_size",
    "budget",
    "force_numeric_cols",
    "force_numeric_names",
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


def identify_routing(bundle, clf, reg, *, max_context: int = CONTEXT_SIZE) -> Dict[str, List[int]]:
    """Return original feature columns routed to classifier vs regressor."""
    from experiments.zeroshot_cf.sampler import ConditionalDensitySampler

    sampler = ConditionalDensitySampler(
        clf=clf,
        reg=reg,
        append_target=True,
        n_permutations=1,
        temperature=TEMPERATURE,
        random_state=42,
        categorical_features_indices=bundle.categorical_features_indices,
    )
    sampler.set_context(
        bundle.X_train,
        y_context=bundle.y_train,
        target_class=None,
        max_context=max_context,
        selection="random",
    )

    classifier_cols: List[int] = []
    regressor_cols: List[int] = []
    for j in range(bundle.X_train.shape[1]):
        if sampler.model.use_classifier_(j, sampler.model.X_[:, j]):
            classifier_cols.append(j)
        else:
            regressor_cols.append(j)
    return {"classifier": classifier_cols, "regressor": regressor_cols}


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


def _run_cell(
    *,
    dataset_name: str,
    cell: str,
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
    force_numeric_cols: List[int],
    budget: int,
    tau: float,
    temperature: float,
    stall_eps: float,
    n_permutations: int,
) -> Dict[str, object]:
    from experiments.zeroshot_cf.greedy import greedy_counterfactual
    from experiments.zeroshot_cf.sampler import ConditionalDensitySampler

    X_train = bundle.X_train
    y_train = bundle.y_train
    n = len(X_test)

    print(
        f"\n--- Exp9 {cell}: {CONTEXT_STRATEGY}@{CONTEXT_SIZE}, "
        f"force_numeric={force_numeric_cols or 'none'} ---"
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

        for i in test_idx:
            sampler = ConditionalDensitySampler(
                clf=clf,
                reg=reg,
                append_target=True,
                n_permutations=n_permutations,
                temperature=temperature,
                random_state=42 + target_cls,
                categorical_features_indices=bundle.categorical_features_indices,
                force_numeric_cols=force_numeric_cols,
            )
            sampler.set_context(
                X_train,
                y_context=y_train,
                target_class=None,
                max_context=CONTEXT_SIZE,
                selection="knn",
                query=X_test[i],
            )
            effective_sizes.append(min(CONTEXT_SIZE, len(X_train)))
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
    info = {
        "bundle": bundle,
        "y_pred": y_pred,
        "y_target": y_target,
        "actionable_idx": actionable_idx,
        "immutable_idx": immutable_idx,
        "disc_model": disc_model,
        "selector": SELECTOR,
        "context_type": "both",
        "tau": tau,
        "budget": budget,
        "temperature": temperature,
        "stall_eps": stall_eps,
        "n_permutations": n_permutations,
        "max_context": CONTEXT_SIZE,
        "force_numeric_cols": force_numeric_cols,
        "changed_per_point": changed_per_point,
        "flipped_per_point": flipped_per_point,
        "steps_per_point": steps_per_point,
    }
    metrics = evaluate_and_report(dataset_name, X_test, y_test, X_cf, info, write_csv=False)

    force_names = [bundle.feature_names[j] for j in force_numeric_cols]
    return {
        "dataset": dataset_name,
        "cell": cell,
        "selector": SELECTOR,
        "strategy": CONTEXT_STRATEGY,
        "size": CONTEXT_SIZE,
        "effective_size": int(max(effective_sizes)) if effective_sizes else 0,
        "budget": budget,
        "force_numeric_cols": ",".join(str(j) for j in force_numeric_cols) or "none",
        "force_numeric_names": ",".join(force_names) or "none",
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


def _write_summary(
    dataset_name: str,
    routing: Dict[str, List[int]],
    rows: List[Dict[str, object]],
    bundle,
) -> None:
    classifier_cols = routing["classifier"]
    regressor_cols = routing["regressor"]
    classifier_names = [bundle.feature_names[j] for j in classifier_cols]

    baseline = rows[0]
    override = rows[1]
    deltas = {
        key: float(override[key]) - float(baseline[key])
        for key in (
            "validity",
            "proximity_l2_jaccard",
            "frac_oob",
            "l0_count_mean",
        )
    }

    verdict = (
        "The override improves proximity/validity enough to consider forcing "
        "ordered treatment for these HELOC columns."
        if deltas["validity"] >= 0 and deltas["proximity_l2_jaccard"] <= 0
        else "The override is not a clear win; keep auto-routing as the default."
    )

    lines = [
        "# Experiment 9: HELOC Routing Override Audit",
        "",
        f"Config: `{SELECTOR}` + `{CONTEXT_STRATEGY}@{CONTEXT_SIZE}`; "
        f"dataset: `{dataset_name}`; n_test: {baseline['n_test']}; "
        f"budget: {baseline['budget']}.",
        "",
        "## Routing Inventory",
        "",
        f"Classifier-routed original columns: {len(classifier_cols)} / {len(bundle.feature_names)}",
        "",
        "| idx | feature | unique_train_values |",
        "|---|---|---|",
    ]
    for j in classifier_cols:
        lines.append(
            f"| {j} | `{bundle.feature_names[j]}` | "
            f"{len(np.unique(bundle.X_train[:, j]))} |"
        )
    lines += [
        "",
        f"Regressor-routed original columns: {len(regressor_cols)} / {len(bundle.feature_names)}.",
        "",
        "## Metrics",
        "",
        "| cell | validity | failure_rate | proximity_l2_jaccard | frac_oob | l0_count_mean | steps_mean | runtime_s | force_numeric |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell"]),
                    _fmt(row["validity"]),
                    _fmt(row["failure_rate"]),
                    _fmt(row["proximity_l2_jaccard"]),
                    _fmt(row["frac_oob"]),
                    _fmt(row["l0_count_mean"]),
                    _fmt(row["steps_mean"]),
                    _fmt(row["runtime_s"]),
                    str(row["force_numeric_cols"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Deltas (override - baseline)",
        "",
        f"- validity: {_fmt(deltas['validity'])}",
        f"- proximity_l2_jaccard: {_fmt(deltas['proximity_l2_jaccard'])}",
        f"- frac_oob: {_fmt(deltas['frac_oob'])}",
        f"- l0_count_mean: {_fmt(deltas['l0_count_mean'])}",
        "",
        f"Verdict: {verdict}",
        "",
        "Forced columns:",
        ", ".join(f"`{j}:{name}`" for j, name in zip(classifier_cols, classifier_names))
        or "none",
        "",
    ]
    out_path = RESULTS_DIR / "exp9_routing_summary.md"
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")


def run(
    dataset_name: str = "heloc",
    *,
    max_test: Optional[int] = None,
    tau: float = TAU,
    temperature: float = TEMPERATURE,
    stall_eps: float = 1e-6,
    n_permutations: int = N_PERMUTATIONS,
    force_numeric_cols: str = "auto",
    budget: int = DEFAULT_BUDGET,
) -> List[Dict[str, object]]:
    from experiments.zeroshot_cf.checkpoints import get_models
    from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset
    from experiments.zeroshot_cf.discriminator import train_discriminator

    if dataset_name != "heloc":
        raise ValueError("Exp9 is defined for HELOC; MOONS is a null-control by inspection.")

    max_test_resolved = DEFAULT_MAX_TEST if max_test is None else max_test
    if max_test_resolved < 0:
        max_test_resolved = None

    print(f"\n########## Exp9 routing audit: {dataset_name.upper()} ##########")
    print(
        f"Audit budget={budget}. Full budget=|A| is intentionally exposed via "
        "--budget, but the override cell can hit the O(|A|^2) worst case."
    )
    bundle = load_dataset(dataset_name)
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test = bundle.X_test[:max_test_resolved]
    y_test = bundle.y_test[:max_test_resolved]
    actionable_idx, immutable_idx = get_actionable_immutable(dataset_name, bundle)

    disc_model = train_discriminator(X_train, y_train, X_test, y_test, dataset_name)
    y_pred = disc_model.predict(X_test)
    y_target = 1 - y_pred

    print("Loading TabPFN models ...")
    clf, reg = get_models(n_estimators=N_ESTIMATORS)

    routing = identify_routing(bundle, clf, reg)
    print(
        "Routing inventory: "
        f"{len(routing['classifier'])} classifier, {len(routing['regressor'])} regressor"
    )
    for j in routing["classifier"]:
        print(f"  classifier {j}: {bundle.feature_names[j]}")

    if force_numeric_cols == "auto":
        override_cols = routing["classifier"]
    else:
        override_cols = parse_force_numeric_cols(force_numeric_cols, bundle)

    rows = [
        _run_cell(
            dataset_name=dataset_name,
            cell="baseline",
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
            force_numeric_cols=[],
            budget=budget,
            tau=tau,
            temperature=temperature,
            stall_eps=stall_eps,
            n_permutations=n_permutations,
        ),
        _run_cell(
            dataset_name=dataset_name,
            cell="override",
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
            force_numeric_cols=override_cols,
            budget=budget,
            tau=tau,
            temperature=temperature,
            stall_eps=stall_eps,
            n_permutations=n_permutations,
        ),
    ]

    csv_path = RESULTS_DIR / f"exp9_routing_{dataset_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_COLUMNS})
    print(f"Wrote {csv_path}")

    _write_summary(dataset_name, routing, rows, bundle)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 9: routing override audit")
    parser.add_argument("--dataset", default="heloc", help="Dataset name (heloc).")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--stall-eps", type=float, default=1e-6)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help=f"Greedy commit budget for each audit cell (default: {DEFAULT_BUDGET}).",
    )
    parser.add_argument(
        "--force-numeric-cols",
        default="auto",
        help="Override cell columns: auto=baseline classifier-routed columns, "
             "none, all, or comma-separated indices/names.",
    )
    args = parser.parse_args()

    run(
        args.dataset,
        max_test=args.max_test,
        tau=args.tau,
        temperature=args.temperature,
        stall_eps=args.stall_eps,
        n_permutations=args.n_permutations,
        force_numeric_cols=args.force_numeric_cols,
        budget=args.budget,
    )


if __name__ == "__main__":
    main()
