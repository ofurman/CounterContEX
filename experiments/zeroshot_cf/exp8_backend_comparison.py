#  Copyright (c) Prior Labs GmbH 2026.

"""TabPFNv3 versus TabICLv2 on two fixed comparison datasets.

The comparison runs MOONS and HELOC with the same counterfactual algorithm and
the already-selected Athena context: ``prob_ascent`` with a 512-row both-class
kNN context. It intentionally contains no context sweep.

TabPFN uses the existing sequential candidate evaluation. TabICL uses the
candidate-expanded fast path. Fast unit tests cover the adapter contract and
``exp8_tabicl_diagnostics`` checks real-model equivalence on Athena. Both
backends use discriminator-predicted context labels, matching the final Athena
Exp7 winner.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.zeroshot_cf.exp4_greedy_cf import TAU, evaluate_and_report
from experiments.zeroshot_cf.exp6_context_ablation import _run_cell
from experiments.zeroshot_cf.exp8_tabicl_cf import (
    ATHENA_CONTEXT_SIZE,
    ATHENA_CONTEXT_STRATEGY,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_TEMPERATURE,
    generate_tabicl_counterfactuals,
)

DATASETS = ("moons", "heloc")
RESULTS_DIR = Path(__file__).parent / "results"
TABPFN_N_PERMUTATIONS = 3


def _load_shared_data(
    dataset_name: str,
    max_test: int | None,
    *,
    drop_heloc_all_minus9: bool = False,
):
    from experiments.zeroshot_cf.data import (
        get_actionable_immutable,
        load_dataset,
    )
    from experiments.zeroshot_cf.discriminator import train_discriminator

    bundle = load_dataset(
        dataset_name,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
    )
    limit = None if max_test is not None and max_test < 0 else max_test
    if limit is None and max_test is None:
        limit = 100 if dataset_name == "moons" else 50
    X_test, y_test = bundle.X_test[:limit], bundle.y_test[:limit]
    actionable_idx, immutable_idx = get_actionable_immutable(dataset_name, bundle)
    discriminator_cache_tag = (
        f"{dataset_name}_drop_all_minus9"
        if bundle.preprocessing_variant == "drop_heloc_all_minus9"
        else dataset_name
    )
    disc_model = train_discriminator(
        bundle.X_train,
        bundle.y_train,
        X_test,
        y_test,
        discriminator_cache_tag,
    )
    y_pred = disc_model.predict(X_test)
    return {
        "bundle": bundle,
        "disc_model": disc_model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_target": 1 - y_pred,
        "actionable_idx": actionable_idx,
        "immutable_idx": immutable_idx,
    }


def run_tabpfn_v3(
    dataset_name: str,
    *,
    max_test: int | None,
    tau: float,
    temperature: float,
    n_estimators: int,
    tabpfn_cache_dir: Path | None,
    drop_heloc_all_minus9: bool = False,
) -> dict[str, Any]:
    """Run the existing TabPFNv3 method at only ``knn_both@512``."""
    from experiments.zeroshot_cf.checkpoints import get_v3_models

    shared = _load_shared_data(
        dataset_name,
        max_test,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
    )
    clf, reg = get_v3_models(
        n_estimators=n_estimators,
        cache_dir=tabpfn_cache_dir,
    )
    row = _run_cell(
        dataset_name,
        "prob_ascent",
        ATHENA_CONTEXT_SIZE,
        ATHENA_CONTEXT_STRATEGY,
        bundle=shared["bundle"],
        disc_model=shared["disc_model"],
        X_test=shared["X_test"],
        y_test=shared["y_test"],
        y_pred=shared["y_pred"],
        y_target=shared["y_target"],
        actionable_idx=shared["actionable_idx"],
        immutable_idx=shared["immutable_idx"],
        clf=clf,
        reg=reg,
        tau=tau,
        temperature=temperature,
        n_permutations=TABPFN_N_PERMUTATIONS,
        context_y=shared["disc_model"].predict(shared["bundle"].X_train),
        project_to_domain=True,
        retain_best=True,
    )
    return {
        "dataset": dataset_name,
        "backend": "tabpfn_v3",
        "candidate_mode": "sequential",
        "context_labels": "disc",
        "n_estimators": n_estimators,
        "temperature": temperature,
        "point_estimate": "mode",
        "project_to_domain": True,
        "retain_best": True,
        "preprocessing_variant": shared["bundle"].preprocessing_variant,
        "n_dropped_rows": shared["bundle"].n_dropped_rows,
        **row,
    }


def run_tabicl_v2(
    dataset_name: str,
    *,
    max_test: int | None,
    tau: float,
    temperature: float,
    n_estimators: int,
    tabicl_cache_dir: Path | None,
    candidate_quantiles: tuple[float, ...] | None = None,
    confidence_quantiles: tuple[float, ...] | None = None,
    lof_first: bool = False,
    probability_slack: float = 0.02,
    max_rounds: int = 1,
    drop_heloc_all_minus9: bool = False,
) -> dict[str, Any]:
    """Run TabICLv2 with candidate expansion at ``knn_both@512``."""
    X_test, y_test, X_cf, info = generate_tabicl_counterfactuals(
        dataset_name,
        tau=tau,
        temperature=temperature,
        n_estimators=n_estimators,
        max_test=max_test,
        context_labels="disc",
        candidate_mode="batched",
        candidate_quantiles=candidate_quantiles,
        confidence_quantiles=confidence_quantiles,
        lof_first=lof_first,
        probability_slack=probability_slack,
        max_rounds=max_rounds,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
        cache_dir=tabicl_cache_dir,
    )
    metrics = evaluate_and_report(
        dataset_name,
        X_test,
        y_test,
        X_cf,
        info,
        write_csv=False,
    )
    diagnostics = None
    if info["lof_per_point"] is not None:
        diagnostics = {
            "dataset": dataset_name,
            "preprocessing_variant": info["preprocessing_variant"],
            "n_dropped_rows": info["n_dropped_rows"],
            "lof_per_point": info["lof_per_point"].tolist(),
            "y_pred": info["y_pred"].tolist(),
            "y_target": info["y_target"].tolist(),
            "target_probability_per_point": info[
                "target_probability_per_point"
            ].tolist(),
            "changed_per_point": info["changed_per_point"],
            "flipped_per_point": info["flipped_per_point"],
            "steps_per_point": info["steps_per_point"],
            "history_per_point": info["history_per_point"],
            "attempt_history_per_point": info["attempt_history_per_point"],
            "selection_history_per_point": info["selection_history_per_point"],
            "confidence_grid_per_point": info["confidence_grid_per_point"],
            "X_test": X_test.tolist(),
            "X_cf": X_cf.tolist(),
        }
    return {
        "dataset": dataset_name,
        "backend": "tabicl_v2",
        "selector": "prob_ascent",
        "size": ATHENA_CONTEXT_SIZE,
        "effective_size": min(ATHENA_CONTEXT_SIZE, len(info["bundle"].X_train)),
        "strategy": ATHENA_CONTEXT_STRATEGY,
        "class_scope": "both",
        "selection": "knn",
        "candidate_mode": "batched",
        "context_update": info["context_update"],
        "point_estimate": info["point_estimate"],
        "project_to_domain": info["project_to_domain"],
        "retain_best": info["retain_best"],
        "candidate_quantiles": info["candidate_quantiles"],
        "confidence_quantiles": info["confidence_quantiles"],
        "lof_first": info["lof_first"],
        "probability_slack": info["probability_slack"],
        "max_rounds": info["max_rounds"],
        "preprocessing_variant": info["preprocessing_variant"],
        "n_dropped_rows": info["n_dropped_rows"],
        "context_labels": "disc",
        "n_estimators": n_estimators,
        "temperature": temperature,
        "n_test": len(X_test),
        "runtime_s": round(float(info["runtime_s"]), 2),
        "_diagnostics": diagnostics,
        **metrics,
    }


def _write_row(row: dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    diagnostics = row.pop("_diagnostics", None)
    output = results_dir / (
        f"exp8_compare_{row['dataset']}_{row['backend']}_metrics.csv"
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote {output}")
    if diagnostics is not None:
        diagnostics_output = output.with_name(
            output.name.replace("_metrics.csv", "_diagnostics.json")
        )
        with diagnostics_output.open("w") as handle:
            json.dump(diagnostics, handle, indent=2)
        print(f"Wrote {diagnostics_output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TabPFNv3 and TabICLv2 at knn_both@512"
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="moons",
    )
    parser.add_argument(
        "--backend",
        choices=["tabpfn", "tabicl", "all"],
        default="all",
    )
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--tabpfn-cache-dir", type=Path, default=None)
    parser.add_argument("--tabicl-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--tabicl-quantiles",
        type=float,
        nargs="+",
        default=None,
        metavar="Q",
        help="Deterministic TabICL candidate quantiles; applies only to TabICL.",
    )
    parser.add_argument(
        "--tabicl-confidence-quantiles",
        type=float,
        nargs="+",
        default=None,
        metavar="Q",
        help=(
            "Empirical target-confidence quantile levels used as TabICL "
            "conditioning candidates; applies only to TabICL."
        ),
    )
    parser.add_argument(
        "--tabicl-lof-first",
        action="store_true",
        help="Use classifier validity as a gate and minimum LOF among valid candidates.",
    )
    parser.add_argument(
        "--tabicl-probability-slack",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--tabicl-max-rounds",
        type=int,
        default=1,
        help="TabICL greedy coordinate passes (default: 1).",
    )
    parser.add_argument(
        "--drop-heloc-all-minus9",
        action="store_true",
        help=(
            "Before splitting HELOC, remove records whose predictors are all "
            "the -9 no-bureau-record sentinel."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for this backend/dataset CSV (default: experiment results).",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    backends = ("tabpfn", "tabicl") if args.backend == "all" else (args.backend,)
    for dataset_name in datasets:
        for backend in backends:
            common = {
                "max_test": args.max_test,
                "tau": args.tau,
                "temperature": args.temperature,
                "n_estimators": args.n_estimators,
                "drop_heloc_all_minus9": args.drop_heloc_all_minus9,
            }
            if backend == "tabpfn":
                row = run_tabpfn_v3(
                    dataset_name,
                    tabpfn_cache_dir=args.tabpfn_cache_dir,
                    **common,
                )
            else:
                row = run_tabicl_v2(
                    dataset_name,
                    tabicl_cache_dir=args.tabicl_cache_dir,
                    candidate_quantiles=(
                        None
                        if args.tabicl_quantiles is None
                        else tuple(args.tabicl_quantiles)
                    ),
                    confidence_quantiles=(
                        None
                        if args.tabicl_confidence_quantiles is None
                        else tuple(args.tabicl_confidence_quantiles)
                    ),
                    lof_first=args.tabicl_lof_first,
                    probability_slack=args.tabicl_probability_slack,
                    max_rounds=args.tabicl_max_rounds,
                    **common,
                )
            _write_row(row, args.results_dir)


if __name__ == "__main__":
    main()
