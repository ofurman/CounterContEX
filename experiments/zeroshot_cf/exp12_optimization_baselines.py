#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: T201

"""Model-agnostic optimization baselines for the fixed Exp9 setting.

The implementations adapt two established counterfactual strategies to the
repository's mixed-data action space:

* ``wachter`` minimizes a Wachter-style distance-plus-classification-loss
  objective through black-box coordinate search.
* ``growing_spheres`` samples expanding neighbourhoods until it finds target
  predictions, then removes unnecessary actions and contracts scalar edits.

Both methods use only ``predict`` and ``predict_proba`` from the target model.
One-hot groups are always changed atomically and immutable features are never
offered to either optimizer. Dataset splitting, factual selection, target
labels, and metrics are identical to Exp9 and Exp11.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.benchmark_protocol import (
    DATASETS,
    DEFAULT_MAX_TEST,
    DEFAULT_VALIDATION_FRACTION,
    aggregate_metrics_path,
    build_common_result_row,
    dataset_result_paths,
    mean_on_valid,
    prepare_benchmark_context,
    write_result_table,
)
from experiments.zeroshot_cf.methods.optimization import (
    WACHTER_QUANTILES as WACHTER_QUANTILES,
)
from experiments.zeroshot_cf.methods.optimization import (
    _action_sparsity,
)
from experiments.zeroshot_cf.methods.optimization import (
    growing_spheres_counterfactual as growing_spheres_counterfactual,
)
from experiments.zeroshot_cf.methods.optimization import (
    wachter_coordinate_counterfactual as wachter_coordinate_counterfactual,
)
from experiments.zeroshot_cf.retained_config import TAU
from experiments.zeroshot_cf.runner_compat import (
    evaluate_result,
    generation_request,
    legacy_candidate_matrix,
    legacy_common_metrics,
    method_context,
    point_diagnostics,
    write_legacy_outputs_with_timing,
)

METHODS = ("wachter", "growing_spheres")
RESULTS_DIR = Path(__file__).parent / "results" / "local" / "exp12_optimization"


def run_dataset(  # noqa: PLR0913
    dataset_name: str,
    method: str,
    *,
    max_test: int = DEFAULT_MAX_TEST,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    drop_heloc_all_minus9: bool = True,
    random_state: int = 42,
    sphere_candidates: int = 512,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run one optimization baseline under the fixed Exp9 protocol."""
    from experiments.zeroshot_cf.methods.optimization import (
        GrowingSpheresConfig,
        GrowingSpheresMethod,
        WachterMethod,
    )
    from experiments.zeroshot_cf.metrics_harness import print_metrics

    if method not in METHODS:
        raise ValueError(f"Unsupported Exp12 method: {method!r}")

    total_started = time.perf_counter()
    context = prepare_benchmark_context(
        dataset_name,
        max_test=max_test,
        validation_fraction=validation_fraction,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
    )
    X_test = context.X_test
    y_test = context.y_test
    prepare_started = time.perf_counter()
    configured_method = (
        WachterMethod()
        if method == "wachter"
        else GrowingSpheresMethod(
            GrowingSpheresConfig(n_candidates=sphere_candidates)
        )
    )
    prepared = configured_method.prepare(method_context(context))
    runtime_prepare = time.perf_counter() - prepare_started
    generation_started = time.perf_counter()
    result = prepared.generate(generation_request(context, seed=random_state))
    runtime_generation = time.perf_counter() - generation_started
    evaluation_started = time.perf_counter()
    report = evaluate_result(
        context,
        result,
        probability_threshold=TAU,
    )
    runtime_evaluation = time.perf_counter() - evaluation_started
    X_cf = legacy_candidate_matrix(result)
    point_info = point_diagnostics(result)
    common_metrics = legacy_common_metrics(report)
    print_metrics(common_metrics, prefix=f"{dataset_name}/{method}")
    valid = result.available[:, 0]
    changed_columns = (X_cf != X_test).sum(axis=1)
    action_counts = np.asarray(
        [
            _action_sparsity(
                X_cf[i],
                X_test[i],
                list(context.scalar_actionable),
                list(context.grouped_actionable),
            )[0]
            for i in range(len(X_test))
        ],
        dtype=float,
    )
    all_l2 = (
        np.linalg.norm(X_cf[valid] - X_test[valid], axis=1)
        if valid.any()
        else np.empty(0, dtype=float)
    )
    evaluations = np.asarray([info["evaluations"] for info in point_info], dtype=float)
    row: dict[str, Any] = build_common_result_row(
        context,
        method=method,
        cf_per_factual=1,
        extra_fields={
            "model_access": "predict_and_predict_proba",
            "categorical_actions": "atomic_one_hot_groups",
            "posthoc_action_pruning": True,
            "posthoc_scalar_contraction": True,
            "sphere_candidates": (
                sphere_candidates if method == "growing_spheres" else 0
            ),
            "prepare_s": runtime_prepare,
            "generate_s": runtime_generation,
            "evaluate_s": runtime_evaluation,
            "runtime_generation_s": round(runtime_generation, 3),
            **common_metrics,
            "sparsity_exact": float((X_test != X_cf).mean()),
            "true_actionability": common_metrics["actionability"],
            "proximity_all_features_euclidean": (
                float(all_l2.mean()) if len(all_l2) else float("nan")
            ),
            "failure_rate": float((~valid).mean()),
            "l0_count_mean": mean_on_valid(changed_columns, valid),
            "action_count_mean": mean_on_valid(action_counts, valid),
            "model_evaluations_mean": float(evaluations.mean()),
            "factual_oob_fraction": float(
                (((X_test < 0.0) | (X_test > 1.0)).any(axis=1)).mean()
            ),
            "cf_oob_fraction": float(
                (((X_cf < 0.0) | (X_cf > 1.0)).any(axis=1)).mean()
            ),
        },
    )
    row["runtime_total_s"] = round(time.perf_counter() - total_started, 3)

    point_rows = [
        {
            "point": index,
            "factual_label": int(y_test[index]),
            "factual_prediction": int(context.y_pred[index]),
            "target": int(context.y_target[index]),
            "cf_prediction": int(
                context.disc_model.predict(X_cf[index : index + 1])[0]
            ),
            "valid": bool(info["valid"]),
            "target_probability": float(info["target_probability"]),
            "changed_columns": int(changed_columns[index]),
            "action_count": int(action_counts[index]),
            "model_evaluations": int(info["evaluations"]),
        }
        for index, info in enumerate(point_info)
    ]
    paths = dataset_result_paths(results_dir, f"exp12_{method}", dataset_name)
    write_legacy_outputs_with_timing(
        paths,
        row,
        point_rows,
        arrays={
            "X_test": X_test,
            "y_test": y_test,
            "X_cf": X_cf,
            "y_pred": context.y_pred,
            "y_target": context.y_target,
            "y_cf_pred": context.disc_model.predict(X_cf),
        },
        total_started=total_started,
    )
    return row


def main() -> None:
    """Run selected optimization baselines locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--sphere-candidates", type=int, default=512)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--drop-heloc-all-minus9",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    methods = METHODS if args.method == "all" else (args.method,)
    rows = [
        run_dataset(
            dataset,
            method,
            max_test=args.max_test,
            validation_fraction=args.validation_fraction,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
            sphere_candidates=args.sphere_candidates,
            results_dir=args.results_dir,
        )
        for dataset in datasets
        for method in methods
    ]
    write_result_table(
        aggregate_metrics_path(args.results_dir, "exp12_optimization"), rows
    )


if __name__ == "__main__":
    main()
