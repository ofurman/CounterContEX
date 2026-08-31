#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: T201

"""NICE-style nearest-unlike-neighbour baseline for the Exp9 setting.

The baseline uses the same fixed split, target classifier, factual subset,
actionability constraints, and metrics as Exp9. For each factual it finds the
nearest training row predicted as the desired class, then greedily copies one
actionable scalar or one complete one-hot group at a time. Before validity,
the action with the largest target-probability gain is selected. If several
one-step actions are valid, the lowest-LOF action is selected, matching the
valid-candidate preference used by the TabICL experiment.
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
    DEFAULT_PROTOCOL_SEED,
    DEFAULT_VALIDATION_FRACTION,
    TARGET_CLASSIFIER_LABELS,
    aggregate_metrics_path,
    build_common_result_row,
    dataset_result_paths,
    mean_on_valid,
    prepare_benchmark_context,
    write_result_table,
)
from experiments.zeroshot_cf.methods.nice import (
    greedy_nice_counterfactual as greedy_nice_counterfactual,
)
from experiments.zeroshot_cf.methods.nice import (
    nearest_unlike_prototypes as nearest_unlike_prototypes,
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

RESULTS_DIR = Path(__file__).parent / "results" / "local" / "exp11_nice_nun"


def run_dataset(
    dataset_name: str,
    *,
    max_test: int = DEFAULT_MAX_TEST,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    drop_heloc_all_minus9: bool = True,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run one NICE-NUN baseline case in the fixed Exp9 setting."""
    from experiments.zeroshot_cf.methods.nice import NiceMethod
    from experiments.zeroshot_cf.metrics_harness import print_metrics

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
    prepared = NiceMethod().prepare(method_context(context))
    runtime_prepare = time.perf_counter() - prepare_started
    generation_started = time.perf_counter()
    result = prepared.generate(
        generation_request(context, seed=DEFAULT_PROTOCOL_SEED)
    )
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
    prototypes = np.asarray(result.artifacts["method.prototypes"])
    prototype_indices = np.asarray(result.artifacts["method.prototype_indices"])
    prototype_distances = np.asarray(
        result.artifacts["method.prototype_distances"]
    )
    common_metrics = legacy_common_metrics(report)
    print_metrics(common_metrics, prefix=f"{dataset_name}/NICE-NUN")
    valid = result.available[:, 0]
    changed_counts = np.asarray(
        [info["changed_columns"] for info in point_info], dtype=float
    )
    steps = np.asarray([info["steps"] for info in point_info], dtype=float)
    all_l2 = (
        np.linalg.norm(X_cf[valid] - X_test[valid], axis=1)
        if valid.any()
        else np.empty(0, dtype=float)
    )
    row: dict[str, Any] = build_common_result_row(
        context,
        method="nice_nun_greedy_lof",
        cf_per_factual=1,
        extra_fields={
        "prototype_pool_labels": TARGET_CLASSIFIER_LABELS,
        "prototype_metric": "euclidean",
        "valid_candidate_selection": "lof",
        "categorical_actions": "atomic_one_hot_groups",
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
        "l0_count_mean": mean_on_valid(changed_counts, valid),
        "steps_mean": mean_on_valid(steps, valid),
        "prototype_distance_mean": float(prototype_distances.mean()),
        "factual_oob_fraction": float(
            (((X_test < 0.0) | (X_test > 1.0)).any(axis=1)).mean()
        ),
        "cf_oob_fraction": float((((X_cf < 0.0) | (X_cf > 1.0)).any(axis=1)).mean()),
        },
    )
    row["runtime_total_s"] = round(time.perf_counter() - total_started, 3)

    point_rows = [
        {
            "point": index,
            "factual_label": int(y_test[index]),
            "factual_prediction": int(context.y_pred[index]),
            "target": int(context.y_target[index]),
            "cf_prediction": int(info["prediction"]),
            "valid": bool(info["valid"]),
            "target_probability": float(info["target_probability"]),
            "changed_columns": int(info["changed_columns"]),
            "steps": int(info["steps"]),
            "prototype_index": int(prototype_indices[index]),
            "prototype_distance": float(prototype_distances[index]),
        }
        for index, info in enumerate(point_info)
    ]
    paths = dataset_result_paths(results_dir, "exp11_nice_nun", dataset_name)
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
            "prototypes": prototypes,
            "prototype_indices": prototype_indices,
        },
        total_started=total_started,
    )
    return row


def main() -> None:
    """Run one dataset or all four fixed-protocol datasets locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
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
    rows = [
        run_dataset(
            dataset,
            max_test=args.max_test,
            validation_fraction=args.validation_fraction,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
            results_dir=args.results_dir,
        )
        for dataset in datasets
    ]
    if args.dataset == "all":
        write_result_table(
            aggregate_metrics_path(args.results_dir, "exp11_nice_nun"), rows
        )


if __name__ == "__main__":
    main()
