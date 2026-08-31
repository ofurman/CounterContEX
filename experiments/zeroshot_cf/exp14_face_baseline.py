#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: T201

"""Density-weighted FACE-kNN baseline under the fixed Exp9 protocol.

FACE represents observed data as a neighbourhood graph and searches for a
short, high-density path to a target-class endpoint. This implementation builds
one reusable kNN graph per dataset in actionable-feature space. During search,
immutable values are copied from the factual into every candidate, and complete
one-hot groups are transferred atomically from observed training rows.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.baseline_common import build_action_units
from experiments.zeroshot_cf.benchmark_protocol import (
    DATASETS,
    DEFAULT_MAX_TEST,
    DEFAULT_PROTOCOL_SEED,
    DEFAULT_VALIDATION_FRACTION,
    aggregate_metrics_path,
    build_common_result_row,
    dataset_result_paths,
    mean_on_valid,
    prepare_benchmark_context,
    write_result_table,
)
from experiments.zeroshot_cf.methods.face import FaceGraph as FaceGraph
from experiments.zeroshot_cf.methods.face import (
    _expanded_actionable_columns as _expanded_actionable_columns,
)
from experiments.zeroshot_cf.methods.face import (
    build_face_knn_graph as build_face_knn_graph,
)
from experiments.zeroshot_cf.methods.face import (
    face_counterfactual as face_counterfactual,
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

RESULTS_DIR = Path(__file__).parent / "results" / "local" / "exp14_face_knn"


def run_dataset(
    dataset_name: str,
    *,
    max_test: int = DEFAULT_MAX_TEST,
    n_neighbors: int = 100,
    density_power: float = 1.0,
    tau: float = TAU,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    drop_heloc_all_minus9: bool = True,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run FACE-kNN for one dataset in the fixed Exp9 setting."""
    from experiments.zeroshot_cf.methods.face import FaceConfig, FaceMethod
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
    action_units = build_action_units(
        list(context.scalar_actionable),
        list(context.grouped_actionable),
    )

    graph_started = time.perf_counter()
    prepared = FaceMethod(
        FaceConfig(
            n_neighbors=n_neighbors,
            density_power=density_power,
            tau=tau,
        )
    ).prepare(method_context(context))
    runtime_graph = time.perf_counter() - graph_started
    search_started = time.perf_counter()
    result = prepared.generate(
        generation_request(context, seed=DEFAULT_PROTOCOL_SEED)
    )
    runtime_search = time.perf_counter() - search_started
    evaluation_started = time.perf_counter()
    report = evaluate_result(
        context,
        result,
        probability_threshold=tau,
    )
    runtime_evaluation = time.perf_counter() - evaluation_started
    X_cf = legacy_candidate_matrix(result)
    point_info = point_diagnostics(result)
    common_metrics = legacy_common_metrics(report)
    print_metrics(common_metrics, prefix=f"{dataset_name}/FACE-kNN")
    valid = result.available[:, 0]
    changed_columns = (X_cf != X_test).sum(axis=1)
    changed_actions = np.asarray(
        [
            sum(
                not np.array_equal(
                    X_cf[index, list(unit.columns)],
                    X_test[index, list(unit.columns)],
                )
                for unit in action_units
            )
            for index in range(len(X_test))
        ],
        dtype=float,
    )
    l2 = (
        np.linalg.norm(X_cf[valid] - X_test[valid], axis=1)
        if valid.any()
        else np.empty(0, dtype=float)
    )
    path_costs = np.asarray([info["path_cost"] for info in point_info], dtype=float)
    path_steps = np.asarray([info["path_steps"] for info in point_info], dtype=float)
    expanded = np.asarray([info["expanded_nodes"] for info in point_info], dtype=float)
    row: dict[str, Any] = build_common_result_row(
        context,
        method="face_knn_density_weighted",
        cf_per_factual=1,
        extra_fields={
        "graph": "symmetric_knn_actionable_space",
        "n_neighbors": prepared.graph.n_neighbors,
        "edge_weight": "euclidean_times_relative_knn_radius",
        "density_power": density_power,
        "tau": tau,
        "endpoint": "observed_actionable_projection",
        "categorical_actions": "atomic_one_hot_groups",
        "prepare_s": runtime_graph,
        "generate_s": runtime_search,
        "evaluate_s": runtime_evaluation,
        "runtime_graph_build_s": round(runtime_graph, 3),
        "runtime_search_s": round(runtime_search, 3),
        "runtime_generation_s": round(runtime_graph + runtime_search, 3),
        **common_metrics,
        "sparsity_exact": float((X_test != X_cf).mean()),
        "true_actionability": common_metrics["actionability"],
        "proximity_all_features_euclidean": (
            float(l2.mean()) if len(l2) else float("nan")
        ),
        "failure_rate": float((~valid).mean()),
        "l0_count_mean": mean_on_valid(changed_columns, valid),
        "action_count_mean": mean_on_valid(changed_actions, valid),
        "path_cost_mean": float(np.nanmean(path_costs)),
        "path_steps_mean": mean_on_valid(path_steps, valid),
        "expanded_nodes_mean": float(expanded.mean()),
        "factual_oob_fraction": float(
            (((X_test < 0.0) | (X_test > 1.0)).any(axis=1)).mean()
        ),
        "cf_oob_fraction": float((((X_cf < 0.0) | (X_cf > 1.0)).any(axis=1)).mean()),
        },
    )
    row["runtime_total_s"] = round(time.perf_counter() - total_started, 3)

    paths = dataset_result_paths(results_dir, "exp14_face_knn", dataset_name)
    write_legacy_outputs_with_timing(
        paths,
        row,
        [
            {
                "point": index,
                "factual_label": int(y_test[index]),
                "factual_prediction": int(context.y_pred[index]),
                "target": int(context.y_target[index]),
                "cf_prediction": int(info["prediction"]),
                "valid": bool(info["valid"]),
                "target_probability": float(info["target_probability"]),
                "endpoint_train_index": int(info["endpoint_index"]),
                "changed_columns": int(changed_columns[index]),
                "changed_actions": int(changed_actions[index]),
                "path_cost": float(info["path_cost"]),
                "path_steps": int(info["path_steps"]),
                "expanded_nodes": int(info["expanded_nodes"]),
                "runtime_s": float(info["runtime_s"]),
            }
            for index, info in enumerate(point_info)
        ],
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
    """Run FACE-kNN locally on one or all fixed-protocol datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--n-neighbors", type=int, default=100)
    parser.add_argument("--density-power", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=TAU)
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
            n_neighbors=args.n_neighbors,
            density_power=args.density_power,
            tau=args.tau,
            validation_fraction=args.validation_fraction,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
            results_dir=args.results_dir,
        )
        for dataset in datasets
    ]
    if len(rows) > 1:
        write_result_table(
            aggregate_metrics_path(args.results_dir, "exp14_face_knn"), rows
        )


if __name__ == "__main__":
    main()
