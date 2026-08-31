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
import heapq
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.baseline_common import build_action_units
from experiments.zeroshot_cf.benchmark_protocol import (
    DATASETS,
    DEFAULT_MAX_TEST,
    DEFAULT_SPARSITY_EPS,
    DEFAULT_VALIDATION_FRACTION,
    aggregate_metrics_path,
    build_common_result_row,
    dataset_result_paths,
    mean_on_valid,
    prepare_benchmark_context,
    write_dataset_outputs,
    write_result_table,
)
from experiments.zeroshot_cf.retained_config import TAU

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

RESULTS_DIR = Path(__file__).parent / "results" / "local" / "exp14_face_knn"
_DISTANCE_EPS = 1e-12


@dataclass(frozen=True)
class FaceGraph:
    """Reusable density-weighted neighbourhood graph in actionable space."""

    X_train: np.ndarray
    actionable_columns: tuple[int, ...]
    adjacency: csr_matrix
    neighbours: NearestNeighbors
    local_scale: np.ndarray
    median_scale: float
    n_neighbors: int
    density_power: float

def _expanded_actionable_columns(
    scalar_actionable: list[int],
    grouped_actionable: list[OneHotActionGroup],
) -> tuple[int, ...]:
    return (
        *scalar_actionable,
        *(column for group in grouped_actionable for column in group.columns),
    )


def build_face_knn_graph(
    X_train: np.ndarray,
    actionable_columns: tuple[int, ...],
    *,
    n_neighbors: int = 100,
    density_power: float = 1.0,
) -> FaceGraph:
    """Build a symmetric kNN graph with a local-density edge penalty."""
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    X_train = np.asarray(X_train, dtype=np.float64)
    if X_train.ndim != 2 or len(X_train) < 2:
        raise ValueError("X_train must contain at least two rows")
    if not actionable_columns:
        raise ValueError("FACE requires at least one actionable feature")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if density_power < 0:
        raise ValueError("density_power must be non-negative")

    action_matrix = X_train[:, actionable_columns]
    graph_k = min(n_neighbors, len(X_train) - 1)
    neighbours = NearestNeighbors(
        n_neighbors=graph_k + 1,
        metric="euclidean",
        n_jobs=-1,
    ).fit(action_matrix)
    distances, indices = neighbours.kneighbors(action_matrix)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    local_scale = np.maximum(distances[:, -1], _DISTANCE_EPS)
    positive_scale = local_scale[local_scale > _DISTANCE_EPS]
    median_scale = float(np.median(positive_scale)) if len(positive_scale) else 1.0

    rows = np.repeat(np.arange(len(X_train)), graph_k)
    columns = indices.reshape(-1)
    base_distance = np.maximum(distances.reshape(-1), _DISTANCE_EPS)
    relative_sparsity = (local_scale[rows] + local_scale[columns]) / (
        2.0 * median_scale
    )
    weights = (
        base_distance * np.maximum(relative_sparsity, _DISTANCE_EPS) ** density_power
    )
    adjacency = csr_matrix(
        (weights, (rows, columns)),
        shape=(len(X_train), len(X_train)),
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    return FaceGraph(
        X_train=X_train,
        actionable_columns=actionable_columns,
        adjacency=adjacency,
        neighbours=neighbours,
        local_scale=local_scale,
        median_scale=median_scale,
        n_neighbors=graph_k,
        density_power=density_power,
    )


def _compose_candidate(
    factual: np.ndarray,
    training_row: np.ndarray,
    actionable_columns: tuple[int, ...],
) -> np.ndarray:
    candidate = np.asarray(factual, dtype=np.float64).copy()
    candidate[list(actionable_columns)] = training_row[list(actionable_columns)]
    return candidate


def face_counterfactual(
    graph: FaceGraph,
    classifier: Any,
    factual: np.ndarray,
    target: int,
    *,
    tau: float = TAU,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the first target endpoint reached by density-weighted Dijkstra."""
    factual = np.asarray(factual, dtype=np.float64)
    query = factual[list(graph.actionable_columns)].reshape(1, -1)
    query_distances, query_indices = graph.neighbours.kneighbors(
        query,
        n_neighbors=graph.n_neighbors,
    )
    query_distances = query_distances[0]
    query_indices = query_indices[0]
    query_scale = max(float(query_distances[-1]), _DISTANCE_EPS)
    density_factor = (query_scale + graph.local_scale[query_indices]) / (
        2.0 * graph.median_scale
    )
    source_weights = (
        np.maximum(query_distances, _DISTANCE_EPS)
        * np.maximum(density_factor, _DISTANCE_EPS) ** graph.density_power
    )

    best_distance = np.full(len(graph.X_train), np.inf, dtype=np.float64)
    predecessor = np.full(len(graph.X_train), -1, dtype=int)
    heap: list[tuple[float, int]] = []
    for node, weight in zip(query_indices, source_weights, strict=True):
        if weight < best_distance[node]:
            best_distance[node] = float(weight)
            heapq.heappush(heap, (float(weight), int(node)))

    visited = np.zeros(len(graph.X_train), dtype=bool)
    expanded = 0
    endpoint = -1
    endpoint_probability = float("nan")
    while heap:
        distance, node = heapq.heappop(heap)
        if visited[node] or distance > best_distance[node]:
            continue
        visited[node] = True
        expanded += 1
        candidate = _compose_candidate(
            factual,
            graph.X_train[node],
            graph.actionable_columns,
        )
        prediction = int(classifier.predict(candidate.reshape(1, -1))[0])
        probability = float(
            classifier.predict_proba(candidate.reshape(1, -1))[0, target]
        )
        if prediction == target and probability >= tau:
            endpoint = node
            endpoint_probability = probability
            break

        row_start = graph.adjacency.indptr[node]
        row_end = graph.adjacency.indptr[node + 1]
        for neighbour, edge_weight in zip(
            graph.adjacency.indices[row_start:row_end],
            graph.adjacency.data[row_start:row_end],
            strict=True,
        ):
            proposal = distance + float(edge_weight)
            if proposal < best_distance[neighbour]:
                best_distance[neighbour] = proposal
                predecessor[neighbour] = node
                heapq.heappush(heap, (proposal, int(neighbour)))

    if endpoint < 0:
        return factual.copy(), {
            "valid": False,
            "prediction": int(classifier.predict(factual.reshape(1, -1))[0]),
            "target_probability": float(
                classifier.predict_proba(factual.reshape(1, -1))[0, target]
            ),
            "endpoint_index": -1,
            "path_cost": float("nan"),
            "path_steps": 0,
            "expanded_nodes": expanded,
        }

    path_steps = 1
    cursor = endpoint
    while predecessor[cursor] >= 0:
        path_steps += 1
        cursor = int(predecessor[cursor])
    counterfactual = _compose_candidate(
        factual,
        graph.X_train[endpoint],
        graph.actionable_columns,
    )
    return counterfactual, {
        "valid": True,
        "prediction": target,
        "target_probability": endpoint_probability,
        "endpoint_index": endpoint,
        "path_cost": float(best_distance[endpoint]),
        "path_steps": path_steps,
        "expanded_nodes": expanded,
    }


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
    from experiments.zeroshot_cf.metrics_harness import (
        compute_dicoflex_common_metrics,
        print_metrics,
    )

    total_started = time.perf_counter()
    context = prepare_benchmark_context(
        dataset_name,
        max_test=max_test,
        validation_fraction=validation_fraction,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
    )
    bundle = context.bundle
    X_test = context.X_test
    y_test = context.y_test
    actionable_columns = _expanded_actionable_columns(
        list(context.scalar_actionable),
        list(context.grouped_actionable),
    )
    action_units = build_action_units(
        list(context.scalar_actionable),
        list(context.grouped_actionable),
    )

    graph_started = time.perf_counter()
    graph = build_face_knn_graph(
        bundle.X_train,
        actionable_columns,
        n_neighbors=n_neighbors,
        density_power=density_power,
    )
    runtime_graph = time.perf_counter() - graph_started
    search_started = time.perf_counter()
    X_cf = np.empty_like(X_test)
    point_info: list[dict[str, Any]] = []
    for index, (factual, target) in enumerate(
        zip(X_test, context.y_target, strict=True)
    ):
        point_started = time.perf_counter()
        X_cf[index], info = face_counterfactual(
            graph,
            context.disc_model,
            factual,
            int(target),
            tau=tau,
        )
        info["runtime_s"] = time.perf_counter() - point_started
        point_info.append(info)
    runtime_search = time.perf_counter() - search_started

    common_metrics = compute_dicoflex_common_metrics(
        context.disc_model,
        X_cf,
        X_test,
        bundle.X_train,
        context.y_target,
        bundle.numerical_features_indices,
        list(context.immutable_idx),
        categorical_groups=context.categorical_groups,
        sparsity_eps=DEFAULT_SPARSITY_EPS,
    )
    print_metrics(common_metrics, prefix=f"{dataset_name}/FACE-kNN")
    valid = np.asarray([info["valid"] for info in point_info], dtype=bool)
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
        "n_neighbors": graph.n_neighbors,
        "edge_weight": "euclidean_times_relative_knn_radius",
        "density_power": density_power,
        "tau": tau,
        "endpoint": "observed_actionable_projection",
        "categorical_actions": "atomic_one_hot_groups",
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

    write_dataset_outputs(
        dataset_result_paths(results_dir, "exp14_face_knn", dataset_name),
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
        write_result_table(aggregate_metrics_path(args.results_dir, "exp14_face_knn"), rows)


if __name__ == "__main__":
    main()
