"""Density-weighted FACE-kNN method adapter."""

from __future__ import annotations

import heapq
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.contracts import GenerationRequest, MethodContext
from experiments.zeroshot_cf.core.validation import target_probabilities
from experiments.zeroshot_cf.methods.base import (
    MethodCapabilities,
    canonical_single_result,
    require_single_counterfactual,
)
from experiments.zeroshot_cf.retained_config import TAU

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors


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
            target_probabilities(
                classifier,
                candidate.reshape(1, -1),
                np.array([target]),
            )[0]
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
                target_probabilities(
                    classifier,
                    factual.reshape(1, -1),
                    np.array([target]),
                )[0]
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


@dataclass(frozen=True)
class FaceConfig:
    n_neighbors: int = 100
    density_power: float = 1.0
    tau: float = TAU

    def __post_init__(self) -> None:
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if self.density_power < 0:
            raise ValueError("density_power must be non-negative")
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be between zero and one")


@dataclass(frozen=True)
class FaceMethod:
    config: FaceConfig = FaceConfig()
    method_id = "face_knn_density_weighted"
    capabilities = MethodCapabilities(
        supports_categorical=True,
        enforces_actionability=True,
        supports_multiple_counterfactuals=False,
        requires_probabilities=True,
        optional_dependencies=("scipy", "scikit-learn"),
    )

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def prepare(self, context: MethodContext) -> PreparedFaceMethod:
        actionable_columns = (
            *context.feature_schema.actionable_scalars,
            *(
                column
                for group in context.feature_schema.actionable_groups
                for column in group.columns
            ),
        )
        graph = build_face_knn_graph(
            context.X_reference,
            actionable_columns,
            n_neighbors=self.config.n_neighbors,
            density_power=self.config.density_power,
        )
        return PreparedFaceMethod(context, self.config, graph)


@dataclass(frozen=True)
class PreparedFaceMethod:
    context: MethodContext
    config: FaceConfig
    graph: FaceGraph

    def generate(self, request: GenerationRequest):
        require_single_counterfactual(request)
        if request.factuals.shape[1] != self.context.X_reference.shape[1]:
            raise ValueError("request feature width does not match method context")
        raw = np.empty_like(request.factuals)
        available = np.zeros(len(request.factuals), dtype=bool)
        diagnostics: list[dict[str, Any]] = []
        for index, (factual, target) in enumerate(
            zip(request.factuals, request.targets, strict=True)
        ):
            started = time.perf_counter()
            candidate, info = face_counterfactual(
                self.graph,
                self.context.oracle,
                factual,
                target.item() if isinstance(target, np.generic) else target,
                tau=self.config.tau,
            )
            raw[index] = candidate
            available[index] = bool(info["valid"])
            diagnostics.append({**info, "runtime_s": time.perf_counter() - started})
        return canonical_single_result(
            raw,
            available,
            point_diagnostics=tuple(diagnostics),
            run_diagnostics={"seed": request.seed},
        )
