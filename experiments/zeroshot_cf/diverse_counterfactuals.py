"""Quality-constrained selection of diverse counterfactual sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from experiments.zeroshot_cf.data import OneHotActionGroup


@dataclass(frozen=True)
class CounterfactualSetDiversity:
    """Pairwise diversity diagnostics for one counterfactual set."""

    mean_action_set_jaccard: float
    minimum_action_set_jaccard: float
    mean_action_value_distance: float
    minimum_action_value_distance: float
    distinct_action_sets: int


def action_unit_signatures(
    rows: np.ndarray,
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
) -> np.ndarray:
    """Encode which numerical features or categorical groups each row changes."""
    matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    reference = np.asarray(factual, dtype=np.float64)
    signatures: list[np.ndarray] = []
    for column in numerical_columns:
        signatures.append(~np.isclose(matrix[:, int(column)], reference[int(column)]))
    for group in categorical_groups:
        columns = list(group.columns)
        factual_category = int(np.argmax(reference[columns]))
        signatures.append(np.argmax(matrix[:, columns], axis=1) != factual_category)
    if not signatures:
        return np.empty((len(matrix), 0), dtype=bool)
    return np.column_stack(signatures)


def pairwise_action_distances(
    rows: np.ndarray,
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
) -> tuple[np.ndarray, np.ndarray]:
    """Return changed-action Jaccard and mixed-value distance matrices."""
    matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    signatures = action_unit_signatures(
        matrix,
        factual,
        numerical_columns,
        categorical_groups,
    )
    intersection = signatures.astype(int) @ signatures.astype(int).T
    sizes = signatures.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    action_set_distance = np.divide(
        union - intersection,
        union,
        out=np.zeros_like(union, dtype=np.float64),
        where=union > 0,
    )

    n_units = len(numerical_columns) + len(categorical_groups)
    action_value_distance = np.zeros((len(matrix), len(matrix)), dtype=np.float64)
    if n_units == 0:
        return action_set_distance, action_value_distance
    for column in numerical_columns:
        values = matrix[:, int(column)]
        action_value_distance += np.clip(
            np.abs(values[:, None] - values[None, :]),
            0.0,
            1.0,
        )
    for group in categorical_groups:
        categories = np.argmax(matrix[:, list(group.columns)], axis=1)
        action_value_distance += categories[:, None] != categories[None, :]
    action_value_distance /= n_units
    return action_set_distance, action_value_distance


def select_diverse_counterfactuals(
    rows: np.ndarray,
    joint_log_density: np.ndarray,
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    *,
    primary_index: int,
    max_outputs: int,
) -> np.ndarray:
    """Select a deterministic quality-preserving, diverse subset.

    ``rows`` have already passed validity, actionability, action-budget, and
    TabICL quality constraints. The existing single-CFE winner is always first.
    Later rows maximize minimum changed-action-set distance, then minimum
    mixed-value distance. TabICL density, sparsity, and proximity break ties.
    """
    matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    scores = np.asarray(joint_log_density, dtype=np.float64)
    if len(matrix) != len(scores):
        raise ValueError("rows and joint_log_density must have equal length")
    if not 0 <= primary_index < len(matrix):
        raise ValueError("primary_index is outside the candidate pool")
    if max_outputs < 1:
        raise ValueError("max_outputs must be at least 1")
    if not np.all(np.isfinite(scores)):
        raise ValueError("joint_log_density must be finite")

    ordered = [primary_index, *(i for i in range(len(matrix)) if i != primary_index)]
    unique_indices: list[int] = []
    seen: set[bytes] = set()
    for index in ordered:
        key = np.ascontiguousarray(matrix[index]).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique_indices.append(index)
    unique = np.asarray(unique_indices, dtype=int)
    unique_rows = matrix[unique]
    unique_scores = scores[unique]
    action_set_distance, action_value_distance = pairwise_action_distances(
        unique_rows,
        factual,
        numerical_columns,
        categorical_groups,
    )
    signatures = action_unit_signatures(
        unique_rows,
        factual,
        numerical_columns,
        categorical_groups,
    )
    sparsity = signatures.sum(axis=1)
    proximity = np.linalg.norm(unique_rows - np.asarray(factual), axis=1)

    selected = [0]
    remaining = set(range(1, len(unique_rows)))
    while remaining and len(selected) < max_outputs:
        def selection_key(index: int) -> tuple[float, float, float, int, float, int]:
            return (
                float(action_set_distance[index, selected].min()),
                float(action_value_distance[index, selected].min()),
                float(unique_scores[index]),
                -int(sparsity[index]),
                -float(proximity[index]),
                -int(unique[index]),
            )

        best = max(remaining, key=selection_key)
        selected.append(best)
        remaining.remove(best)
    return unique[np.asarray(selected, dtype=int)]


def summarize_counterfactual_set(
    rows: np.ndarray,
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
) -> CounterfactualSetDiversity:
    """Summarize pairwise diversity for one non-empty counterfactual set."""
    matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    signatures = action_unit_signatures(
        matrix,
        factual,
        numerical_columns,
        categorical_groups,
    )
    distinct_action_sets = len({tuple(row.tolist()) for row in signatures})
    if len(matrix) < 2:
        return CounterfactualSetDiversity(
            mean_action_set_jaccard=0.0,
            minimum_action_set_jaccard=0.0,
            mean_action_value_distance=0.0,
            minimum_action_value_distance=0.0,
            distinct_action_sets=distinct_action_sets,
        )
    action_set_distance, action_value_distance = pairwise_action_distances(
        matrix,
        factual,
        numerical_columns,
        categorical_groups,
    )
    upper = np.triu_indices(len(matrix), k=1)
    set_values = action_set_distance[upper]
    value_values = action_value_distance[upper]
    return CounterfactualSetDiversity(
        mean_action_set_jaccard=float(set_values.mean()),
        minimum_action_set_jaccard=float(set_values.min()),
        mean_action_value_distance=float(value_values.mean()),
        minimum_action_value_distance=float(value_values.min()),
        distinct_action_sets=distinct_action_sets,
    )
