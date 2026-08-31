"""Shared action primitives for retained benchmark baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.validation import target_probabilities
from experiments.zeroshot_cf.retained_config import TAU


@dataclass(frozen=True)
class ActionUnit:
    """One scalar intervention or one atomic one-hot intervention."""

    name: str
    columns: tuple[int, ...]


def build_action_units(
    scalar_actionable: Sequence[int],
    grouped_actionable: Sequence[OneHotActionGroup],
) -> list[ActionUnit]:
    """Return the atomic scalar and grouped actions available to a baseline."""
    units = [
        ActionUnit(f"feature_{column}", (int(column),)) for column in scalar_actionable
    ]
    units.extend(
        ActionUnit(group.name, tuple(group.columns)) for group in grouped_actionable
    )
    return units


def _is_valid(
    disc_model: Any,
    rows: np.ndarray,
    target: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.atleast_2d(rows)
    probabilities = target_probabilities(
        disc_model,
        matrix,
        np.full(len(matrix), target),
    )
    predictions = np.asarray(disc_model.predict(matrix), dtype=int)
    return (predictions == target) & (probabilities >= tau), probabilities


def _changed_units(
    factual: np.ndarray,
    candidate: np.ndarray,
    action_units: Sequence[ActionUnit],
) -> list[ActionUnit]:
    return [
        unit
        for unit in action_units
        if not np.allclose(
            factual[list(unit.columns)],
            candidate[list(unit.columns)],
        )
    ]


def prune_counterfactual_actions(
    disc_model: Any,
    factual: np.ndarray,
    candidate: np.ndarray,
    target: int,
    action_units: Sequence[ActionUnit],
    *,
    tau: float = TAU,
) -> np.ndarray:
    """Greedily revert actions while preserving the target prediction."""
    current = np.asarray(candidate, dtype=np.float64).copy()
    factual = np.asarray(factual, dtype=np.float64)
    while True:
        changed = _changed_units(factual, current, action_units)
        if not changed:
            break
        trials = np.repeat(current.reshape(1, -1), len(changed), axis=0)
        for index, unit in enumerate(changed):
            columns = list(unit.columns)
            trials[index, columns] = factual[columns]
        valid, _ = _is_valid(disc_model, trials, target, tau)
        if not valid.any():
            break
        eligible = np.flatnonzero(valid)
        distances = np.linalg.norm(trials[eligible] - factual, axis=1)
        current = trials[int(eligible[np.argmin(distances)])]
    return current


def contract_scalar_actions(
    disc_model: Any,
    factual: np.ndarray,
    candidate: np.ndarray,
    target: int,
    scalar_columns: Sequence[int],
    *,
    tau: float = TAU,
    iterations: int = 12,
) -> np.ndarray:
    """Move each changed scalar toward the factual via validity-preserving bisection."""
    current = np.asarray(candidate, dtype=np.float64).copy()
    factual = np.asarray(factual, dtype=np.float64)
    for column in scalar_columns:
        if np.isclose(current[column], factual[column]):
            continue
        valid_value = float(current[column])
        invalid_value = float(factual[column])
        for _ in range(iterations):
            midpoint = 0.5 * (valid_value + invalid_value)
            trial = current.copy()
            trial[column] = midpoint
            valid, _ = _is_valid(disc_model, trial, target, tau)
            if bool(valid[0]):
                valid_value = midpoint
            else:
                invalid_value = midpoint
        current[column] = valid_value
    return current
