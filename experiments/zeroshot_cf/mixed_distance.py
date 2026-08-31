"""Distances and change counts over original mixed-data feature units."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup


def compact_gower_distance(
    rows: np.ndarray,
    reference: np.ndarray,
    categorical_columns: Sequence[int] = (),
) -> np.ndarray:
    """Return Gower distance for a compact, MinMax-scaled mixed matrix.

    Each categorical column contains one category identifier and contributes
    zero or one. All remaining columns are numerical and contribute their
    absolute difference, clipped to the unit interval. Every original feature
    therefore receives equal weight regardless of its representation.
    """
    matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    factual = np.asarray(reference, dtype=np.float64)
    if factual.ndim == 1:
        factual = np.broadcast_to(factual, matrix.shape)
    if factual.shape != matrix.shape:
        raise ValueError("reference must be one row or have the same shape as rows")

    categorical = tuple(int(column) for column in categorical_columns)
    if len(set(categorical)) != len(categorical):
        raise ValueError("categorical_columns must be unique")
    if any(column < 0 or column >= matrix.shape[1] for column in categorical):
        raise IndexError("categorical column index is out of bounds")

    numerical = tuple(
        column for column in range(matrix.shape[1]) if column not in categorical
    )
    distances = np.zeros(len(matrix), dtype=np.float64)
    if numerical:
        distances += np.clip(
            np.abs(matrix[:, numerical] - factual[:, numerical]),
            0.0,
            1.0,
        ).sum(axis=1)
    if categorical:
        distances += (matrix[:, categorical] != factual[:, categorical]).sum(axis=1)
    return distances / matrix.shape[1]


def grouped_gower_distance(
    rows: np.ndarray,
    reference: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
) -> np.ndarray:
    """Return mixed Gower distance from ``reference`` for every row.

    Numerical inputs are expected to use the experiment's [0, 1] scaling, so
    their contribution is absolute distance clipped to one.  A one-hot group
    contributes exactly zero or one regardless of how many dummy columns it
    contains.  The mean is taken over original feature/action units.
    """
    matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    factual = np.asarray(reference, dtype=np.float64)
    if factual.ndim == 1:
        factual = np.broadcast_to(factual, matrix.shape)
    if factual.shape != matrix.shape:
        raise ValueError("reference must be one row or have the same shape as rows")

    numerical = tuple(int(column) for column in numerical_columns)
    groups = tuple(categorical_groups)
    n_units = len(numerical) + len(groups)
    if n_units == 0:
        return np.zeros(len(matrix), dtype=np.float64)

    distances = np.zeros(len(matrix), dtype=np.float64)
    if numerical:
        distances += np.clip(
            np.abs(matrix[:, numerical] - factual[:, numerical]),
            0.0,
            1.0,
        ).sum(axis=1)
    for group in groups:
        columns = list(group.columns)
        factual_categories = np.argmax(factual[:, columns], axis=1)
        distances += np.argmax(matrix[:, columns], axis=1) != factual_categories
    return distances / n_units


def action_unit_change_count(
    rows: np.ndarray,
    reference: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    *,
    numerical_tolerance: float = 0.0,
) -> np.ndarray:
    """Count changed numeric features and changed categorical groups."""
    if numerical_tolerance < 0:
        raise ValueError("numerical_tolerance must be non-negative")
    matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    factual = np.asarray(reference, dtype=np.float64)
    if factual.ndim == 1:
        factual = np.broadcast_to(factual, matrix.shape)
    if factual.shape != matrix.shape:
        raise ValueError("reference must be one row or have the same shape as rows")

    counts = np.zeros(len(matrix), dtype=np.int64)
    numerical = tuple(int(column) for column in numerical_columns)
    if numerical:
        numerical_changed = (
            ~np.isclose(matrix[:, numerical], factual[:, numerical])
            if numerical_tolerance == 0.0
            else np.abs(matrix[:, numerical] - factual[:, numerical])
            > numerical_tolerance
        )
        counts += numerical_changed.sum(axis=1)
    for group in categorical_groups:
        columns = list(group.columns)
        factual_categories = np.argmax(factual[:, columns], axis=1)
        counts += np.argmax(matrix[:, columns], axis=1) != factual_categories
    return counts
