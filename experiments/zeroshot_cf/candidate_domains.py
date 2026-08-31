"""Dependency-light candidate-domain helpers for retained generators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

FeatureDomains: TypeAlias = tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]


def infer_feature_domains(
    X_train: np.ndarray,
    *,
    max_discrete_values: int = 20,
) -> FeatureDomains:
    """Infer training bounds and small empirical supports for projection."""
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X.shape}")
    lower = np.nanmin(X, axis=0)
    upper = np.nanmax(X, axis=0)
    supports: dict[int, np.ndarray] = {}
    for column in range(X.shape[1]):
        values = np.unique(X[:, column][~np.isnan(X[:, column])])
        if 0 < len(values) <= max_discrete_values:
            supports[column] = values
    return lower, upper, supports


def project_candidate_values(
    candidates: Sequence[int],
    values: np.ndarray,
    feature_domains: FeatureDomains | None,
) -> np.ndarray:
    """Project candidate values to training bounds and empirical supports."""
    projected = np.asarray(values, dtype=np.float64).copy()
    if feature_domains is None:
        return projected

    lower, upper, supports = feature_domains
    columns = np.asarray(candidates, dtype=int)
    projected = np.clip(projected, lower[columns], upper[columns])
    for position, column in enumerate(columns):
        support = supports.get(int(column))
        if support is not None:
            nearest = int(np.abs(support - projected[position]).argmin())
            projected[position] = support[nearest]
    return projected
