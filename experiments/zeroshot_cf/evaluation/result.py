"""Versioned, typed evaluator outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from experiments.zeroshot_cf.core.validation import deep_freeze, readonly_array

METRIC_SCHEMA_VERSION = "countercontex.evaluation.v2"
Scalar = str | int | float | bool | None


@dataclass(frozen=True)
class EvaluationSpec:
    metric_version: str = METRIC_SCHEMA_VERSION
    sparsity_epsilon: float = 0.05
    probability_threshold: float = 0.7
    primary_rank: int = 0
    lof_n_neighbors: int = 20
    isolation_forest_estimators: int = 100
    detectability_min_cf_rows: int = 20
    gower_neighbor_k: int = 5

    def __post_init__(self) -> None:
        if self.metric_version != METRIC_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported metric schema version: {self.metric_version}"
            )
        if self.sparsity_epsilon < 0:
            raise ValueError("sparsity_epsilon must be non-negative")
        if not 0 <= self.probability_threshold <= 1:
            raise ValueError("probability_threshold must be between zero and one")
        if self.primary_rank < 0:
            raise ValueError("primary_rank must be non-negative")
        if self.lof_n_neighbors <= 0 or self.isolation_forest_estimators <= 0:
            raise ValueError("novelty estimator sizes must be positive")
        if self.detectability_min_cf_rows < 2:
            raise ValueError("detectability_min_cf_rows must be at least two")
        if self.gower_neighbor_k <= 0:
            raise ValueError("gower_neighbor_k must be positive")


@dataclass(frozen=True)
class SummaryOutput:
    schema_version: str
    values: Mapping[str, Scalar]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class PointOutput:
    point: int
    values: Mapping[str, Scalar]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class CandidateOutput:
    point: int
    rank: int
    values: Mapping[str, Scalar]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class ArrayOutput:
    schema_version: str
    values: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name, value in self.values.items():
            array = readonly_array(value, name=f"arrays[{name!r}]")
            if array.dtype.hasobject:
                raise TypeError(f"arrays[{name!r}] must not use object dtype")
            arrays[str(name)] = array
        object.__setattr__(self, "values", MappingProxyType(arrays))


@dataclass(frozen=True)
class EvaluationReport:
    """All common outputs from one evaluator invocation."""

    schema_version: str
    summary: SummaryOutput
    points: tuple[PointOutput, ...]
    candidates: tuple[CandidateOutput, ...]
    arrays: ArrayOutput
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != METRIC_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported report schema version: {self.schema_version}"
            )
        if self.summary.schema_version != self.schema_version:
            raise ValueError("summary schema version does not match report")
        if self.arrays.schema_version != self.schema_version:
            raise ValueError("array schema version does not match report")
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))
