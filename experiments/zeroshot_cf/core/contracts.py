"""Immutable provider-neutral data contracts for benchmark execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.validation import (
    deep_freeze,
    frozen_mapping,
    readonly_array,
    validate_predictor,
    validate_unique_indices,
)


@dataclass(frozen=True)
class FeatureDomains:
    """Training-derived bounds and optional empirical scalar supports."""

    lower: np.ndarray
    upper: np.ndarray
    discrete: Mapping[int, np.ndarray]

    def __post_init__(self) -> None:
        lower = readonly_array(self.lower, dtype=np.float64, ndim=1, name="lower")
        upper = readonly_array(self.upper, dtype=np.float64, ndim=1, name="upper")
        if lower.shape != upper.shape:
            raise ValueError("domain lower and upper bounds must have equal shape")
        if np.any(lower > upper):
            raise ValueError("domain lower bounds must not exceed upper bounds")
        supports: dict[int, np.ndarray] = {}
        for raw_column, raw_values in self.discrete.items():
            column = int(raw_column)
            if column < 0 or column >= len(lower):
                raise ValueError(f"discrete domain column {column} is out of range")
            values = readonly_array(
                raw_values, dtype=np.float64, ndim=1, name=f"discrete[{column}]"
            )
            if not len(values):
                raise ValueError(f"discrete[{column}] must not be empty")
            if len(np.unique(values)) != len(values):
                raise ValueError(f"discrete[{column}] contains duplicate values")
            if np.any(values < lower[column]) or np.any(values > upper[column]):
                raise ValueError(f"discrete[{column}] lies outside its bounds")
            supports[column] = values
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "discrete", MappingProxyType(supports))

    @property
    def supports(self) -> Mapping[int, np.ndarray]:
        """Compatibility name for empirical discrete domains."""
        return self.discrete


@dataclass(frozen=True)
class FeatureSchema:
    """Feature type and actionability metadata in transformed column order."""

    names: tuple[str, ...]
    numerical: tuple[int, ...]
    categorical_groups: tuple[OneHotActionGroup, ...]
    actionable_scalars: tuple[int, ...]
    actionable_groups: tuple[OneHotActionGroup, ...]
    immutable: tuple[int, ...]
    domains: FeatureDomains

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if not names or any(not name for name in names):
            raise ValueError("feature names must be non-empty")
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        size = len(names)
        numerical = validate_unique_indices(self.numerical, size=size, name="numerical")
        actionable_scalars = validate_unique_indices(
            self.actionable_scalars, size=size, name="actionable_scalars"
        )
        immutable = validate_unique_indices(self.immutable, size=size, name="immutable")

        def validate_groups(
            groups: tuple[OneHotActionGroup, ...], name: str
        ) -> tuple[OneHotActionGroup, ...]:
            normalized = tuple(
                OneHotActionGroup(
                    group.name,
                    validate_unique_indices(
                        group.columns, size=size, name=f"{name}.{group.name}"
                    ),
                )
                for group in groups
            )
            if len({group.name for group in normalized}) != len(normalized):
                raise ValueError(f"{name} contains duplicate group names")
            seen: set[int] = set()
            for group in normalized:
                columns = group.columns
                if len(columns) < 2:
                    raise ValueError(
                        f"{name}.{group.name} must contain at least two columns"
                    )
                overlap = seen.intersection(columns)
                if overlap:
                    raise ValueError(
                        f"{name} groups overlap at columns {sorted(overlap)}"
                    )
                seen.update(columns)
            return normalized

        categorical_groups = validate_groups(
            self.categorical_groups, "categorical_groups"
        )
        actionable_groups = validate_groups(self.actionable_groups, "actionable_groups")
        categorical_columns = {
            column for group in categorical_groups for column in group.columns
        }
        if set(numerical).intersection(categorical_columns):
            raise ValueError(
                "numerical and categorical feature columns must be disjoint"
            )
        if set(numerical) | categorical_columns != set(range(size)):
            raise ValueError(
                "numerical and categorical groups must partition all features"
            )
        categorical_signatures = {
            (group.name, group.columns) for group in categorical_groups
        }
        if any(
            (group.name, group.columns) not in categorical_signatures
            for group in actionable_groups
        ):
            raise ValueError("every actionable group must be a categorical group")
        actionable_columns = set(actionable_scalars)
        actionable_columns.update(
            column for group in actionable_groups for column in group.columns
        )
        if actionable_columns.intersection(immutable):
            raise ValueError(
                "actionable and immutable feature columns must be disjoint"
            )
        if actionable_columns | set(immutable) != set(range(size)):
            raise ValueError(
                "actionable and immutable columns must partition all features"
            )
        if len(self.domains.lower) != size:
            raise ValueError("feature domains must match the schema feature count")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "numerical", numerical)
        object.__setattr__(self, "categorical_groups", categorical_groups)
        object.__setattr__(self, "actionable_scalars", actionable_scalars)
        object.__setattr__(self, "actionable_groups", actionable_groups)
        object.__setattr__(self, "immutable", immutable)


@dataclass(frozen=True)
class DatasetProvenance:
    provider: str
    source_revision: str
    source_hashes: Mapping[str, str]
    preprocessing_id: str
    split_id: str
    fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "provider",
            "source_revision",
            "preprocessing_id",
            "split_id",
            "fingerprint",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(
            self,
            "source_hashes",
            frozen_mapping(self.source_hashes, name="source_hashes"),
        )
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    schema: FeatureSchema
    provenance: DatasetProvenance

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset name must be non-empty")
        pairs = (
            ("train", self.X_train, self.y_train),
            ("validation", self.X_validation, self.y_validation),
            ("test", self.X_test, self.y_test),
        )
        width = len(self.schema.names)
        for split, raw_X, raw_y in pairs:
            X = readonly_array(raw_X, dtype=np.float64, ndim=2, name=f"X_{split}")
            y = readonly_array(raw_y, ndim=1, name=f"y_{split}")
            if len(X) != len(y):
                raise ValueError(f"{split} feature and label row counts differ")
            if X.shape[1] != width:
                raise ValueError(f"{split} feature width does not match schema")
            object.__setattr__(self, f"X_{split}", X)
            object.__setattr__(self, f"y_{split}", y)


@dataclass(frozen=True)
class FactualSelection:
    indices: np.ndarray
    values: np.ndarray
    true_labels: np.ndarray

    def __post_init__(self) -> None:
        indices = readonly_array(self.indices, dtype=np.int64, ndim=1, name="indices")
        values = readonly_array(self.values, dtype=np.float64, ndim=2, name="values")
        labels = readonly_array(self.true_labels, ndim=1, name="true_labels")
        if len(indices) != len(values) or len(indices) != len(labels):
            raise ValueError(
                "factual indices, values, and labels must have equal row counts"
            )
        if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
            raise ValueError("factual source indices must be non-negative and unique")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "true_labels", labels)


@runtime_checkable
class Predictor(Protocol):
    classes_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    dataset: PreparedDataset
    factuals: FactualSelection
    oracle: Predictor
    factual_predictions: np.ndarray
    targets: np.ndarray
    protocol: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        validate_predictor(self.oracle)
        predictions = readonly_array(
            self.factual_predictions, ndim=1, name="factual_predictions"
        )
        targets = readonly_array(self.targets, ndim=1, name="targets")
        if len(predictions) != len(self.factuals.values) or len(targets) != len(
            predictions
        ):
            raise ValueError(
                "benchmark predictions and targets must match factual count"
            )
        if self.factuals.values.shape[1] != len(self.dataset.schema.names):
            raise ValueError("factual feature width must match dataset schema")
        if np.any(self.factuals.indices >= len(self.dataset.X_test)):
            raise ValueError("factual indices must refer to dataset test rows")
        expected_values = self.dataset.X_test[self.factuals.indices]
        if not np.array_equal(self.factuals.values, expected_values, equal_nan=True):
            raise ValueError(
                "factual values must exactly match dataset test rows at source indices"
            )
        expected_labels = self.dataset.y_test[self.factuals.indices]
        if not np.array_equal(self.factuals.true_labels, expected_labels):
            raise ValueError(
                "factual true labels must exactly match dataset test rows "
                "at source indices"
            )
        object.__setattr__(self, "factual_predictions", predictions)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "protocol", deep_freeze(self.protocol))


@dataclass(frozen=True)
class MethodContext:
    X_reference: np.ndarray
    feature_schema: FeatureSchema
    oracle: Predictor

    def __post_init__(self) -> None:
        X = readonly_array(
            self.X_reference, dtype=np.float64, ndim=2, name="X_reference"
        )
        if X.shape[1] != len(self.feature_schema.names):
            raise ValueError("reference feature width must match schema")
        validate_predictor(self.oracle)
        object.__setattr__(self, "X_reference", X)


@dataclass(frozen=True)
class GenerationRequest:
    factuals: np.ndarray
    targets: np.ndarray
    n_counterfactuals: int
    seed: int

    def __post_init__(self) -> None:
        factuals = readonly_array(
            self.factuals, dtype=np.float64, ndim=2, name="factuals"
        )
        targets = readonly_array(self.targets, ndim=1, name="targets")
        if len(factuals) != len(targets):
            raise ValueError(
                "generation factuals and targets must have equal row counts"
            )
        if self.n_counterfactuals <= 0:
            raise ValueError("n_counterfactuals must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "factuals", factuals)
        object.__setattr__(self, "targets", targets)


def _validate_json_value(value: Any, *, path: str) -> None:
    """Reject diagnostics that cannot be represented in a manifest."""
    if value is None or isinstance(value, str | bool | int | float):
        return
    if isinstance(value, np.generic):
        _validate_json_value(value.item(), path=path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} must contain only JSON-serializable values")


@dataclass(frozen=True)
class GenerationResult:
    """Canonical, method-neutral counterfactual generation output."""

    candidates: np.ndarray
    available: np.ndarray
    point_diagnostics: tuple[Mapping[str, Any], ...] = ()
    run_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidates = readonly_array(
            self.candidates,
            dtype=np.float64,
            ndim=3,
            name="candidates",
        )
        available = readonly_array(
            self.available,
            dtype=np.bool_,
            ndim=2,
            name="available",
        )
        if available.shape != candidates.shape[:2]:
            raise ValueError("available must have shape (n_factuals, k)")
        if candidates.shape[1] <= 0 or candidates.shape[2] <= 0:
            raise ValueError("candidates must have positive k and feature dimensions")
        if np.any(~np.isfinite(candidates[available])):
            raise ValueError("available slots must be finite")
        if np.any(~np.isnan(candidates[~available])):
            raise ValueError("unavailable slots must contain only NaN")
        for rows, row_available in zip(candidates, available, strict=True):
            returned = rows[row_available]
            if len(returned) > 1 and len(np.unique(returned, axis=0)) != len(returned):
                raise ValueError("available candidates must not duplicate padding")

        diagnostics = tuple(dict(item) for item in self.point_diagnostics)
        if diagnostics and len(diagnostics) != len(candidates):
            raise ValueError("point diagnostics must match the factual row count")
        for index, item in enumerate(diagnostics):
            _validate_json_value(item, path=f"point_diagnostics[{index}]")
        run_diagnostics = dict(self.run_diagnostics)
        _validate_json_value(run_diagnostics, path="run_diagnostics")

        artifacts: dict[str, np.ndarray] = {}
        for name, value in self.artifacts.items():
            if not isinstance(name, str) or not name.startswith("method."):
                raise ValueError(
                    "artifact names must use the method.* namespace, "
                    "for example method.best_effort"
                )
            artifact = readonly_array(value, name=f"artifacts[{name!r}]")
            if artifact.dtype.hasobject:
                raise TypeError(f"artifacts[{name!r}] must not use object dtype")
            artifacts[name] = artifact

        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "available", available)
        object.__setattr__(
            self, "point_diagnostics", tuple(deep_freeze(item) for item in diagnostics)
        )
        object.__setattr__(self, "run_diagnostics", deep_freeze(run_diagnostics))
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))

    def validate_for_factuals(self, factuals: np.ndarray) -> None:
        """Validate dimensions and reject factual rows represented as candidates."""
        values = np.asarray(factuals, dtype=np.float64)
        if values.ndim != 2 or values.shape != (
            self.candidates.shape[0],
            self.candidates.shape[2],
        ):
            raise ValueError("factuals must match candidate n and feature dimensions")
        factual_padding = self.available & np.all(
            self.candidates == values[:, None, :], axis=2
        )
        if factual_padding.any():
            raise ValueError("available candidates must not contain factual padding")
