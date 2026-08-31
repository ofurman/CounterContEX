"""Validation and immutable-array helpers for portable benchmark contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np


def readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any] | None = None,
    ndim: int | None = None,
    name: str = "array",
) -> np.ndarray:
    """Return an owned, C-contiguous, read-only array."""
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}")
    array.setflags(write=False)
    return array


def frozen_mapping(mapping: Mapping[str, str], *, name: str) -> Mapping[str, str]:
    """Copy a string mapping into an immutable mapping proxy."""
    copied = dict(mapping)
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in copied.items()
    ):
        raise TypeError(f"{name} keys and values must be strings")
    return MappingProxyType(copied)


def deep_freeze(value: Any) -> Any:
    """Recursively freeze JSON-like protocol/configuration values."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(deep_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return readonly_array(value, name="configuration array")
    return value


def validate_unique_indices(
    values: Sequence[int],
    *,
    size: int,
    name: str,
) -> tuple[int, ...]:
    """Normalize and validate a column-index collection."""
    indices = tuple(int(value) for value in values)
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} contains duplicate indices")
    invalid = [index for index in indices if index < 0 or index >= size]
    if invalid:
        raise ValueError(f"{name} contains out-of-range indices: {invalid}")
    return indices


def validate_predictor(predictor: Any) -> np.ndarray:
    """Validate the classifier surface needed by the benchmark."""
    if not callable(getattr(predictor, "predict", None)):
        raise TypeError("predictor must define predict(X)")
    if not callable(getattr(predictor, "predict_proba", None)):
        raise TypeError("predictor must define predict_proba(X)")
    if not hasattr(predictor, "classes_"):
        raise TypeError("predictor must expose classes_")
    classes = np.asarray(predictor.classes_)
    if classes.ndim != 1 or not len(classes):
        raise ValueError("predictor.classes_ must be a non-empty 1D array")
    if len(np.unique(classes)) != len(classes):
        raise ValueError("predictor.classes_ contains duplicate labels")
    return classes


def target_probabilities(
    predictor: Any,
    X: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Read each row's target probability through ``classes_`` label mapping."""
    classes = validate_predictor(predictor)
    matrix = np.asarray(predictor.predict_proba(X), dtype=np.float64)
    labels = np.asarray(targets).reshape(-1)
    if matrix.ndim != 2 or matrix.shape != (len(labels), len(classes)):
        raise ValueError(
            "predict_proba shape must be (n_rows, len(classes_)); "
            f"got {matrix.shape} for {len(labels)} rows and {len(classes)} classes"
        )
    columns = {
        label.item() if isinstance(label, np.generic) else label: index
        for index, label in enumerate(classes)
    }
    try:
        selected = np.fromiter(
            (
                columns[label.item() if isinstance(label, np.generic) else label]
                for label in labels
            ),
            dtype=np.int64,
            count=len(labels),
        )
    except KeyError as error:
        raise ValueError(
            f"target label {error.args[0]!r} is absent from predictor.classes_"
        ) from error
    probabilities = matrix[np.arange(len(labels)), selected]
    return readonly_array(probabilities, dtype=np.float64, ndim=1, name="probabilities")
