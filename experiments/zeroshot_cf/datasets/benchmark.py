"""Reusable benchmark-case construction from a portable prepared dataset."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.core.contracts import (
    BenchmarkCase,
    FactualSelection,
    MethodContext,
    Predictor,
    PreparedDataset,
)
from experiments.zeroshot_cf.core.validation import target_probabilities
from sklearn.model_selection import train_test_split


class PredictorAdapter:
    """Expose a validated classifier contract without changing a legacy model."""

    def __init__(self, model: Any) -> None:
        self.model = model
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "_clf"):
            classes = getattr(model._clf, "classes_", None)
        if classes is None:
            raise TypeError(
                "classifier must expose classes_ (directly or through _clf)"
            )
        self.classes_ = np.asarray(classes).copy()
        self.classes_.setflags(write=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)


def select_factual_indices(
    labels: np.ndarray,
    limit: int | None,
    selection: str = "stratified",
    *,
    seed: int = 42,
) -> np.ndarray:
    """Select stable source indices from a held-out label array."""
    y = np.asarray(labels).reshape(-1)
    if selection not in {"first", "stratified"}:
        raise ValueError("test_selection must be 'first' or 'stratified'")
    if limit is None or limit >= len(y):
        selected = np.arange(len(y), dtype=np.int64)
    elif limit <= 0:
        raise ValueError("max_test must be positive or -1 for the full test set")
    elif selection == "first":
        selected = np.arange(limit, dtype=np.int64)
    elif limit < len(np.unique(y)):
        selected = np.sort(
            np.random.default_rng(seed).choice(len(y), size=limit, replace=False)
        ).astype(np.int64)
    else:
        selected, _ = train_test_split(
            np.arange(len(y)),
            train_size=limit,
            random_state=seed,
            stratify=y,
        )
        selected = np.sort(selected).astype(np.int64)
    selected.setflags(write=False)
    return selected


def select_factuals(
    dataset: PreparedDataset,
    limit: int | None,
    selection: str = "stratified",
    *,
    seed: int = 42,
) -> FactualSelection:
    """Select factual values and truth once while retaining source indices."""
    indices = select_factual_indices(dataset.y_test, limit, selection, seed=seed)
    return FactualSelection(
        indices=indices,
        values=dataset.X_test[indices],
        true_labels=dataset.y_test[indices],
    )


def _opposite_binary_labels(classes: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    if len(classes) != 2:
        raise ValueError(
            f"benchmark target policy requires two classes, got {classes.tolist()}"
        )
    mapping = {classes[0].item(): classes[1], classes[1].item(): classes[0]}
    try:
        return np.asarray([mapping[value.item()] for value in predictions])
    except KeyError as error:
        raise ValueError(
            f"prediction {error.args[0]!r} is absent from classifier.classes_"
        ) from error


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def _canonical_model_value(value: Any, seen: set[int] | None = None) -> Any:
    """Convert estimator configuration and fitted state into stable JSON data."""
    if seen is None:
        seen = set()
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, np.generic):
        return _canonical_model_value(value.item(), seen)
    if isinstance(value, np.ndarray):
        return {
            "ndarray": _array_digest(value),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    if isinstance(value, np.random.RandomState):
        return {
            "random_state": _canonical_model_value(value.get_state(), seen),
        }
    if isinstance(value, np.random.Generator):
        return {
            "generator": _canonical_model_value(value.bit_generator.state, seen),
        }
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_model_value(item, seen)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [_canonical_model_value(item, seen) for item in value]
    if isinstance(value, set | frozenset):
        items = [_canonical_model_value(item, seen) for item in value]
        return sorted(
            items, key=lambda item: json.dumps(item, sort_keys=True, default=str)
        )
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if callable(value):
        return {
            "callable": (
                f"{getattr(value, '__module__', '')}."
                f"{getattr(value, '__qualname__', type(value).__qualname__)}"
            )
        }

    marker = id(value)
    if marker in seen:
        return {"cycle": f"{type(value).__module__}.{type(value).__qualname__}"}
    seen.add(marker)
    try:
        state = getattr(value, "__dict__", None)
        if state is not None:
            return {
                "object": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": _canonical_model_value(state, seen),
            }
        return {
            "object": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }
    finally:
        seen.remove(marker)


def _implementation_digest(model: Any) -> str:
    """Fingerprint the predictor implementation, including compatibility wrappers."""
    implementations: list[dict[str, str]] = []
    current = model
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        cls = type(current)
        try:
            source = inspect.getsource(cls).encode()
        except (OSError, TypeError):
            methods: list[bytes] = []
            for name in ("predict", "predict_proba"):
                method = getattr(cls, name, None)
                code = getattr(method, "__code__", None)
                if code is not None:
                    methods.append(code.co_code + repr(code.co_consts).encode())
            source = b"".join(methods)
        implementations.append(
            {
                "class": f"{cls.__module__}.{cls.__qualname__}",
                "source_sha256": hashlib.sha256(source).hexdigest(),
            }
        )
        if isinstance(current, PredictorAdapter):
            current = current.model
        elif hasattr(current, "_clf"):
            current = current._clf
        else:
            break
    return hashlib.sha256(
        json.dumps(implementations, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _model_identity(
    predictor: Predictor,
    declared_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    """Resolve declared parameters and hash the actual fitted predictor content."""
    if not declared_config:
        raise ValueError("target_model must be a non-empty resolved model config")
    actual: Any = (
        predictor.model if isinstance(predictor, PredictorAdapter) else predictor
    )
    estimator = actual._clf if hasattr(actual, "_clf") else actual
    get_params = getattr(estimator, "get_params", None)
    estimator_params = get_params(deep=True) if callable(get_params) else {}
    resolved_config = {
        "declared": dict(declared_config),
        "estimator_params": estimator_params,
    }
    if not resolved_config["declared"] and not resolved_config["estimator_params"]:
        raise ValueError("resolved target model config must be non-empty")
    content = {
        "implementation_fingerprint": _implementation_digest(predictor),
        "resolved_config": _canonical_model_value(resolved_config),
        "classes": _canonical_model_value(np.asarray(predictor.classes_)),
        "fitted_predictor": _canonical_model_value(actual),
    }
    fingerprint = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return content, fingerprint


def build_benchmark_case(
    dataset: PreparedDataset,
    oracle: Predictor | Any,
    *,
    max_test: int | None = 1000,
    test_selection: str = "stratified",
    seed: int = 42,
    target_model: Mapping[str, Any] | None = None,
) -> BenchmarkCase:
    """Build one immutable, reusable case from real dataset and model inputs."""
    predictor = oracle if hasattr(oracle, "classes_") else PredictorAdapter(oracle)
    model_identity, model_fingerprint = _model_identity(predictor, target_model)
    factuals = select_factuals(dataset, max_test, test_selection, seed=seed)
    predictions = np.asarray(predictor.predict(factuals.values)).reshape(-1)
    if len(predictions) != len(factuals.values):
        raise ValueError("classifier returned the wrong number of factual predictions")
    classes = np.asarray(predictor.classes_)
    targets = _opposite_binary_labels(classes, predictions)
    protocol = {
        "max_test": max_test,
        "test_selection": test_selection,
        "selection_seed": seed,
        "target_policy": "opposite_classifier_prediction",
        "target_model": dict(target_model or {}),
        "resolved_target_model": model_identity["resolved_config"],
        "target_model_fingerprint": model_fingerprint,
        "target_model_implementation_fingerprint": model_identity[
            "implementation_fingerprint"
        ],
    }
    identity = {
        "dataset_fingerprint": dataset.provenance.fingerprint,
        "selection_inputs": {
            "max_test": max_test,
            "test_selection": test_selection,
            "selection_seed": seed,
            "test_pool_size": len(dataset.y_test),
            "test_labels": _array_digest(dataset.y_test),
        },
        "factual_indices": _array_digest(factuals.indices),
        "factual_values": _array_digest(factuals.values),
        "factual_true_labels": _array_digest(factuals.true_labels),
        "factual_predictions": _array_digest(predictions),
        "targets": _array_digest(targets),
        "classes": classes.tolist(),
        "target_policy": protocol["target_policy"],
        "target_model_fingerprint": model_fingerprint,
        "protocol": protocol,
    }
    case_id = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()
    return BenchmarkCase(
        case_id=case_id,
        dataset=dataset,
        factuals=factuals,
        oracle=predictor,
        factual_predictions=predictions,
        targets=targets,
        protocol=protocol,
    )


def prepare_benchmark_case(
    dataset: PreparedDataset,
    oracle_factory: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any],
    **case_options: Any,
) -> BenchmarkCase:
    """Train or load an oracle once and construct the reusable case."""
    evaluation_X = dataset.X_validation if len(dataset.X_validation) else dataset.X_test
    evaluation_y = dataset.y_validation if len(dataset.y_validation) else dataset.y_test
    oracle = oracle_factory(
        dataset.X_train, dataset.y_train, evaluation_X, evaluation_y
    )
    return build_benchmark_case(dataset, oracle, **case_options)


def method_context(case: BenchmarkCase) -> MethodContext:
    """Return the narrow method-owned view without truth labels or output state."""
    return MethodContext(
        X_reference=case.dataset.X_train,
        feature_schema=case.dataset.schema,
        oracle=case.oracle,
    )


def case_target_probabilities(case: BenchmarkCase, X: np.ndarray) -> np.ndarray:
    """Resolve case target labels through the oracle's declared class order."""
    return target_probabilities(case.oracle, X, case.targets)
