"""Serializable scientific and execution specifications for generic runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from experiments.zeroshot_cf.evaluation import EvaluationSpec

_EXECUTION_ONLY_METHOD_KEYS = frozenset(
    {
        "cache_dir",
        "cache_path",
        "cache_paths",
        "checkpoint_path",
        "device",
        "environment",
        "host",
        "legacy_export",
        "local_checkpoint_path",
        "model_path",
        "output_root",
        "resume",
    }
)


def _execution_only_param_paths(
    value: Any,
    *,
    prefix: str = "params",
) -> tuple[str, ...]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if key in _EXECUTION_ONLY_METHOD_KEYS:
                paths.append(path)
            paths.extend(_execution_only_param_paths(item, prefix=path))
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            paths.extend(_execution_only_param_paths(item, prefix=f"{prefix}[{index}]"))
    return tuple(paths)


def _frozen_params(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _reject_execution_only_params(values: Mapping[str, Any], *, kind: str) -> None:
    execution_paths = _execution_only_param_paths(values)
    if execution_paths:
        raise ValueError(
            f"{kind} params contain execution-only settings: {list(execution_paths)}"
        )


def canonical_json(value: Mapping[str, Any]) -> str:
    """Encode one identity payload independently of mapping field order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset name must be non-empty")
        _reject_execution_only_params(self.params, kind="dataset")
        object.__setattr__(self, "params", _frozen_params(self.params))


@dataclass(frozen=True)
class ProtocolSpec:
    max_test: int | None = 1000
    test_selection: str = "stratified"
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_test is not None and (
            isinstance(self.max_test, bool) or not isinstance(self.max_test, int)
        ):
            raise TypeError("max_test must be an integer or null")
        if self.max_test is not None and self.max_test <= 0:
            raise ValueError("max_test must be positive or null")
        if self.test_selection not in {"first", "stratified"}:
            raise ValueError("test_selection must be first or stratified")
        _reject_execution_only_params(self.params, kind="protocol")
        object.__setattr__(self, "params", _frozen_params(self.params))


@dataclass(frozen=True)
class TargetModelSpec:
    name: str = "retained_logistic_regression"
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("target model name must be non-empty")
        _reject_execution_only_params(self.params, kind="target model")
        object.__setattr__(self, "params", _frozen_params(self.params))


@dataclass(frozen=True)
class MethodSpec:
    name: str
    variant: str = "default"
    params: Mapping[str, Any] = field(default_factory=dict)
    n_counterfactuals: int = 1

    def __post_init__(self) -> None:
        if not self.name or not self.variant:
            raise ValueError("method name and variant must be non-empty")
        if isinstance(self.n_counterfactuals, bool) or not isinstance(
            self.n_counterfactuals, int
        ):
            raise TypeError("n_counterfactuals must be an integer")
        if self.n_counterfactuals <= 0:
            raise ValueError("n_counterfactuals must be positive")
        _reject_execution_only_params(self.params, kind="method")
        object.__setattr__(self, "params", _frozen_params(self.params))


@dataclass(frozen=True)
class RunSpec:
    dataset: DatasetSpec
    protocol: ProtocolSpec
    target_model: TargetModelSpec
    method: MethodSpec
    evaluation: EvaluationSpec
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "dataset": {"name": self.dataset.name, "params": dict(self.dataset.params)},
            "protocol": {
                "max_test": self.protocol.max_test,
                "test_selection": self.protocol.test_selection,
                "params": dict(self.protocol.params),
            },
            "target_model": {
                "name": self.target_model.name,
                "params": dict(self.target_model.params),
            },
            "method": {
                "name": self.method.name,
                "variant": self.method.variant,
                "params": dict(self.method.params),
                "n_counterfactuals": self.method.n_counterfactuals,
            },
            "evaluation": asdict(self.evaluation),
            "seed": self.seed,
        }

    @property
    def cell_id(self) -> str:
        return hashlib.sha256(
            canonical_json(self.scientific_payload()).encode()
        ).hexdigest()


@dataclass(frozen=True)
class IdentityVersions:
    dataset_fingerprint: str
    case_fingerprint: str
    method_implementation: str
    backend_implementation: str
    model_content_id: str
    checkpoint_content_ids: Mapping[str, str] = field(default_factory=dict)
    evaluation_version: str = "countercontex.evaluation.v2"
    artifact_schema_version: str = "countercontex.artifacts.v1"

    def __post_init__(self) -> None:
        required = (
            self.dataset_fingerprint,
            self.case_fingerprint,
            self.method_implementation,
            self.backend_implementation,
            self.model_content_id,
            self.evaluation_version,
            self.artifact_schema_version,
        )
        if any(not value for value in required):
            raise ValueError("identity version fields must be non-empty")
        object.__setattr__(
            self,
            "checkpoint_content_ids",
            MappingProxyType(dict(self.checkpoint_content_ids)),
        )


def identity_payload(spec: RunSpec, versions: IdentityVersions) -> dict[str, Any]:
    return {
        "scientific_spec": spec.scientific_payload(),
        "resolved": {
            "dataset_fingerprint": versions.dataset_fingerprint,
            "case_fingerprint": versions.case_fingerprint,
            "method_implementation": versions.method_implementation,
            "backend_implementation": versions.backend_implementation,
            "model_content_id": versions.model_content_id,
            "checkpoint_content_ids": dict(versions.checkpoint_content_ids),
            "evaluation_version": versions.evaluation_version,
            "artifact_schema_version": versions.artifact_schema_version,
        },
    }


def run_id(spec: RunSpec, versions: IdentityVersions) -> str:
    return hashlib.sha256(
        canonical_json(identity_payload(spec, versions)).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ExecutionSpec:
    output_root: Path
    resume: bool = False
    cache_paths: Mapping[str, Path] = field(default_factory=dict)
    device: str | None = None
    host: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    legacy_export: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(
            self,
            "cache_paths",
            MappingProxyType(
                {key: Path(value) for key, value in self.cache_paths.items()}
            ),
        )
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "resume": self.resume,
            "cache_paths": {key: str(value) for key, value in self.cache_paths.items()},
            "device": self.device,
            "host": self.host,
            "environment": dict(self.environment),
            "legacy_export": self.legacy_export,
        }
