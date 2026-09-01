"""Manifest loading and Cartesian expansion into concrete run specs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from experiments.zeroshot_cf.evaluation import EvaluationSpec
from experiments.zeroshot_cf.orchestration.legacy import generic_legacy_paths
from experiments.zeroshot_cf.orchestration.spec import (
    DatasetSpec,
    ExecutionSpec,
    MethodSpec,
    ProtocolSpec,
    RunSpec,
    TargetModelSpec,
)

MATRIX_SCHEMA_VERSION = "countercontex.matrix.v1"


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"matrix {field} must be an integer")
    return value


@dataclass(frozen=True)
class MatrixConfig:
    suite: str
    runs: tuple[RunSpec, ...]
    execution: ExecutionSpec
    source: Path

    @property
    def expected_cells(self) -> tuple[str, ...]:
        return tuple(run.cell_id for run in self.runs)


def _named_spec(value: Any, *, kind: str) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str):
        return value, {}
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        raise ValueError(f"{kind} entries require a name")
    unknown = set(value) - {"name", "params", "variant", "n_counterfactuals"}
    if unknown:
        raise ValueError(f"unknown {kind} fields: {sorted(unknown)}")
    params = value.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{kind} params must be a mapping")
    return value["name"], dict(params)


def load_matrix_config(path: Path | str) -> MatrixConfig:
    source = Path(path)
    if source.suffix.lower() == ".toml":
        import tomllib

        payload = tomllib.loads(source.read_text())
    elif source.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(source.read_text())
    else:
        raise ValueError("matrix config must use .yaml, .yml, or .toml")
    if not isinstance(payload, dict):
        raise ValueError("matrix config must contain a mapping")
    if payload.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError("unsupported matrix schema version")
    allowed = {
        "schema_version",
        "suite",
        "output_root",
        "datasets",
        "methods",
        "seeds",
        "protocol",
        "target_model",
        "target_models",
        "evaluation",
        "legacy_export",
        "cache_paths",
        "device",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown matrix fields: {sorted(unknown)}")
    suite = payload.get("suite")
    if not isinstance(suite, str) or not suite:
        raise ValueError("matrix suite must be non-empty")
    datasets = payload.get("datasets")
    methods = payload.get("methods")
    seeds = payload.get("seeds")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("matrix datasets must be a non-empty list")
    if not isinstance(methods, list) or not methods:
        raise ValueError("matrix methods must be a non-empty list")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("matrix seeds must be a non-empty list")

    protocol_values = dict(payload.get("protocol", {}))
    protocol = ProtocolSpec(
        max_test=protocol_values.pop("max_test", 1000),
        test_selection=protocol_values.pop("test_selection", "stratified"),
        params=protocol_values,
    )
    if "target_model" in payload and "target_models" in payload:
        raise ValueError("matrix target_model and target_models are mutually exclusive")
    raw_target_models = payload.get("target_models")
    if raw_target_models is None:
        raw_target_models = [
            payload.get(
                "target_model", {"name": "retained_logistic_regression"}
            )
        ]
    if not isinstance(raw_target_models, list) or not raw_target_models:
        raise ValueError("matrix target_models must be a non-empty list")
    target_model_specs = []
    for value in raw_target_models:
        name, params = _named_spec(value, kind="target model")
        target_model_specs.append(TargetModelSpec(name, params))
    evaluation = EvaluationSpec(**dict(payload.get("evaluation", {})))

    dataset_specs = []
    for value in datasets:
        name, params = _named_spec(value, kind="dataset")
        dataset_specs.append(DatasetSpec(name, params))
    method_specs = []
    for value in methods:
        name, params = _named_spec(value, kind="method")
        mapping = value if isinstance(value, dict) else {}
        method_specs.append(
            MethodSpec(
                name=name,
                variant=str(mapping.get("variant", "default")),
                params=params,
                n_counterfactuals=_integer(
                    mapping.get("n_counterfactuals", 1),
                    field="method n_counterfactuals",
                ),
            )
        )
    runs = tuple(
        RunSpec(
            dataset,
            protocol,
            target_model,
            method,
            evaluation,
            _integer(seed, field="seed"),
        )
        for dataset, target_model, method, seed in product(
            dataset_specs, target_model_specs, method_specs, seeds
        )
    )
    if len({run.cell_id for run in runs}) != len(runs):
        raise ValueError("matrix expands duplicate scientific cells")
    output_root = payload.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("matrix output_root must be non-empty")
    execution = ExecutionSpec(
        output_root=Path(output_root),
        cache_paths={
            key: Path(value)
            for key, value in dict(payload.get("cache_paths", {})).items()
        },
        device=payload.get("device"),
        legacy_export=bool(payload.get("legacy_export", False)),
    )
    if execution.legacy_export:
        destinations = [
            generic_legacy_paths(
                execution.output_root,
                run.method.name,
                run.dataset.name,
            ).metrics_csv
            for run in runs
        ]
        if len(set(destinations)) != len(destinations):
            raise ValueError(
                "legacy_export requires at most one run per method and dataset"
            )
    return MatrixConfig(suite=suite, runs=runs, execution=execution, source=source)
