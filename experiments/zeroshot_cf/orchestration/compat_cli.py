"""Shared translation helpers for numbered v1 compatibility entry points."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.zeroshot_cf.evaluation import EvaluationSpec
from experiments.zeroshot_cf.orchestration.artifacts import ArtifactStore, StoredRun
from experiments.zeroshot_cf.orchestration.legacy import (
    aggregate_v1_metrics,
    generic_legacy_paths,
    write_result_table,
)
from experiments.zeroshot_cf.orchestration.runner import GenericRunner
from experiments.zeroshot_cf.orchestration.spec import (
    DatasetSpec,
    ExecutionSpec,
    MethodSpec,
    ProtocolSpec,
    RunSpec,
    TargetModelSpec,
)

COMPATIBILITY_SEED = 42
_TARGET_MODEL = TargetModelSpec(
    "retained_logistic_regression",
    {"C": 1.0, "max_iter": 1000, "seed": COMPATIBILITY_SEED},
)
_MANIFEST_ENVIRONMENT_KEYS = (
    "COUNTERCONTEX_SLURM_WALLTIME",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_CPUS_PER_TASK",
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "TABICL_DEVICE",
)


def compatibility_environment() -> dict[str, str]:
    """Capture selected operational settings without affecting run identity."""
    return {
        key: os.environ[key] for key in _MANIFEST_ENVIRONMENT_KEYS if key in os.environ
    }


def legacy_run_spec(
    dataset_name: str,
    method_name: str,
    *,
    method_params: Mapping[str, Any] | None = None,
    method_variant: str = "default",
    n_counterfactuals: int = 1,
    max_test: int | None,
    validation_fraction: float,
    drop_heloc_all_minus9: bool,
    probability_threshold: float,
    seed: int = COMPATIBILITY_SEED,
) -> RunSpec:
    """Translate one numbered-runner invocation into a scientific run spec."""
    return RunSpec(
        DatasetSpec(dataset_name),
        ProtocolSpec(
            max_test=None if max_test is None or max_test < 0 else max_test,
            test_selection="stratified",
            params={
                "validation_fraction": validation_fraction,
                "drop_heloc_all_minus9": drop_heloc_all_minus9,
            },
        ),
        _TARGET_MODEL,
        MethodSpec(
            method_name,
            variant=method_variant,
            params=dict(method_params or {}),
            n_counterfactuals=n_counterfactuals,
        ),
        EvaluationSpec(probability_threshold=probability_threshold),
        seed,
    )


def _coerce_legacy_scalar(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    lowered = value.lower()
    if lowered in {"nan", "+nan", "-nan"}:
        return math.nan
    if lowered in {"inf", "+inf"}:
        return math.inf
    if lowered == "-inf":
        return -math.inf
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_legacy_metrics(
    results_dir: Path,
    method_name: str,
    dataset_name: str,
) -> dict[str, Any]:
    path = generic_legacy_paths(results_dir, method_name, dataset_name).metrics_csv
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"legacy metrics file must contain one row: {path}")
    return {key: _coerce_legacy_scalar(value) for key, value in rows[0].items()}


def run_legacy_dataset_with_stored(
    spec: RunSpec,
    *,
    results_dir: Path,
    tabicl_cache_dir: Path | None = None,
) -> tuple[dict[str, Any], StoredRun]:
    """Run one translated spec and return its v1 row and exact canonical run."""
    cache_paths = {} if tabicl_cache_dir is None else {"tabicl": tabicl_cache_dir}
    runner = GenericRunner(
        ExecutionSpec(
            output_root=results_dir,
            cache_paths=cache_paths,
            environment=compatibility_environment(),
            legacy_export=True,
            resume=True,
        ),
        store=ArtifactStore(results_dir / "runs" / spec.cell_id),
    )
    outcome = runner.run(spec, resume=True)
    row = read_legacy_metrics(results_dir, spec.method.name, spec.dataset.name)
    timings = dict(outcome.stored.manifest.get("timings", {}))
    row.update(
        {
            "prepare_s": timings.get("prepare_s"),
            "generate_s": timings.get("generate_s"),
            "evaluate_s": timings.get("evaluate_s"),
            "write_s": timings.get("write_s"),
            "total_s": timings.get("total_s"),
        }
    )
    return row, outcome.stored


def run_legacy_dataset(
    spec: RunSpec,
    *,
    results_dir: Path,
    tabicl_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one translated spec and return its frozen v1 metrics row."""
    row, _ = run_legacy_dataset_with_stored(
        spec,
        results_dir=results_dir,
        tabicl_cache_dir=tabicl_cache_dir,
    )
    return row


def run_legacy_specs(
    specs: Sequence[RunSpec],
    *,
    results_dir: Path,
    aggregate_name: str,
    tabicl_cache_dir: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Execute a translated CLI suite and write canonical and v1 aggregates."""
    cache_paths = {} if tabicl_cache_dir is None else {"tabicl": tabicl_cache_dir}
    suite_id = hashlib.sha256(
        "\n".join(spec.cell_id for spec in specs).encode()
    ).hexdigest()
    runner = GenericRunner(
        ExecutionSpec(
            output_root=results_dir,
            cache_paths=cache_paths,
            environment=compatibility_environment(),
            legacy_export=True,
            resume=True,
        ),
        store=ArtifactStore(results_dir / "runs" / suite_id),
    )
    runner.run_all(specs, resume=True)
    runner.store.aggregate_expected(
        [spec.cell_id for spec in specs], output=results_dir / "aggregate.csv"
    )
    rows = tuple(
        read_legacy_metrics(results_dir, spec.method.name, spec.dataset.name)
        for spec in specs
    )
    write_result_table(results_dir / aggregate_name, rows)
    return rows


def aggregate_legacy_method(
    results_dir: Path,
    method_name: str,
    datasets: Sequence[str],
    output_name: str,
) -> Path:
    """Aggregate already-exported v1 metrics in declared dataset order."""
    return aggregate_v1_metrics(
        [
            (
                dataset,
                generic_legacy_paths(results_dir, method_name, dataset).metrics_csv,
            )
            for dataset in datasets
        ],
        results_dir / output_name,
    )
