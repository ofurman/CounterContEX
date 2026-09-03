"""Strict loading and multi-seed aggregation of canonical run artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.orchestration.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactStore,
    _read_csv,
    _validate_table_types,
)
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config
from experiments.zeroshot_cf.orchestration.spec import canonical_json

_NON_METRICS = frozenset(
    {
        "run_id",
        "cell_id",
        "dataset",
        "target_model",
        "method",
        "method_variant",
        "n_counterfactuals",
        "backend",
        "seed",
        "scientific_group",
        "artifact_path",
    }
)
_LEGACY_EVALUATION_VERSION = "countercontex.evaluation.v1"


def derive_cell_metrics(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Return the cell with analysis-only derived metrics appended."""
    derived = dict(cell)
    spread = cell.get("set_pairwise_gower_mean")
    proximity = cell.get("proximity_grouped_gower")
    if (
        spread is not None
        and proximity is not None
        and np.isfinite(spread)
        and np.isfinite(proximity)
        and proximity > 0
    ):
        derived["set_pairwise_gower_ratio"] = float(spread) / float(proximity)
    else:
        derived["set_pairwise_gower_ratio"] = None
    return derived


def load_published_cells(
    output_root: Path | str, matrix_config: Path | str
) -> tuple[dict[str, Any], ...]:
    """Read exactly the matrix-declared COMPLETE runs and no survivors-only subset."""
    root = Path(output_root)
    config = load_matrix_config(Path(matrix_config))
    published_versions = {
        json.loads(path.read_text()).get("evaluation_schema_version")
        for path in root.glob("*/manifest.json")
        if (path.parent / "COMPLETE").is_file()
    }
    if len(published_versions) > 1:
        raise ValueError(
            f"mixed evaluation schema versions are not aggregateable: "
            f"{sorted(str(version) for version in published_versions)}"
        )
    configured_version = config.runs[0].evaluation.metric_version
    if published_versions == {_LEGACY_EVALUATION_VERSION}:
        return _load_legacy_cells(root, config)
    if published_versions and published_versions != {configured_version}:
        raise ValueError(
            f"unsupported evaluation schema version: {next(iter(published_versions))}"
        )
    store = ArtifactStore(root)
    summary_rows = store.aggregate_expected(config.expected_cells)
    stored_by_cell = {run.manifest["cell_id"]: run for run in store.completed_runs()}
    rows: list[dict[str, Any]] = []
    for summary in summary_rows:
        stored = stored_by_cell[summary["cell_id"]]
        scientific = stored.manifest["scientific_spec"]
        group_identity = dict(scientific)
        group_identity.pop("seed")
        rows.append(
            derive_cell_metrics(
                {
                    **summary,
                    "target_model": scientific["target_model"]["name"],
                    "scientific_group": canonical_json(group_identity),
                    "artifact_path": str(stored.path),
                    **{
                        f"timing_{name}": value
                        for name, value in stored.manifest.get("timings", {}).items()
                    },
                }
            )
        )
    return tuple(rows)


def _legacy_comparable_scientific(scientific: Mapping[str, Any]) -> str:
    """Normalize only evaluation fields introduced by the v1-to-v2 schema change."""
    comparable = dict(scientific)
    evaluation = dict(comparable.get("evaluation", {}))
    for name in (
        "metric_version",
        "detectability_min_cf_rows",
        "gower_neighbor_k",
    ):
        evaluation.pop(name, None)
    comparable["evaluation"] = evaluation
    return canonical_json(comparable)


def _load_legacy_cells(root: Path, config: Any) -> tuple[dict[str, Any], ...]:
    """Read historical schemas without changing or re-evaluating their values."""
    partial = [
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and not (path / "COMPLETE").is_file()
    ]
    if partial:
        raise ValueError(f"partial run directories are not aggregateable: {partial}")
    expected = {
        _legacy_comparable_scientific(run.scientific_payload()): run
        for run in config.runs
    }
    if len(expected) != len(config.runs):
        raise ValueError("matrix cells collide when matching a historical schema")
    found: dict[str, dict[str, Any]] = {}
    required = {
        "manifest.json",
        "summary.csv",
        "points.csv",
        "candidates.csv",
        "arrays.npz",
        "COMPLETE",
    }
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        missing = sorted(name for name in required if not (path / name).is_file())
        if missing:
            raise ValueError(f"run is missing required artifact files: {missing}")
        payload = json.loads((path / "manifest.json").read_text())
        if payload.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported artifact schema version")
        if payload.get("evaluation_schema_version") != _LEGACY_EVALUATION_VERSION:
            raise ValueError("unsupported evaluation schema version")
        config_payload = payload.get("config")
        if not isinstance(config_payload, dict):
            raise ValueError("manifest is missing config")
        identity = config_payload.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("run manifest is missing matrix identity")
        run_id = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        if (
            run_id != path.name
            or payload.get("run_id") != run_id
            or config_payload.get("run_id") != run_id
        ):
            raise ValueError("run manifest identity does not match its run_id")
        scientific = identity.get("scientific_spec")
        if not isinstance(scientific, dict):
            raise ValueError("run manifest identity is missing scientific_spec")
        if (
            scientific.get("evaluation", {}).get("metric_version")
            != _LEGACY_EVALUATION_VERSION
        ):
            raise ValueError("unsupported scientific evaluation metric version")
        cell_id = hashlib.sha256(canonical_json(scientific).encode()).hexdigest()
        if (
            config_payload.get("cell_id") != cell_id
            or config_payload.get("scientific_spec") != scientific
        ):
            raise ValueError("run manifest scientific identity does not match cell_id")
        key = _legacy_comparable_scientific(scientific)
        if key in found:
            raise ValueError("duplicate completed matrix cell")
        types = _validate_table_types(payload.get("table_types"))
        summaries = _read_csv(path / "summary.csv", types["summary"])
        if len(summaries) != 1:
            raise ValueError("summary.csv must contain exactly one row")
        method = scientific["method"]
        found[key] = {
            "run_id": run_id,
            "cell_id": cell_id,
            "dataset": scientific["dataset"]["name"],
            "target_model": scientific["target_model"]["name"],
            "method": method["name"],
            "method_variant": method["variant"],
            "n_counterfactuals": method["n_counterfactuals"],
            "backend": config_payload.get("resolved_method_config", {})
            .get("foundation", {})
            .get("backend"),
            "seed": scientific["seed"],
            "scientific_group": canonical_json(
                {name: value for name, value in scientific.items() if name != "seed"}
            ),
            "artifact_path": str(path),
            **summaries[0],
            **{
                f"timing_{name}": value
                for name, value in config_payload.get("timings", {}).items()
            },
        }
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    if missing or extra:
        raise ValueError(f"matrix cell mismatch: missing={missing}, extra={extra}")
    return tuple(found[key] for key in expected)


def aggregate_seeds(
    output_root: Path | str, matrix_config: Path | str
) -> tuple[dict[str, Any], ...]:
    """Emit mean, sample standard deviation, and actual n by identity minus seed."""
    cells = load_published_cells(output_root, matrix_config)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[cell["scientific_group"]].append(cell)
    aggregated: list[dict[str, Any]] = []
    for group_key, members in sorted(grouped.items()):
        identity = json.loads(group_key)
        row: dict[str, Any] = {
            "dataset": identity["dataset"]["name"],
            "target_model": identity["target_model"]["name"],
            "method": identity["method"]["name"],
            "method_variant": identity["method"]["variant"],
            "n_counterfactuals": identity["method"]["n_counterfactuals"],
            "backend": identity["method"].get("params", {})
            .get("foundation", {})
            .get("backend"),
            "seed_n": len(members),
        }
        metric_names = sorted(set.intersection(*(set(item) for item in members)))
        for name in metric_names:
            if name in _NON_METRICS or name.startswith("timing_"):
                continue
            values = [item[name] for item in members]
            if not all(
                value is None
                or isinstance(value, int | float | np.integer | np.floating)
                and not isinstance(value, bool | np.bool_)
                for value in values
            ):
                continue
            finite = np.asarray(
                [float(value) for value in values if value is not None], dtype=float
            )
            finite = finite[np.isfinite(finite)]
            row[f"{name}_n"] = int(len(finite))
            row[f"{name}_mean"] = float(np.mean(finite)) if len(finite) else None
            row[f"{name}_std"] = (
                float(np.std(finite, ddof=1))
                if len(finite) > 1
                else 0.0
                if len(finite) == 1
                else None
            )
        aggregated.append(row)
    return tuple(aggregated)


def write_rows(path: Path, rows: tuple[Mapping[str, Any], ...]) -> None:
    """Write auditable analysis rows with stable columns."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
