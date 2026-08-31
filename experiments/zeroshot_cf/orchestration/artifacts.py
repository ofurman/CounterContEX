"""Atomic manifest-backed persistence for versioned evaluator outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.evaluation.result import (
    METRIC_SCHEMA_VERSION,
    ArrayOutput,
    CandidateOutput,
    EvaluationReport,
    PointOutput,
    SummaryOutput,
)

ARTIFACT_SCHEMA_VERSION = "countercontex.artifacts.v1"
_REQUIRED_FILES = (
    "manifest.json",
    "summary.csv",
    "points.csv",
    "candidates.csv",
    "arrays.npz",
    "COMPLETE",
)
_TABLE_KEYS = ("summary", "points", "candidates")
_SCALAR_TYPES = frozenset({"null", "bool", "int", "float", "str"})


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, frozenset | set):
        return sorted(_json_value(item) for item in value)
    return value


def _scalar_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    if isinstance(value, str):
        return "str"
    raise TypeError(f"unsupported table scalar type: {type(value).__name__}")


def _encode_scalar(value: Any) -> str:
    """Encode table cells without conflating the empty string and null."""
    return json.dumps(_json_value(value), ensure_ascii=False, allow_nan=True)


def _decode_scalar(value: str, kind: str) -> Any:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed encoded table scalar: {value!r}") from error
    if decoded is None:
        return None
    if kind == "null":
        raise ValueError("non-null value found in a null-only table column")
    if kind == "bool" and isinstance(decoded, bool):
        return decoded
    if kind == "int" and isinstance(decoded, int) and not isinstance(decoded, bool):
        return decoded
    if (
        kind == "float"
        and isinstance(decoded, int | float)
        and not isinstance(decoded, bool)
    ):
        return float(decoded)
    if kind == "str" and isinstance(decoded, str):
        return decoded
    raise ValueError(f"table value does not match declared {kind!r} scalar type")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    required_columns: Sequence[str] = (),
) -> dict[str, str]:
    columns: list[str] = list(required_columns)
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    types: dict[str, str] = {}
    for name in columns:
        observed = {
            _scalar_type(row.get(name)) for row in rows if row.get(name) is not None
        }
        if not observed:
            types[name] = "null"
        elif len(observed) == 1:
            types[name] = observed.pop()
        elif observed <= {"int", "float"}:
            types[name] = "float"
        else:
            raise TypeError(
                f"column {name!r} has incompatible scalar types: {observed}"
            )
    for name in required_columns:
        types[name] = "int"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {name: _encode_scalar(row.get(name)) for name in columns} for row in rows
        )
    return types


def _read_csv(path: Path, types: Mapping[str, str]) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != set(types):
            raise ValueError(
                f"table columns do not match manifest schema for {path.name}"
            )
        return [
            {name: _decode_scalar(value, types[name]) for name, value in row.items()}
            for row in reader
        ]


def _validate_table_types(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != set(_TABLE_KEYS):
        raise ValueError("manifest has malformed table type schemas")
    validated: dict[str, dict[str, str]] = {}
    for table in _TABLE_KEYS:
        schema = value[table]
        if not isinstance(schema, dict) or any(
            not isinstance(name, str)
            or not isinstance(kind, str)
            or kind not in _SCALAR_TYPES
            for name, kind in schema.items()
        ):
            raise ValueError("manifest has malformed table type schemas")
        validated[table] = dict(schema)
    if validated["points"].get("point") != "int":
        raise ValueError("manifest has malformed points table type schema")
    if (
        validated["candidates"].get("point") != "int"
        or validated["candidates"].get("rank") != "int"
    ):
        raise ValueError("manifest has malformed candidates table type schema")
    return validated


def _validate_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"artifact array {name!r} must not use object dtype")
    return array


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    manifest: Mapping[str, Any]
    report: EvaluationReport
    path: Path


class ArtifactStore:
    """Read and atomically write complete run directories."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def write(
        self,
        run_id: str,
        *,
        manifest: Mapping[str, Any],
        report: EvaluationReport,
        manifest_finalizer: Callable[[Mapping[str, Any], float], Mapping[str, Any]]
        | None = None,
    ) -> StoredRun:
        if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
            raise ValueError("run_id must be one safe path component")
        if report.schema_version != METRIC_SCHEMA_VERSION:
            raise ValueError("unsupported evaluation schema version")
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / run_id
        temporary = self.root / f".{run_id}.{uuid.uuid4().hex}.partial"
        temporary.mkdir()
        published = False
        write_started = time.perf_counter()
        try:
            summary_rows = [dict(report.summary.values)]
            point_rows = [{"point": row.point, **row.values} for row in report.points]
            candidate_rows = [
                {"point": row.point, "rank": row.rank, **row.values}
                for row in report.candidates
            ]
            table_types = {
                "summary": _write_csv(temporary / "summary.csv", summary_rows),
                "points": _write_csv(
                    temporary / "points.csv", point_rows, required_columns=("point",)
                ),
                "candidates": _write_csv(
                    temporary / "candidates.csv",
                    candidate_rows,
                    required_columns=("point", "rank"),
                ),
            }
            arrays = {
                name: _validate_array(name, value)
                for name, value in report.arrays.values.items()
            }
            np.savez_compressed(temporary / "arrays.npz", **arrays)
            payload_write_s = time.perf_counter() - write_started
            finalized_manifest = (
                manifest
                if manifest_finalizer is None
                else manifest_finalizer(manifest, payload_write_s)
            )
            payload = {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "evaluation_schema_version": report.schema_version,
                "run_id": run_id,
                "config": _json_value(finalized_manifest),
                "report_metadata": _json_value(report.metadata),
                "table_types": table_types,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            if destination.exists():
                if (destination / "COMPLETE").exists():
                    raise FileExistsError(
                        f"completed run already exists: {destination}"
                    )
                shutil.rmtree(destination)
            os.replace(temporary, destination)
            published = True
            marker_temporary = destination / f".COMPLETE.{uuid.uuid4().hex}.partial"
            marker_temporary.write_text("complete\n")
            os.replace(marker_temporary, destination / "COMPLETE")
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            if published and not (destination / "COMPLETE").exists():
                shutil.rmtree(destination, ignore_errors=True)
            raise
        return self.read(run_id)

    def read(self, run_id: str) -> StoredRun:
        path = self.root / run_id
        if not (path / "COMPLETE").is_file():
            raise FileNotFoundError(f"run is incomplete or absent: {run_id}")
        missing = [name for name in _REQUIRED_FILES if not (path / name).is_file()]
        if missing:
            raise ValueError(f"run is missing required artifact files: {missing}")
        manifest_path = path / "manifest.json"
        payload = json.loads(manifest_path.read_text())
        if payload.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported artifact schema version")
        if payload.get("evaluation_schema_version") != METRIC_SCHEMA_VERSION:
            raise ValueError("unsupported evaluation schema version")
        if payload.get("run_id") != run_id:
            raise ValueError("manifest run_id does not match its directory")
        if "config" not in payload:
            raise ValueError("manifest is missing config")
        types = _validate_table_types(payload.get("table_types"))
        summary_rows = _read_csv(path / "summary.csv", types["summary"])
        if len(summary_rows) != 1:
            raise ValueError("summary.csv must contain exactly one row")
        point_rows = _read_csv(path / "points.csv", types["points"])
        candidate_rows = _read_csv(path / "candidates.csv", types["candidates"])
        try:
            with np.load(path / "arrays.npz", allow_pickle=False) as archive:
                arrays = {
                    name: _validate_array(name, archive[name]) for name in archive.files
                }
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                "arrays.npz is malformed or contains object arrays"
            ) from error
        report = EvaluationReport(
            schema_version=METRIC_SCHEMA_VERSION,
            summary=SummaryOutput(METRIC_SCHEMA_VERSION, summary_rows[0]),
            points=tuple(
                PointOutput(
                    point=int(row.pop("point")),
                    values=row,
                )
                for row in point_rows
            ),
            candidates=tuple(
                CandidateOutput(
                    point=int(row.pop("point")),
                    rank=int(row.pop("rank")),
                    values=row,
                )
                for row in candidate_rows
            ),
            arrays=ArrayOutput(METRIC_SCHEMA_VERSION, arrays),
            metadata=payload.get("report_metadata", {}),
        )
        return StoredRun(
            run_id=run_id,
            manifest=payload["config"],
            report=report,
            path=path,
        )

    def completed_runs(self) -> tuple[StoredRun, ...]:
        """Return valid completed runs, ignoring partial directories."""
        if not self.root.exists():
            return ()
        runs: list[StoredRun] = []
        for path in sorted(self.root.iterdir()):
            if (
                path.is_dir()
                and not path.name.startswith(".")
                and (path / "COMPLETE").is_file()
            ):
                runs.append(self.read(path.name))
        return tuple(runs)

    def aggregate_summary(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(run.report.summary.values) for run in self.completed_runs())

    def aggregate_expected(
        self,
        expected_cells: Sequence[str],
        *,
        output: Path | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Aggregate exactly the declared complete cells from validated manifests."""
        from experiments.zeroshot_cf.orchestration.spec import canonical_json

        expected = tuple(expected_cells)
        if len(set(expected)) != len(expected):
            raise ValueError("expected matrix contains duplicate cell identities")
        if self.root.exists():
            partial = [
                path.name
                for path in self.root.iterdir()
                if path.is_dir()
                and not path.name.startswith(".")
                and not (path / "COMPLETE").is_file()
            ]
            if partial:
                raise ValueError(
                    f"partial run directories are not aggregateable: {partial}"
                )
        runs = self.completed_runs()
        by_cell: dict[str, StoredRun] = {}
        for stored in runs:
            cell_id = stored.manifest.get("cell_id")
            identity = stored.manifest.get("identity")
            if not isinstance(cell_id, str) or not isinstance(identity, dict):
                raise ValueError("run manifest is missing matrix identity")
            derived = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
            if (
                derived != stored.run_id
                or stored.manifest.get("run_id") != stored.run_id
            ):
                raise ValueError("run manifest identity does not match its run_id")
            identity_scientific = identity.get("scientific_spec")
            manifest_scientific = stored.manifest.get("scientific_spec")
            if not isinstance(identity_scientific, dict):
                raise ValueError("run manifest identity is missing scientific_spec")
            derived_cell = hashlib.sha256(
                canonical_json(identity_scientific).encode()
            ).hexdigest()
            if derived_cell != cell_id:
                raise ValueError(
                    "run manifest scientific identity does not match cell_id"
                )
            if manifest_scientific != identity_scientific:
                raise ValueError(
                    "run manifest scientific_spec does not match identity"
                )
            if cell_id in by_cell:
                raise ValueError(f"duplicate completed matrix cell: {cell_id}")
            by_cell[cell_id] = stored
        missing = sorted(set(expected) - set(by_cell))
        extra = sorted(set(by_cell) - set(expected))
        if missing or extra:
            raise ValueError(f"matrix cell mismatch: missing={missing}, extra={extra}")
        rows = tuple(
            {
                "run_id": by_cell[cell].run_id,
                "cell_id": cell,
                "dataset": by_cell[cell].manifest["scientific_spec"]["dataset"][
                    "name"
                ],
                "method": by_cell[cell].manifest["scientific_spec"]["method"][
                    "name"
                ],
                "method_variant": by_cell[cell].manifest["scientific_spec"][
                    "method"
                ]["variant"],
                "n_counterfactuals": by_cell[cell].manifest["scientific_spec"][
                    "method"
                ]["n_counterfactuals"],
                "backend": by_cell[cell]
                .manifest.get("resolved_method_config", {})
                .get("foundation", {})
                .get(
                    "backend",
                    by_cell[cell].manifest["scientific_spec"]["method"]
                    .get("params", {})
                    .get("foundation", {})
                    .get("backend"),
                ),
                "seed": by_cell[cell].manifest["scientific_spec"]["seed"],
                **dict(by_cell[cell].report.summary.values),
            }
            for cell in expected
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_csv(output, rows)
        return rows
