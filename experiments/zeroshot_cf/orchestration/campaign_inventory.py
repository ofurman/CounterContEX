"""External integrity inventories for immutable canonical campaign runs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.zeroshot_cf.orchestration.artifacts import (
    _REQUIRED_FILES,
    ArtifactStore,
)
from experiments.zeroshot_cf.orchestration.matrix import (
    MATRIX_SCHEMA_VERSION,
    MatrixConfig,
    load_matrix_config,
)

CAMPAIGN_INVENTORY_SCHEMA_VERSION = "campaign-inventory-v1"
REQUIRED_RUN_FILES = _REQUIRED_FILES
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "matrix_identity",
        "matrix_identity_sha256",
        "resolved_runs_sha256",
        "runs",
    }
)
_RUN_KEYS = frozenset({"cell_id", "run_id", "files"})
_FILE_KEYS = frozenset({"name", "size_bytes", "sha256"})


@dataclass(frozen=True)
class ExpectedCampaignRun:
    """One matrix cell and its fully resolved content-addressed run ID."""

    cell_id: str
    run_id: str

    def __post_init__(self) -> None:
        for field, value in (("cell_id", self.cell_id), ("run_id", self.run_id)):
            if not value or value in {".", ".."} or Path(value).name != value:
                raise ValueError(f"{field} must be one non-empty safe path component")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("campaign inventory values must be strict JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _normalize_matrix_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("matrix_identity must be a mapping")
    encoded = _canonical_bytes(dict(value))
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded by Mapping
        raise TypeError("matrix_identity must be a mapping")
    return normalized


def _validate_expected_runs(
    expected_runs: Sequence[ExpectedCampaignRun],
) -> tuple[ExpectedCampaignRun, ...]:
    expected = tuple(expected_runs)
    if not expected:
        raise ValueError("campaign must declare at least one expected run")
    if not all(isinstance(item, ExpectedCampaignRun) for item in expected):
        raise TypeError("expected_runs must contain ExpectedCampaignRun values")
    cells = [item.cell_id for item in expected]
    run_ids = [item.run_id for item in expected]
    if len(set(cells)) != len(cells):
        raise ValueError("campaign contains duplicate expected cell identities")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("campaign contains duplicate expected run identities")
    return expected


def _strict_actual_runs(
    store: ArtifactStore,
    expected: tuple[ExpectedCampaignRun, ...],
) -> tuple[ExpectedCampaignRun, ...]:
    try:
        rows = store.aggregate_expected([item.cell_id for item in expected])
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"campaign run integrity validation failed: {error}"
        ) from error
    actual = tuple(
        ExpectedCampaignRun(cell_id=row["cell_id"], run_id=row["run_id"])
        for row in rows
    )
    if actual != expected:
        raise ValueError(
            "resolved campaign run membership does not match expected run IDs: "
            f"expected={expected!r}, actual={actual!r}"
        )
    return actual


def _run_identity_payload(
    runs: Sequence[ExpectedCampaignRun],
) -> list[dict[str, str]]:
    return [
        {"cell_id": item.cell_id, "run_id": item.run_id}
        for item in runs
    ]


def _file_records(
    store: ArtifactStore, run: ExpectedCampaignRun
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in REQUIRED_RUN_FILES:
        path = store.root / run.run_id / name
        if not path.is_file():
            raise ValueError(f"campaign run is missing required artifact file: {path}")
        size, digest = _sha256_file(path)
        records.append({"name": name, "size_bytes": size, "sha256": digest})
    return records


def _inventory_payload(
    *,
    store: ArtifactStore,
    matrix_identity: dict[str, Any],
    expected: tuple[ExpectedCampaignRun, ...],
) -> dict[str, Any]:
    run_identities = _run_identity_payload(expected)
    return {
        "schema_version": CAMPAIGN_INVENTORY_SCHEMA_VERSION,
        "matrix_identity": matrix_identity,
        "matrix_identity_sha256": _sha256_bytes(_canonical_bytes(matrix_identity)),
        "resolved_runs_sha256": _sha256_bytes(_canonical_bytes(run_identities)),
        "runs": [
            {
                "cell_id": run.cell_id,
                "run_id": run.run_id,
                "files": _file_records(store, run),
            }
            for run in expected
        ],
    }


def _require_external_destination(
    destination: Path,
    store: ArtifactStore,
    expected: Sequence[ExpectedCampaignRun],
) -> None:
    resolved_destination = destination.resolve()
    for run in expected:
        run_path = (store.root / run.run_id).resolve()
        if resolved_destination == run_path or run_path in resolved_destination.parents:
            raise ValueError(
                "campaign inventory must be published outside completed run directories"
            )


@contextmanager
def _inventory_lock(destination: Path):
    lock_path = destination.parent / f".{destination.name}.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def publish_campaign_inventory(
    *,
    store: ArtifactStore,
    expected_runs: Sequence[ExpectedCampaignRun],
    matrix_identity: Mapping[str, Any],
    destination: Path | str,
) -> Path:
    """Validate and atomically publish an immutable external campaign seal."""
    expected = _validate_expected_runs(expected_runs)
    destination_path = Path(destination)
    _require_external_destination(destination_path, store, expected)
    _strict_actual_runs(store, expected)
    normalized_matrix = _normalize_matrix_identity(matrix_identity)
    payload = _inventory_payload(
        store=store,
        matrix_identity=normalized_matrix,
        expected=expected,
    )
    encoded = _canonical_bytes(payload)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with _inventory_lock(destination_path):
        if destination_path.exists():
            raise FileExistsError(
                f"campaign inventory already exists: {destination_path}"
            )
        temporary = destination_path.parent / (
            f".{destination_path.name}.{uuid.uuid4().hex}.partial"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination_path)
        finally:
            temporary.unlink(missing_ok=True)
    return destination_path


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"campaign inventory contains duplicate key: {key}")
        result[key] = value
    return result


def _load_inventory(path: Path) -> dict[str, Any]:
    encoded = path.read_bytes()
    try:
        payload = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("campaign inventory is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("campaign inventory must contain a JSON object")
    if encoded != _canonical_bytes(payload):
        raise ValueError("campaign inventory is not in canonical form")
    return payload


def _validate_inventory_shape(payload: Mapping[str, Any]) -> None:
    if set(payload) != _TOP_LEVEL_KEYS:
        raise ValueError("campaign inventory has unexpected top-level fields")
    if payload.get("schema_version") != CAMPAIGN_INVENTORY_SCHEMA_VERSION:
        raise ValueError("unsupported campaign inventory schema version")
    if not isinstance(payload.get("matrix_identity"), dict):
        raise ValueError("campaign inventory matrix identity must be an object")
    for field in ("matrix_identity_sha256", "resolved_runs_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"campaign inventory has malformed {field}")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("campaign inventory must contain at least one run")
    seen_cells: set[str] = set()
    seen_runs: set[str] = set()
    for run in runs:
        if not isinstance(run, dict) or set(run) != _RUN_KEYS:
            raise ValueError("campaign inventory has malformed run records")
        cell_id = run.get("cell_id")
        run_id = run.get("run_id")
        if not isinstance(cell_id, str) or not isinstance(run_id, str):
            raise ValueError("campaign inventory run identities must be strings")
        if cell_id in seen_cells or run_id in seen_runs:
            raise ValueError("campaign inventory contains duplicate run identities")
        seen_cells.add(cell_id)
        seen_runs.add(run_id)
        files = run.get("files")
        if not isinstance(files, list) or len(files) != len(REQUIRED_RUN_FILES):
            raise ValueError("campaign inventory has malformed artifact file records")
        names = []
        for file_record in files:
            if not isinstance(file_record, dict) or set(file_record) != _FILE_KEYS:
                raise ValueError(
                    "campaign inventory has malformed artifact file records"
                )
            name = file_record.get("name")
            size = file_record.get("size_bytes")
            digest = file_record.get("sha256")
            if (
                not isinstance(name, str)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise ValueError(
                    "campaign inventory has malformed artifact file records"
                )
            names.append(name)
        if tuple(names) != REQUIRED_RUN_FILES:
            raise ValueError("campaign inventory artifact file set or order is invalid")


def verify_campaign_inventory(
    *,
    store: ArtifactStore,
    expected_runs: Sequence[ExpectedCampaignRun],
    matrix_identity: Mapping[str, Any],
    inventory_path: Path | str,
) -> tuple[ExpectedCampaignRun, ...]:
    """Verify inventory bytes, matrix identity, exact runs, and every payload hash."""
    expected = _validate_expected_runs(expected_runs)
    path = Path(inventory_path)
    _require_external_destination(path, store, expected)
    payload = _load_inventory(path)
    _validate_inventory_shape(payload)

    normalized_matrix = _normalize_matrix_identity(matrix_identity)
    stored_matrix = payload["matrix_identity"]
    stored_matrix_digest = _sha256_bytes(_canonical_bytes(stored_matrix))
    if payload["matrix_identity_sha256"] != stored_matrix_digest:
        raise ValueError("campaign inventory matrix identity hash does not match")
    if stored_matrix != normalized_matrix:
        raise ValueError(
            "campaign inventory does not match the expected matrix identity"
        )

    stored_identities = [
        {"cell_id": run["cell_id"], "run_id": run["run_id"]}
        for run in payload["runs"]
    ]
    expected_identities = _run_identity_payload(expected)
    if stored_identities != expected_identities:
        raise ValueError("campaign inventory resolved run membership does not match")
    if payload["resolved_runs_sha256"] != _sha256_bytes(
        _canonical_bytes(stored_identities)
    ):
        raise ValueError("campaign inventory resolved run identity hash does not match")

    _strict_actual_runs(store, expected)
    for run, stored_run in zip(expected, payload["runs"], strict=True):
        for stored_file in stored_run["files"]:
            artifact = store.root / run.run_id / stored_file["name"]
            if not artifact.is_file():
                raise ValueError(
                    f"campaign run is missing required artifact file: {artifact}"
                )
            size, digest = _sha256_file(artifact)
            if size != stored_file["size_bytes"] or digest != stored_file["sha256"]:
                raise ValueError(f"campaign artifact hash mismatch: {artifact}")
    return expected


def matrix_identity(config: MatrixConfig) -> dict[str, Any]:
    """Return the path-independent identity sealed for one resolved matrix."""
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "suite": config.suite,
        "source_sha256": _sha256_file(config.source)[1],
        "expected_cells": list(config.expected_cells),
    }


def _resolved_runs(
    store: ArtifactStore, config: MatrixConfig
) -> tuple[ExpectedCampaignRun, ...]:
    rows = store.aggregate_expected(config.expected_cells)
    return tuple(
        ExpectedCampaignRun(cell_id=row["cell_id"], run_id=row["run_id"])
        for row in rows
    )


def _default_inventory_path(config: MatrixConfig) -> Path:
    root = config.execution.output_root
    return root.parent / f"{root.name}.campaign-inventory.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("seal", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--matrix", required=True)
        command.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_matrix_config(args.matrix)
    store = ArtifactStore(config.execution.output_root)
    expected = _resolved_runs(store, config)
    identity = matrix_identity(config)
    destination = Path(args.output) if args.output else _default_inventory_path(config)
    if args.command == "seal":
        publish_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=identity,
            destination=destination,
        )
        print(f"sealed {len(expected)} runs into {destination}")
        return 0
    if args.command == "verify":
        verify_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=identity,
            inventory_path=destination,
        )
        print(f"verified {len(expected)} runs from {destination}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
