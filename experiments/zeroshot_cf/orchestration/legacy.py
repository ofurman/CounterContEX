"""Frozen v1 CSV/NPZ compatibility exporter."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def write_result_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write legacy tables with stable first-seen column ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty result table")
    columns: list[str] = []
    normalized_rows = [dict(row) for row in rows]
    for row in normalized_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(normalized_rows)
    print(f"Wrote {path}")


def write_v1_dataset_outputs(
    metrics_csv: Path,
    points_csv: Path,
    arrays_npz: Path,
    metrics_row: Mapping[str, Any],
    point_rows: Sequence[Mapping[str, Any]],
    *,
    arrays: Mapping[str, Any] | None = None,
) -> None:
    """Preserve exact legacy paths, row columns, and caller-provided NPZ keys."""
    write_result_table(metrics_csv, [metrics_row])
    write_result_table(points_csv, point_rows)
    if arrays is not None:
        arrays_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays_npz, **arrays)
        print(f"Wrote {arrays_npz}")


def aggregate_v1_metrics(
    paths: Sequence[tuple[str, Path]],
    output: Path,
) -> Path:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset, path in paths:
        if not path.exists():
            missing.append(dataset)
            continue
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if missing:
        raise FileNotFoundError(
            "Missing per-dataset benchmark results for: " + ", ".join(missing)
        )
    write_result_table(output, rows)
    return output
