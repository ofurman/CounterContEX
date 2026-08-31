"""Frozen v1 CSV/NPZ compatibility exporter."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.core.contracts import BenchmarkCase
from experiments.zeroshot_cf.evaluation import EvaluationReport
from experiments.zeroshot_cf.orchestration.v1_contract import V1_CONTRACT

LEGACY_COMPATIBILITY_DIRECTORY = "."


@dataclass(frozen=True)
class GenericLegacyPaths:
    metrics_csv: Path
    points_csv: Path
    arrays_npz: Path
    required_npz_keys: tuple[str, ...]
    summary_columns: tuple[str, ...]
    point_columns: tuple[str, ...]


def generic_legacy_paths(
    output_root: Path,
    method_name: str,
    dataset_name: str,
) -> GenericLegacyPaths:
    try:
        contract = V1_CONTRACT[method_name]
    except KeyError as error:
        raise ValueError(f"unsupported legacy method: {method_name}") from error
    stem = contract["stem"]
    keys = contract["npz_keys"]
    root = Path(output_root)
    prefix = root / f"{stem}_{dataset_name}"
    return GenericLegacyPaths(
        metrics_csv=prefix.with_name(f"{prefix.name}_metrics.csv"),
        points_csv=prefix.with_name(f"{prefix.name}_points.csv"),
        arrays_npz=prefix.with_name(f"{prefix.name}_arrays.npz"),
        required_npz_keys=keys,
        summary_columns=contract["summary_columns"],
        point_columns=contract["point_columns"],
    )


def _legacy_primary(report: EvaluationReport) -> np.ndarray:
    arrays = report.arrays.values
    candidates = np.asarray(arrays["common.candidates"])
    available = np.asarray(arrays["common.available"], dtype=bool)
    rank = int(report.metadata.get("primary_rank", 0))
    primary = candidates[:, rank].copy()
    missing = ~available[:, rank]
    if missing.any():
        best_effort = arrays.get("method.best_effort")
        if best_effort is None or np.asarray(best_effort).shape != primary.shape:
            raise ValueError("legacy export requires best effort rows for failures")
        primary[missing] = np.asarray(best_effort)[missing]
    return primary


def export_generic_v1(
    output_root: Path,
    *,
    dataset_name: str,
    method_name: str,
    case: BenchmarkCase,
    report: EvaluationReport,
    point_diagnostics: Sequence[Mapping[str, Any]] = (),
    manifest: Mapping[str, Any] | None = None,
) -> GenericLegacyPaths:
    """Write the frozen per-dataset v1 surface from canonical run artifacts."""
    paths = generic_legacy_paths(output_root, method_name, dataset_name)
    contract = V1_CONTRACT[method_name]
    legacy_method_id = contract["method_id"]
    values = report.arrays.values
    X_cf = _legacy_primary(report)
    y_cf_pred = np.asarray(case.oracle.predict(X_cf)).reshape(-1)
    valid = y_cf_pred == case.targets
    summary = dict(report.summary.values)
    timings = dict((manifest or {}).get("timings", {}))
    valid_rows = X_cf[valid]
    valid_factuals = case.factuals.values[valid]
    metrics_values = {
        "dataset": dataset_name,
        "method": legacy_method_id,
        "split_variant": case.dataset.provenance.split_id,
        "split_seed": 42,
        "test_selection": case.protocol.get("test_selection"),
        "n_train": len(case.dataset.X_train),
        "n_validation": len(case.dataset.X_validation),
        "n_test_pool": len(case.dataset.X_test),
        "n_test": len(case.factuals.values),
        "cf_per_factual": np.asarray(values["common.candidates"]).shape[1],
        "target_classifier_validation_accuracy": float(
            np.mean(
                case.oracle.predict(case.dataset.X_validation)
                == case.dataset.y_validation
            )
        ),
        "target_classifier_test_accuracy": float(
            np.mean(case.factual_predictions == case.factuals.true_labels)
        ),
        "preprocessing_variant": case.dataset.provenance.preprocessing_id,
        "n_dropped_rows": 0,
        "runtime_generation_s": timings.get("generate_s"),
        **summary,
        "validity": float(valid.mean()),
        "true_actionability": summary.get("actionability"),
        "proximity_all_features_euclidean": (
            float(np.linalg.norm(valid_rows - valid_factuals, axis=1).mean())
            if len(valid_rows)
            else float("nan")
        ),
        "failure_rate": float((~valid).mean()),
        "l0_count_mean": (
            float(np.count_nonzero(valid_rows != valid_factuals, axis=1).mean())
            if len(valid_rows)
            else float("nan")
        ),
        "factual_oob_fraction": float(
            np.mean(
                ((case.factuals.values < 0) | (case.factuals.values > 1)).any(axis=1)
            )
        ),
        "runtime_total_s": timings.get("total_s"),
    }
    metrics = {column: metrics_values.get(column) for column in paths.summary_columns}
    point_rows = []
    for point, output in enumerate(report.points):
        target_probability = output.values.get("target_probability")
        if target_probability is None:
            probabilities = np.asarray(values["common.target_probabilities"])
            rank = int(report.metadata.get("primary_rank", 0))
            target_probability = float(probabilities[point, rank])
        diagnostics = (
            dict(point_diagnostics[point]) if point < len(point_diagnostics) else {}
        )
        point_values = {
            **diagnostics,
            "point": point,
            "factual_label": case.factuals.true_labels[point].item(),
            "factual_prediction": case.factual_predictions[point].item(),
            "target": case.targets[point].item(),
            "cf_prediction": y_cf_pred[point].item(),
            "valid": bool(valid[point]),
            "target_probability": target_probability,
            "changed_columns": int(
                np.count_nonzero(X_cf[point] != case.factuals.values[point])
            ),
        }
        point_values["model_evaluations"] = diagnostics.get("evaluations")
        point_values["search_attempts"] = diagnostics.get("attempts")
        point_values["endpoint_train_index"] = diagnostics.get("endpoint_index")
        point_rows.append(
            {column: point_values.get(column) for column in paths.point_columns}
        )
    arrays: dict[str, np.ndarray] = {
        "X_test": case.factuals.values,
        "y_test": case.factuals.true_labels,
        "X_cf": X_cf,
        "y_pred": case.factual_predictions,
        "y_target": case.targets,
        "y_cf_pred": y_cf_pred,
    }
    if method_name == "dicoflex":
        arrays.update(
            {
                "X_sparse": np.asarray(values["method.sparse_counterfactuals"]),
                "X_cf_sets": np.asarray(values["common.candidates"]),
                "diverse_available_count": np.asarray(values["method.available_count"]),
            }
        )
    elif method_name == "nice":
        arrays.update(
            {
                "prototypes": np.asarray(values["method.prototypes"]),
                "prototype_indices": np.asarray(values["method.prototype_indices"]),
            }
        )
    elif method_name == "dice":
        arrays["X_cf_raw"] = np.asarray(values["method.raw_candidates"])
    if set(arrays) != set(paths.required_npz_keys):
        raise ValueError("legacy exporter produced unexpected NPZ keys")
    arrays = {key: arrays[key] for key in paths.required_npz_keys}
    write_v1_dataset_outputs(
        paths.metrics_csv,
        paths.points_csv,
        paths.arrays_npz,
        metrics,
        point_rows,
        arrays=arrays,
    )
    validate_generic_v1(paths)
    return paths


def validate_generic_v1(paths: GenericLegacyPaths) -> None:
    if not paths.metrics_csv.is_file() or not paths.points_csv.is_file():
        raise FileNotFoundError("legacy CSV artifacts are incomplete")
    with paths.metrics_csv.open(newline="") as handle:
        metrics_header = tuple(next(csv.reader(handle)))
    with paths.points_csv.open(newline="") as handle:
        points_header = tuple(next(csv.reader(handle)))
    if metrics_header != paths.summary_columns or points_header != paths.point_columns:
        raise ValueError("legacy CSV headers do not match the frozen v1 contract")
    try:
        with np.load(paths.arrays_npz, allow_pickle=False) as archive:
            keys = tuple(archive.files)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("legacy NPZ artifact is malformed") from error
    if keys != paths.required_npz_keys:
        raise ValueError("legacy NPZ keys do not match the frozen v1 contract")


def ensure_generic_v1(
    output_root: Path,
    *,
    dataset_name: str,
    method_name: str,
    case: BenchmarkCase,
    report: EvaluationReport,
    point_diagnostics: Sequence[Mapping[str, Any]] = (),
    manifest: Mapping[str, Any] | None = None,
) -> GenericLegacyPaths:
    paths = generic_legacy_paths(output_root, method_name, dataset_name)
    try:
        validate_generic_v1(paths)
    except (FileNotFoundError, ValueError):
        return export_generic_v1(
            output_root,
            dataset_name=dataset_name,
            method_name=method_name,
            case=case,
            report=report,
            point_diagnostics=point_diagnostics,
            manifest=manifest,
        )
    return paths


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
