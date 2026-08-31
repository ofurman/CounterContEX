"""Shared benchmark protocol for the retained Exp9/Exp11-14 suite."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.contracts import BenchmarkCase
from experiments.zeroshot_cf.data import (
    DatasetBundle,
    get_grouped_categorical_action_space,
    get_one_hot_groups,
    load_dataset,
)
from experiments.zeroshot_cf.datasets.benchmark import (
    build_benchmark_case,
    select_factual_indices,
)
from experiments.zeroshot_cf.discriminator import train_discriminator

DATASETS = ("heloc", "bank_marketing", "give_me_some_credit", "lending_club")
DEFAULT_MAX_TEST = 1000
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_TEST_SELECTION = "stratified"
DEFAULT_DROP_HELOC_ALL_MINUS9 = True
DEFAULT_PROTOCOL_SEED = 42
DEFAULT_SPARSITY_EPS = 0.05
TARGET_CLASSIFIER_LABELS = "target_classifier"


@dataclass(frozen=True)
class BenchmarkDatasetContext:
    """Deprecated compatibility view over a portable ``BenchmarkCase``."""

    dataset_name: str
    bundle: DatasetBundle
    X_test: np.ndarray
    y_test: np.ndarray
    disc_model: Any
    y_pred: np.ndarray
    y_target: np.ndarray
    scalar_actionable: tuple[int, ...]
    grouped_actionable: tuple[OneHotActionGroup, ...]
    immutable_idx: tuple[int, ...]
    categorical_groups: tuple[OneHotActionGroup, ...]
    test_selection: str = DEFAULT_TEST_SELECTION
    benchmark_case: BenchmarkCase | None = None

    @property
    def factual_source_indices(self) -> np.ndarray:
        if self.benchmark_case is not None:
            return self.benchmark_case.factuals.indices
        indices = np.arange(len(self.X_test), dtype=np.int64)
        indices.setflags(write=False)
        return indices

    @property
    def validation_accuracy(self) -> float:
        if self.bundle.X_val is None or self.bundle.y_val is None:
            return float("nan")
        predictions = np.asarray(self.disc_model.predict(self.bundle.X_val), dtype=int)
        return float((predictions == self.bundle.y_val).mean())

    @property
    def test_accuracy(self) -> float:
        return float((self.y_pred == self.y_test).mean())


@dataclass(frozen=True)
class BenchmarkResultPaths:
    """Standardized output paths for one benchmark dataset run."""

    prefix: Path
    metrics_csv: Path
    points_csv: Path
    arrays_npz: Path


def resolve_max_test_limit(max_test: int) -> int | None:
    """Map the CLI max-test contract onto a concrete limit."""
    return None if max_test < 0 else max_test


def select_benchmark_test_rows(
    X_test: np.ndarray,
    y_test: np.ndarray,
    limit: int | None,
    selection: str = DEFAULT_TEST_SELECTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility delegate for deterministic factual selection."""
    selected = select_factual_indices(
        y_test,
        limit,
        selection,
        seed=DEFAULT_PROTOCOL_SEED,
    )
    return X_test[selected], y_test[selected]


def build_classifier_targets(
    disc_model: Any,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive benchmark targets from the classifier's factual predictions."""
    y_pred = np.asarray(disc_model.predict(X_test), dtype=int).reshape(-1)
    unique = set(np.unique(y_pred).tolist())
    if not unique <= {0, 1}:
        raise ValueError(f"benchmark targets require binary predictions, got {unique}")
    y_target = 1 - y_pred
    return y_pred, y_target


def build_discriminator_cache_tag(
    dataset_name: str,
    bundle: DatasetBundle,
) -> str:
    """Keep cached classifier naming aligned across benchmark entry points."""
    tag = (
        f"{dataset_name}_drop_all_minus9"
        if bundle.preprocessing_variant == "drop_heloc_all_minus9"
        else dataset_name
    )
    if bundle.X_val is not None:
        tag = f"{tag}_{bundle.split_variant}"
    return tag


def prepare_benchmark_context(
    dataset_name: str,
    *,
    max_test: int = DEFAULT_MAX_TEST,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_selection: str = DEFAULT_TEST_SELECTION,
    drop_heloc_all_minus9: bool = DEFAULT_DROP_HELOC_ALL_MINUS9,
) -> BenchmarkDatasetContext:
    """Compatibility delegate that exposes the reusable portable case."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported benchmark dataset: {dataset_name!r}")

    bundle = load_dataset(
        dataset_name,
        drop_heloc_all_minus9=(
            drop_heloc_all_minus9 if dataset_name == "heloc" else False
        ),
        validation_fraction=validation_fraction,
    )
    scalar_actionable, grouped_actionable, immutable_idx = (
        get_grouped_categorical_action_space(bundle)
    )
    X_disc_eval = bundle.X_val if bundle.X_val is not None else bundle.X_test
    y_disc_eval = bundle.y_val if bundle.y_val is not None else bundle.y_test
    disc_model = train_discriminator(
        bundle.X_train,
        bundle.y_train,
        X_disc_eval,
        y_disc_eval,
        build_discriminator_cache_tag(dataset_name, bundle),
    )
    if bundle.prepared is None:
        raise RuntimeError("CEL compatibility bundle is missing its PreparedDataset")
    case = build_benchmark_case(
        bundle.prepared,
        disc_model,
        max_test=resolve_max_test_limit(max_test),
        test_selection=test_selection,
        seed=DEFAULT_PROTOCOL_SEED,
        target_model={
            "kind": "logistic_regression",
            "C": 1.0,
            "max_iter": 1000,
            "seed": DEFAULT_PROTOCOL_SEED,
            "cache_tag": build_discriminator_cache_tag(dataset_name, bundle),
        },
    )
    return BenchmarkDatasetContext(
        dataset_name=dataset_name,
        bundle=bundle,
        X_test=case.factuals.values,
        y_test=case.factuals.true_labels,
        disc_model=disc_model,
        y_pred=case.factual_predictions,
        y_target=case.targets,
        scalar_actionable=tuple(int(column) for column in scalar_actionable),
        grouped_actionable=tuple(grouped_actionable),
        immutable_idx=tuple(int(column) for column in immutable_idx),
        categorical_groups=tuple(get_one_hot_groups(bundle)),
        test_selection=test_selection,
        benchmark_case=case,
    )


def build_common_result_row(
    context: BenchmarkDatasetContext,
    *,
    method: str,
    cf_per_factual: int,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the protocol-owned shared result row fields."""
    row: dict[str, Any] = {
        "dataset": context.dataset_name,
        "method": method,
        "split_variant": context.bundle.split_variant,
        "split_seed": DEFAULT_PROTOCOL_SEED,
        "test_selection": context.test_selection,
        "n_train": len(context.bundle.X_train),
        "n_validation": 0
        if context.bundle.X_val is None
        else len(context.bundle.X_val),
        "n_test_pool": len(context.bundle.X_test),
        "n_test": len(context.X_test),
        "cf_per_factual": cf_per_factual,
        "target_classifier_validation_accuracy": context.validation_accuracy,
        "target_classifier_test_accuracy": context.test_accuracy,
        "preprocessing_variant": context.bundle.preprocessing_variant,
        "n_dropped_rows": context.bundle.n_dropped_rows,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def dataset_result_paths(
    results_dir: Path,
    stem: str,
    dataset_name: str,
) -> BenchmarkResultPaths:
    """Return the standardized result artefact paths for one dataset."""
    prefix = results_dir / f"{stem}_{dataset_name}"
    return BenchmarkResultPaths(
        prefix=prefix,
        metrics_csv=prefix.with_name(f"{prefix.name}_metrics.csv"),
        points_csv=prefix.with_name(f"{prefix.name}_points.csv"),
        arrays_npz=prefix.with_name(f"{prefix.name}_arrays.npz"),
    )


def aggregate_metrics_path(results_dir: Path, stem: str) -> Path:
    """Return the standardized aggregate-metrics path for one runner."""
    return results_dir / f"{stem}_all_metrics.csv"


def write_result_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a CSV result table with stable first-seen column ordering."""
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


def write_dataset_outputs(
    paths: BenchmarkResultPaths,
    metrics_row: Mapping[str, Any],
    point_rows: Sequence[Mapping[str, Any]],
    *,
    arrays: Mapping[str, Any] | None = None,
) -> None:
    """Write one dataset's metrics/points CSVs and optional NPZ arrays."""
    write_result_table(paths.metrics_csv, [metrics_row])
    write_result_table(paths.points_csv, point_rows)
    if arrays is not None:
        paths.arrays_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(paths.arrays_npz, **arrays)
        print(f"Wrote {paths.arrays_npz}")


def aggregate_dataset_metrics(
    results_dir: Path,
    stem: str,
    *,
    datasets: Sequence[str] = DATASETS,
) -> Path:
    """Combine completed per-dataset metrics in protocol dataset order."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset_name in datasets:
        path = dataset_result_paths(results_dir, stem, dataset_name).metrics_csv
        if not path.exists():
            missing.append(dataset_name)
            continue
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if missing:
        raise FileNotFoundError(
            "Missing per-dataset benchmark results for: " + ", ".join(missing)
        )
    output = aggregate_metrics_path(results_dir, stem)
    write_result_table(output, rows)
    return output


def mean_on_valid(values: Sequence[float] | np.ndarray, valid: np.ndarray) -> float:
    """Return the mean over valid rows, or NaN when none are valid."""
    array = np.asarray(values, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    return float(array[mask].mean()) if mask.any() else float("nan")
