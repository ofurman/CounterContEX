"""Compatibility adapters used while numbered runners remain in service."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from experiments.zeroshot_cf.benchmark_protocol import (
    BenchmarkDatasetContext,
    BenchmarkResultPaths,
    write_dataset_outputs,
    write_result_table,
)
from experiments.zeroshot_cf.core.contracts import (
    GenerationRequest,
    GenerationResult,
    MethodContext,
)
from experiments.zeroshot_cf.evaluation import EvaluationSpec, Evaluator
from experiments.zeroshot_cf.evaluation.result import EvaluationReport


def method_context(context: BenchmarkDatasetContext) -> MethodContext:
    if context.benchmark_case is None:
        raise RuntimeError("portable benchmark case is required for method execution")
    case = context.benchmark_case
    return MethodContext(case.dataset.X_train, case.dataset.schema, case.oracle)


def generation_request(
    context: BenchmarkDatasetContext,
    *,
    seed: int,
) -> GenerationRequest:
    return GenerationRequest(context.X_test, context.y_target, 1, seed)


def evaluate_result(
    context: BenchmarkDatasetContext,
    result: GenerationResult,
    *,
    probability_threshold: float,
) -> EvaluationReport:
    if context.benchmark_case is None:
        raise RuntimeError("portable benchmark case is required for evaluation")
    return (
        Evaluator()
        .prepare(
            context.benchmark_case,
            EvaluationSpec(probability_threshold=probability_threshold),
        )
        .evaluate(result)
    )


def legacy_candidate_matrix(result: GenerationResult) -> np.ndarray:
    """Return raw v1 rows while canonical metrics retain truthful availability."""
    if result.candidates.shape[1] != 1:
        raise ValueError("v1 baseline export requires one candidate rank")
    rows = result.candidates[:, 0].copy()
    unavailable = ~result.available[:, 0]
    if unavailable.any():
        best_effort = result.artifacts.get("method.best_effort")
        if best_effort is None or np.asarray(best_effort).shape != rows.shape:
            raise ValueError("unavailable v1 rows require method.best_effort")
        rows[unavailable] = np.asarray(best_effort)[unavailable]
    return rows


def legacy_common_metrics(report: EvaluationReport) -> dict[str, float]:
    """Map versioned evaluator fields onto the frozen Exp11-14 column names."""
    values = report.summary.values
    mapping = {
        "coverage": "coverage",
        "validity": "validity_returned_class",
        "actionability": "actionability",
        "sparsity": "sparsity",
        "action_unit_sparsity_mean": "action_unit_sparsity_mean",
        "proximity_grouped_gower": "proximity_grouped_gower",
        "proximity_continuous_manhattan": "proximity_continuous_manhattan",
        "proximity_continuous_euclidean": "proximity_continuous_euclidean",
        "lof_scores_cf": "lof_scores_cf",
        "lof_scores_test": "lof_scores_test",
        "isolation_forest_scores_cf": "isolation_forest_scores_cf",
        "isolation_forest_scores_test": "isolation_forest_scores_test",
    }
    return {legacy: float(values[current]) for legacy, current in mapping.items()}


def point_diagnostics(result: GenerationResult) -> list[dict[str, Any]]:
    return [dict(values) for values in result.point_diagnostics]


def write_legacy_outputs_with_timing(
    paths: BenchmarkResultPaths,
    row: dict[str, Any],
    point_rows: Sequence[Mapping[str, Any]],
    *,
    arrays: Mapping[str, Any],
    total_started: float,
) -> None:
    """Write frozen v1 artifacts and persist measured uniform write timing."""
    write_started = time.perf_counter()
    write_dataset_outputs(paths, row, point_rows, arrays=arrays)
    row["write_s"] = time.perf_counter() - write_started
    row["total_s"] = time.perf_counter() - total_started
    # The metrics table is rewritten once so it contains the measured write phase.
    write_result_table(paths.metrics_csv, [row])
