"""Historical Experiment 8 result compatibility over canonical artifacts."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.generator import (
    ATHENA_CONTEXT_SIZE,
    ATHENA_CONTEXT_STRATEGY,
    DEFAULT_CATEGORICAL_PROPOSAL_COUNT,
    DEFAULT_POINT_ESTIMATE,
)
from experiments.zeroshot_cf.orchestration.artifacts import StoredRun
from experiments.zeroshot_cf.orchestration.legacy import generic_legacy_paths
from experiments.zeroshot_cf.orchestration.spec import RunSpec

_EXP8_METADATA_COLUMNS = (
    "dataset",
    "backend",
    "selector",
    "valid_candidate_objective",
    "context_strategy",
    "context_size",
    "context_labels",
    "candidate_mode",
    "context_update",
    "point_estimate",
    "project_to_domain",
    "candidate_quantiles",
    "confidence_quantiles",
    "cf_mode",
    "plausibility_backend",
    "max_validity_steps",
    "allow_revisits",
    "joint_shortlist_size",
    "max_extra_actions",
    "min_joint_log_gain",
    "n_counterfactuals",
    "diversity_beam_width",
    "diversity_candidate_pool_size",
    "diversity_max_extra_actions",
    "diversity_max_gower_ratio",
    "diversity_max_gower_increase",
    "joint_scoring_batch_count_mean",
    "joint_rows_scored_mean",
    "accepted_refinement_count_mean",
    "initial_sparse_action_count_mean",
    "final_action_count_mean",
    "extra_actions_mean",
    "categorical_proposal_count",
    "categorical_confidence_batching",
    "conditional_estimator_cache",
    "tabicl_kv_cache",
    "split_variant",
    "test_selection",
    "n_estimators",
    "temperature",
    "n_test",
    "runtime_s",
)
_EXP8_METRIC_COLUMNS = (
    "validity",
    "lof_scores_cf",
    "sparsity",
    "actionability",
    "true_actionability",
    "proximity_l2_jaccard",
    "frac_oob",
    "l0_count_mean",
    "l0_count_median",
    "l0_count_max",
    "steps_mean",
    "steps_median",
    "steps_max",
    "failure_rate",
    "n_actionable",
    "diverse_coverage_at_k",
    "diverse_returned_count_mean",
)
_EXP8_MULTI_CF_COLUMNS = (
    "diverse_returned_validity",
    "diverse_action_jaccard_mean",
    "diverse_action_jaccard_min",
    "diverse_pairwise_gower_mean",
    "diverse_pairwise_gower_min",
    "diverse_factual_gower_mean",
    "diverse_action_count_mean",
)



def _legacy_arrays(dataset_name: str, results_dir: Path) -> dict[str, np.ndarray]:
    paths = generic_legacy_paths(results_dir, "dicoflex", dataset_name)
    with np.load(paths.arrays_npz, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _point_values(
    diagnostics: Sequence[dict[str, Any]],
    name: str,
    default: Any,
) -> list[Any]:
    return [values.get(name, default) for values in diagnostics]


def _legacy_info(
    spec: RunSpec,
    metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
    stored: StoredRun,
) -> dict[str, Any]:
    """Rebuild the documented Exp8 diagnostics from canonical artifacts."""
    metrics = dict(metrics)
    manifest = stored.manifest
    config = dict(manifest["resolved_method_config"])
    search = dict(config["search"])
    diversity = dict(config["diversity"])
    foundation = dict(config["foundation"])
    run_diagnostics = dict(manifest.get("method_run_diagnostics", {}))
    point_diagnostics = [
        dict(values) for values in manifest.get("method_point_diagnostics", ())
    ]
    if len(point_diagnostics) != len(arrays["X_test"]):
        raise ValueError("canonical Exp8 point diagnostics are incomplete")
    canonical_summary = stored.report.summary.values
    metrics["diverse_returned_validity"] = canonical_summary.get(
        "set_validity_returned_threshold", metrics.get("diverse_returned_validity")
    )
    report_arrays = stored.report.arrays.values
    available = np.asarray(report_arrays["common.available"], dtype=bool)
    lof_grid = np.full(available.shape, np.nan, dtype=float)
    lof_grid[available] = np.asarray(report_arrays["candidate.lof_score"], dtype=float)
    primary_rank = int(stored.report.metadata.get("primary_rank", 0))
    primary_lof = lof_grid[:, primary_rank]
    if np.isfinite(primary_lof).any():
        metrics["lof_scores_cf"] = float(np.nanmean(primary_lof))
    cache = dict(run_diagnostics.get("cache", {}))
    actionable_idx = list(run_diagnostics.get("actionable_idx", ()))
    immutable_idx = list(run_diagnostics.get("immutable_idx", ()))
    flipped = [
        bool(value)
        for value in _point_values(point_diagnostics, "flipped", False)
    ]
    return {
        # The old live bundle and estimator are deliberately unavailable after the
        # generic lifecycle. The keys remain so callers can detect that boundary.
        "bundle": None,
        "y_pred": arrays["y_pred"],
        "y_target": arrays["y_target"],
        "actionable_idx": actionable_idx,
        "immutable_idx": immutable_idx,
        "disc_model": None,
        "tau": float(search["tau"]),
        "temperature": float(foundation["temperature"]),
        "candidate_quantiles": (
            None
            if search.get("candidate_quantiles") is None
            else tuple(search["candidate_quantiles"])
        ),
        "confidence_quantiles": (
            None
            if foundation.get("confidence_quantiles") is None
            else tuple(foundation["confidence_quantiles"])
        ),
        "cf_mode": search["cf_mode"],
        "plausibility_backend": metrics["plausibility_backend"],
        "max_validity_steps": metrics["max_validity_steps"],
        "allow_revisits": bool(search["allow_revisits"]),
        "joint_shortlist_size": int(search["joint_shortlist_size"]),
        "max_extra_actions": int(search["max_extra_actions"]),
        "min_joint_log_gain": float(search["min_joint_log_gain"]),
        "n_counterfactuals": spec.method.n_counterfactuals,
        "diversity_beam_width": int(diversity["beam_width"]),
        "diversity_candidate_pool_size": int(diversity["candidate_pool_size"]),
        "diversity_max_extra_actions": int(diversity["max_extra_actions"]),
        "diversity_max_gower_ratio": float(diversity["max_gower_ratio"]),
        "diversity_max_gower_increase": float(diversity["max_gower_increase"]),
        "diversity_candidate_generation": "bounded_beam",
        "diversity_selector": "exact_fixed_size_dpp_map",
        "categorical_proposal_count": int(
            search.get("categorical_proposal_count", DEFAULT_CATEGORICAL_PROPOSAL_COUNT)
        ),
        "categorical_confidence_batching": bool(
            cache.get("conditional_estimator", False)
        ),
        "conditional_estimator_cache": bool(
            cache.get("conditional_estimator", False)
        ),
        "tabicl_kv_cache": bool(cache.get("key_value", False)),
        "test_selection": spec.protocol.test_selection,
        "split_variant": metrics["split_variant"],
        "preprocessing_variant": metrics["preprocessing_variant"],
        "n_dropped_rows": int(metrics["n_dropped_rows"]),
        "n_estimators": int(foundation["n_estimators"]),
        "runtime_s": float(run_diagnostics["runtime_s"]),
        "X_sparse": arrays["X_sparse"],
        "X_cf_sets": arrays["X_cf_sets"],
        "diverse_available_count_per_point": arrays["diverse_available_count"],
        "diverse_candidate_pool_count_per_point": np.asarray(
            _point_values(point_diagnostics, "candidate_pool_count", 0), dtype=int
        ),
        "diverse_search_depth_per_point": np.asarray(
            _point_values(point_diagnostics, "search_depth", 0), dtype=int
        ),
        "diverse_histories_per_point": _point_values(
            point_diagnostics, "diverse_histories", []
        ),
        "point_runtime_s": np.asarray(
            _point_values(point_diagnostics, "point_runtime_s", np.nan), dtype=float
        ),
        "joint_scoring_runtime_s_per_point": np.asarray(
            _point_values(point_diagnostics, "joint_scoring_runtime_s", 0.0),
            dtype=float,
        ),
        "changed_per_point": _point_values(point_diagnostics, "changed_columns", []),
        "flipped_per_point": flipped,
        "steps_per_point": _point_values(point_diagnostics, "steps", 0),
        "history_per_point": _point_values(point_diagnostics, "history", []),
        "attempt_history_per_point": _point_values(
            point_diagnostics, "attempt_history", []
        ),
        "validity_steps_per_point": _point_values(
            point_diagnostics, "validity_steps", 0
        ),
        "initial_valid_step_per_point": _point_values(
            point_diagnostics, "initial_valid_step", None
        ),
        "refinement_steps_per_point": _point_values(
            point_diagnostics, "refinement_steps", 0
        ),
        "accepted_refinement_count_per_point": _point_values(
            point_diagnostics, "accepted_refinement_count", 0
        ),
        "initial_sparse_action_count_per_point": np.asarray(
            _point_values(point_diagnostics, "initial_sparse_action_count", -1),
            dtype=int,
        ),
        "final_action_count_per_point": np.asarray(
            _point_values(point_diagnostics, "final_action_count", 0), dtype=int
        ),
        "initial_tabicl_joint_log_density_per_point": np.asarray(
            _point_values(
                point_diagnostics, "initial_tabicl_joint_log_density", np.nan
            ),
            dtype=float,
        ),
        "final_tabicl_joint_log_density_per_point": np.asarray(
            _point_values(point_diagnostics, "final_tabicl_joint_log_density", np.nan),
            dtype=float,
        ),
        "tabicl_joint_log_density_gain_per_point": np.asarray(
            _point_values(point_diagnostics, "tabicl_joint_log_density_gain", np.nan),
            dtype=float,
        ),
        "joint_scoring_batch_count_per_point": np.asarray(
            _point_values(point_diagnostics, "joint_scoring_batch_count", 0),
            dtype=int,
        ),
        "joint_rows_scored_per_point": np.asarray(
            _point_values(point_diagnostics, "joint_rows_scored", 0), dtype=int
        ),
        "extra_actions_per_point": np.asarray(
            _point_values(point_diagnostics, "extra_actions", 0), dtype=int
        ),
        "refinement_stopping_reason_per_point": _point_values(
            point_diagnostics, "refinement_stopping_reason", "not_started"
        ),
        "target_probability_per_point": np.asarray(
            _point_values(point_diagnostics, "target_probability", np.nan), dtype=float
        ),
        "metrics": metrics,
        "run_spec": spec,
    }



def _legacy_metrics(
    X_test: np.ndarray,
    X_cf: np.ndarray,
    info: dict[str, Any],
) -> dict[str, float]:
    """Map canonical results to the historical Exp8 metric names and order."""
    shared = info["metrics"]
    flipped = np.asarray(info["flipped_per_point"], dtype=bool)
    changed = np.asarray(
        [len(columns) for columns in info["changed_per_point"]], dtype=float
    )
    steps = np.asarray(info["steps_per_point"], dtype=float)

    def valid_stat(values: np.ndarray, reducer) -> float:
        selected = values[flipped]
        return float(reducer(selected)) if len(selected) else float("nan")

    immutable = np.asarray(info["immutable_idx"], dtype=int)
    true_actionability = (
        float(np.all(X_cf[:, immutable] == X_test[:, immutable], axis=1).mean())
        if len(immutable)
        else 1.0
    )
    metrics = {
        "validity": float(shared["validity"]),
        "lof_scores_cf": float(shared["lof_scores_cf"]),
        "sparsity": float(np.mean(X_test != X_cf)),
        "actionability": float(np.all(X_test == X_cf, axis=1).mean()),
        "true_actionability": true_actionability,
        "proximity_l2_jaccard": float(
            shared["proximity_all_features_euclidean"]
        ),
        "frac_oob": float(np.any((X_cf < 0.0) | (X_cf > 1.0), axis=1).mean()),
        "l0_count_mean": valid_stat(changed, np.mean),
        "l0_count_median": valid_stat(changed, np.median),
        "l0_count_max": valid_stat(changed, np.max),
        "steps_mean": valid_stat(steps, np.mean),
        "steps_median": valid_stat(steps, np.median),
        "steps_max": valid_stat(steps, np.max),
        "failure_rate": float((~flipped).mean()),
        "n_actionable": len(info["actionable_idx"]),
        "diverse_coverage_at_k": float(shared["diverse_coverage_at_k"]),
        "diverse_returned_count_mean": float(
            shared["diverse_returned_count_mean"]
        ),
    }
    if np.asarray(info["X_cf_sets"]).shape[1] > 1:
        metrics.update(
            {
                name: float(shared[name])
                for name in _EXP8_MULTI_CF_COLUMNS
            }
        )
    return metrics


def _legacy_row(
    dataset_name: str,
    X_test: np.ndarray,
    X_cf: np.ndarray,
    info: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    initial_counts = np.asarray(
        info["initial_sparse_action_count_per_point"], dtype=float
    )
    reached_validity = initial_counts >= 0
    metrics = _legacy_metrics(X_test, X_cf, info)
    row: dict[str, Any] = {
        "dataset": dataset_name,
        "backend": "tabicl_v2",
        "selector": "prob_ascent",
        "valid_candidate_objective": "grouped_gower",
        "context_strategy": ATHENA_CONTEXT_STRATEGY,
        "context_size": ATHENA_CONTEXT_SIZE,
        "context_labels": "disc",
        "candidate_mode": "batched",
        "context_update": "replace",
        "point_estimate": DEFAULT_POINT_ESTIMATE,
        "project_to_domain": True,
        "candidate_quantiles": info["candidate_quantiles"],
        "confidence_quantiles": info["confidence_quantiles"],
        "cf_mode": info["cf_mode"],
        "plausibility_backend": info["plausibility_backend"],
        "max_validity_steps": info["max_validity_steps"],
        "allow_revisits": info["allow_revisits"],
        "joint_shortlist_size": info["joint_shortlist_size"],
        "max_extra_actions": info["max_extra_actions"],
        "min_joint_log_gain": info["min_joint_log_gain"],
        "n_counterfactuals": info["n_counterfactuals"],
        "diversity_beam_width": info["diversity_beam_width"],
        "diversity_candidate_pool_size": info["diversity_candidate_pool_size"],
        "diversity_max_extra_actions": info["diversity_max_extra_actions"],
        "diversity_max_gower_ratio": info["diversity_max_gower_ratio"],
        "diversity_max_gower_increase": info["diversity_max_gower_increase"],
        "joint_scoring_batch_count_mean": float(
            np.mean(info["joint_scoring_batch_count_per_point"])
        ),
        "joint_rows_scored_mean": float(
            np.mean(info["joint_rows_scored_per_point"])
        ),
        "accepted_refinement_count_mean": float(
            np.mean(info["accepted_refinement_count_per_point"])
        ),
        "initial_sparse_action_count_mean": (
            float(initial_counts[reached_validity].mean())
            if np.any(reached_validity)
            else float("nan")
        ),
        "final_action_count_mean": float(
            np.mean(info["final_action_count_per_point"])
        ),
        "extra_actions_mean": float(np.mean(info["extra_actions_per_point"])),
        "categorical_proposal_count": info["categorical_proposal_count"],
        "categorical_confidence_batching": info[
            "categorical_confidence_batching"
        ],
        "conditional_estimator_cache": info["conditional_estimator_cache"],
        "tabicl_kv_cache": info["tabicl_kv_cache"],
        "split_variant": info["split_variant"],
        "test_selection": info["test_selection"],
        "n_estimators": info["n_estimators"],
        "temperature": info["temperature"],
        "n_test": len(X_test),
        "runtime_s": round(float(info["runtime_s"]), 2),
        **metrics,
    }
    expected = (
        *_EXP8_METADATA_COLUMNS,
        *_EXP8_METRIC_COLUMNS,
        *(_EXP8_MULTI_CF_COLUMNS if np.asarray(info["X_cf_sets"]).shape[1] > 1 else ()),
    )
    if tuple(row) != expected:
        raise AssertionError("Exp8 compatibility row does not match its frozen order")
    return row, metrics



def load_exp8_result(
    spec: RunSpec,
    metrics: dict[str, Any],
    *,
    stored: StoredRun,
    results_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Adapt one completed canonical run to the historical Exp8 return shape."""
    if (
        stored.manifest.get("cell_id") != spec.cell_id
        or stored.manifest.get("scientific_spec") != spec.scientific_payload()
    ):
        raise ValueError("canonical Exp8 run does not match its requested spec")
    arrays = _legacy_arrays(spec.dataset.name, results_dir)
    info = _legacy_info(
        spec,
        metrics,
        arrays,
        stored,
    )
    return arrays["X_test"], arrays["y_test"], arrays["X_cf"], info


def export_exp8_result(
    dataset_name: str,
    X_test: np.ndarray,
    X_cf: np.ndarray,
    info: dict[str, Any],
    *,
    results_dir: Path,
) -> dict[str, float]:
    """Write the frozen historical Exp8 metrics surface."""
    row, metrics = _legacy_row(dataset_name, X_test, X_cf, info)
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / f"exp8_tabicl_{dataset_name}_metrics.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return metrics
