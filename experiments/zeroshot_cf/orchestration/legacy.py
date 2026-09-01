"""Frozen v1 CSV/NPZ compatibility exporter."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.core.contracts import BenchmarkCase
from experiments.zeroshot_cf.core.validation import target_probabilities
from experiments.zeroshot_cf.evaluation import EvaluationReport
from experiments.zeroshot_cf.mixed_distance import grouped_gower_distance
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


def _levels_text(values: Any) -> str:
    if values is None:
        return "sample"
    return ";".join(str(float(value)) for value in values)


def _finite_mean(values: Sequence[Any]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else float("nan")


def _mean_on_valid(values: Sequence[Any], valid: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    selected = array[np.asarray(valid, dtype=bool)]
    return float(selected.mean()) if len(selected) else float("nan")


def _action_counts(case: BenchmarkCase, rows: np.ndarray) -> np.ndarray:
    factuals = case.factuals.values
    counts = np.zeros(len(rows), dtype=int)
    for column in case.dataset.schema.actionable_scalars:
        counts += rows[:, column] != factuals[:, column]
    for group in case.dataset.schema.actionable_groups:
        columns = list(group.columns)
        counts += np.any(rows[:, columns] != factuals[:, columns], axis=1)
    return counts


def _dataset_metadata(case: BenchmarkCase) -> dict[str, Any]:
    provenance = case.dataset.provenance
    metadata = dict(provenance.metadata)
    split_variant = metadata.get("split_variant")
    if split_variant is None:
        split_variant = provenance.split_id.split(":seed=", 1)[0]
    preprocessing_variant = metadata.get("preprocessing_variant")
    if preprocessing_variant is None:
        preprocessing_variant = provenance.preprocessing_id.split(":", 1)[-1]
    return {
        "split_variant": split_variant,
        "split_seed": metadata.get(
            "split_seed", case.protocol.get("selection_seed", 42)
        ),
        "preprocessing_variant": preprocessing_variant,
        "n_dropped_rows": int(metadata.get("n_dropped_rows", 0)),
    }


def _legacy_method_id(
    method_name: str,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    n_counterfactuals: int,
) -> str:
    if method_name != "dicoflex":
        return str(contract["method_id"])
    ids = contract["method_ids"]
    if n_counterfactuals > 1:
        return str(ids["diverse"])
    mode = str(dict(config.get("search", {})).get("cf_mode", "sparse"))
    try:
        return str(ids[mode])
    except KeyError as error:
        raise ValueError(f"unsupported CounterContEx v1 mode: {mode!r}") from error


def _common_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "coverage",
        "actionability",
        "sparsity",
        "action_unit_sparsity_mean",
        "proximity_grouped_gower",
        "proximity_continuous_manhattan",
        "proximity_continuous_euclidean",
        "lof_scores_cf",
        "lof_scores_test",
        "isolation_forest_scores_cf",
        "isolation_forest_scores_test",
    )
    return {
        **{name: summary.get(name) for name in names},
        "validity": summary.get("validity_returned_class"),
    }


def _method_metrics(
    method_name: str,
    *,
    case: BenchmarkCase,
    report: EvaluationReport,
    config: Mapping[str, Any],
    run_diagnostics: Mapping[str, Any],
    point_diagnostics: Sequence[Mapping[str, Any]],
    X_cf: np.ndarray,
    valid: np.ndarray,
    timings: Mapping[str, Any],
    n_counterfactuals: int,
) -> dict[str, Any]:
    factuals = case.factuals.values
    changed_columns = np.count_nonzero(X_cf != factuals, axis=1)
    action_counts = _action_counts(case, X_cf)
    values = report.arrays.values
    summary = report.summary.values
    valid_l2 = np.linalg.norm(X_cf[valid] - factuals[valid], axis=1)
    result: dict[str, Any] = {
        "runtime_generation_s": timings.get("generate_s"),
        **_common_metrics(summary),
        "validity": float(valid.mean()),
        "sparsity_exact": float((X_cf != factuals).mean()),
        "true_actionability": summary.get("actionability"),
        "proximity_all_features_euclidean": (
            float(valid_l2.mean()) if len(valid_l2) else float("nan")
        ),
        "failure_rate": float((~valid).mean()),
        "l0_count_mean": _mean_on_valid(changed_columns, valid),
        "factual_oob_fraction": float(
            (((factuals < 0) | (factuals > 1)).any(axis=1)).mean()
        ),
        "cf_oob_fraction": float((((X_cf < 0) | (X_cf > 1)).any(axis=1)).mean()),
        "runtime_total_s": timings.get("total_s"),
    }
    diagnostics = [dict(item) for item in point_diagnostics]
    if method_name == "nice":
        distances = np.asarray(
            values.get(
                "method.prototype_distances",
                np.linalg.norm(
                    np.asarray(values["method.prototypes"]) - factuals, axis=1
                ),
            ),
            dtype=float,
        )
        result.update(
            {
                "prototype_pool_labels": "target_classifier",
                "prototype_metric": "euclidean",
                "valid_candidate_selection": "lof",
                "categorical_actions": "atomic_one_hot_groups",
                "steps_mean": _mean_on_valid(
                    [item.get("steps", np.nan) for item in diagnostics], valid
                ),
                "prototype_distance_mean": float(distances.mean()),
            }
        )
    elif method_name in {"wachter", "growing_spheres"}:
        result.update(
            {
                "model_access": "predict_and_predict_proba",
                "categorical_actions": "atomic_one_hot_groups",
                "posthoc_action_pruning": True,
                "posthoc_scalar_contraction": True,
                "sphere_candidates": (
                    int(config.get("n_candidates", 512))
                    if method_name == "growing_spheres"
                    else 0
                ),
                "action_count_mean": _mean_on_valid(action_counts, valid),
                "model_evaluations_mean": _finite_mean(
                    [item.get("evaluations", np.nan) for item in diagnostics]
                ),
            }
        )
    elif method_name == "dice":
        raw = np.asarray(values["method.raw_candidates"])
        raw_changed = np.count_nonzero(raw != factuals, axis=1)
        raw_l2 = np.linalg.norm(raw - factuals, axis=1)
        result.update(
            {
                "backend": "sklearn",
                "search": "genetic",
                "initialization": "kdtree",
                "max_iterations": int(config.get("max_iterations", 200)),
                "search_restarts": int(config.get("search_restarts", 1)),
                "stopping_threshold": float(config.get("stopping_threshold", 0.5)),
                "proximity_weight": 0.2,
                "sparsity_weight": 0.2,
                "categorical_penalty": 0.1,
                "categorical_actions": "compact_atomic_groups",
                "posthoc_action_pruning": True,
                "posthoc_scalar_contraction": True,
                "features_to_vary_count": len(
                    run_diagnostics.get("features_to_vary", ())
                ),
                "dice_found_fraction": float(
                    np.mean([bool(item.get("found")) for item in diagnostics])
                ),
                "dice_returned_fraction": float(
                    np.mean([bool(item.get("returned")) for item in diagnostics])
                ),
                "raw_l0_count_mean": float(raw_changed.mean()),
                "raw_proximity_all_features_euclidean": float(raw_l2.mean()),
            }
        )
    elif method_name == "face":
        result.update(
            {
                "graph": "symmetric_knn_actionable_space",
                "n_neighbors": min(
                    int(config.get("n_neighbors", 100)), len(case.dataset.X_train) - 1
                ),
                "edge_weight": "euclidean_times_relative_knn_radius",
                "density_power": float(config.get("density_power", 1.0)),
                "tau": float(config.get("tau", 0.5)),
                "endpoint": "observed_actionable_projection",
                "categorical_actions": "atomic_one_hot_groups",
                "runtime_graph_build_s": timings.get("prepare_s"),
                "runtime_search_s": timings.get("generate_s"),
                "runtime_generation_s": (
                    float(timings.get("prepare_s", 0.0))
                    + float(timings.get("generate_s", 0.0))
                ),
                "action_count_mean": _mean_on_valid(action_counts, valid),
                "path_cost_mean": _finite_mean(
                    [item.get("path_cost", np.nan) for item in diagnostics]
                ),
                "path_steps_mean": _mean_on_valid(
                    [item.get("path_steps", np.nan) for item in diagnostics], valid
                ),
                "expanded_nodes_mean": _finite_mean(
                    [item.get("expanded_nodes", np.nan) for item in diagnostics]
                ),
            }
        )
    elif method_name == "dicoflex":
        search = dict(config.get("search", {}))
        diversity = dict(config.get("diversity", {}))
        foundation = dict(config.get("foundation", {}))
        mode = str(search.get("cf_mode", "sparse"))
        candidates = np.asarray(values["common.candidates"])
        available = np.asarray(values["common.available"], dtype=bool)
        factual_grid = np.broadcast_to(factuals[:, None, :], candidates.shape)
        flat_candidates = candidates[available]
        flat_factuals = factual_grid[available]
        numerical = case.dataset.schema.numerical
        groups = case.dataset.schema.categorical_groups
        diverse_gower = (
            grouped_gower_distance(
                flat_candidates, flat_factuals, list(numerical), list(groups)
            )
            if len(flat_candidates)
            else np.empty(0)
        )
        sparse = np.asarray(values["method.sparse_counterfactuals"])
        sparse_actions = _action_counts(case, sparse)
        final_actions = action_counts
        point_runtime = [item.get("point_runtime_s", np.nan) for item in diagnostics]
        joint_runtime = [
            item.get("joint_scoring_runtime_s", 0.0) for item in diagnostics
        ]
        reasons = [
            str(item["refinement_stopping_reason"])
            for item in diagnostics
            if item.get("refinement_stopping_reason") is not None
        ]
        reason_counts = ";".join(
            f"{reason}:{reasons.count(reason)}" for reason in sorted(set(reasons))
        )
        result.update(
            {
                "cf_mode": mode,
                "context_strategy": "gower_knn_both",
                "context_size": 512,
                "context_labels": "target_classifier",
                "candidate_mode": "batched",
                "candidate_quantiles": _levels_text(search.get("candidate_quantiles")),
                "confidence_quantiles": _levels_text(
                    foundation.get("confidence_quantiles")
                ),
                "plausibility_backend": (
                    "tabicl_joint_one_shot"
                    if mode == "data_plausible"
                    else "proposal_support"
                ),
                "tabicl_joint_permutations": int(
                    foundation.get("tabicl_joint_permutations", 1)
                ),
                "max_validity_steps": (
                    search.get("max_validity_steps")
                    if search.get("max_validity_steps") is not None
                    else len(case.dataset.schema.actionable_scalars)
                    + len(case.dataset.schema.actionable_groups)
                ),
                "allow_revisits": bool(search.get("allow_revisits", True)),
                "categorical_proposal_count": int(
                    search.get("categorical_proposal_count", 1)
                ),
                "categorical_confidence_batching": bool(
                    run_diagnostics.get("cache", {}).get("conditional_estimator", False)
                ),
                "conditional_estimator_cache": bool(
                    run_diagnostics.get("cache", {}).get("conditional_estimator", False)
                ),
                "tabicl_kv_cache": bool(
                    run_diagnostics.get("cache", {}).get("key_value", False)
                ),
                "joint_shortlist_size": int(search.get("joint_shortlist_size", 16)),
                "max_extra_actions": int(search.get("max_extra_actions", 1)),
                "min_joint_log_gain": float(search.get("min_joint_log_gain", 0.0)),
                "diversity_beam_width": int(diversity.get("beam_width", 8)),
                "diversity_candidate_pool_size": int(
                    diversity.get("candidate_pool_size", 16)
                ),
                "diversity_max_extra_actions": int(
                    diversity.get("max_extra_actions", 2)
                ),
                "diversity_max_gower_ratio": float(
                    diversity.get("max_gower_ratio", 1.5)
                ),
                "diversity_max_gower_increase": float(
                    diversity.get("max_gower_increase", 0.02)
                ),
                "diversity_candidate_generation": "bounded_beam",
                "diversity_selector": "exact_fixed_size_dpp_map",
                "search_schedule": (
                    "bounded_beam_then_exact_fixed_size_dpp_map"
                    if n_counterfactuals > 1
                    else (
                        "probability_ascent_until_valid_then_one_shot_joint_reranking"
                        if mode == "data_plausible"
                        else "probability_ascent_until_valid_then_min_grouped_gower"
                    )
                ),
                "valid_candidate_objective": (
                    "quality_constrained_dpp"
                    if n_counterfactuals > 1
                    else "grouped_gower"
                ),
                "n_estimators": int(foundation.get("n_estimators", 1)),
                "temperature": float(foundation.get("temperature", 1e-9)),
                "tau": float(search.get("tau", 0.5)),
                "runtime_generation_s": run_diagnostics.get(
                    "runtime_s", timings.get("generate_s")
                ),
                "runtime_generation_per_factual_s": float(
                    run_diagnostics.get("runtime_s", timings.get("generate_s", 0.0))
                )
                / len(factuals),
                "joint_scoring_runtime_s": float(np.sum(joint_runtime)),
                "point_runtime_s_mean": _finite_mean(point_runtime),
                "diverse_coverage_at_k": summary.get("set_coverage_at_k"),
                "diverse_returned_count_mean": summary.get("set_returned_count_mean"),
                "diverse_returned_validity": summary.get(
                    "set_validity_returned_class"
                ),
                "diverse_action_jaccard_mean": summary.get(
                    "set_action_jaccard_mean"
                ),
                "diverse_action_jaccard_min": summary.get("set_action_jaccard_min"),
                "diverse_pairwise_gower_mean": summary.get(
                    "set_pairwise_gower_mean"
                ),
                "diverse_pairwise_gower_min": summary.get("set_pairwise_gower_min"),
                "diverse_factual_gower_mean": (
                    float(diverse_gower.mean())
                    if len(diverse_gower)
                    else float("nan")
                ),
                "diverse_action_count_mean": (
                    float(np.asarray(values["candidate.action_unit_changes"]).mean())
                    if len(flat_candidates)
                    else float("nan")
                ),
                "steps_mean": _mean_on_valid(
                    [item.get("steps", np.nan) for item in diagnostics], valid
                ),
                "validity_steps_mean": _finite_mean(
                    [item.get("validity_steps", np.nan) for item in diagnostics]
                ),
                "post_valid_refinement": mode == "data_plausible",
                "refinement_steps_mean": _finite_mean(
                    [item.get("refinement_steps", np.nan) for item in diagnostics]
                ),
                "refined_fraction": float(
                    np.mean(
                        [item.get("refinement_steps", 0) > 0 for item in diagnostics]
                    )
                ),
                "accepted_refinement_count_mean": _finite_mean(
                    [
                        item.get("accepted_refinement_count", np.nan)
                        for item in diagnostics
                    ]
                ),
                "initial_tabicl_joint_log_density_mean": _finite_mean(
                    [
                        item.get("initial_tabicl_joint_log_density", np.nan)
                        for item in diagnostics
                    ]
                ),
                "final_tabicl_joint_log_density_mean": _finite_mean(
                    [
                        item.get("final_tabicl_joint_log_density", np.nan)
                        for item in diagnostics
                    ]
                ),
                "tabicl_joint_log_density_gain_mean": _finite_mean(
                    [
                        item.get("tabicl_joint_log_density_gain", np.nan)
                        for item in diagnostics
                    ]
                ),
                "joint_scoring_batch_count_mean": _finite_mean(
                    [item.get("joint_scoring_batch_count", 0) for item in diagnostics]
                ),
                "joint_rows_scored_mean": _finite_mean(
                    [item.get("joint_rows_scored", 0) for item in diagnostics]
                ),
                "extra_actions_mean": _finite_mean(
                    [item.get("extra_actions", 0) for item in diagnostics]
                ),
                "initial_sparse_action_count_mean": _finite_mean(
                    [
                        item.get("initial_sparse_action_count", sparse_actions[index])
                        for index, item in enumerate(diagnostics)
                    ]
                ),
                "final_action_count_mean": _finite_mean(
                    [
                        item.get("final_action_count", final_actions[index])
                        for index, item in enumerate(diagnostics)
                    ]
                ),
                "refinement_stopping_reasons": reason_counts,
                "initial_valid_action_sparsity_mean": _finite_mean(
                    [
                        item.get("initial_valid_action_sparsity", np.nan)
                        for item in diagnostics
                    ]
                ),
                "initial_valid_grouped_gower_mean": _finite_mean(
                    [
                        item.get("initial_valid_grouped_gower", np.nan)
                        for item in diagnostics
                    ]
                ),
                "final_action_sparsity_mean": _finite_mean(
                    [
                        item.get("final_action_sparsity", np.nan)
                        for item in diagnostics
                    ]
                ),
                "categorical_first_fraction": float(
                    np.mean(
                        [
                            item.get("first_action_type") == "categorical"
                            for item in diagnostics
                        ]
                    )
                ),
            }
        )
    return result


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
    if manifest is None:
        raise ValueError("legacy export requires a finalized canonical manifest")
    paths = generic_legacy_paths(output_root, method_name, dataset_name)
    contract = V1_CONTRACT[method_name]
    values = report.arrays.values
    X_cf = _legacy_primary(report)
    y_cf_pred = np.asarray(case.oracle.predict(X_cf)).reshape(-1)
    valid = y_cf_pred == case.targets
    timings = dict(manifest.get("timings", {}))
    if set(timings) != {
        "prepare_s",
        "generate_s",
        "evaluate_s",
        "write_s",
        "total_s",
    }:
        raise ValueError("legacy export requires finalized phase timings")
    config = dict(manifest.get("resolved_method_config", {}))
    run_diagnostics = dict(manifest.get("method_run_diagnostics", {}))
    scientific = dict(manifest.get("scientific_spec", {}))
    method_spec = dict(scientific.get("method", {}))
    n_counterfactuals = int(
        method_spec.get(
            "n_counterfactuals", np.asarray(values["common.candidates"]).shape[1]
        )
    )
    legacy_method_id = _legacy_method_id(
        method_name, contract, config, n_counterfactuals
    )
    metadata = _dataset_metadata(case)
    metrics_values = {
        "dataset": dataset_name,
        "method": legacy_method_id,
        **metadata,
        "test_selection": case.protocol.get("test_selection"),
        "n_train": len(case.dataset.X_train),
        "n_validation": len(case.dataset.X_validation),
        "n_test_pool": len(case.dataset.X_test),
        "n_test": len(case.factuals.values),
        "cf_per_factual": n_counterfactuals,
        "target_classifier_validation_accuracy": (
            float(
                np.mean(
                    case.oracle.predict(case.dataset.X_validation)
                    == case.dataset.y_validation
                )
            )
            if len(case.dataset.X_validation)
            else float("nan")
        ),
        "target_classifier_test_accuracy": float(
            np.mean(case.factual_predictions == case.factuals.true_labels)
        ),
        **_method_metrics(
            method_name,
            case=case,
            report=report,
            config=config,
            run_diagnostics=run_diagnostics,
            point_diagnostics=point_diagnostics,
            X_cf=X_cf,
            valid=valid,
            timings=timings,
            n_counterfactuals=n_counterfactuals,
        ),
    }
    metrics = {column: metrics_values.get(column) for column in paths.summary_columns}
    probabilities = target_probabilities(case.oracle, X_cf, case.targets)
    changed_columns = np.count_nonzero(X_cf != case.factuals.values, axis=1)
    action_counts = _action_counts(case, X_cf)
    raw = np.asarray(values.get("method.raw_candidates", X_cf))
    raw_changed = np.count_nonzero(raw != case.factuals.values, axis=1)
    raw_l2 = np.linalg.norm(raw - case.factuals.values, axis=1)
    sparse = np.asarray(values.get("method.sparse_counterfactuals", X_cf))
    sparse_actions = _action_counts(case, sparse)
    primary_rank = int(report.metadata.get("primary_rank", 0))
    available = np.asarray(values["common.available"], dtype=bool)
    lof_grid = np.full(available.shape, np.nan)
    gower_grid = np.full(available.shape, np.nan)
    lof_grid[available] = np.asarray(values["candidate.lof_score"])
    gower_grid[available] = np.asarray(values["candidate.grouped_gower"])
    point_rows = []
    for point in range(len(report.points)):
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
            "target_probability": float(probabilities[point]),
            "changed_columns": int(changed_columns[point]),
            "changed_actions": int(action_counts[point]),
            "action_count": int(action_counts[point]),
            "raw_changed_columns": int(raw_changed[point]),
            "raw_proximity_l2": float(raw_l2[point]),
            "lof_score": float(lof_grid[point, primary_rank]),
            "initial_sparse_action_count": int(
                diagnostics.get("initial_sparse_action_count", sparse_actions[point])
            ),
            "final_action_count": int(
                diagnostics.get("final_action_count", action_counts[point])
            ),
            "initial_valid_action_sparsity": float(
                diagnostics.get(
                    "initial_valid_action_sparsity",
                    np.mean(sparse[point] != case.factuals.values[point]),
                )
            ),
            "initial_valid_grouped_gower": float(
                diagnostics.get(
                    "initial_valid_grouped_gower",
                    grouped_gower_distance(
                        sparse[point : point + 1],
                        case.factuals.values[point : point + 1],
                        list(case.dataset.schema.numerical),
                        list(case.dataset.schema.categorical_groups),
                    )[0],
                )
            ),
            "final_grouped_gower": float(gower_grid[point, primary_rank]),
            "final_action_sparsity": float(
                diagnostics.get(
                    "final_action_sparsity",
                    action_counts[point]
                    / max(
                        1,
                        len(case.dataset.schema.actionable_scalars)
                        + len(case.dataset.schema.actionable_groups),
                    ),
                )
            ),
        }
        if method_name == "nice":
            prototypes = np.asarray(values["method.prototypes"])
            indices = np.asarray(values["method.prototype_indices"])
            point_values["prototype_index"] = int(indices[point])
            point_values["prototype_distance"] = float(
                np.linalg.norm(prototypes[point] - case.factuals.values[point])
            )
        point_values["model_evaluations"] = diagnostics.get("evaluations")
        point_values["search_attempts"] = diagnostics.get("attempts")
        point_values["endpoint_train_index"] = diagnostics.get("endpoint_index")
        point_values["dice_found"] = diagnostics.get("found")
        point_values["dice_returned"] = diagnostics.get("returned")
        point_values["attempt_steps"] = diagnostics.get(
            "attempt_steps", len(diagnostics.get("attempt_history", ()))
        )
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
    # Rebuild from the canonical run on every resume. Header/key-only validation
    # cannot detect stale values from an older exporter implementation.
    return export_generic_v1(
        output_root,
        dataset_name=dataset_name,
        method_name=method_name,
        case=case,
        report=report,
        point_diagnostics=point_diagnostics,
        manifest=manifest,
    )


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
