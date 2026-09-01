"""Reusable common metric kernels and compatibility adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.diverse_search import (
    action_set_jaccard_distance,
    action_unit_signature,
)
from experiments.zeroshot_cf.mixed_distance import (
    action_unit_change_count,
    grouped_gower_distance,
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def prepare_novelty_models(
    X_reference: np.ndarray,
    *,
    lof_n_neighbors: int,
    isolation_forest_estimators: int,
) -> tuple[LocalOutlierFactor, IsolationForest]:
    """Fit reusable novelty state exactly once for a benchmark case."""
    reference = np.asarray(X_reference, dtype=np.float64)
    if reference.ndim != 2 or len(reference) < 2:
        raise ValueError("novelty metrics require at least two reference rows")
    neighbors = min(lof_n_neighbors, len(reference) - 1)
    lof = LocalOutlierFactor(n_neighbors=neighbors, novelty=True).fit(reference)
    isolation = IsolationForest(
        n_estimators=isolation_forest_estimators,
        random_state=42,
    ).fit(reference)
    return lof, isolation


def common_candidate_metrics(
    *,
    candidates: np.ndarray,
    factuals: np.ndarray,
    available: np.ndarray,
    class_success: np.ndarray,
    numerical: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    immutable: Sequence[int],
    sparsity_epsilon: float,
    lof: LocalOutlierFactor,
    isolation: IsolationForest,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute candidate metrics without method diagnostics or method identity."""
    flat_candidates = candidates[available]
    factual_grid = np.broadcast_to(factuals[:, None, :], candidates.shape)
    flat_factuals = factual_grid[available]
    flat_valid = class_success[available]
    numerical_columns = [int(column) for column in numerical]

    if len(flat_candidates):
        changed = np.abs(flat_candidates - flat_factuals) > sparsity_epsilon
        exact_changed = flat_candidates != flat_factuals
        grouped = grouped_gower_distance(
            flat_candidates, flat_factuals, numerical_columns, categorical_groups
        )
        action_counts = action_unit_change_count(
            flat_candidates,
            flat_factuals,
            numerical_columns,
            categorical_groups,
            numerical_tolerance=sparsity_epsilon,
        )
        immutable_columns = np.asarray(tuple(immutable), dtype=int)
        preserved = (
            np.ones(len(flat_candidates), dtype=bool)
            if not len(immutable_columns)
            else np.all(
                flat_candidates[:, immutable_columns]
                == flat_factuals[:, immutable_columns],
                axis=1,
            )
        )
        out_of_bounds = ((flat_candidates < 0) | (flat_candidates > 1)).any(axis=1)
        lof_scores = -lof.score_samples(flat_candidates)
        isolation_scores = isolation.decision_function(flat_candidates)
    else:
        changed = np.empty((0, candidates.shape[2]), dtype=bool)
        exact_changed = changed.copy()
        grouped = np.empty(0, dtype=float)
        action_counts = np.empty(0, dtype=float)
        preserved = np.empty(0, dtype=bool)
        out_of_bounds = np.empty(0, dtype=bool)
        lof_scores = np.empty(0, dtype=float)
        isolation_scores = np.empty(0, dtype=float)

    valid_candidates = flat_candidates[flat_valid]
    valid_factuals = flat_factuals[flat_valid]
    if len(valid_candidates) and numerical_columns:
        continuous = (
            valid_candidates[:, numerical_columns]
            - valid_factuals[:, numerical_columns]
        )
        manhattan = float(np.abs(continuous).sum(axis=1).mean())
        euclidean = float(np.linalg.norm(continuous, axis=1).mean())
    else:
        manhattan = float("nan")
        euclidean = float("nan")

    summary = {
        "actionability": float(preserved.mean()) if len(preserved) else float("nan"),
        "sparsity": float(changed.mean()) if changed.size else float("nan"),
        "sparsity_exact": (
            float(exact_changed.mean()) if exact_changed.size else float("nan")
        ),
        "action_unit_sparsity_mean": (
            float(action_counts.mean()) if len(action_counts) else float("nan")
        ),
        "proximity_grouped_gower": (
            float(grouped[flat_valid].mean()) if flat_valid.any() else float("nan")
        ),
        "proximity_continuous_manhattan": manhattan,
        "proximity_continuous_euclidean": euclidean,
        "lof_scores_cf": (
            float(lof_scores.mean()) if len(lof_scores) else float("nan")
        ),
        "isolation_forest_scores_cf": (
            float(isolation_scores.mean()) if len(isolation_scores) else float("nan")
        ),
        "cf_oob_fraction": (
            float(out_of_bounds.mean()) if len(out_of_bounds) else float("nan")
        ),
    }
    arrays = {
        "candidate.grouped_gower": grouped,
        "candidate.action_unit_changes": action_counts,
        "candidate.lof_score": lof_scores,
        "candidate.isolation_forest_score": isolation_scores,
    }
    return summary, arrays


def evaluate_diverse_candidate_sets(
    *,
    factuals: np.ndarray,
    candidates: np.ndarray,
    available: np.ndarray,
    class_success: np.ndarray,
    threshold_success: np.ndarray,
    numerical: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
) -> dict[str, float]:
    """Compute set coverage and diversity from canonical candidates."""
    requested = candidates.shape[1]
    counts = available.sum(axis=1)
    action_distances: list[float] = []
    value_distances: list[float] = []
    for point, count in enumerate(counts):
        rows = candidates[point][available[point]]
        signatures = [
            action_unit_signature(row, factuals[point], numerical, categorical_groups)
            for row in rows
        ]
        for left in range(int(count)):
            for right in range(left + 1, int(count)):
                action_distances.append(
                    action_set_jaccard_distance(signatures[left], signatures[right])
                )
                value_distances.append(
                    float(
                        grouped_gower_distance(
                            rows[left], rows[right], numerical, categorical_groups
                        )[0]
                    )
                )
    returned = int(available.sum())
    return {
        "set_coverage_at_k": float(np.mean(counts >= requested)),
        "set_returned_count_mean": float(np.mean(counts)),
        "set_validity_returned_class": (
            float(class_success.sum() / returned) if returned else float("nan")
        ),
        "set_validity_returned_threshold": (
            float(threshold_success.sum() / returned) if returned else float("nan")
        ),
        "set_action_jaccard_mean": (
            float(np.mean(action_distances)) if action_distances else float("nan")
        ),
        "set_action_jaccard_min": (
            float(np.min(action_distances)) if action_distances else float("nan")
        ),
        "set_pairwise_gower_mean": (
            float(np.mean(value_distances)) if value_distances else float("nan")
        ),
        "set_pairwise_gower_min": (
            float(np.min(value_distances)) if value_distances else float("nan")
        ),
    }


def compute_legacy_diverse_metrics(
    *,
    factuals: np.ndarray,
    oracle: Any,
    targets: np.ndarray,
    candidates: np.ndarray,
    counts: np.ndarray,
    probability_threshold: float,
    numerical: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
) -> dict[str, float]:
    """Legacy-name view of canonical set evaluation for retained runners."""
    sets = np.asarray(candidates, dtype=np.float64)
    available = np.arange(sets.shape[1])[None, :] < np.asarray(counts)[:, None]
    basic_metrics = {
        "diverse_coverage_at_k": float(np.mean(available.all(axis=1))),
        "diverse_returned_count_mean": float(np.mean(counts)),
    }
    if sets.shape[1] == 1:
        return basic_metrics
    predictions = np.full(available.shape, None, dtype=object)
    probabilities = np.full(available.shape, np.nan, dtype=float)
    if available.any():
        rows = sets[available]
        row_targets = np.broadcast_to(np.asarray(targets)[:, None], available.shape)[
            available
        ]
        predictions[available] = np.asarray(oracle.predict(rows)).reshape(-1)
        from experiments.zeroshot_cf.core.validation import target_probabilities

        probabilities[available] = target_probabilities(oracle, rows, row_targets)
    target_grid = np.broadcast_to(np.asarray(targets)[:, None], available.shape)
    class_success = available & (predictions == target_grid)
    threshold_success = class_success & (probabilities >= probability_threshold)
    canonical = evaluate_diverse_candidate_sets(
        factuals=np.asarray(factuals),
        candidates=sets,
        available=available,
        class_success=class_success,
        threshold_success=threshold_success,
        numerical=numerical,
        categorical_groups=categorical_groups,
    )
    factual_distances: list[float] = []
    action_counts: list[float] = []
    for point in range(len(sets)):
        rows = sets[point][available[point]]
        if len(rows):
            factual_distances.extend(
                grouped_gower_distance(
                    rows, factuals[point], numerical, categorical_groups
                ).tolist()
            )
            action_counts.extend(
                action_unit_change_count(
                    rows, factuals[point], numerical, categorical_groups
                ).tolist()
            )
    return {
        **basic_metrics,
        "diverse_returned_validity": canonical["set_validity_returned_threshold"],
        "diverse_action_jaccard_mean": canonical["set_action_jaccard_mean"],
        "diverse_action_jaccard_min": canonical["set_action_jaccard_min"],
        "diverse_pairwise_gower_mean": canonical["set_pairwise_gower_mean"],
        "diverse_pairwise_gower_min": canonical["set_pairwise_gower_min"],
        "diverse_factual_gower_mean": (
            float(np.mean(factual_distances)) if factual_distances else float("nan")
        ),
        "diverse_action_count_mean": (
            float(np.mean(action_counts)) if action_counts else float("nan")
        ),
    }


def compute_legacy_common_metrics(
    disc_model: Any,
    X_cf: np.ndarray,
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_target: np.ndarray,
    numerical_idx: Sequence[int],
    immutable_idx: Sequence[int] = (),
    *,
    categorical_groups: Sequence[OneHotActionGroup] = (),
    sparsity_eps: float = 0.05,
    lof_n_neighbors: int = 20,
    isolation_forest_estimators: int = 100,
) -> dict[str, float]:
    """Compatibility view backed by the same method-blind metric kernels."""
    if sparsity_eps < 0:
        raise ValueError("sparsity_eps must be non-negative")
    counterfactuals = np.asarray(X_cf, dtype=np.float64)
    factuals = np.asarray(X_test, dtype=np.float64)
    reference = np.asarray(X_train, dtype=np.float64)
    targets = np.asarray(y_target).reshape(-1)
    if counterfactuals.ndim != 2 or factuals.shape != counterfactuals.shape:
        raise ValueError("X_cf and X_test must be equally shaped 2D arrays")
    if reference.ndim != 2 or reference.shape[1] != counterfactuals.shape[1]:
        raise ValueError("X_cf and X_train must have the same columns")
    if targets.shape != (len(counterfactuals),):
        raise ValueError("y_target must contain one label per counterfactual")
    available = np.isfinite(counterfactuals).all(axis=1)[:, None]
    if not available.all():
        raise ValueError(
            "CounterContEx common metrics require complete counterfactuals; "
            f"coverage was {float(available.mean()):.4f}"
        )
    candidate_cube = counterfactuals[:, None, :]
    predictions = np.asarray(disc_model.predict(counterfactuals)).reshape(-1)
    class_success = (predictions == targets)[:, None]
    lof, isolation = prepare_novelty_models(
        reference,
        lof_n_neighbors=lof_n_neighbors,
        isolation_forest_estimators=isolation_forest_estimators,
    )
    metrics, _ = common_candidate_metrics(
        candidates=candidate_cube,
        factuals=factuals,
        available=available,
        class_success=class_success,
        numerical=numerical_idx,
        categorical_groups=categorical_groups,
        immutable=immutable_idx,
        sparsity_epsilon=sparsity_eps,
        lof=lof,
        isolation=isolation,
    )
    metrics.update(
        {
            "coverage": 1.0,
            "validity": float(class_success.mean()),
            "lof_scores_test": float((-lof.score_samples(factuals)).mean()),
            "isolation_forest_scores_test": float(
                isolation.decision_function(factuals).mean()
            ),
        }
    )
    return {
        key: metrics[key]
        for key in (
            "coverage",
            "validity",
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
    }
