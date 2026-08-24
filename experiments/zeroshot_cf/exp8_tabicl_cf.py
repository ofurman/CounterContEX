#  Copyright (c) Prior Labs GmbH 2026.

"""Experiment 8: greedy counterfactuals with the TabICLv2 backend.

This runner intentionally uses one fixed configuration:

* selector: ``prob_ascent``
* context: 512 nearest neighbours from both classes (``knn_both@512``)
* labels: predictions of the discriminator being explained
* iterative greedy search; on mixed data, numerical proposals and atomic
  categorical swaps compete globally at every step

Numerical candidate interventions for each greedy step are expanded into one
matrix and imputed in one TabICL call. They are then scored together with every
legal whole-category swap. While no proposal is valid, the search commits the
largest target-probability improvement. Once valid proposals exist, it commits
the lowest grouped-Gower row, with each categorical group counted once rather
than by dummy-column width. The overall search remains iterative. Context
remains per-factual because the kNN context is query-specific.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.data import get_one_hot_groups
from experiments.zeroshot_cf.exp4_greedy_cf import (
    _DATASET_PARAMS,
    TAU,
    evaluate_and_report,
)
from experiments.zeroshot_cf.mixed_distance import (
    action_unit_change_count,
    grouped_gower_distance,
)
from experiments.zeroshot_cf.tabicl_joint_plausibility import (
    TabICLJointScorer,
)

RESULTS_DIR = Path(__file__).parent / "results"
ATHENA_CONTEXT_SIZE = 512
ATHENA_CONTEXT_STRATEGY = "knn_both"
CATEGORICAL_PROPOSAL_COUNT = 1
DEFAULT_TEMPERATURE = 1e-9  # deterministic point estimate / categorical mode
DEFAULT_N_ESTIMATORS = 4
DEFAULT_POINT_ESTIMATE = "mode"
CF_MODES = ("sparse", "data_plausible")


def empirical_confidence_grid(
    confidences: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    quantile_levels: tuple[float, ...],
) -> tuple[float, ...]:
    """Derive query-confidence candidates from the selected target-class rows."""
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if levels.ndim != 1 or len(levels) == 0:
        raise ValueError("confidence quantile levels must be a non-empty sequence")
    if np.any((levels <= 0) | (levels >= 1)) or np.any(np.diff(levels) <= 0):
        raise ValueError(
            "confidence quantile levels must be strictly increasing inside (0, 1)"
        )
    scores = np.asarray(confidences, dtype=np.float64)
    context_labels = np.asarray(labels)
    target_scores = scores[context_labels == target_class]
    if len(target_scores) == 0:
        target_scores = scores
    values = np.quantile(target_scores, levels)
    return tuple(float(v) for v in np.unique(values))


def _resolve_max_test(dataset_name: str, max_test: int | None) -> int | None:
    if max_test is not None and max_test < 0:
        return None
    if max_test is not None:
        return max_test
    return _DATASET_PARAMS.get(dataset_name, {"max_test": 50})["max_test"]


def _select_test_rows(
    X_test: np.ndarray,
    y_test: np.ndarray,
    limit: int | None,
    selection: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic held-out evaluation subset."""
    if selection not in {"first", "stratified"}:
        raise ValueError("test_selection must be 'first' or 'stratified'")
    if limit is None or limit >= len(X_test):
        return X_test, y_test
    if limit <= 0:
        raise ValueError("max_test must be positive or -1 for the full test set")
    if selection == "first":
        return X_test[:limit], y_test[:limit]

    from sklearn.model_selection import train_test_split

    if limit < len(np.unique(y_test)):
        rng = np.random.default_rng(42)
        selected = np.sort(rng.choice(len(X_test), size=limit, replace=False))
        return X_test[selected], y_test[selected]

    selected, _ = train_test_split(
        np.arange(len(X_test)),
        train_size=limit,
        random_state=42,
        stratify=y_test,
    )
    selected.sort()
    return X_test[selected], y_test[selected]


def generate_tabicl_counterfactuals(
    dataset_name: str,
    *,
    tau: float = TAU,
    temperature: float = DEFAULT_TEMPERATURE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_test: int | None = None,
    candidate_quantiles: tuple[float, ...] | None = None,
    confidence_quantiles: tuple[float, ...] | None = None,
    cf_mode: str = "sparse",
    tabicl_joint_permutations: int = 1,
    max_validity_steps: int | None = None,
    allow_revisits: bool = True,
    joint_shortlist_size: int = 16,
    max_extra_actions: int = 1,
    min_joint_log_gain: float = 0.0,
    validation_fraction: float = 0.0,
    test_selection: str = "first",
    drop_heloc_all_minus9: bool = False,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate TabICL counterfactuals under the fixed configuration."""
    if candidate_quantiles is not None:
        candidate_quantiles = tuple(float(q) for q in candidate_quantiles)
    if confidence_quantiles is not None:
        confidence_quantiles = tuple(float(q) for q in confidence_quantiles)
        if candidate_quantiles is None:
            raise ValueError("confidence_quantiles require candidate_quantiles")
    if cf_mode not in CF_MODES:
        raise ValueError(f"cf_mode must be one of {CF_MODES}, got {cf_mode!r}")
    if tabicl_joint_permutations < 1:
        raise ValueError("tabicl_joint_permutations must be positive")
    if max_validity_steps is not None and max_validity_steps < 1:
        raise ValueError("max_validity_steps must be at least 1")
    if joint_shortlist_size < 1:
        raise ValueError("joint_shortlist_size must be at least 1")
    if max_extra_actions < 0:
        raise ValueError("max_extra_actions must be non-negative")
    if min_joint_log_gain < 0:
        raise ValueError("min_joint_log_gain must be non-negative")
    if test_selection not in {"first", "stratified"}:
        raise ValueError("test_selection must be 'first' or 'stratified'")

    from experiments.zeroshot_cf.data import (
        get_grouped_categorical_action_space,
        load_dataset,
    )
    from experiments.zeroshot_cf.discriminator import train_discriminator
    from experiments.zeroshot_cf.greedy import infer_feature_domains
    from experiments.zeroshot_cf.grouped_categorical import (
        CompactMixedSampler,
        ConditionedCategoryDistribution,
        GroupedCategoricalCodec,
        greedy_mixed_counterfactual,
    )
    from experiments.zeroshot_cf.tabicl_checkpoints import TABICL_DEVICE
    from experiments.zeroshot_cf.tabicl_sampler import (
        TabICLConditionalDensitySampler,
    )

    limit = _resolve_max_test(dataset_name, max_test)
    bundle = load_dataset(
        dataset_name,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
        validation_fraction=validation_fraction,
    )
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test, y_test = _select_test_rows(
        bundle.X_test,
        bundle.y_test,
        limit,
        test_selection,
    )
    (
        numerical_actionable_idx,
        grouped_actionable,
        immutable_idx,
    ) = get_grouped_categorical_action_space(bundle)
    all_one_hot_groups = get_one_hot_groups(bundle)
    categorical_codec = None
    if all_one_hot_groups:
        categorical_codec = GroupedCategoricalCodec.from_matrix(
            X_train,
            all_one_hot_groups,
        )
    actionable_idx = list(numerical_actionable_idx)
    for group in grouped_actionable:
        actionable_idx.extend(group.columns)
    effective_max_validity_steps = (
        len(numerical_actionable_idx) + len(grouped_actionable)
        if max_validity_steps is None
        else max_validity_steps
    )

    discriminator_cache_tag = (
        f"{dataset_name}_drop_all_minus9"
        if bundle.preprocessing_variant == "drop_heloc_all_minus9"
        else dataset_name
    )
    if bundle.X_val is not None:
        discriminator_cache_tag = f"{discriminator_cache_tag}_{bundle.split_variant}"
    X_disc_eval = bundle.X_val if bundle.X_val is not None else X_test
    y_disc_eval = bundle.y_val if bundle.y_val is not None else y_test
    disc_model = train_discriminator(
        X_train,
        y_train,
        X_disc_eval,
        y_disc_eval,
        discriminator_cache_tag,
    )
    y_pred = disc_model.predict(X_test)
    y_target = 1 - y_pred
    y_context = disc_model.predict(X_train)
    context_probabilities = (
        np.asarray(disc_model.predict_proba(X_train))
        if confidence_quantiles is not None
        else None
    )

    print(f"\n=== Experiment 8 (TabICL): {dataset_name.upper()} ===")
    print(
        f"  selector=prob_ascent, context={ATHENA_CONTEXT_STRATEGY}"
        f"@{ATHENA_CONTEXT_SIZE}, labels=disc, "
        f"point_estimate={DEFAULT_POINT_ESTIMATE}, project_to_domain=True, "
        f"candidate_quantiles={candidate_quantiles}, "
        f"confidence_quantiles={confidence_quantiles}, "
        f"cf_mode={cf_mode}, joint_shortlist_size={joint_shortlist_size}, "
        f"max_extra_actions={max_extra_actions}, "
        f"min_joint_log_gain={min_joint_log_gain}, "
        f"max_validity_steps={effective_max_validity_steps}, "
        f"allow_revisits={allow_revisits}, "
        f"split={bundle.split_variant}, test_selection={test_selection}, "
        f"preprocessing={bundle.preprocessing_variant}, "
        f"n_dropped_rows={bundle.n_dropped_rows}, "
        f"temperature={temperature}, "
        f"n_estimators={n_estimators}, n_test={len(X_test)}"
    )
    print(
        f"  Features: {X_train.shape[1]} total, "
        f"{len(numerical_actionable_idx)} scalar actionable, "
        f"{len(grouped_actionable)} grouped categorical actionable, "
        f"{len(immutable_idx)} immutable"
    )

    sampler_context = TabICLConditionalDensitySampler(
        n_estimators=n_estimators,
        temperature=temperature,
        random_state=42,
        device=TABICL_DEVICE,
        cache_dir=cache_dir,
        numerical_point_estimate=DEFAULT_POINT_ESTIMATE,
        categorical_features=(
            None if categorical_codec is None else categorical_codec.categorical_columns
        ),
    )
    if categorical_codec is None:
        X_sampler_train = X_train
        sampler = sampler_context
    else:
        X_sampler_train = categorical_codec.encode(X_train)
        sampler = CompactMixedSampler(sampler_context, categorical_codec)
    joint_sampler_context = None
    joint_sampler = None
    if cf_mode == "data_plausible":
        # Whole-row scoring uses its own [X, Y] context. Proposal confidence
        # conditioning remains available on ``sampler_context`` but is not
        # folded into the meaning of complete-row density.
        joint_sampler_context = TabICLConditionalDensitySampler(
            n_estimators=n_estimators,
            temperature=temperature,
            random_state=42,
            device=TABICL_DEVICE,
            cache_dir=cache_dir,
            numerical_point_estimate=DEFAULT_POINT_ESTIMATE,
            categorical_features=(
                None
                if categorical_codec is None
                else categorical_codec.categorical_columns
            ),
        )
        joint_sampler = (
            joint_sampler_context
            if categorical_codec is None
            else CompactMixedSampler(joint_sampler_context, categorical_codec)
        )
    feature_domains = infer_feature_domains(X_train)

    X_cf = X_test.copy()
    X_sparse = X_test.copy()
    changed_per_point: list[list[int]] = [[] for _ in range(len(X_test))]
    flipped_per_point = [False] * len(X_test)
    steps_per_point = [0] * len(X_test)
    history_per_point: list[list[tuple]] = [[] for _ in range(len(X_test))]
    attempt_history_per_point: list[list[tuple]] = [[] for _ in range(len(X_test))]
    validity_steps_per_point = [0] * len(X_test)
    initial_valid_step_per_point: list[int | None] = [None for _ in range(len(X_test))]
    refinement_steps_per_point = [0] * len(X_test)
    accepted_refinement_count_per_point = [0] * len(X_test)
    initial_sparse_action_count_per_point = np.full(len(X_test), -1, dtype=int)
    final_action_count_per_point = np.zeros(len(X_test), dtype=int)
    initial_tabicl_joint_log_density_per_point = np.full(len(X_test), np.nan)
    final_tabicl_joint_log_density_per_point = np.full(len(X_test), np.nan)
    tabicl_joint_log_density_gain_per_point = np.full(len(X_test), np.nan)
    joint_scoring_batch_count_per_point = np.zeros(len(X_test), dtype=int)
    joint_rows_scored_per_point = np.zeros(len(X_test), dtype=int)
    extra_actions_per_point = np.zeros(len(X_test), dtype=int)
    refinement_stopping_reason_per_point = ["not_started"] * len(X_test)
    point_runtime_s = np.zeros(len(X_test), dtype=np.float64)
    joint_scoring_runtime_s_per_point = np.zeros(len(X_test), dtype=np.float64)

    started = time.perf_counter()
    for i, (x, target) in enumerate(zip(X_test, y_target, strict=True)):
        point_started = time.perf_counter()
        # Both-class pool with a per-factual 512-row kNN context.
        sampler_query = (
            x if categorical_codec is None else categorical_codec.encode_row(x)
        )
        sampler_context.set_context(
            X_sampler_train,
            y_context=y_context,
            confidence_context=(
                None
                if context_probabilities is None
                else context_probabilities[:, int(target)]
            ),
            target_class=None,
            max_context=ATHENA_CONTEXT_SIZE,
            selection="knn",
            query=sampler_query,
        )
        confidence_grid = None
        if confidence_quantiles is not None:
            selected_confidences = sampler_context.selected_confidences_
            selected_labels = sampler_context.selected_labels_
            if selected_confidences is None or selected_labels is None:
                raise RuntimeError(
                    "confidence-conditioned context diagnostics are unavailable"
                )
            confidence_grid = empirical_confidence_grid(
                selected_confidences,
                selected_labels,
                int(target),
                confidence_quantiles,
            )
        target_class = int(target)
        point_confidence_grid = confidence_grid

        tabicl_joint_plausibility = None
        if cf_mode == "data_plausible":
            if joint_sampler_context is None or joint_sampler is None:
                raise RuntimeError("TabICL joint scorer is unavailable")
            joint_sampler_context.set_context(
                X_sampler_train,
                y_context=y_context,
                confidence_context=None,
                target_class=None,
                max_context=ATHENA_CONTEXT_SIZE,
                selection="knn",
                query=sampler_query,
            )
            tabicl_joint_plausibility = TabICLJointScorer(
                sampler=joint_sampler,
                target_class=target_class,
                n_permutations=tabicl_joint_permutations,
            )

        category_distribution: ConditionedCategoryDistribution | None = None
        if categorical_codec is not None and grouped_actionable:
            category_distribution_cache: dict[
                tuple[bytes, str], tuple[np.ndarray, np.ndarray]
            ] = {}
            categorical_confidence_anchors = (
                (None,)
                if point_confidence_grid is None
                else tuple(float(value) for value in point_confidence_grid)
            )

            def conditioned_category_distribution(
                row: np.ndarray,
                group: Any,
                confidence: float | None,
                _target_class: int = target_class,
                _cache: dict[
                    tuple[bytes, str], tuple[np.ndarray, np.ndarray]
                ] = category_distribution_cache,
                _anchors: tuple[float | None, ...] = (
                    categorical_confidence_anchors
                ),
                _codec: Any = categorical_codec,
                _sampler_context: Any = sampler_context,
            ) -> tuple[np.ndarray, np.ndarray]:
                key = (np.ascontiguousarray(row).tobytes(), group.name)
                if key not in _cache:
                    encoded_row = _codec.encode_row(row)
                    encoded_col = _codec.encoded_column_for_group(group)
                    fixed_confidences = (
                        None
                        if _anchors == (None,)
                        else np.asarray(
                            _anchors,
                            dtype=np.float32,
                        )
                    )
                    categories, probability_grid = (
                        _sampler_context.categorical_distribution(
                            encoded_row.reshape(1, -1),
                            encoded_col,
                            fixed_target=_target_class,
                            fixed_confidence=fixed_confidences,
                        )
                    )
                    _cache[key] = (
                        np.asarray(categories, dtype=int),
                        np.atleast_2d(
                            np.asarray(probability_grid, dtype=np.float64)
                        ),
                    )
                categories, probability_grid = _cache[key]
                if confidence is None:
                    anchor_index = 0
                else:
                    matches = np.flatnonzero(
                        np.isclose(
                            np.asarray(_anchors, dtype=float),
                            confidence,
                        )
                    )
                    if not len(matches):
                        raise ValueError(
                            f"unknown categorical confidence anchor: {confidence}"
                        )
                    anchor_index = int(matches[0])
                return categories, probability_grid[anchor_index]

            category_distribution = conditioned_category_distribution

        x_cf, changed, greedy_info = greedy_mixed_counterfactual(
            sampler,
            disc_model,
            x,
            target_class,
            numerical_actionable_idx,
            grouped_actionable,
            candidate_quantiles=candidate_quantiles,
            candidate_confidences=point_confidence_grid,
            feature_domains=feature_domains,
            cf_mode=cf_mode,
            tabicl_joint_plausibility=tabicl_joint_plausibility,
            max_validity_steps=effective_max_validity_steps,
            allow_revisits=allow_revisits,
            joint_shortlist_size=joint_shortlist_size,
            max_extra_actions=max_extra_actions,
            min_joint_log_gain=min_joint_log_gain,
            tau=tau,
            temperature=temperature,
            category_distribution=category_distribution,
            categorical_proposal_count=CATEGORICAL_PROPOSAL_COUNT,
        )
        X_cf[i] = x_cf
        initial_sparse_row = greedy_info.get("initial_sparse_row")
        if initial_sparse_row is not None:
            X_sparse[i] = np.asarray(initial_sparse_row, dtype=X_sparse.dtype)
        changed_per_point[i] = changed
        flipped_per_point[i] = greedy_info["flipped"]
        steps_per_point[i] = greedy_info["steps"]
        history_per_point[i] = greedy_info["history"]
        attempt_history_per_point[i] = greedy_info["attempt_history"]
        validity_steps_per_point[i] = greedy_info["validity_steps"]
        initial_valid_step_per_point[i] = greedy_info.get("initial_valid_step")
        refinement_steps_per_point[i] = greedy_info.get("refinement_steps", 0)
        accepted_refinement_count_per_point[i] = greedy_info.get(
            "accepted_refinement_count", 0
        )
        initial_action_count = greedy_info.get("initial_sparse_action_count")
        if initial_action_count is not None:
            initial_sparse_action_count_per_point[i] = int(initial_action_count)
        final_action_count_per_point[i] = int(
            greedy_info.get("final_action_count", len(changed))
        )
        initial_joint_score = greedy_info.get("initial_tabicl_joint_log_density")
        if initial_joint_score is not None:
            initial_tabicl_joint_log_density_per_point[i] = float(
                initial_joint_score
            )
        final_joint_score = greedy_info.get("final_tabicl_joint_log_density")
        if final_joint_score is not None:
            final_tabicl_joint_log_density_per_point[i] = float(final_joint_score)
        joint_score_gain = greedy_info.get("tabicl_joint_log_density_gain")
        if joint_score_gain is not None:
            tabicl_joint_log_density_gain_per_point[i] = float(joint_score_gain)
        joint_scoring_batch_count_per_point[i] = int(
            greedy_info.get("joint_scoring_batch_count", 0)
        )
        joint_rows_scored_per_point[i] = int(
            greedy_info.get("joint_rows_scored", 0)
        )
        extra_actions_per_point[i] = int(greedy_info.get("extra_actions", 0))
        refinement_stopping_reason_per_point[i] = str(
            greedy_info.get("refinement_stopping_reason", "unknown")
        )
        joint_scoring_runtime_s_per_point[i] = float(
            greedy_info.get("joint_scoring_runtime_s", 0.0)
        )
        point_runtime_s[i] = time.perf_counter() - point_started
        if i == 0:
            first_s = time.perf_counter() - started
            print(
                f"  [timing] first point: {first_s:.2f}s "
                f"(~{first_s * len(X_test) / 60:.1f} min linear estimate)"
            )

    runtime_s = time.perf_counter() - started
    target_probability_per_point = np.asarray(disc_model.predict_proba(X_cf))[
        np.arange(len(X_cf)), y_target.astype(int)
    ]
    info: dict[str, Any] = {
        "bundle": bundle,
        "y_pred": y_pred,
        "y_target": y_target,
        "actionable_idx": actionable_idx,
        "immutable_idx": immutable_idx,
        "disc_model": disc_model,
        "temperature": temperature,
        "candidate_quantiles": candidate_quantiles,
        "confidence_quantiles": confidence_quantiles,
        "cf_mode": cf_mode,
        "plausibility_backend": (
            "tabicl_joint_one_shot"
            if cf_mode == "data_plausible"
            else "proposal_support"
        ),
        "max_validity_steps": effective_max_validity_steps,
        "allow_revisits": allow_revisits,
        "joint_shortlist_size": joint_shortlist_size,
        "max_extra_actions": max_extra_actions,
        "min_joint_log_gain": min_joint_log_gain,
        "categorical_proposal_count": CATEGORICAL_PROPOSAL_COUNT,
        "categorical_confidence_batching": True,
        "conditional_estimator_cache": True,
        "tabicl_kv_cache": sampler_context.estimator_params.get("kv_cache", False),
        "test_selection": test_selection,
        "split_variant": bundle.split_variant,
        "preprocessing_variant": bundle.preprocessing_variant,
        "n_dropped_rows": bundle.n_dropped_rows,
        "n_estimators": n_estimators,
        "runtime_s": runtime_s,
        "X_sparse": X_sparse,
        "point_runtime_s": point_runtime_s,
        "joint_scoring_runtime_s_per_point": (
            joint_scoring_runtime_s_per_point
        ),
        "changed_per_point": changed_per_point,
        "flipped_per_point": flipped_per_point,
        "steps_per_point": steps_per_point,
        "history_per_point": history_per_point,
        "attempt_history_per_point": attempt_history_per_point,
        "validity_steps_per_point": validity_steps_per_point,
        "initial_valid_step_per_point": initial_valid_step_per_point,
        "refinement_steps_per_point": refinement_steps_per_point,
        "accepted_refinement_count_per_point": (
            accepted_refinement_count_per_point
        ),
        "initial_sparse_action_count_per_point": (
            initial_sparse_action_count_per_point
        ),
        "final_action_count_per_point": final_action_count_per_point,
        "initial_tabicl_joint_log_density_per_point": (
            initial_tabicl_joint_log_density_per_point
        ),
        "final_tabicl_joint_log_density_per_point": (
            final_tabicl_joint_log_density_per_point
        ),
        "tabicl_joint_log_density_gain_per_point": (
            tabicl_joint_log_density_gain_per_point
        ),
        "joint_scoring_batch_count_per_point": (
            joint_scoring_batch_count_per_point
        ),
        "joint_rows_scored_per_point": joint_rows_scored_per_point,
        "extra_actions_per_point": extra_actions_per_point,
        "refinement_stopping_reason_per_point": (
            refinement_stopping_reason_per_point
        ),
        "target_probability_per_point": target_probability_per_point,
    }
    return X_test, y_test, X_cf, info


def run_and_report(
    dataset_name: str,
    **kwargs: Any,
) -> dict[str, float]:
    """Run one dataset, evaluate it, and write one backend-comparison row."""
    X_test, y_test, X_cf, info = generate_tabicl_counterfactuals(dataset_name, **kwargs)
    metrics = evaluate_and_report(
        dataset_name,
        X_test,
        y_test,
        X_cf,
        info,
        write_csv=False,
    )
    bundle = info["bundle"]
    categorical_groups = get_one_hot_groups(bundle)
    valid = np.asarray(info["disc_model"].predict(X_cf)) == info["y_target"]
    grouped_gower = grouped_gower_distance(
        X_cf,
        X_test,
        bundle.numerical_features_indices,
        categorical_groups,
    )
    action_counts = action_unit_change_count(
        X_cf,
        X_test,
        bundle.numerical_features_indices,
        categorical_groups,
        numerical_tolerance=0.05,
    )
    metrics["proximity_grouped_gower"] = (
        float(grouped_gower[valid].mean()) if valid.any() else float("nan")
    )
    metrics["action_unit_sparsity_mean"] = float(action_counts.mean())
    initial_action_counts = np.asarray(
        info["initial_sparse_action_count_per_point"], dtype=float
    )
    reached_validity = initial_action_counts >= 0
    initial_action_count_mean = (
        float(initial_action_counts[reached_validity].mean())
        if np.any(reached_validity)
        else float("nan")
    )

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
        "joint_scoring_batch_count_mean": float(
            np.mean(info["joint_scoring_batch_count_per_point"])
        ),
        "joint_rows_scored_mean": float(
            np.mean(info["joint_rows_scored_per_point"])
        ),
        "accepted_refinement_count_mean": float(
            np.mean(info["accepted_refinement_count_per_point"])
        ),
        "initial_sparse_action_count_mean": initial_action_count_mean,
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"exp8_tabicl_{dataset_name}_metrics.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"\n  Wrote {output}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TabICL greedy counterfactuals with a fixed kNN context"
    )
    parser.add_argument(
        "--dataset",
        choices=["moons", "heloc", "german_credit", "all"],
        default="moons",
    )
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Default: moons=100, heloc=50; use -1 for the full test split.",
    )
    parser.add_argument(
        "--candidate-quantiles",
        type=float,
        nargs="+",
        default=None,
        metavar="Q",
        help=(
            "Generate deterministic conditional proposals per feature. Prefer "
            "a central grid, e.g. --candidate-quantiles 0.1 0.3 0.5 0.7 0.9."
        ),
    )
    parser.add_argument(
        "--confidence-quantiles",
        type=float,
        nargs="+",
        default=None,
        metavar="Q",
        help=(
            "Quantile levels of the selected context's target-class confidence "
            "distribution. The resulting empirical confidence values are appended "
            "to TabICL queries; requires --candidate-quantiles."
        ),
    )
    parser.add_argument(
        "--cf-mode",
        choices=["sparse", "data-plausible"],
        default="sparse",
        help=(
            "Sparse stops after selecting the lowest grouped-Gower valid "
            "proposal; data-plausible then performs one bounded TabICL "
            "complete-row reranking batch."
        ),
    )
    parser.add_argument(
        "--tabicl-joint-permutations",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-validity-steps",
        type=int,
        default=None,
        help=(
            "Maximum committed probability-ascent actions before validity. "
            "Defaults to the number of actionable feature units."
        ),
    )
    parser.add_argument(
        "--allow-revisits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow an action unit to be proposed again after another edit.",
    )
    parser.add_argument(
        "--joint-shortlist-size",
        type=int,
        default=16,
        help="Maximum valid alternatives in the one-shot whole-row batch.",
    )
    parser.add_argument(
        "--max-extra-actions",
        type=int,
        default=1,
        help="Maximum final action-count increase over the sparse CFE.",
    )
    parser.add_argument(
        "--min-joint-log-gain",
        type=float,
        default=0.0,
        help="Minimum raw whole-row log-density gain over the sparse CFE.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of the provisional 80%% train partition reserved for "
            "validation; 0.2 gives a fixed 64/16/20 split."
        ),
    )
    parser.add_argument(
        "--test-selection",
        choices=["first", "stratified"],
        default="first",
        help="How --max-test selects held-out factuals (default: first).",
    )
    parser.add_argument(
        "--drop-heloc-all-minus9",
        action="store_true",
        help=(
            "Before splitting HELOC, remove rows whose 23 predictors are all "
            "the -9 no-bureau-record sentinel."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    datasets = (
        ["moons", "heloc", "german_credit"] if args.dataset == "all" else [args.dataset]
    )
    for dataset_name in datasets:
        run_and_report(
            dataset_name,
            tau=args.tau,
            temperature=args.temperature,
            n_estimators=args.n_estimators,
            max_test=args.max_test,
            candidate_quantiles=(
                None
                if args.candidate_quantiles is None
                else tuple(args.candidate_quantiles)
            ),
            confidence_quantiles=(
                None
                if args.confidence_quantiles is None
                else tuple(args.confidence_quantiles)
            ),
            cf_mode=args.cf_mode.replace("-", "_"),
            tabicl_joint_permutations=args.tabicl_joint_permutations,
            max_validity_steps=args.max_validity_steps,
            allow_revisits=args.allow_revisits,
            joint_shortlist_size=args.joint_shortlist_size,
            max_extra_actions=args.max_extra_actions,
            min_joint_log_gain=args.min_joint_log_gain,
            validation_fraction=args.validation_fraction,
            test_selection=args.test_selection,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
