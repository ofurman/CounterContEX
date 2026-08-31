"""Stable public API for retained TabICL counterfactual generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.candidate_domains import FeatureDomains
from experiments.zeroshot_cf.diverse_search import (
    DiverseBeamSearchConfig,
    generate_diverse_counterfactuals,
)
from experiments.zeroshot_cf.grouped_categorical import (
    ConditionedCategoryDistribution,
    greedy_mixed_counterfactual,
)
from experiments.zeroshot_cf.mixed_distance import action_unit_change_count

ATHENA_CONTEXT_SIZE = 512
ATHENA_CONTEXT_STRATEGY = "gower_knn_both"
DEFAULT_CATEGORICAL_PROPOSAL_COUNT = 1
DEFAULT_TEMPERATURE = 1e-9
DEFAULT_N_ESTIMATORS = 4
DEFAULT_POINT_ESTIMATE = "mode"
CF_MODES = ("sparse", "data_plausible")


class DiscriminatorProtocol(Protocol):
    """Target classifier interface required by the public generator API."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class TabICLGeneratorConfig:
    """Stable search configuration for single or multiple counterfactuals."""

    tau: float = 0.5
    temperature: float = DEFAULT_TEMPERATURE
    candidate_quantiles: tuple[float, ...] | None = None
    confidence_quantiles: tuple[float, ...] | None = None
    cf_mode: str = "sparse"
    tabicl_joint_permutations: int = 1
    max_validity_steps: int | None = None
    allow_revisits: bool = True
    joint_shortlist_size: int = 16
    max_extra_actions: int = 1
    min_joint_log_gain: float = 0.0
    diversity_config: DiverseBeamSearchConfig = field(
        default_factory=lambda: DiverseBeamSearchConfig(n_counterfactuals=1)
    )
    categorical_proposal_count: int = DEFAULT_CATEGORICAL_PROPOSAL_COUNT

    def __post_init__(self) -> None:
        if self.cf_mode not in CF_MODES:
            raise ValueError(f"cf_mode must be one of {CF_MODES}, got {self.cf_mode!r}")
        if self.confidence_quantiles is not None and self.candidate_quantiles is None:
            raise ValueError("confidence_quantiles require candidate_quantiles")
        if self.tabicl_joint_permutations < 1:
            raise ValueError("tabicl_joint_permutations must be positive")
        if self.max_validity_steps is not None and self.max_validity_steps < 1:
            raise ValueError("max_validity_steps must be at least 1")
        if self.joint_shortlist_size < 1:
            raise ValueError("joint_shortlist_size must be at least 1")
        if self.max_extra_actions < 0:
            raise ValueError("max_extra_actions must be non-negative")
        if self.min_joint_log_gain < 0:
            raise ValueError("min_joint_log_gain must be non-negative")
        if self.categorical_proposal_count < 1:
            raise ValueError("categorical_proposal_count must be at least 1")
        if self.n_counterfactuals > 1 and self.cf_mode != "sparse":
            raise ValueError(
                "the separate diverse generator currently requires sparse mode"
            )

    @property
    def n_counterfactuals(self) -> int:
        return self.diversity_config.n_counterfactuals


@dataclass(frozen=True)
class TabICLGeneratorInputs:
    """Per-batch factuals and structural constraints for retained search."""

    factuals: np.ndarray
    targets: np.ndarray
    numerical_columns: tuple[int, ...]
    categorical_groups: tuple[OneHotActionGroup, ...]
    immutable_idx: tuple[int, ...] = ()
    feature_domains: FeatureDomains | None = None


@dataclass(frozen=True)
class TabICLGeneratorPointBackend:
    """Prepared per-factual backend state for the stable generator."""

    sampler: Any
    candidate_confidences: tuple[float, ...] | None = None
    category_distribution: ConditionedCategoryDistribution | None = None
    joint_scorer: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TabICLGeneratorDiagnostics:
    """Typed runtime diagnostics emitted by the stable generator API."""

    tau: float
    temperature: float
    candidate_quantiles: tuple[float, ...] | None
    confidence_quantiles: tuple[float, ...] | None
    cf_mode: str
    plausibility_backend: str
    max_validity_steps: int
    allow_revisits: bool
    joint_shortlist_size: int
    max_extra_actions: int
    min_joint_log_gain: float
    n_counterfactuals: int
    diversity_config: DiverseBeamSearchConfig
    categorical_proposal_count: int
    categorical_confidence_batching: bool
    conditional_estimator_cache: bool
    tabicl_kv_cache: bool
    runtime_s: float
    point_runtime_s: np.ndarray
    joint_scoring_runtime_s_per_point: np.ndarray
    changed_per_point: tuple[tuple[int, ...], ...]
    flipped_per_point: tuple[bool, ...]
    steps_per_point: tuple[int, ...]
    history_per_point: tuple[tuple[Any, ...], ...]
    attempt_history_per_point: tuple[tuple[Any, ...], ...]
    validity_steps_per_point: tuple[int, ...]
    initial_valid_step_per_point: tuple[int | None, ...]
    refinement_steps_per_point: tuple[int, ...]
    accepted_refinement_count_per_point: tuple[int, ...]
    initial_sparse_action_count_per_point: np.ndarray
    final_action_count_per_point: np.ndarray
    initial_tabicl_joint_log_density_per_point: np.ndarray
    final_tabicl_joint_log_density_per_point: np.ndarray
    tabicl_joint_log_density_gain_per_point: np.ndarray
    joint_scoring_batch_count_per_point: np.ndarray
    joint_rows_scored_per_point: np.ndarray
    extra_actions_per_point: np.ndarray
    refinement_stopping_reason_per_point: tuple[str, ...]
    diverse_available_count_per_point: np.ndarray
    diverse_candidate_pool_count_per_point: np.ndarray
    diverse_search_depth_per_point: np.ndarray
    diverse_histories_per_point: tuple[tuple[tuple[Any, ...], ...], ...]
    target_probability_per_point: np.ndarray


@dataclass(frozen=True)
class TabICLGeneratorResult:
    """Counterfactual batch and typed diagnostics from one generator run."""

    factuals: np.ndarray
    targets: np.ndarray
    counterfactuals: np.ndarray
    sparse_counterfactuals: np.ndarray
    counterfactual_sets: np.ndarray
    diagnostics: TabICLGeneratorDiagnostics


PointBackendFactory = Callable[[np.ndarray, int], TabICLGeneratorPointBackend]


def empirical_confidence_grid(
    confidences: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    quantile_levels: tuple[float, ...],
) -> tuple[float, ...]:
    """Derive query-confidence anchors from the selected target-class rows."""
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
    return tuple(float(value) for value in np.unique(values))


def select_test_rows(
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


def _freeze_items(items: Sequence[Any]) -> tuple[Any, ...]:
    frozen: list[Any] = []
    for item in items:
        frozen.append(dict(item) if isinstance(item, dict) else item)
    return tuple(frozen)


def _freeze_nested_histories(histories: Sequence[Sequence[Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(_freeze_items(history) for history in histories)


def _target_probabilities(
    discriminator: DiscriminatorProtocol,
    rows: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    if len(rows) == 0:
        return np.empty(0, dtype=np.float64)
    probability_matrix = np.asarray(discriminator.predict_proba(rows))
    classes = np.asarray(
        getattr(discriminator, "classes_", np.arange(probability_matrix.shape[1]))
    )
    probabilities = np.empty(len(rows), dtype=np.float64)
    for index, target in enumerate(np.asarray(targets, dtype=int)):
        target_positions = np.flatnonzero(classes == target)
        if len(target_positions) != 1:
            raise ValueError(
                f"target class {target} is absent from classifier classes"
            )
        probabilities[index] = float(probability_matrix[index, int(target_positions[0])])
    return probabilities


def _metadata_bool(metadata: Mapping[str, Any], key: str, default: bool = False) -> bool:
    if key not in metadata:
        return default
    return bool(metadata[key])


def generate_counterfactual_batch(
    inputs: TabICLGeneratorInputs,
    *,
    discriminator: DiscriminatorProtocol,
    config: TabICLGeneratorConfig,
    point_backend_factory: PointBackendFactory,
) -> TabICLGeneratorResult:
    """Generate a batch of retained single- or multiple-counterfactual outputs."""
    factuals = np.asarray(inputs.factuals, dtype=np.float64)
    targets = np.asarray(inputs.targets, dtype=int).reshape(-1)
    if factuals.ndim != 2:
        raise ValueError(f"factuals must be 2D, got shape {factuals.shape}")
    if targets.shape != (len(factuals),):
        raise ValueError("targets must contain one label per factual row")

    numerical_columns = tuple(int(column) for column in inputs.numerical_columns)
    categorical_groups = tuple(inputs.categorical_groups)
    effective_max_validity_steps = (
        len(numerical_columns) + len(categorical_groups)
        if config.max_validity_steps is None
        else config.max_validity_steps
    )

    X_cf = factuals.copy()
    X_sparse = factuals.copy()
    X_cf_sets = np.full(
        (len(factuals), config.n_counterfactuals, factuals.shape[1]),
        np.nan,
        dtype=factuals.dtype,
    )
    diverse_available_count_per_point = np.zeros(len(factuals), dtype=int)
    diverse_candidate_pool_count_per_point = np.zeros(len(factuals), dtype=int)
    diverse_search_depth_per_point = np.zeros(len(factuals), dtype=int)
    diverse_histories_per_point: list[tuple[tuple[Any, ...], ...]] = [
        () for _ in range(len(factuals))
    ]
    changed_per_point: list[tuple[int, ...]] = [() for _ in range(len(factuals))]
    flipped_per_point = [False] * len(factuals)
    steps_per_point = [0] * len(factuals)
    history_per_point: list[tuple[Any, ...]] = [() for _ in range(len(factuals))]
    attempt_history_per_point: list[tuple[Any, ...]] = [() for _ in range(len(factuals))]
    validity_steps_per_point = [0] * len(factuals)
    initial_valid_step_per_point: list[int | None] = [None for _ in range(len(factuals))]
    refinement_steps_per_point = [0] * len(factuals)
    accepted_refinement_count_per_point = [0] * len(factuals)
    initial_sparse_action_count_per_point = np.full(len(factuals), -1, dtype=int)
    final_action_count_per_point = np.zeros(len(factuals), dtype=int)
    initial_tabicl_joint_log_density_per_point = np.full(len(factuals), np.nan)
    final_tabicl_joint_log_density_per_point = np.full(len(factuals), np.nan)
    tabicl_joint_log_density_gain_per_point = np.full(len(factuals), np.nan)
    joint_scoring_batch_count_per_point = np.zeros(len(factuals), dtype=int)
    joint_rows_scored_per_point = np.zeros(len(factuals), dtype=int)
    extra_actions_per_point = np.zeros(len(factuals), dtype=int)
    refinement_stopping_reason_per_point = ["not_started"] * len(factuals)
    point_runtime_s = np.zeros(len(factuals), dtype=np.float64)
    joint_scoring_runtime_s_per_point = np.zeros(len(factuals), dtype=np.float64)

    started = perf_counter()
    metadata: Mapping[str, Any] = {}
    for index, (factual, target_class) in enumerate(zip(factuals, targets, strict=True)):
        point_started = perf_counter()
        point_backend = point_backend_factory(np.asarray(factual).copy(), int(target_class))
        if index == 0:
            metadata = point_backend.metadata

        if config.n_counterfactuals == 1:
            if config.cf_mode == "data_plausible" and point_backend.joint_scorer is None:
                raise ValueError("data_plausible mode requires a prepared joint scorer")
            x_cf, changed, greedy_info = greedy_mixed_counterfactual(
                point_backend.sampler,
                discriminator,
                factual,
                int(target_class),
                numerical_columns,
                categorical_groups,
                candidate_quantiles=config.candidate_quantiles,
                candidate_confidences=point_backend.candidate_confidences,
                feature_domains=inputs.feature_domains,
                cf_mode=config.cf_mode,
                tabicl_joint_plausibility=point_backend.joint_scorer,
                max_validity_steps=effective_max_validity_steps,
                allow_revisits=config.allow_revisits,
                joint_shortlist_size=config.joint_shortlist_size,
                max_extra_actions=config.max_extra_actions,
                min_joint_log_gain=config.min_joint_log_gain,
                tau=config.tau,
                temperature=config.temperature,
                category_distribution=point_backend.category_distribution,
                categorical_proposal_count=config.categorical_proposal_count,
            )
            if greedy_info["flipped"]:
                X_cf_sets[index, 0] = x_cf
                diverse_available_count_per_point[index] = 1
                diverse_candidate_pool_count_per_point[index] = 1
        else:
            diverse_result = generate_diverse_counterfactuals(
                point_backend.sampler,
                discriminator,
                factual,
                int(target_class),
                numerical_columns,
                categorical_groups,
                config=config.diversity_config,
                candidate_quantiles=config.candidate_quantiles,
                candidate_confidences=point_backend.candidate_confidences,
                feature_domains=inputs.feature_domains,
                max_validity_steps=effective_max_validity_steps,
                allow_revisits=config.allow_revisits,
                tau=config.tau,
                temperature=config.temperature,
                category_distribution=point_backend.category_distribution,
            )
            available_count = diverse_result.available_count
            if available_count:
                X_cf_sets[index, :available_count] = diverse_result.counterfactuals
            diverse_available_count_per_point[index] = available_count
            diverse_candidate_pool_count_per_point[index] = (
                diverse_result.candidate_pool_count
            )
            diverse_search_depth_per_point[index] = diverse_result.search_depth
            diverse_histories_per_point[index] = _freeze_nested_histories(
                diverse_result.histories
            )
            if available_count:
                x_cf = diverse_result.counterfactuals[0].copy()
                primary_history = tuple(diverse_histories_per_point[index][0])
                primary_depth = int(diverse_result.depths[0])
                changed = tuple(np.flatnonzero(~np.isclose(x_cf, factual)).tolist())
            else:
                x_cf = factual.copy()
                primary_history = ()
                primary_depth = 0
                changed = ()
            primary_action_count = int(
                action_unit_change_count(
                    x_cf,
                    factual,
                    numerical_columns,
                    categorical_groups,
                )[0]
            )
            greedy_info = {
                "flipped": available_count > 0,
                "steps": primary_depth,
                "history": primary_history,
                "attempt_history": (),
                "validity_steps": primary_depth,
                "initial_valid_step": primary_depth if available_count else None,
                "initial_sparse_row": x_cf.copy() if available_count else None,
                "initial_sparse_action_count": (
                    primary_action_count if available_count else None
                ),
                "final_action_count": primary_action_count,
                "refinement_steps": 0,
                "accepted_refinement_count": 0,
                "extra_actions": 0,
                "refinement_stopping_reason": "diverse_method",
                "joint_scoring_batch_count": 0,
                "joint_rows_scored": 0,
                "joint_scoring_runtime_s": 0.0,
            }

        X_cf[index] = x_cf
        initial_sparse_row = greedy_info.get("initial_sparse_row")
        if initial_sparse_row is not None:
            X_sparse[index] = np.asarray(initial_sparse_row, dtype=X_sparse.dtype)
        changed_per_point[index] = tuple(int(column) for column in changed)
        flipped_per_point[index] = bool(greedy_info["flipped"])
        steps_per_point[index] = int(greedy_info["steps"])
        history_per_point[index] = _freeze_items(greedy_info["history"])
        attempt_history_per_point[index] = _freeze_items(
            greedy_info["attempt_history"]
        )
        validity_steps_per_point[index] = int(greedy_info["validity_steps"])
        initial_valid_step_per_point[index] = greedy_info.get("initial_valid_step")
        refinement_steps_per_point[index] = int(greedy_info.get("refinement_steps", 0))
        accepted_refinement_count_per_point[index] = int(
            greedy_info.get("accepted_refinement_count", 0)
        )
        initial_action_count = greedy_info.get("initial_sparse_action_count")
        if initial_action_count is not None:
            initial_sparse_action_count_per_point[index] = int(initial_action_count)
        final_action_count_per_point[index] = int(
            greedy_info.get("final_action_count", len(changed_per_point[index]))
        )
        initial_joint_score = greedy_info.get("initial_tabicl_joint_log_density")
        if initial_joint_score is not None:
            initial_tabicl_joint_log_density_per_point[index] = float(initial_joint_score)
        final_joint_score = greedy_info.get("final_tabicl_joint_log_density")
        if final_joint_score is not None:
            final_tabicl_joint_log_density_per_point[index] = float(final_joint_score)
        joint_score_gain = greedy_info.get("tabicl_joint_log_density_gain")
        if joint_score_gain is not None:
            tabicl_joint_log_density_gain_per_point[index] = float(joint_score_gain)
        joint_scoring_batch_count_per_point[index] = int(
            greedy_info.get("joint_scoring_batch_count", 0)
        )
        joint_rows_scored_per_point[index] = int(
            greedy_info.get("joint_rows_scored", 0)
        )
        extra_actions_per_point[index] = int(greedy_info.get("extra_actions", 0))
        refinement_stopping_reason_per_point[index] = str(
            greedy_info.get("refinement_stopping_reason", "unknown")
        )
        joint_scoring_runtime_s_per_point[index] = float(
            greedy_info.get("joint_scoring_runtime_s", 0.0)
        )
        point_runtime_s[index] = perf_counter() - point_started

    runtime_s = perf_counter() - started
    target_probability_per_point = _target_probabilities(discriminator, X_cf, targets)
    diagnostics = TabICLGeneratorDiagnostics(
        tau=config.tau,
        temperature=config.temperature,
        candidate_quantiles=config.candidate_quantiles,
        confidence_quantiles=config.confidence_quantiles,
        cf_mode=config.cf_mode,
        plausibility_backend=(
            "tabicl_joint_one_shot"
            if config.cf_mode == "data_plausible"
            else "proposal_support"
        ),
        max_validity_steps=effective_max_validity_steps,
        allow_revisits=config.allow_revisits,
        joint_shortlist_size=config.joint_shortlist_size,
        max_extra_actions=config.max_extra_actions,
        min_joint_log_gain=config.min_joint_log_gain,
        n_counterfactuals=config.n_counterfactuals,
        diversity_config=config.diversity_config,
        categorical_proposal_count=config.categorical_proposal_count,
        categorical_confidence_batching=_metadata_bool(
            metadata, "categorical_confidence_batching"
        ),
        conditional_estimator_cache=_metadata_bool(
            metadata, "conditional_estimator_cache"
        ),
        tabicl_kv_cache=_metadata_bool(metadata, "tabicl_kv_cache"),
        runtime_s=runtime_s,
        point_runtime_s=point_runtime_s,
        joint_scoring_runtime_s_per_point=joint_scoring_runtime_s_per_point,
        changed_per_point=tuple(changed_per_point),
        flipped_per_point=tuple(flipped_per_point),
        steps_per_point=tuple(steps_per_point),
        history_per_point=tuple(history_per_point),
        attempt_history_per_point=tuple(attempt_history_per_point),
        validity_steps_per_point=tuple(validity_steps_per_point),
        initial_valid_step_per_point=tuple(initial_valid_step_per_point),
        refinement_steps_per_point=tuple(refinement_steps_per_point),
        accepted_refinement_count_per_point=tuple(
            accepted_refinement_count_per_point
        ),
        initial_sparse_action_count_per_point=initial_sparse_action_count_per_point,
        final_action_count_per_point=final_action_count_per_point,
        initial_tabicl_joint_log_density_per_point=(
            initial_tabicl_joint_log_density_per_point
        ),
        final_tabicl_joint_log_density_per_point=(
            final_tabicl_joint_log_density_per_point
        ),
        tabicl_joint_log_density_gain_per_point=(
            tabicl_joint_log_density_gain_per_point
        ),
        joint_scoring_batch_count_per_point=joint_scoring_batch_count_per_point,
        joint_rows_scored_per_point=joint_rows_scored_per_point,
        extra_actions_per_point=extra_actions_per_point,
        refinement_stopping_reason_per_point=tuple(
            refinement_stopping_reason_per_point
        ),
        diverse_available_count_per_point=diverse_available_count_per_point,
        diverse_candidate_pool_count_per_point=diverse_candidate_pool_count_per_point,
        diverse_search_depth_per_point=diverse_search_depth_per_point,
        diverse_histories_per_point=tuple(diverse_histories_per_point),
        target_probability_per_point=target_probability_per_point,
    )
    return TabICLGeneratorResult(
        factuals=factuals,
        targets=targets,
        counterfactuals=X_cf,
        sparse_counterfactuals=X_sparse,
        counterfactual_sets=X_cf_sets,
        diagnostics=diagnostics,
    )
