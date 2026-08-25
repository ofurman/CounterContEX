# Copyright (c) Prior Labs GmbH 2026.

"""Bounded diverse beam search with joint DPP counterfactual selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
from experiments.zeroshot_cf.greedy import project_candidate_values
from experiments.zeroshot_cf.mixed_distance import (
    action_unit_change_count,
    grouped_gower_distance,
)

if TYPE_CHECKING:
    from experiments.zeroshot_cf.data import OneHotActionGroup
    from experiments.zeroshot_cf.grouped_categorical import (
        ConditionedCategoryDistribution,
    )


@dataclass(frozen=True)
class DiverseBeamSearchConfig:
    """Budgets and objectives for the separate diverse generator."""

    n_counterfactuals: int = 3
    beam_width: int = 8
    candidate_pool_size: int = 16
    max_extra_actions: int = 2
    max_gower_ratio: float = 1.5
    max_gower_increase: float = 0.02
    states_per_action_set: int = 2
    categorical_proposal_count: int | None = None
    dpp_action_weight: float = 0.75
    dpp_gower_quality_weight: float = 4.0
    dpp_sparsity_quality_weight: float = 1.0

    def __post_init__(self) -> None:  # noqa: C901
        if self.n_counterfactuals < 1:
            raise ValueError("n_counterfactuals must be at least 1")
        if self.beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        if self.candidate_pool_size < self.n_counterfactuals:
            raise ValueError("candidate_pool_size must be at least n_counterfactuals")
        if self.max_extra_actions < 0:
            raise ValueError("max_extra_actions must be non-negative")
        if not np.isfinite(self.max_gower_ratio) or self.max_gower_ratio < 1.0:
            raise ValueError("max_gower_ratio must be at least 1")
        if not np.isfinite(self.max_gower_increase) or self.max_gower_increase < 0:
            raise ValueError("max_gower_increase must be non-negative")
        if self.states_per_action_set < 1:
            raise ValueError("states_per_action_set must be at least 1")
        if (
            self.categorical_proposal_count is not None
            and self.categorical_proposal_count < 1
        ):
            raise ValueError("categorical_proposal_count must be positive")
        if not 0.0 <= self.dpp_action_weight <= 1.0:
            raise ValueError("dpp_action_weight must be between zero and one")
        if self.dpp_gower_quality_weight < 0:
            raise ValueError("dpp_gower_quality_weight must be non-negative")
        if self.dpp_sparsity_quality_weight < 0:
            raise ValueError("dpp_sparsity_quality_weight must be non-negative")


@dataclass(frozen=True)
class DiverseCounterfactualResult:
    """A variable-length set containing valid, unique counterfactuals only."""

    counterfactuals: np.ndarray
    target_probabilities: np.ndarray
    histories: tuple[tuple[dict[str, Any], ...], ...]
    depths: np.ndarray
    requested_count: int
    valid_candidate_count: int
    candidate_pool_count: int
    search_depth: int
    dpp_logdet: float | None

    @property
    def available_count(self) -> int:
        """Return the number of valid counterfactuals found."""
        return len(self.counterfactuals)

@dataclass(frozen=True)
class _BeamState:
    row: np.ndarray
    probability: float
    depth: int
    used_numerical: frozenset[int]
    used_groups: frozenset[str]
    history: tuple[dict[str, Any], ...]


def action_unit_signature(
    row: np.ndarray,
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
) -> frozenset[tuple[str, int | str]]:
    """Return changed original feature units, counting each group once."""
    candidate = np.asarray(row)
    reference = np.asarray(factual)
    signature: set[tuple[str, int | str]] = {
        ("numerical", int(column))
        for column in numerical_columns
        if not np.isclose(candidate[int(column)], reference[int(column)])
    }
    for group in categorical_groups:
        columns = list(group.columns)
        if np.argmax(candidate[columns]) != np.argmax(reference[columns]):
            signature.add(("categorical", group.name))
    return frozenset(signature)


def action_set_jaccard_distance(
    left: frozenset[tuple[str, int | str]],
    right: frozenset[tuple[str, int | str]],
) -> float:
    """Return Jaccard distance between two sets of changed feature units."""
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def _classifier_outputs(
    disc: Any,
    rows: np.ndarray,
    y_target: int,
) -> tuple[np.ndarray, np.ndarray]:
    probability_matrix = np.asarray(disc.predict_proba(np.atleast_2d(rows)))
    classes = np.asarray(
        getattr(disc, "classes_", np.arange(probability_matrix.shape[1]))
    )
    target_positions = np.flatnonzero(classes == y_target)
    if len(target_positions) != 1:
        raise ValueError(f"target class {y_target} is absent from classifier classes")
    predictions = classes[np.argmax(probability_matrix, axis=1)]
    return probability_matrix[:, int(target_positions[0])], predictions


def _available_numerical(
    state: _BeamState,
    numerical_columns: Sequence[int],
    *,
    allow_revisits: bool,
) -> list[int]:
    return [
        int(column)
        for column in numerical_columns
        if allow_revisits or int(column) not in state.used_numerical
    ]


def _numerical_trials_for_beam(
    sampler: Any,
    beam: Sequence[_BeamState],
    numerical_columns: Sequence[int],
    y_target: int,
    candidate_quantiles: np.ndarray | None,
    candidate_confidences: np.ndarray | None,
    feature_domains: Any,
    temperature: float,
    *,
    allow_revisits: bool,
) -> tuple[list[np.ndarray], list[_BeamState], list[dict[str, Any]]]:
    """Expand numerical branches in one TabICL call when supported."""
    pair_states: list[_BeamState] = []
    pair_columns: list[int] = []
    for state in beam:
        columns = _available_numerical(
            state,
            numerical_columns,
            allow_revisits=allow_revisits,
        )
        pair_states.extend([state] * len(columns))
        pair_columns.extend(columns)
    if not pair_columns:
        return [], [], []

    queries = np.stack([state.row for state in pair_states])
    columns_array = np.asarray(pair_columns, dtype=int)
    n_confidences = 1 if candidate_confidences is None else len(candidate_confidences)
    n_quantiles = 1 if candidate_quantiles is None else len(candidate_quantiles)

    if candidate_quantiles is None:
        if hasattr(sampler, "sample_candidates_batch"):
            raw_values = sampler.sample_candidates_batch(
                queries,
                pair_columns,
                sample_temperature=temperature,
                fixed_target=y_target,
            )
        else:
            raw_values = [
                sampler.sample_candidates(
                    state.row.reshape(1, -1),
                    [column],
                    sample_temperature=temperature,
                    fixed_target=y_target,
                )[0]
                for state, column in zip(pair_states, pair_columns, strict=True)
            ]
        values = np.asarray(raw_values, dtype=np.float64).reshape(
            len(pair_columns), 1, 1
        )
    elif hasattr(sampler, "sample_candidate_grid_batch"):
        values = np.asarray(
            sampler.sample_candidate_grid_batch(
                queries,
                pair_columns,
                quantiles=candidate_quantiles,
                fixed_target=y_target,
                confidences=candidate_confidences,
            ),
            dtype=np.float64,
        )
    else:
        grids = [
            np.asarray(
                sampler.sample_candidate_grid(
                    state.row.reshape(1, -1),
                    [column],
                    quantiles=candidate_quantiles,
                    fixed_target=y_target,
                    confidences=candidate_confidences,
                ),
                dtype=np.float64,
            ).reshape(n_confidences, n_quantiles)
            for state, column in zip(pair_states, pair_columns, strict=True)
        ]
        values = np.stack(grids)

    expected = (len(pair_columns), n_confidences, n_quantiles)
    if values.shape != expected:
        raise ValueError(
            "batched TabICL proposals returned an unexpected shape; "
            f"expected {expected}, got {values.shape}"
        )

    repeated_columns = np.repeat(columns_array, n_confidences * n_quantiles)
    flat_values = project_candidate_values(
        repeated_columns.tolist(),
        values.reshape(-1),
        feature_domains,
    )
    rows: list[np.ndarray] = []
    parents: list[_BeamState] = []
    metadata: list[dict[str, Any]] = []
    position = 0
    for state, column in zip(pair_states, pair_columns, strict=True):
        for confidence_index in range(n_confidences):
            for quantile_index in range(n_quantiles):
                row = state.row.copy()
                row[column] = flat_values[position]
                rows.append(row)
                parents.append(state)
                metadata.append(
                    {
                        "action_type": "numerical",
                        "feature": int(column),
                        "quantile": (
                            None
                            if candidate_quantiles is None
                            else float(candidate_quantiles[quantile_index])
                        ),
                        "confidence": (
                            None
                            if candidate_confidences is None
                            else float(candidate_confidences[confidence_index])
                        ),
                    }
                )
                position += 1
    return rows, parents, metadata


def _categorical_trials_for_beam(
    beam: Sequence[_BeamState],
    categorical_groups: Sequence[OneHotActionGroup],
    candidate_confidences: np.ndarray | None,
    category_distribution: ConditionedCategoryDistribution | None,
    proposal_count: int | None,
    *,
    allow_revisits: bool,
) -> tuple[list[np.ndarray], list[_BeamState], list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    parents: list[_BeamState] = []
    metadata: list[dict[str, Any]] = []
    for state in beam:
        groups = [
            group
            for group in categorical_groups
            if allow_revisits or group.name not in state.used_groups
        ]
        for group in groups:
            columns = list(group.columns)
            values = state.row[columns]
            if not np.isclose(values.sum(), 1.0):
                raise ValueError(f"one-hot group {group.name!r} is invalid")
            previous_category = int(np.argmax(values))
            scores: dict[int, tuple[float, float | None]] = {}
            if category_distribution is None:
                scores = dict.fromkeys(range(len(columns)), (1.0, None))
            else:
                anchors = (
                    [None]
                    if candidate_confidences is None
                    else candidate_confidences.tolist()
                )
                for anchor in anchors:
                    categories, probabilities = category_distribution(
                        state.row,
                        group,
                        anchor,
                    )
                    for category, probability in zip(
                        np.asarray(categories, dtype=int),
                        np.asarray(probabilities, dtype=np.float64),
                        strict=True,
                    ):
                        previous = scores.get(int(category))
                        if previous is None or probability > previous[0]:
                            scores[int(category)] = (
                                float(probability),
                                None if anchor is None else float(anchor),
                            )
            alternatives = [
                category
                for category in range(len(columns))
                if category != previous_category
            ]
            alternatives.sort(
                key=lambda category: scores.get(category, (0.0, None))[0],
                reverse=True,
            )
            if proposal_count is not None:
                alternatives = alternatives[:proposal_count]
            for proposal_rank, category in enumerate(alternatives, start=1):
                probability, confidence = scores.get(category, (0.0, None))
                trial = state.row.copy()
                trial[columns] = 0.0
                trial[group.columns[category]] = 1.0
                rows.append(trial)
                parents.append(state)
                metadata.append(
                    {
                        "action_type": "categorical",
                        "group": group.name,
                        "from_category": previous_category,
                        "to_category": category,
                        "tabicl_conditional_probability": probability,
                        "tabicl_confidence_anchor": confidence,
                        "tabicl_proposal_rank": proposal_rank,
                        "in_tabicl_support": category in scores,
                    }
                )
    return rows, parents, metadata


def _state_quality_key(
    state: _BeamState,
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
) -> tuple[float, int, float, bytes]:
    gower = float(
        grouped_gower_distance(
            state.row, factual, numerical_columns, categorical_groups
        )[0]
    )
    sparsity = int(
        action_unit_change_count(
            state.row, factual, numerical_columns, categorical_groups
        )[0]
    )
    return gower, sparsity, -state.probability, state.row.tobytes()


def _prune_beam(
    states: Sequence[_BeamState],
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    config: DiverseBeamSearchConfig,
) -> list[_BeamState]:
    """Keep strong representatives from distinct changed-feature niches."""
    unique: dict[bytes, _BeamState] = {}
    for state in states:
        key = state.row.tobytes()
        previous = unique.get(key)
        if previous is None or state.probability > previous.probability:
            unique[key] = state

    niches: dict[frozenset[tuple[str, int | str]], list[_BeamState]] = {}
    for state in unique.values():
        signature = action_unit_signature(
            state.row, factual, numerical_columns, categorical_groups
        )
        niches.setdefault(signature, []).append(state)
    for niche in niches.values():
        niche.sort(
            key=lambda state: (
                -state.probability,
                *_state_quality_key(
                    state, factual, numerical_columns, categorical_groups
                ),
            )
        )

    selected: list[_BeamState] = []
    for rank in range(config.states_per_action_set):
        candidates = [niche[rank] for niche in niches.values() if len(niche) > rank]
        candidates.sort(
            key=lambda state: (
                -state.probability,
                *_state_quality_key(
                    state, factual, numerical_columns, categorical_groups
                ),
            )
        )
        selected.extend(candidates[: config.beam_width - len(selected)])
        if len(selected) == config.beam_width:
            break
    return selected


def _quality_eligible_candidates(
    candidates: Sequence[_BeamState],
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    config: DiverseBeamSearchConfig,
) -> list[_BeamState]:
    if not candidates:
        return []
    pool = list({state.row.tobytes(): state for state in candidates}.values())
    anchor = min(
        pool,
        key=lambda state: _state_quality_key(
            state, factual, numerical_columns, categorical_groups
        ),
    )
    anchor_gower, anchor_sparsity, _, _ = _state_quality_key(
        anchor, factual, numerical_columns, categorical_groups
    )
    max_gower = config.max_gower_ratio * anchor_gower + config.max_gower_increase
    max_sparsity = anchor_sparsity + config.max_extra_actions
    eligible: list[_BeamState] = []
    for state in pool:
        gower, sparsity, _, _ = _state_quality_key(
            state, factual, numerical_columns, categorical_groups
        )
        if gower <= max_gower + 1e-12 and sparsity <= max_sparsity:
            eligible.append(state)
    return eligible


def _curate_candidate_pool(
    candidates: Sequence[_BeamState],
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    config: DiverseBeamSearchConfig,
) -> list[_BeamState]:
    """Bound the DPP pool while preserving distinct changed-feature sets."""
    eligible = _quality_eligible_candidates(
        candidates, factual, numerical_columns, categorical_groups, config
    )
    niches: dict[frozenset[tuple[str, int | str]], list[_BeamState]] = {}
    for state in eligible:
        signature = action_unit_signature(
            state.row, factual, numerical_columns, categorical_groups
        )
        niches.setdefault(signature, []).append(state)
    for niche in niches.values():
        niche.sort(
            key=lambda state: _state_quality_key(
                state, factual, numerical_columns, categorical_groups
            )
        )

    selected: list[_BeamState] = []
    rank = 0
    while len(selected) < config.candidate_pool_size:
        candidates_at_rank = [
            niche[rank] for niche in niches.values() if len(niche) > rank
        ]
        if not candidates_at_rank:
            break
        candidates_at_rank.sort(
            key=lambda state: _state_quality_key(
                state, factual, numerical_columns, categorical_groups
            )
        )
        selected.extend(
            candidates_at_rank[: config.candidate_pool_size - len(selected)]
        )
        rank += 1
    return selected


def _dpp_embedding(
    rows: np.ndarray,
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    action_weight: float,
) -> np.ndarray:
    """Build a unit-balanced action/value embedding for an RBF DPP kernel."""
    matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    reference = np.asarray(factual, dtype=np.float64)
    numerical = tuple(int(column) for column in numerical_columns)
    groups = tuple(categorical_groups)
    n_units = len(numerical) + len(groups)
    if n_units == 0:
        return np.zeros((len(matrix), 1), dtype=np.float64)

    action_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    if numerical:
        numeric_values = np.clip(matrix[:, numerical], 0.0, 1.0)
        action_parts.append(
            (~np.isclose(numeric_values, reference[list(numerical)])).astype(float)
        )
        value_parts.append(numeric_values)
    for group in groups:
        columns = list(group.columns)
        categories = np.argmax(matrix[:, columns], axis=1)
        factual_category = int(np.argmax(reference[columns]))
        action_parts.append((categories != factual_category).reshape(-1, 1))
        # A one-hot category switch has unit value distance after this scaling.
        value_parts.append(matrix[:, columns] / np.sqrt(2.0))

    action = np.concatenate(action_parts, axis=1) / np.sqrt(n_units)
    values = np.concatenate(value_parts, axis=1) / np.sqrt(n_units)
    return np.concatenate(
        [
            np.sqrt(action_weight) * action,
            np.sqrt(1.0 - action_weight) * values,
        ],
        axis=1,
    )


def select_dpp_subset(
    rows: np.ndarray,
    probabilities: np.ndarray,
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    config: DiverseBeamSearchConfig,
) -> tuple[np.ndarray, float | None]:
    """Select the exact fixed-size DPP MAP subset from a small valid pool."""
    matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    target_probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(matrix) != len(target_probabilities):
        raise ValueError("rows and probabilities must have the same length")
    k = min(config.n_counterfactuals, len(matrix))
    if k == 0:
        return np.empty(0, dtype=int), None

    gower = grouped_gower_distance(
        matrix, factual, numerical_columns, categorical_groups
    )
    sparsity = action_unit_change_count(
        matrix, factual, numerical_columns, categorical_groups
    )
    n_units = max(1, len(numerical_columns) + len(categorical_groups))
    quality = np.exp(
        -config.dpp_gower_quality_weight * gower
        - config.dpp_sparsity_quality_weight * sparsity / n_units
    )
    embedding = _dpp_embedding(
        matrix,
        factual,
        numerical_columns,
        categorical_groups,
        config.dpp_action_weight,
    )
    differences = embedding[:, None, :] - embedding[None, :, :]
    squared_distances = np.einsum("ijk,ijk->ij", differences, differences)
    positive = squared_distances[squared_distances > 1e-12]
    bandwidth_squared = float(np.median(positive)) if len(positive) else 1.0
    similarity = np.exp(-squared_distances / (2.0 * bandwidth_squared))
    kernel = quality[:, None] * similarity * quality[None, :]

    best_combination: tuple[int, ...] | None = None
    best_logdet = -np.inf
    best_tie: tuple[float, int, float, tuple[int, ...]] | None = None
    for combination in combinations(range(len(matrix)), k):
        indices = np.asarray(combination, dtype=int)
        subkernel = kernel[np.ix_(indices, indices)]
        sign, logdet = np.linalg.slogdet(
            subkernel + np.eye(k, dtype=np.float64) * 1e-12
        )
        score = float(logdet) if sign > 0 else -np.inf
        tie = (
            float(gower[indices].sum()),
            int(sparsity[indices].sum()),
            -float(target_probabilities[indices].sum()),
            combination,
        )
        if score > best_logdet + 1e-12 or (
            np.isclose(score, best_logdet, atol=1e-12, rtol=0.0)
            and (best_tie is None or tie < best_tie)
        ):
            best_combination = combination
            best_logdet = score
            best_tie = tie

    if best_combination is None:
        return np.empty(0, dtype=int), None
    ordered = sorted(
        best_combination,
        key=lambda index: (
            float(gower[index]),
            int(sparsity[index]),
            -float(target_probabilities[index]),
            matrix[index].tobytes(),
        ),
    )
    return np.asarray(ordered, dtype=int), best_logdet


def generate_diverse_counterfactuals(  # noqa: C901, PLR0912, PLR0913
    sampler: Any,
    disc: Any,
    x: np.ndarray,
    y_target: int,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    *,
    config: DiverseBeamSearchConfig,
    candidate_quantiles: Sequence[float] | None = None,
    candidate_confidences: Sequence[float] | None = None,
    feature_domains: Any = None,
    max_validity_steps: int | None = None,
    allow_revisits: bool = True,
    tau: float = 0.5,
    temperature: float = 1e-9,
    category_distribution: ConditionedCategoryDistribution | None = None,
) -> DiverseCounterfactualResult:
    """Generate a valid beam pool and jointly select a diverse subset.

    This method is independent of the existing greedy single-CFE method.
    Invalid candidates stay in the beam only after strict target-probability
    improvement. Valid candidates enter a bounded, quality-controlled pool.
    Exact fixed-size DPP MAP selection returns no invalid or duplicate padding.
    """
    factual = np.asarray(x, dtype=np.float64)
    numerical = tuple(int(column) for column in numerical_columns)
    groups = tuple(categorical_groups)
    n_action_units = len(numerical) + len(groups)
    if max_validity_steps is None:
        max_validity_steps = n_action_units
    if max_validity_steps < 1:
        raise ValueError("max_validity_steps must be at least 1")
    quantiles = (
        None
        if candidate_quantiles is None
        else np.asarray(candidate_quantiles, dtype=np.float64)
    )
    confidences = (
        None
        if candidate_confidences is None
        else np.asarray(candidate_confidences, dtype=np.float64)
    )
    if confidences is not None and quantiles is None:
        raise ValueError("candidate_confidences require candidate_quantiles")

    factual_probabilities, factual_predictions = _classifier_outputs(
        disc, factual, y_target
    )
    initial = _BeamState(
        row=factual.copy(),
        probability=float(factual_probabilities[0]),
        depth=0,
        used_numerical=frozenset(),
        used_groups=frozenset(),
        history=(),
    )
    initial_is_valid = bool(
        factual_predictions[0] == y_target and factual_probabilities[0] >= tau
    )
    valid_candidates: dict[bytes, _BeamState] = {}
    if initial_is_valid:
        valid_candidates[initial.row.tobytes()] = initial
    beam = [] if initial_is_valid else [initial]
    visited = {initial.row.tobytes()}
    search_depth = 0

    for depth in range(1, max_validity_steps + 1):
        if not beam:
            break
        search_depth = depth
        numerical_rows, numerical_parents, numerical_metadata = (
            _numerical_trials_for_beam(
                sampler,
                beam,
                numerical,
                y_target,
                quantiles,
                confidences,
                feature_domains,
                temperature,
                allow_revisits=allow_revisits,
            )
        )
        categorical_rows, categorical_parents, categorical_metadata = (
            _categorical_trials_for_beam(
                beam,
                groups,
                confidences,
                category_distribution,
                config.categorical_proposal_count,
                allow_revisits=allow_revisits,
            )
        )
        trial_rows = numerical_rows + categorical_rows
        trial_parents = numerical_parents + categorical_parents
        trial_metadata = numerical_metadata + categorical_metadata

        # Projection and repeated quantiles can create identical rows.
        unique_trials: dict[bytes, tuple[np.ndarray, _BeamState, dict[str, Any]]] = {}
        for row, parent, metadata in zip(
            trial_rows, trial_parents, trial_metadata, strict=True
        ):
            key = np.ascontiguousarray(row).tobytes()
            if key in visited or key in unique_trials:
                continue
            unique_trials[key] = (row, parent, metadata)
        if not unique_trials:
            break
        visited.update(unique_trials)

        trials = np.stack([item[0] for item in unique_trials.values()])
        parents = [item[1] for item in unique_trials.values()]
        metadata_items = [item[2] for item in unique_trials.values()]
        probabilities, predictions = _classifier_outputs(disc, trials, y_target)
        next_states: list[_BeamState] = []
        for row, probability, prediction, parent, raw_metadata in zip(
            trials,
            probabilities,
            predictions,
            parents,
            metadata_items,
            strict=True,
        ):
            immediate_valid = bool(prediction == y_target and probability >= tau)
            metadata = dict(raw_metadata)
            metadata.update(
                {
                    "selection_phase": "diverse_beam_search",
                    "target_probability": float(probability),
                    "immediate_valid": immediate_valid,
                    "search_depth": depth,
                }
            )
            used_numerical = set(parent.used_numerical)
            used_groups = set(parent.used_groups)
            if metadata["action_type"] == "numerical":
                used_numerical.add(int(metadata["feature"]))
            else:
                used_groups.add(str(metadata["group"]))
            state = _BeamState(
                row=np.asarray(row).copy(),
                probability=float(probability),
                depth=depth,
                used_numerical=frozenset(used_numerical),
                used_groups=frozenset(used_groups),
                history=(*parent.history, metadata),
            )
            if immediate_valid:
                valid_candidates[state.row.tobytes()] = state
            if probability > parent.probability + 1e-12:
                next_states.append(state)

        pool = _curate_candidate_pool(
            list(valid_candidates.values()), factual, numerical, groups, config
        )
        if len(pool) >= config.candidate_pool_size:
            break
        beam = _prune_beam(next_states, factual, numerical, groups, config)

    pool = _curate_candidate_pool(
        list(valid_candidates.values()), factual, numerical, groups, config
    )
    if pool:
        pool_rows = np.stack([state.row for state in pool])
        pool_probabilities = np.asarray(
            [state.probability for state in pool], dtype=np.float64
        )
        selected_indices, dpp_logdet = select_dpp_subset(
            pool_rows,
            pool_probabilities,
            factual,
            numerical,
            groups,
            config,
        )
        selected = [pool[int(index)] for index in selected_indices]
        rows = np.stack([state.row for state in selected])
        probabilities = np.asarray(
            [state.probability for state in selected], dtype=np.float64
        )
        depths = np.asarray([state.depth for state in selected], dtype=int)
    else:
        selected = []
        rows = np.empty((0, factual.shape[0]), dtype=np.float64)
        probabilities = np.empty(0, dtype=np.float64)
        depths = np.empty(0, dtype=int)
        dpp_logdet = None
    return DiverseCounterfactualResult(
        counterfactuals=rows,
        target_probabilities=probabilities,
        histories=tuple(state.history for state in selected),
        depths=depths,
        requested_count=config.n_counterfactuals,
        valid_candidate_count=len(valid_candidates),
        candidate_pool_count=len(pool),
        search_depth=search_depth,
        dpp_logdet=dpp_logdet,
    )
