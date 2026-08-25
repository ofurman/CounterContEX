# Copyright (c) Prior Labs GmbH 2026.

"""Validity-constrained diverse search for mixed-data counterfactuals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
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
class DiverseSearchConfig:
    """Search and quality budgets for a set of counterfactuals."""

    n_counterfactuals: int = 3
    beam_width: int = 8
    archive_size: int = 64
    max_extra_actions: int = 2
    max_gower_ratio: float = 1.5
    max_gower_increase: float = 0.02
    states_per_action_set: int = 2
    categorical_proposal_count: int | None = None

    def __post_init__(self) -> None:
        if self.n_counterfactuals < 1:
            raise ValueError("n_counterfactuals must be at least 1")
        if self.beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        if self.archive_size < self.n_counterfactuals:
            raise ValueError("archive_size must be at least n_counterfactuals")
        if self.max_extra_actions < 0:
            raise ValueError("max_extra_actions must be non-negative")
        if not np.isfinite(self.max_gower_ratio) or self.max_gower_ratio < 1.0:
            raise ValueError("max_gower_ratio must be at least 1")
        if not np.isfinite(self.max_gower_increase) or self.max_gower_increase < 0.0:
            raise ValueError("max_gower_increase must be non-negative")
        if self.states_per_action_set < 1:
            raise ValueError("states_per_action_set must be at least 1")
        if (
            self.categorical_proposal_count is not None
            and self.categorical_proposal_count < 1
        ):
            raise ValueError("categorical_proposal_count must be positive")


@dataclass(frozen=True)
class DiverseCounterfactualResult:
    """A variable-length set containing valid, unique counterfactuals only."""

    counterfactuals: np.ndarray
    target_probabilities: np.ndarray
    histories: tuple[tuple[dict[str, Any], ...], ...]
    requested_count: int
    archive_count: int
    search_depth: int

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


def _numerical_trials(
    sampler: Any,
    state: _BeamState,
    columns: Sequence[int],
    y_target: int,
    candidate_quantiles: np.ndarray | None,
    candidate_confidences: np.ndarray | None,
    feature_domains: Any,
    temperature: float,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    if not columns:
        return [], []
    if candidate_quantiles is None:
        values = np.asarray(
            sampler.sample_candidates(
                state.row.reshape(1, -1),
                columns,
                sample_temperature=temperature,
                fixed_target=y_target,
            ),
            dtype=np.float64,
        )
        expected = (len(columns),)
        if values.shape != expected:
            raise ValueError(
                "sample_candidates returned an unexpected shape; "
                f"expected {expected}, got {values.shape}"
            )
        expanded_columns = np.asarray(columns, dtype=int)
        expanded_quantiles: np.ndarray | None = None
        expanded_confidences: np.ndarray | None = None
    else:
        values = np.asarray(
            sampler.sample_candidate_grid(
                state.row.reshape(1, -1),
                columns,
                quantiles=candidate_quantiles,
                fixed_target=y_target,
                confidences=candidate_confidences,
            ),
            dtype=np.float64,
        )
        n_confidences = (
            1 if candidate_confidences is None else len(candidate_confidences)
        )
        expected = (
            (len(columns), len(candidate_quantiles))
            if candidate_confidences is None
            else (len(columns), n_confidences, len(candidate_quantiles))
        )
        if values.shape != expected:
            raise ValueError(
                "sample_candidate_grid returned an unexpected shape; "
                f"expected {expected}, got {values.shape}"
            )
        expanded_columns = np.repeat(
            np.asarray(columns, dtype=int),
            n_confidences * len(candidate_quantiles),
        )
        expanded_quantiles = np.tile(
            candidate_quantiles,
            len(columns) * n_confidences,
        )
        expanded_confidences = (
            None
            if candidate_confidences is None
            else np.tile(
                np.repeat(candidate_confidences, len(candidate_quantiles)),
                len(columns),
            )
        )

    flat_values = project_candidate_values(
        expanded_columns.tolist(),
        values.reshape(-1),
        feature_domains,
    )
    rows = np.repeat(state.row.reshape(1, -1), len(flat_values), axis=0)
    rows[np.arange(len(flat_values)), expanded_columns] = flat_values
    metadata = [
        {
            "action_type": "numerical",
            "feature": int(column),
            "quantile": (
                None
                if expanded_quantiles is None
                else float(expanded_quantiles[position])
            ),
            "confidence": (
                None
                if expanded_confidences is None
                else float(expanded_confidences[position])
            ),
        }
        for position, column in enumerate(expanded_columns)
    ]
    return list(rows), metadata


def _categorical_trials(
    state: _BeamState,
    groups: Sequence[OneHotActionGroup],
    candidate_confidences: np.ndarray | None,
    category_distribution: ConditionedCategoryDistribution | None,
    proposal_count: int | None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
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
    return rows, metadata


def _expand_state(  # noqa: PLR0913
    sampler: Any,
    state: _BeamState,
    y_target: int,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    candidate_quantiles: np.ndarray | None,
    candidate_confidences: np.ndarray | None,
    feature_domains: Any,
    temperature: float,
    category_distribution: ConditionedCategoryDistribution | None,
    config: DiverseSearchConfig,
    *,
    allow_revisits: bool,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    numerical = [
        int(column)
        for column in numerical_columns
        if allow_revisits or int(column) not in state.used_numerical
    ]
    groups = [
        group
        for group in categorical_groups
        if allow_revisits or group.name not in state.used_groups
    ]
    numerical_rows, numerical_metadata = _numerical_trials(
        sampler,
        state,
        numerical,
        y_target,
        candidate_quantiles,
        candidate_confidences,
        feature_domains,
        temperature,
    )
    categorical_rows, categorical_metadata = _categorical_trials(
        state,
        groups,
        candidate_confidences,
        category_distribution,
        config.categorical_proposal_count,
    )
    return (
        numerical_rows + categorical_rows,
        numerical_metadata + categorical_metadata,
    )


def _state_sort_key(
    state: _BeamState,
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
) -> tuple[float, float, int, bytes]:
    gower = float(
        grouped_gower_distance(
            state.row,
            factual,
            numerical_columns,
            categorical_groups,
        )[0]
    )
    sparsity = int(
        action_unit_change_count(
            state.row,
            factual,
            numerical_columns,
            categorical_groups,
        )[0]
    )
    return -state.probability, gower, sparsity, state.row.tobytes()


def _prune_beam(
    states: Sequence[_BeamState],
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    config: DiverseSearchConfig,
) -> list[_BeamState]:
    """Keep strong representatives from distinct action-set niches."""
    unique_rows: dict[bytes, _BeamState] = {}
    for state in states:
        key = state.row.tobytes()
        previous = unique_rows.get(key)
        if previous is None or state.probability > previous.probability:
            unique_rows[key] = state

    niches: dict[frozenset[tuple[str, int | str]], list[_BeamState]] = {}
    for state in unique_rows.values():
        signature = action_unit_signature(
            state.row,
            factual,
            numerical_columns,
            categorical_groups,
        )
        niches.setdefault(signature, []).append(state)
    for niche in niches.values():
        niche.sort(
            key=lambda state: _state_sort_key(
                state,
                factual,
                numerical_columns,
                categorical_groups,
            )
        )

    selected: list[_BeamState] = []
    for rank in range(config.states_per_action_set):
        candidates = [niche[rank] for niche in niches.values() if len(niche) > rank]
        candidates.sort(
            key=lambda state: _state_sort_key(
                state,
                factual,
                numerical_columns,
                categorical_groups,
            )
        )
        for state in candidates:
            selected.append(state)
            if len(selected) == config.beam_width:
                return selected
    return selected


def _select_diverse_set(
    candidates: Sequence[_BeamState],
    primary: np.ndarray,
    *,
    primary_is_valid: bool,
    factual: np.ndarray,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    config: DiverseSearchConfig,
) -> list[_BeamState]:
    if not candidates:
        return []
    unique = {candidate.row.tobytes(): candidate for candidate in candidates}
    pool = list(unique.values())

    primary_key = primary.tobytes()
    if primary_is_valid and primary_key in unique:
        anchor = unique[primary_key]
    else:
        pool.sort(
            key=lambda state: (
                _state_sort_key(
                    state,
                    factual,
                    numerical_columns,
                    categorical_groups,
                )[1:3],
                -state.probability,
                state.row.tobytes(),
            )
        )
        anchor = pool[0]

    anchor_gower = float(
        grouped_gower_distance(
            anchor.row,
            factual,
            numerical_columns,
            categorical_groups,
        )[0]
    )
    anchor_sparsity = int(
        action_unit_change_count(
            anchor.row,
            factual,
            numerical_columns,
            categorical_groups,
        )[0]
    )
    max_gower = config.max_gower_ratio * anchor_gower + config.max_gower_increase
    max_sparsity = anchor_sparsity + config.max_extra_actions
    eligible = []
    for candidate in pool:
        gower = float(
            grouped_gower_distance(
                candidate.row,
                factual,
                numerical_columns,
                categorical_groups,
            )[0]
        )
        sparsity = int(
            action_unit_change_count(
                candidate.row,
                factual,
                numerical_columns,
                categorical_groups,
            )[0]
        )
        if gower <= max_gower + 1e-12 and sparsity <= max_sparsity:
            eligible.append(candidate)

    selected = [anchor]
    remaining = [
        item for item in eligible if item.row.tobytes() != anchor.row.tobytes()
    ]
    signatures = {
        item.row.tobytes(): action_unit_signature(
            item.row,
            factual,
            numerical_columns,
            categorical_groups,
        )
        for item in eligible
    }
    while remaining and len(selected) < config.n_counterfactuals:
        best: _BeamState | None = None
        best_key: tuple[float, float, float, int, float] | None = None
        for candidate in remaining:
            signature = signatures[candidate.row.tobytes()]
            action_diversity = min(
                action_set_jaccard_distance(
                    signature,
                    signatures[chosen.row.tobytes()],
                )
                for chosen in selected
            )
            value_diversity = min(
                float(
                    grouped_gower_distance(
                        candidate.row,
                        chosen.row,
                        numerical_columns,
                        categorical_groups,
                    )[0]
                )
                for chosen in selected
            )
            factual_gower = float(
                grouped_gower_distance(
                    candidate.row,
                    factual,
                    numerical_columns,
                    categorical_groups,
                )[0]
            )
            sparsity = int(
                action_unit_change_count(
                    candidate.row,
                    factual,
                    numerical_columns,
                    categorical_groups,
                )[0]
            )
            key = (
                action_diversity,
                value_diversity,
                -factual_gower,
                -sparsity,
                candidate.probability,
            )
            if best_key is None or key > best_key:
                best = candidate
                best_key = key
        if best is None:
            break
        selected.append(best)
        remaining = [candidate for candidate in remaining if candidate is not best]
    return selected


def generate_diverse_counterfactuals(  # noqa: C901, PLR0912, PLR0913
    sampler: Any,
    disc: Any,
    x: np.ndarray,
    y_target: int,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    *,
    primary_counterfactual: np.ndarray,
    primary_info: dict[str, Any],
    config: DiverseSearchConfig,
    candidate_quantiles: Sequence[float] | None = None,
    candidate_confidences: Sequence[float] | None = None,
    feature_domains: Any = None,
    max_validity_steps: int | None = None,
    allow_revisits: bool = True,
    tau: float = 0.5,
    temperature: float = 1e-9,
    category_distribution: ConditionedCategoryDistribution | None = None,
) -> DiverseCounterfactualResult:
    """Search multiple paths and return a quality-constrained diverse set.

    The supplied primary counterfactual is retained as the first result when
    it is valid. Invalid rows are never returned. If the search cannot find the
    requested number of valid, unique rows within the budgets, it returns the
    smaller set without padding.
    """
    factual = np.asarray(x, dtype=np.float64)
    primary = np.asarray(primary_counterfactual, dtype=np.float64)
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
        disc,
        factual,
        y_target,
    )
    initial = _BeamState(
        row=factual.copy(),
        probability=float(factual_probabilities[0]),
        depth=0,
        used_numerical=frozenset(),
        used_groups=frozenset(),
        history=(),
    )
    initial_is_valid = (
        factual_predictions[0] == y_target and factual_probabilities[0] >= tau
    )
    archive: dict[bytes, _BeamState] = {}
    if initial_is_valid:
        archive[initial.row.tobytes()] = initial

    primary_probabilities, primary_predictions = _classifier_outputs(
        disc,
        primary,
        y_target,
    )
    primary_is_valid = bool(
        primary_predictions[0] == y_target and primary_probabilities[0] >= tau
    )
    if primary_is_valid:
        archive[primary.tobytes()] = _BeamState(
            row=primary.copy(),
            probability=float(primary_probabilities[0]),
            depth=int(primary_info.get("validity_steps", 0)),
            used_numerical=frozenset(),
            used_groups=frozenset(),
            history=tuple(primary_info.get("history", ())),
        )

    beam = [] if initial_is_valid else [initial]
    search_depth = 0
    for depth in range(1, max_validity_steps + 1):
        if not beam:
            break
        search_depth = depth
        trial_rows: list[np.ndarray] = []
        trial_parents: list[_BeamState] = []
        trial_metadata: list[dict[str, Any]] = []
        for state in beam:
            rows, metadata = _expand_state(
                sampler,
                state,
                y_target,
                numerical,
                groups,
                quantiles,
                confidences,
                feature_domains,
                temperature,
                category_distribution,
                config,
                allow_revisits=allow_revisits,
            )
            trial_rows.extend(rows)
            trial_parents.extend([state] * len(rows))
            trial_metadata.extend(metadata)
        if not trial_rows:
            break

        trials = np.stack(trial_rows)
        probabilities, predictions = _classifier_outputs(disc, trials, y_target)
        next_states: list[_BeamState] = []
        for row, probability, prediction, parent, raw_metadata in zip(
            trials,
            probabilities,
            predictions,
            trial_parents,
            trial_metadata,
            strict=True,
        ):
            if probability <= parent.probability + 1e-12:
                continue
            metadata = dict(raw_metadata)
            metadata.update(
                {
                    "selection_phase": "diverse_validity_search",
                    "target_probability": float(probability),
                    "immediate_valid": bool(
                        prediction == y_target and probability >= tau
                    ),
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
            if prediction == y_target and probability >= tau:
                archive[state.row.tobytes()] = state
            else:
                next_states.append(state)

        beam = _prune_beam(
            next_states,
            factual,
            numerical,
            groups,
            config,
        )
        if len(archive) > config.archive_size:
            retained = _select_diverse_set(
                list(archive.values()),
                primary,
                primary_is_valid=primary_is_valid,
                factual=factual,
                numerical_columns=numerical,
                categorical_groups=groups,
                config=replace(
                    config,
                    n_counterfactuals=config.archive_size,
                    max_gower_ratio=1.0,
                    max_gower_increase=1.0,
                    max_extra_actions=n_action_units,
                ),
            )
            archive = {state.row.tobytes(): state for state in retained}

    selected = _select_diverse_set(
        list(archive.values()),
        primary,
        primary_is_valid=primary_is_valid,
        factual=factual,
        numerical_columns=numerical,
        categorical_groups=groups,
        config=config,
    )
    if selected:
        rows = np.stack([state.row for state in selected])
        probabilities = np.asarray(
            [state.probability for state in selected],
            dtype=np.float64,
        )
    else:
        rows = np.empty((0, factual.shape[0]), dtype=np.float64)
        probabilities = np.empty(0, dtype=np.float64)
    return DiverseCounterfactualResult(
        counterfactuals=rows,
        target_probabilities=probabilities,
        histories=tuple(state.history for state in selected),
        requested_count=config.n_counterfactuals,
        archive_count=len(archive),
        search_depth=search_depth,
    )
