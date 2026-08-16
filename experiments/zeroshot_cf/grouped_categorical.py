"""Atomic categorical actions for mixed-data counterfactual search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.greedy import project_candidate_values


@dataclass(frozen=True)
class GroupedCategoricalCodec:
    """Encode one-hot groups as single categorical columns for TabICL."""

    scalar_columns: tuple[int, ...]
    groups: tuple[OneHotActionGroup, ...]
    n_original_features: int

    @classmethod
    def from_matrix(
        cls,
        X: np.ndarray,
        groups: Sequence[OneHotActionGroup],
    ) -> "GroupedCategoricalCodec":
        matrix = np.asarray(X)
        if matrix.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {matrix.shape}")
        grouped = {col for group in groups for col in group.columns}
        scalar = tuple(i for i in range(matrix.shape[1]) if i not in grouped)
        codec = cls(scalar, tuple(groups), matrix.shape[1])
        codec.encode(matrix)  # validate the training representation immediately
        return codec

    @property
    def categorical_columns(self) -> tuple[int, ...]:
        start = len(self.scalar_columns)
        return tuple(range(start, start + len(self.groups)))

    def encoded_column_for_group(self, group: OneHotActionGroup) -> int:
        return len(self.scalar_columns) + self.groups.index(group)

    def encoded_columns_for_scalars(
        self,
        columns: Sequence[int],
    ) -> tuple[int, ...]:
        """Map original scalar columns to their compact encoded positions."""
        positions = {column: i for i, column in enumerate(self.scalar_columns)}
        try:
            return tuple(positions[int(column)] for column in columns)
        except KeyError as exc:
            raise ValueError(
                f"column {exc.args[0]} belongs to a categorical group"
            ) from exc

    def encode(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_original_features:
            raise ValueError(
                "X has incompatible shape for grouped categorical encoding: "
                f"{matrix.shape}"
            )

        encoded = [matrix[:, self.scalar_columns]]
        for group in self.groups:
            values = matrix[:, group.columns]
            binary = np.isclose(values, 0.0) | np.isclose(values, 1.0)
            if not np.all(binary) or not np.allclose(values.sum(axis=1), 1.0):
                raise ValueError(
                    f"one-hot group {group.name!r} contains an invalid row"
                )
            encoded.append(np.argmax(values, axis=1).reshape(-1, 1))
        return np.concatenate(encoded, axis=1).astype(np.float64, copy=False)

    def encode_row(self, x: np.ndarray) -> np.ndarray:
        return self.encode(np.asarray(x).reshape(1, -1))[0]


class CompactMixedSampler:
    """Present the original feature API over a compact mixed TabICL sampler.

    The greedy search continues to use original transformed column indices, but
    TabICL receives one column per categorical variable instead of every dummy.
    Only scalar numerical columns are sampled through this adapter; categorical
    changes are handled atomically by the mixed search or categorical fallback.
    """

    def __init__(self, sampler, codec: GroupedCategoricalCodec) -> None:
        self.sampler = sampler
        self.codec = codec

    def _encoded_candidates(self, columns: Sequence[int]) -> tuple[int, ...]:
        return self.codec.encoded_columns_for_scalars(columns)

    def sample_candidates(
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        **kwargs,
    ) -> np.ndarray:
        return self.sampler.sample_candidates(
            self.codec.encode(X_query),
            self._encoded_candidates(candidate_cols),
            **kwargs,
        )

    def sample_candidate_grid(
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        **kwargs,
    ) -> np.ndarray:
        return self.sampler.sample_candidate_grid(
            self.codec.encode(X_query),
            self._encoded_candidates(candidate_cols),
            **kwargs,
        )

    def sample_feature(
        self,
        X_query: np.ndarray,
        target_col: int,
        **kwargs,
    ) -> np.ndarray:
        encoded_col = self._encoded_candidates([target_col])[0]
        return self.sampler.sample_feature(
            self.codec.encode(X_query),
            encoded_col,
            **kwargs,
        )


CategoryDistribution = Callable[
    [np.ndarray, OneHotActionGroup], tuple[np.ndarray, np.ndarray]
]
ConditionedCategoryDistribution = Callable[
    [np.ndarray, OneHotActionGroup, float | None], tuple[np.ndarray, np.ndarray]
]


def quantile_grid_log_density(
    values: np.ndarray,
    quantiles: Sequence[float],
    *,
    min_slope: float = 1e-6,
    max_slope: float = 1e6,
) -> np.ndarray:
    """Approximate TabICL log-density from an already evaluated quantile grid.

    TabICL represents a numerical conditional through a monotone quantile
    function ``Q(alpha)``. Its density satisfies
    ``log p(Q(alpha)) = -log(dQ / d alpha)``. Central finite differences use
    the returned counterfactual grid directly, so this score requires no
    additional TabICL prediction. The last axis of ``values`` must correspond
    to ``quantiles``; any leading feature/confidence dimensions are preserved.
    """
    grid = np.asarray(values, dtype=np.float64)
    levels = np.asarray(quantiles, dtype=np.float64)
    if grid.ndim == 0 or grid.shape[-1] != len(levels):
        raise ValueError(
            "values must have one trailing entry per quantile; "
            f"got shape {grid.shape} and {len(levels)} levels"
        )
    if levels.ndim != 1 or len(levels) == 0:
        raise ValueError("quantiles must be a non-empty 1D sequence")
    if not np.all(np.isfinite(levels)) or np.any(np.diff(levels) <= 0):
        raise ValueError("quantiles must be finite, strictly increasing values")
    if min_slope <= 0 or max_slope < min_slope:
        raise ValueError("density slope bounds are invalid")
    if len(levels) == 1:
        return np.zeros_like(grid, dtype=np.float64)

    slopes = np.gradient(grid, levels, axis=-1, edge_order=1)
    slopes = np.clip(slopes, min_slope, max_slope)
    return -np.log(slopes)


def greedy_mixed_counterfactual(  # noqa: PLR0913
    sampler,
    disc,
    x: np.ndarray,
    y_target: int,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    *,
    candidate_quantiles: Sequence[float] | None = None,
    candidate_confidences: Sequence[float] | None = None,
    feature_domains=None,
    plausibility_model=None,
    use_tabicl_local_plausibility: bool = False,
    max_validity_steps: int | None = None,
    allow_revisits: bool = True,
    max_refinement_steps: int = 2,
    min_relative_lof_gain: float = 0.05,
    refinement_lof_threshold: float | None = None,
    tau: float = 0.5,
    temperature: float = 1e-9,
    category_distribution: ConditionedCategoryDistribution | None = None,
    categorical_proposal_count: int = 1,
) -> tuple[np.ndarray, list[int], dict]:
    """Greedily select the best action across numerical and categorical types.

    At every step, all remaining scalar feature proposals and all legal atomic
    category swaps compete in one discriminator batch. Before a valid proposal
    exists, the proposal with the highest target-class probability wins,
    provided it strictly improves on the incumbent. As soon as one or more
    proposals are valid, validity becomes a hard gate. The valid proposal is
    then selected either by LOF or by the local conditional score already
    produced by TabICL. LOF supports post-valid refinement; the local TabICL
    score deliberately stops at the validity boundary because it describes
    only the proposed feature, not the joint plausibility of the completed row.

    TabICL ranks categorical alternatives within each group before the target
    classifier compares the resulting rows globally. Each action unit is used
    once unless ``allow_revisits`` is enabled.
    """
    n_action_units = len(numerical_columns) + len(categorical_groups)
    if max_validity_steps is None:
        max_validity_steps = n_action_units
    if max_validity_steps < 1:
        raise ValueError("max_validity_steps must be at least 1")
    if max_refinement_steps < 0:
        raise ValueError("max_refinement_steps must be non-negative")
    if not 0.0 <= min_relative_lof_gain < 1.0:
        raise ValueError("min_relative_lof_gain must be in [0, 1)")
    if refinement_lof_threshold is not None and refinement_lof_threshold <= 0:
        raise ValueError("refinement_lof_threshold must be positive")
    if use_tabicl_local_plausibility and plausibility_model is not None:
        raise ValueError("TabICL local plausibility and LOF are mutually exclusive")
    if use_tabicl_local_plausibility and candidate_quantiles is None:
        raise ValueError("TabICL local plausibility requires candidate_quantiles")
    if categorical_proposal_count < 1:
        raise ValueError("categorical_proposal_count must be at least 1")
    numerical = [int(column) for column in numerical_columns]
    groups = list(categorical_groups)
    quantiles = (
        None
        if candidate_quantiles is None
        else np.asarray(candidate_quantiles, dtype=np.float64)
    )
    if use_tabicl_local_plausibility and len(quantiles) < 2:
        raise ValueError(
            "TabICL local plausibility requires at least two candidate quantiles"
        )
    confidences = (
        None
        if candidate_confidences is None
        else np.asarray(candidate_confidences, dtype=np.float64)
    )
    if confidences is not None and quantiles is None:
        raise ValueError("candidate_confidences require candidate_quantiles")

    factual = np.asarray(x, dtype=np.float64)
    current = factual.copy()
    changed_order: list[int] = []
    history: list[dict] = []
    categorical_history: list[dict] = []
    search_passes_used = 0
    flipped = False

    def classifier_outputs(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return target probabilities and labels from one classifier call."""
        probability_matrix = np.asarray(disc.predict_proba(np.atleast_2d(rows)))
        classes = np.asarray(
            getattr(disc, "classes_", np.arange(probability_matrix.shape[1]))
        )
        target_positions = np.flatnonzero(classes == y_target)
        if len(target_positions) != 1:
            raise ValueError(
                f"target class {y_target} is absent from classifier classes"
            )
        predictions = classes[np.argmax(probability_matrix, axis=1)]
        return probability_matrix[:, int(target_positions[0])], predictions

    def flip_state(row: np.ndarray) -> tuple[bool, float]:
        probabilities, predictions = classifier_outputs(row.reshape(1, -1))
        probability = float(probabilities[0])
        prediction = int(predictions[0])
        return prediction == y_target and probability >= tau, probability

    def counterfactual_costs(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return action-level sparsity and Euclidean factual distance."""
        matrix = np.atleast_2d(rows)
        sparsity = np.zeros(len(matrix), dtype=np.int64)
        if numerical:
            sparsity += np.count_nonzero(
                ~np.isclose(matrix[:, numerical], factual[numerical]),
                axis=1,
            )
        for group in groups:
            columns = list(group.columns)
            factual_category = int(np.argmax(factual[columns]))
            sparsity += np.argmax(matrix[:, columns], axis=1) != factual_category
        proximity = np.linalg.norm(matrix - factual, axis=1)
        return sparsity, proximity

    def valid_lof_scores(
        rows: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray | None:
        """Score only valid candidates; invalid rows do not use plausibility."""
        if plausibility_model is None or not np.any(valid_mask):
            return None
        scores = np.full(len(rows), np.inf, dtype=np.float64)
        scores[valid_mask] = -np.asarray(
            plausibility_model.score_samples(rows[valid_mask]),
            dtype=np.float64,
        )
        return scores

    def select_valid_candidate(
        eligible: np.ndarray,
        probabilities: np.ndarray,
        lof_scores: np.ndarray | None,
        tabicl_local_scores: np.ndarray,
    ) -> int:
        """Apply the configured plausibility rule within the validity gate."""
        if use_tabicl_local_plausibility and np.any(
            np.isfinite(tabicl_local_scores[eligible])
        ):
            eligible_scores = tabicl_local_scores[eligible]
            best_score = np.max(eligible_scores)
            tied = eligible[
                np.isclose(eligible_scores, best_score, rtol=1e-9, atol=1e-12)
            ]
            return int(tied[np.argmax(probabilities[tied])])
        if lof_scores is not None:
            return int(eligible[np.argmin(lof_scores[eligible])])
        return int(eligible[np.argmax(probabilities[eligible])])

    flipped, current_probability = flip_state(current)
    current_lof = (
        None
        if plausibility_model is None or not flipped
        else float(-plausibility_model.score_samples(current.reshape(1, -1))[0])
    )
    best_row = current.copy()
    best_probability = current_probability
    best_history_length = 0
    initial_valid_step = 0 if flipped else None
    refinement_steps = 0
    validity_steps = 0
    visited_rows: set[bytes] = {current.tobytes()}
    refined_numerical: set[int] = set()
    refined_groups: set[str] = set()

    round_limit = max_validity_steps if allow_revisits else 1
    for round_index in range(round_limit):
        if flipped and (
            plausibility_model is None
            or refinement_steps >= max_refinement_steps
            or (
                refinement_lof_threshold is not None
                and current_lof is not None
                and current_lof <= refinement_lof_threshold
            )
        ):
            break
        search_passes_used = round_index + 1
        used_numerical: set[int] = set()
        used_groups: set[str] = set()
        committed_this_round = 0

        while True:
            if not flipped and validity_steps >= max_validity_steps:
                break
            if flipped and (
                plausibility_model is None
                or refinement_steps >= max_refinement_steps
                or (
                    refinement_lof_threshold is not None
                    and current_lof is not None
                    and current_lof <= refinement_lof_threshold
                )
            ):
                break
            trial_rows: list[np.ndarray] = []
            metadata: list[dict] = []
            available_numerical = [
                column
                for column in numerical
                if column not in used_numerical
                and (not flipped or column not in refined_numerical)
            ]
            if available_numerical:
                if quantiles is None:
                    numerical_values = np.asarray(
                        sampler.sample_candidates(
                            current.reshape(1, -1),
                            available_numerical,
                            sample_temperature=temperature,
                            fixed_target=y_target,
                        ),
                        dtype=np.float64,
                    )
                    expected = (len(available_numerical),)
                    if numerical_values.shape != expected:
                        raise ValueError(
                            "sample_candidates returned an unexpected shape; "
                            f"expected {expected}, got {numerical_values.shape}"
                        )
                    expanded_columns = np.asarray(available_numerical, dtype=int)
                    flat_values = project_candidate_values(
                        available_numerical,
                        numerical_values,
                        feature_domains,
                    )
                    numerical_metadata = [
                        {
                            "action_type": "numerical",
                            "feature": int(column),
                            "quantile": None,
                            "confidence": None,
                        }
                        for column in expanded_columns
                    ]
                else:
                    numerical_values = np.asarray(
                        sampler.sample_candidate_grid(
                            current.reshape(1, -1),
                            available_numerical,
                            quantiles=quantiles,
                            fixed_target=y_target,
                            confidences=confidences,
                        ),
                        dtype=np.float64,
                    )
                    n_confidences = 1 if confidences is None else len(confidences)
                    expected = (
                        (len(available_numerical), len(quantiles))
                        if confidences is None
                        else (
                            len(available_numerical),
                            n_confidences,
                            len(quantiles),
                        )
                    )
                    if numerical_values.shape != expected:
                        raise ValueError(
                            "sample_candidate_grid returned an unexpected shape; "
                            f"expected {expected}, got {numerical_values.shape}"
                        )
                    numerical_log_density = quantile_grid_log_density(
                        numerical_values,
                        quantiles,
                    )
                    expanded_columns = np.repeat(
                        np.asarray(available_numerical, dtype=int),
                        n_confidences * len(quantiles),
                    )
                    expanded_quantiles = np.tile(
                        quantiles,
                        len(available_numerical) * n_confidences,
                    )
                    expanded_confidences = (
                        None
                        if confidences is None
                        else np.tile(
                            np.repeat(confidences, len(quantiles)),
                            len(available_numerical),
                        )
                    )
                    flat_values = project_candidate_values(
                        expanded_columns.tolist(),
                        numerical_values.reshape(-1),
                        feature_domains,
                    )
                    numerical_metadata = [
                        {
                            "action_type": "numerical",
                            "feature": int(column),
                            "quantile": float(expanded_quantiles[position]),
                            "confidence": (
                                None
                                if expanded_confidences is None
                                else float(expanded_confidences[position])
                            ),
                            "tabicl_local_log_score": float(
                                numerical_log_density.reshape(-1)[position]
                            ),
                            "tabicl_local_score_kind": "log_density",
                        }
                        for position, column in enumerate(expanded_columns)
                    ]

                numerical_trials = np.repeat(
                    current.reshape(1, -1),
                    len(flat_values),
                    axis=0,
                )
                numerical_trials[
                    np.arange(len(flat_values)),
                    expanded_columns,
                ] = flat_values
                trial_rows.extend(numerical_trials)
                metadata.extend(numerical_metadata)

            category_scores_by_group: dict[
                str, dict[int, tuple[float, float | None]]
            ] = {}
            for group in groups:
                if group.name in used_groups or (
                    flipped and group.name in refined_groups
                ):
                    continue
                columns = list(group.columns)
                group_values = current[columns]
                if not np.isclose(group_values.sum(), 1.0):
                    raise ValueError(f"one-hot group {group.name!r} is invalid")
                previous_category = int(np.argmax(group_values))
                category_scores: dict[int, tuple[float, float | None]] = {}
                if category_distribution is not None:
                    anchors = [None] if confidences is None else confidences.tolist()
                    for anchor in anchors:
                        categories, conditional_probabilities = category_distribution(
                            current,
                            group,
                            anchor,
                        )
                        for category, probability in zip(
                            np.asarray(categories, dtype=int),
                            np.asarray(conditional_probabilities, dtype=np.float64),
                            strict=True,
                        ):
                            previous_score = category_scores.get(int(category))
                            if (
                                previous_score is None
                                or probability > previous_score[0]
                            ):
                                category_scores[int(category)] = (
                                    float(probability),
                                    None if anchor is None else float(anchor),
                                )
                else:
                    category_scores = {
                        category: (1.0, None) for category in range(len(columns))
                    }
                category_scores_by_group[group.name] = category_scores
                alternatives = [
                    category
                    for category in range(len(columns))
                    if category != previous_category
                ]
                alternatives.sort(
                    key=lambda category: category_scores.get(category, (0.0, None))[0],
                    reverse=True,
                )
                # TabICL ranks proposals within a categorical action unit. The
                # target classifier subsequently compares their complete rows
                # against all numerical proposals.
                alternatives = alternatives[:categorical_proposal_count]
                for proposal_rank, category in enumerate(alternatives, start=1):
                    if category == previous_category:
                        continue
                    conditional_probability, confidence_anchor = category_scores.get(
                        category, (0.0, None)
                    )
                    trial = current.copy()
                    trial[columns] = 0.0
                    trial[group.columns[category]] = 1.0
                    trial_rows.append(trial)
                    metadata.append(
                        {
                            "action_type": "categorical",
                            "group": group.name,
                            "group_object": group,
                            "from_category": previous_category,
                            "to_category": category,
                            "tabicl_conditional_probability": (conditional_probability),
                            "tabicl_local_log_score": float(
                                np.log(max(conditional_probability, 1e-12))
                            ),
                            "tabicl_local_score_kind": "log_probability",
                            "tabicl_confidence_anchor": confidence_anchor,
                            "tabicl_proposal_rank": proposal_rank,
                            "in_tabicl_support": category in category_scores,
                        }
                    )

            if not trial_rows:
                break
            trials = np.stack(trial_rows)
            probabilities, predictions = classifier_outputs(trials)
            valid = (predictions == y_target) & (probabilities >= tau)
            lof_scores = valid_lof_scores(trials, valid)
            tabicl_local_scores = np.asarray(
                [
                    float(item.get("tabicl_local_log_score", -np.inf))
                    for item in metadata
                ],
                dtype=np.float64,
            )

            if flipped:
                if (
                    lof_scores is None
                    or current_lof is None
                    or refinement_steps >= max_refinement_steps
                    or (
                        refinement_lof_threshold is not None
                        and current_lof <= refinement_lof_threshold
                    )
                ):
                    break
                relative_lof_gain = (current_lof - lof_scores) / max(
                    abs(current_lof), 1e-12
                )
                eligible = np.flatnonzero(
                    valid
                    & (relative_lof_gain >= min_relative_lof_gain)
                    & np.asarray([row.tobytes() not in visited_rows for row in trials])
                )
                if not len(eligible):
                    break
                candidate_sparsity, candidate_proximity = counterfactual_costs(trials)
                ranked = np.lexsort(
                    (
                        lof_scores[eligible],
                        candidate_proximity[eligible],
                        candidate_sparsity[eligible],
                    )
                )
                best = int(eligible[ranked[0]])
            elif valid.any():
                eligible = np.flatnonzero(valid)
                best = select_valid_candidate(
                    eligible,
                    probabilities,
                    lof_scores,
                    tabicl_local_scores,
                )
            else:
                best = int(np.argmax(probabilities))

            selected_probability = float(probabilities[best])
            if (
                not flipped
                and selected_probability <= current_probability + 1e-12
                and category_distribution is not None
            ):
                # The ranked shortlist is the normal path. If it cannot make
                # progress, expose every remaining legal category once so
                # local TabICL support gaps cannot reduce coverage.
                existing = {
                    (item.get("group"), item.get("to_category")) for item in metadata
                }
                for fallback_group in groups:
                    if fallback_group.name in used_groups:
                        continue
                    columns = list(fallback_group.columns)
                    previous_category = int(np.argmax(current[columns]))
                    for category in range(len(columns)):
                        if (
                            category == previous_category
                            or (fallback_group.name, category) in existing
                        ):
                            continue
                        trial = current.copy()
                        trial[columns] = 0.0
                        trial[fallback_group.columns[category]] = 1.0
                        fallback_scores = category_scores_by_group.get(
                            fallback_group.name, {}
                        )
                        conditional_probability, confidence_anchor = (
                            fallback_scores.get(category, (0.0, None))
                        )
                        trial_rows.append(trial)
                        metadata.append(
                            {
                                "action_type": "categorical",
                                "group": fallback_group.name,
                                "group_object": fallback_group,
                                "from_category": previous_category,
                                "to_category": category,
                                "tabicl_conditional_probability": (
                                    conditional_probability
                                ),
                                "tabicl_local_log_score": float(
                                    np.log(max(conditional_probability, 1e-12))
                                ),
                                "tabicl_local_score_kind": "log_probability",
                                "tabicl_confidence_anchor": confidence_anchor,
                                "tabicl_proposal_rank": None,
                                "in_tabicl_support": category in fallback_scores,
                                "coverage_fallback": True,
                            }
                        )
                trials = np.stack(trial_rows)
                probabilities, predictions = classifier_outputs(trials)
                valid = (predictions == y_target) & (probabilities >= tau)
                lof_scores = valid_lof_scores(trials, valid)
                tabicl_local_scores = np.asarray(
                    [
                        float(item.get("tabicl_local_log_score", -np.inf))
                        for item in metadata
                    ],
                    dtype=np.float64,
                )
                if valid.any():
                    eligible = np.flatnonzero(valid)
                    best = select_valid_candidate(
                        eligible,
                        probabilities,
                        lof_scores,
                        tabicl_local_scores,
                    )
                else:
                    best = int(np.argmax(probabilities))
                selected_probability = float(probabilities[best])
            if not flipped and selected_probability <= current_probability + 1e-12:
                break
            if trials[best].tobytes() in visited_rows:
                break

            was_flipped = flipped
            selected = dict(metadata[best])
            selected.pop("group_object", None)
            selected.update(
                {
                    "search_pass": search_passes_used,
                    "selection_phase": (
                        "plausibility_refinement" if was_flipped else "validity_search"
                    ),
                    "target_probability": selected_probability,
                    "lof": (None if lof_scores is None else float(lof_scores[best])),
                    "tabicl_local_log_score": (
                        None
                        if not np.isfinite(tabicl_local_scores[best])
                        else float(tabicl_local_scores[best])
                    ),
                    "immediate_valid": bool(valid[best]),
                    "n_candidates": len(trials),
                    "n_valid_candidates": int(valid.sum()),
                }
            )
            selected_sparsity, selected_proximity = counterfactual_costs(trials[best])
            selected["action_sparsity"] = int(selected_sparsity[0])
            selected["proximity_l2"] = float(selected_proximity[0])
            if was_flipped and current_lof is not None and lof_scores is not None:
                selected["relative_lof_gain"] = float(
                    (current_lof - lof_scores[best]) / max(abs(current_lof), 1e-12)
                )
            current = trials[best]
            visited_rows.add(current.tobytes())
            current_probability = selected_probability
            if lof_scores is not None:
                current_lof = float(lof_scores[best])
            committed_this_round += 1

            raw_selected = metadata[best]
            if raw_selected["action_type"] == "numerical":
                feature = int(raw_selected["feature"])
                used_numerical.add(feature)
                if was_flipped:
                    refined_numerical.add(feature)
            else:
                group = raw_selected["group_object"]
                used_groups.add(group.name)
                if was_flipped:
                    refined_groups.add(group.name)
                categorical_history.append(selected.copy())

            history.append(selected)
            flipped, current_probability = flip_state(current)
            if was_flipped:
                refinement_steps += 1
            elif flipped and initial_valid_step is None:
                initial_valid_step = len(history)
            if not was_flipped:
                validity_steps += 1
            if current_probability > best_probability:
                best_row = current.copy()
                best_probability = current_probability
                best_history_length = len(history)

        if committed_this_round == 0:
            break

    attempt_history = history.copy()
    if not flipped:
        current = best_row
        history = history[:best_history_length]
        categorical_history = [
            item for item in history if item["action_type"] == "categorical"
        ]

    for column in np.flatnonzero(current != factual):
        changed_order.append(int(column))
    return (
        current,
        changed_order,
        {
            "flipped": bool(flipped),
            "steps": len(history),
            "search_passes": search_passes_used,
            "validity_steps": validity_steps,
            "max_validity_steps": max_validity_steps,
            "allow_revisits": allow_revisits,
            "history": history,
            "attempt_history": attempt_history,
            "selection_history": history,
            "attempt_selection_history": attempt_history,
            "categorical_history": categorical_history,
            "best_target_probability": float(best_probability),
            "initial_valid_step": initial_valid_step,
            "refinement_steps": refinement_steps,
        },
    )


def grouped_categorical_fallback(
    x_start: np.ndarray,
    *,
    disc,
    y_target: int,
    groups: Sequence[OneHotActionGroup],
    category_distribution: CategoryDistribution,
    plausibility_model=None,
    tau: float = 0.5,
) -> tuple[np.ndarray, list[int], dict]:
    """Greedily apply valid whole-group category swaps.

    Every category in the metadata-defined domain is considered. Before a flip,
    the candidate with maximum target-class probability is committed. As soon
    as any candidate is valid, validity becomes a hard gate and the lowest-LOF
    valid candidate is selected. TabICL is queried only for the selected
    group's conditional distribution; querying all groups cannot change this
    selection rule and is needlessly expensive. A group is edited at most once.
    """
    x_cf = np.asarray(x_start, dtype=np.float64).copy()
    groups = list(groups)
    used_groups: set[str] = set()
    changed_columns: list[int] = []
    history: list[dict] = []

    def classifier_outputs(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probability_matrix = np.asarray(disc.predict_proba(np.atleast_2d(rows)))
        classes = np.asarray(
            getattr(disc, "classes_", np.arange(probability_matrix.shape[1]))
        )
        target_positions = np.flatnonzero(classes == y_target)
        if len(target_positions) != 1:
            raise ValueError(
                f"target class {y_target} is absent from classifier classes"
            )
        predictions = classes[np.argmax(probability_matrix, axis=1)]
        return probability_matrix[:, int(target_positions[0])], predictions

    def flip_state(row: np.ndarray) -> tuple[bool, float]:
        probabilities, predictions = classifier_outputs(row.reshape(1, -1))
        probability = float(probabilities[0])
        prediction = int(predictions[0])
        return prediction == y_target and probability >= tau, probability

    flipped, current_probability = flip_state(x_cf)
    while not flipped:
        trials: list[np.ndarray] = []
        trial_groups: list[OneHotActionGroup] = []
        trial_categories: list[int] = []
        for group in groups:
            if group.name in used_groups:
                continue
            group_values = x_cf[list(group.columns)]
            if not np.isclose(group_values.sum(), 1.0):
                raise ValueError(f"one-hot group {group.name!r} is invalid")
            current_category = int(np.argmax(group_values))
            # Enumerate the complete metadata domain to protect coverage. A
            # category missing from the local kNN context remains a valid action.
            for category in range(len(group.columns)):
                if category == current_category:
                    continue
                trial = x_cf.copy()
                trial[list(group.columns)] = 0.0
                trial[group.columns[category]] = 1.0
                trials.append(trial)
                trial_groups.append(group)
                trial_categories.append(category)

        if not trials:
            break

        trial_matrix = np.stack(trials)
        target_probabilities, predictions = classifier_outputs(trial_matrix)
        valid = (predictions == y_target) & (target_probabilities >= tau)
        lof_scores = None
        if plausibility_model is not None and valid.any():
            lof_scores = np.full(len(trial_matrix), np.inf, dtype=np.float64)
            lof_scores[valid] = -np.asarray(
                plausibility_model.score_samples(trial_matrix[valid]),
                dtype=np.float64,
            )

        if valid.any():
            eligible = np.flatnonzero(valid)
            if lof_scores is None:
                best = int(eligible[np.argmax(target_probabilities[eligible])])
            else:
                best = int(eligible[np.argmin(lof_scores[eligible])])
        else:
            best = int(np.argmax(target_probabilities))
            if float(target_probabilities[best]) <= current_probability:
                break

        selected_group = trial_groups[best]
        categories, conditional_probabilities = category_distribution(
            x_cf,
            selected_group,
        )
        categories = np.asarray(categories, dtype=int)
        conditional_probabilities = np.asarray(
            conditional_probabilities,
            dtype=np.float64,
        )
        if categories.ndim != 1 or conditional_probabilities.shape != categories.shape:
            raise ValueError("category_distribution must return aligned 1D arrays")
        if any(
            category < 0 or category >= len(selected_group.columns)
            for category in categories
        ):
            raise ValueError(
                f"TabICL returned a category outside group {selected_group.name!r}"
            )
        conditional_probability = dict(
            zip(categories.tolist(), conditional_probabilities.tolist())
        ).get(trial_categories[best], 0.0)
        previous_category = int(np.argmax(x_cf[list(selected_group.columns)]))
        x_cf = trial_matrix[best]
        used_groups.add(selected_group.name)
        for column in (
            selected_group.columns[previous_category],
            selected_group.columns[trial_categories[best]],
        ):
            if column not in changed_columns:
                changed_columns.append(column)
        flipped, current_probability = flip_state(x_cf)
        history.append(
            {
                "group": selected_group.name,
                "from_category": previous_category,
                "to_category": trial_categories[best],
                "target_probability": current_probability,
                "tabicl_conditional_probability": float(conditional_probability),
                "lof": None if lof_scores is None else float(lof_scores[best]),
                "immediate_valid": bool(valid[best]),
                "n_candidates": len(trials),
                "n_valid_candidates": int(valid.sum()),
            }
        )

    return (
        x_cf,
        changed_columns,
        {
            "flipped": bool(flipped),
            "steps": len(history),
            "history": history,
            "final_target_probability": float(current_probability),
        },
    )
