"""Atomic categorical actions for mixed-data counterfactual search."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np
from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.greedy import project_candidate_values
from experiments.zeroshot_cf.mixed_distance import (
    action_unit_change_count,
    grouped_gower_distance,
)


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

        encoded: list[np.ndarray] = [matrix[:, self.scalar_columns]]
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

    def __init__(self, sampler: Any, codec: GroupedCategoricalCodec) -> None:
        super().__init__()
        self.sampler = sampler
        self.codec = codec

    def _encoded_candidates(self, columns: Sequence[int]) -> tuple[int, ...]:
        return self.codec.encoded_columns_for_scalars(columns)

    def sample_candidates(
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        **kwargs: Any,
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
        **kwargs: Any,
    ) -> np.ndarray:
        return self.sampler.sample_candidate_grid(
            self.codec.encode(X_query),
            self._encoded_candidates(candidate_cols),
            **kwargs,
        )

    def sample_candidates_batch(
        self,
        X_queries: np.ndarray,
        candidate_cols: Sequence[int],
        **kwargs: Any,
    ) -> np.ndarray:
        """Impute original-space query/feature pairs in one compact batch."""
        return self.sampler.sample_candidates_batch(
            self.codec.encode(X_queries),
            self._encoded_candidates(candidate_cols),
            **kwargs,
        )

    def sample_candidate_grid_batch(
        self,
        X_queries: np.ndarray,
        candidate_cols: Sequence[int],
        **kwargs: Any,
    ) -> np.ndarray:
        """Evaluate original-space query/feature quantile grids in one batch."""
        return self.sampler.sample_candidate_grid_batch(
            self.codec.encode(X_queries),
            self._encoded_candidates(candidate_cols),
            **kwargs,
        )

    def sample_feature(
        self,
        X_query: np.ndarray,
        target_col: int,
        **kwargs: Any,
    ) -> np.ndarray:
        encoded_col = self._encoded_candidates([target_col])[0]
        return self.sampler.sample_feature(
            self.codec.encode(X_query),
            encoded_col,
            **kwargs,
        )

    def score_joint_rows(self, X_rows: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Evaluate complete original-space rows in the compact TabICL space."""
        return self.sampler.score_joint_rows(self.codec.encode(X_rows), **kwargs)


ConditionedCategoryDistribution = Callable[
    [np.ndarray, OneHotActionGroup, float | None], tuple[np.ndarray, np.ndarray]
]


def greedy_mixed_counterfactual(  # noqa: PLR0913
    sampler: Any,
    disc: Any,
    x: np.ndarray,
    y_target: int,
    numerical_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    *,
    candidate_quantiles: Sequence[float] | None = None,
    candidate_confidences: Sequence[float] | None = None,
    feature_domains: Any = None,
    cf_mode: str = "sparse",
    tabicl_joint_plausibility: Any = None,
    max_validity_steps: int | None = None,
    allow_revisits: bool = True,
    joint_shortlist_size: int = 16,
    max_extra_actions: int = 1,
    min_joint_log_gain: float = 0.0,
    tau: float = 0.5,
    temperature: float = 1e-9,
    category_distribution: ConditionedCategoryDistribution | None = None,
    categorical_proposal_count: int = 1,
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    """Greedily select the best action across numerical and categorical types.

    At every validity-search step, all scalar and categorical TabICL proposals
    compete in one discriminator batch. Invalid states are ranked only by
    target-probability ascent. Once valid proposals exist, the candidate with
    the lowest grouped Gower distance is selected subject to validity and the
    search/action budget, then committed without a joint-density call.
    ``sparse`` mode returns it immediately. ``data_plausible`` mode performs
    one validity-preserving refinement attempt. It keeps at most one
    representative per action unit, fills a bounded shortlist, and scores the
    sparse incumbent plus that shortlist in one whole-row TabICL batch. The
    incumbent remains the fallback unless raw joint log density improves.

    TabICL ranks categorical alternatives within each group before the target
    classifier compares the resulting rows globally. Each action unit is used
    once unless ``allow_revisits`` is enabled.
    """
    n_action_units = len(numerical_columns) + len(categorical_groups)
    if max_validity_steps is None:
        max_validity_steps = n_action_units
    if max_validity_steps < 1:
        raise ValueError("max_validity_steps must be at least 1")
    if cf_mode not in {"sparse", "data_plausible"}:
        raise ValueError("cf_mode must be 'sparse' or 'data_plausible'")
    if joint_shortlist_size < 1:
        raise ValueError("joint_shortlist_size must be at least 1")
    if max_extra_actions < 0:
        raise ValueError("max_extra_actions must be non-negative")
    if min_joint_log_gain < 0:
        raise ValueError("min_joint_log_gain must be non-negative")
    if cf_mode == "data_plausible" and tabicl_joint_plausibility is None:
        raise ValueError("data_plausible mode requires a TabICL joint scorer")
    if cf_mode == "sparse" and tabicl_joint_plausibility is not None:
        raise ValueError("sparse mode must not receive a TabICL joint scorer")
    if categorical_proposal_count < 1:
        raise ValueError("categorical_proposal_count must be at least 1")
    numerical = [int(column) for column in numerical_columns]
    groups = list(categorical_groups)
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
        """Return action-unit change count and grouped Gower distance."""
        return (
            action_unit_change_count(rows, factual, numerical, groups),
            grouped_gower_distance(rows, factual, numerical, groups),
        )

    def proposal_supports(candidate_metadata: Sequence[dict]) -> np.ndarray:
        """Return comparable local-support ranks for cheap prescreening."""
        return np.asarray(
            [
                (
                    1.0
                    if item.get("quantile") is None
                    else 2.0
                    * min(float(item["quantile"]), 1.0 - float(item["quantile"]))
                )
                if item["action_type"] == "numerical"
                else float(item.get("tabicl_conditional_probability", 0.0))
                for item in candidate_metadata
            ],
            dtype=np.float64,
        )

    def joint_shortlist(
        eligible: np.ndarray,
        rows: np.ndarray,
        probabilities: np.ndarray,
        candidate_metadata: Sequence[dict],
    ) -> np.ndarray:
        """Select a small per-action batch before whole-row scoring."""
        if not len(eligible):
            return np.empty(0, dtype=int)
        _, candidate_gower = counterfactual_costs(rows)
        support_penalty = -np.log(
            np.clip(proposal_supports(candidate_metadata), 1e-12, 1.0)
        )

        def action_key(index: int) -> tuple[str, int | str]:
            item = candidate_metadata[index]
            if item["action_type"] == "numerical":
                return "numerical", int(item["feature"])
            return "categorical", str(item["group"])

        ranked = eligible[
            np.lexsort(
                (
                    -probabilities[eligible],
                    support_penalty[eligible],
                    candidate_gower[eligible],
                )
            )
        ]
        representatives: list[int] = []
        represented: set[tuple[str, int | str]] = set()
        for raw_index in ranked:
            index = int(raw_index)
            key = action_key(index)
            if key in represented:
                continue
            representatives.append(index)
            represented.add(key)
            if len(representatives) == joint_shortlist_size:
                return np.asarray(representatives, dtype=int)
        selected = set(representatives)
        for raw_index in ranked:
            index = int(raw_index)
            if index in selected:
                continue
            representatives.append(index)
            if len(representatives) == joint_shortlist_size:
                break
        return np.asarray(representatives, dtype=int)

    def select_valid_candidate(
        eligible: np.ndarray,
        probabilities: np.ndarray,
        rows: np.ndarray,
        candidate_metadata: Sequence[dict],
    ) -> int:
        """Choose the closest valid candidate without querying joint TabICL."""
        _, candidate_gower = counterfactual_costs(rows)
        proposal_support = proposal_supports(candidate_metadata)
        support_penalty = -np.log(np.clip(proposal_support, 1e-12, 1.0))
        ranked = np.lexsort(
            (
                -probabilities[eligible],
                support_penalty[eligible],
                candidate_gower[eligible],
            )
        )
        return int(eligible[ranked[0]])

    flipped, current_probability = flip_state(current)
    current_joint_log_density: float | None = None
    initial_joint_log_density: float | None = None
    initial_sparse_action_count: int | None = (
        int(counterfactual_costs(current)[0][0]) if flipped else None
    )
    initial_sparse_row: np.ndarray | None = current.copy() if flipped else None
    best_row = current.copy()
    best_probability = current_probability
    best_history_length = 0
    initial_valid_step = 0 if flipped else None
    refinement_steps = 0
    validity_steps = 0
    visited_rows: set[bytes] = {current.tobytes()}
    refined_numerical: set[int] = set()
    refined_groups: set[str] = set()
    joint_rerank_attempted = False
    post_valid_refinement = tabicl_joint_plausibility is not None
    refinement_stopping_reason = "not_started"
    joint_scoring_runtime_s = 0.0

    round_limit = max_validity_steps if allow_revisits else 1
    for round_index in range(round_limit):
        refinement_limit = 1
        if flipped and (
            not post_valid_refinement
            or refinement_steps >= refinement_limit
            or joint_rerank_attempted
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
                not post_valid_refinement
                or refinement_steps >= refinement_limit
                or joint_rerank_attempted
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

                if flipped:
                    reversion_columns = [
                        column
                        for column in available_numerical
                        if not np.isclose(current[column], factual[column])
                    ]
                    if reversion_columns:
                        reversion_trials = np.repeat(
                            current.reshape(1, -1),
                            len(reversion_columns),
                            axis=0,
                        )
                        reversion_trials[
                            np.arange(len(reversion_columns)),
                            reversion_columns,
                        ] = factual[reversion_columns]
                        trial_rows.extend(reversion_trials)
                        metadata.extend(
                            {
                                "action_type": "numerical",
                                "feature": int(column),
                                "quantile": None,
                                "confidence": None,
                                "proposal_kind": "revert",
                            }
                            for column in reversion_columns
                        )

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
                factual_category = int(np.argmax(factual[columns]))
                if (
                    flipped
                    and previous_category != factual_category
                    and factual_category not in alternatives
                ):
                    alternatives.append(factual_category)
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
                            "tabicl_confidence_anchor": confidence_anchor,
                            "tabicl_proposal_rank": proposal_rank,
                            "in_tabicl_support": category in category_scores,
                            "proposal_kind": (
                                "revert"
                                if flipped and category == factual_category
                                else "conditional"
                            ),
                        }
                    )

            if not trial_rows:
                if flipped and tabicl_joint_plausibility is not None:
                    joint_rerank_attempted = True
                    refinement_stopping_reason = "no_candidates"
                break
            trials = np.stack(trial_rows)
            probabilities, predictions = classifier_outputs(trials)
            valid = (predictions == y_target) & (probabilities >= tau)
            tabicl_joint_scores = None

            best: int | None = None
            if flipped:
                unvisited = np.asarray(
                    [row.tobytes() not in visited_rows for row in trials]
                )
                if tabicl_joint_plausibility is not None:
                    joint_rerank_attempted = True
                    if initial_sparse_action_count is None:
                        raise RuntimeError("initial sparse action count is unavailable")
                    candidate_sparsity, candidate_gower = counterfactual_costs(trials)
                    eligible_for_scoring = np.flatnonzero(
                        valid
                        & unvisited
                        & (
                            candidate_sparsity
                            <= initial_sparse_action_count + max_extra_actions
                        )
                    )
                    shortlist = joint_shortlist(
                        eligible_for_scoring,
                        trials,
                        probabilities,
                        metadata,
                    )
                    if not len(shortlist):
                        refinement_stopping_reason = "no_eligible_candidate"
                        break
                    scoring_rows = np.vstack(
                        (current.reshape(1, -1), trials[shortlist])
                    )
                    joint_started = perf_counter()
                    joint_batch = tabicl_joint_plausibility.score_rows(
                        scoring_rows,
                        y_target,
                    )
                    joint_scoring_runtime_s += perf_counter() - joint_started
                    current_joint_log_density = float(joint_batch.joint_log_density[0])
                    initial_joint_log_density = current_joint_log_density
                    tabicl_joint_scores = {
                        "joint_log_density": np.full(
                            len(trials), np.nan, dtype=np.float64
                        )
                    }
                    tabicl_joint_scores["joint_log_density"][shortlist] = (
                        joint_batch.joint_log_density[1:]
                    )
                    improving = (
                        joint_batch.joint_log_density[1:]
                        > current_joint_log_density + min_joint_log_gain
                    )
                    eligible = shortlist[improving]
                    if not len(eligible):
                        refinement_stopping_reason = "no_improving_candidate"
                        break
                    ranked = np.lexsort(
                        (
                            candidate_gower[eligible],
                            -tabicl_joint_scores["joint_log_density"][eligible],
                        )
                    )
                    best = int(eligible[ranked[0]])
                    refinement_stopping_reason = "one_shot_accepted"
                else:
                    break
            elif valid.any():
                eligible = np.flatnonzero(valid)
                best = select_valid_candidate(
                    eligible,
                    probabilities,
                    trials,
                    metadata,
                )
            else:
                best = int(np.argmax(probabilities))

            if best is None:
                break
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
                                "tabicl_confidence_anchor": confidence_anchor,
                                "tabicl_proposal_rank": None,
                                "in_tabicl_support": category in fallback_scores,
                                "coverage_fallback": True,
                            }
                        )
                trials = np.stack(trial_rows)
                probabilities, predictions = classifier_outputs(trials)
                valid = (predictions == y_target) & (probabilities >= tau)
                tabicl_joint_scores = None
                if valid.any():
                    eligible = np.flatnonzero(valid)
                    best = select_valid_candidate(
                        eligible,
                        probabilities,
                        trials,
                        metadata,
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
                    "tabicl_joint_log_density": (
                        None
                        if tabicl_joint_scores is None
                        else float(tabicl_joint_scores["joint_log_density"][best])
                    ),
                    "immediate_valid": bool(valid[best]),
                    "n_candidates": len(trials),
                    "n_valid_candidates": int(valid.sum()),
                }
            )
            selected_sparsity, selected_gower = counterfactual_costs(trials[best])
            selected["action_sparsity"] = int(selected_sparsity[0])
            selected["grouped_gower"] = float(selected_gower[0])
            current = trials[best]
            visited_rows.add(current.tobytes())
            current_probability = selected_probability
            if tabicl_joint_scores is not None:
                current_joint_log_density = float(
                    tabicl_joint_scores["joint_log_density"][best]
                )
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
                initial_sparse_action_count = int(counterfactual_costs(current)[0][0])
                initial_sparse_row = current.copy()
                if tabicl_joint_plausibility is not None:
                    # The one-shot refinement may replace the action that
                    # first reached validity, so expose every action unit once
                    # more before constructing the joint-scoring shortlist.
                    used_numerical.clear()
                    used_groups.clear()
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

    final_action_count = int(counterfactual_costs(current)[0][0])
    if not flipped:
        refinement_stopping_reason = "validity_not_reached"
    elif tabicl_joint_plausibility is None:
        refinement_stopping_reason = "sparse_valid"
    elif refinement_stopping_reason == "not_started":
        refinement_stopping_reason = "one_shot_not_attempted"
    joint_log_density_gain = (
        None
        if initial_joint_log_density is None or current_joint_log_density is None
        else current_joint_log_density - initial_joint_log_density
    )
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
            "accepted_refinement_count": refinement_steps,
            "initial_sparse_action_count": initial_sparse_action_count,
            "initial_sparse_row": initial_sparse_row,
            "final_action_count": final_action_count,
            "initial_tabicl_joint_log_density": initial_joint_log_density,
            "final_tabicl_joint_log_density": current_joint_log_density,
            "tabicl_joint_log_density_gain": joint_log_density_gain,
            "joint_scoring_batch_count": (
                0
                if tabicl_joint_plausibility is None
                else int(getattr(tabicl_joint_plausibility, "batch_count", 0))
            ),
            "joint_rows_scored": (
                0
                if tabicl_joint_plausibility is None
                else int(getattr(tabicl_joint_plausibility, "row_count", 0))
            ),
            "joint_shortlist_size": joint_shortlist_size,
            "max_extra_actions": max_extra_actions,
            "min_joint_log_gain": min_joint_log_gain,
            "joint_scoring_runtime_s": joint_scoring_runtime_s,
            "extra_actions": (
                0
                if initial_sparse_action_count is None
                else max(0, final_action_count - initial_sparse_action_count)
            ),
            "refinement_stopping_reason": refinement_stopping_reason,
        },
    )
