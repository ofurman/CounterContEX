"""Atomic categorical actions for mixed-data counterfactual search."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.diverse_counterfactuals import (
    select_diverse_counterfactuals,
)
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

    def score_joint_rows(self, X_rows: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate complete original-space rows in the compact TabICL space."""
        return self.sampler.score_joint_rows(self.codec.encode(X_rows), **kwargs)


CategoryDistribution = Callable[
    [np.ndarray, OneHotActionGroup], tuple[np.ndarray, np.ndarray]
]
ConditionedCategoryDistribution = Callable[
    [np.ndarray, OneHotActionGroup, float | None], tuple[np.ndarray, np.ndarray]
]


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
    cf_mode: str = "sparse",
    plausibility_model=None,
    tabicl_joint_plausibility=None,
    max_validity_steps: int | None = None,
    allow_revisits: bool = True,
    joint_shortlist_size: int = 16,
    max_extra_actions: int = 1,
    min_joint_log_gain: float = 0.0,
    n_counterfactuals: int = 1,
    max_refinement_steps: int = 2,
    min_relative_lof_gain: float = 0.05,
    refinement_lof_threshold: float | None = None,
    tau: float = 0.5,
    temperature: float = 1e-9,
    category_distribution: ConditionedCategoryDistribution | None = None,
    categorical_proposal_count: int = 1,
) -> tuple[np.ndarray, list[int], dict]:
    """Greedily select the best action across numerical and categorical types.

    At every validity-search step, all scalar and categorical TabICL proposals
    compete in one discriminator batch. Invalid states are ranked only by
    target-probability ascent. The first valid state is selected by action-unit
    sparsity and factual proximity, then committed without any joint-density
    call. ``sparse`` mode returns it immediately. ``data_plausible`` mode then
    performs one validity-preserving refinement attempt. It cheaply keeps at
    most one representative per action unit, fills a bounded shortlist, and
    scores the sparse incumbent plus that shortlist in one whole-row TabICL
    batch. The incumbent remains the fallback unless raw joint log density
    improves. When multiple counterfactuals are requested, the same scored
    batch supplies a quality-constrained pool. The existing single-CFE winner
    remains first, and later rows are selected by changed-action-set diversity.

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
    if n_counterfactuals < 1:
        raise ValueError("n_counterfactuals must be at least 1")
    if n_counterfactuals > 1 and cf_mode != "data_plausible":
        raise ValueError("multiple counterfactuals require data_plausible mode")
    if cf_mode == "data_plausible" and tabicl_joint_plausibility is None:
        raise ValueError("data_plausible mode requires a TabICL joint scorer")
    if cf_mode == "sparse" and tabicl_joint_plausibility is not None:
        raise ValueError("sparse mode must not receive a TabICL joint scorer")
    if max_refinement_steps < 0:
        raise ValueError("max_refinement_steps must be non-negative")
    if not 0.0 <= min_relative_lof_gain < 1.0:
        raise ValueError("min_relative_lof_gain must be in [0, 1)")
    if refinement_lof_threshold is not None and refinement_lof_threshold <= 0:
        raise ValueError("refinement_lof_threshold must be positive")
    if plausibility_model is not None and cf_mode != "sparse":
        raise ValueError("legacy LOF refinement cannot be combined with cf_mode")
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
        """Select a small, action-diverse batch before whole-row scoring."""
        if not len(eligible):
            return np.empty(0, dtype=int)
        candidate_sparsity, candidate_proximity = counterfactual_costs(rows)
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
                    candidate_proximity[eligible],
                    support_penalty[eligible],
                    candidate_sparsity[eligible],
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
        lof_scores: np.ndarray | None,
        rows: np.ndarray,
        candidate_metadata: Sequence[dict],
    ) -> int:
        """Choose a sparse valid candidate without querying joint TabICL."""
        if lof_scores is not None:
            return int(eligible[np.argmin(lof_scores[eligible])])
        candidate_sparsity, candidate_proximity = counterfactual_costs(rows)
        proposal_support = proposal_supports(candidate_metadata)
        support_penalty = -np.log(np.clip(proposal_support, 1e-12, 1.0))
        ranked = np.lexsort(
            (
                -probabilities[eligible],
                support_penalty[eligible],
                candidate_proximity[eligible],
                candidate_sparsity[eligible],
            )
        )
        return int(eligible[ranked[0]])

    flipped, current_probability = flip_state(current)
    current_lof = (
        None
        if plausibility_model is None or not flipped
        else float(-plausibility_model.score_samples(current.reshape(1, -1))[0])
    )
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
    post_valid_refinement = (
        plausibility_model is not None or tabicl_joint_plausibility is not None
    )
    refinement_stopping_reason = "not_started"
    diverse_counterfactuals: np.ndarray | None = None
    diverse_joint_log_densities: np.ndarray | None = None
    diverse_target_probabilities: np.ndarray | None = None
    joint_scoring_runtime_s = 0.0
    diversity_selection_runtime_s = 0.0

    def plausibility_threshold_reached() -> bool:
        return bool(
            plausibility_model is not None
            and refinement_lof_threshold is not None
            and current_lof is not None
            and current_lof <= refinement_lof_threshold
        )

    round_limit = max_validity_steps if allow_revisits else 1
    for round_index in range(round_limit):
        refinement_limit = (
            1 if tabicl_joint_plausibility is not None else max_refinement_steps
        )
        if flipped and (
            not post_valid_refinement
            or refinement_steps >= refinement_limit
            or joint_rerank_attempted
            or plausibility_threshold_reached()
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
                or plausibility_threshold_reached()
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
            lof_scores = valid_lof_scores(trials, valid)
            tabicl_joint_scores = None

            if flipped:
                unvisited = np.asarray(
                    [row.tobytes() not in visited_rows for row in trials]
                )
                if tabicl_joint_plausibility is not None:
                    joint_rerank_attempted = True
                    if initial_sparse_action_count is None:
                        raise RuntimeError("initial sparse action count is unavailable")
                    candidate_sparsity, candidate_proximity = counterfactual_costs(
                        trials
                    )
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
                    current_joint_log_density = float(
                        joint_batch.joint_log_density[0]
                    )
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
                    if len(eligible):
                        ranked = np.lexsort(
                            (
                                candidate_proximity[eligible],
                                candidate_sparsity[eligible],
                                -tabicl_joint_scores["joint_log_density"][eligible],
                            )
                        )
                        best = int(eligible[ranked[0]])
                    if n_counterfactuals > 1:
                        quality_preserving = (
                            joint_batch.joint_log_density[1:]
                            >= current_joint_log_density + min_joint_log_gain
                        )
                        pool_scoring_indices = np.concatenate(
                            (
                                np.asarray([0], dtype=int),
                                np.flatnonzero(quality_preserving) + 1,
                            )
                        )
                        best_scoring_index = (
                            0
                            if not len(eligible)
                            else int(np.flatnonzero(shortlist == best)[0]) + 1
                        )
                        primary_pool_index = int(
                            np.flatnonzero(
                                pool_scoring_indices == best_scoring_index
                            )[0]
                        )
                        pool_rows = scoring_rows[pool_scoring_indices]
                        pool_joint_scores = joint_batch.joint_log_density[
                            pool_scoring_indices
                        ]
                        pool_target_probabilities = np.concatenate(
                            (
                                np.asarray([current_probability], dtype=np.float64),
                                probabilities[shortlist],
                            )
                        )[pool_scoring_indices]
                        diversity_started = perf_counter()
                        selected_pool_indices = select_diverse_counterfactuals(
                            pool_rows,
                            pool_joint_scores,
                            factual,
                            numerical,
                            groups,
                            primary_index=primary_pool_index,
                            max_outputs=n_counterfactuals,
                        )
                        diversity_selection_runtime_s += (
                            perf_counter() - diversity_started
                        )
                        diverse_counterfactuals = pool_rows[selected_pool_indices]
                        diverse_joint_log_densities = pool_joint_scores[
                            selected_pool_indices
                        ]
                        diverse_target_probabilities = pool_target_probabilities[
                            selected_pool_indices
                        ]
                    if not len(eligible):
                        refinement_stopping_reason = "no_improving_candidate"
                        break
                    refinement_stopping_reason = "one_shot_accepted"
                else:
                    if lof_scores is None or current_lof is None:
                        break
                    relative_lof_gain = (current_lof - lof_scores) / max(
                        abs(current_lof), 1e-12
                    )
                    eligible = np.flatnonzero(
                        valid
                        & (relative_lof_gain >= min_relative_lof_gain)
                        & unvisited
                    )
                    if not len(eligible):
                        break
                    candidate_sparsity, candidate_proximity = counterfactual_costs(
                        trials
                    )
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
                    trials,
                    metadata,
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
                tabicl_joint_scores = None
                if valid.any():
                    eligible = np.flatnonzero(valid)
                    best = select_valid_candidate(
                        eligible,
                        probabilities,
                        lof_scores,
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
                    "lof": (None if lof_scores is None else float(lof_scores[best])),
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
                initial_sparse_action_count = int(
                    counterfactual_costs(current)[0][0]
                )
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
        refinement_stopping_reason = (
            "legacy_lof_complete" if plausibility_model is not None else "sparse_valid"
        )
    elif refinement_stopping_reason == "not_started":
        refinement_stopping_reason = "one_shot_not_attempted"
    joint_log_density_gain = (
        None
        if initial_joint_log_density is None or current_joint_log_density is None
        else current_joint_log_density - initial_joint_log_density
    )
    if diverse_counterfactuals is None:
        diverse_counterfactuals = current.reshape(1, -1).copy()
        diverse_joint_log_densities = np.asarray(
            [
                np.nan
                if current_joint_log_density is None
                else current_joint_log_density
            ],
            dtype=np.float64,
        )
        diverse_target_probabilities = np.asarray(
            [current_probability],
            dtype=np.float64,
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
            "n_counterfactuals_requested": n_counterfactuals,
            "diverse_counterfactuals": diverse_counterfactuals,
            "diverse_joint_log_densities": diverse_joint_log_densities,
            "diverse_target_probabilities": diverse_target_probabilities,
            "joint_scoring_runtime_s": joint_scoring_runtime_s,
            "diversity_selection_runtime_s": diversity_selection_runtime_s,
            "extra_actions": (
                0
                if initial_sparse_action_count is None
                else max(0, final_action_count - initial_sparse_action_count)
            ),
            "refinement_stopping_reason": refinement_stopping_reason,
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
