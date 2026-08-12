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
    validity_first: bool = False,
    probability_slack: float = 0.0,
    max_rounds: int = 1,
    tau: float = 0.5,
    temperature: float = 1e-9,
    category_distribution: CategoryDistribution | None = None,
) -> tuple[np.ndarray, list[int], dict]:
    """Greedily select the best action across numerical and categorical types.

    At every step, all remaining scalar feature proposals and all legal atomic
    category swaps compete in one discriminator batch. If any candidate is
    valid, validity is a hard gate and the lowest-LOF valid candidate wins.
    Otherwise the candidate with the highest target-class probability wins
    (with optional LOF tie-breaking inside ``probability_slack``).

    Each scalar feature or categorical group may be changed once per round.
    Later rounds revisit all action units, but only strict target-probability
    improvements are committed.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    if probability_slack < 0:
        raise ValueError("probability_slack must be non-negative")
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
    rounds_used = 0
    flipped = False

    def flip_state(row: np.ndarray) -> tuple[bool, float]:
        batch = row.reshape(1, -1)
        probability = float(disc.predict_proba(batch)[0, y_target])
        prediction = int(disc.predict(batch)[0])
        return prediction == y_target and probability >= tau, probability

    flipped, current_probability = flip_state(current)
    best_row = current.copy()
    best_probability = current_probability
    best_history_length = 0

    for round_index in range(max_rounds):
        if flipped:
            break
        rounds_used = round_index + 1
        used_numerical: set[int] = set()
        used_groups: set[str] = set()
        committed_this_round = 0

        while not flipped:
            trial_rows: list[np.ndarray] = []
            metadata: list[dict] = []
            available_numerical = [
                column for column in numerical if column not in used_numerical
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

            for group in groups:
                if group.name in used_groups:
                    continue
                columns = list(group.columns)
                group_values = current[columns]
                if not np.isclose(group_values.sum(), 1.0):
                    raise ValueError(f"one-hot group {group.name!r} is invalid")
                previous_category = int(np.argmax(group_values))
                for category in range(len(columns)):
                    if category == previous_category:
                        continue
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
                        }
                    )

            if not trial_rows:
                break
            trials = np.stack(trial_rows)
            probabilities = np.asarray(disc.predict_proba(trials))[:, y_target]
            predictions = np.asarray(disc.predict(trials))
            valid = (predictions == y_target) & (probabilities >= tau)
            lof_scores = (
                None
                if plausibility_model is None
                else -np.asarray(plausibility_model.score_samples(trials))
            )

            if validity_first and valid.any():
                eligible = np.flatnonzero(valid)
                best = (
                    int(eligible[np.argmax(probabilities[eligible])])
                    if lof_scores is None
                    else int(eligible[np.argmin(lof_scores[eligible])])
                )
            elif validity_first and lof_scores is not None:
                maximum = float(np.max(probabilities))
                eligible = np.flatnonzero(
                    probabilities >= maximum - probability_slack
                )
                best = int(eligible[np.argmin(lof_scores[eligible])])
            else:
                best = int(np.argmax(probabilities))

            selected_probability = float(probabilities[best])
            if round_index > 0 and selected_probability <= current_probability:
                break

            previous = current.copy()
            selected = dict(metadata[best])
            selected.pop("group_object", None)
            selected.update(
                {
                    "round": rounds_used,
                    "target_probability": selected_probability,
                    "lof": (
                        None if lof_scores is None else float(lof_scores[best])
                    ),
                    "immediate_valid": bool(valid[best]),
                    "n_candidates": len(trials),
                    "n_valid_candidates": int(valid.sum()),
                }
            )
            current = trials[best]
            current_probability = selected_probability
            committed_this_round += 1

            raw_selected = metadata[best]
            if raw_selected["action_type"] == "numerical":
                used_numerical.add(int(raw_selected["feature"]))
            else:
                group = raw_selected["group_object"]
                used_groups.add(group.name)
                if category_distribution is not None:
                    categories, conditional_probabilities = category_distribution(
                        previous,
                        group,
                    )
                    conditional = dict(
                        zip(
                            np.asarray(categories, dtype=int).tolist(),
                            np.asarray(
                                conditional_probabilities,
                                dtype=np.float64,
                            ).tolist(),
                        )
                    )
                    selected["tabicl_conditional_probability"] = float(
                        conditional.get(int(raw_selected["to_category"]), 0.0)
                    )
                categorical_history.append(selected.copy())

            history.append(selected)
            flipped, current_probability = flip_state(current)
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
    return current, changed_order, {
        "flipped": bool(flipped),
        "steps": len(history),
        "rounds": rounds_used,
        "history": history,
        "attempt_history": attempt_history,
        "selection_history": history,
        "attempt_selection_history": attempt_history,
        "categorical_history": categorical_history,
        "round_history": [],
        "best_target_probability": float(best_probability),
    }


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

    def flip_state(row: np.ndarray) -> tuple[bool, float]:
        batch = row.reshape(1, -1)
        probability = float(disc.predict_proba(batch)[0, y_target])
        prediction = int(disc.predict(batch)[0])
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
        target_probabilities = np.asarray(disc.predict_proba(trial_matrix))[
            :, y_target
        ]
        predictions = np.asarray(disc.predict(trial_matrix))
        valid = (predictions == y_target) & (target_probabilities >= tau)
        lof_scores = (
            None
            if plausibility_model is None
            else -np.asarray(plausibility_model.score_samples(trial_matrix))
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
        if (
            categories.ndim != 1
            or conditional_probabilities.shape != categories.shape
        ):
            raise ValueError(
                "category_distribution must return aligned 1D arrays"
            )
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
        previous_category = int(
            np.argmax(x_cf[list(selected_group.columns)])
        )
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

    return x_cf, changed_columns, {
        "flipped": bool(flipped),
        "steps": len(history),
        "history": history,
        "final_target_probability": float(current_probability),
    }
