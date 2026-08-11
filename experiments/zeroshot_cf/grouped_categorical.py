"""Atomic categorical actions for mixed-data counterfactual search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from experiments.zeroshot_cf.data import OneHotActionGroup


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


CategoryDistribution = Callable[
    [np.ndarray, OneHotActionGroup], tuple[np.ndarray, np.ndarray]
]


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

    Every category in TabICL's conditional support is considered.  Before a
    flip, the candidate with maximum target-class probability is committed.
    As soon as any candidate is valid, validity becomes a hard gate and the
    lowest-LOF valid candidate is selected.  A group is edited at most once.
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
        conditional_probabilities: list[float] = []

        for group in groups:
            if group.name in used_groups:
                continue
            group_values = x_cf[list(group.columns)]
            if not np.isclose(group_values.sum(), 1.0):
                raise ValueError(f"one-hot group {group.name!r} is invalid")
            current_category = int(np.argmax(group_values))
            categories, probabilities = category_distribution(x_cf, group)
            categories = np.asarray(categories, dtype=int)
            probabilities = np.asarray(probabilities, dtype=np.float64)
            if categories.ndim != 1 or probabilities.shape != categories.shape:
                raise ValueError(
                    "category_distribution must return aligned 1D arrays"
                )
            probability_by_category = dict(zip(categories.tolist(), probabilities))

            # Enumerating the entire learned support protects coverage. TabICL's
            # probability is retained for diagnostics rather than used as a
            # rejection threshold, because even a low-probability category may
            # be the only valid recourse for a hard factual.
            if any(
                category < 0 or category >= len(group.columns)
                for category in categories
            ):
                raise ValueError(
                    f"TabICL returned a category outside group {group.name!r}"
                )
            for category in range(len(group.columns)):
                if category == current_category:
                    continue
                trial = x_cf.copy()
                trial[list(group.columns)] = 0.0
                trial[group.columns[category]] = 1.0
                trials.append(trial)
                trial_groups.append(group)
                trial_categories.append(category)
                conditional_probabilities.append(
                    float(probability_by_category.get(category, 0.0))
                )

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
                "tabicl_conditional_probability": conditional_probabilities[best],
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
