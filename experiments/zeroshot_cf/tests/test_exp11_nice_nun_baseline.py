#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the NICE-style Exp11 local baseline."""

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.baseline_common import (
    ActionUnit,
    build_action_units,
)
from experiments.zeroshot_cf.exp11_nice_nun_baseline import (
    greedy_nice_counterfactual,
    nearest_unlike_prototypes,
)


class _ToyClassifier:
    """Small deterministic classifier with an sklearn-like interface."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X)
        target_probability = 0.1 + 0.45 * matrix[:, 0] + 0.6 * matrix[:, 2]
        target_probability = np.clip(target_probability, 0.0, 1.0)
        return np.column_stack([1.0 - target_probability, target_probability])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def test_nearest_unlike_prototypes_use_classifier_target_pool() -> None:
    """Every selected prototype comes from the requested prediction class."""
    X_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.8, 0.9],
            [0.1, 0.1],
        ]
    )
    train_predictions = np.array([0, 1, 1, 0])
    X_test = np.array([[0.7, 0.8], [0.2, 0.2]])
    targets = np.array([1, 0])

    prototypes, indices, distances = nearest_unlike_prototypes(
        X_train,
        train_predictions,
        X_test,
        targets,
    )

    np.testing.assert_array_equal(indices, [2, 3])
    np.testing.assert_allclose(prototypes, X_train[indices])
    assert np.all(distances >= 0.0)


def test_greedy_nice_copies_group_atomically_and_preserves_immutable() -> None:
    """A grouped action changes together and excluded columns remain factual."""
    factual = np.array([0.0, 1.0, 0.0, 5.0])
    prototype = np.array([1.0, 0.0, 1.0, 99.0])
    actions = [
        ActionUnit("scalar", (0,)),
        ActionUnit("category", (1, 2)),
    ]

    counterfactual, info = greedy_nice_counterfactual(
        _ToyClassifier(),
        factual,
        prototype,
        target=1,
        action_units=actions,
    )

    np.testing.assert_array_equal(counterfactual[1:3], [0.0, 1.0])
    assert counterfactual[3] == factual[3]
    assert info["valid"] is True
    assert info["changed_columns"] == 2
    assert info["selected_units"] == ["category"]


def test_greedy_nice_requires_tau_even_after_target_prediction_flips() -> None:
    class _ThresholdTrapClassifier:
        classes_ = np.array([0, 1])

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            matrix = np.asarray(X)
            target_probability = 0.1 + 0.3 * matrix[:, 0] + 0.45 * matrix[:, 2]
            return np.column_stack([1.0 - target_probability, target_probability])

        def predict(self, X: np.ndarray) -> np.ndarray:
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    factual = np.array([0.0, 1.0, 0.0, 5.0])
    prototype = np.array([1.0, 0.0, 1.0, 5.0])
    actions = [
        ActionUnit("scalar", (0,)),
        ActionUnit("category", (1, 2)),
    ]

    counterfactual, info = greedy_nice_counterfactual(
        _ThresholdTrapClassifier(),
        factual,
        prototype,
        target=1,
        action_units=actions,
        tau=0.9,
    )

    np.testing.assert_array_equal(counterfactual[1:3], [0.0, 1.0])
    assert _ThresholdTrapClassifier().predict(counterfactual.reshape(1, -1))[0] == 1
    assert info["target_probability"] < 0.9
    assert info["valid"] is False


def test_build_action_units_keeps_scalar_and_group_boundaries() -> None:
    actions = build_action_units([0, 3], [])

    assert actions == [
        ActionUnit("feature_0", (0,)),
        ActionUnit("feature_3", (3,)),
    ]
