#  Copyright (c) Prior Labs GmbH 2026.

"""Focused tests for the density-weighted FACE-kNN baseline."""

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.exp14_face_baseline import (
    build_face_knn_graph,
    face_counterfactual,
)


class ThresholdClassifier:
    """Small deterministic classifier used to test graph search."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities determined by the first feature."""
        probability = np.clip(np.asarray(X)[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - probability, probability))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify rows at the fixed 0.5 threshold."""
        return (np.asarray(X)[:, 0] >= 0.5).astype(int)


def test_face_finds_training_supported_target_and_preserves_immutable() -> None:
    """FACE must reach a target row without altering immutable columns."""
    X_train = np.array(
        [
            [0.0, 0.0, 0.1],
            [0.2, 0.0, 0.2],
            [0.6, 1.0, 0.3],
            [0.9, 1.0, 0.4],
        ]
    )
    factual = np.array([0.1, 1.0, 0.77])
    graph = build_face_knn_graph(X_train, (0, 1), n_neighbors=2)

    counterfactual, info = face_counterfactual(
        graph,
        ThresholdClassifier(),
        factual,
        target=1,
    )

    assert info["valid"] is True
    assert counterfactual[2] == factual[2]
    assert any(
        np.array_equal(counterfactual[:2], training_row[:2]) for training_row in X_train
    )
    assert ThresholdClassifier().predict(counterfactual[None])[0] == 1


def test_face_rejects_empty_action_space() -> None:
    """A graph cannot be constructed when no intervention is permitted."""
    with np.testing.assert_raises_regex(ValueError, "actionable"):
        build_face_knn_graph(np.array([[0.0], [1.0]]), ())
