#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for mixed-data optimization counterfactual baselines."""

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.exp11_nice_nun_baseline import ActionUnit
from experiments.zeroshot_cf.exp12_optimization_baselines import (
    growing_spheres_counterfactual,
    prune_counterfactual_actions,
    wachter_coordinate_counterfactual,
)


class _ToyClassifier:
    def predict_proba(self, X):
        matrix = np.asarray(X)
        probability = np.clip(0.1 + 0.7 * matrix[:, 0] + 0.5 * matrix[:, 2], 0, 1)
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _space():
    group = OneHotActionGroup("job", (1, 2))
    actions = [ActionUnit("scalar", (0,)), ActionUnit("job", (1, 2))]
    return group, actions


def test_pruning_removes_redundant_actions_and_preserves_one_hot_group() -> None:
    group, actions = _space()
    factual = np.array([0.0, 1.0, 0.0, 7.0])
    candidate = np.array([1.0, 0.0, 1.0, 7.0])

    pruned = prune_counterfactual_actions(
        _ToyClassifier(), factual, candidate, 1, actions
    )

    assert _ToyClassifier().predict(pruned.reshape(1, -1))[0] == 1
    assert np.isclose(pruned[list(group.columns)].sum(), 1.0)
    assert pruned[3] == factual[3]
    assert np.count_nonzero(pruned != factual) < np.count_nonzero(candidate != factual)


def test_wachter_coordinate_search_finds_atomic_valid_counterfactual() -> None:
    group, actions = _space()
    factual = np.array([0.0, 1.0, 0.0, 7.0])

    counterfactual, info = wachter_coordinate_counterfactual(
        _ToyClassifier(),
        factual,
        1,
        {0: np.array([0.25, 0.75, 1.0])},
        [group],
        actions,
    )

    assert info["valid"] is True
    assert np.isclose(counterfactual[list(group.columns)].sum(), 1.0)
    assert counterfactual[3] == factual[3]


def test_growing_spheres_is_reproducible_valid_and_actionable() -> None:
    group, actions = _space()
    factual = np.array([0.0, 1.0, 0.0, 7.0])
    kwargs = dict(
        disc_model=_ToyClassifier(),
        factual=factual,
        target=1,
        scalar_columns=[0],
        categorical_groups=[group],
        action_units=actions,
        n_candidates=128,
        max_shells=10,
        random_state=11,
    )

    first, first_info = growing_spheres_counterfactual(**kwargs)
    second, second_info = growing_spheres_counterfactual(**kwargs)

    np.testing.assert_allclose(first, second)
    assert first_info["valid"] is True
    assert second_info["valid"] is True
    assert np.isclose(first[list(group.columns)].sum(), 1.0)
    assert first[3] == factual[3]
