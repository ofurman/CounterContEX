"""Tests for quality-constrained diverse counterfactual selection."""

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.diverse_counterfactuals import (
    action_unit_signatures,
    pairwise_action_distances,
    select_diverse_counterfactuals,
    summarize_counterfactual_set,
)


def test_selection_keeps_primary_then_prefers_distinct_action_sets() -> None:
    factual = np.zeros(3)
    rows = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    selected = select_diverse_counterfactuals(
        rows,
        np.asarray([10.0, 9.9, 9.8, 9.7, 9.6]),
        factual,
        numerical_columns=[0, 1, 2],
        categorical_groups=[],
        primary_index=0,
        max_outputs=3,
    )

    np.testing.assert_array_equal(selected, [0, 2, 3])


def test_selection_deduplicates_rows_without_losing_primary() -> None:
    factual = np.zeros(2)
    rows = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    selected = select_diverse_counterfactuals(
        rows,
        np.asarray([0.8, 0.9, 0.7]),
        factual,
        numerical_columns=[0, 1],
        categorical_groups=[],
        primary_index=0,
        max_outputs=3,
    )

    np.testing.assert_array_equal(selected, [0, 2])


def test_mixed_action_distances_treat_one_hot_group_as_one_unit() -> None:
    group = OneHotActionGroup("job", (1, 2, 3))
    factual = np.asarray([0.0, 1.0, 0.0, 0.0])
    rows = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )

    signatures = action_unit_signatures(rows, factual, [0], [group])
    np.testing.assert_array_equal(
        signatures,
        [[True, False], [False, True], [True, True]],
    )
    action_set_distance, value_distance = pairwise_action_distances(
        rows,
        factual,
        [0],
        [group],
    )
    assert action_set_distance[0, 1] == 1.0
    assert action_set_distance[0, 2] == 0.5
    assert value_distance[1, 2] == 1.0

    summary = summarize_counterfactual_set(rows, factual, [0], [group])
    assert summary.distinct_action_sets == 3
    assert summary.minimum_action_set_jaccard == 0.5
