"""Tests for grouped mixed-data costs."""

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.mixed_distance import (
    action_unit_change_count,
    compact_gower_distance,
    grouped_gower_distance,
)


def test_compact_gower_treats_category_identifiers_as_nominal() -> None:
    rows = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 100.0],
            [0.6, 0.6, 0.0],
        ]
    )

    np.testing.assert_allclose(
        compact_gower_distance(rows, np.zeros(3), categorical_columns=[2]),
        [1 / 3, 1 / 3, 0.4],
    )


def test_grouped_gower_counts_a_category_once_regardless_of_one_hot_width() -> None:
    narrow = OneHotActionGroup("narrow", (1, 2))
    wide = OneHotActionGroup("wide", (1, 2, 3, 4, 5))

    narrow_distance = grouped_gower_distance(
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.0, 1.0, 0.0]),
        numerical_columns=[0],
        categorical_groups=[narrow],
    )
    wide_distance = grouped_gower_distance(
        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        numerical_columns=[0],
        categorical_groups=[wide],
    )

    np.testing.assert_allclose(narrow_distance, [0.5])
    np.testing.assert_allclose(wide_distance, [0.5])


def test_grouped_costs_support_one_reference_per_row() -> None:
    group = OneHotActionGroup("kind", (1, 2, 3))
    factuals = np.array(
        [[0.1, 1.0, 0.0, 0.0], [0.8, 0.0, 1.0, 0.0]]
    )
    rows = np.array(
        [[0.3, 1.0, 0.0, 0.0], [0.8, 0.0, 0.0, 1.0]]
    )

    np.testing.assert_allclose(
        grouped_gower_distance(rows, factuals, [0], [group]),
        [0.1, 0.5],
    )
    np.testing.assert_array_equal(
        action_unit_change_count(
            rows,
            factuals,
            [0],
            [group],
            numerical_tolerance=0.05,
        ),
        [1, 1],
    )
