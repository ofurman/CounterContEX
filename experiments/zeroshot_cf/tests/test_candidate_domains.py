"""Tests for dependency-light candidate-domain projection helpers."""

from __future__ import annotations

import numpy as np

from experiments.zeroshot_cf.candidate_domains import (
    infer_feature_domains,
    project_candidate_values,
)


def test_infer_feature_domains_tracks_bounds_and_small_supports() -> None:
    X_train = np.array(
        [
            [0.0, 0.0, 0.10],
            [0.5, 1.0, 0.90],
            [1.0, 0.0, 0.60],
            [0.5, 1.0, 0.40],
        ],
        dtype=np.float64,
    )

    lower, upper, supports = infer_feature_domains(X_train, max_discrete_values=2)

    np.testing.assert_allclose(lower, [0.0, 0.0, 0.10])
    np.testing.assert_allclose(upper, [1.0, 1.0, 0.90])
    np.testing.assert_array_equal(supports[1], [0.0, 1.0])
    assert 0 not in supports
    assert 2 not in supports


def test_projection_clips_ranges_and_snaps_to_empirical_support() -> None:
    X_train = np.array(
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        dtype=np.float64,
    )
    domains = infer_feature_domains(X_train, max_discrete_values=3)

    projected = project_candidate_values([0, 1], np.array([-2.0, 0.61]), domains)

    np.testing.assert_allclose(projected, [0.0, 0.5])


def test_projection_preserves_integer_and_batch_candidate_semantics() -> None:
    X_train = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 5.0],
            [1.0, 4.0, 10.0],
        ],
        dtype=np.float64,
    )
    domains = infer_feature_domains(X_train, max_discrete_values=4)

    projected = project_candidate_values(
        [1, 1, 2],
        np.array([1.6, 3.6, 12.0]),
        domains,
    )

    np.testing.assert_allclose(projected, [2.0, 4.0, 10.0])
