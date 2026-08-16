"""Tests for validation-calibrated TabICL whole-row plausibility."""

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.tabicl_joint_plausibility import (
    ValidationCalibratedTabICLJointScorer,
    empirical_percentiles,
)


def test_empirical_percentiles_preserve_higher_is_better_order() -> None:
    percentiles = empirical_percentiles(
        np.array([-4.0, -3.0, -2.0, -1.0]),
        np.array([-3.5, -1.5, 0.0]),
    )

    np.testing.assert_allclose(percentiles, [0.25, 0.75, 1.0])


def test_joint_scorer_returns_calibrated_density_without_classifier_margin():
    class Sampler:
        def score_joint_rows(
            self,
            rows,
            *,
            fixed_target,
            n_permutations,
        ):
            del fixed_target, n_permutations
            return np.asarray(rows)[:, 0]

    scorer = ValidationCalibratedTabICLJointScorer(
        sampler=Sampler(),
        target_class=1,
        joint_calibration_scores=np.array([0.0, 0.25, 0.5, 0.75]),
    )

    batch = scorer.score_rows(
        np.array([[0.6, 0.9], [0.9, 0.2]]),
        target_class=1,
    )

    np.testing.assert_allclose(batch.joint_percentile, [0.75, 1.0])
    np.testing.assert_allclose(batch.joint_log_density, [0.6, 0.9])
