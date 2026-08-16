"""Tests for validation-calibrated TabICL whole-row plausibility."""

from __future__ import annotations

import numpy as np

from experiments.zeroshot_cf.tabicl_joint_plausibility import (
    ValidationCalibratedTabICLJointScorer,
    empirical_percentiles,
)
from experiments.zeroshot_cf.tabicl_row_plausibility import one_vs_rest_logit


def test_empirical_percentiles_preserve_higher_is_better_order() -> None:
    percentiles = empirical_percentiles(
        np.array([-4.0, -3.0, -2.0, -1.0]),
        np.array([-3.5, -1.5, 0.0]),
    )

    np.testing.assert_allclose(percentiles, [0.25, 0.75, 1.0])


def test_one_vs_rest_logit_matches_binary_log_odds() -> None:
    probabilities = np.array([[0.8, 0.2], [0.25, 0.75]])

    scores = one_vs_rest_logit(probabilities, np.array([0, 1]))

    np.testing.assert_allclose(scores, np.log([4.0, 3.0]))


def test_joint_scorer_combines_density_and_classifier_by_minimum_percentile():
    class Sampler:
        def score_joint_rows(
            self,
            rows,
            *,
            fixed_target,
            fixed_confidence,
            n_permutations,
        ):
            del fixed_target, fixed_confidence, n_permutations
            return np.asarray(rows)[:, 0]

    class MarginScorer:
        def percentile_rows(self, rows, target_class):
            del target_class
            return np.asarray(rows)[:, 1]

    scorer = ValidationCalibratedTabICLJointScorer(
        sampler=Sampler(),
        target_class=1,
        joint_calibration_scores=np.array([0.0, 0.25, 0.5, 0.75]),
        threshold_quantile=0.10,
        classifier_margin_scorer=MarginScorer(),
    )

    batch = scorer.score_rows(
        np.array([[0.6, 0.9], [0.9, 0.2]]),
        target_class=1,
        target_probabilities=np.array([0.7, 0.8]),
    )

    np.testing.assert_allclose(batch.joint_percentile, [0.75, 1.0])
    np.testing.assert_allclose(batch.combined_percentile, [0.75, 0.2])
