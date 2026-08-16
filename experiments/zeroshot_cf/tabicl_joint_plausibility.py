"""Whole-row plausibility from TabICL's built-in joint-density estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def empirical_percentiles(
    calibration_scores: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Map scores to an empirical validation CDF where higher is better."""
    calibration = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    values = np.asarray(scores, dtype=np.float64)
    if calibration.ndim != 1 or len(calibration) == 0:
        raise ValueError("calibration_scores must be a non-empty 1D array")
    if values.ndim != 1:
        raise ValueError("scores must be a 1D array")
    if not np.all(np.isfinite(calibration)) or not np.all(np.isfinite(values)):
        raise ValueError("plausibility scores must be finite")
    return np.searchsorted(calibration, values, side="right") / len(calibration)


@dataclass(frozen=True)
class TabICLJointScoreBatch:
    """Components of one whole-row TabICL plausibility evaluation."""

    combined_percentile: np.ndarray
    joint_log_density: np.ndarray
    joint_percentile: np.ndarray
    classifier_margin_percentile: np.ndarray | None


@dataclass
class ValidationCalibratedTabICLJointScorer:
    """Calibrate TabICL joint density on local target-class validation rows.

    The primary signal is the chain-rule log-density produced by
    ``TabICLUnsupervised`` for complete augmented rows ``[X, Y, confidence]``.
    An optional TabICL classifier-margin percentile provides the additional
    TabPFNGEN-inspired signal. When present, the combined score is the lower of
    the two validation percentiles, so a row is only as plausible as its weaker
    TabICL assessment.
    """

    sampler: Any
    target_class: int
    joint_calibration_scores: np.ndarray
    threshold_quantile: float
    n_permutations: int = 1
    classifier_margin_scorer: Any | None = None

    @classmethod
    def calibrate(
        cls,
        sampler: Any,
        validation_rows: np.ndarray,
        validation_target_probabilities: np.ndarray,
        *,
        target_class: int,
        threshold_quantile: float = 0.10,
        n_permutations: int = 1,
        classifier_margin_scorer: Any | None = None,
    ) -> "ValidationCalibratedTabICLJointScorer":
        """Evaluate the target-class validation reference distribution once."""
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        if n_permutations < 1:
            raise ValueError("n_permutations must be positive")
        rows = np.asarray(validation_rows)
        probabilities = np.asarray(validation_target_probabilities, dtype=np.float64)
        if rows.ndim != 2 or len(rows) == 0:
            raise ValueError("validation_rows must be a non-empty 2D matrix")
        if probabilities.shape != (len(rows),):
            raise ValueError(
                "validation_target_probabilities must contain one value per row"
            )
        joint_scores = sampler.score_joint_rows(
            rows,
            fixed_target=target_class,
            fixed_confidence=probabilities,
            n_permutations=n_permutations,
        )
        return cls(
            sampler=sampler,
            target_class=int(target_class),
            joint_calibration_scores=np.sort(
                np.asarray(joint_scores, dtype=np.float64)
            ),
            threshold_quantile=threshold_quantile,
            n_permutations=n_permutations,
            classifier_margin_scorer=classifier_margin_scorer,
        )

    @property
    def threshold(self) -> float:
        """Return the shared validation-percentile acceptance threshold."""
        return self.threshold_quantile

    @property
    def calibration_count(self) -> int:
        return len(self.joint_calibration_scores)

    def score_rows(
        self,
        rows: np.ndarray,
        target_class: int,
        target_probabilities: np.ndarray,
    ) -> TabICLJointScoreBatch:
        """Score complete candidate rows and return calibrated components."""
        if int(target_class) != self.target_class:
            raise ValueError(
                f"scorer is calibrated for class {self.target_class}, "
                f"not {target_class}"
            )
        matrix = np.atleast_2d(np.asarray(rows))
        probabilities = np.asarray(target_probabilities, dtype=np.float64)
        if probabilities.shape != (len(matrix),):
            raise ValueError("target_probabilities must contain one value per row")
        joint_log_density = self.sampler.score_joint_rows(
            matrix,
            fixed_target=target_class,
            fixed_confidence=probabilities,
            n_permutations=self.n_permutations,
        )
        joint_percentile = empirical_percentiles(
            self.joint_calibration_scores,
            joint_log_density,
        )
        margin_percentile = None
        combined = joint_percentile
        if self.classifier_margin_scorer is not None:
            margin_percentile = self.classifier_margin_scorer.percentile_rows(
                matrix,
                target_class,
            )
            combined = np.minimum(joint_percentile, margin_percentile)
        return TabICLJointScoreBatch(
            combined_percentile=np.asarray(combined, dtype=np.float64),
            joint_log_density=np.asarray(joint_log_density, dtype=np.float64),
            joint_percentile=np.asarray(joint_percentile, dtype=np.float64),
            classifier_margin_percentile=(
                None
                if margin_percentile is None
                else np.asarray(margin_percentile, dtype=np.float64)
            ),
        )
