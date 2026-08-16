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

    joint_log_density: np.ndarray
    joint_percentile: np.ndarray


@dataclass
class ValidationCalibratedTabICLJointScorer:
    """Calibrate TabICL ``[X, Y]`` joint density on validation rows.

    The scorer deliberately excludes the explained classifier's confidence.
    It therefore measures complete-row density under TabICL rather than mixing
    density with a second classifier-agreement signal.
    """

    sampler: Any
    target_class: int
    joint_calibration_scores: np.ndarray
    n_permutations: int = 1

    @classmethod
    def calibrate(
        cls,
        sampler: Any,
        validation_rows: np.ndarray,
        *,
        target_class: int,
        n_permutations: int = 1,
    ) -> "ValidationCalibratedTabICLJointScorer":
        """Evaluate the target-class validation reference distribution once."""
        if n_permutations < 1:
            raise ValueError("n_permutations must be positive")
        rows = np.asarray(validation_rows)
        if rows.ndim != 2 or len(rows) == 0:
            raise ValueError("validation_rows must be a non-empty 2D matrix")
        joint_scores = sampler.score_joint_rows(
            rows,
            fixed_target=target_class,
            n_permutations=n_permutations,
        )
        return cls(
            sampler=sampler,
            target_class=int(target_class),
            joint_calibration_scores=np.sort(
                np.asarray(joint_scores, dtype=np.float64)
            ),
            n_permutations=n_permutations,
        )

    @property
    def calibration_count(self) -> int:
        return len(self.joint_calibration_scores)

    def score_rows(
        self,
        rows: np.ndarray,
        target_class: int,
    ) -> TabICLJointScoreBatch:
        """Score complete candidate rows and return calibrated components."""
        if int(target_class) != self.target_class:
            raise ValueError(
                f"scorer is calibrated for class {self.target_class}, "
                f"not {target_class}"
            )
        matrix = np.atleast_2d(np.asarray(rows))
        joint_log_density = self.sampler.score_joint_rows(
            matrix,
            fixed_target=target_class,
            n_permutations=self.n_permutations,
        )
        joint_percentile = empirical_percentiles(
            self.joint_calibration_scores,
            joint_log_density,
        )
        return TabICLJointScoreBatch(
            joint_log_density=np.asarray(joint_log_density, dtype=np.float64),
            joint_percentile=np.asarray(joint_percentile, dtype=np.float64),
        )
