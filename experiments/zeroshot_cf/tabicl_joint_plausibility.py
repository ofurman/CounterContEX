"""One-shot whole-row plausibility scoring with TabICL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TabICLJointScoreBatch:
    """Raw complete-row TabICL log densities from one scoring batch."""

    joint_log_density: np.ndarray


@dataclass
class TabICLJointScorer:
    """Compare complete rows under one factual-specific ``[X, Y]`` context.

    Raw log densities are comparable within a factual because every candidate
    uses the same context, target class, feature order, and dimensionality.
    No per-factual validation calibration is required for this relative rank.
    """

    sampler: Any
    target_class: int
    n_permutations: int = 1
    batch_count: int = 0
    row_count: int = 0

    def score_rows(
        self,
        rows: np.ndarray,
        target_class: int,
    ) -> TabICLJointScoreBatch:
        """Score one complete-row batch and record its computational size."""
        if int(target_class) != self.target_class:
            raise ValueError(
                f"scorer is configured for class {self.target_class}, "
                f"not {target_class}"
            )
        matrix = np.atleast_2d(np.asarray(rows))
        if matrix.ndim != 2 or len(matrix) == 0:
            raise ValueError("rows must be a non-empty 2D matrix")
        joint_log_density = np.asarray(
            self.sampler.score_joint_rows(
                matrix,
                fixed_target=target_class,
                n_permutations=self.n_permutations,
            ),
            dtype=np.float64,
        )
        if joint_log_density.shape != (len(matrix),):
            raise ValueError(
                "score_joint_rows returned an unexpected shape; "
                f"expected {(len(matrix),)}, got {joint_log_density.shape}"
            )
        if not np.all(np.isfinite(joint_log_density)):
            raise ValueError("TabICL joint scores must be finite")
        self.batch_count += 1
        self.row_count += len(matrix)
        return TabICLJointScoreBatch(joint_log_density=joint_log_density)
