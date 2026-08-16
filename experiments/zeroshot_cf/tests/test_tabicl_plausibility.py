"""Tests for one-shot TabICL whole-row plausibility scoring."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.zeroshot_cf.tabicl_joint_plausibility import TabICLJointScorer


class _Sampler:
    def score_joint_rows(
        self,
        rows,
        *,
        fixed_target,
        n_permutations,
    ):
        assert fixed_target == 1
        assert n_permutations == 1
        return np.asarray(rows)[:, 0]


def test_joint_scorer_returns_raw_density_and_tracks_one_batch() -> None:
    scorer = TabICLJointScorer(sampler=_Sampler(), target_class=1)

    batch = scorer.score_rows(
        np.array([[0.6, 0.9], [0.9, 0.2]]),
        target_class=1,
    )

    np.testing.assert_allclose(batch.joint_log_density, [0.6, 0.9])
    assert scorer.batch_count == 1
    assert scorer.row_count == 2


def test_joint_scorer_rejects_a_different_target_class() -> None:
    scorer = TabICLJointScorer(sampler=_Sampler(), target_class=1)

    with pytest.raises(ValueError, match="configured for class 1"):
        scorer.score_rows(np.array([[0.6, 0.9]]), target_class=0)
