"""Independent witnesses for evaluation-v2 plausibility metrics."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.zeroshot_cf.evaluation.metrics import (
    detectability_auc,
    kth_grouped_gower_distance,
)


def test_detectability_auc_distinguishes_copied_and_shifted_counterfactuals() -> None:
    rng = np.random.default_rng(42)
    real = rng.normal(0.0, 0.2, size=(80, 3))
    targets = np.repeat([0, 1], 40)

    copied = detectability_auc(
        reference=real,
        reference_labels=targets,
        counterfactuals=real.copy(),
        counterfactual_targets=targets,
        minimum_cf_rows=20,
    )
    shifted = detectability_auc(
        reference=real,
        reference_labels=targets,
        counterfactuals=real + 5.0,
        counterfactual_targets=targets,
        minimum_cf_rows=20,
    )

    assert copied.status == "MEASURED"
    assert copied.auc == pytest.approx(0.5, abs=0.08)
    assert copied.auc is not None and 0.5 <= copied.auc <= 1.0
    assert copied.n_reference == copied.n_counterfactual == 80
    assert shifted.status == "MEASURED"
    assert shifted.auc is not None and shifted.auc >= 0.99


def test_detectability_auc_empty_arm_is_not_measured() -> None:
    measured = detectability_auc(
        reference=np.zeros((40, 2)),
        reference_labels=np.repeat([0, 1], 20),
        counterfactuals=np.empty((0, 2)),
        counterfactual_targets=np.empty(0, dtype=int),
        minimum_cf_rows=20,
    )

    assert measured.status == "NOT_MEASURED"
    assert measured.auc is None
    assert measured.n_counterfactual == 0


def test_kth_grouped_gower_distance_uses_original_feature_units() -> None:
    reference = np.array([[0.0], [0.2], [0.4], [0.8]])
    candidates = np.array([[0.1], [1.0]])

    distances = kth_grouped_gower_distance(
        candidates,
        reference,
        numerical=(0,),
        categorical_groups=(),
        k=2,
    )

    np.testing.assert_allclose(distances, [0.1, 0.6])
