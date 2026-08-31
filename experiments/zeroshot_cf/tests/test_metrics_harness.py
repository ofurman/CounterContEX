"""Regression test for the validity reference fix (post-review P1).

The bug: validity was scored as (y_cf_pred != y_test).mean() instead of
(y_cf_pred == y_target).mean(). These diverge on misclassified factuals where
y_pred != y_test, so the generation target y_target = 1 - y_pred != 1 - y_test.

This test constructs a deliberately misclassified factual and asserts that
compute_metrics counts a CF as valid iff predict(X_cf) == y_target, NOT != y_test.
"""

from __future__ import annotations

import numpy as np
import pytest
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.metrics_harness import (
    compute_dicoflex_common_metrics,
    compute_metrics,
)


class _MockDisc:
    """Discriminator that returns a fixed prediction array."""

    def __init__(self, preds: np.ndarray) -> None:
        self._preds = np.asarray(preds)

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._preds.copy()

    def eval(self) -> None:
        pass


def _make_dummy_data(n: int = 6, d: int = 4, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d)).astype(np.float64)
    return X


# ---------------------------------------------------------------------------
# Core validity regression test
# ---------------------------------------------------------------------------

def test_validity_uses_y_target_not_y_test():
    """validity must be (y_cf_pred == y_target).mean(), not (y_cf_pred != y_test).

    Setup:
      n=4 factuals; one is misclassified (y_pred != y_test).

      idx  y_test  y_pred  y_target  y_cf_pred  valid(correct)  valid(bug)
       0     0       0        1          1           True            True
       1     1       1        0          0           True            True
       2     0       1        0          0           True            False  ← misclassified
       3     1       0        1          0           False           True   ← misclassified

    Correct validity = 3/4 = 0.75
    Buggy  validity = 3/4 = 0.75  — same by coincidence on this example?
    Let's pick a case that clearly separates them.

    Simpler setup:
      idx  y_test  y_pred  y_target  y_cf_pred
       0     0       1        0          0     → y_cf==y_target → VALID; y_cf!=y_test → INVALID
       1     1       0        1          1     → y_cf==y_target → VALID; y_cf!=y_test → INVALID

    Correct validity = 2/2 = 1.0
    Buggy   validity = 0/2 = 0.0
    """
    n, d = 2, 3
    X_test = _make_dummy_data(n, d, seed=0)
    X_cf = _make_dummy_data(n, d, seed=1)
    X_train = _make_dummy_data(10, d, seed=2)

    # Both factuals are misclassified: y_pred != y_test
    y_test = np.array([0, 1])
    y_pred = np.array([1, 0])       # misclassified
    y_target = 1 - y_pred           # [0, 1]
    y_cf_pred_arr = np.array([0, 1])  # CF hits the target class

    disc = _MockDisc(y_cf_pred_arr)

    metrics = compute_metrics(
        disc_model=disc,
        X_cf=X_cf,
        X_test=X_test,
        X_train=X_train,
        y_test=y_test,
        y_target=y_target,
    )

    # Correct definition: CF pred matches the intended target class
    assert metrics["validity"] == pytest.approx(1.0), (
        f"Expected validity=1.0 (both CFs hit target class) but got {metrics['validity']:.3f}. "
        "This suggests validity is still scored against y_test instead of y_target."
    )


def test_validity_partial():
    """validity correctly counts only CFs that land on y_target."""
    n, d = 4, 3
    X_test = _make_dummy_data(n, d, seed=3)
    X_cf = _make_dummy_data(n, d, seed=4)
    X_train = _make_dummy_data(20, d, seed=5)

    y_test = np.array([0, 1, 0, 1])
    y_target = np.array([1, 0, 1, 0])  # standard flip
    # CFs: first two hit target, last two miss
    y_cf_pred_arr = np.array([1, 0, 0, 1])

    disc = _MockDisc(y_cf_pred_arr)

    metrics = compute_metrics(
        disc_model=disc,
        X_cf=X_cf,
        X_test=X_test,
        X_train=X_train,
        y_test=y_test,
        y_target=y_target,
    )

    assert metrics["validity"] == pytest.approx(0.5), (
        f"Expected validity=0.5 (2/4 CFs hit target) but got {metrics['validity']:.3f}."
    )


def test_validity_misclassified_factuals_diverge_from_buggy_definition():
    """Demonstrate that the corrected definition differs from the buggy != y_test one.

    With misclassified factuals, (y_cf_pred == y_target) != (y_cf_pred != y_test).
    """
    n, d = 3, 3
    X_test = _make_dummy_data(n, d, seed=6)
    X_cf = _make_dummy_data(n, d, seed=7)
    X_train = _make_dummy_data(20, d, seed=8)

    # idx 0: correctly classified, idx 1 & 2: misclassified
    y_test   = np.array([0,  1,  0])
    y_pred   = np.array([0,  0,  1])   # idx 1,2 misclassified
    y_target = 1 - y_pred              # [1, 1, 0]
    # CF predictions land on y_target for all three
    y_cf_pred_arr = np.array([1, 1, 0])

    disc = _MockDisc(y_cf_pred_arr)

    metrics = compute_metrics(
        disc_model=disc,
        X_cf=X_cf,
        X_test=X_test,
        X_train=X_train,
        y_test=y_test,
        y_target=y_target,
    )

    # All three CFs hit y_target → correct validity = 1.0
    assert metrics["validity"] == pytest.approx(1.0), (
        f"Expected validity=1.0 (all CFs hit y_target) but got {metrics['validity']:.3f}."
    )

    # Under the buggy definition (y_cf_pred != y_test):
    buggy_validity = float((y_cf_pred_arr != y_test).mean())
    # idx 0: 1 != 0 = True, idx 1: 1 != 1 = False, idx 2: 0 != 0 = False → 1/3
    assert buggy_validity == pytest.approx(1 / 3), (
        "Sanity check: buggy definition gives 1/3 on this example"
    )
    # Confirm they differ
    assert metrics["validity"] != pytest.approx(buggy_validity), (
        "Correct and buggy validity should differ when factuals are misclassified"
    )


def test_proximity_uses_valid_mask_from_y_target():
    """proximity_l2_jaccard must be computed on CFs where y_cf_pred == y_target."""
    n, d = 4, 2
    # X_test at origin, X_cf at varying distances
    X_test = np.zeros((n, d))
    X_cf = np.array([
        [1.0, 0.0],   # idx 0: valid, L2=1.0
        [0.5, 0.5],   # idx 1: invalid
        [0.0, 2.0],   # idx 2: valid, L2=2.0
        [3.0, 0.0],   # idx 3: invalid
    ])
    X_train = np.random.default_rng(9).uniform(0, 1, (20, d))

    y_test = np.array([0, 1, 0, 1])
    y_target = np.array([1, 0, 1, 0])
    y_cf_pred_arr = np.array([1, 1, 1, 1])  # idx 0,2 valid (hit target); 1,3 invalid

    disc = _MockDisc(y_cf_pred_arr)

    metrics = compute_metrics(
        disc_model=disc,
        X_cf=X_cf,
        X_test=X_test,
        X_train=X_train,
        y_test=y_test,
        y_target=y_target,
    )

    # validity: 2/4 (idx 0 and 2 hit y_target=[1,0,1,0])
    assert metrics["validity"] == pytest.approx(0.5)
    # proximity: mean L2 of idx 0 and 2 = (1.0 + 2.0) / 2 = 1.5
    assert metrics["proximity_l2_jaccard"] == pytest.approx(1.5)


def test_dicoflex_common_metrics_match_reference_definitions():
    X_test = np.zeros((3, 3), dtype=float)
    X_cf = np.array(
        [
            [0.04, 0.00, 0.0],
            [0.20, 0.30, 0.0],
            [0.60, 0.80, 0.0],
        ]
    )
    X_train = np.random.default_rng(21).uniform(0, 1, (40, 3))
    disc = _MockDisc(np.array([1, 1, 0]))

    metrics = compute_dicoflex_common_metrics(
        disc,
        X_cf,
        X_test,
        X_train,
        y_target=np.ones(3, dtype=int),
        numerical_idx=[0, 1],
        immutable_idx=[2],
    )

    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["validity"] == pytest.approx(2 / 3)
    assert metrics["actionability"] == pytest.approx(1.0)
    # Four of nine transformed entries exceed DiCoFlex's epsilon of 0.05.
    assert metrics["sparsity"] == pytest.approx(4 / 9)
    assert metrics["action_unit_sparsity_mean"] == pytest.approx(4 / 3)
    # Valid rows have mixed distances 0.02 and 0.25 over two numeric units.
    assert metrics["proximity_grouped_gower"] == pytest.approx(0.135)
    # Only the first two, valid rows contribute: (0.04 + 0.50) / 2.
    assert metrics["proximity_continuous_manhattan"] == pytest.approx(0.27)
    assert metrics["proximity_continuous_euclidean"] == pytest.approx(
        (0.04 + np.hypot(0.2, 0.3)) / 2
    )
    assert np.isfinite(metrics["lof_scores_cf"])
    assert np.isfinite(metrics["lof_scores_test"])
    assert np.isfinite(metrics["isolation_forest_scores_cf"])
    assert np.isfinite(metrics["isolation_forest_scores_test"])


def test_dicoflex_grouped_gower_treats_one_hot_group_as_one_feature() -> None:
    X_test = np.array([[0.0, 1.0, 0.0, 0.0]])
    X_cf = np.array([[0.2, 0.0, 0.0, 1.0]])
    X_train = np.random.default_rng(23).uniform(0, 1, (40, 4))
    metrics = compute_dicoflex_common_metrics(
        _MockDisc(np.array([1])),
        X_cf,
        X_test,
        X_train,
        y_target=np.array([1]),
        numerical_idx=[0],
        categorical_groups=[OneHotActionGroup("kind", (1, 2, 3))],
    )

    # (0.2 numeric contribution + 1 categorical mismatch) / 2 units.
    assert metrics["proximity_grouped_gower"] == pytest.approx(0.6)
    assert metrics["action_unit_sparsity_mean"] == pytest.approx(2.0)


def test_dicoflex_common_metrics_keep_singleton_target_dimension():
    X_train = np.random.default_rng(22).uniform(0, 1, (40, 2))
    metrics = compute_dicoflex_common_metrics(
        _MockDisc(np.array([1])),
        X_cf=np.array([[0.2, 0.3]]),
        X_test=np.array([[0.1, 0.3]]),
        X_train=X_train,
        y_target=np.array([1]),
        numerical_idx=[0, 1],
    )

    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["validity"] == pytest.approx(1.0)
