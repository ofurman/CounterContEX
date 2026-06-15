"""Unit tests for ConditionalDensitySampler (Stage 3).

Tests on a small synthetic dataset to avoid long runtime:
  x0, x1 ~ Uniform(0, 1)
  x2 = x0 + x1 + small_noise

The conditional mean E[x2 | x0, x1] is informative; a sampler that
conditions on x0 and x1 should reconstruct x2 with MSE below the
marginal-mean baseline (which knows nothing about x0, x1).

All three verification requirements from the stage spec are tested:
  1. sample_feature reconstructs target below marginal-mean MSE baseline.
  2. impute_masked preserves non-masked (immutable) columns byte-exactly.
  3. Switching target_class shifts sampled values on a 2-class toy set.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.zeroshot_cf.checkpoints import get_models
from experiments.zeroshot_cf.sampler import ConditionalDensitySampler


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def models():
    clf, reg = get_models(n_estimators=2)
    return clf, reg


def _make_synthetic(n: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a 3-feature regression-like dataset.

    x2 = x0 + x1 + 0.05 * noise
    y  = (x2 > 1.0).astype(int)  — binary label for class-conditioning tests
    """
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 1, n)
    x2 = x0 + x1 + 0.05 * rng.standard_normal(n)
    X = np.stack([x0, x1, x2], axis=1).astype(np.float64)
    y = (x2 > 1.0).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Test 1: sample_feature beats marginal-mean baseline
# ---------------------------------------------------------------------------

def test_sample_feature_beats_marginal_mean(models):
    """Reconstructing x2 from (x0, x1) context beats predicting the train mean."""
    clf, reg = models
    X, _ = _make_synthetic(n=80, seed=0)

    X_train, X_test = X[:60], X[60:]

    sampler = ConditionalDensitySampler(
        clf, reg,
        append_target=False,
        n_permutations=5,
        temperature=1e-9,
        random_state=42,
    )
    sampler.set_context(X_train)

    # Reconstruct x2 (col 2) from x0 and x1
    x2_true = X_test[:, 2]
    x2_pred = sampler.sample_feature(X_test, target_col=2, n_samples=1)

    mse_sampler = float(np.mean((x2_pred - x2_true) ** 2))
    marginal_mean = float(X_train[:, 2].mean())
    mse_baseline = float(np.mean((marginal_mean - x2_true) ** 2))

    print(f"\nMSE (sampler) = {mse_sampler:.4f}  |  MSE (marginal mean) = {mse_baseline:.4f}")
    assert mse_sampler < mse_baseline, (
        f"Sampler MSE ({mse_sampler:.4f}) should be below marginal-mean "
        f"baseline ({mse_baseline:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 2: impute_masked preserves non-masked columns exactly
# ---------------------------------------------------------------------------

def test_impute_masked_preserves_non_masked(models):
    """Non-masked columns must be byte-identical to the input after imputation."""
    clf, reg = models
    X, _ = _make_synthetic(n=60, seed=1)

    X_train, X_test = X[:40], X[40:]
    sampler = ConditionalDensitySampler(
        clf, reg,
        n_permutations=3,
        temperature=1e-9,
        random_state=0,
    )
    sampler.set_context(X_train)

    mask_cols = [2]  # mask x2
    X_filled = sampler.impute_masked(X_test, mask_cols=mask_cols)

    # Columns 0 and 1 must be preserved exactly
    non_masked = [0, 1]
    np.testing.assert_array_equal(
        X_filled[:, non_masked],
        X_test[:, non_masked],
        err_msg="Non-masked columns must be byte-identical to input",
    )

    # Masked column must have no NaNs in output
    assert not np.any(np.isnan(X_filled[:, 2])), "Masked column still has NaN after imputation"


# ---------------------------------------------------------------------------
# Test 3: impute_masked with append_target drops Y column correctly
# ---------------------------------------------------------------------------

def test_impute_masked_with_append_target_shape(models):
    """With append_target=True the output shape equals the original feature count."""
    clf, reg = models
    X, y = _make_synthetic(n=60, seed=2)

    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]

    sampler = ConditionalDensitySampler(
        clf, reg,
        append_target=True,
        n_permutations=3,
        temperature=1e-9,
        random_state=0,
    )
    sampler.set_context(X_train, y_context=y_train, target_class=1)

    X_filled = sampler.impute_masked(X_test, mask_cols=[2], fixed_target=1)

    assert X_filled.shape == X_test.shape, (
        f"Output shape {X_filled.shape} should equal input shape {X_test.shape}"
    )
    assert not np.any(np.isnan(X_filled)), "Output must be NaN-free"


# ---------------------------------------------------------------------------
# Test 4: target_class conditioning shifts sampled values
# ---------------------------------------------------------------------------

def test_target_class_shifts_samples(models):
    """Conditioning on different target classes should produce visibly different distributions."""
    clf, reg = models
    rng = np.random.default_rng(10)

    # Construct a clearly separable 2-class dataset:
    # class 0: x0~U(0,0.4), x1~U(0,0.4), x2 = x0+x1
    # class 1: x0~U(0.6,1), x1~U(0.6,1), x2 = x0+x1
    n = 50
    x0_c0 = rng.uniform(0.0, 0.4, n)
    x1_c0 = rng.uniform(0.0, 0.4, n)
    x2_c0 = x0_c0 + x1_c0
    X_c0 = np.stack([x0_c0, x1_c0, x2_c0], axis=1)

    x0_c1 = rng.uniform(0.6, 1.0, n)
    x1_c1 = rng.uniform(0.6, 1.0, n)
    x2_c1 = x0_c1 + x1_c1
    X_c1 = np.stack([x0_c1, x1_c1, x2_c1], axis=1)

    X_all = np.concatenate([X_c0, X_c1], axis=0).astype(np.float64)
    y_all = np.concatenate([np.zeros(n, dtype=np.int64),
                            np.ones(n, dtype=np.int64)])

    # Query: a mid-point row whose x2 is unknown
    X_query = np.array([[0.5, 0.5, np.nan]], dtype=np.float64)
    # We'll mask x2 and check that conditioning on class 0 vs 1 shifts the output

    sampler_c0 = ConditionalDensitySampler(
        clf, reg,
        append_target=True,
        n_permutations=5,
        temperature=1.0,  # use posterior sampling for visible spread
        random_state=0,
    )
    # Set NaN in query correctly
    X_query_clean = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)

    sampler_c0.set_context(X_all, y_context=y_all, target_class=0)
    filled_c0 = sampler_c0.impute_masked(X_query_clean, mask_cols=[2], fixed_target=0)

    sampler_c1 = ConditionalDensitySampler(
        clf, reg,
        append_target=True,
        n_permutations=5,
        temperature=1.0,
        random_state=0,
    )
    sampler_c1.set_context(X_all, y_context=y_all, target_class=1)
    filled_c1 = sampler_c1.impute_masked(X_query_clean, mask_cols=[2], fixed_target=1)

    val_c0 = float(filled_c0[0, 2])
    val_c1 = float(filled_c1[0, 2])
    print(f"\nClass-0 x2 sample: {val_c0:.3f}  |  Class-1 x2 sample: {val_c1:.3f}")

    # Class 0 x2 should be in ~(0, 0.8) and class 1 x2 in ~(1.2, 2.0)
    # We just assert the direction is right (c1 > c0)
    assert val_c1 > val_c0, (
        f"Expected class-1 sample ({val_c1:.3f}) > class-0 sample ({val_c0:.3f})"
    )
