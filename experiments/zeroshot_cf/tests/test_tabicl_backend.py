#  Copyright (c) Prior Labs GmbH 2026.

"""Fast tests for the TabICL sampler and candidate-expanded greedy path.

The tests inject a deterministic fake unsupervised model, so they verify our
context/masking/batching logic without downloading TabICL checkpoints.
"""

from __future__ import annotations

import numpy as np
import torch
from experiments.zeroshot_cf.data import get_actionable_immutable
from experiments.zeroshot_cf.greedy import greedy_counterfactual
from experiments.zeroshot_cf.exp8_tabicl_cf import (
    _select_test_rows,
    empirical_confidence_grid,
)
from experiments.zeroshot_cf.tabicl_sampler import (
    TabICLConditionalDensitySampler,
    _knn_indices,
    quantile_mode,
)
from tabicl._model.quantile_dist import QuantileDistribution


class _FakeCategoricalEstimator:
    def __init__(self, y):
        self.classes_, counts = np.unique(
            np.asarray(y, dtype=int),
            return_counts=True,
        )
        self.probabilities = counts / counts.sum()

    def predict_proba(self, X):
        return np.repeat(self.probabilities.reshape(1, -1), len(X), axis=0)


def test_stratified_selector_supports_one_point_smoke_tests():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([0, 1] * 5)

    X_selected, y_selected = _select_test_rows(X, y, 1, "stratified")
    X_repeated, y_repeated = _select_test_rows(X, y, 1, "stratified")

    assert X_selected.shape == (1, 2)
    assert y_selected.shape == (1,)
    np.testing.assert_array_equal(X_selected, X_repeated)
    np.testing.assert_array_equal(y_selected, y_repeated)


class _FakeTabICLUnsupervised:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.random_state = kwargs["random_state"]
        self.fit_calls = 0
        self.impute_calls = 0
        self.conditional_fit_calls = 0

    def fit(self, X):
        self.fit_calls += 1
        self.X_ = np.asarray(X, dtype=np.float32).copy()
        self.n_features_in_ = self.X_.shape[1]
        return self

    def impute(self, X, temperature=1e-8, n_iterations=1):
        self.impute_calls += 1
        out = np.asarray(X, dtype=np.float32).copy()
        target_idx = self.kwargs["categorical_features"][0]
        target = out[:, target_idx].copy()
        confidence = (
            out[:, target_idx + 1].copy()
            if out.shape[1] > target_idx + 1
            else np.zeros(len(out), dtype=np.float32)
        )
        missing_rows, missing_cols = np.where(np.isnan(out))
        quantile_grid = getattr(self, "_numerical_quantile_grid", None)
        for col in np.unique(missing_cols):
            rows = missing_rows[missing_cols == col]
            # Deterministic and class-conditional. Candidate j gets
            # 0.1*(j+1) under class 0 and an additional 0.5 under class 1.
            values = (
                0.1 * (col + 1)
                + 0.5 * target[rows]
                + 0.2 * confidence[rows]
            )
            if quantile_grid is not None:
                values = values + np.asarray(quantile_grid)
            out[rows, col] = values
        return out

    def _prepare_conditional_data(
        self, *, tgt_idx, cond_features, train_mask, X_test, rng
    ):
        del rng
        return (
            self.X_[train_mask][:, cond_features],
            self.X_[train_mask, tgt_idx],
            X_test[:, cond_features],
        )

    def _fit_conditional_estimator(self, col_idx, X_train, y_train):
        del X_train
        self.conditional_fit_calls += 1
        return _FakeCategoricalEstimator(y_train), (
            col_idx in self.kwargs["categorical_features"]
        )


def _factory(**kwargs):
    return _FakeTabICLUnsupervised(**kwargs)


def _context():
    x0 = np.linspace(0.0, 1.0, 20)
    X = np.column_stack([x0, 1.0 - x0, x0**2]).astype(np.float64)
    y = (x0 >= 0.5).astype(np.int64)
    return X, y


def _sampler(**kwargs):
    return TabICLConditionalDensitySampler(
        n_estimators=2,
        temperature=1e-9,
        random_state=7,
        device="cpu",
        model_factory=_factory,
        **kwargs,
    )


def test_audit_exposes_all_continuous_features_as_actionable():
    class _AuditBundle:
        feature_names = [f"feature_{j}" for j in range(23)]

    actionable, immutable = get_actionable_immutable("audit", _AuditBundle())

    assert actionable == list(range(23))
    assert immutable == []


def test_knn_both_context_is_selected_and_y_is_appended():
    X, y = _context()
    query = np.array([0.52, 0.48, 0.27])
    sampler = _sampler().set_context(
        X,
        y_context=y,
        target_class=None,
        max_context=8,
        selection="knn",
        query=query,
    )

    expected_idx = _knn_indices(X.astype(np.float32), query, 8)
    np.testing.assert_allclose(sampler.selected_context_, X[expected_idx])
    np.testing.assert_array_equal(sampler.selected_labels_, y[expected_idx])
    np.testing.assert_allclose(sampler.model.X_[:, :-1], X[expected_idx])
    np.testing.assert_array_equal(sampler.model.X_[:, -1], y[expected_idx])
    assert sampler.model.kwargs["categorical_features"] == [X.shape[1]]
    assert sampler.model.kwargs["estimator_params"]["kv_cache"] is True


def test_context_update_reuses_loaded_model_weights():
    X, y = _context()
    sampler = _sampler()
    sampler.set_context(X, y_context=y, max_context=6, selection="knn", query=X[2])
    model = sampler.model
    first_context = model.X_.copy()

    sampler.set_context(X, y_context=y, max_context=6, selection="knn", query=X[-3])

    assert sampler.model is model
    assert model.fit_calls == 1
    assert not np.array_equal(model.X_, first_context)
    assert len(model.X_) == 6


def test_explicit_categorical_feature_returns_complete_distribution():
    X, y = _context()
    X[:, 0] = np.resize([0.0, 1.0, 2.0], len(X))
    sampler = _sampler(categorical_features=[0]).set_context(X, y_context=y)

    categories, probabilities = sampler.categorical_distribution(
        X[[0]],
        0,
        fixed_target=1,
    )

    np.testing.assert_array_equal(categories, [0, 1, 2])
    np.testing.assert_allclose(probabilities, [0.35, 0.35, 0.30])
    assert sampler.model.kwargs["categorical_features"] == [0, X.shape[1]]


def test_categorical_confidence_conditions_share_one_conditional_fit() -> None:
    """Confidence anchors for one category share one fitted conditional."""
    X, y = _context()
    X[:, 0] = np.resize([0.0, 1.0, 2.0], len(X))
    confidence = np.linspace(0.1, 0.9, len(X))
    sampler = _sampler(categorical_features=[0]).set_context(
        X,
        y_context=y,
        confidence_context=confidence,
    )

    before = sampler.model.conditional_fit_calls
    categories, probabilities = sampler.categorical_distribution(
        X[[0]],
        0,
        fixed_target=1,
        fixed_confidence=[0.25, 0.75],
    )

    np.testing.assert_array_equal(categories, [0, 1, 2])
    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(probabilities[0], probabilities[1])
    assert sampler.model.conditional_fit_calls == before + 1


def test_refit_context_update_calls_upstream_fit_each_time():
    X, y = _context()
    sampler = _sampler(context_update="refit")
    sampler.set_context(X, y_context=y, max_context=6, selection="knn", query=X[2])
    model = sampler.model

    sampler.set_context(X, y_context=y, max_context=6, selection="knn", query=X[-3])

    assert sampler.model is model
    assert model.fit_calls == 2
    expected_idx = _knn_indices(X.astype(np.float32), X[-3], 6)
    np.testing.assert_allclose(model.X_[:, :-1], X[expected_idx])


def test_invalid_context_update_is_rejected():
    with np.testing.assert_raises_regex(ValueError, "context_update"):
        _sampler(context_update="shortcut")


def test_invalid_numerical_point_estimate_is_rejected():
    with np.testing.assert_raises_regex(ValueError, "numerical_point_estimate"):
        _sampler(numerical_point_estimate="mean")


def test_quantile_mode_differs_from_median_on_skewed_distribution():
    dist = QuantileDistribution(torch.tensor([[0.0, 10.0, 20.0, 20.1]]))

    median = float(dist.icdf(torch.tensor(0.5)).item())
    mode = float(quantile_mode(dist)[0])

    assert 14.0 < median < 16.0
    assert 19.9 < mode < 20.2


def test_candidate_expansion_uses_one_imputation_call():
    X, y = _context()
    sampler = _sampler().set_context(
        X, y_context=y, max_context=8, selection="knn", query=X[10]
    )
    before = sampler.model.impute_calls
    values = sampler.sample_candidates(
        X[[0]], [0, 2], sample_temperature=1e-9, fixed_target=1
    )

    np.testing.assert_allclose(values, [0.6, 0.8], atol=1e-6)
    assert sampler.model.impute_calls == before + 1


def test_quantile_grid_expands_rows_but_uses_one_imputation_call():
    X, y = _context()
    sampler = _sampler().set_context(
        X, y_context=y, max_context=8, selection="knn", query=X[10]
    )
    before = sampler.model.impute_calls

    values = sampler.sample_candidate_grid(
        X[[0]],
        [0, 2],
        quantiles=[0.1, 0.5, 0.9],
        fixed_target=1,
    )

    np.testing.assert_allclose(
        values,
        [[0.7, 1.1, 1.5], [0.9, 1.3, 1.7]],
        atol=1e-6,
    )
    assert sampler.model.impute_calls == before + 1
    assert not hasattr(sampler.model, "_numerical_quantile_grid")


def test_confidence_conditioning_uses_empirical_grid_in_one_call():
    X, y = _context()
    confidence = np.linspace(0.1, 0.9, len(X))
    sampler = _sampler().set_context(
        X,
        y_context=y,
        confidence_context=confidence,
        max_context=8,
        selection="knn",
        query=X[10],
    )
    before = sampler.model.impute_calls

    values = sampler.sample_candidate_grid(
        X[[0]],
        [0],
        quantiles=[0.1, 0.5],
        confidences=[0.55, 0.85],
        fixed_target=1,
    )

    assert values.shape == (1, 2, 2)
    np.testing.assert_allclose(
        values[0],
        [[0.81, 1.21], [0.87, 1.27]],
        atol=1e-6,
    )
    assert sampler.model.impute_calls == before + 1
    np.testing.assert_allclose(
        sampler.model.X_[:, -1],
        sampler.selected_confidences_,
    )
    assert sampler.model.kwargs["categorical_features"] == [X.shape[1]]


def test_empirical_confidence_grid_uses_target_class_context_distribution():
    scores = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1, 1])

    grid = empirical_confidence_grid(scores, labels, 1, (0.25, 0.5, 0.75))

    np.testing.assert_allclose(grid, [0.7, 0.8, 0.85])


def test_quantile_grid_rejects_invalid_probability_levels():
    X, y = _context()
    sampler = _sampler().set_context(X, y_context=y)

    with np.testing.assert_raises_regex(ValueError, "strictly between"):
        sampler.sample_candidate_grid(
            X[[0]], [0], quantiles=[0.0, 0.5], fixed_target=1
        )


class _LinearDisc:
    def predict_proba(self, X):
        p1 = np.clip(0.15 * X[:, 0] + 0.85 * X[:, 2], 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _GridDisc:
    def predict_proba(self, X):
        p1 = np.clip(0.05 * X[:, 0] + 0.4 * X[:, 2], 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _PreferMiddleLOF:
    def score_samples(self, X):
        # Negative outlier score; the lowest resulting LOF is at x2=1.3.
        return -np.abs(X[:, 2] - 1.3)


def test_quantile_grid_greedy_selects_best_feature_value_pair():
    X, y = _context()
    query = np.array([0.0, 1.0, 0.0])
    sampler = _sampler().set_context(X, y_context=y)

    x_cf, changed, info = greedy_counterfactual(
        sampler,
        _GridDisc(),
        query,
        y_target=1,
        actionable_idx=[0, 2],
        selector="prob_ascent",
        batch_candidates=True,
        candidate_quantiles=(0.1, 0.5, 0.9),
    )

    assert changed == [2]
    np.testing.assert_allclose(x_cf[2], 1.7, atol=1e-6)
    assert info["flipped"]
    assert info["candidate_quantiles"] == (0.1, 0.5, 0.9)


def test_validity_first_selects_lowest_lof_not_highest_probability():
    X, y = _context()
    query = np.array([0.0, 1.0, 0.0])
    sampler = _sampler().set_context(X, y_context=y)

    x_cf, changed, info = greedy_counterfactual(
        sampler,
        _GridDisc(),
        query,
        y_target=1,
        actionable_idx=[2],
        selector="prob_ascent",
        batch_candidates=True,
        candidate_quantiles=(0.1, 0.5, 0.9),
        plausibility_model=_PreferMiddleLOF(),
        validity_first=True,
    )

    # q=.5 gives x2=1.3 and p=.52; q=.9 gives x2=1.7 and p=.68.
    # Both flip, so LOF alone must select q=.5 despite its lower confidence.
    assert changed == [2]
    np.testing.assert_allclose(x_cf[2], 1.3, atol=1e-6)
    assert info["flipped"]
    assert info["selection_history"][0]["quantile"] == 0.5
    assert abs(info["selection_history"][0]["lof"]) < 1e-6


def test_batched_and_sequential_greedy_are_equivalent_and_preserve_immutable():
    X, y = _context()
    query = X[0].copy()
    disc = _LinearDisc()

    sequential_sampler = _sampler().set_context(
        X, y_context=y, max_context=8, selection="knn", query=query
    )
    batched_sampler = _sampler().set_context(
        X, y_context=y, max_context=8, selection="knn", query=query
    )

    seq_cf, seq_changed, seq_info = greedy_counterfactual(
        sequential_sampler,
        disc,
        query,
        y_target=1,
        actionable_idx=[0, 2],
        selector="prob_ascent",
        temperature=1e-9,
        batch_candidates=False,
    )
    bat_cf, bat_changed, bat_info = greedy_counterfactual(
        batched_sampler,
        disc,
        query,
        y_target=1,
        actionable_idx=[0, 2],
        selector="prob_ascent",
        temperature=1e-9,
        batch_candidates=True,
    )

    np.testing.assert_allclose(bat_cf, seq_cf)
    assert bat_changed == seq_changed
    assert bat_info["flipped"] == seq_info["flipped"]
    assert bat_info["steps"] == seq_info["steps"]
    # Column 1 was immutable and remains bit-identical to the factual.
    assert bat_cf[1] == query[1]
    # One batched call evaluates both initial candidates; the sequential path
    # needs one call for each of them.
    assert batched_sampler.model.impute_calls < sequential_sampler.model.impute_calls
