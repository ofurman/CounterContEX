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
from experiments.zeroshot_cf.tabicl_sampler import (
    TabICLConditionalDensitySampler,
    _knn_indices,
    quantile_mode,
)
from tabicl._model.quantile_dist import QuantileDistribution


class _FakeTabICLUnsupervised:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.random_state = kwargs["random_state"]
        self.fit_calls = 0
        self.impute_calls = 0

    def fit(self, X):
        self.fit_calls += 1
        self.X_ = np.asarray(X, dtype=np.float32).copy()
        self.n_features_in_ = self.X_.shape[1]
        return self

    def impute(self, X, temperature=1e-8, n_iterations=1):
        self.impute_calls += 1
        out = np.asarray(X, dtype=np.float32).copy()
        target = out[:, -1].copy()
        for row, col in zip(*np.where(np.isnan(out))):
            # Deterministic and class-conditional. Candidate j gets
            # 0.1*(j+1) under class 0 and an additional 0.5 under class 1.
            out[row, col] = 0.1 * (col + 1) + 0.5 * target[row]
        return out


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


class _LinearDisc:
    def predict_proba(self, X):
        p1 = np.clip(0.15 * X[:, 0] + 0.85 * X[:, 2], 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


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
