#  Copyright (c) Prior Labs GmbH 2026.

"""Fast tests for the retained TabICL sampler surface.

The tests inject a deterministic fake unsupervised model, so they verify our
context/masking/batching logic without downloading TabICL checkpoints.
"""

from __future__ import annotations

import numpy as np
import torch
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.data import get_actionable_immutable
from experiments.zeroshot_cf.generator import (
    empirical_confidence_grid,
    select_test_rows,
)
from experiments.zeroshot_cf.grouped_categorical import (
    CompactMixedSampler,
    GroupedCategoricalCodec,
)
from experiments.zeroshot_cf.tabicl_sampler import (
    TabICLConditionalDensitySampler,
    _knn_indices,
    _local_tabicl_model_factory,
    _NumericalDistributionRequest,
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

    X_selected, y_selected = select_test_rows(X, y, 1, "stratified")
    X_repeated, y_repeated = select_test_rows(X, y, 1, "stratified")

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
        self.log_score_calls = 0

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
        distribution_request = getattr(self, "_numerical_distribution_request", None)
        columns_with_nan = sorted(
            np.unique(missing_cols),
            key=lambda col: np.isnan(out[:, col]).mean(),
        )
        for col in columns_with_nan:
            rows = missing_rows[missing_cols == col]
            # Deterministic and class-conditional. Candidate j gets
            # 0.1*(j+1) under class 0 and an additional 0.5 under class 1.
            values = 0.1 * (col + 1) + 0.5 * target[rows] + 0.2 * confidence[rows]
            if distribution_request is not None:
                values = values + distribution_request.next_alphas(len(rows))
                distribution_request.record(np.zeros(len(rows), dtype=np.float64))
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

    def log_score_samples(self, X, n_permutations=1):
        self.log_score_calls += 1
        self.last_scored_rows = np.asarray(X).copy()
        self.last_n_permutations = n_permutations
        return np.asarray(X).sum(axis=1)


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


def test_knn_context_uses_gower_for_compact_categories():
    X = np.array(
        [
            [0.0, 0.0, 100.0],
            [0.6, 0.6, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 1, 1])
    query = np.zeros(3)

    sampler = _sampler(categorical_features=[2]).set_context(
        X,
        y_context=y,
        max_context=1,
        selection="knn",
        query=query,
    )

    # A category mismatch contributes one unit, not the magnitude of its ID.
    np.testing.assert_array_equal(sampler.selected_context_, X[[0]])
    np.testing.assert_array_equal(sampler.selected_labels_, y[[0]])


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


def test_quantile_grid_batches_multiple_query_feature_pairs():
    X, y = _context()
    sampler = _sampler().set_context(
        X, y_context=y, max_context=8, selection="knn", query=X[10]
    )
    before = sampler.model.impute_calls

    values = sampler.sample_candidate_grid_batch(
        X[[0, 1, 2]],
        [0, 0, 2],
        quantiles=[0.1, 0.5],
        fixed_target=1,
    )

    assert values.shape == (3, 1, 2)
    np.testing.assert_allclose(
        values[:, 0],
        [[0.7, 1.1], [0.7, 1.1], [0.9, 1.3]],
        atol=1e-6,
    )
    assert sampler.model.impute_calls == before + 1


def test_numerical_distribution_returns_icdf_and_log_probability_in_one_call():
    X, y = _context()
    sampler = _sampler().set_context(
        X, y_context=y, max_context=8, selection="knn", query=X[10]
    )
    before = sampler.model.impute_calls

    values, log_probabilities = sampler.numerical_distribution_batch(
        X[[0, 1, 2]],
        [0, 0, 2],
        quantiles=[0.2, 0.8],
        fixed_target=1,
    )

    assert values.shape == (3, 1, 2)
    np.testing.assert_allclose(
        values[:, 0],
        [[0.8, 1.4], [0.8, 1.4], [1.0, 1.6]],
        atol=1e-6,
    )
    np.testing.assert_array_equal(log_probabilities, np.zeros((3, 1, 2)))
    assert sampler.model.impute_calls == before + 1
    assert not hasattr(type(sampler.model), "_numerical_distribution_request")


def test_numerical_distribution_preserves_iid_uniform_order_and_repetition():
    X, y = _context()
    sampler = _sampler().set_context(X, y_context=y)

    values, _ = sampler.numerical_distribution_batch(
        X[[0]],
        [0],
        quantiles=[0.8, 0.2, 0.8],
        fixed_target=1,
    )

    np.testing.assert_allclose(values[0, 0], [1.4, 0.8, 1.4], atol=1e-6)


def test_real_distribution_callback_evaluates_matching_log_density(tmp_path):
    model = _local_tabicl_model_factory(
        classifier_path=tmp_path / "classifier.ckpt",
        regressor_path=tmp_path / "regressor.ckpt",
        numerical_point_estimate="median",
        n_estimators=1,
        categorical_features=[],
        batch_size=2,
        random_state=7,
        device="cpu",
        estimator_params={},
    )
    model_type = type(model)
    request = _NumericalDistributionRequest(
        row_indices=(np.array([0, 1]),),
        alphas=(np.array([0.2, 0.8]),),
        log_probabilities=np.full(2, np.nan),
    )
    distribution = QuantileDistribution(
        torch.tensor(
            [
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 2.9, 5.2],
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 2.9, 5.2],
            ]
        )
    )
    model_type._numerical_distribution_request = request
    try:
        values = model_type._sample_numerical(
            distribution, 1.0, np.random.default_rng(99)
        )
    finally:
        delattr(model_type, "_numerical_distribution_request")

    np.testing.assert_allclose(values, [0.1, 2.9], atol=1e-6)
    np.testing.assert_allclose(request.finish(), [0.0, -np.log(23.0)], atol=1e-5)


def test_mixed_distribution_codec_matches_historical_grid_column_mapping():
    group = OneHotActionGroup("segment", (1, 2))
    rows = np.array([[0.1, 1.0, 0.0, 0.7], [0.2, 0.0, 1.0, 0.8]])
    codec = GroupedCategoricalCodec.from_matrix(rows, [group])

    class RecordingSampler:
        def __init__(self):
            self.calls = []

        def sample_candidate_grid_batch(self, encoded, columns, **kwargs):
            self.calls.append(("grid", np.asarray(encoded).copy(), tuple(columns)))
            return np.array([[[0.25, 0.75]], [[1.25, 1.75]]])

        def numerical_distribution_batch(self, encoded, columns, **kwargs):
            self.calls.append(("full", np.asarray(encoded).copy(), tuple(columns)))
            values = np.array([[[0.25, 0.75]], [[1.25, 1.75]]])
            return values, np.zeros_like(values)

    native = RecordingSampler()
    compact = CompactMixedSampler(native, codec)
    grid = compact.sample_candidate_grid_batch(
        rows, [0, 3], quantiles=[0.25, 0.75], fixed_target=1
    )
    values, log_probabilities = compact.numerical_distribution_batch(
        rows, [0, 3], quantiles=[0.25, 0.75], fixed_target=1
    )

    np.testing.assert_array_equal(values, grid)
    np.testing.assert_array_equal(log_probabilities, np.zeros_like(values))
    assert native.calls[0][2] == native.calls[1][2] == (0, 1)
    np.testing.assert_array_equal(native.calls[0][1], native.calls[1][1])


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


def test_joint_score_augments_complete_rows_with_target_and_confidence():
    X, y = _context()
    confidence = np.linspace(0.1, 0.9, len(X))
    sampler = _sampler().set_context(
        X,
        y_context=y,
        confidence_context=confidence,
    )

    scores = sampler.score_joint_rows(
        X[:2],
        fixed_target=1,
        fixed_confidence=[0.7, 0.8],
        n_permutations=2,
    )

    expected_rows = np.column_stack([X[:2], np.ones(2), [0.7, 0.8]])
    np.testing.assert_allclose(sampler.model.last_scored_rows, expected_rows)
    np.testing.assert_allclose(scores, expected_rows.sum(axis=1))
    assert sampler.model.last_n_permutations == 2
    assert sampler.model.log_score_calls == 1


def test_empirical_confidence_grid_uses_target_class_context_distribution():
    scores = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1, 1])

    grid = empirical_confidence_grid(scores, labels, 1, (0.25, 0.5, 0.75))

    np.testing.assert_allclose(grid, [0.7, 0.8, 0.85])


def test_quantile_grid_rejects_invalid_probability_levels():
    X, y = _context()
    sampler = _sampler().set_context(X, y_context=y)

    with np.testing.assert_raises_regex(ValueError, "strictly between"):
        sampler.sample_candidate_grid(X[[0]], [0], quantiles=[0.0, 0.5], fixed_target=1)
