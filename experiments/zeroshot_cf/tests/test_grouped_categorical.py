"""Tests for compact TabICL categorical encoding and atomic group actions."""

from __future__ import annotations

import numpy as np

from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.grouped_categorical import (
    CompactMixedSampler,
    GroupedCategoricalCodec,
    greedy_mixed_counterfactual,
    grouped_categorical_fallback,
)


class _CategoricalDisc:
    def predict_proba(self, X):
        # Categories 1 and 2 both flip. Category 2 has more classifier
        # confidence, while category 1 will be more plausible under LOF.
        p1 = 0.1 + 0.45 * X[:, 2] + 0.8 * X[:, 3]
        p1 = np.clip(p1, 0.0, 0.99)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _PreferCategoryOneLOF:
    def score_samples(self, X):
        # score_samples is negative LOF; category 1 has LOF=1, category 2=2.
        return -(1.0 + X[:, 3])


def test_codec_round_trips_categories_without_treating_dummies_as_scalars():
    group = OneHotActionGroup("job", (1, 2, 3))
    X = np.array(
        [
            [0.2, 1.0, 0.0, 0.0, 0.7],
            [0.4, 0.0, 0.0, 1.0, 0.8],
        ]
    )

    codec = GroupedCategoricalCodec.from_matrix(X, [group])

    assert codec.scalar_columns == (0, 4)
    assert codec.categorical_columns == (2,)
    np.testing.assert_array_equal(
        codec.encode(X),
        [[0.2, 0.7, 0.0], [0.4, 0.8, 2.0]],
    )


def test_codec_rejects_malformed_one_hot_rows():
    group = OneHotActionGroup("job", (1, 2, 3))
    malformed = np.array([[0.2, 1.0, 1.0, 0.0]])

    with np.testing.assert_raises_regex(ValueError, "invalid row"):
        GroupedCategoricalCodec.from_matrix(malformed, [group])


def test_compact_sampler_maps_original_scalar_columns():
    group = OneHotActionGroup("job", (1, 2, 3))
    X = np.array([[0.2, 1.0, 0.0, 0.0, 0.7]])
    codec = GroupedCategoricalCodec.from_matrix(X, [group])

    class _Recorder:
        def sample_candidate_grid(self, query, columns, **kwargs):
            np.testing.assert_array_equal(query, [[0.2, 0.7, 0.0]])
            assert columns == (0, 1)
            assert kwargs["fixed_target"] == 1
            return np.array([[0.1, 0.2], [0.3, 0.4]])

    sampler = CompactMixedSampler(_Recorder(), codec)
    values = sampler.sample_candidate_grid(
        X,
        [0, 4],
        quantiles=(0.25, 0.75),
        fixed_target=1,
    )

    np.testing.assert_array_equal(values, [[0.1, 0.2], [0.3, 0.4]])


def test_grouped_fallback_uses_validity_gate_then_minimum_lof():
    group = OneHotActionGroup("job", (1, 2, 3))
    factual = np.array([0.2, 1.0, 0.0, 0.0])

    def distribution(_row, _group):
        return np.array([0, 1, 2]), np.array([0.7, 0.2, 0.1])

    x_cf, changed, info = grouped_categorical_fallback(
        factual,
        disc=_CategoricalDisc(),
        y_target=1,
        groups=[group],
        category_distribution=distribution,
        plausibility_model=_PreferCategoryOneLOF(),
    )

    # Both alternatives flip, hence LOF chooses category 1 even though category
    # 2 has higher target-class probability.
    np.testing.assert_array_equal(x_cf[1:4], [0.0, 1.0, 0.0])
    assert changed == [1, 2]
    assert info["flipped"]
    assert info["steps"] == 1
    assert info["history"][0]["to_category"] == 1
    assert info["history"][0]["tabicl_conditional_probability"] == 0.2


def test_grouped_fallback_keeps_categories_absent_from_local_context():
    group = OneHotActionGroup("job", (1, 2, 3))
    factual = np.array([0.2, 1.0, 0.0, 0.0])

    def local_distribution(_row, _group):
        # Category 1 is absent from the selected kNN context, but it must remain
        # a candidate so the method does not lose counterfactual coverage.
        return np.array([0, 2]), np.array([0.8, 0.2])

    x_cf, _, info = grouped_categorical_fallback(
        factual,
        disc=_CategoricalDisc(),
        y_target=1,
        groups=[group],
        category_distribution=local_distribution,
        plausibility_model=_PreferCategoryOneLOF(),
    )

    np.testing.assert_array_equal(x_cf[1:4], [0.0, 1.0, 0.0])
    assert info["history"][0]["tabicl_conditional_probability"] == 0.0


def test_grouped_fallback_queries_tabicl_only_for_selected_group():
    job = OneHotActionGroup("job", (1, 2, 3))
    housing = OneHotActionGroup("housing", (4, 5))
    factual = np.array([0.2, 1.0, 0.0, 0.0, 1.0, 0.0])
    queried = []

    def distribution(_row, group):
        queried.append(group.name)
        return np.arange(len(group.columns)), np.full(
            len(group.columns),
            1.0 / len(group.columns),
        )

    _, _, info = grouped_categorical_fallback(
        factual,
        disc=_CategoricalDisc(),
        y_target=1,
        groups=[job, housing],
        category_distribution=distribution,
        plausibility_model=_PreferCategoryOneLOF(),
    )

    assert info["flipped"]
    assert queried == ["job"]


class _MixedGridSampler:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def sample_candidate_grid(
        self,
        _query,
        columns,
        *,
        quantiles,
        fixed_target,
        confidences=None,
    ):
        assert fixed_target == 1
        assert len(columns) == self.values.shape[0]
        assert len(quantiles) == self.values.shape[-1]
        assert confidences is None
        return self.values


class _MixedActionDisc:
    def __init__(self, numerical_weight, categorical_weight):
        self.numerical_weight = numerical_weight
        self.categorical_weight = categorical_weight

    def predict_proba(self, X):
        p1 = (
            0.1
            + self.numerical_weight * X[:, 0]
            + self.categorical_weight * X[:, 2]
        )
        p1 = np.clip(p1, 0.0, 0.99)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _PreferCategoricalLOF:
    def score_samples(self, X):
        # The categorical-only trial has LOF 1; numerical-only has LOF 2.
        return -(2.0 - X[:, 2])


def test_global_mixed_search_chooses_categorical_action_over_numerical() -> None:
    """A category swap goes first when it has the largest classifier effect."""
    group = OneHotActionGroup("job", (1, 2))
    factual = np.array([0.0, 1.0, 0.0])

    counterfactual, changed, info = greedy_mixed_counterfactual(
        _MixedGridSampler([[0.2, 0.8]]),
        _MixedActionDisc(numerical_weight=0.3, categorical_weight=0.5),
        factual,
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[group],
        candidate_quantiles=(0.25, 0.75),
        validity_first=True,
    )

    np.testing.assert_array_equal(counterfactual, [0.0, 0.0, 1.0])
    assert changed == [1, 2]
    assert info["history"][0]["action_type"] == "categorical"
    assert info["flipped"] is True


def test_global_mixed_search_chooses_numerical_action_over_categorical() -> None:
    """A scalar proposal goes first when it has the largest classifier effect."""
    group = OneHotActionGroup("job", (1, 2))
    factual = np.array([0.0, 1.0, 0.0])

    counterfactual, changed, info = greedy_mixed_counterfactual(
        _MixedGridSampler([[0.2, 0.8]]),
        _MixedActionDisc(numerical_weight=0.7, categorical_weight=0.3),
        factual,
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[group],
        candidate_quantiles=(0.25, 0.75),
        validity_first=True,
    )

    np.testing.assert_array_equal(counterfactual, [0.8, 1.0, 0.0])
    assert changed == [0]
    assert info["history"][0]["action_type"] == "numerical"
    assert info["flipped"] is True


def test_global_mixed_validity_gate_compares_lof_across_action_types() -> None:
    """LOF chooses across all globally valid numerical and category trials."""
    group = OneHotActionGroup("job", (1, 2))
    factual = np.array([0.0, 1.0, 0.0])

    counterfactual, _, info = greedy_mixed_counterfactual(
        _MixedGridSampler([[1.0]]),
        _MixedActionDisc(numerical_weight=0.8, categorical_weight=0.5),
        factual,
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[group],
        candidate_quantiles=(0.5,),
        plausibility_model=_PreferCategoricalLOF(),
        validity_first=True,
    )

    # Both trials flip; numerical has higher confidence, but categorical has
    # lower LOF and therefore wins inside the global valid-candidate set.
    np.testing.assert_array_equal(counterfactual, [0.0, 0.0, 1.0])
    assert info["history"][0]["action_type"] == "categorical"
    assert info["history"][0]["n_valid_candidates"] == 2
