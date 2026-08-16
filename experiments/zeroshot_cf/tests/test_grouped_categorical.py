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
from experiments.zeroshot_cf.tabicl_joint_plausibility import TabICLJointScoreBatch


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
        p1 = 0.1 + self.numerical_weight * X[:, 0] + self.categorical_weight * X[:, 2]
        p1 = np.clip(p1, 0.0, 0.99)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _PreferCategoricalLOF:
    def score_samples(self, X):
        # The categorical-only trial has LOF 1; numerical-only has LOF 2.
        return -(2.0 - X[:, 2])


class _NumericalRefinementLOF:
    def score_samples(self, X):
        # After a categorical flip establishes validity, increasing the scalar
        # feature moves the valid point into a denser region.
        return -(3.0 - 2.0 * X[:, 0])


class _RejectNumericalRefinementLOF:
    def score_samples(self, X):
        # The same validity-preserving scalar edit would make LOF worse.
        return -(1.0 + 2.0 * X[:, 0])


class _ProximityTradeoffLOF:
    def score_samples(self, X):
        # Both scalar values improve plausibility by at least 5%, but the more
        # distant value has the lowest absolute LOF.
        return -(2.0 - X[:, 0])


class _WholeRowTabICLScorer:
    def __init__(self):
        self.scored_rows = []

    def score_rows(self, rows, target_class):
        assert target_class == 1
        self.scored_rows.append(np.asarray(rows).copy())
        joint = 0.05 + 0.5 * (
            np.count_nonzero(~np.isclose(rows, 0.0), axis=1) - 1
        )
        return TabICLJointScoreBatch(
            joint_log_density=joint - 10.0,
        )


def test_data_plausible_mode_refines_relative_to_initial_sparse_score() -> None:
    class ReusableSampler:
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
            assert confidences is None
            return np.ones((len(columns), len(quantiles)))

    class AdditiveDisc:
        def predict_proba(self, X):
            p1 = np.clip(0.1 + 0.5 * X[:, 0] + 0.5 * X[:, 1], 0.0, 0.99)
            return np.column_stack([1.0 - p1, p1])

    scorer = _WholeRowTabICLScorer()
    counterfactual, changed, info = greedy_mixed_counterfactual(
        ReusableSampler(),
        AdditiveDisc(),
        np.array([0.0, 0.0]),
        y_target=1,
        numerical_columns=[0, 1],
        categorical_groups=[],
        candidate_quantiles=(0.5,),
        cf_mode="data_plausible",
        tabicl_joint_plausibility=scorer,
        joint_shortlist_size=16,
        min_joint_log_gain=0.0,
    )

    np.testing.assert_array_equal(counterfactual, [1.0, 1.0])
    assert changed == [0, 1]
    assert info["flipped"] is True
    assert info["initial_valid_step"] == 1
    assert info["accepted_refinement_count"] == 1
    assert info["initial_tabicl_joint_log_density"] == -9.95
    assert info["final_tabicl_joint_log_density"] == -9.45
    assert info["tabicl_joint_log_density_gain"] == 0.5
    assert len(scorer.scored_rows) == 1
    assert len(scorer.scored_rows[0]) == 2
    np.testing.assert_array_equal(info["initial_sparse_row"], [1.0, 0.0])
    assert info["initial_sparse_action_count"] == 1
    assert info["final_action_count"] == 2
    assert info["extra_actions"] == 1


def test_data_plausible_joint_score_is_only_evaluated_after_validity() -> None:
    class ValidOnlyScorer:
        def __init__(self):
            self.calls = 0

        def score_rows(self, rows, target_class):
            assert target_class == 1
            assert np.all(rows[:, 0] >= 0.8)
            self.calls += 1
            values = np.full(len(rows), 0.5)
            return TabICLJointScoreBatch(values)

    class ScalarDisc:
        def predict_proba(self, X):
            p1 = 0.1 + 0.6 * X[:, 0]
            return np.column_stack([1.0 - p1, p1])

    scorer = ValidOnlyScorer()
    counterfactual, _, info = greedy_mixed_counterfactual(
        _MixedGridSampler([[0.2, 0.8]]),
        ScalarDisc(),
        np.array([0.0]),
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[],
        candidate_quantiles=(0.25, 0.75),
        cf_mode="data_plausible",
        tabicl_joint_plausibility=scorer,
    )

    np.testing.assert_array_equal(counterfactual, [0.8])
    assert scorer.calls == 0
    assert info["refinement_stopping_reason"] == "no_eligible_candidate"


def test_data_plausible_can_replace_action_without_increasing_sparsity() -> None:
    class StatefulSampler:
        def __init__(self):
            self.calls = 0

        def sample_candidate_grid(
            self,
            _query,
            columns,
            *,
            quantiles,
            fixed_target,
            confidences=None,
        ):
            del quantiles, fixed_target, confidences
            self.calls += 1
            values = {
                0: 1.0 if self.calls == 1 else 0.8,
                1: 0.2,
            }
            return np.asarray([[values[column]] for column in columns])

    class ScalarDisc:
        def predict_proba(self, X):
            p1 = 0.1 + 0.6 * X[:, 0]
            return np.column_stack([1.0 - p1, p1])

    class PreferReplacementScorer:
        def score_rows(self, rows, target_class):
            assert target_class == 1
            scores = np.where(np.isclose(rows[:, 0], 0.8), 0.9, 0.5)
            scores = np.where(rows[:, 1] > 0.0, 0.4, scores)
            return TabICLJointScoreBatch(scores)

    counterfactual, changed, info = greedy_mixed_counterfactual(
        StatefulSampler(),
        ScalarDisc(),
        np.array([0.0, 0.0]),
        y_target=1,
        numerical_columns=[0, 1],
        categorical_groups=[],
        candidate_quantiles=(0.5,),
        cf_mode="data_plausible",
        tabicl_joint_plausibility=PreferReplacementScorer(),
        max_extra_actions=0,
        min_joint_log_gain=0.0,
    )

    np.testing.assert_allclose(counterfactual, [0.8, 0.0])
    assert changed == [0]
    assert info["accepted_refinement_count"] == 1
    assert info["initial_sparse_action_count"] == 1
    assert info["final_action_count"] == 1
    assert info["extra_actions"] == 0
    assert info["refinement_stopping_reason"] == "one_shot_accepted"


def test_data_plausible_scores_one_bounded_action_diverse_batch() -> None:
    class UniformSampler:
        def sample_candidate_grid(
            self,
            _query,
            columns,
            *,
            quantiles,
            fixed_target,
            confidences=None,
        ):
            del quantiles, fixed_target, confidences
            return np.ones((len(columns), 1))

    class AnyActionDisc:
        def predict_proba(self, X):
            p1 = 0.1 + 0.6 * np.max(X, axis=1)
            return np.column_stack([1.0 - p1, p1])

    class CountingScorer:
        def __init__(self):
            self.batch_count = 0
            self.row_count = 0

        def score_rows(self, rows, target_class):
            assert target_class == 1
            self.batch_count += 1
            self.row_count += len(rows)
            return TabICLJointScoreBatch(np.count_nonzero(rows, axis=1).astype(float))

    scorer = CountingScorer()
    counterfactual, _, info = greedy_mixed_counterfactual(
        UniformSampler(),
        AnyActionDisc(),
        np.zeros(20),
        y_target=1,
        numerical_columns=list(range(20)),
        categorical_groups=[],
        candidate_quantiles=(0.5,),
        cf_mode="data_plausible",
        tabicl_joint_plausibility=scorer,
        joint_shortlist_size=4,
        max_extra_actions=1,
    )

    assert np.count_nonzero(counterfactual) == 2
    assert scorer.batch_count == 1
    assert scorer.row_count == 5  # sparse incumbent plus four action units
    assert info["joint_scoring_batch_count"] == 1
    assert info["joint_rows_scored"] == 5
    assert info["accepted_refinement_count"] == 1


def test_data_plausible_reuses_one_joint_batch_for_multiple_counterfactuals() -> None:
    class UniformSampler:
        def sample_candidate_grid(
            self,
            _query,
            columns,
            *,
            quantiles,
            fixed_target,
            confidences=None,
        ):
            del quantiles, fixed_target, confidences
            return np.ones((len(columns), 1))

    class AnyActionDisc:
        def predict_proba(self, X):
            p1 = 0.1 + 0.6 * np.max(X, axis=1)
            return np.column_stack([1.0 - p1, p1])

    class CountingScorer:
        def __init__(self):
            self.batch_count = 0
            self.row_count = 0

        def score_rows(self, rows, target_class):
            assert target_class == 1
            self.batch_count += 1
            self.row_count += len(rows)
            return TabICLJointScoreBatch(
                np.count_nonzero(rows, axis=1).astype(float)
            )

    scorer = CountingScorer()
    counterfactual, _, info = greedy_mixed_counterfactual(
        UniformSampler(),
        AnyActionDisc(),
        np.zeros(8),
        y_target=1,
        numerical_columns=list(range(8)),
        categorical_groups=[],
        candidate_quantiles=(0.5,),
        cf_mode="data_plausible",
        tabicl_joint_plausibility=scorer,
        joint_shortlist_size=7,
        max_extra_actions=1,
        n_counterfactuals=5,
    )

    diverse = np.asarray(info["diverse_counterfactuals"])
    np.testing.assert_array_equal(diverse[0], counterfactual)
    assert diverse.shape == (5, 8)
    assert len({tuple(row) for row in diverse}) == 5
    assert np.all(np.count_nonzero(diverse, axis=1) <= 2)
    assert np.all(info["diverse_joint_log_densities"] >= 1.0)
    assert scorer.batch_count == 1
    assert scorer.row_count == 8
    assert info["joint_scoring_batch_count"] == 1
    assert info["n_counterfactuals_requested"] == 5


def test_diversity_pool_does_not_change_frozen_primary_shortlist() -> None:
    class UniformSampler:
        def sample_candidate_grid(
            self,
            _query,
            columns,
            *,
            quantiles,
            fixed_target,
            confidences=None,
        ):
            del quantiles, fixed_target, confidences
            return np.ones((len(columns), 1))

    class AnyActionDisc:
        def predict_proba(self, X):
            p1 = 0.1 + 0.6 * np.max(X, axis=1)
            return np.column_stack([1.0 - p1, p1])

    class PreferLateCandidateScorer:
        def score_rows(self, rows, target_class):
            assert target_class == 1
            scores = rows[:, 1] + 2.0 * rows[:, 2] + 3.0 * rows[:, 3]
            return TabICLJointScoreBatch(scores)

    counterfactual, _, info = greedy_mixed_counterfactual(
        UniformSampler(),
        AnyActionDisc(),
        np.zeros(4),
        y_target=1,
        numerical_columns=list(range(4)),
        categorical_groups=[],
        candidate_quantiles=(0.5,),
        cf_mode="data_plausible",
        tabicl_joint_plausibility=PreferLateCandidateScorer(),
        joint_shortlist_size=3,
        primary_shortlist_size=1,
        max_extra_actions=1,
        n_counterfactuals=4,
    )

    np.testing.assert_array_equal(counterfactual, [1.0, 1.0, 0.0, 0.0])
    diverse = np.asarray(info["diverse_counterfactuals"])
    np.testing.assert_array_equal(diverse[0], counterfactual)
    assert any(np.array_equal(row, [1.0, 0.0, 0.0, 1.0]) for row in diverse)


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
    )

    np.testing.assert_array_equal(counterfactual, [0.8, 1.0, 0.0])
    assert changed == [0]
    assert info["history"][0]["action_type"] == "numerical"
    assert info["flipped"] is True


def test_validity_search_requires_progress_from_first_step() -> None:
    factual = np.array([0.0])

    class ScalarDisc:
        def predict_proba(self, X):
            p1 = np.clip(0.1 + 0.3 * X[:, 0], 0.0, 0.99)
            return np.column_stack([1.0 - p1, p1])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    counterfactual, changed, info = greedy_mixed_counterfactual(
        _MixedGridSampler([[-0.2]]),
        ScalarDisc(),
        factual,
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[],
        candidate_quantiles=(0.5,),
    )

    np.testing.assert_array_equal(counterfactual, factual)
    assert changed == []
    assert info["validity_steps"] == 0


def test_tabicl_probability_ranks_categorical_proposals() -> None:
    group = OneHotActionGroup("job", (0, 1, 2))
    factual = np.array([1.0, 0.0, 0.0])

    class CategoryDisc:
        def predict_proba(self, X):
            p1 = 0.1 + 0.3 * X[:, 1] + 0.7 * X[:, 2]
            return np.column_stack([1.0 - p1, p1])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def distribution(row, action_group, confidence):
        del row, action_group, confidence
        return np.array([0, 1, 2]), np.array([0.1, 0.2, 0.7])

    counterfactual, _, info = greedy_mixed_counterfactual(
        _MixedGridSampler([]),
        CategoryDisc(),
        factual,
        y_target=1,
        numerical_columns=[],
        categorical_groups=[group],
        category_distribution=distribution,
    )

    np.testing.assert_array_equal(counterfactual, [0.0, 0.0, 1.0])
    assert info["history"][0]["tabicl_proposal_rank"] == 1
    assert info["history"][0]["tabicl_conditional_probability"] == 0.7


def test_categorical_proposal_expands_for_coverage_when_top_rank_stalls() -> None:
    group = OneHotActionGroup("job", (0, 1, 2))
    factual = np.array([1.0, 0.0, 0.0])

    class CategoryDisc:
        def predict_proba(self, X):
            p1 = 0.1 - 0.05 * X[:, 1] + 0.7 * X[:, 2]
            return np.column_stack([1.0 - p1, p1])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def distribution(row, action_group, confidence):
        del row, action_group, confidence
        # TabICL prefers category 1, but it cannot improve classifier confidence.
        return np.array([0, 1, 2]), np.array([0.1, 0.8, 0.1])

    counterfactual, _, info = greedy_mixed_counterfactual(
        _MixedGridSampler([]),
        CategoryDisc(),
        factual,
        y_target=1,
        numerical_columns=[],
        categorical_groups=[group],
        category_distribution=distribution,
    )

    np.testing.assert_array_equal(counterfactual, [0.0, 0.0, 1.0])
    assert info["history"][0]["coverage_fallback"] is True
    assert info["history"][0]["tabicl_conditional_probability"] == 0.1


def test_valid_mixed_candidates_use_lof_at_the_validity_boundary() -> None:
    """Once proposals are valid, the lowest-LOF mixed action wins."""
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
    )

    # Both first-step trials flip. The categorical action has lower LOF even
    # though the numerical action has greater target-class confidence.
    np.testing.assert_array_equal(counterfactual, [0.0, 0.0, 1.0])
    assert info["history"][0]["action_type"] == "categorical"
    assert info["history"][0]["selection_phase"] == "validity_search"
    assert info["history"][0]["n_valid_candidates"] == 2
    assert info["refinement_steps"] == 0


def test_validity_search_scores_lof_only_for_valid_candidates() -> None:
    """Invalid probability-ascent proposals must not incur LOF evaluation."""

    class ScalarDisc:
        def predict_proba(self, X):
            p1 = 0.1 + 0.6 * X[:, 0]
            return np.column_stack([1.0 - p1, p1])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    class ValidOnlyLOF:
        def score_samples(self, X):
            assert np.all(X[:, 0] >= 0.8)
            return -np.ones(len(X))

    counterfactual, _, info = greedy_mixed_counterfactual(
        _MixedGridSampler([[0.2, 0.8]]),
        ScalarDisc(),
        np.array([0.0]),
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[],
        candidate_quantiles=(0.25, 0.75),
        plausibility_model=ValidOnlyLOF(),
    )

    np.testing.assert_array_equal(counterfactual, [0.8])
    assert info["flipped"] is True


def test_global_mixed_search_refines_lof_after_reaching_validity() -> None:
    """A valid incumbent can be improved without losing target validity."""
    group = OneHotActionGroup("job", (1, 2))
    factual = np.array([0.0, 1.0, 0.0])

    counterfactual, changed, info = greedy_mixed_counterfactual(
        _MixedGridSampler([[0.8]]),
        _MixedActionDisc(numerical_weight=0.2, categorical_weight=0.5),
        factual,
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[group],
        candidate_quantiles=(0.5,),
        plausibility_model=_NumericalRefinementLOF(),
    )

    np.testing.assert_array_equal(counterfactual, [0.8, 0.0, 1.0])
    assert changed == [0, 1, 2]
    assert info["flipped"] is True
    assert info["initial_valid_step"] == 1
    assert info["refinement_steps"] == 1
    assert [step["selection_phase"] for step in info["history"]] == [
        "validity_search",
        "plausibility_refinement",
    ]


def test_global_mixed_search_rejects_valid_but_less_plausible_refinement() -> None:
    """Validity alone is insufficient to replace the valid incumbent."""
    group = OneHotActionGroup("job", (1, 2))
    factual = np.array([0.0, 1.0, 0.0])

    counterfactual, changed, info = greedy_mixed_counterfactual(
        _MixedGridSampler([[0.8]]),
        _MixedActionDisc(numerical_weight=0.2, categorical_weight=0.5),
        factual,
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[group],
        candidate_quantiles=(0.5,),
        plausibility_model=_RejectNumericalRefinementLOF(),
    )

    np.testing.assert_array_equal(counterfactual, [0.0, 0.0, 1.0])
    assert changed == [1, 2]
    assert info["flipped"] is True
    assert info["initial_valid_step"] == 1
    assert info["refinement_steps"] == 0
    assert len(info["history"]) == 1


def test_global_mixed_search_does_not_refine_below_lof_threshold() -> None:
    """An already plausible valid CF keeps its minimal one-action solution."""
    group = OneHotActionGroup("job", (1, 2))
    factual = np.array([0.0, 1.0, 0.0])

    counterfactual, changed, info = greedy_mixed_counterfactual(
        _MixedGridSampler([[0.8]]),
        _MixedActionDisc(numerical_weight=0.2, categorical_weight=0.5),
        factual,
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[group],
        candidate_quantiles=(0.5,),
        plausibility_model=_NumericalRefinementLOF(),
        refinement_lof_threshold=3.1,
    )

    np.testing.assert_array_equal(counterfactual, [0.0, 0.0, 1.0])
    assert changed == [1, 2]
    assert info["refinement_steps"] == 0


def test_global_mixed_refinement_prefers_proximity_after_lof_gain_gate() -> None:
    """Meaningful LOF gains are ranked by sparsity and factual proximity."""
    group = OneHotActionGroup("job", (1, 2))
    factual = np.array([0.0, 1.0, 0.0])

    counterfactual, _, info = greedy_mixed_counterfactual(
        _MixedGridSampler([[0.2, 0.8]]),
        _MixedActionDisc(numerical_weight=0.2, categorical_weight=0.5),
        factual,
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[group],
        candidate_quantiles=(0.25, 0.75),
        plausibility_model=_ProximityTradeoffLOF(),
    )

    # x=0.8 has lower LOF, but x=0.2 already clears the relative-gain gate and
    # is closer to the factual while having the same action-level sparsity.
    np.testing.assert_allclose(counterfactual, [0.2, 0.0, 1.0])
    assert info["refinement_steps"] == 1
    assert info["history"][1]["action_sparsity"] == 2
