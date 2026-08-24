"""Tests for compact TabICL categorical encoding and atomic group actions."""

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.grouped_categorical import (
    CompactMixedSampler,
    GroupedCategoricalCodec,
    greedy_mixed_counterfactual,
)
from experiments.zeroshot_cf.tabicl_joint_plausibility import TabICLJointScoreBatch


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


def test_data_plausible_scores_one_bounded_per_action_batch() -> None:
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


def test_valid_candidates_are_ranked_by_grouped_gower_before_probability() -> None:
    group = OneHotActionGroup("job", (1, 2))
    factual = np.array([0.0, 1.0, 0.0])

    counterfactual, _, info = greedy_mixed_counterfactual(
        _MixedGridSampler([[0.9]]),
        _MixedActionDisc(numerical_weight=0.5, categorical_weight=0.7),
        factual,
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[group],
        candidate_quantiles=(0.5,),
    )

    # Both proposals are valid and the category has higher target probability,
    # but 0.9 / 2 feature units is closer than 1 / 2 feature units.
    np.testing.assert_array_equal(counterfactual, [0.9, 1.0, 0.0])
    assert info["history"][0]["grouped_gower"] == 0.45


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
