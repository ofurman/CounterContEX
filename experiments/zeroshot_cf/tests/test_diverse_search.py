# Copyright (c) Prior Labs GmbH 2026.

"""Tests for validity-constrained diverse counterfactual search."""

# ruff: noqa: ANN001, ANN202, D103

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.diverse_search import (
    DiverseSearchConfig,
    action_set_jaccard_distance,
    action_unit_signature,
    generate_diverse_counterfactuals,
)
from experiments.zeroshot_cf.grouped_categorical import (
    greedy_mixed_counterfactual,
)


class _OnesSampler:
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


class _AdditiveDisc:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p1 = np.clip(0.1 + 0.25 * np.asarray(X).sum(axis=1), 0.0, 0.99)
        return np.column_stack([1.0 - p1, p1])


def _primary(sampler, disc, factual, numerical_columns):
    return greedy_mixed_counterfactual(
        sampler,
        disc,
        factual,
        y_target=1,
        numerical_columns=numerical_columns,
        categorical_groups=[],
        candidate_quantiles=(0.5,),
    )


def test_diverse_beam_finds_distinct_valid_action_sets() -> None:
    sampler = _OnesSampler()
    disc = _AdditiveDisc()
    factual = np.zeros(3)
    primary, _, primary_info = _primary(sampler, disc, factual, [0, 1, 2])

    result = generate_diverse_counterfactuals(
        sampler,
        disc,
        factual,
        y_target=1,
        numerical_columns=[0, 1, 2],
        categorical_groups=[],
        primary_counterfactual=primary,
        primary_info=primary_info,
        config=DiverseSearchConfig(
            n_counterfactuals=3,
            beam_width=3,
            max_gower_ratio=1.0,
            max_gower_increase=0.0,
        ),
        candidate_quantiles=(0.5,),
        allow_revisits=False,
    )

    assert result.available_count == 3
    np.testing.assert_array_equal(result.counterfactuals[0], primary)
    predictions = np.argmax(disc.predict_proba(result.counterfactuals), axis=1)
    np.testing.assert_array_equal(predictions, np.ones(3, dtype=int))
    signatures = {
        action_unit_signature(row, factual, [0, 1, 2], [])
        for row in result.counterfactuals
    }
    assert len(signatures) == 3
    assert all(len(signature) == 2 for signature in signatures)


def test_diverse_search_never_pads_with_invalid_or_duplicate_rows() -> None:
    class OneFeatureDisc:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            p1 = 0.1 + 0.6 * np.asarray(X)[:, 0]
            return np.column_stack([1.0 - p1, p1])

    sampler = _OnesSampler()
    disc = OneFeatureDisc()
    factual = np.zeros(1)
    primary, _, primary_info = _primary(sampler, disc, factual, [0])

    result = generate_diverse_counterfactuals(
        sampler,
        disc,
        factual,
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[],
        primary_counterfactual=primary,
        primary_info=primary_info,
        config=DiverseSearchConfig(n_counterfactuals=3, beam_width=3),
        candidate_quantiles=(0.5,),
    )

    assert result.available_count == 1
    np.testing.assert_array_equal(result.counterfactuals, [[1.0]])
    assert result.target_probabilities[0] >= 0.5


def test_action_signature_counts_a_one_hot_group_once() -> None:
    group = OneHotActionGroup("job", (1, 2, 3))
    factual = np.array([0.0, 1.0, 0.0, 0.0])
    categorical_change = np.array([0.0, 0.0, 1.0, 0.0])
    numerical_change = np.array([1.0, 1.0, 0.0, 0.0])

    categorical_signature = action_unit_signature(
        categorical_change,
        factual,
        [0],
        [group],
    )
    numerical_signature = action_unit_signature(
        numerical_change,
        factual,
        [0],
        [group],
    )

    assert categorical_signature == frozenset({("categorical", "job")})
    assert numerical_signature == frozenset({("numerical", 0)})
    assert (
        action_set_jaccard_distance(
            categorical_signature,
            numerical_signature,
        )
        == 1.0
    )
