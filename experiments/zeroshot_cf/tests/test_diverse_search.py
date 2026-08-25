# Copyright (c) Prior Labs GmbH 2026.

"""Tests for bounded diverse beam search and DPP selection."""

# ruff: noqa: ANN001, ANN202, D103

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.diverse_search import (
    DiverseBeamSearchConfig,
    action_set_jaccard_distance,
    action_unit_signature,
    generate_diverse_counterfactuals,
    select_dpp_subset,
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


def test_diverse_beam_finds_distinct_valid_action_sets() -> None:
    sampler = _OnesSampler()
    disc = _AdditiveDisc()
    factual = np.zeros(3)
    result = generate_diverse_counterfactuals(
        sampler,
        disc,
        factual,
        y_target=1,
        numerical_columns=[0, 1, 2],
        categorical_groups=[],
        config=DiverseBeamSearchConfig(
            n_counterfactuals=3,
            beam_width=3,
            candidate_pool_size=3,
            max_gower_ratio=1.0,
            max_gower_increase=0.0,
        ),
        candidate_quantiles=(0.5,),
        allow_revisits=False,
    )

    assert result.available_count == 3
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
    result = generate_diverse_counterfactuals(
        sampler,
        disc,
        factual,
        y_target=1,
        numerical_columns=[0],
        categorical_groups=[],
        config=DiverseBeamSearchConfig(
            n_counterfactuals=3,
            beam_width=3,
            candidate_pool_size=3,
        ),
        candidate_quantiles=(0.5,),
    )

    assert result.available_count == 1
    np.testing.assert_array_equal(result.counterfactuals, [[1.0]])
    assert result.target_probabilities[0] >= 0.5


def test_dpp_prefers_distinct_action_sets() -> None:
    factual = np.zeros(3)
    rows = np.asarray(
        [
            [0.6, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.6, 0.0],
            [0.0, 0.0, 0.6],
        ]
    )
    selected, logdet = select_dpp_subset(
        rows,
        np.full(len(rows), 0.8),
        factual,
        [0, 1, 2],
        [],
        DiverseBeamSearchConfig(
            n_counterfactuals=3,
            candidate_pool_size=5,
            dpp_action_weight=1.0,
            dpp_gower_quality_weight=0.0,
            dpp_sparsity_quality_weight=0.0,
        ),
    )

    signatures = {
        action_unit_signature(rows[index], factual, [0, 1, 2], []) for index in selected
    }
    assert len(signatures) == 3
    assert logdet is not None


def test_beam_batches_all_numerical_pairs_per_depth() -> None:
    class BatchedSampler(_OnesSampler):
        def __init__(self):
            self.batch_calls = 0

        def sample_candidate_grid_batch(
            self,
            queries,
            columns,
            *,
            quantiles,
            fixed_target,
            confidences=None,
        ):
            del fixed_target
            assert confidences is None
            assert len(queries) == len(columns)
            self.batch_calls += 1
            return np.ones((len(columns), 1, len(quantiles)))

    sampler = BatchedSampler()
    result = generate_diverse_counterfactuals(
        sampler,
        _AdditiveDisc(),
        np.zeros(3),
        y_target=1,
        numerical_columns=[0, 1, 2],
        categorical_groups=[],
        config=DiverseBeamSearchConfig(
            n_counterfactuals=3,
            beam_width=3,
            candidate_pool_size=3,
        ),
        candidate_quantiles=(0.5,),
        allow_revisits=False,
    )

    assert result.available_count == 3
    assert sampler.batch_calls == 2


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
