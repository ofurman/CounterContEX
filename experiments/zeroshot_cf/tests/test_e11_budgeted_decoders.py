"""Scientific-contract gates for strict E11 proposal decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.generator import TabICLGeneratorInputs
from experiments.zeroshot_cf.methods.countercontex.backends.base import (
    CategoryProposals,
    NumericalDistribution,
    ProposalCapabilities,
)
from experiments.zeroshot_cf.methods.countercontex.config import (
    CounterContExConfig,
    CounterContExFoundationConfig,
    CounterContExSearchConfig,
)
from experiments.zeroshot_cf.methods.countercontex.method import adapt_generator_result
from experiments.zeroshot_cf.methods.countercontex.search import (
    _SessionSampler,
    generate_with_backend,
)


class _Oracle:
    classes_ = np.array([0, 1])

    def __init__(self):
        self.rows: list[np.ndarray] = []

    def predict_proba(self, rows):
        matrix = np.asarray(rows)
        self.rows.extend(row.copy() for row in matrix)
        probabilities = np.clip(matrix[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - probabilities, probabilities))


class _SumOracle(_Oracle):
    def predict_proba(self, rows):
        matrix = np.asarray(rows)
        self.rows.extend(row.copy() for row in matrix)
        probabilities = np.clip(matrix.sum(axis=1), 0.0, 1.0)
        return np.column_stack((1.0 - probabilities, probabilities))


@dataclass(frozen=True)
class _DistributionSession:
    confidence_anchors = None
    diagnostics = {}

    def propose_numerical(
        self, rows, columns, *, quantiles, confidence, temperature
    ):
        del rows, confidence, temperature
        if quantiles is None:
            return np.full(len(columns), 0.9)
        return np.broadcast_to(
            np.asarray(quantiles, dtype=np.float64),
            (len(columns), len(quantiles)),
        ).copy()

    def propose_numerical_batch(
        self, rows, columns, *, quantiles, confidences, temperature
    ):
        del rows, confidences, temperature
        if quantiles is None:
            return np.full(len(columns), 0.9)
        return np.broadcast_to(
            np.asarray(quantiles, dtype=np.float64),
            (len(columns), 1, len(quantiles)),
        ).copy()

    def numerical_distribution_batch(
        self, rows, columns, *, quantiles, confidences
    ):
        del rows, confidences
        q = np.asarray(quantiles, dtype=np.float64)
        values = np.broadcast_to(q, (len(columns), 1, len(q))).copy()
        # A peak around .75 makes truncation non-trivial while deterministic.
        log_probabilities = -np.square(values - 0.75)
        return NumericalDistribution(q, values, log_probabilities)

    def categorical_distribution(self, row, group, *, confidence):
        del row, confidence
        support = np.arange(len(group.columns), dtype=np.int64)
        probabilities = np.arange(1, len(support) + 1, dtype=np.float64)
        probabilities /= probabilities.sum()
        return CategoryProposals(support, probabilities)

    def score_joint(self, rows, target):
        raise AssertionError((rows, target))


@dataclass(frozen=True)
class _Backend:
    backend_id = "fake"
    capabilities = ProposalCapabilities(
        numerical_distribution=True,
        categorical_distribution=True,
    )

    def for_factual(self, factual, target, *, seed):
        del factual, target, seed
        return _DistributionSession()


@dataclass(frozen=True)
class _NoOpSession(_DistributionSession):
    def numerical_distribution_batch(
        self, rows, columns, *, quantiles, confidences
    ):
        del rows, confidences
        q = np.asarray(quantiles, dtype=np.float64)
        shape = (len(columns), 1, len(q))
        return NumericalDistribution(q, np.full(shape, 0.1), np.zeros(shape))

    def categorical_distribution(self, row, group, *, confidence):
        del row, confidence
        return CategoryProposals(
            np.arange(len(group.columns), dtype=np.int64),
            np.array([1.0, *([0.0] * (len(group.columns) - 1))]),
        )


@dataclass(frozen=True)
class _ModeSession(_DistributionSession):
    def propose_numerical(
        self, rows, columns, *, quantiles, confidence, temperature
    ):
        del rows, quantiles, confidence, temperature
        return np.full(len(columns), 0.5)


@dataclass(frozen=True)
class _NoOpBackend(_Backend):
    def for_factual(self, factual, target, *, seed):
        del factual, target, seed
        return _NoOpSession()


@dataclass(frozen=True)
class _ModeBackend(_Backend):
    def for_factual(self, factual, target, *, seed):
        del factual, target, seed
        return _ModeSession()


def _strict_search(**changes) -> CounterContExSearchConfig:
    values = {
        "numerical_decoder": "iid",
        "numerical_proposal_budget": 9,
        "categorical_decoder": "iid",
        "categorical_proposal_budget": 9,
        "strict_proposal_budget": True,
    }
    values.update(changes)
    return CounterContExSearchConfig(**values)


def _strict_config(**changes) -> CounterContExConfig:
    return CounterContExConfig(
        search=_strict_search(**changes),
        foundation=CounterContExFoundationConfig(
            backend="fake", n_estimators=1, temperature=1.0
        ),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"numerical_proposal_budget": None},
        {"categorical_proposal_budget": None},
        {"numerical_decoder": "mode"},
        {"categorical_decoder": "greedy"},
        {"numerical_decoder": "grid"},
        {"numerical_decoder": "top-k", "truncation_k": 257},
        {"categorical_decoder": "top-p", "truncation_p": 0.0},
    ],
)
def test_strict_configuration_rejects_inapplicable_budget_combinations(changes):
    with pytest.raises(ValueError):
        _strict_search(**changes)


def test_full_distribution_sampling_requires_unit_temperature() -> None:
    with pytest.raises(ValueError, match="temperature=1.0"):
        CounterContExConfig(
            search=_strict_search(),
            foundation=CounterContExFoundationConfig(
                backend="fake", temperature=0.5
            ),
        )


def test_iid_decoding_is_invariant_to_action_order_and_chunking() -> None:
    config = _strict_config()
    row = np.array([0.2, 0.4])
    together = _SessionSampler(
        _DistributionSession(), config=config, seed=42, factual_source_index=7
    ).sample_budgeted_candidates(row, [0, 1], state_depth=2)
    reordered = _SessionSampler(
        _DistributionSession(), config=config, seed=42, factual_source_index=7
    ).sample_budgeted_candidates(row, [1, 0], state_depth=2)
    first = _SessionSampler(
        _DistributionSession(), config=config, seed=42, factual_source_index=7
    ).sample_budgeted_candidates(row, [0], state_depth=2)
    second = _SessionSampler(
        _DistributionSession(), config=config, seed=42, factual_source_index=7
    ).sample_budgeted_candidates(row, [1], state_depth=2)

    np.testing.assert_array_equal(together.values[0], reordered.values[1])
    np.testing.assert_array_equal(together.values[1], reordered.values[0])
    np.testing.assert_array_equal(together.values[0], first.values[0])
    np.testing.assert_array_equal(together.values[1], second.values[0])
    np.testing.assert_array_equal(together.quantiles[0], first.quantiles[0])

    forward_accounting = _SessionSampler(
        _DistributionSession(), config=config, seed=42, factual_source_index=7
    )
    reverse_accounting = _SessionSampler(
        _DistributionSession(), config=config, seed=42, factual_source_index=7
    )
    action_rows = {
        "0": np.array([[0.2, 0.4], [0.8, 0.4], [0.8, 0.4]]),
        "1": np.array([[0.2, 0.4], [0.2, 0.6], [0.2, 0.7]]),
    }
    for action in ("0", "1"):
        forward_accounting.record_projected_rows(
            row, action_rows[action], action_unit_id=action, state_depth=2
        )
    for action in ("1", "0"):
        reverse_accounting.record_projected_rows(
            row, action_rows[action], action_unit_id=action, state_depth=2
        )
    scored_rows = np.array([[0.8, 0.4], [0.2, 0.6], [0.2, 0.7]])
    scores = np.array([0.8, 0.2, 0.2])
    forward_accounting.record_classifier_rows(scored_rows, scores)
    reverse_accounting.record_classifier_rows(scored_rows[::-1], scores[::-1])

    assert forward_accounting.accounting.raw_count == 6
    assert reverse_accounting.accounting.raw_count == 6
    np.testing.assert_array_equal(
        forward_accounting.accounting.terminal_arrays()[0],
        reverse_accounting.accounting.terminal_arrays()[0],
    )

    opposite_target = _SessionSampler(
        _DistributionSession(),
        config=config,
        seed=42,
        factual_source_index=7,
        target=1,
    ).sample_budgeted_candidates(row, [0], state_depth=2)
    assert not np.array_equal(together.quantiles[0], opposite_target.quantiles[0])


def test_strict_search_counts_no_ops_duplicates_and_unique_classifier_rows() -> None:
    group = OneHotActionGroup("segment", (1, 2))
    oracle = _Oracle()
    result = generate_with_backend(
        TabICLGeneratorInputs(
            factuals=np.array([[0.1, 1.0, 0.0]]),
            targets=np.array([1]),
            numerical_columns=(0,),
            categorical_groups=(group,),
            factual_source_indices=(7,),
        ),
        discriminator=oracle,
        config=_strict_config(),
        backend=_Backend(),
        seed=42,
        n_counterfactuals=1,
    )

    diagnostics = result.diagnostics
    assert diagnostics.proposal_policy == "iid/iid"
    assert diagnostics.proposal_raw_count_per_point.tolist() == [18]
    assert diagnostics.proposal_projected_count_per_point.tolist() == [18]
    assert (
        diagnostics.proposal_unique_count_per_point
        + diagnostics.proposal_no_op_count_per_point
        + diagnostics.proposal_duplicate_count_per_point
    ).tolist() == [18]
    assert diagnostics.proposal_classifier_rows_per_point.tolist() == [
        1 + diagnostics.proposal_unique_count_per_point[0]
    ]
    assert len(diagnostics.proposal_terminal_dispositions) == 18
    assert len(diagnostics.proposal_unique_index_by_raw) == 18
    assert len(oracle.rows) == diagnostics.proposal_classifier_rows_per_point[0]
    # The numerical q draw reaches validity in the first expansion; no fallback
    # may enumerate additional categories after the declared 9 raw draws.
    assert diagnostics.validity_steps_per_point == (1,)
    adapted = adapt_generator_result(result, seed=42, proposal_backend="fake")
    assert adapted.run_diagnostics["proposal_policy"] == "iid/iid"
    assert adapted.artifacts["method.proposal_raw_count"].tolist() == [18]
    assert adapted.artifacts["method.proposal_classifier_rows"].tolist() == [
        1 + diagnostics.proposal_unique_count_per_point[0]
    ]


def test_grid_and_mode_references_have_exact_declared_budget() -> None:
    grid = _strict_search(
        numerical_decoder="grid",
        numerical_proposal_budget=9,
        candidate_quantiles=tuple(np.arange(1, 10) / 10),
    )
    mode = _strict_search(
        numerical_decoder="mode",
        numerical_proposal_budget=1,
    )

    assert grid.candidate_quantiles == tuple(np.arange(1, 10) / 10)
    assert mode.numerical_proposal_budget == 1


def test_strict_search_never_refills_after_no_ops_exhaust_the_budget() -> None:
    result = generate_with_backend(
        TabICLGeneratorInputs(
            factuals=np.array([[0.1, 1.0, 0.0]]),
            targets=np.array([1]),
            numerical_columns=(0,),
            categorical_groups=(OneHotActionGroup("segment", (1, 2)),),
            factual_source_indices=(7,),
        ),
        discriminator=_Oracle(),
        config=_strict_config(),
        backend=_NoOpBackend(),
        seed=42,
        n_counterfactuals=1,
    )

    diagnostics = result.diagnostics
    assert diagnostics.proposal_raw_count_per_point.tolist() == [18]
    assert diagnostics.proposal_no_op_count_per_point.tolist() == [18]
    assert diagnostics.proposal_unique_count_per_point.tolist() == [0]
    assert diagnostics.proposal_classifier_rows_per_point.tolist() == [1]
    np.testing.assert_array_equal(result.counterfactuals[0], result.factuals[0])


def test_full_strict_result_is_invariant_to_action_unit_order_on_ties() -> None:
    config = _strict_config(
        numerical_decoder="mode",
        numerical_proposal_budget=1,
        categorical_decoder="greedy",
        categorical_proposal_budget=1,
    )

    def run(columns):
        return generate_with_backend(
            TabICLGeneratorInputs(
                factuals=np.array([[0.0, 0.0]]),
                targets=np.array([1]),
                numerical_columns=columns,
                categorical_groups=(),
                factual_source_indices=(11,),
            ),
            discriminator=_SumOracle(),
            config=config,
            backend=_ModeBackend(),
            seed=42,
            n_counterfactuals=1,
        )

    forward = run((0, 1))
    reverse = run((1, 0))

    np.testing.assert_array_equal(forward.counterfactuals, reverse.counterfactuals)
    np.testing.assert_array_equal(
        forward.diagnostics.proposal_terminal_dispositions,
        reverse.diagnostics.proposal_terminal_dispositions,
    )


def test_factual_source_ids_make_reordered_batches_reproducible() -> None:
    config = _strict_config()

    def run(factuals, source_indices):
        return generate_with_backend(
            TabICLGeneratorInputs(
                factuals=np.asarray(factuals, dtype=np.float64),
                targets=np.ones(len(factuals), dtype=int),
                numerical_columns=(0,),
                categorical_groups=(),
                factual_source_indices=tuple(source_indices),
            ),
            discriminator=_Oracle(),
            config=config,
            backend=_Backend(),
            seed=42,
            n_counterfactuals=1,
        )

    forward = run([[0.1], [0.2]], [7, 8])
    reverse = run([[0.2], [0.1]], [8, 7])

    np.testing.assert_array_equal(
        forward.counterfactuals, reverse.counterfactuals[::-1]
    )
    np.testing.assert_array_equal(
        forward.diagnostics.proposal_raw_count_per_point,
        reverse.diagnostics.proposal_raw_count_per_point[::-1],
    )


def test_top_k_trace_preserves_support_draws_and_reverse_links() -> None:
    result = generate_with_backend(
        TabICLGeneratorInputs(
            factuals=np.array([[0.1]]),
            targets=np.array([1]),
            numerical_columns=(0,),
            categorical_groups=(),
            factual_source_indices=(7,),
        ),
        discriminator=_Oracle(),
        config=_strict_config(numerical_decoder="top-k"),
        backend=_Backend(),
        seed=42,
        n_counterfactuals=1,
    )
    trace = result.diagnostics.proposal_trace_arrays

    np.testing.assert_array_equal(trace["support_offsets"], [0, 256])
    np.testing.assert_array_equal(trace["selected_offsets"], [0, 9])
    assert np.count_nonzero(trace["support_retained"]) == 5
    np.testing.assert_array_equal(np.sort(trace["support_order"]), np.arange(256))
    assert np.all(np.isfinite(trace["within_bin_uniforms"]))
    np.testing.assert_allclose(
        np.exp(trace["selected_log_probabilities"]),
        trace["selected_probabilities"],
    )
    assert len(result.diagnostics.proposal_unique_index_by_raw) == 9


def test_sparse_categorical_support_is_decoded_without_fabricating_classes() -> None:
    class SparseSession(_DistributionSession):
        def categorical_distribution(self, row, group, *, confidence):
            del row, group, confidence
            return CategoryProposals(
                np.array([1], dtype=np.int64), np.array([1.0])
            )

    class SparseBackend(_Backend):
        def for_factual(self, factual, target, *, seed):
            del factual, target, seed
            return SparseSession()

    result = generate_with_backend(
        TabICLGeneratorInputs(
            factuals=np.array([[0.0, 1.0, 0.0]]),
            targets=np.array([1]),
            numerical_columns=(),
            categorical_groups=(OneHotActionGroup("segment", (0, 1, 2)),),
            factual_source_indices=(7,),
        ),
        discriminator=_Oracle(),
        config=_strict_config(
            categorical_decoder="greedy", categorical_proposal_budget=1
        ),
        backend=SparseBackend(),
        seed=42,
        n_counterfactuals=1,
    )

    trace = result.diagnostics.proposal_trace_arrays
    np.testing.assert_array_equal(trace["support_values"], [1.0])
    np.testing.assert_array_equal(trace["selected_values"], [1.0])


def test_reverse_map_dereferences_persisted_classifier_rows_and_scores() -> None:
    result = generate_with_backend(
        TabICLGeneratorInputs(
            factuals=np.array([[0.1]]),
            targets=np.array([1]),
            numerical_columns=(0,),
            categorical_groups=(),
            factual_source_indices=(7,),
        ),
        discriminator=_Oracle(),
        config=_strict_config(),
        backend=_Backend(),
        seed=42,
        n_counterfactuals=1,
    )
    diagnostics = result.diagnostics

    np.testing.assert_array_equal(diagnostics.proposal_raw_offsets, [0, 9])
    np.testing.assert_array_equal(
        diagnostics.proposal_classifier_row_offsets,
        [0, diagnostics.proposal_classifier_rows_per_point[0]],
    )
    linked = diagnostics.proposal_unique_index_by_raw
    assert np.all(linked >= 0)
    np.testing.assert_allclose(
        diagnostics.proposal_scored_target_probabilities,
        diagnostics.proposal_scored_rows[:, 0],
    )
    np.testing.assert_allclose(
        diagnostics.proposal_scored_target_probabilities[linked],
        diagnostics.proposal_scored_rows[linked, 0],
    )


def test_k_greater_than_one_reconciles_raw_dispositions_with_scored_rows() -> None:
    config = _strict_config(
        numerical_decoder="mode",
        numerical_proposal_budget=1,
        categorical_decoder="greedy",
        categorical_proposal_budget=1,
    )
    result = generate_with_backend(
        TabICLGeneratorInputs(
            factuals=np.array([[0.0, 0.0]]),
            targets=np.array([1]),
            numerical_columns=(0, 1),
            categorical_groups=(),
            factual_source_indices=(11,),
        ),
        discriminator=_SumOracle(),
        config=config,
        backend=_ModeBackend(),
        seed=42,
        n_counterfactuals=2,
    )
    diagnostics = result.diagnostics

    assert diagnostics.proposal_classifier_rows_per_point.tolist() == [
        1 + diagnostics.proposal_unique_count_per_point[0]
    ]
    assert (
        diagnostics.proposal_unique_count_per_point
        + diagnostics.proposal_no_op_count_per_point
        + diagnostics.proposal_duplicate_count_per_point
    ).tolist() == diagnostics.proposal_raw_count_per_point.tolist()
