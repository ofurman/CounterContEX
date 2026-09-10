"""Scientific-contract gates for common-pool classifier guidance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.generator import TabICLGeneratorInputs
from experiments.zeroshot_cf.methods.countercontex.backends.base import (
    CategoryProposals,
)
from experiments.zeroshot_cf.methods.countercontex.config import (
    CounterContExSearchConfig,
)
from experiments.zeroshot_cf.methods.countercontex.search import (
    _SessionSampler,
    generate_with_backend,
)
from experiments.zeroshot_cf.tests.test_e11_budgeted_decoders import (
    _Backend,
    _DistributionSession,
    _strict_config,
)


class _ClassifierSpy:
    classes_ = np.array([0, 1])

    def __init__(self) -> None:
        self.batches: list[np.ndarray] = []

    def predict_proba(self, rows):
        matrix = np.asarray(rows, dtype=np.float64)
        self.batches.append(matrix.copy())
        scores = np.clip(matrix[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - scores, scores))


def _guided_config(beta: float, **changes):
    values = {
        "guidance_policy": "pi-beta",
        "guidance_beta": beta,
        "guidance_pool_size": 64,
    }
    values.update(changes)
    return _strict_config(
        numerical_proposal_budget=9,
        categorical_proposal_budget=9,
        **values,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"guidance_policy": "pi-beta", "guidance_beta": -1.0},
        {"guidance_policy": "pi-beta", "guidance_beta": np.inf},
        {"guidance_policy": "pi-beta", "guidance_pool_size": 8},
        {"guidance_policy": "pi-beta", "numerical_decoder": "grid"},
        {
            "guidance_policy": "classifier-top-b",
            "guidance_beta": 1.0,
        },
    ],
)
def test_guidance_configuration_rejects_non_protocol_combinations(changes) -> None:
    values = {
        "numerical_decoder": "iid",
        "numerical_proposal_budget": 9,
        "categorical_decoder": "iid",
        "categorical_proposal_budget": 9,
        "strict_proposal_budget": True,
    }
    values.update(changes)
    with pytest.raises(ValueError):
        CounterContExSearchConfig(**values)


def _run_numerical(beta: float):
    spy = _ClassifierSpy()
    result = generate_with_backend(
        TabICLGeneratorInputs(
            factuals=np.array([[0.1]]),
            targets=np.array([1]),
            numerical_columns=(0,),
            categorical_groups=(),
            factual_source_indices=(7,),
        ),
        discriminator=spy,
        config=_guided_config(beta),
        backend=_Backend(),
        seed=42,
        n_counterfactuals=1,
    )
    return result, spy


def test_beta_zero_recovers_empirical_pool_without_q_squared() -> None:
    result, _ = _run_numerical(0.0)
    guidance = result.diagnostics.proposal_guidance_trace_arrays

    np.testing.assert_allclose(guidance["raw_weights"], np.full(64, 1.0 / 64))
    expected = np.floor(guidance["selection_uniforms"] * 64).astype(np.int64)
    np.testing.assert_array_equal(guidance["selected_pool_indices"], expected)
    assert guidance["effective_sample_size"].tolist() == pytest.approx([64.0])
    assert guidance["log_normalizer"].tolist() == pytest.approx([0.0])


def test_beta_arms_share_pool_and_uniforms_but_change_only_weights_selection() -> None:
    control, _ = _run_numerical(0.0)
    guided, _ = _run_numerical(2.0)
    control_q = control.diagnostics.proposal_trace_arrays
    guided_q = guided.diagnostics.proposal_trace_arrays
    control_g = control.diagnostics.proposal_guidance_trace_arrays
    guided_g = guided.diagnostics.proposal_guidance_trace_arrays

    for key in ("selected_quantiles", "selected_values", "selection_uniforms"):
        np.testing.assert_array_equal(control_q[key], guided_q[key])
    np.testing.assert_array_equal(control_g["pool_id"], guided_g["pool_id"])
    np.testing.assert_array_equal(
        control_g["selection_uniforms"], guided_g["selection_uniforms"]
    )
    assert not np.array_equal(
        control_g["raw_weights"], guided_g["raw_weights"]
    )
    assert not np.array_equal(
        control_g["selected_pool_indices"], guided_g["selected_pool_indices"]
    )


def test_pool_rows_are_scored_once_and_selected_rows_reuse_cache() -> None:
    result, spy = _run_numerical(1.0)
    diagnostics = result.diagnostics

    assert len(spy.batches) == 2  # factual, then unique M-pool rows
    assert diagnostics.proposal_classifier_calls_per_point.tolist() == [2]
    assert len(np.vstack(spy.batches)) == 65
    assert diagnostics.proposal_classifier_rows_per_point.tolist() == [65]
    assert diagnostics.proposal_raw_count_per_point.tolist() == [9]
    assert (
        diagnostics.proposal_classifier_calls_to_first_validity_per_point.tolist()
        == [2]
    )
    selected = diagnostics.history_per_point[0][0]
    pool_index = selected["guidance_pool_index"]
    assert selected["quantile"] == pytest.approx(
        diagnostics.proposal_trace_arrays["selected_quantiles"][pool_index]
    )
    assert result.counterfactuals[0, 0] == pytest.approx(
        diagnostics.proposal_trace_arrays["selected_values"][pool_index]
    )


def test_categorical_common_pool_is_sampled_from_exact_q_support() -> None:
    @dataclass(frozen=True)
    class CategoricalSession(_DistributionSession):
        def categorical_distribution(self, row, group, *, confidence):
            del row, group, confidence
            return CategoryProposals(
                np.array([0, 1], dtype=np.int64), np.array([0.25, 0.75])
            )

    @dataclass(frozen=True)
    class CategoricalBackend(_Backend):
        def for_factual(self, factual, target, *, seed):
            del factual, target, seed
            return CategoricalSession()

    spy = _ClassifierSpy()
    result = generate_with_backend(
        TabICLGeneratorInputs(
                factuals=np.array([[0.0, 1.0]]),
            targets=np.array([1]),
            numerical_columns=(),
            categorical_groups=(OneHotActionGroup("segment", (0, 1)),),
            factual_source_indices=(7,),
        ),
        discriminator=spy,
        config=_guided_config(0.0),
        backend=CategoricalBackend(),
        seed=42,
        n_counterfactuals=1,
    )
    q_trace = result.diagnostics.proposal_trace_arrays
    uniforms = q_trace["selection_uniforms"]
    expected_categories = np.where(uniforms < 0.25, 0.0, 1.0)

    np.testing.assert_array_equal(q_trace["selected_values"], expected_categories)
    np.testing.assert_allclose(
        result.diagnostics.proposal_guidance_trace_arrays["raw_weights"],
        np.full(64, 1.0 / 64),
    )


def test_classifier_top_b_uses_unique_best_pool_rows_without_extra_calls() -> None:
    spy = _ClassifierSpy()
    result = generate_with_backend(
        TabICLGeneratorInputs(
            factuals=np.array([[0.1]]),
            targets=np.array([1]),
            numerical_columns=(0,),
            categorical_groups=(),
            factual_source_indices=(7,),
        ),
        discriminator=spy,
        config=_strict_config(
            numerical_proposal_budget=9,
            categorical_proposal_budget=9,
            guidance_policy="classifier-top-b",
            guidance_pool_size=64,
        ),
        backend=_Backend(),
        seed=42,
        n_counterfactuals=1,
    )
    guidance = result.diagnostics.proposal_guidance_trace_arrays

    assert guidance["policy"].tolist() == ["classifier-top-b"]
    assert len(guidance["selected_pool_indices"]) == 9
    selected_scores = guidance["raw_target_probabilities"][
        guidance["selected_pool_indices"]
    ]
    assert np.all(selected_scores[:-1] >= selected_scores[1:])
    assert len(spy.batches) == 2
    assert result.diagnostics.proposal_classifier_calls_per_point.tolist() == [2]


def test_divergent_states_receive_distinct_reproducible_pool_keys() -> None:
    def pool_id(row: np.ndarray) -> str:
        sampler = _SessionSampler(
            _DistributionSession(),
            config=_guided_config(1.0),
            seed=42,
            factual_source_index=7,
            target=1,
            discriminator=_ClassifierSpy(),
        )
        decoded = sampler.sample_budgeted_candidates(row, [1], state_depth=1)
        pool = np.repeat(row.reshape(1, -1), 64, axis=0)
        pool[:, 1] = decoded.values[0]
        sampler.guide_projected_rows(
            row,
            pool,
            action_unit_id="1",
            state_depth=1,
            proposal_budget=9,
        )
        return sampler.accounting.guidance_traces[0].pool_id

    first = np.array([0.1, 0.0])
    second = np.array([0.2, 0.0])
    assert pool_id(first) == pool_id(first.copy())
    assert pool_id(first) != pool_id(second)


def test_single_support_no_op_pool_consumes_budget_without_padding_or_rescore() -> None:
    @dataclass(frozen=True)
    class SingleSupportSession(_DistributionSession):
        def categorical_distribution(self, row, group, *, confidence):
            del row, group, confidence
            return CategoryProposals(
                np.array([1], dtype=np.int64), np.array([1.0])
            )

    @dataclass(frozen=True)
    class SingleSupportBackend(_Backend):
        def for_factual(self, factual, target, *, seed):
            del factual, target, seed
            return SingleSupportSession()

    spy = _ClassifierSpy()
    result = generate_with_backend(
        TabICLGeneratorInputs(
            factuals=np.array([[0.0, 1.0]]),
            targets=np.array([1]),
            numerical_columns=(),
            categorical_groups=(OneHotActionGroup("segment", (0, 1)),),
            factual_source_indices=(7,),
        ),
        discriminator=spy,
        config=_guided_config(2.0),
        backend=SingleSupportBackend(),
        seed=42,
        n_counterfactuals=1,
    )
    diagnostics = result.diagnostics

    assert diagnostics.proposal_raw_count_per_point.tolist() == [9]
    assert diagnostics.proposal_no_op_count_per_point.tolist() == [9]
    assert diagnostics.proposal_unique_count_per_point.tolist() == [0]
    assert diagnostics.proposal_classifier_calls_per_point.tolist() == [1]
    assert diagnostics.proposal_guidance_trace_arrays[
        "effective_sample_size"
    ].tolist() == pytest.approx([64.0])
