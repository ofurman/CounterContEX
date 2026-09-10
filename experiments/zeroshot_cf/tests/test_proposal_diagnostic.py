"""Independent witnesses for the read-only proposal diagnostic."""

import json
from types import MappingProxyType

import numpy as np
from experiments.zeroshot_cf.diagnostics.proposal_backends import (
    SearchTrace,
    compare_candidates,
    serial,
)
from experiments.zeroshot_cf.diverse_search import (
    DiverseBeamSearchConfig,
    generate_diverse_counterfactuals,
)


def test_trace_json_preserves_nested_frozen_diagnostics() -> None:
    assert json.loads(
        json.dumps(serial({"history": (MappingProxyType({"p": np.float64(0.7)}),)}))
    ) == {"history": [{"p": 0.7}]}


def test_candidate_comparison_separates_rank_set_and_unavailable() -> None:
    a = np.array([[[0.1], [0.9]], [[np.nan], [np.nan]]])
    b = np.array([[[0.9], [0.1]], [[np.nan], [np.nan]]])
    available = np.array([[True, True], [False, False]])
    rows = compare_candidates(a, available, b, available)
    assert rows[0]["rank_exact_count"] == 0
    assert rows[0]["set_exact"] is True
    assert rows[0]["matched_exact_count"] == 2
    assert rows[1]["set_exact"] is None
    assert rows[1]["joint_available_count"] == 0


def test_candidate_comparison_tolerance_is_not_exact_equality() -> None:
    a = np.array([[[0.1], [0.9]]])
    b = a + 1e-10
    mask = np.ones((1, 2), dtype=bool)
    row = compare_candidates(a, mask, b, mask)[0]
    assert row["set_exact"] is False
    assert row["set_close"] is True


def test_trace_does_not_change_beam_output_or_add_sampler_calls() -> None:
    class Sampler:
        calls = 0

        def sample_candidate_grid_batch(self, rows, columns, **kwargs):
            self.calls += 1
            return np.ones((len(columns), 1, 1))

    class Oracle:
        classes_ = np.array([0, 1])

        def predict_proba(self, rows):
            p = np.clip(0.1 + 0.3 * np.asarray(rows).sum(axis=-1), 0, 1)
            return np.column_stack([1 - p, p])

    def run(sampler):
        return generate_diverse_counterfactuals(
            sampler,
            Oracle(),
            np.zeros(3),
            1,
            [0, 1, 2],
            [],
            config=DiverseBeamSearchConfig(n_counterfactuals=3, candidate_pool_size=3),
            candidate_quantiles=[0.5],
        )

    plain = Sampler()
    expected = run(plain)
    traced = Sampler()
    with SearchTrace() as trace:
        actual = run(traced)
    np.testing.assert_array_equal(actual.counterfactuals, expected.counterfactuals)
    assert actual.histories == expected.histories
    assert plain.calls == traced.calls
    assert any(event["stage"] == "numerical_trials" for event in trace.events)
    assert any(event["stage"] == "selection" for event in trace.events)
