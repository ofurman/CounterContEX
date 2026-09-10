"""Independent contract tests for proposal decoding, RNG, and accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from experiments.zeroshot_cf.candidate_domains import project_candidate_values
from experiments.zeroshot_cf.methods.countercontex.proposal_policy import (
    DISPOSITION_DUPLICATE,
    DISPOSITION_NO_OP,
    DISPOSITION_UNIQUE,
    NUMERICAL_BIN_COUNT,
    NormalizedWeights,
    ProposalBudget,
    SamplingKey,
    account_projected_rows,
    categorical_truncation,
    equal_mass_bin_truncation,
    keyed_uniform,
    normalize_log_weights,
    sample_equal_mass_bins,
    weighted_resample,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "proposal_sampling"
_REPOSITORY = Path(__file__).parents[3]


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def test_authenticated_e3_witness_preserves_historical_nine_quantiles() -> None:
    witness = _fixture("e3_nine_quantile_witness.json")

    assert witness["quantile_levels"] == pytest.approx(np.arange(0.1, 1.0, 0.1))
    assert witness["raw_values"] == witness["projected_values"]
    assert len(witness["raw_values"]) == 9
    assert witness["provenance"]["source_sha256"] == (
        "d50d476bb5d476f96cd39cf757b59a0a68be5164b12f74a85b502ac90324fb8a"
    )
    assert witness["provenance"]["complete_sha256"] == (
        "49c50de0e41be3d31817e895135fd5c2f4f8afddbfe68b32a2263edf7db1f687"
    )
    assert witness["provenance"]["trace_analysis_sha256"] == (
        "6596636a03da030d13f681e6169b79b4ed9173dc92b55f855e75212131bc0eda"
    )


def test_local_e3_sources_reauthenticate_committed_witness_when_available() -> None:
    witness = _fixture("e3_nine_quantile_witness.json")
    provenance = witness["provenance"]
    source = _REPOSITORY / provenance["source_path"]
    complete_path = _REPOSITORY / provenance["complete_path"]
    trace_analysis = _REPOSITORY / provenance["trace_analysis_path"]
    paths = (source, complete_path, trace_analysis)

    if not any(path.exists() for path in paths):
        pytest.skip("preserved local E3 source bundle is not materialized")
    assert all(path.exists() for path in paths), (
        "E3 source bundle is only partially present"
    )

    assert (
        hashlib.sha256(source.read_bytes()).hexdigest() == provenance["source_sha256"]
    )
    assert (
        hashlib.sha256(complete_path.read_bytes()).hexdigest()
        == provenance["complete_sha256"]
    )
    assert (
        hashlib.sha256(trace_analysis.read_bytes()).hexdigest()
        == provenance["trace_analysis_sha256"]
    )
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    assert complete["files"][source.name] == provenance["source_sha256"]
    payload = json.loads(source.read_text(encoding="utf-8"))
    selector = provenance["source_selector"]
    assert payload["columns"][selector["raw_index"]] == selector["column"]
    assert (
        payload["raw"][selector["raw_index"]][selector["query_index"]]
        == witness["raw_values"]
    )


def test_equal_mass_top_k_and_top_p_match_hand_derived_supports() -> None:
    fixture = _fixture("analytical.json")["equal_mass_bins"]
    scores = np.asarray(fixture["log_densities"])

    top_k = equal_mass_bin_truncation(scores, policy="top-k", k=fixture["top_k"])
    top_p = equal_mass_bin_truncation(scores, policy="top-p", p=fixture["top_p"])

    np.testing.assert_array_equal(top_k.density_order, fixture["density_order"])
    np.testing.assert_array_equal(
        np.flatnonzero(top_k.retained), fixture["top_k_retained"]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(top_p.retained), fixture["top_p_retained"]
    )


def test_default_numerical_partition_is_fixed_at_256_equal_mass_bins() -> None:
    truncation = equal_mass_bin_truncation(
        np.zeros(NUMERICAL_BIN_COUNT), policy="top-p", p=0.9
    )

    assert truncation.bin_count == 256
    assert truncation.retained_count == 231
    np.testing.assert_array_equal(np.flatnonzero(truncation.retained), np.arange(231))


def test_equal_mass_sampling_uses_retained_q_mass_and_open_bin_coordinates() -> None:
    fixture = _fixture("analytical.json")["equal_mass_bins"]
    truncation = equal_mass_bin_truncation(
        np.asarray(fixture["log_densities"]), policy="top-k", k=3
    )

    sample = sample_equal_mass_bins(
        truncation,
        selection_uniforms=np.array([0.0, 0.34, 0.99]),
        within_bin_uniforms=np.array([0.0, 0.5, 1.0]),
    )

    np.testing.assert_array_equal(sample.bin_indices, [1, 4, 7])
    assert np.all(sample.quantiles > 0.0)
    assert np.all(sample.quantiles < 1.0)
    assert sample.quantiles[1] == pytest.approx((4.0 + 0.5) / 8.0)


def test_categorical_truncation_preserves_q_and_canonical_ties() -> None:
    probabilities = np.array([0.35, 0.35, 0.20, 0.10])

    top_k = categorical_truncation(probabilities, policy="top-k", k=2)
    top_p = categorical_truncation(probabilities, policy="top-p", p=0.7)

    np.testing.assert_array_equal(top_k.order, [0, 1, 2, 3])
    np.testing.assert_array_equal(np.flatnonzero(top_k.retained), [0, 1])
    np.testing.assert_allclose(top_k.normalized_probabilities, [0.5, 0.5, 0.0, 0.0])
    np.testing.assert_array_equal(np.flatnonzero(top_p.retained), [0, 1])


def test_pi_beta_uses_one_q_factor_and_beta_zero_recovers_base_mass() -> None:
    fixture = _fixture("analytical.json")["categorical"]
    q = np.asarray(fixture["probabilities"])
    scores = np.asarray(fixture["classifier_probabilities"])

    guided = normalize_log_weights(np.log(q), scores, beta=fixture["beta"])
    control = normalize_log_weights(np.log(q), scores, beta=0.0)

    np.testing.assert_allclose(guided.weights, fixture["expected_pi_beta"])
    assert guided.log_normalizer == pytest.approx(fixture["expected_log_normalizer"])
    assert guided.effective_sample_size == pytest.approx(
        fixture["expected_effective_sample_size"]
    )
    np.testing.assert_allclose(control.weights, q)
    assert control.log_normalizer == pytest.approx(0.0)
    assert control.effective_sample_size == pytest.approx(1.0 / np.square(q).sum())


def test_pool_guidance_handles_zero_and_tiny_scores_without_q_squared() -> None:
    scores = np.array([0.0, 1e-300, 0.5, 1.0])

    guided = normalize_log_weights(np.zeros(4), scores, beta=1.0, epsilon=1e-12)
    control = normalize_log_weights(np.zeros(4), scores, beta=0.0)

    np.testing.assert_allclose(control.weights, np.full(4, 0.25))
    assert np.all(np.isfinite(guided.weights))
    assert guided.weights[0] == pytest.approx(guided.weights[1])
    assert guided.weights[-1] > guided.weights[-2]


def test_weighted_resampling_keeps_repeated_indices_and_raw_budget() -> None:
    selected = weighted_resample(
        np.array([0.1, 0.2, 0.7]),
        np.array([0.05, 0.15, 0.31, 0.99]),
    )

    np.testing.assert_array_equal(selected, [0, 1, 2, 2])


def test_keyed_uniforms_are_stable_under_ordering_and_chunking() -> None:
    base = SamplingKey(
        run_seed=42,
        factual_source_index=17,
        state_depth=2,
        parent_id="parent-3",
        action_unit_id="income",
        draw_index=0,
        stream="q-pool",
    )
    keys = [replace(base, draw_index=index) for index in range(12)]
    witness = _fixture("analytical.json")["stable_rng"]

    direct = {key.draw_index: keyed_uniform(key) for key in keys}
    reordered = {key.draw_index: keyed_uniform(key) for key in reversed(keys)}
    chunked = [keyed_uniform(key) for key in keys[:5]] + [
        keyed_uniform(key) for key in keys[5:]
    ]

    assert direct == reordered
    assert base.canonical_bytes().decode("utf-8") == witness["canonical_bytes"]
    assert hashlib.sha256(base.canonical_bytes()).hexdigest() == witness["sha256"]
    assert direct[0] == witness["expected_uniform"]
    np.testing.assert_array_equal(chunked, [direct[index] for index in range(12)])
    assert all(0.0 < value < 1.0 for value in direct.values())
    assert keyed_uniform(replace(base, action_unit_id="age")) != direct[0]
    assert keyed_uniform(replace(base, stream="resample")) != direct[0]


def test_projection_accounting_preserves_noops_duplicates_onehot_and_immutable() -> (
    None
):
    fixture = _fixture("analytical.json")["projection"]
    accounting = account_projected_rows(
        np.asarray(fixture["factual"]), np.asarray(fixture["projected_rows"])
    )

    np.testing.assert_array_equal(
        accounting.terminal_dispositions, fixture["terminal_dispositions"]
    )
    np.testing.assert_array_equal(
        accounting.unique_index_by_raw, fixture["unique_index_by_raw"]
    )
    assert accounting.budget.raw_count == fixture["raw_count"]
    assert accounting.budget.projected_count == fixture["projected_count"]
    assert (
        accounting.budget.unique_classifier_count == fixture["unique_classifier_count"]
    )
    assert accounting.budget.no_op_count == fixture["no_op_count"]
    assert accounting.budget.duplicate_count == fixture["duplicate_count"]
    np.testing.assert_array_equal(accounting.unique_classifier_rows[:, 3], 7.0)
    np.testing.assert_array_equal(
        accounting.unique_classifier_rows[:, 1:3].sum(axis=1), 1.0
    )


def test_candidate_projection_clips_and_snaps_before_accounting() -> None:
    domains = (
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        {1: np.array([0.0, 0.5, 1.0])},
    )
    projected = project_candidate_values(
        [0, 1, 2], np.array([-0.2, 0.59, 1.2]), domains
    )

    np.testing.assert_allclose(projected, [0.0, 0.5, 1.0])


def test_records_reject_invalid_probabilities_shapes_and_sampling_keys() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        replace(
            SamplingKey(1, 2, 0, "p", "a", 0, "q"),
            factual_source_index=-1,
        )
    with pytest.raises(ValueError, match="finite"):
        normalize_log_weights(np.zeros(2), np.array([0.2, np.nan]), beta=1.0)
    with pytest.raises(ValueError, match="same shape"):
        normalize_log_weights(np.zeros(2), np.array([0.2]), beta=1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        equal_mass_bin_truncation(np.zeros((2, 2)), policy="top-k", k=1)
    with pytest.raises(ValueError, match="effective sample size"):
        NormalizedWeights(np.array([0.5, 0.5]), 0.0, 0.0)
    with pytest.raises(ValueError, match="partition"):
        ProposalBudget(2, 2, 1, 0, 0)


def test_array_records_are_immutable_after_validation() -> None:
    truncation = equal_mass_bin_truncation(np.zeros(8), policy="top-k", k=3)
    accounting = account_projected_rows(np.array([0.0]), np.array([[0.1], [0.1]]))

    with pytest.raises(ValueError, match="read-only"):
        truncation.retained[0] = False
    with pytest.raises(ValueError, match="read-only"):
        accounting.unique_index_by_raw[0] = 4


@pytest.mark.parametrize("target", [0, 1])
def test_target_direction_is_part_of_the_rng_action_identity(target: int) -> None:
    key = SamplingKey(
        run_seed=17,
        factual_source_index=4,
        state_depth=0,
        parent_id="root",
        action_unit_id=f"feature-2:target-{target}",
        draw_index=0,
        stream="q-pool",
    )

    opposite = replace(key, action_unit_id=f"feature-2:target-{1 - target}")
    assert keyed_uniform(key) != keyed_uniform(opposite)


def test_terminal_disposition_constants_are_frozen_by_fixture() -> None:
    assert (DISPOSITION_NO_OP, DISPOSITION_UNIQUE, DISPOSITION_DUPLICATE) == (0, 1, 2)
