"""Frozen cell counts and controlled axes for E11-E13 campaign specs."""

from dataclasses import replace
from pathlib import Path

from experiments.zeroshot_cf.datasets.target_models import (
    DEFAULT_TARGET_MODEL_REGISTRY,
)
from experiments.zeroshot_cf.methods.registry import DEFAULT_METHOD_REGISTRY
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config
from experiments.zeroshot_cf.orchestration.spec import MethodSpec

_CONFIG = Path(__file__).parents[1] / "configs"
_LAUNCHER = Path(__file__).parents[1] / "athena" / "run_distribution_campaign.sh"
_EXPECTED = {
    "matrices/campaign_e11_distribution_sampling.yaml": 144,
    "matrices/campaign_e11_distribution_sampling_confirmation.yaml": 240,
    "matrices/campaign_e13_classifier_guidance.yaml": 60,
    "matrices/campaign_e13_classifier_guidance_confirmation.yaml": 100,
    "matrices/campaign_e11_e13_k3_replication.yaml": 48,
    "diagnostics/campaign_e12_pushforward.yaml": 36,
    "diagnostics/campaign_e12_pushforward_confirmation.yaml": 12,
}


def _load(relative: str):
    return load_matrix_config(_CONFIG / relative)


def test_distribution_campaigns_resolve_exact_unique_cells() -> None:
    all_cells: list[str] = []
    for relative, expected in _EXPECTED.items():
        config = _load(relative)
        assert len(config.runs) == expected
        assert len(set(config.expected_cells)) == expected
        assert not config.execution.legacy_export
        all_cells.extend(config.expected_cells)
        for method in {
            (
                run.method.name,
                run.method.variant,
                repr(dict(run.method.params)),
            ): run.method
            for run in config.runs
        }.values():
            DEFAULT_METHOD_REGISTRY.create(
                method.name, method.params, variant=method.variant
            )
        for target in {
            (run.target_model.name, repr(dict(run.target_model.params))): (
                run.target_model
            )
            for run in config.runs
        }.values():
            DEFAULT_TARGET_MODEL_REGISTRY.resolve(target.name, target.params)
    assert len(set(all_cells)) == sum(_EXPECTED.values())


def test_pilot_and_confirmation_partitions_and_seeds_are_disjoint_protocols() -> None:
    for experiment in ("e11_distribution_sampling", "e13_classifier_guidance"):
        pilot = _load(f"matrices/campaign_{experiment}.yaml")
        confirmation = _load(
            f"matrices/campaign_{experiment}_confirmation.yaml"
        )
        assert {run.protocol.factual_partition for run in pilot.runs} == {
            "validation"
        }
        assert {run.protocol.factual_partition for run in confirmation.runs} == {
            "test"
        }
        assert {run.protocol.test_selection for run in pilot.runs} == {
            "target_stratified"
        }
        assert {run.protocol.max_test for run in pilot.runs} == {20}
        assert {run.protocol.max_test for run in confirmation.runs} == {100}
        assert {run.seed for run in pilot.runs} == {17, 42, 101}
        assert {run.seed for run in confirmation.runs} == {
            17,
            42,
            101,
            202,
            303,
        }
        assert set(pilot.expected_cells).isdisjoint(confirmation.expected_cells)


def test_e11_changes_one_declared_decoder_axis_at_a_time() -> None:
    config = _load("matrices/campaign_e11_distribution_sampling.yaml")
    searches = {
        repr(dict(run.method.params["search"])): dict(
            run.method.params["search"]
        )
        for run in config.runs
    }.values()
    assert len(searches) == 12

    numerical = [
        search
        for search in searches
        if search["categorical_decoder"] == "iid"
        and search["categorical_proposal_budget"] == 9
    ]
    assert {
        (search["numerical_decoder"], search["numerical_proposal_budget"])
        for search in numerical
    } == {
        ("mode", 1),
        ("grid", 9),
        ("iid", 9),
        ("top-k", 9),
        ("top-p", 9),
        ("grid", 49),
        ("iid", 49),
        ("top-k", 49),
        ("top-p", 49),
    }
    categorical = [
        search
        for search in searches
        if search["numerical_decoder"] == "iid"
        and search["numerical_proposal_budget"] == 9
    ]
    assert {
        (search["categorical_decoder"], search["categorical_proposal_budget"])
        for search in categorical
    } == {("iid", 9), ("greedy", 1), ("top-k", 9), ("top-p", 9)}


def test_e13_beta_arms_share_m_b_and_q_policy() -> None:
    config = _load("matrices/campaign_e13_classifier_guidance.yaml")
    searches = {
        repr(dict(run.method.params["search"])): dict(
            run.method.params["search"]
        )
        for run in config.runs
    }.values()
    pi_beta = [s for s in searches if s["guidance_policy"] == "pi-beta"]
    assert {s["guidance_beta"] for s in pi_beta} == {0.0, 0.5, 1.0, 2.0}
    for search in searches:
        assert search["guidance_pool_size"] == 64
        assert search["numerical_proposal_budget"] == 9
        assert search["categorical_proposal_budget"] == 9
        assert search["numerical_decoder"] == "iid"
        assert search["categorical_decoder"] == "iid"
    comparator = [
        s for s in searches if s["guidance_policy"] == "classifier-top-b"
    ]
    assert len(comparator) == 1


def test_k3_replication_and_e12_classifier_axis_are_explicit() -> None:
    replication = _load("matrices/campaign_e11_e13_k3_replication.yaml")
    assert {run.method.n_counterfactuals for run in replication.runs} == {3}
    assert {run.protocol.max_test for run in replication.runs} == {50}

    diagnostic = _load("diagnostics/campaign_e12_pushforward.yaml")
    assert {run.target_model.name for run in diagnostic.runs} == {
        "retained_logistic_regression",
        "retained_mlp",
        "retained_xgboost",
    }
    assert {run.protocol.factual_partition for run in diagnostic.runs} == {
        "validation"
    }
    confirmation = _load(
        "diagnostics/campaign_e12_pushforward_confirmation.yaml"
    )
    assert {run.target_model.name for run in confirmation.runs} == {
        "retained_logistic_regression",
        "retained_mlp",
        "retained_xgboost",
    }
    assert {run.protocol.factual_partition for run in confirmation.runs} == {"test"}
    assert {run.protocol.max_test for run in confirmation.runs} == {100}
    assert {run.seed for run in confirmation.runs} == {42}


def test_campaign_launcher_defaults_to_dry_run_and_separates_authority() -> None:
    launchers = (
        _LAUNCHER,
        _LAUNCHER.with_name("run_e12_diagnostic_campaign.sh"),
    )
    for launcher in launchers:
        script = launcher.read_text()
        assert 'CAMPAIGN_MODE="${CAMPAIGN_MODE:-dry-run}"' in script
        assert "approved_matrix_sha256=$MATRIX_SHA" in script
        assert "scope=pilot" in script
        assert "scope=confirmation" in script
        assert "refusing non-empty output root" in script
        assert "require_checkpoints" in script
    assert "orchestration.campaign_inventory" in _LAUNCHER.read_text()
    e12_script = launchers[1].read_text()
    assert "--mc-bank-count" not in e12_script
    assert "--mc-bank-size" not in e12_script
    assert "status=starting" in e12_script
    assert "status=complete" in e12_script


def test_e12_policy_is_complete_and_every_field_changes_identity() -> None:
    for relative, expected_banks in (
        ("diagnostics/campaign_e12_pushforward.yaml", 5),
        ("diagnostics/campaign_e12_pushforward_confirmation.yaml", 0),
    ):
        run = _load(relative).runs[0]
        policy = dict(run.protocol.params["proposal_pushforward"])
        assert policy == {
            "diagnostic_version": "proposal-diagnostic-v1",
            "numerical_integration_points": 256,
            "categorical_policy": "exact-support",
            "mc_bank_count": expected_banks,
            "mc_bank_size": 64,
            "best_of_b_budgets": [9, 49],
        }
        mutations = {
            "diagnostic_version": "mutation-v2",
            "numerical_integration_points": 255,
            "categorical_policy": "mutation",
            "mc_bank_count": expected_banks + 1,
            "mc_bank_size": 63,
            "best_of_b_budgets": (8, 49),
        }
        for field, value in mutations.items():
            changed = {**policy, field: value}
            protocol = replace(
                run.protocol,
                params={"proposal_pushforward": changed},
            )
            assert replace(run, protocol=protocol).cell_id != run.cell_id, field


def test_every_distribution_policy_field_changes_scientific_cell_identity() -> None:
    run = _load("matrices/campaign_e13_classifier_guidance.yaml").runs[0]
    base_params = dict(run.method.params)
    base_search = dict(base_params["search"])
    mutations = {
        "numerical_decoder": "top-k",
        "numerical_proposal_budget": 10,
        "categorical_decoder": "top-p",
        "categorical_proposal_budget": 10,
        "truncation_k": 4,
        "truncation_p": 0.8,
        "strict_proposal_budget": False,
        "proposal_rng_scheme": "mutation-witness",
        "proposal_accounting_policy": "mutation-witness",
        "guidance_policy": "classifier-top-b",
        "guidance_beta": 0.5,
        "guidance_pool_size": 65,
        "guidance_epsilon": 1e-9,
        "guidance_with_replacement": False,
        "guidance_score_cache": False,
        "guidance_resampling_policy": "mutation-witness",
        "guidance_common_rng_scheme": "mutation-witness",
    }
    for field, value in mutations.items():
        changed_search = {**base_search, field: value}
        changed_method = MethodSpec(
            run.method.name,
            run.method.variant,
            {**base_params, "search": changed_search},
            run.method.n_counterfactuals,
        )
        assert replace(run, method=changed_method).cell_id != run.cell_id, field
