"""Frozen paper-campaign matrix contracts."""

from pathlib import Path

from experiments.zeroshot_cf.datasets.target_models import (
    DEFAULT_TARGET_MODEL_REGISTRY,
)
from experiments.zeroshot_cf.evaluation import METRIC_SCHEMA_VERSION
from experiments.zeroshot_cf.methods.registry import DEFAULT_METHOD_REGISTRY
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config

_ROOT = Path(__file__).parents[1] / "configs" / "matrices"
_DGX = Path(__file__).parents[1] / "dgx"
_EXPECTED = {
    "campaign_e1_main.yaml": 540,
    "campaign_e2_diverse.yaml": 60,
    "campaign_e3_backend.yaml": 36,
    "campaign_e4_confidence.yaml": 200,
    "campaign_e5_search.yaml": 210,
    "campaign_e6_context.yaml": 240,
    "campaign_e7_cost.yaml": 80,
    "campaign_e9_fmswap.yaml": 12,
    "campaign_e10_headline.yaml": 6,
}


def test_campaign_matrices_have_frozen_counts_and_shared_protocol():
    all_cell_ids = []
    for filename, expected in _EXPECTED.items():
        config = load_matrix_config(_ROOT / filename)
        assert len(config.runs) == expected
        assert len(set(config.expected_cells)) == expected
        all_cell_ids.extend(config.expected_cells)
        assert not config.execution.legacy_export
        assert all(
            run.evaluation.metric_version == METRIC_SCHEMA_VERSION
            for run in config.runs
        )
        if filename != "campaign_e10_headline.yaml":
            assert {run.seed for run in config.runs} <= {17, 42, 101, 202, 303}
        if filename not in {
            "campaign_e5_search.yaml",
            "campaign_e6_context.yaml",
            "campaign_e10_headline.yaml",
        }:
            assert all(run.protocol.max_test == 250 for run in config.runs)

    assert len(all_cell_ids) == 1384
    assert len(set(all_cell_ids)) == len(all_cell_ids)


def test_e1_uses_classifier_axis_and_k1_for_every_method():
    config = load_matrix_config(_ROOT / "campaign_e1_main.yaml")

    assert {run.target_model.name for run in config.runs} == {
        "retained_logistic_regression",
        "retained_mlp",
        "retained_xgboost",
    }
    assert {run.method.n_counterfactuals for run in config.runs} == {1}


def test_campaign_target_model_specs_match_fixed_registry_params():
    for filename in _EXPECTED:
        config = load_matrix_config(_ROOT / filename)
        target_models = {
            (
                run.target_model.name,
                repr(dict(run.target_model.params)),
            ): run.target_model
            for run in config.runs
        }
        for target_model in target_models.values():
            DEFAULT_TARGET_MODEL_REGISTRY.resolve(
                target_model.name,
                target_model.params,
            )


def test_future_ablation_axes_have_unambiguous_raw_values():
    e5 = load_matrix_config(_ROOT / "campaign_e5_search.yaml")
    e5_searches = {
        str(run.method.params["search"]): run.method.params["search"]
        for run in e5.runs
    }
    assert all(
        "allow_revisits" not in search
        for search in e5_searches.values()
        if search.get("allow_revisits") is not False
    )

    e6 = load_matrix_config(_ROOT / "campaign_e6_context.yaml")
    assert {
        run.method.params["foundation"]["context_labels"] for run in e6.runs
    } == {"predictions", "true"}

    e7 = load_matrix_config(_ROOT / "campaign_e7_cost.yaml")
    full_reference = [
        run
        for run in e7.runs
        if run.method.params.get("foundation", {}).get("confidence_quantiles")
    ]
    assert full_reference
    assert all(
        run.method.params["foundation"]["context_size"] == 512
        for run in full_reference
    )


def test_pre_ablation_campaign_method_configs_are_registry_valid():
    for filename in (
        "campaign_e1_main.yaml",
        "campaign_e2_diverse.yaml",
        "campaign_e3_backend.yaml",
        "campaign_e4_confidence.yaml",
        "campaign_e10_headline.yaml",
    ):
        config = load_matrix_config(_ROOT / filename)
        methods = {
            (
                run.method.name,
                run.method.variant,
                repr(dict(run.method.params)),
            ): run.method
            for run in config.runs
        }
        for method in methods.values():
            DEFAULT_METHOD_REGISTRY.create(
                method.name,
                method.params,
                variant=method.variant,
            )


def test_dgx_launchers_clear_the_exact_success_marker_before_nohup():
    expected_markers = {
        "07": "stage07.DONE",
        "08": "stage08.DONE",
        "09": "stage09.DONE",
        "10": "stage10.DONE",
        "11": "stage11.E9_DONE",
        "12": "stage12.DONE",
    }
    for stage, marker in expected_markers.items():
        script = (_DGX / f"launch_stage{stage}.sh").read_text()
        assignment = f"MARKER=$RUN_DIR/{marker}"
        clear = 'rm -f "$MARKER"'
        launch = 'nohup "$PROJECT_DIR/experiments/zeroshot_cf/dgx/run_stage.sh"'
        assert script.index(assignment) < script.index(clear) < script.index(launch)
        assert script.count('"$MARKER"') == 2
