"""Public-API tests for the retained TabICL generator boundary."""

from __future__ import annotations

import copy
import csv
import os
from types import SimpleNamespace

import experiments.zeroshot_cf.exp8_tabicl_cf as exp8
import numpy as np
import pytest
from experiments.zeroshot_cf.diverse_search import DiverseBeamSearchConfig
from experiments.zeroshot_cf.generator import (
    TabICLGeneratorConfig,
    TabICLGeneratorInputs,
    TabICLGeneratorPointBackend,
    generate_counterfactual_batch,
)
from experiments.zeroshot_cf.orchestration import exp8_compat
from experiments.zeroshot_cf.tabicl_joint_plausibility import TabICLJointScoreBatch

EXP8_HISTORICAL_INFO_KEYS = (
    "bundle",
    "y_pred",
    "y_target",
    "actionable_idx",
    "immutable_idx",
    "disc_model",
    "tau",
    "temperature",
    "candidate_quantiles",
    "confidence_quantiles",
    "cf_mode",
    "plausibility_backend",
    "max_validity_steps",
    "allow_revisits",
    "joint_shortlist_size",
    "max_extra_actions",
    "min_joint_log_gain",
    "n_counterfactuals",
    "diversity_beam_width",
    "diversity_candidate_pool_size",
    "diversity_max_extra_actions",
    "diversity_max_gower_ratio",
    "diversity_max_gower_increase",
    "diversity_candidate_generation",
    "diversity_selector",
    "categorical_proposal_count",
    "categorical_confidence_batching",
    "conditional_estimator_cache",
    "tabicl_kv_cache",
    "test_selection",
    "split_variant",
    "preprocessing_variant",
    "n_dropped_rows",
    "n_estimators",
    "runtime_s",
    "X_sparse",
    "X_cf_sets",
    "diverse_available_count_per_point",
    "diverse_candidate_pool_count_per_point",
    "diverse_search_depth_per_point",
    "diverse_histories_per_point",
    "point_runtime_s",
    "joint_scoring_runtime_s_per_point",
    "changed_per_point",
    "flipped_per_point",
    "steps_per_point",
    "history_per_point",
    "attempt_history_per_point",
    "validity_steps_per_point",
    "initial_valid_step_per_point",
    "refinement_steps_per_point",
    "accepted_refinement_count_per_point",
    "initial_sparse_action_count_per_point",
    "final_action_count_per_point",
    "initial_tabicl_joint_log_density_per_point",
    "final_tabicl_joint_log_density_per_point",
    "tabicl_joint_log_density_gain_per_point",
    "joint_scoring_batch_count_per_point",
    "joint_rows_scored_per_point",
    "extra_actions_per_point",
    "refinement_stopping_reason_per_point",
    "target_probability_per_point",
)

EXP8_HISTORICAL_MULTI_CF_HEADER = (
    "dataset",
    "backend",
    "selector",
    "valid_candidate_objective",
    "context_strategy",
    "context_size",
    "context_labels",
    "candidate_mode",
    "context_update",
    "point_estimate",
    "project_to_domain",
    "candidate_quantiles",
    "confidence_quantiles",
    "cf_mode",
    "plausibility_backend",
    "max_validity_steps",
    "allow_revisits",
    "joint_shortlist_size",
    "max_extra_actions",
    "min_joint_log_gain",
    "n_counterfactuals",
    "diversity_beam_width",
    "diversity_candidate_pool_size",
    "diversity_max_extra_actions",
    "diversity_max_gower_ratio",
    "diversity_max_gower_increase",
    "joint_scoring_batch_count_mean",
    "joint_rows_scored_mean",
    "accepted_refinement_count_mean",
    "initial_sparse_action_count_mean",
    "final_action_count_mean",
    "extra_actions_mean",
    "categorical_proposal_count",
    "categorical_confidence_batching",
    "conditional_estimator_cache",
    "tabicl_kv_cache",
    "split_variant",
    "test_selection",
    "n_estimators",
    "temperature",
    "n_test",
    "runtime_s",
    "validity",
    "lof_scores_cf",
    "sparsity",
    "actionability",
    "true_actionability",
    "proximity_l2_jaccard",
    "frac_oob",
    "l0_count_mean",
    "l0_count_median",
    "l0_count_max",
    "steps_mean",
    "steps_median",
    "steps_max",
    "failure_rate",
    "n_actionable",
    "diverse_coverage_at_k",
    "diverse_returned_count_mean",
    "diverse_returned_validity",
    "diverse_action_jaccard_mean",
    "diverse_action_jaccard_min",
    "diverse_pairwise_gower_mean",
    "diverse_pairwise_gower_min",
    "diverse_factual_gower_mean",
    "diverse_action_count_mean",
)


class _LinearDisc:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        matrix = np.asarray(X, dtype=float)
        p1 = np.clip(0.1 + 0.5 * matrix[:, 0] + 0.05 * matrix[:, 1], 0.0, 0.99)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _SingleFeatureDisc:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        matrix = np.asarray(X, dtype=float)
        p1 = np.clip(0.1 + 0.6 * matrix[:, 0], 0.0, 0.99)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _ConfidenceGridSampler:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def sample_candidate_grid(
        self,
        _query,
        columns,
        *,
        quantiles,
        fixed_target,
        confidences=None,
    ):
        self.calls.append(
            {
                "columns": tuple(columns),
                "quantiles": tuple(float(value) for value in quantiles),
                "confidences": None
                if confidences is None
                else tuple(float(value) for value in confidences),
                "fixed_target": int(fixed_target),
            }
        )
        assert tuple(columns) == (0,)
        return np.array([[[0.2, 0.4], [0.7, 0.9]]], dtype=float)


class _StatefulRevisitSampler:
    def sample_candidate_grid(
        self,
        query,
        columns,
        *,
        quantiles,
        fixed_target,
        confidences=None,
    ):
        del quantiles, fixed_target, confidences
        row = np.asarray(query, dtype=float)[0]
        values = []
        for column in columns:
            if column == 0:
                values.append([1.0] if row[0] >= 0.59 else [0.6])
            else:
                values.append([0.2])
        return np.asarray(values, dtype=float)


class _TwoFeatureSampler:
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
        return np.ones((len(columns), 1), dtype=float)


class _AcceptanceScorer:
    def score_rows(self, rows, target_class):
        assert target_class == 1
        return TabICLJointScoreBatch(
            joint_log_density=np.array([-9.95, -9.45], dtype=float)
        )


class _RejectionScorer:
    def score_rows(self, rows, target_class):
        assert target_class == 1
        return TabICLJointScoreBatch(
            joint_log_density=np.array([-9.95, -9.96], dtype=float)
        )


class _SingleFeatureSampler:
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
        return np.ones((len(columns), 1), dtype=float)


def test_public_generator_forwards_confidence_quantiles_and_preserves_immutable() -> (
    None
):
    sampler = _ConfidenceGridSampler()
    result = generate_counterfactual_batch(
        TabICLGeneratorInputs(
            factuals=np.array([[0.0, 5.0]], dtype=float),
            targets=np.array([1], dtype=int),
            numerical_columns=(0,),
            categorical_groups=(),
            immutable_idx=(1,),
        ),
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.25, 0.75),
            confidence_quantiles=(0.25, 0.75),
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=sampler,
            candidate_confidences=(0.55, 0.85),
            metadata={
                "categorical_confidence_batching": True,
                "conditional_estimator_cache": True,
                "tabicl_kv_cache": True,
            },
        ),
    )

    np.testing.assert_allclose(result.counterfactuals, [[0.7, 5.0]])
    assert sampler.calls == [
        {
            "columns": (0,),
            "quantiles": (0.25, 0.75),
            "confidences": (0.55, 0.85),
            "fixed_target": 1,
        }
    ]
    assert result.diagnostics.flipped_per_point == (True,)
    np.testing.assert_allclose(result.diagnostics.target_probability_per_point, [0.52])


def test_public_generator_revisit_control_changes_whether_validity_is_reached() -> None:
    factuals = np.array([[0.0, 0.0]], dtype=float)
    targets = np.array([1], dtype=int)
    inputs = TabICLGeneratorInputs(
        factuals=factuals,
        targets=targets,
        numerical_columns=(0, 1),
        categorical_groups=(),
    )

    no_revisit = generate_counterfactual_batch(
        inputs,
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.5,),
            max_validity_steps=2,
            allow_revisits=False,
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=_StatefulRevisitSampler()
        ),
    )
    with_revisit = generate_counterfactual_batch(
        inputs,
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.5,),
            max_validity_steps=2,
            allow_revisits=True,
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=_StatefulRevisitSampler()
        ),
    )

    assert no_revisit.diagnostics.flipped_per_point == (False,)
    np.testing.assert_allclose(no_revisit.counterfactuals, [[0.6, 0.0]])
    assert with_revisit.diagnostics.flipped_per_point == (True,)
    np.testing.assert_allclose(with_revisit.counterfactuals, [[1.0, 0.0]])


def test_public_generator_caps_pre_validity_steps() -> None:
    result = generate_counterfactual_batch(
        TabICLGeneratorInputs(
            factuals=np.array([[0.0, 0.0]], dtype=float),
            targets=np.array([1], dtype=int),
            numerical_columns=(0, 1),
            categorical_groups=(),
        ),
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.5,),
            max_validity_steps=1,
            allow_revisits=True,
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=_StatefulRevisitSampler()
        ),
    )

    assert result.diagnostics.flipped_per_point == (False,)
    assert result.diagnostics.validity_steps_per_point == (1,)
    np.testing.assert_allclose(result.counterfactuals, [[0.6, 0.0]])


@pytest.mark.parametrize(
    ("scorer", "expected", "accepted", "reason"),
    [
        (_AcceptanceScorer(), [1.0, 1.0], 1, "one_shot_accepted"),
        (_RejectionScorer(), [1.0, 0.0], 0, "no_improving_candidate"),
    ],
)
def test_public_generator_exposes_refinement_acceptance_and_rejection(
    scorer,
    expected,
    accepted,
    reason,
) -> None:
    result = generate_counterfactual_batch(
        TabICLGeneratorInputs(
            factuals=np.array([[0.0, 0.0]], dtype=float),
            targets=np.array([1], dtype=int),
            numerical_columns=(0, 1),
            categorical_groups=(),
        ),
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.5,),
            cf_mode="data_plausible",
            max_extra_actions=1,
            min_joint_log_gain=0.0,
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=_TwoFeatureSampler(),
            joint_scorer=scorer,
        ),
    )

    np.testing.assert_allclose(result.counterfactuals, [expected])
    assert result.diagnostics.accepted_refinement_count_per_point == (accepted,)
    assert result.diagnostics.refinement_stopping_reason_per_point == (reason,)


def test_public_generator_never_pads_diverse_results_with_invalid_rows() -> None:
    result = generate_counterfactual_batch(
        TabICLGeneratorInputs(
            factuals=np.array([[0.0]], dtype=float),
            targets=np.array([1], dtype=int),
            numerical_columns=(0,),
            categorical_groups=(),
        ),
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.5,),
            diversity_config=DiverseBeamSearchConfig(
                n_counterfactuals=3,
                beam_width=3,
                candidate_pool_size=3,
            ),
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=_SingleFeatureSampler()
        ),
    )

    assert result.diagnostics.diverse_available_count_per_point[0] == 1
    np.testing.assert_allclose(result.counterfactual_sets[0, 0], [1.0])
    assert np.isnan(result.counterfactual_sets[0, 1:]).all()


def test_exp8_adapter_translates_every_scientific_option() -> None:
    spec = exp8._spec(
        "german_credit",
        tau=0.73,
        temperature=0.8,
        n_estimators=3,
        max_test=-1,
        candidate_quantiles=(0.2, 0.8),
        confidence_quantiles=(0.25, 0.75),
        cf_mode="data-plausible",
        tabicl_joint_permutations=4,
        max_validity_steps=9,
        allow_revisits=False,
        joint_shortlist_size=7,
        max_extra_actions=2,
        min_joint_log_gain=0.1,
        n_counterfactuals=3,
        diversity_beam_width=5,
        diversity_candidate_pool_size=11,
        diversity_max_extra_actions=4,
        diversity_max_gower_ratio=1.2,
        diversity_max_gower_increase=0.03,
        validation_fraction=0.2,
        test_selection="first",
        drop_heloc_all_minus9=True,
    )

    assert spec.dataset.name == "german_credit"
    assert spec.protocol.max_test is None
    assert spec.protocol.test_selection == "first"
    assert spec.protocol.params == {
        "validation_fraction": 0.2,
        "drop_heloc_all_minus9": True,
    }
    assert spec.method.name == "countercontex"
    assert spec.method.variant == "default"
    assert spec.method.n_counterfactuals == 3
    assert spec.method.params == {
        "search": {
            "tau": 0.73,
            "candidate_quantiles": (0.2, 0.8),
            "cf_mode": "data_plausible",
            "max_validity_steps": 9,
            "allow_revisits": False,
            "joint_shortlist_size": 7,
            "max_extra_actions": 2,
            "min_joint_log_gain": 0.1,
        },
        "diversity": {
            "beam_width": 5,
            "candidate_pool_size": 11,
            "max_extra_actions": 4,
            "max_gower_ratio": 1.2,
            "max_gower_increase": 0.03,
        },
        "foundation": {
            "backend": "tabicl",
            "n_estimators": 3,
            "temperature": 0.8,
            "confidence_quantiles": (0.25, 0.75),
            "tabicl_joint_permutations": 4,
        },
    }
    assert spec.evaluation.probability_threshold == 0.73


def test_exp8_adapter_uses_exact_current_run_when_stale_version_is_newer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured = {}
    arrays = {
        "X_test": np.array([[0.0]]),
        "y_test": np.array([0]),
        "X_cf": np.array([[1.0]]),
        "X_sparse": np.array([[0.9]]),
        "X_cf_sets": np.array([[[1.0], [np.nan]]]),
        "diverse_available_count": np.array([1]),
        "y_pred": np.array([0]),
        "y_target": np.array([1]),
        "y_cf_pred": np.array([1]),
    }

    metrics = {
        "validity": 1.0,
        "lof_scores_cf": 1.25,
        "proximity_all_features_euclidean": 1.0,
        "diverse_coverage_at_k": 0.0,
        "diverse_returned_count_mean": 1.0,
        "diverse_returned_validity": 1.0,
        "diverse_action_jaccard_mean": float("nan"),
        "diverse_action_jaccard_min": float("nan"),
        "diverse_pairwise_gower_mean": float("nan"),
        "diverse_pairwise_gower_min": float("nan"),
        "diverse_factual_gower_mean": 1.0,
        "diverse_action_count_mean": 1.0,
        "plausibility_backend": "proposal_support",
        "max_validity_steps": 1,
        "split_variant": "train_test",
        "preprocessing_variant": "baseline",
        "n_dropped_rows": 0,
    }
    manifest = {
        "resolved_method_config": {
            "search": {
                "tau": 0.5,
                "candidate_quantiles": (0.5,),
                "cf_mode": "sparse",
                "max_validity_steps": None,
                "allow_revisits": True,
                "joint_shortlist_size": 16,
                "max_extra_actions": 1,
                "min_joint_log_gain": 0.0,
                "categorical_proposal_count": 1,
            },
            "diversity": {
                "beam_width": 8,
                "candidate_pool_size": 16,
                "max_extra_actions": 2,
                "max_gower_ratio": 1.5,
                "max_gower_increase": 0.02,
            },
            "foundation": {
                "backend": "tabicl",
                "n_estimators": 1,
                "temperature": 1e-09,
                "confidence_quantiles": None,
                "tabicl_joint_permutations": 1,
                "cache_dir": None,
            },
        },
        "method_run_diagnostics": {
            "runtime_s": 0.125,
            "cache": {"conditional_estimator": True, "key_value": False},
            "actionable_idx": [0],
            "immutable_idx": [],
        },
        "method_point_diagnostics": [
            {
                "flipped": True,
                "changed_columns": [0],
                "steps": 1,
                "validity_steps": 1,
                "refinement_steps": 0,
                "accepted_refinement_count": 0,
                "history": [{"column": 0}],
                "attempt_history": [{"column": 0}],
                "diverse_histories": [[{"column": 0}]],
                "initial_valid_step": 1,
                "initial_sparse_action_count": 1,
                "final_action_count": 1,
                "initial_tabicl_joint_log_density": float("nan"),
                "final_tabicl_joint_log_density": float("nan"),
                "tabicl_joint_log_density_gain": float("nan"),
                "joint_scoring_batch_count": 0,
                "joint_rows_scored": 0,
                "extra_actions": 0,
                "refinement_stopping_reason": "not_requested",
                "candidate_pool_count": 1,
                "search_depth": 1,
                "target_probability": 0.9,
                "point_runtime_s": 0.125,
                "joint_scoring_runtime_s": 0.0,
            }
        ],
    }

    def fake_run(spec, *, results_dir, tabicl_cache_dir):
        captured.update(
            spec=spec,
            results_dir=results_dir,
            tabicl_cache_dir=tabicl_cache_dir,
        )
        stored.manifest.update(
            cell_id=spec.cell_id,
            scientific_spec=spec.scientific_payload(),
        )
        stale_manifest = copy.deepcopy(stored.manifest)
        stale_manifest["method_point_diagnostics"][0]["history"] = [{"column": "stale"}]
        current_path = tmp_path / "runs" / spec.cell_id / "v3-current"
        stale_path = tmp_path / "runs" / spec.cell_id / "v2-stale"
        current_path.mkdir(parents=True)
        stale_path.mkdir()
        os.utime(current_path, ns=(1, 1))
        os.utime(stale_path, ns=(2, 2))
        stored.path = current_path
        captured["stale"] = SimpleNamespace(
            manifest=stale_manifest,
            report=stored.report,
            path=stale_path,
        )
        return (
            metrics,
            stored,
            {
                "dataset_adapter": "live-adapter",
                "oracle": "live-oracle",
            },
        )

    stored = SimpleNamespace(
        manifest=manifest,
        report=SimpleNamespace(
            summary=SimpleNamespace(values={"set_validity_returned_threshold": 1.0}),
            arrays=SimpleNamespace(
                values={
                    "common.available": np.array([[True, False]]),
                    "candidate.lof_score": np.array([1.5]),
                }
            ),
            metadata={"primary_rank": 0},
        ),
    )

    monkeypatch.setattr(exp8, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(exp8, "run_legacy_dataset_with_context", fake_run)
    monkeypatch.setattr(exp8_compat, "_legacy_arrays", lambda *args: arrays)
    monkeypatch.setattr(
        exp8_compat,
        "dataset_bundle_from_adapter",
        lambda name, adapter: (name, adapter),
    )

    X_test, y_test, X_cf, info = exp8.generate_tabicl_counterfactuals(
        "heloc",
        candidate_quantiles=(0.5,),
        max_test=1,
        n_counterfactuals=2,
        cache_dir=tmp_path / "cache",
    )

    np.testing.assert_allclose(X_test, [[0.0]])
    np.testing.assert_allclose(X_cf, [[1.0]])
    np.testing.assert_array_equal(y_test, [0])
    assert captured["spec"].method.name == "countercontex"
    assert captured["spec"].protocol.test_selection == "first"
    assert captured["results_dir"] == tmp_path
    assert captured["tabicl_cache_dir"] == tmp_path / "cache"
    assert info["metrics"]["validity"] == 1.0
    assert info["bundle"] == ("heloc", "live-adapter")
    assert info["disc_model"] == "live-oracle"
    assert info["metrics"]["lof_scores_cf"] == 1.5
    assert info["diverse_available_count_per_point"].tolist() == [1]
    assert info["flipped_per_point"] == [True]
    assert info["history_per_point"] == [[{"column": 0}]]
    assert info["attempt_history_per_point"] == [[{"column": 0}]]
    assert info["diverse_histories_per_point"] == [[[{"column": 0}]]]
    assert tuple(info) == (*EXP8_HISTORICAL_INFO_KEYS, "metrics", "run_spec")
    assert captured["stale"].path.stat().st_mtime_ns > stored.path.stat().st_mtime_ns
    assert captured["stale"].manifest["cell_id"] == stored.manifest["cell_id"]

    monkeypatch.setattr(
        exp8,
        "generate_tabicl_counterfactuals",
        lambda dataset_name, **kwargs: (X_test, y_test, X_cf, info),
    )
    returned = exp8.run_and_report("heloc")
    output = tmp_path / "exp8_tabicl_heloc_metrics.csv"
    assert output.is_file()
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        assert tuple(rows[0]) == EXP8_HISTORICAL_MULTI_CF_HEADER
    assert rows[0]["backend"] == "tabicl_v2"
    assert rows[0]["runtime_s"] == "0.12"
    assert rows[0]["n_actionable"] == "1"
    assert returned["validity"] == 1.0


def test_exp8_help_stays_runnable() -> None:
    with pytest.raises(SystemExit) as excinfo:
        exp8.main(["--help"])
    assert excinfo.value.code == 0
