#  Copyright (c) Prior Labs GmbH 2026.

"""Fast tests for the single-split DiCoFlex-compatible benchmark runner."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import experiments.zeroshot_cf.benchmark_protocol as protocol
import experiments.zeroshot_cf.exp9_dicoflex_benchmark as benchmark
import numpy as np
import pytest
from experiments.zeroshot_cf.benchmark_protocol import BenchmarkDatasetContext
from experiments.zeroshot_cf.exp9_dicoflex_benchmark import (
    DATASETS,
    DEFAULT_CANDIDATE_QUANTILES,
    DEFAULT_CONFIDENCE_QUANTILES,
    DEFAULT_DIVERSITY_BEAM_WIDTH,
    DEFAULT_DIVERSITY_CANDIDATE_POOL_SIZE,
    DEFAULT_DIVERSITY_MAX_EXTRA_ACTIONS,
    DEFAULT_DIVERSITY_MAX_GOWER_INCREASE,
    DEFAULT_DIVERSITY_MAX_GOWER_RATIO,
    DEFAULT_JOINT_SHORTLIST_SIZE,
    DEFAULT_MAX_EXTRA_ACTIONS,
    DEFAULT_MAX_TEST,
    DEFAULT_MAX_VALIDITY_STEPS,
    DEFAULT_MIN_JOINT_LOG_GAIN,
    DEFAULT_N_COUNTERFACTUALS,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_TABICL_JOINT_PERMUTATIONS,
    DEFAULT_TEMPERATURE,
    DEFAULT_VALIDATION_FRACTION,
    TAU,
    aggregate_results,
    run_dataset,
)
from experiments.zeroshot_cf.tests.conftest import REPO_ROOT, RETAINED_FOCUSED_TESTS


def test_exp9_excludes_adult_and_uses_larger_common_test_set() -> None:
    """The fixed suite contains four suitable datasets and 1,000 factuals."""
    assert "adult" not in DATASETS
    assert DATASETS == (
        "heloc",
        "bank_marketing",
        "give_me_some_credit",
        "lending_club",
    )
    assert DEFAULT_MAX_TEST == 1000
    assert DEFAULT_N_COUNTERFACTUALS == 3
    assert DEFAULT_CANDIDATE_QUANTILES == tuple(i / 10 for i in range(1, 10))


def test_exp9_aggregates_independent_dataset_outputs(tmp_path: Path) -> None:
    """Per-dataset jobs combine in the declared benchmark order."""
    for index, dataset_name in enumerate(DATASETS):
        path = tmp_path / f"exp9_tabicl_{dataset_name}_metrics.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "validity"])
            writer.writeheader()
            writer.writerow({"dataset": dataset_name, "validity": index / 10})

    output = aggregate_results(tmp_path)
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["dataset"] for row in rows] == list(DATASETS)
    assert len(rows) == 4


def test_retained_manifest_matches_stage_one_focused_suite() -> None:
    """Stage 1 freezes one explicit focused suite instead of implicit collection."""
    assert RETAINED_FOCUSED_TESTS == (
        "experiments/zeroshot_cf/tests/test_candidate_domains.py",
        "experiments/zeroshot_cf/tests/test_data_cleaning.py",
        "experiments/zeroshot_cf/tests/test_diverse_search.py",
        "experiments/zeroshot_cf/tests/test_exp9_benchmark.py",
        "experiments/zeroshot_cf/tests/test_exp11_nice_nun_baseline.py",
        "experiments/zeroshot_cf/tests/test_exp12_optimization_baselines.py",
        "experiments/zeroshot_cf/tests/test_exp13_dice_baseline.py",
        "experiments/zeroshot_cf/tests/test_exp14_face_baseline.py",
        "experiments/zeroshot_cf/tests/test_generator.py",
        "experiments/zeroshot_cf/tests/test_grouped_categorical.py",
        "experiments/zeroshot_cf/tests/test_metrics_harness.py",
        "experiments/zeroshot_cf/tests/test_mixed_distance.py",
        "experiments/zeroshot_cf/tests/test_tabicl_backend.py",
        "experiments/zeroshot_cf/tests/test_tabicl_plausibility.py",
        "experiments/zeroshot_cf/tests/test_tabicl_checkpoints.py",
        "experiments/zeroshot_cf/tests/test_dataset_contract.py",
    )
    assert all((REPO_ROOT / path).is_file() for path in RETAINED_FOCUSED_TESTS)


class _PredictByLength:
    def __init__(self, predictions_by_size: dict[int, np.ndarray]) -> None:
        self._predictions_by_size = predictions_by_size

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predictions_by_size[len(X)].copy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions = self.predict(X)
        return np.column_stack(
            [
                np.where(predictions == 0, 0.8, 0.2),
                np.where(predictions == 1, 0.8, 0.2),
            ]
        )


def _stub_run_dataset(
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_counterfactuals: int = DEFAULT_N_COUNTERFACTUALS,
    portable_case: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    captures: dict[str, object] = {}
    written: dict[str, object] = {}
    X_train = np.array([[0.0, 0.0], [0.9, 1.0], [0.2, 0.3], [0.8, 0.7]])
    X_val = np.array([[0.1, 0.1], [0.7, 0.7]])
    y_val = np.array([0, 1], dtype=int)
    X_test = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    y_test = np.array([1, 1, 0], dtype=int)
    X_cf = np.array([[0.8, 0.8], [0.9, 0.9], [0.4, 0.4]])
    y_pred = np.array([0, 1, 1], dtype=int)
    y_target = np.array([1, 0, 0], dtype=int)
    y_cf_pred = np.array([1, 0, 1], dtype=int)
    bundle = SimpleNamespace(
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
        X_test=np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]]),
        split_variant="train_val_test_0.64_0.16_0.20",
        numerical_features_indices=[0, 1],
        preprocessing_variant="drop_heloc_all_minus9",
        n_dropped_rows=588,
    )
    disc_model = _PredictByLength({len(X_val): y_val, len(X_cf): y_cf_pred})
    context = BenchmarkDatasetContext(
        dataset_name="heloc",
        bundle=bundle,
        X_test=X_test,
        y_test=y_test,
        disc_model=disc_model,
        y_pred=y_pred,
        y_target=y_target,
        scalar_actionable=(0,),
        grouped_actionable=(),
        immutable_idx=(1,),
        categorical_groups=(),
        benchmark_case=object() if portable_case else None,
    )

    diagnostics = SimpleNamespace(
        plausibility_backend="tabicl_sparse",
        categorical_proposal_count=1,
        categorical_confidence_batching=True,
        conditional_estimator_cache=True,
        tabicl_kv_cache=True,
        runtime_s=0.5,
        joint_scoring_runtime_s_per_point=np.array([0.0, 0.1, 0.0]),
        point_runtime_s=np.array([0.2, 0.3, 0.4]),
        changed_per_point=((0,), (0, 1), (0,)),
        steps_per_point=(1, 2, 1),
        validity_steps_per_point=(1, 2, 1),
        refinement_steps_per_point=(0, 1, 0),
        history_per_point=(
            (
                {
                    "immediate_valid": True,
                    "action_sparsity": 0.1,
                    "grouped_gower": 0.2,
                    "action_type": "numerical",
                },
            ),
            (
                {
                    "immediate_valid": True,
                    "action_sparsity": 0.3,
                    "grouped_gower": 0.4,
                    "action_type": "categorical",
                },
                {"action_sparsity": 0.5},
            ),
            (
                {
                    "immediate_valid": False,
                    "action_sparsity": 0.2,
                    "grouped_gower": 0.6,
                    "action_type": "numerical",
                },
            ),
        ),
        initial_valid_step_per_point=(1, 2, None),
        initial_sparse_action_count_per_point=np.array([1, 1, 1]),
        final_action_count_per_point=np.array([1, 2, 9]),
        accepted_refinement_count_per_point=np.array([0, 1, 0]),
        extra_actions_per_point=np.array([0, 1, 0]),
        initial_tabicl_joint_log_density_per_point=np.array([0.0, 0.3, 0.1]),
        final_tabicl_joint_log_density_per_point=np.array([0.0, 0.5, 0.1]),
        tabicl_joint_log_density_gain_per_point=np.array([0.0, 0.2, 0.0]),
        joint_scoring_batch_count_per_point=np.array([0, 1, 0]),
        joint_rows_scored_per_point=np.array([0, 2, 0]),
        refinement_stopping_reason_per_point=(
            "no_refinement",
            "one_shot_accepted",
            "no_refinement",
        ),
        diverse_available_count_per_point=np.array([n_counterfactuals, 1, 0]),
        target_probability_per_point=np.array([0.9, 0.8, 0.4]),
        attempt_history_per_point=(({},), ({}, {}), ({},)),
        max_validity_steps=DEFAULT_MAX_VALIDITY_STEPS,
    )

    def fake_prepare_benchmark_context(dataset_name: str, **kwargs):
        captures["dataset_name"] = dataset_name
        captures["prepare_kwargs"] = kwargs
        return context

    def fake_run_tabicl_benchmark(runtime_context, **kwargs):
        captures["runtime_context"] = runtime_context
        captures["runtime_kwargs"] = kwargs
        return SimpleNamespace(
            counterfactuals=X_cf,
            sparse_counterfactuals=X_test.copy(),
            counterfactual_sets=np.repeat(X_cf[:, None, :], n_counterfactuals, axis=1),
            diagnostics=diagnostics,
            result=object(),
        )

    def fake_write_outputs(paths, row, point_rows, *, arrays=None):
        written["paths"] = paths
        written["row"] = row
        written["point_rows"] = point_rows
        written["arrays"] = arrays

    class FakeLocalOutlierFactor:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def fit(self, X: np.ndarray) -> FakeLocalOutlierFactor:
            del X
            return self

        def score_samples(self, X: np.ndarray) -> np.ndarray:
            return -np.ones(len(X))

    monkeypatch.setattr(
        benchmark, "prepare_benchmark_context", fake_prepare_benchmark_context
    )
    monkeypatch.setattr(benchmark, "run_tabicl_benchmark", fake_run_tabicl_benchmark)
    monkeypatch.setattr(benchmark, "get_one_hot_groups", lambda bundle: [])
    monkeypatch.setattr(
        benchmark,
        "compute_dicoflex_common_metrics",
        lambda *args, **kwargs: {
            "coverage": 1.0,
            "validity": 2 / 3,
            "actionability": 1.0,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_diverse_counterfactual_sets",
        lambda *args, **kwargs: {
            "diverse_coverage_at_k": 1 / 3,
            "diverse_returned_count_mean": 1.0,
        },
    )
    monkeypatch.setattr(benchmark, "print_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark, "LocalOutlierFactor", FakeLocalOutlierFactor)
    monkeypatch.setattr(
        benchmark,
        "grouped_gower_distance",
        lambda *args, **kwargs: np.array([0.2, 0.3, 0.4]),
    )
    monkeypatch.setattr(benchmark, "write_dataset_outputs", fake_write_outputs)
    return captures, written


def test_protocol_target_selection_is_prediction_derived_and_deterministic() -> None:
    X_test = np.arange(24, dtype=float).reshape(12, 2)
    y_test = np.array([0, 1] * 6, dtype=int)
    selected_first, labels_first = protocol.select_benchmark_test_rows(
        X_test,
        y_test,
        6,
    )
    selected_second, labels_second = protocol.select_benchmark_test_rows(
        X_test,
        y_test,
        6,
    )
    y_pred, y_target = protocol.build_classifier_targets(
        _PredictByLength({6: np.array([0, 1, 1, 0, 0, 1], dtype=int)}),
        selected_first,
    )

    np.testing.assert_array_equal(selected_first, selected_second)
    np.testing.assert_array_equal(labels_first, labels_second)
    np.testing.assert_array_equal(y_pred, [0, 1, 1, 0, 0, 1])
    np.testing.assert_array_equal(y_target, [1, 0, 0, 1, 1, 0])


def test_exp9_run_dataset_uses_protocol_outputs_and_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmark must keep the frozen split, target, and search defaults."""
    captures, written = _stub_run_dataset(monkeypatch)
    cache_dir = Path("/tmp/fake-tabicl-cache")
    row = run_dataset("heloc", tabicl_cache_dir=cache_dir)
    forwarded = captures["runtime_kwargs"]
    point_rows = written["point_rows"]

    assert captures["dataset_name"] == "heloc"
    assert captures["prepare_kwargs"]["max_test"] == DEFAULT_MAX_TEST
    assert (
        captures["prepare_kwargs"]["validation_fraction"] == DEFAULT_VALIDATION_FRACTION
    )
    assert captures["prepare_kwargs"]["test_selection"] == "stratified"
    assert forwarded["n_estimators"] == DEFAULT_N_ESTIMATORS
    assert forwarded["temperature"] == DEFAULT_TEMPERATURE
    assert forwarded["tau"] == TAU
    assert forwarded["candidate_quantiles"] == DEFAULT_CANDIDATE_QUANTILES
    assert forwarded["confidence_quantiles"] == DEFAULT_CONFIDENCE_QUANTILES
    assert forwarded["tabicl_joint_permutations"] == DEFAULT_TABICL_JOINT_PERMUTATIONS
    assert forwarded["max_validity_steps"] == DEFAULT_MAX_VALIDITY_STEPS
    assert forwarded["allow_revisits"] is True
    assert forwarded["joint_shortlist_size"] == DEFAULT_JOINT_SHORTLIST_SIZE
    assert forwarded["max_extra_actions"] == DEFAULT_MAX_EXTRA_ACTIONS
    assert forwarded["min_joint_log_gain"] == DEFAULT_MIN_JOINT_LOG_GAIN
    assert forwarded["n_counterfactuals"] == DEFAULT_N_COUNTERFACTUALS
    assert forwarded["diversity_beam_width"] == DEFAULT_DIVERSITY_BEAM_WIDTH
    assert forwarded["diversity_candidate_pool_size"] == (
        DEFAULT_DIVERSITY_CANDIDATE_POOL_SIZE
    )
    assert forwarded["diversity_max_extra_actions"] == (
        DEFAULT_DIVERSITY_MAX_EXTRA_ACTIONS
    )
    assert forwarded["diversity_max_gower_ratio"] == DEFAULT_DIVERSITY_MAX_GOWER_RATIO
    assert forwarded["diversity_max_gower_increase"] == (
        DEFAULT_DIVERSITY_MAX_GOWER_INCREASE
    )
    assert forwarded["cache_dir"] == cache_dir

    assert row["context_labels"] == "target_classifier"
    assert row["search_schedule"] == "bounded_beam_then_exact_fixed_size_dpp_map"
    assert row["valid_candidate_objective"] == "quality_constrained_dpp"
    assert row["candidate_mode"] == "batched"
    assert row["method"] == "tabicl_v2_diverse_dpp"
    assert {"coverage", "validity", "diverse_coverage_at_k"} <= set(row)
    assert row["l0_count_mean"] == pytest.approx(1.5)
    assert row["target_classifier_test_accuracy"] == pytest.approx(1 / 3)
    assert written["paths"].metrics_csv.name == "exp9_tabicl_heloc_metrics.csv"
    assert written["paths"].points_csv.name == "exp9_tabicl_heloc_points.csv"
    assert written["paths"].arrays_npz.name == "exp9_tabicl_heloc_arrays.npz"
    assert {"X_sparse", "X_cf_sets", "y_target"} <= set(written["arrays"])
    assert point_rows[0]["target"] == 1
    assert point_rows[1]["target"] == 0
    assert point_rows[0]["factual_prediction"] == 0
    assert point_rows[2]["valid"] is False


def test_exp9_portable_case_uses_canonical_evaluator_and_stage3_set_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures, _ = _stub_run_dataset(monkeypatch, portable_case=True)
    canonical = object()
    report = SimpleNamespace(
        summary=SimpleNamespace(
            values={
                "primary_coverage": 2 / 3,
                "primary_validity_returned_class": 1 / 2,
            }
        )
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        benchmark,
        "compute_dicoflex_common_metrics",
        lambda *args, **kwargs: pytest.fail("legacy common evaluator was used"),
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_diverse_counterfactual_sets",
        lambda *args, **kwargs: pytest.fail("legacy diverse evaluator was used"),
    )
    monkeypatch.setattr(
        benchmark,
        "adapt_generator_result",
        lambda result, *, seed: calls.update(result=result, seed=seed) or canonical,
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_result",
        lambda context, result, *, probability_threshold: (
            calls.update(
                context=context,
                canonical=result,
                probability_threshold=probability_threshold,
            )
            or report
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "legacy_common_metrics",
        lambda actual_report: {
            "coverage": 99.0,
            "validity": 99.0,
            "actionability": 0.75,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "compute_legacy_diverse_metrics",
        lambda **kwargs: (
            calls.update(set_kwargs=kwargs)
            or {
                "diverse_coverage_at_k": 1 / 3,
                "diverse_returned_count_mean": 4 / 3,
            }
        ),
    )

    row = run_dataset("heloc")

    assert calls["seed"] == 42
    assert calls["canonical"] is canonical
    assert calls["probability_threshold"] == TAU
    assert calls["context"] is captures["runtime_context"]
    assert row["coverage"] == pytest.approx(2 / 3)
    assert row["validity"] == pytest.approx(1 / 2)
    assert row["diverse_returned_count_mean"] == pytest.approx(4 / 3)


@pytest.mark.parametrize(
    ("n_counterfactuals", "cf_mode", "expected_method", "expected_schedule"),
    [
        (
            1,
            "sparse",
            "tabicl_v2_sparse",
            "probability_ascent_until_valid_then_min_grouped_gower",
        ),
        (
            1,
            "data_plausible",
            "tabicl_v2_data_plausible",
            "probability_ascent_until_valid_then_one_shot_joint_reranking",
        ),
    ],
)
def test_exp9_single_counterfactual_modes_keep_distinct_objectives(
    monkeypatch: pytest.MonkeyPatch,
    n_counterfactuals: int,
    cf_mode: str,
    expected_method: str,
    expected_schedule: str,
) -> None:
    """Single-CF modes must stay distinct from the diverse benchmark path."""
    captures, _ = _stub_run_dataset(monkeypatch, n_counterfactuals=n_counterfactuals)
    row = run_dataset("heloc", n_counterfactuals=n_counterfactuals, cf_mode=cf_mode)

    assert captures["runtime_kwargs"]["n_counterfactuals"] == 1
    assert row["method"] == expected_method
    assert row["search_schedule"] == expected_schedule
    assert row["valid_candidate_objective"] == "grouped_gower"
    assert row["post_valid_refinement"] is (cf_mode == "data_plausible")


def test_exp9_incomplete_common_coverage_raises_instead_of_fabricating_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = SimpleNamespace(
        X_train=np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.2], [0.8, 0.8]]),
        X_val=np.array([[0.1, 0.1], [0.9, 0.9]]),
        y_val=np.array([0, 1], dtype=int),
        X_test=np.array([[0.1, 0.1], [0.3, 0.3]]),
        split_variant="train_val_test_0.64_0.16_0.20",
        numerical_features_indices=[0, 1],
        preprocessing_variant="original",
        n_dropped_rows=0,
    )
    context = BenchmarkDatasetContext(
        dataset_name="heloc",
        bundle=bundle,
        X_test=bundle.X_test,
        y_test=np.array([0, 1], dtype=int),
        disc_model=_PredictByLength({2: np.array([0, 1], dtype=int)}),
        y_pred=np.array([0, 1], dtype=int),
        y_target=np.array([1, 0], dtype=int),
        scalar_actionable=(0, 1),
        grouped_actionable=(),
        immutable_idx=(),
        categorical_groups=(),
    )

    monkeypatch.setattr(
        benchmark, "prepare_benchmark_context", lambda *args, **kwargs: context
    )
    monkeypatch.setattr(
        benchmark,
        "run_tabicl_benchmark",
        lambda *args, **kwargs: SimpleNamespace(
            counterfactuals=np.array([[0.9, 0.9], [np.nan, 0.4]]),
            sparse_counterfactuals=context.X_test.copy(),
            counterfactual_sets=np.array(
                [[[0.9, 0.9]], [[np.nan, 0.4]]],
                dtype=float,
            ),
            diagnostics=SimpleNamespace(
                diverse_available_count_per_point=np.array([1, 1]),
            ),
        ),
    )
    monkeypatch.setattr(benchmark, "get_one_hot_groups", lambda bundle: [])

    with pytest.raises(ValueError, match="complete counterfactuals"):
        run_dataset("heloc", n_counterfactuals=1)
