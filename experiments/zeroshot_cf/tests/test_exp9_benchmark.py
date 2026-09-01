#  Copyright (c) Prior Labs GmbH 2026.
"""Compatibility tests for the thin Exp9 CounterContEx entry point."""

from __future__ import annotations

import csv
from pathlib import Path

import experiments.zeroshot_cf.benchmark_protocol as protocol
import numpy as np
from experiments.zeroshot_cf.exp9_dicoflex_benchmark import (
    DATASETS,
    DEFAULT_CANDIDATE_QUANTILES,
    DEFAULT_MAX_TEST,
    DEFAULT_N_COUNTERFACTUALS,
    _spec,
    aggregate_results,
)
from experiments.zeroshot_cf.tests.conftest import REPO_ROOT, RETAINED_FOCUSED_TESTS


class _PredictByLength:
    def __init__(self, predictions_by_size: dict[int, np.ndarray]) -> None:
        self._predictions_by_size = predictions_by_size

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predictions_by_size[len(X)].copy()


def test_exp9_excludes_adult_and_uses_larger_common_test_set() -> None:
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
    assert "experiments/zeroshot_cf/tests/test_exp9_benchmark.py" in (
        RETAINED_FOCUSED_TESTS
    )
    assert all((REPO_ROOT / path).is_file() for path in RETAINED_FOCUSED_TESTS)


def test_protocol_target_selection_is_prediction_derived_and_deterministic() -> None:
    X_test = np.arange(24, dtype=float).reshape(12, 2)
    y_test = np.array([0, 1] * 6, dtype=int)
    selected_first, labels_first = protocol.select_benchmark_test_rows(
        X_test, y_test, 6
    )
    selected_second, labels_second = protocol.select_benchmark_test_rows(
        X_test, y_test, 6
    )
    y_pred, y_target = protocol.build_classifier_targets(
        _PredictByLength({6: np.array([0, 1, 1, 0, 0, 1], dtype=int)}),
        selected_first,
    )

    np.testing.assert_array_equal(selected_first, selected_second)
    np.testing.assert_array_equal(labels_first, labels_second)
    np.testing.assert_array_equal(y_pred, [0, 1, 1, 0, 0, 1])
    np.testing.assert_array_equal(y_target, [1, 0, 0, 1, 1, 0])


def test_exp9_translates_legacy_settings_into_one_run_spec() -> None:
    spec = _spec(
        "heloc",
        max_test=7,
        n_estimators=2,
        temperature=0.1,
        tau=0.8,
        candidate_quantiles=(0.25, 0.75),
        confidence_quantiles=(0.5,),
        cf_mode="data-plausible",
        tabicl_joint_permutations=3,
        max_validity_steps=9,
        allow_revisits=False,
        joint_shortlist_size=5,
        max_extra_actions=2,
        min_joint_log_gain=0.2,
        n_counterfactuals=1,
        diversity_beam_width=4,
        diversity_candidate_pool_size=6,
        diversity_max_extra_actions=1,
        diversity_max_gower_ratio=1.2,
        diversity_max_gower_increase=0.01,
        validation_fraction=0.25,
        drop_heloc_all_minus9=False,
    )

    assert spec.dataset.name == "heloc"
    assert spec.protocol.max_test == 7
    assert spec.protocol.params["validation_fraction"] == 0.25
    assert spec.method.name == "countercontex"
    assert spec.method.variant == "default"
    assert spec.method.params["search"]["cf_mode"] == "data_plausible"
    assert spec.method.params["foundation"]["n_estimators"] == 2
    assert spec.evaluation.probability_threshold == 0.8
