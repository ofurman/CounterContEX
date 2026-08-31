"""Named semantic cases exercised through the production evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from experiments.zeroshot_cf.core.contracts import (
    BenchmarkCase,
    DatasetProvenance,
    FactualSelection,
    FeatureDomains,
    FeatureSchema,
    GenerationResult,
    PreparedDataset,
)
from experiments.zeroshot_cf.evaluation import EvaluationSpec, Evaluator

FIXTURE = Path(__file__).parent / "fixtures" / "architecture_v1" / "semantic_cases.json"


def _cases():
    fixture = json.loads(FIXTURE.read_text())
    return [*fixture["cases"], *fixture.get("evaluator_cases", [])]


class _FixtureOracle:
    def __init__(self, rows, classes=(0, 1)):
        self.rows = rows
        self.classes_ = np.asarray(classes)

    def predict(self, X):
        return np.array([self.rows[tuple(row)][0] for row in np.asarray(X)])

    def predict_proba(self, X):
        p = np.array([self.rows[tuple(row)][1] for row in np.asarray(X)])
        return np.column_stack([1 - p, p])


def _evaluation_inputs(
    case, *, classes=(0, 1), point_diagnostics=(), run_diagnostics=None
):
    candidates = np.asarray(case["candidates"], dtype=float)
    available = np.asarray(case["available"], dtype=bool)
    factuals = np.full((len(candidates), candidates.shape[2]), 0.05)
    rows = {}
    for point, rank in zip(*np.where(available), strict=True):
        rows[tuple(candidates[point, rank])] = (
            int(case["predictions"][point][rank]),
            float(case["target_probabilities"][point][rank]),
        )
    oracle = _FixtureOracle(rows, classes)
    dataset = PreparedDataset(
        name="semantic",
        X_train=np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.4], [0.7, 0.3], [0.9, 0.1]]),
        y_train=np.array([0, 1, 0, 1, 1]),
        X_validation=np.array([[0.1, 0.1], [0.8, 0.8]]),
        y_validation=np.array([0, 1]),
        X_test=factuals,
        y_test=np.zeros(len(factuals), dtype=int),
        schema=FeatureSchema(
            names=("a", "b"),
            numerical=(0, 1),
            categorical_groups=(),
            actionable_scalars=(0, 1),
            actionable_groups=(),
            immutable=(),
            domains=FeatureDomains(np.zeros(2), np.ones(2), {}),
        ),
        provenance=DatasetProvenance(
            provider="fixture",
            source_revision="v1",
            source_hashes={"semantic_cases": "tracked"},
            preprocessing_id="identity",
            split_id="fixed",
            fingerprint="semantic-fixture",
        ),
    )
    targets = np.asarray(case["targets"], dtype=int)
    benchmark = BenchmarkCase(
        case_id=f"semantic-{case['id']}",
        dataset=dataset,
        factuals=FactualSelection(np.arange(len(factuals)), factuals, dataset.y_test),
        oracle=oracle,
        factual_predictions=np.full(len(targets), classes[0]),
        targets=targets,
        protocol={"target_policy": "fixture"},
    )
    result = GenerationResult(
        candidates,
        available,
        point_diagnostics=point_diagnostics,
        run_diagnostics={} if run_diagnostics is None else run_diagnostics,
        artifacts={
            name: np.asarray(value, dtype=float)
            for name, value in case.get("artifacts", {}).items()
        },
    )
    spec = EvaluationSpec(
        probability_threshold=float(case["tau"]),
        primary_rank=int(case.get("primary_rank", 0)),
    )
    return benchmark, result, spec


def _evaluate(case):
    benchmark, result, spec = _evaluation_inputs(case)
    return Evaluator().prepare(benchmark, spec).evaluate(result)


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["id"])
def test_evaluation_semantics_are_derived_from_available_candidates(case):
    report = _evaluate(case)
    for name, expected in case["expected"].items():
        actual = report.summary.values[name]
        if expected is None:
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected)


def test_best_effort_artifact_does_not_count_as_coverage():
    case = next(case for case in _cases() if case["id"] == "best_effort_only")
    report = _evaluate(case)
    assert report.summary.values["coverage"] == 0.0
    assert "method.best_effort" in report.arrays.values


def test_class_and_probability_threshold_validity_are_not_substituted():
    case = next(
        case for case in _cases() if case["id"] == "target_class_below_threshold"
    )
    summary = _evaluate(case).summary.values
    assert summary["validity_returned_class"] == 1.0
    assert summary["validity_returned_threshold"] == 0.0


def test_evaluator_is_method_blind_and_distances_use_only_valid_candidates():
    case = next(case for case in _cases() if case["id"] == "partial_k3")
    report = _evaluate(case)
    assert not any(name.startswith("method") for name in report.summary.values)
    assert report.summary.values["proximity_continuous_manhattan"] == pytest.approx(0.9)
    assert report.summary.values["proximity_continuous_euclidean"] == pytest.approx(
        np.sqrt(0.75**2 + 0.15**2)
    )
    assert report.summary.values["proximity_grouped_gower"] == pytest.approx(0.45)
    np.testing.assert_allclose(
        report.arrays.values["candidate.grouped_gower"], [0.45, 0.2]
    )


def test_common_outputs_ignore_contradictory_method_diagnostics():
    case = next(case for case in _cases() if case["id"] == "partial_k3")
    benchmark, result_a, spec = _evaluation_inputs(
        case,
        point_diagnostics=({"claimed_valid": True, "reason": "success"},),
        run_diagnostics={"claimed_coverage": 1.0, "method": "left"},
    )
    _, result_b, _ = _evaluation_inputs(
        case,
        point_diagnostics=({"claimed_valid": False, "reason": "failure"},),
        run_diagnostics={"claimed_coverage": 0.0, "method": "right"},
    )
    prepared = Evaluator().prepare(benchmark, spec)
    left = prepared.evaluate(result_a)
    right = prepared.evaluate(result_b)

    assert left.summary.values == right.summary.values
    assert left.points == right.points
    assert left.candidates == right.candidates
    common_names = {name for name in left.arrays.values if name.startswith("common.")}
    assert common_names == {
        name for name in right.arrays.values if name.startswith("common.")
    }
    for name in common_names:
        np.testing.assert_equal(left.arrays.values[name], right.arrays.values[name])


def test_reversed_nonzero_classes_map_target_probability_by_label():
    case = {
        "id": "reversed-nonzero",
        "tau": 0.7,
        "targets": [2],
        "candidates": [[[0.8, 0.2]]],
        "available": [[True]],
        "predictions": [[2]],
        "target_probabilities": [[0.8]],
    }
    benchmark, result, spec = _evaluation_inputs(case, classes=(7, 2))
    report = Evaluator().prepare(benchmark, spec).evaluate(result)
    assert report.points[0].values["target_probability"] == pytest.approx(0.8)
    assert report.summary.values["validity_returned_threshold"] == 1.0


def test_prepare_fits_novelty_models_once_for_repeated_evaluation(monkeypatch):
    case = next(case for case in _cases() if case["id"] == "partial_k3")
    benchmark, result, spec = _evaluation_inputs(case)
    from experiments.zeroshot_cf.evaluation import evaluator as evaluator_module
    from experiments.zeroshot_cf.evaluation import metrics as metrics_module

    calls = {"prepare": 0, "lof_fit": 0, "isolation_fit": 0}
    real_prepare = evaluator_module.prepare_novelty_models
    real_lof_fit = metrics_module.LocalOutlierFactor.fit
    real_isolation_fit = metrics_module.IsolationForest.fit

    def observed_prepare(*args, **kwargs):
        calls["prepare"] += 1
        return real_prepare(*args, **kwargs)

    def observed_lof_fit(self, *args, **kwargs):
        calls["lof_fit"] += 1
        return real_lof_fit(self, *args, **kwargs)

    def observed_isolation_fit(self, *args, **kwargs):
        calls["isolation_fit"] += 1
        return real_isolation_fit(self, *args, **kwargs)

    monkeypatch.setattr(evaluator_module, "prepare_novelty_models", observed_prepare)
    monkeypatch.setattr(metrics_module.LocalOutlierFactor, "fit", observed_lof_fit)
    monkeypatch.setattr(metrics_module.IsolationForest, "fit", observed_isolation_fit)

    prepared = Evaluator().prepare(benchmark, spec)
    prepared.evaluate(result)
    prepared.evaluate(result)
    assert calls == {"prepare": 1, "lof_fit": 1, "isolation_fit": 1}
