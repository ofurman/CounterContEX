"""Reasoned examples for truthful availability and split-validity metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "architecture_v1" / "semantic_cases.json"


def _cases() -> list[dict[str, object]]:
    return json.loads(FIXTURE.read_text())["cases"]


def _evaluate(case: dict[str, object]) -> dict[str, float]:
    available = np.asarray(case["available"], dtype=bool)
    targets = np.asarray(case["targets"], dtype=int)[:, None]
    predictions = np.asarray(
        [
            [-1 if value is None else value for value in row]
            for row in case["predictions"]
        ],
        dtype=int,
    )
    probabilities = np.asarray(case["target_probabilities"], dtype=np.float64)
    class_success = available & (predictions == targets)
    threshold_success = class_success & (probabilities >= float(case["tau"]))
    returned = int(available.sum())
    requested = int(available.size)

    return {
        "coverage": float(available.any(axis=1).mean()),
        "validity_returned_class": (
            float(class_success.sum() / returned) if returned else float("nan")
        ),
        "validity_returned_threshold": (
            float(threshold_success.sum() / returned) if returned else float("nan")
        ),
        "valid_success_rate_class_per_requested_slot": float(
            class_success.sum() / requested
        ),
        "valid_success_rate_threshold_per_requested_slot": float(
            threshold_success.sum() / requested
        ),
    }


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["id"])
def test_evaluation_semantics_are_derived_from_available_candidates(case) -> None:
    actual = _evaluate(case)

    for name, expected in case["expected"].items():
        if expected is None:
            assert np.isnan(actual[name])
        else:
            assert actual[name] == pytest.approx(expected)


def test_best_effort_artifact_does_not_count_as_coverage() -> None:
    case = next(case for case in _cases() if case["id"] == "best_effort_only")
    assert np.isfinite(np.asarray(case["artifacts"]["method.best_effort"])).all()
    assert _evaluate(case)["coverage"] == 0.0


def test_class_and_probability_threshold_validity_are_not_substituted() -> None:
    case = next(
        case for case in _cases() if case["id"] == "target_class_below_threshold"
    )
    metrics = _evaluate(case)
    assert metrics["validity_returned_class"] == 1.0
    assert metrics["validity_returned_threshold"] == 0.0
