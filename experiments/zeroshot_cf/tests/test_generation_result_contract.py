"""Executable v1 fixtures for the canonical generation-result contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from experiments.zeroshot_cf.core.contracts import GenerationResult

FIXTURE = Path(__file__).parent / "fixtures" / "architecture_v1" / "semantic_cases.json"


def _cases() -> dict[str, object]:
    return json.loads(FIXTURE.read_text())


def _arrays(case: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(case["candidates"], dtype=np.float64)
    available = np.asarray(case["available"], dtype=bool)
    return candidates, available


@pytest.mark.parametrize("case", _cases()["cases"], ids=lambda case: case["id"])
def test_reasoned_generation_results_obey_canonical_shape_and_padding(case) -> None:
    candidates, available = _arrays(case)
    GenerationResult(
        candidates,
        available,
        artifacts={
            name: np.asarray(value, dtype=float)
            for name, value in case.get("artifacts", {}).items()
        },
    )

    assert candidates.ndim == 3
    assert available.shape == candidates.shape[:2]
    if case["id"] == "returned_valid_k1":
        assert candidates.shape[1] == 1
    if case["id"] == "best_effort_only":
        assert not available.any()
        assert np.isfinite(np.asarray(case["artifacts"]["method.best_effort"])).all()


@pytest.mark.parametrize(
    "case", _cases()["invalid_results"], ids=lambda case: case["id"]
)
def test_invalid_or_duplicate_padding_is_rejected(case) -> None:
    candidates, available = _arrays(case)

    with pytest.raises(ValueError, match=case["error"]):
        GenerationResult(candidates, available)


def test_factual_padding_is_rejected_when_bound_to_generation_request() -> None:
    result = GenerationResult(np.array([[[0.1, 0.2]]]), np.array([[True]]))
    with pytest.raises(ValueError, match="factual padding"):
        result.validate_for_factuals(np.array([[0.1, 0.2]]))


@pytest.mark.parametrize(
    "name",
    [
        "best_effort",
        "common.shadow",
        "common.candidates",
        "common.available",
        "candidate.shadow",
        "candidate.grouped_gower",
    ],
)
def test_method_artifacts_cannot_claim_evaluator_owned_namespaces(name) -> None:
    with pytest.raises(ValueError, match=r"method\.\* namespace"):
        GenerationResult(
            np.array([[[np.nan, np.nan]]]),
            np.array([[False]]),
            artifacts={name: np.array([[0.2, 0.3]])},
        )


def test_method_best_effort_artifact_is_accepted() -> None:
    result = GenerationResult(
        np.array([[[np.nan, np.nan]]]),
        np.array([[False]]),
        artifacts={"method.best_effort": np.array([[0.2, 0.3]])},
    )
    np.testing.assert_array_equal(
        result.artifacts["method.best_effort"], np.array([[0.2, 0.3]])
    )


def test_fixture_covers_every_required_generation_state() -> None:
    identifiers = {case["id"] for case in _cases()["cases"]}
    assert identifiers == {
        "returned_valid_k1",
        "returned_invalid",
        "unavailable",
        "best_effort_only",
        "target_class_below_threshold",
        "partial_k3",
    }
