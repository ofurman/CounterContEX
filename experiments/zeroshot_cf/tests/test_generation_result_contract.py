"""Executable v1 fixtures for the canonical generation-result contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "architecture_v1" / "semantic_cases.json"


def _cases() -> dict[str, object]:
    return json.loads(FIXTURE.read_text())


def _arrays(case: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(case["candidates"], dtype=np.float64)
    available = np.asarray(case["available"], dtype=bool)
    return candidates, available


def _validate_generation_result(candidates: np.ndarray, available: np.ndarray) -> None:
    """Reference validation required of the future production validator."""
    if candidates.ndim != 3:
        raise ValueError("candidates must have shape (n_factuals, k, n_features)")
    if available.shape != candidates.shape[:2]:
        raise ValueError("available must have shape (n_factuals, k)")
    if np.any(~np.isfinite(candidates[available])):
        raise ValueError("available slots must be finite")
    if np.any(~np.isnan(candidates[~available])):
        raise ValueError("unavailable slots must contain only NaN")

    # Missing diverse results must remain unavailable instead of being represented by
    # repeated returned rows. A duplicate in k>1 is therefore invalid padding.
    for rows, row_available in zip(candidates, available, strict=True):
        returned = rows[row_available]
        if len(returned) > 1 and len(np.unique(returned, axis=0)) != len(returned):
            raise ValueError("available candidates must not duplicate padding")


@pytest.mark.parametrize("case", _cases()["cases"], ids=lambda case: case["id"])
def test_reasoned_generation_results_obey_canonical_shape_and_padding(case) -> None:
    candidates, available = _arrays(case)

    _validate_generation_result(candidates, available)

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
        _validate_generation_result(candidates, available)


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
