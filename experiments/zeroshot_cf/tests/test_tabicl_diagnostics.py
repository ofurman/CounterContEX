#  Copyright (c) Prior Labs GmbH 2026.

"""Fast checks for the real-model diagnostic comparison logic."""

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.exp8_tabicl_diagnostics import _as_builtin, compare_runs


def _result(X_cf: list[list[float]]) -> dict:
    X_cf = np.asarray(X_cf, dtype=float)
    return {
        "X_cf": X_cf,
        "target_probability": X_cf[:, 0],
        "changed": [[0] for _ in X_cf],
        "flipped": [True for _ in X_cf],
        "steps": [1 for _ in X_cf],
        "runtime_s": 1.0,
    }


def test_compare_runs_reports_equivalent_outputs() -> None:
    """Numerically equivalent outputs pass every paired comparison."""
    reference = _result([[0.1, 0.2], [0.3, 0.4]])
    candidate = _result([[0.1, 0.2], [0.3 + 1e-8, 0.4]])

    row = compare_runs(
        "candidate_batching",
        "batched",
        reference,
        "sequential",
        candidate,
    )

    assert row["cf_allclose"] is True
    assert row["target_probability_allclose"] is True
    assert row["changed_equal"] is True


def test_context_comparison_can_skip_shared_first_fit() -> None:
    """The context test ignores point zero, which uses fit in both modes."""
    reference = _result([[0.1, 0.2], [0.3, 0.4]])
    candidate = _result([[9.0, 9.0], [0.3, 0.4]])

    row = compare_runs(
        "context_update",
        "replace",
        reference,
        "refit",
        candidate,
        start_index=1,
    )

    assert row["n_points"] == 1
    assert row["cf_allclose"] is True


def test_nested_numpy_diagnostic_values_are_json_ready() -> None:
    """Nested arrays, tuples, and numpy scalars convert to JSON primitives."""
    converted = _as_builtin(
        {"array": np.array([1.0]), "history": [[(np.int64(2), np.float32(0.3))]]}
    )

    assert converted == {"array": [1.0], "history": [[[2, np.float32(0.3).item()]]]}
