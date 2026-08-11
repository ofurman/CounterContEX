#  Copyright (c) Prior Labs GmbH 2026.

"""Fast paired-analysis tests for the HELOC LOF ablation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from experiments.zeroshot_cf.exp10_lof_ablation import (
    VARIANTS,
    aggregate_results,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_paired_lof_aggregation_requires_and_compares_same_factuals(
    tmp_path: Path,
) -> None:
    """The summary compares plausibility and validity point by point."""
    X_test = np.array([[0.1, 0.2], [0.3, 0.4]])
    targets = np.array([1, 0])
    settings = {
        "lof_first": {
            "lof": [1.0, 0.9],
            "valid": [True, True],
            "X_cf": np.array([[0.7, 0.2], [0.3, 0.8]]),
        },
        "probability_first": {
            "lof": [1.2, 0.8],
            "valid": [True, False],
            "X_cf": np.array([[0.8, 0.2], [0.3, 0.8]]),
        },
    }
    for variant in VARIANTS:
        prefix = tmp_path / variant / "exp9_tabicl_heloc"
        _write_csv(
            prefix.with_name(f"{prefix.name}_metrics.csv"),
            [{"dataset": "heloc", "validity": 1.0}],
        )
        _write_csv(
            prefix.with_name(f"{prefix.name}_points.csv"),
            [
                {
                    "point": index,
                    "target": targets[index],
                    "valid": settings[variant]["valid"][index],
                    "lof_score": settings[variant]["lof"][index],
                }
                for index in range(2)
            ],
        )
        np.savez_compressed(
            prefix.with_name(f"{prefix.name}_arrays.npz"),
            X_test=X_test,
            X_cf=settings[variant]["X_cf"],
            y_target=targets,
        )

    summary = aggregate_results(tmp_path)

    assert summary["n_test"] == 2
    assert summary["n_different_counterfactuals"] == 1
    assert summary["validity_lof_first"] == 1.0
    assert summary["validity_probability_first"] == 0.5
    assert summary["n_lof_first_more_plausible"] == 1
    assert summary["n_probability_first_more_plausible"] == 1
    assert summary["n_valid_only_lof_first"] == 1
