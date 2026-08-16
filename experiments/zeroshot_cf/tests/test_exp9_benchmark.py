#  Copyright (c) Prior Labs GmbH 2026.

"""Fast tests for the single-split DiCoFlex-compatible benchmark runner."""

from __future__ import annotations

import csv
from pathlib import Path

from experiments.zeroshot_cf.exp9_dicoflex_benchmark import (
    DATASETS,
    DEFAULT_CANDIDATE_QUANTILES,
    DEFAULT_MAX_TEST,
    aggregate_results,
)


def test_exp9_excludes_adult_and_uses_larger_common_test_set() -> None:
    """The fixed suite contains five suitable datasets and 1,000 factuals."""
    assert "adult" not in DATASETS
    assert DATASETS == (
        "heloc",
        "bank_marketing",
        "give_me_some_credit",
        "lending_club",
        "credit_default",
    )
    assert DEFAULT_MAX_TEST == 1000
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
    assert len(rows) == 5
