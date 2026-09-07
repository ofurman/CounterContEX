"""Hand-calculated contracts for the clean E3 paired analysis."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.zeroshot_cf.analysis import e3
from experiments.zeroshot_cf.analysis.e3 import _analyse_cells


def _arm(backend, *, available, valid_class, valid_threshold, values):
    run_id = f"run-{backend}"
    scientific = {
        "dataset": {"name": "heloc", "params": {}},
        "protocol": {"max_test": 3, "test_selection": "stratified", "params": {}},
        "target_model": {"name": "retained_mlp", "params": {}},
        "method": {
            "name": "countercontex",
            "variant": "default",
            "n_counterfactuals": 1,
            "params": {"foundation": {"backend": backend}},
        },
        "evaluation": {"probability_threshold": 0.7},
        "seed": 42,
    }
    available = np.asarray(available, dtype=bool)
    points = tuple(
        SimpleNamespace(
            point=point,
            values={
                "factual_label": point % 2,
                "factual_prediction": point % 2,
                "target": 1 - point % 2,
                "available": bool(available[point]),
                "target_probability": values[point] if available[point] else None,
                "valid_class": bool(valid_class[point]),
                "valid_threshold": bool(valid_threshold[point]),
            },
        )
        for point in range(3)
    )
    selected = np.asarray(values, dtype=float)[available]
    arrays = {
        "common.available": available[:, None],
        "candidate.grouped_gower": selected / 10,
        "candidate.action_unit_changes": selected,
        "candidate.lof_score": selected + 10,
        "candidate.isolation_forest_score": selected + 20,
        "candidate.gower_kth_neighbor": selected + 30,
    }
    run = SimpleNamespace(
        manifest={
            "scientific_spec": scientific,
            "identity": {"resolved": {"case_fingerprint": "same-case"}},
        },
        report=SimpleNamespace(
            points=points,
            arrays=SimpleNamespace(values=arrays),
        ),
    )
    cell = {
        "run_id": run_id,
        "cell_id": f"cell-{backend}",
        "artifact_path": f"artifacts/{run_id}",
        "dataset": "heloc",
        "target_model": "retained_mlp",
        "method": "countercontex",
        "method_variant": "default",
        "n_counterfactuals": 1,
        "backend": backend,
        "seed": 42,
        "scientific_group": "fixture",
        "coverage": float(np.mean(available)),
        "set_pairwise_gower_mean": None,
        "timing_total_s": float(values[0]),
    }
    return cell, run


def _fixture():
    arms = (
        _arm(
            "empirical",
            available=[1, 1, 0],
            valid_class=[0, 1, 0],
            valid_threshold=[0, 1, 0],
            values=[1, 2, 3],
        ),
        _arm(
            "empirical_local",
            available=[1, 1, 0],
            valid_class=[1, 1, 0],
            valid_threshold=[1, 1, 0],
            values=[2, 4, 6],
        ),
        _arm(
            "tabicl",
            available=[1, 0, 1],
            valid_class=[1, 0, 1],
            valid_threshold=[1, 0, 1],
            values=[5, 7, 9],
        ),
    )
    return tuple(cell for cell, _run in arms), {
        cell["run_id"]: run for cell, run in arms
    }


def test_e3_pairing_counts_missing_slots_and_conditional_populations():
    cells, runs = _fixture()

    points, contrasts = _analyse_cells(cells, runs)
    learned = [row for row in contrasts if row["contrast"] == "learned_conditional"]
    primary = next(
        row
        for row in learned
        if row["metric"] == "valid_success_rate_threshold_per_requested_slot"
    )
    sparsity = next(
        row for row in learned if row["metric"] == "action_unit_sparsity_mean"
    )
    proximity = next(
        row for row in learned if row["metric"] == "proximity_grouped_gower"
    )

    assert primary == {
        "dataset": "heloc",
        "target_model": "retained_mlp",
        "contrast": "learned_conditional",
        "left_backend": "empirical_local",
        "right_backend": "tabicl",
        "metric": "valid_success_rate_threshold_per_requested_slot",
        "population": "all_requested_factuals",
        "n_total": 3,
        "n_joint_defined": 3,
        "left_mean": pytest.approx(2 / 3),
        "right_mean": pytest.approx(2 / 3),
        "mean_delta": pytest.approx(0),
        "ci_95_low": pytest.approx(-1),
        "ci_95_high": pytest.approx(1),
    }
    assert sparsity["n_joint_defined"] == 1
    assert sparsity["mean_delta"] == pytest.approx(3.0)
    assert proximity["population"] == "jointly_target_class_valid"
    assert proximity["n_joint_defined"] == 1
    assert proximity["mean_delta"] == pytest.approx(0.3)
    assert proximity["ci_95_low"] == pytest.approx(0.3)
    assert proximity["ci_95_high"] == pytest.approx(0.3)
    learned_points = [row for row in points if row["contrast"] == "learned_conditional"]
    assert [
        row["valid_success_rate_threshold_per_requested_slot_delta"]
        for row in learned_points
    ] == [0.0, -1.0, 1.0]
    assert learned_points[1]["action_unit_sparsity_mean_delta"] is None


@pytest.mark.parametrize("failure", ["missing", "duplicate"])
def test_e3_rejects_incomplete_or_duplicate_arms(failure):
    cells, runs = _fixture()
    if failure == "missing":
        cells = cells[:-1]
    else:
        cells = (*cells, cells[0])

    with pytest.raises(ValueError, match="incomplete E3 block|duplicate E3 arm"):
        _analyse_cells(cells, runs)


def test_e3_analysis_is_deterministic_under_input_order():
    cells, runs = _fixture()

    assert _analyse_cells(cells, runs) == _analyse_cells(tuple(reversed(cells)), runs)


def test_e3_builder_writes_deterministic_raw_metrics_and_manifest(
    tmp_path, monkeypatch
):
    cells, runs = _fixture()
    monkeypatch.setattr(e3, "load_published_cells", lambda *_: cells)
    monkeypatch.setattr(
        e3,
        "ArtifactStore",
        lambda _root: SimpleNamespace(read=lambda run_id: runs[run_id]),
    )

    products = e3.build_e3_analysis("artifacts", "matrix.yaml", tmp_path)
    first = {path.name: path.read_bytes() for path in products}
    products = e3.build_e3_analysis("artifacts", "matrix.yaml", tmp_path)

    assert {path.name: path.read_bytes() for path in products} == first
    with (tmp_path / "e3_raw_summaries.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert "coverage" in rows[0]
    assert "timing_total_s" in rows[0]
    assert "set_pairwise_gower_mean" not in rows[0]
    manifest = json.loads((tmp_path / "e3_analysis_manifest.json").read_text())
    assert manifest["uncertainty"] == {
        "confidence": 0.95,
        "method": "within-block paired factual bootstrap percentile interval",
        "resamples": 2000,
        "scope": (
            "selected matched factuals in each dataset-target-model block; not "
            "alternative train/test splits or independent datasets"
        ),
        "seed": 42,
    }
