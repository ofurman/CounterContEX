"""Artifact-only analysis contracts."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from experiments.zeroshot_cf.analysis import (
    builders,
    holm_wilcoxon,
    load_published_cells,
)
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config
from experiments.zeroshot_cf.orchestration.spec import canonical_json


def _matrix(path: Path, output: Path, *, probability_threshold: float = 0.7) -> Path:
    path.write_text(
        "\n".join(
            [
                "schema_version: countercontex.matrix.v1",
                "suite: analysis_fixture",
                f"output_root: {output}",
                "datasets: [heloc]",
                "methods: [nice]",
                "seeds: [42]",
                "evaluation:",
                f"  probability_threshold: {probability_threshold}",
                "legacy_export: false",
            ]
        )
        + "\n"
    )
    return path


def _legacy_run(output: Path, config: Path) -> Path:
    scientific = load_matrix_config(config).runs[0].scientific_payload()
    scientific["evaluation"] = {
        **scientific["evaluation"],
        "metric_version": "countercontex.evaluation.v1",
    }
    scientific["evaluation"].pop("detectability_min_cf_rows")
    scientific["evaluation"].pop("gower_neighbor_k")
    identity = {"scientific_spec": scientific, "resolved": {"fixture": "legacy"}}
    run_id = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    cell_id = hashlib.sha256(canonical_json(scientific).encode()).hexdigest()
    run = output / run_id
    run.mkdir(parents=True)
    manifest = {
        "artifact_schema_version": "countercontex.artifacts.v1",
        "evaluation_schema_version": "countercontex.evaluation.v1",
        "run_id": run_id,
        "config": {
            "run_id": run_id,
            "cell_id": cell_id,
            "identity": identity,
            "scientific_spec": scientific,
            "resolved_method_config": {},
            "timings": {},
        },
        "table_types": {
            "summary": {"coverage": "float"},
            "points": {"point": "int"},
            "candidates": {"point": "int", "rank": "int"},
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "summary.csv").write_text('coverage\n"1.0"\n')
    (run / "points.csv").write_text("point\n")
    (run / "candidates.csv").write_text("point,rank\n")
    np.savez(run / "arrays.npz", fixture=np.asarray([1.0]))
    (run / "COMPLETE").touch()
    return run


def test_analysis_rejects_an_incomplete_published_cell(tmp_path):
    output = tmp_path / "results"
    (output / "interrupted-cell").mkdir(parents=True)
    config = _matrix(tmp_path / "matrix.yaml", output)

    with pytest.raises(ValueError, match="partial run directories"):
        load_published_cells(output, config)


def test_legacy_analysis_rejects_changed_probability_threshold(tmp_path):
    output = tmp_path / "results"
    original = _matrix(tmp_path / "original.yaml", output)
    _legacy_run(output, original)
    changed = _matrix(
        tmp_path / "changed.yaml", output, probability_threshold=0.8
    )

    with pytest.raises(ValueError, match="matrix cell mismatch"):
        load_published_cells(output, changed)


def test_legacy_analysis_rejects_tampered_top_level_run_id(tmp_path):
    output = tmp_path / "results"
    config = _matrix(tmp_path / "matrix.yaml", output)
    run = _legacy_run(output, config)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["run_id"] = "tampered"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="identity does not match its run_id"):
        load_published_cells(output, config)


def test_legacy_analysis_rejects_unsupported_evaluation_schema(tmp_path):
    output = tmp_path / "results"
    config = _matrix(tmp_path / "matrix.yaml", output)
    run = _legacy_run(output, config)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_schema_version"] = "countercontex.evaluation.v3"
    scientific = manifest["config"]["identity"]["scientific_spec"]
    scientific["evaluation"]["metric_version"] = "countercontex.evaluation.v3"
    manifest["config"]["scientific_spec"] = scientific
    cell_id = hashlib.sha256(canonical_json(scientific).encode()).hexdigest()
    manifest["config"]["cell_id"] = cell_id
    run_id = hashlib.sha256(
        canonical_json(manifest["config"]["identity"]).encode()
    ).hexdigest()
    manifest["run_id"] = run_id
    manifest["config"]["run_id"] = run_id
    manifest_path.write_text(json.dumps(manifest))
    run.rename(output / run_id)

    with pytest.raises(ValueError, match="unsupported evaluation schema version"):
        load_published_cells(output, config)


def test_analysis_rejects_mixed_evaluation_schema_roots(tmp_path):
    output = tmp_path / "results"
    config = _matrix(tmp_path / "matrix.yaml", output)
    legacy = _legacy_run(output, config)
    current = output / "current-schema-cell"
    shutil.copytree(legacy, current)
    manifest_path = current / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_schema_version"] = "countercontex.evaluation.v2"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="mixed evaluation schema versions"):
        load_published_cells(output, config)


def test_holm_wilcoxon_distinguishes_identical_and_separated_inputs():
    identical = holm_wilcoxon(
        {"same": ([1.0] * 8, [1.0] * 8)}, noise_floor=0.0
    )[0]
    separated = holm_wilcoxon(
        {"apart": ([2.0] * 8, [1.0] * 8)}, noise_floor=0.0
    )[0]

    assert identical.statistic == 0.0
    assert identical.corrected_p == 1.0
    assert not identical.significant
    assert identical.n == 8
    assert separated.corrected_p < 0.05
    assert separated.significant
    assert separated.n == 8


def test_significance_flags_effects_below_the_noise_floor():
    result = holm_wilcoxon(
        {"small": ([1.01] * 8, [1.0] * 8)}, noise_floor=0.02
    )[0]

    assert result.significant
    assert result.below_noise_floor


def test_paper_builders_accept_only_artifact_and_destination_paths():
    names = [
        "build_f3_critical_difference",
        "build_f4_confidence_pareto",
        "build_f5_cost_quality",
        "build_f6_target_probability",
        "build_f7_qualitative_case",
        "build_t1_main",
        "build_t2_diversity",
        "build_t3_backend",
    ]
    for name in names:
        signature = inspect.signature(getattr(builders, name))
        assert tuple(signature.parameters) == (
            "output_root",
            "matrix_config",
            "output_dir",
        )
