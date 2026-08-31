"""Round-trip and completion tests for manifest-backed artifacts."""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.zeroshot_cf.evaluation.result import (
    METRIC_SCHEMA_VERSION,
    ArrayOutput,
    CandidateOutput,
    EvaluationReport,
    PointOutput,
    SummaryOutput,
)
from experiments.zeroshot_cf.orchestration import artifacts as artifact_module
from experiments.zeroshot_cf.orchestration.artifacts import ArtifactStore


def _payload_entries(path):
    return [entry for entry in path.iterdir() if entry.name != ".locks"]


def _report():
    return EvaluationReport(
        schema_version=METRIC_SCHEMA_VERSION,
        summary=SummaryOutput(
            METRIC_SCHEMA_VERSION,
            {"count": 2, "coverage": 0.5, "enabled": True, "label": "fixture"},
        ),
        points=(
            PointOutput(0, {"available": True, "score": 0.75, "note": ""}),
            PointOutput(1, {"available": False, "score": None, "note": None}),
        ),
        candidates=(
            CandidateOutput(0, 0, {"available": True, "prediction": 1, "note": ""}),
            CandidateOutput(
                1, 0, {"available": False, "prediction": None, "note": None}
            ),
        ),
        arrays=ArrayOutput(
            METRIC_SCHEMA_VERSION,
            {"common.candidates": np.array([[[0.2, 0.8]]], dtype=np.float32)},
        ),
        metadata={"primary_rank": 0},
    )


def test_artifact_round_trip_preserves_config_scalars_and_array_types(tmp_path):
    store = ArtifactStore(tmp_path)
    manifest = {
        "method": "fixture",
        "seed": 7,
        "threshold": 0.7,
        "enabled": False,
        "nested": {"ranks": [0, 1]},
    }
    written = store.write("run-1", manifest=manifest, report=_report())
    assert (written.path / "COMPLETE").is_file()
    loaded = store.read("run-1")
    assert (
        loaded.manifest == manifest
        and loaded.report.summary.values == _report().summary.values
    )
    assert isinstance(loaded.report.summary.values["count"], int) and isinstance(
        loaded.report.summary.values["enabled"], bool
    )
    actual = loaded.report.arrays.values["common.candidates"]
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.array([[[0.2, 0.8]]], dtype=np.float32))
    assert loaded.report.points[0].values["note"] == ""
    assert loaded.report.points[1].values["note"] is None
    assert loaded.report.candidates[0].values["note"] == ""
    assert loaded.report.candidates[1].values["note"] is None


def test_complete_marker_is_published_only_after_the_full_directory(
    tmp_path, monkeypatch
):
    store = ArtifactStore(tmp_path)
    real_replace = artifact_module.os.replace
    observations = []

    def observed_replace(source, destination):
        source = artifact_module.Path(source)
        destination = artifact_module.Path(destination)
        if destination == tmp_path / "run":
            assert not (source / "COMPLETE").exists()
            assert {path.name for path in source.iterdir()} == {
                "manifest.json",
                "summary.csv",
                "points.csv",
                "candidates.csv",
                "arrays.npz",
            }
            real_replace(source, destination)
            assert store.completed_runs() == ()
            observations.append("directory")
            return
        if destination == tmp_path / "run" / "COMPLETE":
            assert observations == ["directory"]
            assert all(
                (tmp_path / "run" / name).is_file()
                for name in (
                    "manifest.json",
                    "summary.csv",
                    "points.csv",
                    "candidates.csv",
                    "arrays.npz",
                )
            )
            assert store.completed_runs() == ()
            observations.append("marker")
        real_replace(source, destination)

    monkeypatch.setattr(artifact_module.os, "replace", observed_replace)
    store.write("run", manifest={}, report=_report())
    assert observations == ["directory", "marker"]
    assert [run.run_id for run in store.completed_runs()] == ["run"]


def test_partial_runs_are_ignored_and_not_readable(tmp_path):
    store = ArtifactStore(tmp_path)
    store.write("complete", manifest={"seed": 1}, report=_report())
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "manifest.json").write_text("{}")
    assert [run.run_id for run in store.completed_runs()] == ["complete"]
    with pytest.raises(FileNotFoundError, match="incomplete"):
        store.read("partial")

    fully_written = tmp_path / "fully-written-without-marker"
    shutil.copytree(tmp_path / "complete", fully_written)
    (fully_written / "COMPLETE").unlink()
    assert [run.run_id for run in store.completed_runs()] == ["complete"]


def test_publish_failure_removes_destination_and_hidden_temporaries(
    tmp_path, monkeypatch
):
    store = ArtifactStore(tmp_path)
    real_replace = artifact_module.os.replace

    def fail_marker(source, destination):
        if artifact_module.Path(destination).name == "COMPLETE":
            raise OSError("marker failure")
        return real_replace(source, destination)

    monkeypatch.setattr(artifact_module.os, "replace", fail_marker)
    with pytest.raises(OSError, match="marker failure"):
        store.write("run", manifest={}, report=_report())
    assert not (tmp_path / "run").exists()
    assert _payload_entries(tmp_path) == []


def test_write_failure_cleans_hidden_temporary_directory(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)

    def fail_arrays(*args, **kwargs):
        raise OSError("array failure")

    monkeypatch.setattr(artifact_module.np, "savez_compressed", fail_arrays)
    with pytest.raises(OSError, match="array failure"):
        store.write("run", manifest={}, report=_report())
    assert _payload_entries(tmp_path) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_schema_version", "future", "artifact schema"),
        ("evaluation_schema_version", "future", "evaluation schema"),
        ("run_id", "wrong", "run_id"),
    ],
)
def test_reader_rejects_manifest_schema_or_identity_drift(
    tmp_path, field, value, message
):
    store = ArtifactStore(tmp_path)
    run = store.write("run", manifest={}, report=_report())
    path = run.path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        store.read("run")


@pytest.mark.parametrize(
    "filename",
    ["manifest.json", "summary.csv", "points.csv", "candidates.csv", "arrays.npz"],
)
def test_reader_rejects_every_missing_required_payload_file(tmp_path, filename):
    store = ArtifactStore(tmp_path)
    run = store.write("run", manifest={}, report=_report())
    (run.path / filename).unlink()
    with pytest.raises(ValueError, match="missing required artifact files"):
        store.read("run")


def test_reader_requires_complete_marker(tmp_path):
    store = ArtifactStore(tmp_path)
    run = store.write("run", manifest={}, report=_report())
    (run.path / "COMPLETE").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        store.read("run")


@pytest.mark.parametrize(
    "table_types",
    [
        None,
        {},
        {"summary": {}, "points": {"point": "integer"}, "candidates": {}},
        {"summary": {}, "points": {"point": "int"}, "candidates": {"point": "int"}},
    ],
)
def test_reader_rejects_malformed_table_type_schema(tmp_path, table_types):
    store = ArtifactStore(tmp_path)
    run = store.write("run", manifest={}, report=_report())
    path = run.path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["table_types"] = table_types
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="table type schema"):
        store.read("run")


def test_reader_rejects_table_columns_that_disagree_with_manifest(tmp_path):
    store = ArtifactStore(tmp_path)
    run = store.write("run", manifest={}, report=_report())
    path = run.path / "summary.csv"
    path.write_text("unexpected\n1\n")
    with pytest.raises(ValueError, match="columns do not match"):
        store.read("run")


def test_object_arrays_are_rejected_on_contract_write_and_read(tmp_path):
    with pytest.raises(TypeError, match="object dtype"):
        ArrayOutput(METRIC_SCHEMA_VERSION, {"bad": np.array([None], dtype=object)})

    report = _report()
    fake_report = SimpleNamespace(
        schema_version=report.schema_version,
        summary=report.summary,
        points=report.points,
        candidates=report.candidates,
        arrays=SimpleNamespace(values={"bad": np.array([None], dtype=object)}),
        metadata=report.metadata,
    )
    store = ArtifactStore(tmp_path)
    with pytest.raises(TypeError, match="object dtype"):
        store.write("write-object", manifest={}, report=fake_report)
    assert _payload_entries(tmp_path) == []

    run = store.write("read-object", manifest={}, report=report)
    np.savez_compressed(run.path / "arrays.npz", bad=np.array([None], dtype=object))
    with pytest.raises(ValueError, match="object arrays"):
        store.read("read-object")


def test_manifest_writer_rejects_non_standard_json_numbers(tmp_path):
    store = ArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="JSON compliant"):
        store.write("run", manifest={"score": float("nan")}, report=_report())

    assert _payload_entries(tmp_path) == []


def test_concurrent_same_run_writers_cannot_delete_each_other(tmp_path):
    store = ArtifactStore(tmp_path)
    first_finalizing = Event()
    allow_first_publish = Event()

    def pause_first(manifest, _write_s):
        first_finalizing.set()
        assert allow_first_publish.wait(timeout=5)
        return manifest

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            store.write,
            "run",
            manifest={"writer": 1},
            report=_report(),
            manifest_finalizer=pause_first,
        )
        assert first_finalizing.wait(timeout=5)
        second = executor.submit(
            store.write,
            "run",
            manifest={"writer": 2},
            report=_report(),
        )
        time.sleep(0.05)
        assert not second.done()
        allow_first_publish.set()
        assert first.result(timeout=5).manifest["writer"] == 1
        with pytest.raises(FileExistsError, match="completed run"):
            second.result(timeout=5)

    stored = store.read("run")
    assert stored.manifest["writer"] == 1
    assert {name for name in artifact_module._REQUIRED_FILES} <= {
        path.name for path in stored.path.iterdir()
    }
