"""Lifecycle, reuse, resume, and artifact tests for the generic runner."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from experiments.zeroshot_cf.core.contracts import (
    DatasetProvenance,
    FeatureDomains,
    FeatureSchema,
    GenerationResult,
    PreparedDataset,
)
from experiments.zeroshot_cf.datasets.benchmark import build_benchmark_case
from experiments.zeroshot_cf.evaluation import EvaluationSpec, Evaluator
from experiments.zeroshot_cf.methods.registry import MethodRegistry, RegistryEntry
from experiments.zeroshot_cf.orchestration.artifacts import ArtifactStore
from experiments.zeroshot_cf.orchestration.runner import (
    GenericRunner,
    _default_case_loader,
)
from experiments.zeroshot_cf.orchestration.spec import (
    DatasetSpec,
    ExecutionSpec,
    IdentityVersions,
    MethodSpec,
    ProtocolSpec,
    RunSpec,
    TargetModelSpec,
)

_V1_COMPATIBILITY = json.loads(
    (
        Path(__file__).parent / "fixtures" / "architecture_v1" / "compatibility.json"
    ).read_text()
)


class _Oracle:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return (np.asarray(X)[:, 0] >= 0.5).astype(int)

    def predict_proba(self, X):
        probability = np.asarray(X)[:, 0]
        return np.column_stack([1 - probability, probability])


def _case(name: str):
    schema = FeatureSchema(
        names=("amount",),
        numerical=(0,),
        categorical_groups=(),
        actionable_scalars=(0,),
        actionable_groups=(),
        immutable=(),
        domains=FeatureDomains(np.zeros(1), np.ones(1), {}),
    )
    dataset = PreparedDataset(
        name=name,
        X_train=np.array([[0.0], [0.2], [0.8], [1.0]]),
        y_train=np.array([0, 0, 1, 1]),
        X_validation=np.array([[0.1], [0.9]]),
        y_validation=np.array([0, 1]),
        X_test=np.array([[0.1], [0.9]]),
        y_test=np.array([0, 1]),
        schema=schema,
        provenance=DatasetProvenance(
            "fixture", "v1", {"rows": name}, "identity", "fixed", f"fp-{name}"
        ),
    )
    return build_benchmark_case(
        dataset,
        _Oracle(),
        max_test=1,
        test_selection="first",
        target_model={"kind": "fixture", "name": name},
    )


@dataclass(frozen=True)
class _FakeMethod:
    name: str
    calls: list[str]

    def config_dict(self):
        return {"name": self.name}

    def prepare(self, context):
        self.calls.append(f"prepare:{self.name}")
        return _Prepared(self.name, self.calls)


@dataclass(frozen=True)
class _Prepared:
    name: str
    calls: list[str]

    def generate(self, request):
        self.calls.append(f"generate:{self.name}")
        candidates = np.full((len(request.factuals), 1, 1), 0.9)
        return GenerationResult(
            candidates,
            np.ones((len(request.factuals), 1), dtype=bool),
            point_diagnostics=tuple({"method": self.name} for _ in request.factuals),
            run_diagnostics={"seed": request.seed},
        )


def _spec(dataset: str, method: str) -> RunSpec:
    return RunSpec(
        DatasetSpec(dataset),
        ProtocolSpec(1, "first"),
        TargetModelSpec("fixture"),
        MethodSpec(method),
        EvaluationSpec(
            probability_threshold=0.7,
            lof_n_neighbors=2,
            isolation_forest_estimators=5,
        ),
        42,
    )


def test_runner_reuses_cases_and_evaluators_and_writes_four_complete_cells(tmp_path):
    calls: list[str] = []
    loaded: list[str] = []
    evaluator_prepared: list[str] = []
    registry = MethodRegistry(
        tuple(
            RegistryEntry(
                name,
                "fake",
                "Fake",
                "Config",
                f"{name}-v1",
                lambda params, name=name: _FakeMethod(name, calls),
            )
            for name in ("alpha", "beta")
        )
    )

    def case_loader(spec):
        loaded.append(spec.dataset.name)
        return _case(spec.dataset.name)

    def evaluator_factory(case, spec):
        evaluator_prepared.append(case.case_id)
        return Evaluator().prepare(case, spec)

    specs = tuple(
        _spec(dataset, method)
        for dataset in ("one", "two")
        for method in ("alpha", "beta")
    )
    runner = GenericRunner(
        ExecutionSpec(tmp_path),
        registry=registry,
        case_loader=case_loader,
        evaluator_factory=evaluator_factory,
    )
    outcomes = runner.run_all(specs)

    assert loaded == ["one", "two"]
    assert len(evaluator_prepared) == 2
    assert len(outcomes) == 4 and all(not outcome.skipped for outcome in outcomes)
    assert calls.count("prepare:alpha") == 2
    assert calls.count("prepare:beta") == 2
    assert all((outcome.stored.path / "COMPLETE").is_file() for outcome in outcomes)
    assert all(outcome.timings.write_s >= 0 for outcome in outcomes)
    assert not tuple(tmp_path.glob("exp*_metrics.csv"))
    assert all(
        set(outcome.stored.manifest["timings"])
        == {"prepare_s", "generate_s", "evaluate_s", "write_s", "total_s"}
        for outcome in outcomes
    )

    resumed = runner.run_all(specs, resume=True)
    assert all(outcome.skipped for outcome in resumed)
    assert calls.count("generate:alpha") == 2
    rows = ArtifactStore(tmp_path).aggregate_expected([spec.cell_id for spec in specs])
    assert len(rows) == 4


def test_aggregation_rejects_missing_partial_extra_and_identity_drift(tmp_path):
    calls: list[str] = []
    registry = MethodRegistry(
        (
            RegistryEntry(
                "alpha",
                "fake",
                "Fake",
                "Config",
                "alpha-v1",
                lambda params: _FakeMethod("alpha", calls),
            ),
        )
    )
    spec = _spec("one", "alpha")
    runner = GenericRunner(
        ExecutionSpec(tmp_path),
        registry=registry,
        case_loader=lambda spec: _case(spec.dataset.name),
    )
    outcome = runner.run(spec)
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        store.aggregate_expected([spec.cell_id, "missing-cell"])

    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(ValueError, match="partial"):
        store.aggregate_expected([spec.cell_id])
    partial.rmdir()

    manifest_path = outcome.stored.path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["config"]["identity"]["resolved"]["model_content_id"] = "tampered"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity"):
        store.aggregate_expected([spec.cell_id])


def test_runner_routes_dicoflex_execution_settings_outside_identity(
    tmp_path, monkeypatch
):
    captured: dict[str, object] = {}
    calls: list[str] = []

    @dataclass(frozen=True)
    class _RuntimeMethod(_FakeMethod):
        def prepare(self, context):
            from experiments.zeroshot_cf import tabicl_checkpoints

            captured["device_during_prepare"] = tabicl_checkpoints.TABICL_DEVICE
            return super().prepare(context)

    def factory(params):
        captured["params"] = params
        return _RuntimeMethod("dicoflex", calls)

    registry = MethodRegistry(
        (
            RegistryEntry(
                "dicoflex",
                "fake",
                "Fake",
                "Config",
                "dicoflex-v1",
                factory,
            ),
        )
    )
    spec = _spec("one", "dicoflex")
    execution = ExecutionSpec(
        tmp_path,
        cache_paths={"tabicl": tmp_path / "tabicl-cache"},
        device="cpu",
    )
    runner = GenericRunner(
        execution,
        registry=registry,
        case_loader=lambda spec: _case(spec.dataset.name),
    )
    monkeypatch.setattr(
        runner,
        "_versions",
        lambda spec, case: IdentityVersions(
            dataset_fingerprint=case.dataset.provenance.fingerprint,
            case_fingerprint=case.case_id,
            method_implementation="dicoflex-v1",
            backend_implementation="retained-tabicl-v1",
            model_content_id="fixture-model",
        ),
    )
    from experiments.zeroshot_cf import tabicl_checkpoints

    monkeypatch.setattr(tabicl_checkpoints, "TABICL_DEVICE", "auto")

    outcome = runner.run(spec)

    assert captured["params"] == {
        "foundation": {"cache_dir": tmp_path / "tabicl-cache"}
    }
    assert captured["device_during_prepare"] == "cpu"
    assert tabicl_checkpoints.TABICL_DEVICE == "auto"
    assert outcome.stored.manifest["identity"]["scientific_spec"] == (
        spec.scientific_payload()
    )
    assert outcome.stored.manifest["scientific_spec"]["method"]["params"] == {}
    assert outcome.stored.manifest["execution"]["cache_paths"] == {
        "tabicl": str(tmp_path / "tabicl-cache")
    }


@pytest.mark.parametrize("tamper", ("cell_id", "scientific_spec"))
def test_aggregation_rejects_broken_scientific_identity_links(tmp_path, tamper):
    calls: list[str] = []
    registry = MethodRegistry(
        (
            RegistryEntry(
                "alpha",
                "fake",
                "Fake",
                "Config",
                "alpha-v1",
                lambda params: _FakeMethod("alpha", calls),
            ),
        )
    )
    spec = _spec("one", "alpha")
    outcome = GenericRunner(
        ExecutionSpec(tmp_path),
        registry=registry,
        case_loader=lambda spec: _case(spec.dataset.name),
    ).run(spec)
    manifest_path = outcome.stored.path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    if tamper == "cell_id":
        payload["config"]["cell_id"] = "0" * 64
    else:
        payload["config"]["scientific_spec"]["seed"] += 1
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="scientific"):
        ArtifactStore(tmp_path).aggregate_expected([spec.cell_id])


@pytest.mark.parametrize(
    "tamper",
    ("run_id", "cell_id", "scientific_spec", "identity_scientific", "identity"),
)
def test_resume_rejects_every_tampered_manifest_identity_link(tmp_path, tamper):
    calls: list[str] = []
    registry = MethodRegistry(
        (
            RegistryEntry(
                "alpha",
                "fake",
                "Fake",
                "Config",
                "alpha-v1",
                lambda params: _FakeMethod("alpha", calls),
            ),
        )
    )
    spec = _spec("one", "alpha")
    runner = GenericRunner(
        ExecutionSpec(tmp_path),
        registry=registry,
        case_loader=lambda spec: _case(spec.dataset.name),
    )
    outcome = runner.run(spec)
    manifest_path = outcome.stored.path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    config = payload["config"]
    if tamper == "run_id":
        config["run_id"] = "0" * 64
    elif tamper == "cell_id":
        config["cell_id"] = "0" * 64
    elif tamper == "scientific_spec":
        config["scientific_spec"]["seed"] += 1
    elif tamper == "identity_scientific":
        config["identity"]["scientific_spec"]["seed"] += 1
    else:
        config["identity"]["resolved"]["model_content_id"] = "tampered"
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="completed manifest identity"):
        runner.run(spec, resume=True)


@pytest.mark.parametrize(
    "target_model",
    (
        TargetModelSpec("other", {"C": 1.0, "max_iter": 1000, "seed": 42}),
        TargetModelSpec(
            "retained_logistic_regression",
            {"C": 2.0, "max_iter": 1000, "seed": 42},
        ),
        TargetModelSpec("retained_logistic_regression", {"C": 1.0, "max_iter": 1000}),
    ),
)
def test_default_case_loader_rejects_unexecuted_target_model_specs(target_model):
    spec = replace(_spec("one", "alpha"), target_model=target_model)
    with pytest.raises(ValueError, match="supports only"):
        _default_case_loader(spec)


def test_runner_passes_method_variant_to_registry(tmp_path):
    calls: list[str] = []
    registry = MethodRegistry(
        (
            RegistryEntry(
                "alpha",
                "fake",
                "Fake",
                "Config",
                "alpha-v1",
                lambda params: _FakeMethod("alpha", calls),
                ("default", "tuned"),
            ),
        )
    )
    spec = replace(_spec("one", "alpha"), method=MethodSpec("alpha", variant="tuned"))

    outcome = GenericRunner(
        ExecutionSpec(tmp_path),
        registry=registry,
        case_loader=lambda spec: _case(spec.dataset.name),
    ).run(spec)

    assert outcome.stored.manifest["scientific_spec"]["method"]["variant"] == "tuned"


@pytest.mark.parametrize(
    ("method_name", "stem", "expected_keys"),
    (
        (
            "dicoflex",
            "exp9_tabicl",
            {
                "X_test",
                "X_sparse",
                "y_test",
                "X_cf",
                "X_cf_sets",
                "diverse_available_count",
                "y_pred",
                "y_target",
                "y_cf_pred",
            },
        ),
        (
            "nice",
            "exp11_nice_nun",
            {
                "X_test",
                "y_test",
                "X_cf",
                "y_pred",
                "y_target",
                "y_cf_pred",
                "prototypes",
                "prototype_indices",
            },
        ),
        (
            "wachter",
            "exp12_wachter",
            {"X_test", "y_test", "X_cf", "y_pred", "y_target", "y_cf_pred"},
        ),
        (
            "growing_spheres",
            "exp12_growing_spheres",
            {"X_test", "y_test", "X_cf", "y_pred", "y_target", "y_cf_pred"},
        ),
        (
            "dice",
            "exp13_dice_genetic",
            {
                "X_test",
                "y_test",
                "X_cf",
                "y_pred",
                "y_target",
                "y_cf_pred",
                "X_cf_raw",
            },
        ),
        (
            "face",
            "exp14_face_knn",
            {"X_test", "y_test", "X_cf", "y_pred", "y_target", "y_cf_pred"},
        ),
    ),
)
def test_legacy_export_writes_frozen_files_and_resume_restores_without_generation(
    tmp_path, monkeypatch, method_name, stem, expected_keys
):
    calls: list[str] = []

    @dataclass(frozen=True)
    class _LegacyMethod(_FakeMethod):
        def prepare(self, context):
            calls.append(f"prepare:{self.name}")
            return _LegacyPrepared(self.name, calls)

    @dataclass(frozen=True)
    class _LegacyPrepared(_Prepared):
        def generate(self, request):
            calls.append(f"generate:{self.name}")
            candidates = np.full((len(request.factuals), 1, 1), 0.9)
            artifacts = {}
            if self.name == "dicoflex":
                artifacts = {
                    "method.best_effort": candidates[:, 0],
                    "method.sparse_counterfactuals": candidates[:, 0],
                    "method.available_count": np.ones(len(candidates), dtype=int),
                }
            elif self.name == "nice":
                artifacts = {
                    "method.prototypes": candidates[:, 0],
                    "method.prototype_indices": np.zeros(len(candidates), dtype=int),
                }
            elif self.name == "dice":
                artifacts = {"method.raw_candidates": candidates[:, 0]}
            return GenerationResult(
                candidates,
                np.ones((len(candidates), 1), dtype=bool),
                point_diagnostics=tuple({} for _ in candidates),
                artifacts=artifacts,
            )

    variants = (
        ("default", "tabicl_sparse") if method_name == "dicoflex" else ("default",)
    )
    registry = MethodRegistry(
        (
            RegistryEntry(
                method_name,
                "fake",
                "Fake",
                "Config",
                f"{method_name}-v1",
                lambda params: _LegacyMethod(method_name, calls),
                variants,
            ),
        )
    )
    runner = GenericRunner(
        ExecutionSpec(tmp_path, legacy_export=True),
        registry=registry,
        case_loader=lambda spec: _case(spec.dataset.name),
    )
    if method_name == "dicoflex":
        monkeypatch.setattr(
            runner,
            "_versions",
            lambda spec, case: IdentityVersions(
                dataset_fingerprint=case.dataset.provenance.fingerprint,
                case_fingerprint=case.case_id,
                method_implementation="dicoflex-v1",
                backend_implementation="retained-tabicl-v1",
                model_content_id="fixture-model",
            ),
        )
    spec = _spec("one", method_name)

    runner.run(spec)

    prefix = tmp_path / f"{stem}_one"
    metrics = prefix.with_name(f"{prefix.name}_metrics.csv")
    points = prefix.with_name(f"{prefix.name}_points.csv")
    arrays = prefix.with_name(f"{prefix.name}_arrays.npz")
    assert metrics.is_file() and points.is_file() and arrays.is_file()
    contract = _V1_COMPATIBILITY["methods"][method_name]
    with metrics.open(newline="") as handle:
        assert next(csv.reader(handle)) == contract["summary_columns"]
    with points.open(newline="") as handle:
        assert next(csv.reader(handle)) == contract["point_columns"]
    with np.load(arrays, allow_pickle=False) as archive:
        assert set(archive.files) == expected_keys
    metrics.write_text("tampered\n")

    resumed = runner.run(spec, resume=True)

    assert resumed.skipped
    with metrics.open(newline="") as handle:
        assert next(csv.reader(handle)) == contract["summary_columns"]
    points.write_text("tampered\n")

    runner.run(spec, resume=True)

    with points.open(newline="") as handle:
        assert next(csv.reader(handle)) == contract["point_columns"]
    arrays.unlink()

    runner.run(spec, resume=True)

    assert arrays.is_file()
    assert calls.count(f"generate:{method_name}") == 1
    assert len(ArtifactStore(tmp_path).aggregate_expected([spec.cell_id])) == 1
