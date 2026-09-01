"""Lifecycle, reuse, resume, and artifact tests for the generic runner."""

from __future__ import annotations

import csv
import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

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
from experiments.zeroshot_cf.methods.registry import (
    MethodRegistry,
    RegistryEntry,
    ResolvedMethodRuntime,
)
from experiments.zeroshot_cf.orchestration.artifacts import ArtifactStore
from experiments.zeroshot_cf.orchestration.legacy import _legacy_method_id
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
from experiments.zeroshot_cf.orchestration.v1_contract import V1_CONTRACT

_V1_COMPATIBILITY = json.loads(
    (
        Path(__file__).parent / "fixtures" / "architecture_v1" / "compatibility.json"
    ).read_text()
)
_COUNTERCONTEX_V1_SEMANTICS = json.loads(
    (
        Path(__file__).parent
        / "fixtures"
        / "architecture_v1"
        / "countercontex_v1_semantics.json"
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
            "fixture",
            "v1",
            {"rows": name},
            "identity",
            "fixed",
            f"fp-{name}",
            metadata={
                "split_variant": "fixture_split",
                "split_seed": 13,
                "preprocessing_variant": "fixture_clean",
                "n_dropped_rows": 7,
            },
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
    identifiers = {
        (row["dataset"], row["method"], row["method_variant"], row["seed"])
        for row in rows
    }
    assert identifiers == {
        (dataset, method, "default", 42)
        for dataset in ("one", "two")
        for method in ("alpha", "beta")
    }
    assert all(row["n_counterfactuals"] == 1 and row["backend"] is None for row in rows)


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


def test_runner_routes_countercontex_execution_settings_outside_identity(
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
        return _RuntimeMethod("countercontex", calls)

    def runtime_resolver(params, cache_paths, device):
        resolved = dict(params)
        resolved["foundation"] = {"cache_dir": cache_paths["tabicl"]}

        @contextmanager
        def activate():
            from experiments.zeroshot_cf import tabicl_checkpoints

            previous = tabicl_checkpoints.TABICL_DEVICE
            tabicl_checkpoints.TABICL_DEVICE = device
            try:
                yield
            finally:
                tabicl_checkpoints.TABICL_DEVICE = previous

        return ResolvedMethodRuntime(resolved, activate=activate)

    registry = MethodRegistry(
        (
            RegistryEntry(
                "countercontex",
                "fake",
                "Fake",
                "Config",
                "countercontex-v3",
                factory,
                runtime_resolver=runtime_resolver,
            ),
        )
    )
    spec = _spec("one", "countercontex")
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
            method_implementation="countercontex-v3",
            backend_implementation="tabicl-proposal-v1",
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


def test_empirical_countercontex_identity_does_not_require_tabicl_checkpoints(
    tmp_path, monkeypatch
) -> None:
    from experiments.zeroshot_cf import tabicl_checkpoints
    from experiments.zeroshot_cf.methods.registry import DEFAULT_METHOD_REGISTRY

    monkeypatch.setattr(
        tabicl_checkpoints,
        "require_checkpoints",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("empirical backend must not resolve TabICL checkpoints")
        ),
    )
    spec = replace(
        _spec("one", "countercontex"),
        method=MethodSpec(
            "countercontex",
            "tabicl_sparse",
            {"foundation": {"backend": "empirical"}},
        ),
    )
    runner = GenericRunner(
        ExecutionSpec(tmp_path),
        registry=DEFAULT_METHOD_REGISTRY,
        case_loader=lambda spec: _case(spec.dataset.name),
    )

    versions = runner._versions(spec, _case("one"))

    assert versions.method_implementation == "countercontex-v3"
    assert versions.backend_implementation == "empirical-reference-v1"
    assert dict(versions.checkpoint_content_ids) == {}


def test_countercontex_does_not_select_or_resume_old_identity_manifest(
    tmp_path,
) -> None:
    calls: list[str] = []
    pre_rename_name = "dico" + "flex"
    registry = MethodRegistry(
        tuple(
            RegistryEntry(
                name,
                "fake",
                "Fake",
                "Config",
                version,
                lambda params, name=name: _FakeMethod(name, calls),
            )
            for name, version in (
                (pre_rename_name, f"{pre_rename_name}-v3"),
                ("countercontex", "countercontex-v3"),
            )
        )
    )
    runner = GenericRunner(
        ExecutionSpec(tmp_path),
        registry=registry,
        case_loader=lambda spec: _case(spec.dataset.name),
    )
    old_spec = _spec("one", pre_rename_name)
    new_spec = _spec("one", "countercontex")

    old_outcome = runner.run(old_spec)
    new_outcome = runner.run(new_spec, resume=True)

    assert not new_outcome.skipped
    assert old_outcome.stored.path != new_outcome.stored.path
    assert calls.count(f"generate:{pre_rename_name}") == 1
    assert calls.count("generate:countercontex") == 1

    shutil.copyfile(
        old_outcome.stored.path / "manifest.json",
        new_outcome.stored.path / "manifest.json",
    )
    with pytest.raises(ValueError, match="identity|run_id|cell_id"):
        runner.run(new_spec, resume=True)


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
    ("target_model", "error_type", "message"),
    (
        (
            TargetModelSpec("other", {"C": 1.0, "max_iter": 1000, "seed": 42}),
            KeyError,
            "unknown target model",
        ),
        (
            TargetModelSpec(
                "retained_logistic_regression",
                {"C": 2.0, "max_iter": 1000, "seed": 42},
            ),
            ValueError,
            "fixed params",
        ),
        (
            TargetModelSpec(
                "retained_logistic_regression", {"C": 1.0, "max_iter": 1000}
            ),
            ValueError,
            "fixed params",
        ),
    ),
)
def test_default_case_loader_rejects_unknown_or_nonfixed_target_model_specs(
    target_model, error_type, message
):
    spec = replace(_spec("one", "alpha"), target_model=target_model)
    with pytest.raises(error_type, match=message):
        _default_case_loader(spec)


def test_default_case_loader_uses_portable_provider_without_benchmark_runner(
    monkeypatch,
):
    from experiments.zeroshot_cf import discriminator as discriminator_module
    from experiments.zeroshot_cf.datasets import cel as cel_module

    source_case = _case("one")
    captured = {}

    def fake_prepare(self, provider_spec):
        captured["provider_spec"] = provider_spec
        return SimpleNamespace(prepared=source_case.dataset)

    monkeypatch.setattr(cel_module.CelDatasetProvider, "prepare_adapter", fake_prepare)
    def fake_train(*args, **kwargs):
        captured["cache_tag"] = args[4]
        captured["disc_type"] = kwargs["disc_type"]
        return _Oracle()

    monkeypatch.setattr(discriminator_module, "train_discriminator", fake_train)
    spec = replace(
        _spec("one", "alpha"),
        target_model=TargetModelSpec(
            "retained_logistic_regression",
            {"C": 1.0, "max_iter": 1000, "seed": 42},
        ),
    )

    loaded = _default_case_loader(spec)

    assert loaded.case.dataset is source_case.dataset
    assert loaded.case.protocol["test_selection"] == "first"
    assert loaded.runtime_context["dataset_adapter"].prepared is source_case.dataset
    assert isinstance(loaded.runtime_context["oracle"], _Oracle)
    assert captured["provider_spec"].validation_fraction == 0.2
    assert captured["provider_spec"].drop_heloc_all_minus9
    assert captured["provider_spec"].split_seed == 42
    assert captured["cache_tag"] == "one_fixture_split"
    assert captured["disc_type"] == "lr"


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


def test_run_all_rejects_legacy_path_collisions_before_loading_cases(tmp_path):
    calls = []

    def fail_case_loader(spec):
        calls.append(spec.cell_id)
        raise AssertionError("colliding matrix must fail before execution")

    runner = GenericRunner(
        ExecutionSpec(tmp_path, legacy_export=True),
        case_loader=fail_case_loader,
    )
    first = _spec("one", "nice")
    second = replace(first, seed=first.seed + 1)

    with pytest.raises(ValueError, match="at most one run"):
        runner.run_all((first, second))

    assert calls == []


@pytest.mark.parametrize(
    ("n_counterfactuals", "cf_mode", "expected"),
    (
        (1, "sparse", "tabicl_v2_sparse"),
        (1, "data_plausible", "tabicl_v2_data_plausible"),
        (3, "sparse", "tabicl_v2_diverse_dpp"),
    ),
)
def test_countercontex_legacy_method_id_tracks_resolved_mode_and_k(
    n_counterfactuals, cf_mode, expected
):
    assert (
        _legacy_method_id(
            "countercontex",
            V1_CONTRACT["countercontex"],
            {"search": {"cf_mode": cf_mode}},
            n_counterfactuals,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("method_name", "stem", "expected_keys"),
    (
        (
            "countercontex",
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
        def config_dict(self):
            if self.name == "countercontex":
                return {"foundation": {"backend": "tabicl"}}
            return super().config_dict()

        def prepare(self, context):
            calls.append(f"prepare:{self.name}")
            return _LegacyPrepared(self.name, calls)

    @dataclass(frozen=True)
    class _LegacyPrepared(_Prepared):
        def generate(self, request):
            calls.append(f"generate:{self.name}")
            candidates = np.full((len(request.factuals), 1, 1), 0.9)
            artifacts = {}
            if self.name == "countercontex":
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
        ("default", "tabicl_sparse") if method_name == "countercontex" else ("default",)
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
    if method_name == "countercontex":
        monkeypatch.setattr(
            runner,
            "_versions",
            lambda spec, case: IdentityVersions(
                dataset_fingerprint=case.dataset.provenance.fingerprint,
                case_fingerprint=case.case_id,
                method_implementation="countercontex-v3",
                backend_implementation="tabicl-proposal-v1",
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
        reader = csv.DictReader(handle)
        assert reader.fieldnames == contract["summary_columns"]
        metrics_row = next(reader)
    assert metrics_row["method"] == contract["legacy_method_ids"][0]
    assert metrics_row["split_variant"] == "fixture_split"
    assert metrics_row["split_seed"] == "13"
    assert metrics_row["preprocessing_variant"] == "fixture_clean"
    assert metrics_row["n_dropped_rows"] == "7"
    assert float(metrics_row["runtime_generation_s"]) >= 0
    assert float(metrics_row["runtime_total_s"]) > 0
    expected_method_value = {
        "countercontex": ("cf_mode", "sparse"),
        "nice": ("prototype_metric", "euclidean"),
        "wachter": ("model_access", "predict_and_predict_proba"),
        "growing_spheres": ("sphere_candidates", "512"),
        "dice": ("max_iterations", "200"),
        "face": ("graph", "symmetric_knn_actionable_space"),
    }[method_name]
    assert metrics_row[expected_method_value[0]] == expected_method_value[1]
    with points.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == contract["point_columns"]
        point_row = next(reader)
    assert point_row["factual_prediction"] == "0"
    assert point_row["target"] == "1"
    assert point_row["cf_prediction"] == "1"
    assert point_row["valid"] == "True"
    assert point_row["changed_columns"] == "1"
    with np.load(arrays, allow_pickle=False) as archive:
        assert set(archive.files) == expected_keys
        if method_name == "countercontex":
            timing_columns = set(
                _COUNTERCONTEX_V1_SEMANTICS["summary_timing_columns"]
            )
            assert {
                key: value
                for key, value in metrics_row.items()
                if key not in timing_columns
            } == _COUNTERCONTEX_V1_SEMANTICS["summary"]
            assert set(metrics_row) - set(
                _COUNTERCONTEX_V1_SEMANTICS["summary"]
            ) == timing_columns
            assert all(
                isinstance(float(metrics_row[column]), float)
                for column in timing_columns
            )
            assert point_row == _COUNTERCONTEX_V1_SEMANTICS["point"]
            assert {
                key: {
                    "values": archive[key].tolist(),
                    "dtype": str(archive[key].dtype),
                    "shape": list(archive[key].shape),
                }
                for key in archive.files
            } == _COUNTERCONTEX_V1_SEMANTICS["arrays"]
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
    aggregate_row = ArtifactStore(tmp_path).aggregate_expected([spec.cell_id])[0]
    expected_backend = "tabicl" if method_name == "countercontex" else None
    assert aggregate_row["backend"] == expected_backend
