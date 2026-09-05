"""One typed prepare/generate/evaluate/persist lifecycle for all methods."""

from __future__ import annotations

import hashlib
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from experiments.zeroshot_cf.core.contracts import BenchmarkCase, GenerationRequest
from experiments.zeroshot_cf.datasets.benchmark import method_context
from experiments.zeroshot_cf.evaluation import EvaluationReport, Evaluator
from experiments.zeroshot_cf.methods.registry import (
    DEFAULT_METHOD_REGISTRY,
    MethodRegistry,
    ResolvedMethodRuntime,
)
from experiments.zeroshot_cf.orchestration.artifacts import ArtifactStore, StoredRun
from experiments.zeroshot_cf.orchestration.legacy import (
    ensure_generic_v1,
    export_generic_v1,
    generic_legacy_paths,
)
from experiments.zeroshot_cf.orchestration.spec import (
    ExecutionSpec,
    IdentityVersions,
    RunSpec,
    canonical_json,
    identity_payload,
    run_id,
)

EvaluatorFactory = Callable[[BenchmarkCase, Any], Any]
_SUPPORTED_TARGET_MODEL_NAME = "retained_logistic_regression"
_SUPPORTED_TARGET_MODEL_PARAMS = {"C": 1.0, "max_iter": 1000, "seed": 42}


@dataclass(frozen=True)
class PhaseTimings:
    prepare_s: float
    generate_s: float
    evaluate_s: float
    write_s: float
    total_s: float


@dataclass(frozen=True)
class LoadedCase:
    """Portable case plus optional process-local compatibility objects."""

    case: BenchmarkCase
    runtime_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "runtime_context", MappingProxyType(dict(self.runtime_context))
        )


CaseLoader = Callable[[RunSpec], BenchmarkCase | LoadedCase]


@dataclass(frozen=True)
class RunOutcome:
    spec: RunSpec
    run_id: str
    stored: StoredRun
    timings: PhaseTimings
    skipped: bool = False
    runtime_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "runtime_context", MappingProxyType(dict(self.runtime_context))
        )


def _default_case_loader(spec: RunSpec) -> LoadedCase:
    if (
        spec.target_model.name != _SUPPORTED_TARGET_MODEL_NAME
        or dict(spec.target_model.params) != _SUPPORTED_TARGET_MODEL_PARAMS
    ):
        raise ValueError(
            "default case loader supports only retained_logistic_regression "
            "with params {'C': 1.0, 'max_iter': 1000, 'seed': 42}"
        )
    # CEL is a pinned, local-only checkout rather than a locked workspace
    # dependency. ``uv run`` reconciles the environment before launching, so an
    # editable install performed by vendor_setup is not guaranteed to persist.
    # Expose the verified checkout for the lazy provider imports used below.
    from experiments.zeroshot_cf.vendor_setup import CEL_REPO

    cel_root = str(CEL_REPO.resolve())
    if cel_root not in sys.path:
        sys.path.insert(0, cel_root)

    from experiments.zeroshot_cf.datasets.base import DatasetSpec as ProviderDatasetSpec
    from experiments.zeroshot_cf.datasets.benchmark import build_benchmark_case
    from experiments.zeroshot_cf.datasets.cel import CelDatasetProvider
    from experiments.zeroshot_cf.discriminator import train_discriminator

    params = {
        "validation_fraction": 0.2,
        "drop_heloc_all_minus9": True,
        "split_seed": 42,
        **dict(spec.dataset.params),
        **dict(spec.protocol.params),
    }
    unknown = set(params) - {
        "validation_fraction",
        "drop_heloc_all_minus9",
        "split_seed",
    }
    if unknown:
        raise ValueError(
            f"unsupported default dataset/protocol params: {sorted(unknown)}"
        )
    provider_spec = ProviderDatasetSpec(spec.dataset.name, **params)
    adapter = CelDatasetProvider().prepare_adapter(provider_spec)
    dataset = adapter.prepared
    metadata = dataset.provenance.metadata
    cache_tag = (
        f"{spec.dataset.name}_drop_all_minus9"
        if metadata.get("preprocessing_variant") == "drop_heloc_all_minus9"
        else spec.dataset.name
    )
    if len(dataset.X_validation):
        cache_tag = f"{cache_tag}_{metadata['split_variant']}"
    evaluation_X = dataset.X_validation if len(dataset.X_validation) else dataset.X_test
    evaluation_y = dataset.y_validation if len(dataset.y_validation) else dataset.y_test
    oracle = train_discriminator(
        dataset.X_train,
        dataset.y_train,
        evaluation_X,
        evaluation_y,
        cache_tag,
    )
    return LoadedCase(
        build_benchmark_case(
            dataset,
            oracle,
            max_test=spec.protocol.max_test,
            test_selection=spec.protocol.test_selection,
            seed=42,
            target_model={
                "kind": "logistic_regression",
                "C": 1.0,
                "max_iter": 1000,
                "seed": 42,
                "cache_tag": cache_tag,
            },
        ),
        {"dataset_adapter": adapter, "oracle": oracle},
    )


def _default_evaluator_factory(case: BenchmarkCase, evaluation_spec: Any):
    return Evaluator().prepare(case, evaluation_spec)


class GenericRunner:
    """Execute a matrix while sharing case and evaluator preparation."""

    def __init__(
        self,
        execution: ExecutionSpec,
        *,
        registry: MethodRegistry = DEFAULT_METHOD_REGISTRY,
        case_loader: CaseLoader = _default_case_loader,
        evaluator_factory: EvaluatorFactory = _default_evaluator_factory,
        store: ArtifactStore | None = None,
    ) -> None:
        self.execution = execution
        self.registry = registry
        self.case_loader = case_loader
        self.evaluator_factory = evaluator_factory
        self.store = store or ArtifactStore(execution.output_root)
        self._cases: dict[str, LoadedCase] = {}
        self._evaluators: dict[str, Any] = {}
        self._method_runtimes: dict[str, ResolvedMethodRuntime] = {}

    @staticmethod
    def _case_key(spec: RunSpec) -> str:
        return "|".join(
            (
                spec.dataset.name,
                repr(sorted(spec.dataset.params.items())),
                repr(spec.protocol),
                repr(spec.target_model),
            )
        )

    def _loaded_case(self, spec: RunSpec) -> LoadedCase:
        key = self._case_key(spec)
        if key not in self._cases:
            loaded = self.case_loader(spec)
            self._cases[key] = (
                loaded if isinstance(loaded, LoadedCase) else LoadedCase(loaded)
            )
        return self._cases[key]

    def _case(self, spec: RunSpec) -> BenchmarkCase:
        return self._loaded_case(spec).case

    def _evaluator(self, case: BenchmarkCase, spec: RunSpec):
        key = f"{case.case_id}|{spec.evaluation!r}"
        if key not in self._evaluators:
            self._evaluators[key] = self.evaluator_factory(case, spec.evaluation)
        return self._evaluators[key]

    def _method_runtime(self, spec: RunSpec) -> ResolvedMethodRuntime:
        if spec.cell_id not in self._method_runtimes:
            self._method_runtimes[spec.cell_id] = self.registry.resolve_runtime(
                spec.method.name,
                spec.method.params,
                cache_paths=self.execution.cache_paths,
                device=self.execution.device,
            )
        return self._method_runtimes[spec.cell_id]

    def _versions(self, spec: RunSpec, case: BenchmarkCase) -> IdentityVersions:
        entry = self.registry.entry(spec.method.name)
        target_fingerprint = str(
            case.protocol.get("target_model_fingerprint", case.case_id)
        )
        runtime = self._method_runtime(spec)
        return IdentityVersions(
            dataset_fingerprint=case.dataset.provenance.fingerprint,
            case_fingerprint=case.case_id,
            method_implementation=entry.implementation_version,
            backend_implementation=runtime.backend_implementation,
            model_content_id=target_fingerprint,
            checkpoint_content_ids=runtime.checkpoint_content_ids,
            evaluation_version=spec.evaluation.metric_version,
        )

    def _wandb_log(
        self, spec: RunSpec, resolved_run_id: str, report: EvaluationReport
    ) -> None:
        """Push every metric in ``report`` to Weights & Biases, right where it
        was computed.

        No-op unless the matrix config sets ``wandb.project``. Athena's GPU
        compute nodes are offline, so point them at wandb's own offline mode
        (``WANDB_MODE=offline``) rather than skipping this: runs are written
        locally and uploaded later with a plain ``wandb sync``, no bespoke
        re-upload script required.
        """
        if not self.execution.wandb_project:
            return
        import pandas as pd
        import wandb

        scientific = spec.scientific_payload()
        dataset_name = scientific["dataset"]["name"]
        method_name = scientific["method"]["name"]
        wandb.init(
            project=self.execution.wandb_project,
            entity=self.execution.wandb_entity,
            group=self.execution.wandb_group,
            job_type=method_name,
            id=resolved_run_id,
            resume="allow",
            reinit=True,
            name=(
                f"{dataset_name}-{method_name}-{scientific['method']['variant']}"
                f"-seed{scientific['seed']}"
            ),
            tags=[dataset_name, method_name, f"seed{scientific['seed']}"],
            config={**scientific, "run_id": resolved_run_id, "cell_id": spec.cell_id},
        )
        try:
            wandb.log(dict(report.summary.values))
            point_rows = [{"point": row.point, **row.values} for row in report.points]
            if point_rows:
                wandb.log({"points": wandb.Table(dataframe=pd.DataFrame(point_rows))})
            candidate_rows = [
                {"point": row.point, "rank": row.rank, **row.values}
                for row in report.candidates
            ]
            if candidate_rows:
                wandb.log(
                    {"candidates": wandb.Table(dataframe=pd.DataFrame(candidate_rows))}
                )
            for name, array in report.arrays.values.items():
                flat = array.reshape(-1)
                if flat.size:
                    wandb.log({f"arrays/{name}": wandb.Histogram(flat)})
        finally:
            wandb.finish()

    def _method_params(self, spec: RunSpec) -> dict[str, Any]:
        """Add execution-only backend settings after scientific identity resolves."""
        return dict(self._method_runtime(spec).params)

    @staticmethod
    def _validate_resumed_manifest(
        stored: StoredRun,
        spec: RunSpec,
        identity: dict[str, Any],
        resolved_run_id: str,
    ) -> None:
        manifest = stored.manifest
        scientific = spec.scientific_payload()
        manifest_identity = manifest.get("identity")
        identity_scientific = (
            manifest_identity.get("scientific_spec")
            if isinstance(manifest_identity, dict)
            else None
        )
        derived_run_id = (
            hashlib.sha256(canonical_json(manifest_identity).encode()).hexdigest()
            if isinstance(manifest_identity, dict)
            else None
        )
        if (
            manifest.get("run_id") != resolved_run_id
            or stored.run_id != resolved_run_id
            or manifest.get("cell_id") != spec.cell_id
            or manifest.get("scientific_spec") != scientific
            or identity_scientific != scientific
            or manifest_identity != identity
            or derived_run_id != resolved_run_id
        ):
            raise ValueError("completed manifest identity does not match resolved run")

    def run(self, spec: RunSpec, *, resume: bool | None = None) -> RunOutcome:
        total_started = time.perf_counter()
        loaded_case = self._loaded_case(spec)
        case = loaded_case.case
        evaluator = self._evaluator(case, spec)
        versions = self._versions(spec, case)
        identity = identity_payload(spec, versions)
        resolved_run_id = run_id(spec, versions)
        should_resume = self.execution.resume if resume is None else resume
        if should_resume and (self.store.root / resolved_run_id / "COMPLETE").is_file():
            stored = self.store.read(resolved_run_id)
            self._validate_resumed_manifest(stored, spec, identity, resolved_run_id)
            self._wandb_log(spec, resolved_run_id, stored.report)
            if self.execution.legacy_export:
                ensure_generic_v1(
                    self.execution.output_root,
                    dataset_name=spec.dataset.name,
                    method_name=spec.method.name,
                    case=case,
                    report=stored.report,
                    point_diagnostics=tuple(
                        stored.manifest.get("method_point_diagnostics", ())
                    ),
                    manifest=stored.manifest,
                )
            elapsed = time.perf_counter() - total_started
            return RunOutcome(
                spec,
                resolved_run_id,
                stored,
                PhaseTimings(0.0, 0.0, 0.0, 0.0, elapsed),
                skipped=True,
                runtime_context=loaded_case.runtime_context,
            )

        method = self.registry.create(
            spec.method.name,
            self._method_params(spec),
            variant=spec.method.variant,
        )
        prepare_started = time.perf_counter()
        with self._method_runtime(spec).activate():
            prepared = method.prepare(method_context(case))
        prepare_s = time.perf_counter() - prepare_started

        generate_started = time.perf_counter()
        generated = prepared.generate(
            GenerationRequest(
                factuals=case.factuals.values,
                targets=case.targets,
                n_counterfactuals=spec.method.n_counterfactuals,
                seed=spec.seed,
            )
        )
        generated.validate_for_factuals(case.factuals.values)
        generate_s = time.perf_counter() - generate_started

        evaluate_started = time.perf_counter()
        report: EvaluationReport = evaluator.evaluate(generated)
        evaluate_s = time.perf_counter() - evaluate_started
        self._wandb_log(spec, resolved_run_id, report)

        manifest = {
            "run_id": resolved_run_id,
            "cell_id": spec.cell_id,
            "identity": identity,
            "scientific_spec": spec.scientific_payload(),
            "resolved_method_config": method.config_dict(),
            "execution": {
                **self.execution.manifest_payload(),
                "python": platform.python_version(),
            },
            "timings": {
                "prepare_s": prepare_s,
                "generate_s": generate_s,
                "evaluate_s": evaluate_s,
            },
            "method_run_diagnostics": dict(generated.run_diagnostics),
            "method_point_diagnostics": [
                dict(values) for values in generated.point_diagnostics
            ],
        }
        total_before_write_s = time.perf_counter() - total_started

        def finalize_manifest(values, payload_write_s):
            finalized = dict(values)
            finalized["timings"] = {
                **dict(values["timings"]),
                "write_s": payload_write_s,
                "total_s": total_before_write_s + payload_write_s,
            }
            return finalized

        write_started = time.perf_counter()
        stored = self.store.write(
            resolved_run_id,
            manifest=manifest,
            report=report,
            manifest_finalizer=finalize_manifest,
        )
        if self.execution.legacy_export:
            export_generic_v1(
                self.execution.output_root,
                dataset_name=spec.dataset.name,
                method_name=spec.method.name,
                case=case,
                report=report,
                point_diagnostics=generated.point_diagnostics,
                manifest=stored.manifest,
            )
        write_s = time.perf_counter() - write_started
        total_s = time.perf_counter() - total_started
        return RunOutcome(
            spec,
            resolved_run_id,
            stored,
            PhaseTimings(prepare_s, generate_s, evaluate_s, write_s, total_s),
            runtime_context=loaded_case.runtime_context,
        )

    def run_all(
        self,
        specs: Sequence[RunSpec],
        *,
        resume: bool | None = None,
    ) -> tuple[RunOutcome, ...]:
        if self.execution.legacy_export:
            destinations = [
                generic_legacy_paths(
                    self.execution.output_root,
                    spec.method.name,
                    spec.dataset.name,
                ).metrics_csv
                for spec in specs
            ]
            if len(set(destinations)) != len(destinations):
                raise ValueError(
                    "legacy_export requires at most one run per method and dataset"
                )
        return tuple(self.run(spec, resume=resume) for spec in specs)
