"""One typed prepare/generate/evaluate/persist lifecycle for all methods."""

from __future__ import annotations

import hashlib
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from experiments.zeroshot_cf.core.contracts import BenchmarkCase, GenerationRequest
from experiments.zeroshot_cf.datasets.benchmark import method_context
from experiments.zeroshot_cf.evaluation import EvaluationReport, Evaluator
from experiments.zeroshot_cf.methods.registry import (
    DEFAULT_METHOD_REGISTRY,
    MethodRegistry,
)
from experiments.zeroshot_cf.orchestration.artifacts import ArtifactStore, StoredRun
from experiments.zeroshot_cf.orchestration.legacy import (
    ensure_generic_v1,
    export_generic_v1,
)
from experiments.zeroshot_cf.orchestration.spec import (
    ExecutionSpec,
    IdentityVersions,
    RunSpec,
    canonical_json,
    identity_payload,
    run_id,
)

CaseLoader = Callable[[RunSpec], BenchmarkCase]
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
class RunOutcome:
    spec: RunSpec
    run_id: str
    stored: StoredRun
    timings: PhaseTimings
    skipped: bool = False


def _default_case_loader(spec: RunSpec) -> BenchmarkCase:
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

    from experiments.zeroshot_cf.benchmark_protocol import prepare_benchmark_context

    params = dict(spec.dataset.params)
    params.update(spec.protocol.params)
    context = prepare_benchmark_context(
        spec.dataset.name,
        max_test=(-1 if spec.protocol.max_test is None else spec.protocol.max_test),
        test_selection=spec.protocol.test_selection,
        **params,
    )
    if context.benchmark_case is None:
        raise RuntimeError("generic runner requires a portable benchmark case")
    return context.benchmark_case


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
        self._cases: dict[str, BenchmarkCase] = {}
        self._evaluators: dict[str, Any] = {}
        self._checkpoint_ids: dict[str, str] | None = None

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

    def _case(self, spec: RunSpec) -> BenchmarkCase:
        key = self._case_key(spec)
        if key not in self._cases:
            self._cases[key] = self.case_loader(spec)
        return self._cases[key]

    def _evaluator(self, case: BenchmarkCase, spec: RunSpec):
        key = f"{case.case_id}|{spec.evaluation!r}"
        if key not in self._evaluators:
            self._evaluators[key] = self.evaluator_factory(case, spec.evaluation)
        return self._evaluators[key]

    @staticmethod
    def _dicoflex_backend(spec: RunSpec) -> str | None:
        if spec.method.name != "dicoflex":
            return None
        foundation = spec.method.params.get("foundation", {})
        if not isinstance(foundation, Mapping):
            raise ValueError("dicoflex foundation params must be a mapping")
        backend = foundation.get("backend", "tabicl")
        if backend not in {"tabicl", "empirical"}:
            raise ValueError(f"unknown DiCoFlex proposal backend: {backend!r}")
        return str(backend)

    def _versions(self, spec: RunSpec, case: BenchmarkCase) -> IdentityVersions:
        entry = self.registry.entry(spec.method.name)
        target_fingerprint = str(
            case.protocol.get("target_model_fingerprint", case.case_id)
        )
        proposal_backend = self._dicoflex_backend(spec)
        backend_versions = {
            None: "none-v1",
            "tabicl": "tabicl-proposal-v1",
            "empirical": "empirical-reference-v1",
        }
        backend = backend_versions[proposal_backend]
        checkpoints: dict[str, str] = {}
        if proposal_backend == "tabicl":
            if self._checkpoint_ids is None:
                from experiments.zeroshot_cf import tabicl_checkpoints

                paths = tabicl_checkpoints.require_checkpoints(
                    self.execution.cache_paths.get("tabicl")
                )
                self._checkpoint_ids = {
                    path.name: tabicl_checkpoints._CHECKPOINT_SHA256[path.name]
                    for path in paths
                }
            checkpoints = dict(self._checkpoint_ids)
        return IdentityVersions(
            dataset_fingerprint=case.dataset.provenance.fingerprint,
            case_fingerprint=case.case_id,
            method_implementation=entry.implementation_version,
            backend_implementation=backend,
            model_content_id=target_fingerprint,
            checkpoint_content_ids=checkpoints,
            evaluation_version=spec.evaluation.metric_version,
        )

    def _method_params(self, spec: RunSpec) -> dict[str, Any]:
        """Add execution-only backend settings after scientific identity resolves."""
        params = deepcopy(dict(spec.method.params))
        if self._dicoflex_backend(spec) != "tabicl":
            return params
        cache_dir = self.execution.cache_paths.get("tabicl")
        if cache_dir is not None:
            foundation = dict(params.get("foundation", {}))
            foundation["cache_dir"] = cache_dir
            params["foundation"] = foundation
        return params

    @contextmanager
    def _runtime_environment(self, spec: RunSpec):
        if self._dicoflex_backend(spec) != "tabicl" or self.execution.device is None:
            yield
            return
        # DiCoFlex loads this value lazily while preparing its TabICL backend.
        # ``_versions`` may have imported the checkpoint module first, so set
        # the module value rather than relying on a late environment update.
        from experiments.zeroshot_cf import tabicl_checkpoints

        previous_device = tabicl_checkpoints.TABICL_DEVICE
        try:
            tabicl_checkpoints.TABICL_DEVICE = self.execution.device
            yield
        finally:
            tabicl_checkpoints.TABICL_DEVICE = previous_device

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
        case = self._case(spec)
        evaluator = self._evaluator(case, spec)
        versions = self._versions(spec, case)
        identity = identity_payload(spec, versions)
        resolved_run_id = run_id(spec, versions)
        should_resume = self.execution.resume if resume is None else resume
        if should_resume and (self.store.root / resolved_run_id / "COMPLETE").is_file():
            stored = self.store.read(resolved_run_id)
            self._validate_resumed_manifest(stored, spec, identity, resolved_run_id)
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
            )

        method = self.registry.create(
            spec.method.name,
            self._method_params(spec),
            variant=spec.method.variant,
        )
        prepare_started = time.perf_counter()
        with self._runtime_environment(spec):
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
        if self.execution.legacy_export:
            export_generic_v1(
                self.execution.output_root,
                dataset_name=spec.dataset.name,
                method_name=spec.method.name,
                case=case,
                report=report,
                point_diagnostics=generated.point_diagnostics,
                manifest=manifest,
            )
        stored = self.store.write(
            resolved_run_id,
            manifest=manifest,
            report=report,
            manifest_finalizer=finalize_manifest,
        )
        write_s = time.perf_counter() - write_started
        total_s = time.perf_counter() - total_started
        return RunOutcome(
            spec,
            resolved_run_id,
            stored,
            PhaseTimings(prepare_s, generate_s, evaluate_s, write_s, total_s),
        )

    def run_all(
        self,
        specs: Sequence[RunSpec],
        *,
        resume: bool | None = None,
    ) -> tuple[RunOutcome, ...]:
        return tuple(self.run(spec, resume=resume) for spec in specs)
