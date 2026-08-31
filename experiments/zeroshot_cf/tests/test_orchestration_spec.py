"""Semantic identity tests for generic run specifications."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from experiments.zeroshot_cf.evaluation import EvaluationSpec
from experiments.zeroshot_cf.orchestration.spec import (
    DatasetSpec,
    ExecutionSpec,
    IdentityVersions,
    MethodSpec,
    ProtocolSpec,
    RunSpec,
    TargetModelSpec,
    run_id,
)


def _spec() -> RunSpec:
    return RunSpec(
        DatasetSpec("fixture", {"cleaning": "v1", "fold": 2}),
        ProtocolSpec(5, "first", {"target_policy": "opposite"}),
        TargetModelSpec("oracle", {"C": 1.0}),
        MethodSpec("nice", "base", {"tau": 0.7}),
        EvaluationSpec(probability_threshold=0.7),
        seed=42,
    )


def _versions() -> IdentityVersions:
    return IdentityVersions(
        dataset_fingerprint="dataset-a",
        case_fingerprint="case-a",
        method_implementation="method-a",
        backend_implementation="backend-a",
        model_content_id="model-a",
        checkpoint_content_ids={"proposal": "checkpoint-a"},
    )


def test_run_identity_is_order_independent_and_excludes_execution_metadata() -> None:
    spec = _spec()
    reordered = replace(
        spec,
        dataset=DatasetSpec("fixture", {"fold": 2, "cleaning": "v1"}),
    )
    first = run_id(spec, _versions())
    second = run_id(reordered, _versions())
    assert first == second

    execution_a = ExecutionSpec(
        Path("one"),
        cache_paths={"models": Path("cache-a")},
        device="cpu",
        host="host-a",
    )
    execution_b = ExecutionSpec(
        Path("two"),
        resume=True,
        cache_paths={"models": Path("cache-b")},
        device="cuda",
        host="host-b",
    )
    assert execution_a != execution_b
    assert run_id(spec, _versions()) == first


@pytest.mark.parametrize(
    "params",
    (
        {"cache_dir": "local-cache"},
        {"foundation": {"cache_dir": "local-cache"}},
        {"foundation": {"device": "cuda"}},
        {"runtime": {"host": "worker-1"}},
    ),
)
def test_method_spec_rejects_execution_only_params(params) -> None:
    with pytest.raises(ValueError, match="execution-only"):
        MethodSpec("dicoflex", params=params)


@pytest.mark.parametrize(
    "build",
    (
        lambda: DatasetSpec("fixture", {"source": {"cache_path": "cache"}}),
        lambda: ProtocolSpec(params={"checkpoint_path": "checkpoint.ckpt"}),
        lambda: TargetModelSpec("oracle", {"model_path": "model.bin"}),
        lambda: MethodSpec(
            "dicoflex",
            params={"foundation": {"local_checkpoint_path": "model.ckpt"}},
        ),
    ),
    ids=("dataset", "protocol", "target-model", "nested-method"),
)
def test_scientific_specs_reject_local_execution_paths(build) -> None:
    with pytest.raises(ValueError, match="execution-only"):
        build()


def test_every_scientific_and_content_version_axis_changes_run_identity() -> None:
    spec = _spec()
    versions = _versions()
    baseline = run_id(spec, versions)
    changed_specs = (
        replace(spec, dataset=DatasetSpec("other", spec.dataset.params)),
        replace(spec, protocol=ProtocolSpec(6, "first", spec.protocol.params)),
        replace(spec, target_model=TargetModelSpec("oracle", {"C": 2.0})),
        replace(spec, method=MethodSpec("nice", "base", {"tau": 0.8})),
        replace(spec, evaluation=EvaluationSpec(probability_threshold=0.8)),
        replace(spec, seed=43),
    )
    changed_versions = (
        replace(versions, dataset_fingerprint="dataset-b"),
        replace(versions, case_fingerprint="case-b"),
        replace(versions, method_implementation="method-b"),
        replace(versions, backend_implementation="backend-b"),
        replace(versions, model_content_id="model-b"),
        replace(versions, checkpoint_content_ids={"proposal": "checkpoint-b"}),
        replace(versions, evaluation_version="evaluation-v2"),
    )
    assert all(run_id(changed, versions) != baseline for changed in changed_specs)
    assert all(run_id(spec, changed) != baseline for changed in changed_versions)
