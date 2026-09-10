"""Authentication witness for legacy E3 replay after protocol identity expansion."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.zeroshot_cf.diagnostics import proposal_backends
from experiments.zeroshot_cf.evaluation import EvaluationSpec
from experiments.zeroshot_cf.orchestration.spec import (
    DatasetSpec,
    MethodSpec,
    ProtocolSpec,
    RunSpec,
    TargetModelSpec,
    canonical_json,
)


def _fixture(monkeypatch, tmp_path):
    specs = tuple(
        RunSpec(
            DatasetSpec("heloc"),
            ProtocolSpec(max_test=1),
            TargetModelSpec(),
            MethodSpec(
                "countercontex",
                params={"foundation": {"backend": backend}},
            ),
            EvaluationSpec(),
            42,
        )
        for backend in ("tabicl", "empirical")
    )
    stored = []
    for spec in specs:
        scientific = deepcopy(spec.scientific_payload())
        scientific["protocol"].pop("factual_partition")
        identity = {
            "scientific_spec": scientific,
            "resolved": {
                "backend": spec.method.params["foundation"]["backend"]
            },
        }
        run_id = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        cell_id = hashlib.sha256(canonical_json(scientific).encode()).hexdigest()
        stored.append(
            SimpleNamespace(
                run_id=run_id,
                manifest={
                    "run_id": run_id,
                    "cell_id": cell_id,
                    "identity": identity,
                    "scientific_spec": scientific,
                },
            )
        )
    config = SimpleNamespace(runs=specs)
    store = SimpleNamespace(root=tmp_path, completed_runs=lambda: tuple(stored))
    monkeypatch.setattr(proposal_backends, "load_matrix_config", lambda _: config)
    monkeypatch.setattr(proposal_backends, "ArtifactStore", lambda _: store)
    return specs, stored


def test_historical_e3_replay_authenticates_then_normalizes_test_partition(
    tmp_path, monkeypatch
) -> None:
    _fixture(monkeypatch, tmp_path)

    pairs = proposal_backends.historical_e3_pairs(tmp_path, "matrix.yaml")

    assert set(pairs) == {("heloc", 42)}
    assert set(pairs[("heloc", 42)]) == {"tabicl", "empirical"}


def test_historical_e3_replay_rejects_identity_drift(tmp_path, monkeypatch) -> None:
    _specs, stored = _fixture(monkeypatch, tmp_path)
    stored[0].manifest["scientific_spec"]["protocol"]["max_test"] = 2

    with pytest.raises(ValueError, match="identity"):
        proposal_backends.historical_e3_pairs(tmp_path, "matrix.yaml")


def test_historical_case_fingerprint_reconstructs_pre_partition_identity() -> None:
    factuals = SimpleNamespace(
        partition="test",
        indices=np.array([0]),
        values=np.array([[0.1]]),
        true_labels=np.array([0]),
    )
    case = SimpleNamespace(
        dataset=SimpleNamespace(
            provenance=SimpleNamespace(fingerprint="dataset-v1"),
            y_test=np.array([0, 1]),
        ),
        factuals=factuals,
        factual_predictions=np.array([0]),
        targets=np.array([1]),
        oracle=SimpleNamespace(classes_=np.array([0, 1])),
        protocol={
            "max_test": 1,
            "test_selection": "first",
            "factual_partition": "test",
            "selection_seed": 42,
            "target_policy": "opposite_classifier_prediction",
            "target_model": {"kind": "fixture"},
            "resolved_target_model": {"kind": "fixture"},
            "target_model_fingerprint": "model-v1",
            "target_model_implementation_fingerprint": "implementation-v1",
        },
    )

    assert proposal_backends.historical_e3_case_id(case) == (
        "847f936423cbfdcbb74a541b5018008a7a70689be322cb256d6e6a87ebc0e56f"
    )
    case.factuals = SimpleNamespace(**{**vars(factuals), "partition": "validation"})
    with pytest.raises(ValueError, match="test factuals"):
        proposal_backends.historical_e3_case_id(case)
