"""External integrity seals for complete canonical campaign runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from experiments.zeroshot_cf.evaluation import EvaluationSpec
from experiments.zeroshot_cf.evaluation.result import (
    METRIC_SCHEMA_VERSION,
    ArrayOutput,
    EvaluationReport,
    SummaryOutput,
)
from experiments.zeroshot_cf.orchestration.artifacts import ArtifactStore
from experiments.zeroshot_cf.orchestration.campaign_inventory import (
    CAMPAIGN_INVENTORY_SCHEMA_VERSION,
    REQUIRED_RUN_FILES,
    ExpectedCampaignRun,
    main,
    publish_campaign_inventory,
    verify_campaign_inventory,
)
from experiments.zeroshot_cf.orchestration.spec import (
    DatasetSpec,
    MethodSpec,
    ProtocolSpec,
    RunSpec,
    TargetModelSpec,
    canonical_json,
)


def _report() -> EvaluationReport:
    return EvaluationReport(
        schema_version=METRIC_SCHEMA_VERSION,
        summary=SummaryOutput(METRIC_SCHEMA_VERSION, {"coverage": 1.0}),
        points=(),
        candidates=(),
        arrays=ArrayOutput(METRIC_SCHEMA_VERSION, {}),
        metadata={},
    )


def _write_run(store: ArtifactStore, *, dataset: str, seed: int) -> ExpectedCampaignRun:
    spec = RunSpec(
        dataset=DatasetSpec(dataset),
        protocol=ProtocolSpec(max_test=1),
        target_model=TargetModelSpec(),
        method=MethodSpec("fixture"),
        evaluation=EvaluationSpec(),
        seed=seed,
    )
    identity = {
        "scientific_spec": spec.scientific_payload(),
        "resolved": {"fixture_content": f"{dataset}-{seed}"},
    }
    resolved_run_id = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    store.write(
        resolved_run_id,
        manifest={
            "run_id": resolved_run_id,
            "cell_id": spec.cell_id,
            "identity": identity,
            "scientific_spec": spec.scientific_payload(),
        },
        report=_report(),
    )
    return ExpectedCampaignRun(cell_id=spec.cell_id, run_id=resolved_run_id)


def _campaign(tmp_path: Path, count: int = 2):
    store = ArtifactStore(tmp_path / "runs")
    expected = tuple(
        _write_run(store, dataset=f"dataset-{index}", seed=index)
        for index in range(count)
    )
    matrix_identity = {
        "schema_version": "countercontex.matrix.v1",
        "suite": "fixture-campaign",
        "profile": "pilot",
    }
    inventory_path = tmp_path / "campaign" / "campaign-inventory.json"
    return store, expected, matrix_identity, inventory_path


def _run_snapshot(store: ArtifactStore) -> dict[str, bytes]:
    return {
        str(path.relative_to(store.root)): path.read_bytes()
        for path in sorted(store.root.glob("*/*"))
        if path.is_file()
    }


def _write_canonical_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def test_campaign_inventory_is_atomic_external_and_verifiable(
    tmp_path, monkeypatch
):
    store, expected, matrix_identity, inventory_path = _campaign(tmp_path)
    before = _run_snapshot(store)
    real_replace = os.replace
    observations = []

    def observed_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == inventory_path:
            assert not inventory_path.exists()
            payload = json.loads(source_path.read_text())
            assert payload["schema_version"] == CAMPAIGN_INVENTORY_SCHEMA_VERSION
            assert len(payload["runs"]) == len(expected)
            observations.append("complete-temporary")
        real_replace(source, destination)

    monkeypatch.setattr(
        "experiments.zeroshot_cf.orchestration.campaign_inventory.os.replace",
        observed_replace,
    )

    published = publish_campaign_inventory(
        store=store,
        expected_runs=expected,
        matrix_identity=matrix_identity,
        destination=inventory_path,
    )

    assert published == inventory_path
    assert observations == ["complete-temporary"]
    assert verify_campaign_inventory(
        store=store,
        expected_runs=expected,
        matrix_identity=matrix_identity,
        inventory_path=inventory_path,
    ) == expected
    assert _run_snapshot(store) == before
    assert not tuple(inventory_path.parent.glob(".*.partial"))

    with pytest.raises(FileExistsError, match="already exists"):
        publish_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=matrix_identity,
            destination=inventory_path,
        )


def test_failed_atomic_publish_leaves_no_inventory_or_temporary(tmp_path, monkeypatch):
    store, expected, matrix_identity, inventory_path = _campaign(tmp_path, count=1)
    before = _run_snapshot(store)

    def fail_publish(_source, _destination):
        raise OSError("publish failure")

    monkeypatch.setattr(
        "experiments.zeroshot_cf.orchestration.campaign_inventory.os.replace",
        fail_publish,
    )
    with pytest.raises(OSError, match="publish failure"):
        publish_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=matrix_identity,
            destination=inventory_path,
        )

    assert not inventory_path.exists()
    assert not tuple(inventory_path.parent.glob(".*.partial"))
    assert _run_snapshot(store) == before


@pytest.mark.parametrize("filename", REQUIRED_RUN_FILES)
def test_verifier_rejects_each_changed_canonical_run_file(tmp_path, filename):
    store, expected, matrix_identity, inventory_path = _campaign(tmp_path, count=1)
    publish_campaign_inventory(
        store=store,
        expected_runs=expected,
        matrix_identity=matrix_identity,
        destination=inventory_path,
    )
    with (store.root / expected[0].run_id / filename).open("ab") as handle:
        handle.write(b"mutation")

    with pytest.raises(
        (FileNotFoundError, ValueError), match="hash|integrity|artifact"
    ):
        verify_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=matrix_identity,
            inventory_path=inventory_path,
        )


def test_publisher_requires_complete_exact_cell_and_run_membership(tmp_path):
    store, expected, matrix_identity, inventory_path = _campaign(tmp_path)
    wrong_run = ExpectedCampaignRun(expected[0].cell_id, "0" * 64)

    invalid_memberships = (
        expected[:1],
        (*expected, expected[0]),
        (wrong_run, expected[1]),
    )
    for invalid in invalid_memberships:
        with pytest.raises(ValueError, match="duplicate|missing|extra|run"):
            publish_campaign_inventory(
                store=store,
                expected_runs=invalid,
                matrix_identity=matrix_identity,
                destination=inventory_path,
            )
        assert not inventory_path.exists()

    incomplete = store.root / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="partial"):
        publish_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=matrix_identity,
            destination=inventory_path,
        )
    assert not inventory_path.exists()


def test_publisher_rejects_mixed_manifest_identity_without_publishing(tmp_path):
    store, expected, matrix_identity, inventory_path = _campaign(tmp_path, count=1)
    manifest_path = store.root / expected[0].run_id / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["config"]["identity"]["resolved"]["fixture_content"] = "foreign"
    _write_canonical_json(manifest_path, payload)

    with pytest.raises(ValueError, match="identity"):
        publish_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=matrix_identity,
            destination=inventory_path,
        )
    assert not inventory_path.exists()


def test_verifier_rejects_mixed_manifest_identity_after_seal(tmp_path):
    store, expected, matrix_identity, inventory_path = _campaign(tmp_path, count=1)
    publish_campaign_inventory(
        store=store,
        expected_runs=expected,
        matrix_identity=matrix_identity,
        destination=inventory_path,
    )
    manifest_path = store.root / expected[0].run_id / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["config"]["identity"]["resolved"]["fixture_content"] = "foreign"
    _write_canonical_json(manifest_path, payload)

    with pytest.raises(ValueError, match="integrity.*identity"):
        verify_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=matrix_identity,
            inventory_path=inventory_path,
        )


def test_verifier_rejects_missing_extra_and_duplicate_members_after_seal(tmp_path):
    store, expected, matrix_identity, inventory_path = _campaign(tmp_path, count=1)
    publish_campaign_inventory(
        store=store,
        expected_runs=expected,
        matrix_identity=matrix_identity,
        destination=inventory_path,
    )

    missing_path = store.root / expected[0].run_id
    held_path = tmp_path / "held-run"
    missing_path.rename(held_path)
    with pytest.raises(ValueError, match="missing"):
        verify_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=matrix_identity,
            inventory_path=inventory_path,
        )
    held_path.rename(missing_path)

    _write_run(store, dataset="extra", seed=99)
    with pytest.raises(ValueError, match="extra"):
        verify_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=matrix_identity,
            inventory_path=inventory_path,
        )

    payload = json.loads(inventory_path.read_text())
    payload["runs"].append(payload["runs"][0])
    _write_canonical_json(inventory_path, payload)
    with pytest.raises(ValueError, match="duplicate|resolved"):
        verify_campaign_inventory(
            store=ArtifactStore(tmp_path / "other-runs"),
            expected_runs=expected,
            matrix_identity=matrix_identity,
            inventory_path=inventory_path,
        )


def test_verifier_rejects_matrix_and_inventory_mutations(tmp_path):
    store, expected, matrix_identity, inventory_path = _campaign(tmp_path, count=1)
    publish_campaign_inventory(
        store=store,
        expected_runs=expected,
        matrix_identity=matrix_identity,
        destination=inventory_path,
    )

    with pytest.raises(ValueError, match="matrix"):
        verify_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity={**matrix_identity, "profile": "confirmation"},
            inventory_path=inventory_path,
        )

    with inventory_path.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="canonical|inventory"):
        verify_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=matrix_identity,
            inventory_path=inventory_path,
        )


def test_inventory_cannot_be_published_inside_a_completed_run(tmp_path):
    store, expected, matrix_identity, _inventory_path = _campaign(tmp_path, count=1)
    before = _run_snapshot(store)
    destination = store.root / expected[0].run_id / "campaign-inventory.json"

    with pytest.raises(ValueError, match="outside completed run directories"):
        publish_campaign_inventory(
            store=store,
            expected_runs=expected,
            matrix_identity=matrix_identity,
            destination=destination,
        )

    assert _run_snapshot(store) == before
    assert not destination.exists()


def test_campaign_inventory_cli_seals_and_verifies_resolved_matrix(
    tmp_path, capsys
):
    store = ArtifactStore(tmp_path / "runs")
    expected = _write_run(store, dataset="dataset-0", seed=0)
    matrix_path = tmp_path / "campaign.yaml"
    matrix_path.write_text(
        "\n".join(
            (
                "schema_version: countercontex.matrix.v1",
                "suite: fixture-campaign",
                f"output_root: {store.root}",
                "datasets: [dataset-0]",
                "methods: [fixture]",
                "seeds: [0]",
                "protocol:",
                "  max_test: 1",
            )
        )
        + "\n"
    )

    assert main(["seal", "--matrix", str(matrix_path)]) == 0
    inventory_path = tmp_path / "runs.campaign-inventory.json"
    assert inventory_path.is_file()
    assert "sealed 1 runs" in capsys.readouterr().out

    assert main(["verify", "--matrix", str(matrix_path)]) == 0
    assert "verified 1 runs" in capsys.readouterr().out
    payload = json.loads(inventory_path.read_text())
    assert payload["runs"][0]["cell_id"] == expected.cell_id
    assert payload["runs"][0]["run_id"] == expected.run_id
