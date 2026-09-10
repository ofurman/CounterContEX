"""Independent synthetic witnesses for the E12 pushforward diagnostic."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.analysis.proposal_pushforward import (
    analyze_diagnostic_bundle,
    hierarchical_bootstrap,
    summarize_action_unit,
    summarize_pushforward,
)
from experiments.zeroshot_cf.core.contracts import FeatureDomains, FeatureSchema
from experiments.zeroshot_cf.diagnostics.proposal_pushforward import (
    read_diagnostic_bundle,
    run_matrix_cell,
    trace_fixed_state_pushforward,
    write_diagnostic_bundle,
)
from experiments.zeroshot_cf.evaluation import METRIC_SCHEMA_VERSION
from experiments.zeroshot_cf.methods.countercontex.backends.base import (
    CategoryProposals,
    NumericalDistribution,
)


def _record(
    action_type: str,
    action_unit: str,
    delta: float,
    mass: float,
    *,
    no_op: bool = False,
    source_index: int = 0,
    dataset: str = "mixed",
    classifier_id: str = "classifier",
    sampling_bank: str = "deterministic",
    sampling_seed: int | None = None,
) -> dict:
    before = 0.2
    return {
        "dataset": dataset,
        "classifier_id": classifier_id,
        "factual_partition": "validation",
        "factual_source_index": source_index,
        "target": 1,
        "state_id": "factual",
        "action_type": action_type,
        "action_unit": action_unit,
        "sampling_bank": sampling_bank,
        "sampling_seed": sampling_seed,
        "q_mass": mass,
        "no_op": no_op,
        "duplicate_of": None,
        "unique_row_id": 1,
        "projection_displacement": 0.0,
        "target_probability_before": before,
        "target_probability_after": before + delta,
        "target_class_after": before + delta >= 0.5,
        "crosses_generation_threshold": before < 0.5 <= before + delta,
        "crosses_evaluation_threshold": before < 0.7 <= before + delta,
    }


def _unit(action_type: str, name: str, delta: float, count: int, **kwargs):
    return [
        _record(action_type, name, delta, 1.0 / count, **kwargs) for _ in range(count)
    ]


def test_trace_is_single_pass_atomic_and_preserves_projection_duplicates() -> None:
    class Session:
        numerical_calls = 0
        categorical_calls = 0

        def numerical_distribution_batch(self, rows, columns, **kwargs):
            self.numerical_calls += 1
            quantiles = np.asarray(kwargs["quantiles"])
            values = np.array([[[0.1, 0.3, 0.9]]])
            return NumericalDistribution(quantiles, values, np.zeros_like(values))

        def categorical_distribution(self, row, group, *, confidence):
            self.categorical_calls += 1
            return CategoryProposals(np.array([0, 1]), np.array([0.25, 0.75]))

    class Classifier:
        classes_ = np.array([0, 1])
        calls = 0

        def predict_proba(self, rows):
            self.calls += 1
            probability = 0.1 + 0.6 * np.asarray(rows)[:, 0]
            return np.column_stack((1.0 - probability, probability))

    group = OneHotActionGroup("segment", (1, 2))
    schema = FeatureSchema(
        names=("amount", "segment_a", "segment_b"),
        numerical=(0,),
        categorical_groups=(group,),
        actionable_scalars=(0,),
        actionable_groups=(group,),
        immutable=(),
        domains=FeatureDomains(np.zeros(3), np.ones(3), {0: np.array([0.2, 0.8])}),
    )
    session = Session()
    classifier = Classifier()
    records = trace_fixed_state_pushforward(
        session,
        classifier,
        np.array([0.2, 1.0, 0.0]),
        0,
        schema,
        numerical_quantiles=(0.2, 0.5, 0.8),
        factual_source_index=12,
    )

    assert session.numerical_calls == session.categorical_calls == 1
    assert classifier.calls == 1
    numerical = [record for record in records if record["action_type"] == "numerical"]
    assert [record["projected_value"] for record in numerical] == [0.2, 0.2, 0.8]
    assert [record["no_op"] for record in numerical] == [True, True, False]
    assert numerical[1]["duplicate_of"] == numerical[0]["record_id"]
    assert all(record["target"] == 0 for record in records)
    assert numerical[-1]["delta_probability"] < 0


@pytest.mark.parametrize(
    ("categorical_delta", "numerical_delta", "expected_sign"),
    [(0.6, 0.1, 1), (0.1, 0.6, -1), (0.3, 0.3, 0)],
)
def test_factual_estimand_detects_categorical_numerical_and_tied_cases(
    categorical_delta, numerical_delta, expected_sign
) -> None:
    records = [
        *_unit("categorical", "category", categorical_delta, 2),
        *_unit("numerical", "number", numerical_delta, 2),
    ]
    estimate = summarize_pushforward(records)[0]["categorical_minus_numerical"]
    assert np.sign(estimate) == expected_sign


def test_action_units_are_equal_weighted_despite_unequal_cardinality() -> None:
    records = [
        *_unit("categorical", "binary", 0.8, 2),
        *_unit("categorical", "four_way", 0.2, 4),
        *_unit("numerical", "number", 0.1, 2),
    ]
    summary = summarize_pushforward(records)[0]
    assert summary["conditional_abs_delta_by_type"]["categorical"] == pytest.approx(0.5)
    assert summary["categorical_minus_numerical"] == pytest.approx(0.4)


def test_best_of_b_is_the_q_order_statistic_not_the_observed_maximum() -> None:
    records = [
        _record("numerical", "number", 0.1, 0.5),
        _record("numerical", "number", 0.5, 0.5),
    ]
    summary = summarize_action_unit(records, best_of_b_budgets=(2,))
    assert summary["expected_best_abs_delta_by_budget"]["2"] == pytest.approx(0.4)


def test_all_no_op_and_heloc_like_rows_are_not_estimable() -> None:
    all_no_op = [
        _record("categorical", "category", 0.0, 1.0, no_op=True),
        _record("numerical", "number", 0.0, 1.0, no_op=True),
    ]
    no_op_summary = summarize_pushforward(all_no_op)[0]
    assert no_op_summary["categorical_minus_numerical"] is None
    assert no_op_summary["not_estimable_units_by_type"] == {
        "categorical": 1,
        "numerical": 1,
    }

    heloc_records = _unit("numerical", "number", 0.2, 2, dataset="heloc")
    heloc = summarize_pushforward(heloc_records)[0]
    assert heloc["categorical_minus_numerical"] is None
    assert heloc["unconditional_abs_delta_by_type"]["numerical"] == pytest.approx(0.2)


def test_hierarchical_bootstrap_resamples_factuals_inside_datasets() -> None:
    records = []
    for dataset, source_index, effect in (("a", 0, 0.2), ("a", 1, 0.4), ("b", 0, 0.8)):
        records.extend(
            _unit(
                "categorical",
                "category",
                effect + 0.1,
                2,
                dataset=dataset,
                source_index=source_index,
            )
        )
        records.extend(
            _unit(
                "numerical",
                "number",
                0.1,
                2,
                dataset=dataset,
                source_index=source_index,
            )
        )
    summaries = summarize_pushforward(records)
    first = hierarchical_bootstrap(summaries, draws=200, seed=19)
    second = hierarchical_bootstrap(summaries, draws=200, seed=19)
    assert first == second
    assert first["strata"][0]["datasets"] == 2
    assert first["strata"][0]["estimate"] == pytest.approx(0.55)


def test_sampling_banks_are_averaged_within_factual_and_classifiers_stay_separate() -> (
    None
):
    records = []
    for sampling_seed, effect in zip((17, 42, 101), (0.1, 0.3, 0.5), strict=True):
        records.extend(
            _unit(
                "numerical",
                "number",
                0.1,
                2,
                sampling_bank="mc-0",
                sampling_seed=sampling_seed,
            )
        )
        records.extend(
            _unit(
                "categorical",
                "category",
                effect,
                2,
                sampling_bank="exact",
                sampling_seed=sampling_seed,
            )
        )
    records.extend(
        _unit(
            "categorical",
            "category",
            0.9,
            2,
            classifier_id="other",
        )
    )
    records.extend(_unit("numerical", "number", 0.1, 2, classifier_id="other"))
    summaries = summarize_pushforward(records)
    assert len(summaries) == 2
    by_classifier = {summary["classifier_id"]: summary for summary in summaries}
    assert by_classifier["classifier"]["categorical_minus_numerical"] == pytest.approx(
        0.2
    )
    bootstrap = hierarchical_bootstrap(summaries, draws=20, seed=3)
    assert {row["classifier_id"] for row in bootstrap["strata"]} == {
        "classifier",
        "other",
    }
    assert bootstrap["strata"][0]["dataset_estimates"][0]["dataset"] == "mixed"


def test_fake_cli_is_byte_stable_and_bundle_reader_rejects_drift(tmp_path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    environment = {**os.environ, "SOURCE_DATE_EPOCH": "0"}
    command = [
        sys.executable,
        "-m",
        "experiments.zeroshot_cf.diagnostics.proposal_pushforward",
        "fake",
        "--output",
    ]
    subprocess.run([*command, str(left)], check=True, env=environment)
    subprocess.run([*command, str(right)], check=True, env=environment)
    assert {path.name: path.read_bytes() for path in left.iterdir()} == {
        path.name: path.read_bytes() for path in right.iterdir()
    }

    metadata, records = read_diagnostic_bundle(
        left, expected_metadata={"case_id": "synthetic-case-v1"}
    )
    assert len(records) == metadata["record_count"] == 4
    analyzed = analyze_diagnostic_bundle(left)
    assert analyzed[0]["dataset"] == "synthetic"
    assert analyzed[0]["classifier_id"] == "synthetic-linear-v1"
    with pytest.raises(ValueError, match="metadata mismatch"):
        read_diagnostic_bundle(left, expected_metadata={"case_id": "mutated"})

    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.zeroshot_cf.diagnostics.proposal_pushforward",
            "verify",
            str(left),
            "--expected-metadata",
            str(expected_path),
        ],
        check=True,
        env=environment,
    )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        write_diagnostic_bundle(existing, metadata, records)
    with pytest.raises(ValueError, match="must be finite"):
        write_diagnostic_bundle(
            tmp_path / "nonfinite", {**metadata, "case_id": float("nan")}, records
        )

    with pytest.raises(ValueError, match="expected record count"):
        write_diagnostic_bundle(tmp_path / "incomplete", metadata, records[:-1])

    forged_metadata = dict(metadata)
    forged_metadata["expected_record_count"] = len(records) - 1
    forged_metadata["action_unit_inventory"] = [
        *metadata["action_unit_inventory"][:-1],
        {
            **metadata["action_unit_inventory"][-1],
            "draw_count": 1,
        },
    ]
    forged = tmp_path / "forged"
    write_diagnostic_bundle(forged, forged_metadata, records[:-1])
    with pytest.raises(ValueError, match="metadata mismatch"):
        read_diagnostic_bundle(
            forged,
            expected_metadata={"expected_record_count": len(records)},
        )

    mismatched = tmp_path / "mismatched"
    changed_records = [dict(record) for record in records]
    changed_records[0]["target"] = 0
    write_diagnostic_bundle(mismatched, metadata, changed_records)
    with pytest.raises(ValueError, match="state mismatch"):
        read_diagnostic_bundle(mismatched)

    (left / "proposals.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_diagnostic_bundle(left)


def test_real_matrix_driver_runs_and_resumes_without_repeating_backend_work(
    tmp_path, monkeypatch
) -> None:
    calls = {
        "create": 0,
        "prepare": 0,
        "session": 0,
        "distribution": 0,
        "classifier": 0,
    }

    class Classifier:
        classes_ = np.array([0, 1])

        def predict(self, rows):
            return (np.asarray(rows)[:, 0] >= 0.5).astype(int)

        def predict_proba(self, rows):
            calls["classifier"] += 1
            probability = np.asarray(rows)[:, 0]
            return np.column_stack((1.0 - probability, probability))

    class Session:
        def numerical_distribution_batch(self, rows, columns, **kwargs):
            calls["distribution"] += 1
            quantiles = np.asarray(kwargs["quantiles"], dtype=np.float64)
            values = quantiles[None, None, :]
            return NumericalDistribution(quantiles, values, np.zeros_like(values))

    class Backend:
        capabilities = SimpleNamespace(numerical_distribution=True)

        def for_factual(self, factual, target, *, seed):
            calls["session"] += 1
            return Session()

    class Method:
        def prepare(self, context):
            calls["prepare"] += 1
            return SimpleNamespace(backend=Backend())

    class Registry:
        def create(self, *args, **kwargs):
            calls["create"] += 1
            return Method()

    schema = FeatureSchema(
        names=("amount",),
        numerical=(0,),
        categorical_groups=(),
        actionable_scalars=(0,),
        actionable_groups=(),
        immutable=(),
        domains=FeatureDomains(np.zeros(1), np.ones(1), {}),
    )
    oracle = Classifier()
    dataset = SimpleNamespace(
        X_train=np.array([[0.0], [1.0]]),
        y_train=np.array([0, 1]),
        schema=schema,
    )
    case = SimpleNamespace(
        case_id="case-v1",
        dataset=dataset,
        oracle=oracle,
        factuals=SimpleNamespace(values=np.array([[0.2]]), indices=np.array([4])),
        targets=np.array([1]),
    )
    spec = SimpleNamespace(
        method=SimpleNamespace(name="countercontex", variant="default"),
        dataset=SimpleNamespace(name="synthetic"),
        protocol=SimpleNamespace(factual_partition="validation"),
        evaluation=SimpleNamespace(probability_threshold=0.7),
        seed=17,
        cell_id="cell-v1",
        scientific_payload=lambda: {"cell": "synthetic"},
    )
    runtime = SimpleNamespace(
        params={"foundation": {"backend": "tabicl"}},
        activate=nullcontext,
    )
    versions = SimpleNamespace(
        model_content_id="classifier-v1",
        method_implementation="countercontex-v4",
        backend_implementation="tabicl-proposal-v2",
        checkpoint_content_ids={"classifier": "a", "regressor": "b"},
    )

    class Runner:
        registry = Registry()

        def __init__(self, execution):
            pass

        def _case(self, run_spec):
            return case

        def _method_runtime(self, run_spec):
            return runtime

        def _versions(self, run_spec, loaded_case):
            return versions

        def _method_params(self, run_spec):
            return runtime.params

    matrix = SimpleNamespace(runs=(spec,), execution=SimpleNamespace())
    import experiments.zeroshot_cf.orchestration.matrix as matrix_module
    import experiments.zeroshot_cf.orchestration.runner as runner_module

    monkeypatch.setattr(matrix_module, "load_matrix_config", lambda path: matrix)
    monkeypatch.setattr(runner_module, "GenericRunner", Runner)
    output = tmp_path / "real"
    run_matrix_cell(
        tmp_path / "matrix.yaml",
        0,
        output,
        mc_bank_count=2,
        mc_bank_size=3,
    )
    metadata, records = read_diagnostic_bundle(output / "00000")
    assert metadata["factual_partition"] == "validation"
    assert len(records) == 256 + 2 * 3
    assert calls == {
        "create": 1,
        "prepare": 1,
        "session": 1,
        "distribution": 1,
        "classifier": 1,
    }

    run_matrix_cell(
        tmp_path / "matrix.yaml",
        0,
        output,
        mc_bank_count=2,
        mc_bank_size=3,
    )
    assert calls["create"] == calls["prepare"] == calls["session"] == 1


def test_e12_remains_method_specific_and_evaluation_schema_is_unchanged() -> None:
    assert METRIC_SCHEMA_VERSION == "countercontex.evaluation.v2"
    evaluation_root = Path(__file__).parents[1] / "evaluation"
    sources = "\n".join(path.read_text() for path in evaluation_root.glob("*.py"))
    assert "proposal_pushforward" not in sources
    assert "methods.countercontex" not in sources
