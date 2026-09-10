"""Trace fixed-state CounterContEx proposals through a classifier.

The diagnostic is method-specific evidence. It does not modify common evaluator
metrics or the adaptive search policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.candidate_domains import project_candidate_values
from experiments.zeroshot_cf.core.contracts import FeatureDomains, FeatureSchema
from experiments.zeroshot_cf.methods.countercontex.backends.base import (
    CategoryProposals,
    NumericalDistribution,
)

PROPOSAL_DIAGNOSTIC_VERSION = "proposal-diagnostic-v1"
_PAYLOAD_FILES = ("metadata.json", "proposals.json")
_REQUIRED_METADATA = {
    "dataset",
    "cell_id",
    "scientific_spec",
    "case_id",
    "classifier_id",
    "method_implementation",
    "backend_implementation",
    "checkpoint_content_ids",
    "context_fingerprint",
    "factual_partition",
    "factual_source_index",
    "target",
    "state_id",
    "action_schema",
    "policy",
    "rng_key_scheme",
    "generation_threshold",
    "evaluation_threshold",
    "expected_record_count",
    "action_unit_inventory",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("diagnostic JSON values must be finite or explicit null")
    return value


def _encoded_json(value: Any) -> bytes:
    return (
        json.dumps(
            _json_value(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _target_outputs(
    classifier: Any, rows: np.ndarray, target: int
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(classifier.predict_proba(np.atleast_2d(rows)))
    classes = np.asarray(
        getattr(classifier, "classes_", np.arange(probabilities.shape[1]))
    )
    positions = np.flatnonzero(classes == target)
    if len(positions) != 1:
        raise ValueError(f"target class {target} is absent from classifier classes")
    predictions = classes[np.argmax(probabilities, axis=1)]
    return probabilities[:, int(positions[0])], predictions


def _project_numerical(
    values: np.ndarray, column: int, domains: FeatureDomains
) -> np.ndarray:
    return project_candidate_values(
        [column] * len(values),
        values,
        (domains.lower, domains.upper, dict(domains.discrete)),
    )


def _validated_quantile_banks(
    banks: Mapping[str, Sequence[float]],
) -> dict[str, np.ndarray]:
    normalized = {
        str(name): np.asarray(levels, dtype=np.float64)
        for name, levels in banks.items()
    }
    if not normalized or any(
        not name
        or levels.ndim != 1
        or len(levels) == 0
        or np.any(~np.isfinite(levels))
        or np.any((levels <= 0.0) | (levels >= 1.0))
        for name, levels in normalized.items()
    ):
        raise ValueError(
            "numerical quantile banks must contain finite levels in (0, 1)"
        )
    return normalized


def trace_fixed_state_pushforward(
    session: Any,
    classifier: Any,
    factual: np.ndarray,
    target: int,
    schema: FeatureSchema,
    *,
    numerical_quantiles: Sequence[float] | None = None,
    numerical_quantile_banks: Mapping[str, Sequence[float]] | None = None,
    numerical_quantile_bank_factory: (
        Callable[[int, str], Mapping[str, Sequence[float]]] | None
    ) = None,
    sampling_seed: int | None = None,
    generation_threshold: float = 0.5,
    evaluation_threshold: float = 0.7,
    factual_partition: str = "validation",
    factual_source_index: int = 0,
    state_id: str = "factual",
) -> tuple[dict[str, Any], ...]:
    """Trace every actionable unit while calling each proposal/scoring path once."""
    factual = np.asarray(factual, dtype=np.float64)
    if factual.shape != (len(schema.names),):
        raise ValueError("factual must match the feature schema")
    supplied_sources = sum(
        source is not None
        for source in (
            numerical_quantiles,
            numerical_quantile_banks,
            numerical_quantile_bank_factory,
        )
    )
    if supplied_sources != 1:
        raise ValueError("provide exactly one numerical quantile source")
    common_banks = (
        None
        if numerical_quantile_bank_factory is not None
        else _validated_quantile_banks(
            {"deterministic": numerical_quantiles}
            if numerical_quantile_banks is None
            else numerical_quantile_banks
        )
    )

    pending: list[dict[str, Any]] = []
    for column in schema.actionable_scalars:
        normalized_banks = (
            common_banks
            if common_banks is not None
            else _validated_quantile_banks(
                numerical_quantile_bank_factory(column, schema.names[column])
            )
        )
        combined_quantiles = np.concatenate(list(normalized_banks.values()))
        distribution: NumericalDistribution = session.numerical_distribution_batch(
            factual.reshape(1, -1),
            [column],
            quantiles=combined_quantiles,
            confidences=None,
        )
        expected = (1, 1, len(combined_quantiles))
        if distribution.values.shape != expected:
            raise ValueError(
                f"numerical distribution shape drifted: expected {expected}, "
                f"got {distribution.values.shape}"
            )
        raw_values = distribution.values[0, 0]
        projected_values = _project_numerical(raw_values, column, schema.domains)
        offset = 0
        for bank_name, bank_quantiles in normalized_banks.items():
            width = len(bank_quantiles)
            bank_slice = slice(offset, offset + width)
            for draw, (quantile, raw, projected, log_density) in enumerate(
                zip(
                    bank_quantiles,
                    raw_values[bank_slice],
                    projected_values[bank_slice],
                    distribution.log_probabilities[0, 0, bank_slice],
                    strict=True,
                )
            ):
                row = factual.copy()
                row[column] = projected
                pending.append(
                    {
                        "action_unit": schema.names[column],
                        "action_type": "numerical",
                        "action_columns": [int(column)],
                        "cardinality": None,
                        "draw": draw,
                        "sampling_bank": bank_name,
                        "sampling_seed": sampling_seed,
                        "uniform_or_quantile": float(quantile),
                        "category": None,
                        "raw_value": float(raw),
                        "q_mass": float(1.0 / width),
                        "q_log_density": float(log_density)
                        if np.isfinite(log_density)
                        else None,
                        "projected_value": float(projected),
                        "projection_displacement": float(abs(projected - raw)),
                        "projected_row": row,
                    }
                )
            offset += width

    for group in schema.actionable_groups:
        proposal: CategoryProposals = session.categorical_distribution(
            factual, group, confidence=None
        )
        current = int(np.argmax(factual[list(group.columns)]))
        for draw, (category, probability) in enumerate(
            zip(proposal.categories, proposal.probabilities, strict=True)
        ):
            if category >= len(group.columns):
                raise ValueError(f"category {category} is outside group {group.name}")
            row = factual.copy()
            row[list(group.columns)] = 0.0
            row[group.columns[int(category)]] = 1.0
            pending.append(
                {
                    "action_unit": group.name,
                    "action_type": "categorical",
                    "action_columns": [int(column) for column in group.columns],
                    "cardinality": len(proposal.categories),
                    "draw": draw,
                    "sampling_bank": "exact",
                    "sampling_seed": sampling_seed,
                    "uniform_or_quantile": None,
                    "category": int(category),
                    "raw_value": int(category),
                    "q_mass": float(probability),
                    "q_log_density": float(np.log(probability))
                    if probability > 0
                    else None,
                    "projected_value": int(category),
                    "projection_displacement": 0.0,
                    "projected_row": row,
                    "current_category": current,
                }
            )

    factual_key = np.ascontiguousarray(factual).tobytes()
    unique_rows: list[np.ndarray] = [factual]
    row_positions = {factual_key: 0}
    first_record_by_position: dict[int, int] = {}
    for record_id, record in enumerate(pending):
        row = np.asarray(record["projected_row"], dtype=np.float64)
        key = np.ascontiguousarray(row).tobytes()
        if key not in row_positions:
            row_positions[key] = len(unique_rows)
            unique_rows.append(row)
        position = row_positions[key]
        record["unique_row_id"] = position
        record["duplicate_of"] = first_record_by_position.get(position)
        first_record_by_position.setdefault(position, record_id)

    target_probabilities, predictions = _target_outputs(
        classifier, np.stack(unique_rows), target
    )
    before = float(target_probabilities[0])
    records: list[dict[str, Any]] = []
    for record_id, record in enumerate(pending):
        position = int(record["unique_row_id"])
        after = float(target_probabilities[position])
        row = np.asarray(record.pop("projected_row"), dtype=np.float64)
        no_op = position == 0
        if no_op:
            stages = ["proposed", "projected", "no_op", "scored_cached"]
        elif record["duplicate_of"] is not None:
            stages = ["proposed", "projected", "duplicate", "scored_cached"]
        else:
            stages = ["proposed", "projected", "unique", "scored"]
        record.update(
            {
                "record_id": record_id,
                "state_id": state_id,
                "parent_id": None,
                "depth": 0,
                "factual_partition": factual_partition,
                "factual_source_index": int(factual_source_index),
                "target": int(target),
                "projected_row": row.tolist(),
                "no_op": no_op,
                "terminal_disposition": "no_op"
                if no_op
                else ("duplicate" if record["duplicate_of"] is not None else "scored"),
                "classifier_batch": 0,
                "target_probability_before": before,
                "target_probability_after": after,
                "delta_probability": after - before,
                "prediction_after": int(predictions[position]),
                "target_class_after": bool(predictions[position] == target),
                "crosses_generation_threshold": bool(
                    before < generation_threshold <= after
                ),
                "crosses_evaluation_threshold": bool(
                    before < evaluation_threshold <= after
                ),
                "stages": stages,
            }
        )
        records.append(record)
    return tuple(records)


def write_diagnostic_bundle(
    output: Path, metadata: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> Path:
    """Publish an immutable diagnostic bundle with an atomic hash inventory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    missing = sorted(_REQUIRED_METADATA - set(metadata))
    if missing:
        raise ValueError(f"diagnostic metadata is missing required fields: {missing}")
    if metadata["expected_record_count"] != len(records):
        raise ValueError("diagnostic proposal rows do not match expected record count")
    output.mkdir(parents=True)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    frozen_metadata = {
        **dict(metadata),
        "diagnostic_version": PROPOSAL_DIAGNOSTIC_VERSION,
        "created_at": datetime.fromtimestamp(epoch, tz=UTC).isoformat(),
        "record_count": len(records),
    }
    (output / "metadata.json").write_bytes(_encoded_json(frozen_metadata))
    (output / "proposals.json").write_bytes(_encoded_json(list(records)))
    inventory = {
        "diagnostic_version": PROPOSAL_DIAGNOSTIC_VERSION,
        "record_count": len(records),
        "files": {name: _digest(output / name) for name in _PAYLOAD_FILES},
    }
    temporary = output / ".COMPLETE.json.tmp"
    temporary.write_bytes(_encoded_json(inventory))
    temporary.replace(output / "COMPLETE.json")
    return output


def read_diagnostic_bundle(
    directory: Path, *, expected_metadata: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Verify a complete bundle before returning any proposal rows."""
    directory = Path(directory)
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    expected_files = {*_PAYLOAD_FILES, "COMPLETE.json"}
    if actual_files != expected_files:
        raise ValueError("diagnostic bundle has missing or unexpected files")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    complete = json.loads(
        (directory / "COMPLETE.json").read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if complete.get("diagnostic_version") != PROPOSAL_DIAGNOSTIC_VERSION:
        raise ValueError("diagnostic version mismatch")
    for name in _PAYLOAD_FILES:
        if complete.get("files", {}).get(name) != _digest(directory / name):
            raise ValueError(f"diagnostic payload hash mismatch: {name}")
    metadata = json.loads(
        (directory / "metadata.json").read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    records = json.loads(
        (directory / "proposals.json").read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if len(records) != complete.get("record_count") or len(records) != metadata.get(
        "record_count"
    ):
        raise ValueError("diagnostic proposal rows are incomplete")
    if len(records) != metadata.get("expected_record_count"):
        raise ValueError("diagnostic proposal rows do not match expected record count")
    if expected_metadata is not None:
        for key, expected in expected_metadata.items():
            if metadata.get(key) != _json_value(expected):
                raise ValueError(f"diagnostic metadata mismatch: {key}")
    required_ids = list(range(len(records)))
    if [record.get("record_id") for record in records] != required_ids:
        raise ValueError("diagnostic proposal rows are missing or out of order")
    expected_inventory = {
        (item["action_type"], item["action_unit"], item["sampling_bank"]): int(
            item["draw_count"]
        )
        for item in metadata["action_unit_inventory"]
    }
    actual_inventory: dict[tuple[str, str, str], list[int]] = {}
    for record in records:
        key = (
            record["action_type"],
            record["action_unit"],
            record["sampling_bank"],
        )
        actual_inventory.setdefault(key, []).append(int(record["draw"]))
    if set(actual_inventory) != set(expected_inventory) or any(
        sorted(draws) != list(range(expected_inventory[key]))
        for key, draws in actual_inventory.items()
    ):
        raise ValueError("diagnostic proposal action inventory is incomplete")
    identity_fields = ("factual_partition", "factual_source_index", "target")
    for record in records:
        for field in identity_fields:
            if record.get(field) != metadata.get(field):
                raise ValueError(f"diagnostic proposal state mismatch: {field}")
        if record.get("state_id") != metadata.get("state_id"):
            raise ValueError("diagnostic proposal state mismatch: state_id")
    return metadata, tuple(records)


def _context_fingerprint(X: np.ndarray, y: np.ndarray | None) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    if y is not None:
        digest.update(np.ascontiguousarray(y).tobytes())
    return digest.hexdigest()


def _root_inventory(output: Path, point_names: Sequence[str]) -> dict[str, Any]:
    return {
        "diagnostic_version": PROPOSAL_DIAGNOSTIC_VERSION,
        "points": len(point_names),
        "files": {
            "spec.json": _digest(output / "spec.json"),
            **{
                f"{name}/COMPLETE.json": _digest(output / name / "COMPLETE.json")
                for name in point_names
            },
        },
    }


def _verify_cell_output(
    output: Path,
    expected_spec: Mapping[str, Any],
    expected_point_metadata: Sequence[Mapping[str, Any]],
) -> None:
    expected_spec_bytes = _encoded_json(expected_spec)
    if (output / "spec.json").read_bytes() != expected_spec_bytes:
        raise ValueError("diagnostic cell source/spec mismatch")
    point_names = [f"{index:05d}" for index in range(len(expected_point_metadata))]
    actual_entries = {path.name for path in output.iterdir()}
    if actual_entries != {"spec.json", "COMPLETE.json", *point_names}:
        raise ValueError("diagnostic cell has missing or unexpected members")
    for name, metadata in zip(point_names, expected_point_metadata, strict=True):
        read_diagnostic_bundle(output / name, expected_metadata=metadata)
    complete = json.loads((output / "COMPLETE.json").read_text(encoding="utf-8"))
    if complete != _root_inventory(output, point_names):
        raise ValueError("diagnostic cell inventory mismatch")


def run_matrix_cell(
    matrix_path: Path,
    cell_index: int,
    output: Path,
    *,
    mc_bank_count: int | None = None,
    mc_bank_size: int | None = None,
) -> Path:
    """Run or resume one real E12 matrix cell through the prepared TabICL backend."""
    from experiments.zeroshot_cf.datasets.benchmark import method_context
    from experiments.zeroshot_cf.methods.countercontex.proposal_policy import (
        SamplingKey,
        keyed_uniform,
    )
    from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config
    from experiments.zeroshot_cf.orchestration.runner import GenericRunner

    matrix = load_matrix_config(matrix_path)
    if not 0 <= cell_index < len(matrix.runs):
        raise IndexError("diagnostic matrix cell index is out of range")
    spec = matrix.runs[cell_index]
    if spec.method.name != "countercontex":
        raise ValueError("E12 requires the CounterContEx method")
    diagnostic = dict(spec.protocol.params).get("proposal_pushforward")
    required_policy = {
        "diagnostic_version",
        "numerical_integration_points",
        "categorical_policy",
        "mc_bank_count",
        "mc_bank_size",
        "best_of_b_budgets",
    }
    if not isinstance(diagnostic, Mapping) or set(diagnostic) != required_policy:
        raise ValueError(
            "E12 matrix must identify the complete proposal_pushforward policy"
        )
    if (
        diagnostic["diagnostic_version"] != PROPOSAL_DIAGNOSTIC_VERSION
        or diagnostic["numerical_integration_points"] != 256
        or diagnostic["categorical_policy"] != "exact-support"
        or diagnostic["best_of_b_budgets"] != [9, 49]
    ):
        raise ValueError("E12 matrix proposal_pushforward policy is unsupported")
    configured_bank_count = diagnostic["mc_bank_count"]
    configured_bank_size = diagnostic["mc_bank_size"]
    if (
        isinstance(configured_bank_count, bool)
        or not isinstance(configured_bank_count, int)
        or configured_bank_count < 0
        or isinstance(configured_bank_size, bool)
        or not isinstance(configured_bank_size, int)
        or configured_bank_size < 1
    ):
        raise ValueError("Monte Carlo bank count/size must be non-negative/positive")
    if mc_bank_count is not None and mc_bank_count != configured_bank_count:
        raise ValueError("mc_bank_count override conflicts with matrix identity")
    if mc_bank_size is not None and mc_bank_size != configured_bank_size:
        raise ValueError("mc_bank_size override conflicts with matrix identity")
    mc_bank_count = configured_bank_count
    mc_bank_size = configured_bank_size
    runner = GenericRunner(matrix.execution)
    case_protocol = replace(
        spec.protocol,
        params={
            key: value
            for key, value in spec.protocol.params.items()
            if key != "proposal_pushforward"
        },
    )
    case = runner._case(replace(spec, protocol=case_protocol))
    runtime = runner._method_runtime(spec)
    versions = runner._versions(spec, case)
    context = method_context(case)
    if not runtime.params["foundation"]["backend"] == "tabicl":
        raise ValueError("E12 requires the TabICL proposal backend")

    partition = getattr(spec.protocol, "factual_partition", "test")
    integration = (np.arange(256, dtype=np.float64) + 0.5) / 256.0
    bank_names = ["integration-256", *[f"mc-{index}" for index in range(mc_bank_count)]]
    action_inventory = [
        {
            "action_type": "numerical",
            "action_unit": context.feature_schema.names[column],
            "sampling_bank": bank_name,
            "draw_count": 256 if bank_name == "integration-256" else mc_bank_size,
        }
        for column in context.feature_schema.actionable_scalars
        for bank_name in bank_names
    ] + [
        {
            "action_type": "categorical",
            "action_unit": group.name,
            "sampling_bank": "exact",
            "draw_count": len(group.columns),
        }
        for group in context.feature_schema.actionable_groups
    ]
    expected_count = sum(item["draw_count"] for item in action_inventory)
    expected_spec = {
        "diagnostic_version": PROPOSAL_DIAGNOSTIC_VERSION,
        "cell_id": spec.cell_id,
        "scientific_spec": spec.scientific_payload(),
        "mc_bank_count": mc_bank_count,
        "mc_bank_size": mc_bank_size,
    }
    common_metadata = {
        "dataset": spec.dataset.name,
        "cell_id": spec.cell_id,
        "scientific_spec": spec.scientific_payload(),
        "case_id": case.case_id,
        "classifier_id": versions.model_content_id,
        "method_implementation": versions.method_implementation,
        "backend_implementation": versions.backend_implementation,
        "checkpoint_content_ids": dict(versions.checkpoint_content_ids),
        "context_fingerprint": _context_fingerprint(
            context.X_reference, context.y_reference
        ),
        "factual_partition": partition,
        "state_id": "factual",
        "action_schema": {
            "names": list(context.feature_schema.names),
            "numerical": list(context.feature_schema.actionable_scalars),
            "categorical": {
                group.name: list(group.columns)
                for group in context.feature_schema.actionable_groups
            },
            "immutable": list(context.feature_schema.immutable),
        },
        "policy": {
            "numerical": "stratified-icdf-256",
            "categorical": "exact-support",
            "best_of_b_budgets": [9, 49],
            "mc_bank_count": mc_bank_count,
            "mc_bank_size": mc_bank_size,
        },
        "rng_key_scheme": "sha256-canonical-v1",
        "generation_threshold": 0.5,
        "evaluation_threshold": spec.evaluation.probability_threshold,
        "expected_record_count": expected_count,
        "action_unit_inventory": action_inventory,
    }
    expected_point_metadata = [
        {
            **common_metadata,
            "factual_source_index": int(source_index),
            "target": int(target),
        }
        for source_index, target in zip(
            case.factuals.indices, case.targets, strict=True
        )
    ]
    output = Path(output)
    if (output / "COMPLETE.json").is_file():
        _verify_cell_output(output, expected_spec, expected_point_metadata)
        return output
    method = runner.registry.create(
        spec.method.name,
        runner._method_params(spec),
        variant=spec.method.variant,
    )
    with runtime.activate():
        prepared = method.prepare(context)
    if not prepared.backend.capabilities.numerical_distribution:
        raise ValueError("E12 requires portable numerical distributions")
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / "spec.json"
    encoded_spec = _encoded_json(expected_spec)
    if spec_path.exists() and spec_path.read_bytes() != encoded_spec:
        raise ValueError("diagnostic cell source/spec mismatch")
    spec_path.write_bytes(encoded_spec)

    for index, (factual, target, source_index, metadata) in enumerate(
        zip(
            case.factuals.values,
            case.targets,
            case.factuals.indices,
            expected_point_metadata,
            strict=True,
        )
    ):
        point_output = output / f"{index:05d}"
        if point_output.exists():
            read_diagnostic_bundle(point_output, expected_metadata=metadata)
            continue

        def quantile_banks(
            column: int, action_unit: str
        ) -> Mapping[str, Sequence[float]]:
            result: dict[str, Sequence[float]] = {"integration-256": integration}
            for bank in range(mc_bank_count):
                result[f"mc-{bank}"] = [
                    keyed_uniform(
                        SamplingKey(
                            run_seed=spec.seed,
                            factual_source_index=int(source_index),
                            state_depth=0,
                            parent_id="factual",
                            action_unit_id=action_unit,
                            draw_index=draw,
                            stream=f"e12-mc-{bank}",
                        )
                    )
                    for draw in range(mc_bank_size)
                ]
            return result

        with runtime.activate():
            session = prepared.backend.for_factual(
                np.asarray(factual), int(target), seed=spec.seed
            )
            records = trace_fixed_state_pushforward(
                session,
                case.oracle,
                factual,
                int(target),
                context.feature_schema,
                numerical_quantile_bank_factory=quantile_banks,
                sampling_seed=spec.seed,
                generation_threshold=0.5,
                evaluation_threshold=spec.evaluation.probability_threshold,
                factual_partition=partition,
                factual_source_index=int(source_index),
            )
        write_diagnostic_bundle(point_output, metadata, records)

    point_names = [f"{index:05d}" for index in range(len(expected_point_metadata))]
    temporary = output / ".COMPLETE.json.tmp"
    temporary.write_bytes(_encoded_json(_root_inventory(output, point_names)))
    temporary.replace(output / "COMPLETE.json")
    _verify_cell_output(output, expected_spec, expected_point_metadata)
    return output


class _FakeSession:
    def numerical_distribution_batch(self, rows, columns, **kwargs):
        quantiles = np.asarray(kwargs["quantiles"], dtype=np.float64)
        values = (0.2 + 0.6 * quantiles)[None, None, :]
        return NumericalDistribution(quantiles, values, np.zeros_like(values))

    def categorical_distribution(self, row, group, *, confidence):
        del row, group, confidence
        return CategoryProposals(np.array([0, 1]), np.array([0.25, 0.75]))


class _FakeClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, rows):
        rows = np.asarray(rows)
        probability = np.clip(0.05 + 0.45 * rows[:, 0] + 0.35 * rows[:, 2], 0, 1)
        return np.column_stack((1.0 - probability, probability))


def _fake_bundle(output: Path) -> None:
    group = OneHotActionGroup("segment", (1, 2))
    schema = FeatureSchema(
        names=("amount", "segment_a", "segment_b"),
        numerical=(0,),
        categorical_groups=(group,),
        actionable_scalars=(0,),
        actionable_groups=(group,),
        immutable=(),
        domains=FeatureDomains(np.zeros(3), np.ones(3), MappingProxyType({})),
    )
    records = trace_fixed_state_pushforward(
        _FakeSession(),
        _FakeClassifier(),
        np.array([0.2, 1.0, 0.0]),
        1,
        schema,
        numerical_quantiles=(0.25, 0.75),
        factual_source_index=7,
    )
    metadata = {
        "dataset": "synthetic",
        "cell_id": "synthetic-cell-v1",
        "scientific_spec": {"kind": "synthetic"},
        "case_id": "synthetic-case-v1",
        "classifier_id": "synthetic-linear-v1",
        "method_implementation": "countercontex-test-v1",
        "backend_implementation": "synthetic-q-v1",
        "checkpoint_content_ids": {},
        "context_fingerprint": "synthetic-context-v1",
        "factual_partition": "validation",
        "factual_source_index": 7,
        "target": 1,
        "state_id": "factual",
        "action_schema": {"numerical": [0], "categorical": {"segment": [1, 2]}},
        "policy": {"numerical": "grid", "quantiles": [0.25, 0.75]},
        "rng_key_scheme": "sha256-canonical-v1",
        "generation_threshold": 0.5,
        "evaluation_threshold": 0.7,
        "expected_record_count": len(records),
        "action_unit_inventory": [
            {
                "action_type": "numerical",
                "action_unit": "amount",
                "sampling_bank": "deterministic",
                "draw_count": 2,
            },
            {
                "action_type": "categorical",
                "action_unit": "segment",
                "sampling_bank": "exact",
                "draw_count": 2,
            },
        ],
    }
    write_diagnostic_bundle(output, metadata, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fake = subparsers.add_parser("fake", help="write a deterministic offline fixture")
    fake.add_argument("--output", required=True, type=Path)
    run = subparsers.add_parser("run", help="run one real E12 matrix cell")
    run.add_argument("--matrix", required=True, type=Path)
    run.add_argument("--cell-index", required=True, type=int)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--mc-bank-count", type=int)
    run.add_argument("--mc-bank-size", type=int)
    verify = subparsers.add_parser("verify", help="verify an existing bundle")
    verify.add_argument("directory", type=Path)
    verify.add_argument("--expected-metadata", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "fake":
        _fake_bundle(args.output)
    elif args.command == "run":
        run_matrix_cell(
            args.matrix,
            args.cell_index,
            args.output,
            mc_bank_count=args.mc_bank_count,
            mc_bank_size=args.mc_bank_size,
        )
    else:
        expected = json.loads(args.expected_metadata.read_text(encoding="utf-8"))
        missing = sorted(_REQUIRED_METADATA - set(expected))
        if missing:
            raise ValueError(
                f"external expected metadata is missing required fields: {missing}"
            )
        read_diagnostic_bundle(args.directory, expected_metadata=expected)


if __name__ == "__main__":
    main()
