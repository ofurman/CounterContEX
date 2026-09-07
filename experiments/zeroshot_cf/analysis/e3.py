"""Paired, artifact-only analysis for the clean three-arm E3 experiment."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.analysis.core import load_published_cells, write_rows
from experiments.zeroshot_cf.orchestration.artifacts import ArtifactStore, StoredRun
from experiments.zeroshot_cf.orchestration.spec import canonical_json

_BACKENDS = ("empirical", "empirical_local", "tabicl")
_CONTRASTS = (
    ("locality", "empirical", "empirical_local"),
    ("learned_conditional", "empirical_local", "tabicl"),
    ("backend_bundle", "empirical", "tabicl"),
)
_POINT_METRICS = {
    "coverage": ("all_requested_factuals", "available"),
    "primary_coverage": ("all_requested_factuals", "available"),
    "valid_success_rate_class_per_requested_slot": (
        "all_requested_factuals",
        "valid_class",
    ),
    "valid_success_rate_threshold_per_requested_slot": (
        "all_requested_factuals",
        "valid_threshold",
    ),
    "valid_success_rate_class_per_factual": (
        "all_requested_factuals",
        "valid_class",
    ),
    "valid_success_rate_threshold_per_factual": (
        "all_requested_factuals",
        "valid_threshold",
    ),
    "target_probability": ("jointly_available", "target_probability"),
    "proximity_grouped_gower": ("jointly_target_class_valid", "grouped_gower"),
    "action_unit_sparsity_mean": ("jointly_available", "action_unit_changes"),
    "lof_scores_cf": ("jointly_available", "lof_score"),
    "isolation_forest_scores_cf": ("jointly_available", "isolation_forest_score"),
    "plausibility_gower_kth_neighbor_mean": (
        "jointly_available",
        "gower_kth_neighbor",
    ),
}
_BOOTSTRAP_RESAMPLES = 2_000
_BOOTSTRAP_SEED = 42


def _paired_bootstrap_ci(values: Sequence[float | None]) -> tuple[float | None, ...]:
    deltas = np.asarray([np.nan if value is None else value for value in values])
    if not np.isfinite(deltas).any():
        return None, None
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    sampled = deltas[
        rng.integers(0, len(deltas), size=(_BOOTSTRAP_RESAMPLES, len(deltas)))
    ]
    counts = np.isfinite(sampled).sum(axis=1)
    means = np.nansum(sampled, axis=1)[counts > 0] / counts[counts > 0]
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _without_backend(scientific: Mapping[str, Any]) -> str:
    payload = json.loads(canonical_json(scientific))
    payload["method"]["params"]["foundation"].pop("backend", None)
    return canonical_json(payload)


def _point_values(run: StoredRun) -> dict[int, dict[str, float]]:
    points = {point.point: dict(point.values) for point in run.report.points}
    if sorted(points) != list(range(len(points))):
        raise ValueError("E3 points must be unique contiguous factual positions")
    arrays = run.report.arrays.values
    available = np.asarray(arrays["common.available"], dtype=bool)
    if available.shape != (len(points), 1):
        raise ValueError("E3 analysis requires k=1 candidate arrays")
    available_positions = np.flatnonzero(available[:, 0])
    array_names = {
        "grouped_gower": "candidate.grouped_gower",
        "action_unit_changes": "candidate.action_unit_changes",
        "lof_score": "candidate.lof_score",
        "isolation_forest_score": "candidate.isolation_forest_score",
        "gower_kth_neighbor": "candidate.gower_kth_neighbor",
    }
    for output_name, array_name in array_names.items():
        values = np.asarray(arrays[array_name], dtype=float)
        if values.shape != (len(available_positions),):
            raise ValueError(
                f"artifact array {array_name!r} does not match availability"
            )
        for point, value in zip(available_positions, values, strict=True):
            points[int(point)][output_name] = float(value)
    for values in points.values():
        values["available"] = float(bool(values["available"]))
        values["valid_class"] = float(bool(values["valid_class"]))
        values["valid_threshold"] = float(bool(values["valid_threshold"]))
        if values.get("target_probability") is not None:
            values["target_probability"] = float(values["target_probability"])
    return points


def _validate_and_group(
    cells: Sequence[Mapping[str, Any]], runs: Mapping[str, StoredRun]
) -> dict[tuple[str, str], dict[str, tuple[Mapping[str, Any], StoredRun]]]:
    blocks: dict[tuple[str, str], dict[str, tuple[Mapping[str, Any], StoredRun]]] = {}
    for cell in cells:
        if cell["method"] != "countercontex" or cell["n_counterfactuals"] != 1:
            raise ValueError("E3 analysis requires CounterContEx with k=1")
        backend = str(cell["backend"])
        if backend not in _BACKENDS:
            raise ValueError(f"unexpected E3 backend: {backend}")
        block = (str(cell["dataset"]), str(cell["target_model"]))
        arms = blocks.setdefault(block, {})
        if backend in arms:
            raise ValueError(f"duplicate E3 arm for {block}: {backend}")
        arms[backend] = (cell, runs[str(cell["run_id"])])
    for block, arms in blocks.items():
        missing = sorted(set(_BACKENDS) - set(arms))
        if missing:
            raise ValueError(f"incomplete E3 block {block}: missing={missing}")
        stored = [run for _cell, run in arms.values()]
        scientific_specs = {
            _without_backend(run.manifest["scientific_spec"]) for run in stored
        }
        if len(scientific_specs) != 1:
            raise ValueError(f"E3 block {block} differs on more than backend")
        if (
            len(
                {
                    run.manifest["identity"]["resolved"]["case_fingerprint"]
                    for run in stored
                }
            )
            != 1
        ):
            raise ValueError(f"E3 block {block} does not share one benchmark case")
        point_rows = [
            {point.point: dict(point.values) for point in run.report.points}
            for run in stored
        ]
        factual_fields = ("factual_label", "factual_prediction", "target")
        reference = {
            point: tuple(values[name] for name in factual_fields)
            for point, values in point_rows[0].items()
        }
        if any(
            {
                point: tuple(values[name] for name in factual_fields)
                for point, values in rows.items()
            }
            != reference
            for rows in point_rows[1:]
        ):
            raise ValueError(f"E3 block {block} factual positions do not match")
    return blocks


def _analyse_cells(
    cells: Sequence[Mapping[str, Any]], runs: Mapping[str, StoredRun]
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    blocks = _validate_and_group(cells, runs)
    point_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for (dataset, target_model), arms in sorted(blocks.items()):
        values = {backend: _point_values(run) for backend, (_cell, run) in arms.items()}
        for contrast, left_backend, right_backend in _CONTRASTS:
            left, right = values[left_backend], values[right_backend]
            if set(left) != set(right):
                raise ValueError(
                    f"E3 contrast {contrast} has unmatched factual positions"
                )
            rows_for_contrast: list[dict[str, Any]] = []
            for point in sorted(left):
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "target_model": target_model,
                    "contrast": contrast,
                    "left_backend": left_backend,
                    "right_backend": right_backend,
                    "point": point,
                }
                for metric, (population, field) in _POINT_METRICS.items():
                    is_defined = population == "all_requested_factuals" or (
                        left[point]["available"] and right[point]["available"]
                    )
                    if population == "jointly_target_class_valid":
                        is_defined = bool(
                            left[point]["valid_class"] and right[point]["valid_class"]
                        )
                    left_value = left[point].get(field) if is_defined else None
                    right_value = right[point].get(field) if is_defined else None
                    if left_value is None or right_value is None:
                        left_value = right_value = delta = None
                    else:
                        delta = float(right_value) - float(left_value)
                    row[f"{metric}_left"] = left_value
                    row[f"{metric}_right"] = right_value
                    row[f"{metric}_delta"] = delta
                rows_for_contrast.append(row)
            point_rows.extend(rows_for_contrast)
            for metric, (population, _field) in _POINT_METRICS.items():
                defined = [
                    row
                    for row in rows_for_contrast
                    if row[f"{metric}_delta"] is not None
                ]
                ci_low, ci_high = _paired_bootstrap_ci(
                    [row[f"{metric}_delta"] for row in rows_for_contrast]
                )
                contrast_rows.append(
                    {
                        "dataset": dataset,
                        "target_model": target_model,
                        "contrast": contrast,
                        "left_backend": left_backend,
                        "right_backend": right_backend,
                        "metric": metric,
                        "population": population,
                        "n_total": len(rows_for_contrast),
                        "n_joint_defined": len(defined),
                        "left_mean": (
                            float(np.mean([row[f"{metric}_left"] for row in defined]))
                            if defined
                            else None
                        ),
                        "right_mean": (
                            float(np.mean([row[f"{metric}_right"] for row in defined]))
                            if defined
                            else None
                        ),
                        "mean_delta": (
                            float(np.mean([row[f"{metric}_delta"] for row in defined]))
                            if defined
                            else None
                        ),
                        "ci_95_low": ci_low,
                        "ci_95_high": ci_high,
                    }
                )
    return tuple(point_rows), tuple(contrast_rows)


def build_e3_analysis(
    output_root: Path | str, matrix_config: Path | str, output_dir: Path | str
) -> tuple[Path, ...]:
    """Build deterministic raw and paired E3 products from canonical artifacts."""
    cells = load_published_cells(output_root, matrix_config)
    store = ArtifactStore(output_root)
    runs = {str(cell["run_id"]): store.read(str(cell["run_id"])) for cell in cells}
    point_rows, contrast_rows = _analyse_cells(cells, runs)
    root = Path(output_dir)
    raw_path = root / "e3_raw_summaries.csv"
    point_path = root / "e3_paired_points.csv"
    contrast_path = root / "e3_paired_contrasts.csv"
    manifest_path = root / "e3_analysis_manifest.json"
    raw_rows = tuple(
        {
            key: cell[key]
            for key in (
                "dataset",
                "target_model",
                "backend",
                "seed",
                "run_id",
                "cell_id",
                "artifact_path",
                *sorted(
                    name
                    for name in cell
                    if name not in {"scientific_group"}
                    and name
                    not in {
                        "dataset",
                        "target_model",
                        "backend",
                        "seed",
                        "run_id",
                        "cell_id",
                        "artifact_path",
                        "method",
                        "method_variant",
                        "n_counterfactuals",
                    }
                    and not name.startswith("set_")
                ),
            )
        }
        for cell in sorted(
            cells, key=lambda row: (row["dataset"], row["target_model"], row["backend"])
        )
    )
    write_rows(raw_path, raw_rows)
    write_rows(point_path, point_rows)
    write_rows(contrast_path, contrast_rows)
    manifest = {
        "schema_version": "countercontex.e3_analysis.v1",
        "primary_contrast": "learned_conditional",
        "primary_metric": "valid_success_rate_threshold_per_requested_slot",
        "contrast_direction": "right_backend_minus_left_backend",
        "blocking_unit": (
            "dataset; target-model blocks are reported within dataset, not treated as "
            "independent datasets"
        ),
        "uncertainty": {
            "method": "within-block paired factual bootstrap percentile interval",
            "confidence": 0.95,
            "resamples": _BOOTSTRAP_RESAMPLES,
            "seed": _BOOTSTRAP_SEED,
            "scope": (
                "selected matched factuals in each dataset-target-model block; not "
                "alternative train/test splits or independent datasets"
            ),
        },
        "populations": {
            metric: population
            for metric, (population, _field) in _POINT_METRICS.items()
        },
        "outputs": [raw_path.name, point_path.name, contrast_path.name],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return raw_path, point_path, contrast_path, manifest_path
