"""Factual-clustered summaries for the E12 proposal pushforward diagnostic."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from experiments.zeroshot_cf.diagnostics.proposal_pushforward import (
    read_diagnostic_bundle,
)


def _weighted(values: np.ndarray, masses: np.ndarray) -> float:
    total = float(masses.sum())
    if total <= 0:
        raise ValueError("proposal mass must be positive")
    return float(np.sum(values * masses) / total)


def _log_odds(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return np.log(clipped) - np.log1p(-clipped)


def _weighted_quantile(
    values: np.ndarray, masses: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(masses[order]) / masses.sum()
    index = min(np.searchsorted(cumulative, quantile), len(values) - 1)
    return float(ordered_values[index])


def _expected_best_of_b(values: np.ndarray, masses: np.ndarray, budget: int) -> float:
    if budget < 1:
        raise ValueError("best-of-B budgets must be positive")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_masses = masses[order] / masses.sum()
    cumulative = np.cumsum(ordered_masses)
    previous = np.concatenate(([0.0], cumulative[:-1]))
    maximum_mass = cumulative**budget - previous**budget
    return float(np.sum(ordered_values * maximum_mass))


def summarize_action_unit(
    records: Sequence[Mapping[str, Any]], *, best_of_b_budgets: Sequence[int] = (9, 49)
) -> dict[str, Any]:
    """Summarize one action unit under its declared q measure."""
    if not records:
        raise ValueError("action-unit records must be non-empty")
    masses = np.asarray([record["q_mass"] for record in records], dtype=np.float64)
    if np.any(masses < 0) or not np.isclose(masses.sum(), 1.0):
        raise ValueError("action-unit q masses must be non-negative and sum to one")
    before = np.asarray(
        [record["target_probability_before"] for record in records], dtype=np.float64
    )
    after = np.asarray(
        [record["target_probability_after"] for record in records], dtype=np.float64
    )
    delta = after - before
    changed = ~np.asarray([record["no_op"] for record in records], dtype=bool)
    changed_mass = masses[changed]
    conditional_abs = (
        _weighted(np.abs(delta[changed]), changed_mass)
        if changed_mass.sum() > 0
        else None
    )
    return {
        "action_unit": records[0]["action_unit"],
        "action_type": records[0]["action_type"],
        "sampling_bank": records[0].get("sampling_bank", "deterministic"),
        "sampling_seed": records[0].get("sampling_seed"),
        "unconditional_abs_delta": _weighted(np.abs(delta), masses),
        "conditional_abs_delta": conditional_abs,
        "positive_delta": _weighted(np.maximum(delta, 0.0), masses),
        "probability_positive_delta": _weighted((delta > 0).astype(float), masses),
        "generation_crossing_mass": _weighted(
            np.asarray(
                [record["crosses_generation_threshold"] for record in records],
                dtype=float,
            ),
            masses,
        ),
        "evaluation_crossing_mass": _weighted(
            np.asarray(
                [record["crosses_evaluation_threshold"] for record in records],
                dtype=float,
            ),
            masses,
        ),
        "unconditional_abs_log_odds_delta": _weighted(
            np.abs(_log_odds(after) - _log_odds(before)), masses
        ),
        "median_abs_delta": _weighted_quantile(np.abs(delta), masses, 0.5),
        "q90_abs_delta": _weighted_quantile(np.abs(delta), masses, 0.9),
        "expected_best_abs_delta_by_budget": {
            str(int(budget)): _expected_best_of_b(np.abs(delta), masses, int(budget))
            for budget in best_of_b_budgets
        },
        "target_class_mass": _weighted(
            np.asarray(
                [record["target_class_after"] for record in records], dtype=float
            ),
            masses,
        ),
        "no_op_mass": float(masses[~changed].sum()),
        "duplicate_mass": float(
            masses[
                np.asarray(
                    [record["duplicate_of"] is not None for record in records],
                    dtype=bool,
                )
            ].sum()
        ),
        "mean_projection_displacement": _weighted(
            np.asarray(
                [record["projection_displacement"] for record in records],
                dtype=np.float64,
            ),
            masses,
        ),
        "raw_count": len(records),
        "unique_projected_count": len(
            {int(record["unique_row_id"]) for record in records}
        ),
    }


def summarize_pushforward(
    records: Sequence[Mapping[str, Any]],
    *,
    best_of_b_budgets: Sequence[int] = (9, 49),
) -> tuple[dict[str, Any], ...]:
    """Average action units within factuals before any across-factual inference."""
    factual_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record.get("dataset"),
            record.get("classifier_id"),
            record["factual_partition"],
            int(record["factual_source_index"]),
            int(record["target"]),
            record["state_id"],
        )
        factual_groups[key].append(record)

    factual_summaries: list[dict[str, Any]] = []
    ordered_factual_groups = sorted(
        factual_groups.items(), key=lambda item: str(item[0])
    )
    for key, factual_records in ordered_factual_groups:
        banks: dict[tuple[str, str, Any, str], list[Mapping[str, Any]]] = defaultdict(
            list
        )
        for record in factual_records:
            banks[
                (
                    record["action_type"],
                    record["action_unit"],
                    record.get("sampling_seed"),
                    record.get("sampling_bank", "deterministic"),
                )
            ].append(record)
        bank_summaries = [
            summarize_action_unit(bank, best_of_b_budgets=best_of_b_budgets)
            for bank in banks.values()
        ]
        grouped_units: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
            list
        )
        for bank in bank_summaries:
            grouped_units[(bank["action_type"], bank["action_unit"])].append(bank)
        averaged_keys = (
            "unconditional_abs_delta",
            "positive_delta",
            "probability_positive_delta",
            "generation_crossing_mass",
            "evaluation_crossing_mass",
            "unconditional_abs_log_odds_delta",
            "median_abs_delta",
            "q90_abs_delta",
            "target_class_mass",
            "no_op_mass",
            "duplicate_mass",
            "mean_projection_displacement",
        )
        unit_summaries = []
        for (action_type, action_unit), unit_banks in grouped_units.items():
            integration_banks = [
                bank
                for bank in unit_banks
                if bank["sampling_bank"] == "integration-256"
            ]
            primary_banks = integration_banks or unit_banks
            conditional = [
                bank["conditional_abs_delta"]
                for bank in primary_banks
                if bank["conditional_abs_delta"] is not None
            ]
            mc_conditional = [
                bank["conditional_abs_delta"]
                for bank in unit_banks
                if str(bank["sampling_bank"]).startswith("mc-")
                and bank["conditional_abs_delta"] is not None
            ]
            unit_summaries.append(
                {
                    "action_type": action_type,
                    "action_unit": action_unit,
                    **{
                        metric: float(np.mean([bank[metric] for bank in primary_banks]))
                        for metric in averaged_keys
                    },
                    "conditional_abs_delta": (
                        float(np.mean(conditional)) if conditional else None
                    ),
                    "expected_best_abs_delta_by_budget": {
                        str(int(budget)): float(
                            np.mean(
                                [
                                    bank["expected_best_abs_delta_by_budget"][
                                        str(int(budget))
                                    ]
                                    for bank in primary_banks
                                ]
                            )
                        )
                        for budget in best_of_b_budgets
                    },
                    "bank_count": len(unit_banks),
                    "mc_conditional_abs_delta_mean": (
                        float(np.mean(mc_conditional)) if mc_conditional else None
                    ),
                    "mc_conditional_abs_delta_std": (
                        float(np.std(mc_conditional)) if mc_conditional else None
                    ),
                    "raw_count_by_bank": [bank["raw_count"] for bank in unit_banks],
                    "unique_projected_count_by_bank": [
                        bank["unique_projected_count"] for bank in unit_banks
                    ],
                }
            )
        conditional_means = {}
        unconditional_means = {}
        not_estimable = {}
        for action_type in ("categorical", "numerical"):
            typed = [
                unit for unit in unit_summaries if unit["action_type"] == action_type
            ]
            estimable = [
                unit["conditional_abs_delta"]
                for unit in typed
                if unit["conditional_abs_delta"] is not None
            ]
            conditional_means[action_type] = (
                float(np.mean(estimable)) if estimable else None
            )
            unconditional_means[action_type] = (
                float(np.mean([unit["unconditional_abs_delta"] for unit in typed]))
                if typed
                else None
            )
            not_estimable[action_type] = len(typed) - len(estimable)
        categorical = conditional_means["categorical"]
        numerical = conditional_means["numerical"]
        factual_summaries.append(
            {
                "dataset": key[0],
                "classifier_id": key[1],
                "factual_partition": key[2],
                "factual_source_index": key[3],
                "target": key[4],
                "state_id": key[5],
                "action_units": sorted(
                    unit_summaries,
                    key=lambda unit: (unit["action_type"], unit["action_unit"]),
                ),
                "conditional_abs_delta_by_type": conditional_means,
                "unconditional_abs_delta_by_type": unconditional_means,
                "not_estimable_units_by_type": not_estimable,
                "categorical_minus_numerical": (
                    None
                    if categorical is None or numerical is None
                    else float(categorical - numerical)
                ),
            }
        )
    return tuple(factual_summaries)


def analyze_diagnostic_bundle(directory: Any) -> tuple[dict[str, Any], ...]:
    """Verify a bundle, attach its stratum identity, then analyze its rows."""
    metadata, records = read_diagnostic_bundle(directory)
    enriched = [
        {
            **record,
            "dataset": metadata["dataset"],
            "classifier_id": metadata["classifier_id"],
        }
        for record in records
    ]
    budgets = metadata["policy"].get("best_of_b_budgets", [9, 49])
    return summarize_pushforward(enriched, best_of_b_budgets=budgets)


def hierarchical_bootstrap(
    factual_summaries: Sequence[Mapping[str, Any]],
    *,
    draws: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap datasets then factuals independently within classifier strata."""
    if draws < 1:
        raise ValueError("draws must be positive")
    strata: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for summary in factual_summaries:
        strata[
            (
                summary["classifier_id"],
                int(summary["target"]),
                summary["factual_partition"],
                summary["state_id"],
            )
        ].append(summary)
    rng = np.random.default_rng(seed)
    results = []
    for stratum, summaries in sorted(strata.items(), key=lambda item: str(item[0])):
        grouped: dict[str, list[float]] = defaultdict(list)
        for summary in summaries:
            effect = summary["categorical_minus_numerical"]
            if effect is not None and np.isfinite(effect):
                grouped[str(summary["dataset"])].append(float(effect))
        estimate = None
        interval = None
        if grouped:
            datasets = sorted(grouped)
            estimate = float(
                np.mean([np.mean(grouped[dataset]) for dataset in datasets])
            )
            samples = np.empty(draws, dtype=np.float64)
            for draw in range(draws):
                selected = rng.choice(datasets, size=len(datasets), replace=True)
                dataset_means = []
                for dataset in selected:
                    values = np.asarray(grouped[str(dataset)], dtype=np.float64)
                    resampled = rng.choice(values, len(values), replace=True)
                    dataset_means.append(float(np.mean(resampled)))
                samples[draw] = np.mean(dataset_means)
            interval = [float(value) for value in np.quantile(samples, [0.025, 0.975])]
        results.append(
            {
                "classifier_id": stratum[0],
                "target": stratum[1],
                "factual_partition": stratum[2],
                "state_id": stratum[3],
                "estimate": estimate,
                "interval_95": interval,
                "datasets": len(grouped),
                "dataset_estimates": [
                    {
                        "dataset": dataset,
                        "estimate": float(np.mean(values)),
                        "factuals": len(values),
                    }
                    for dataset, values in sorted(grouped.items())
                ],
            }
        )
    return {"draws": draws, "strata": results}
