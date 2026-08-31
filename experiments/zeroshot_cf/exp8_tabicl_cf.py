#  Copyright (c) Prior Labs GmbH 2026.
"""Thin Experiment 8 compatibility shim over the generic benchmark runner."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.generator import DEFAULT_N_ESTIMATORS, DEFAULT_TEMPERATURE
from experiments.zeroshot_cf.orchestration.compat_cli import (
    legacy_run_spec,
    run_legacy_dataset_with_stored,
)
from experiments.zeroshot_cf.orchestration.exp8_compat import (
    export_exp8_result,
    load_exp8_result,
)
from experiments.zeroshot_cf.orchestration.spec import RunSpec
from experiments.zeroshot_cf.retained_config import DATASET_PARAMS, TAU

RESULTS_DIR = Path(__file__).parent / "results"
CF_MODES = ("sparse", "data_plausible")

def _resolve_max_test(dataset_name: str, max_test: int | None) -> int | None:
    if max_test is not None and max_test < 0:
        return None
    if max_test is not None:
        return max_test
    return DATASET_PARAMS.get(dataset_name, {"max_test": 50})["max_test"]


def _spec(  # noqa: PLR0913
    dataset_name: str,
    *,
    tau: float,
    temperature: float,
    n_estimators: int,
    max_test: int | None,
    candidate_quantiles: tuple[float, ...] | None,
    confidence_quantiles: tuple[float, ...] | None,
    cf_mode: str,
    tabicl_joint_permutations: int,
    max_validity_steps: int | None,
    allow_revisits: bool,
    joint_shortlist_size: int,
    max_extra_actions: int,
    min_joint_log_gain: float,
    n_counterfactuals: int,
    diversity_beam_width: int,
    diversity_candidate_pool_size: int,
    diversity_max_extra_actions: int,
    diversity_max_gower_ratio: float,
    diversity_max_gower_increase: float,
    validation_fraction: float,
    test_selection: str,
    drop_heloc_all_minus9: bool,
) -> RunSpec:
    """Translate the historical Experiment 8 options into one scientific spec."""
    normalized_mode = cf_mode.replace("-", "_")
    if normalized_mode not in CF_MODES:
        raise ValueError(f"cf_mode must be one of {CF_MODES}, got {cf_mode!r}")
    normalized_candidate_quantiles = (
        None
        if candidate_quantiles is None
        else tuple(float(value) for value in candidate_quantiles)
    )
    normalized_confidence_quantiles = (
        None
        if confidence_quantiles is None
        else tuple(float(value) for value in confidence_quantiles)
    )
    resolved_max_test = _resolve_max_test(dataset_name, max_test)
    spec = legacy_run_spec(
        dataset_name,
        "dicoflex",
        method_variant=(
            "tabicl_sparse" if normalized_mode == "sparse" else "default"
        ),
        method_params={
            "search": {
                "tau": tau,
                "candidate_quantiles": normalized_candidate_quantiles,
                "cf_mode": normalized_mode,
                "max_validity_steps": max_validity_steps,
                "allow_revisits": allow_revisits,
                "joint_shortlist_size": joint_shortlist_size,
                "max_extra_actions": max_extra_actions,
                "min_joint_log_gain": min_joint_log_gain,
            },
            "diversity": {
                "beam_width": diversity_beam_width,
                "candidate_pool_size": diversity_candidate_pool_size,
                "max_extra_actions": diversity_max_extra_actions,
                "max_gower_ratio": diversity_max_gower_ratio,
                "max_gower_increase": diversity_max_gower_increase,
            },
            "foundation": {
                "backend": "tabicl",
                "n_estimators": n_estimators,
                "temperature": temperature,
                "confidence_quantiles": normalized_confidence_quantiles,
                "tabicl_joint_permutations": tabicl_joint_permutations,
            },
        },
        n_counterfactuals=n_counterfactuals,
        max_test=-1 if resolved_max_test is None else resolved_max_test,
        validation_fraction=validation_fraction,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
        probability_threshold=tau,
    )
    return replace(spec, protocol=replace(spec.protocol, test_selection=test_selection))



def generate_tabicl_counterfactuals(  # noqa: PLR0913
    dataset_name: str,
    *,
    tau: float = TAU,
    temperature: float = DEFAULT_TEMPERATURE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_test: int | None = None,
    candidate_quantiles: tuple[float, ...] | None = None,
    confidence_quantiles: tuple[float, ...] | None = None,
    cf_mode: str = "sparse",
    tabicl_joint_permutations: int = 1,
    max_validity_steps: int | None = None,
    allow_revisits: bool = True,
    joint_shortlist_size: int = 16,
    max_extra_actions: int = 1,
    min_joint_log_gain: float = 0.0,
    n_counterfactuals: int = 1,
    diversity_beam_width: int = 8,
    diversity_candidate_pool_size: int = 16,
    diversity_max_extra_actions: int = 2,
    diversity_max_gower_ratio: float = 1.5,
    diversity_max_gower_increase: float = 0.02,
    validation_fraction: float = 0.0,
    test_selection: str = "first",
    drop_heloc_all_minus9: bool = False,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the generic lifecycle and adapt its v1 arrays to the old return shape."""
    spec = _spec(
        dataset_name,
        tau=tau,
        temperature=temperature,
        n_estimators=n_estimators,
        max_test=max_test,
        candidate_quantiles=candidate_quantiles,
        confidence_quantiles=confidence_quantiles,
        cf_mode=cf_mode,
        tabicl_joint_permutations=tabicl_joint_permutations,
        max_validity_steps=max_validity_steps,
        allow_revisits=allow_revisits,
        joint_shortlist_size=joint_shortlist_size,
        max_extra_actions=max_extra_actions,
        min_joint_log_gain=min_joint_log_gain,
        n_counterfactuals=n_counterfactuals,
        diversity_beam_width=diversity_beam_width,
        diversity_candidate_pool_size=diversity_candidate_pool_size,
        diversity_max_extra_actions=diversity_max_extra_actions,
        diversity_max_gower_ratio=diversity_max_gower_ratio,
        diversity_max_gower_increase=diversity_max_gower_increase,
        validation_fraction=validation_fraction,
        test_selection=test_selection,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
    )
    metrics, stored = run_legacy_dataset_with_stored(
        spec,
        results_dir=RESULTS_DIR,
        tabicl_cache_dir=cache_dir,
    )
    return load_exp8_result(
        spec,
        metrics,
        stored=stored,
        results_dir=RESULTS_DIR,
    )



def run_and_report(dataset_name: str, **kwargs: Any) -> dict[str, float]:
    """Run one dataset and request its historical compatibility export."""
    X_test, _, X_cf, info = generate_tabicl_counterfactuals(dataset_name, **kwargs)
    return export_exp8_result(
        dataset_name,
        X_test,
        X_cf,
        info,
        results_dir=RESULTS_DIR,
    )


def main(argv: Sequence[str] | None = None) -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="TabICL greedy counterfactuals with a fixed kNN context"
    )
    parser.add_argument(
        "--dataset",
        choices=["moons", "heloc", "german_credit", "all"],
        default="moons",
    )
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Default: moons=100, heloc=50; use -1 for the full test split.",
    )
    parser.add_argument(
        "--candidate-quantiles",
        type=float,
        nargs="+",
        default=None,
        metavar="Q",
        help=(
            "Generate deterministic conditional proposals per feature. Prefer "
            "a central grid, e.g. --candidate-quantiles 0.1 0.3 0.5 0.7 0.9."
        ),
    )
    parser.add_argument(
        "--confidence-quantiles",
        type=float,
        nargs="+",
        default=None,
        metavar="Q",
        help=(
            "Quantile levels of the selected context's target-class confidence "
            "distribution. The resulting empirical confidence values are appended "
            "to TabICL queries; requires --candidate-quantiles."
        ),
    )
    parser.add_argument(
        "--cf-mode", choices=["sparse", "data-plausible"], default="sparse"
    )
    parser.add_argument("--tabicl-joint-permutations", type=int, default=1)
    parser.add_argument("--max-validity-steps", type=int, default=None)
    parser.add_argument(
        "--allow-revisits", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--joint-shortlist-size", type=int, default=16)
    parser.add_argument("--max-extra-actions", type=int, default=1)
    parser.add_argument("--min-joint-log-gain", type=float, default=0.0)
    parser.add_argument("--n-counterfactuals", type=int, default=1)
    parser.add_argument("--diversity-beam-width", type=int, default=8)
    parser.add_argument("--diversity-candidate-pool-size", type=int, default=16)
    parser.add_argument("--diversity-max-extra-actions", type=int, default=2)
    parser.add_argument("--diversity-max-gower-ratio", type=float, default=1.5)
    parser.add_argument("--diversity-max-gower-increase", type=float, default=0.02)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument(
        "--test-selection", choices=["first", "stratified"], default="first"
    )
    parser.add_argument("--drop-heloc-all-minus9", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    datasets = (
        ["moons", "heloc", "german_credit"]
        if args.dataset == "all"
        else [args.dataset]
    )
    for dataset_name in datasets:
        run_and_report(
            dataset_name,
            tau=args.tau,
            temperature=args.temperature,
            n_estimators=args.n_estimators,
            max_test=args.max_test,
            candidate_quantiles=(
                None
                if args.candidate_quantiles is None
                else tuple(args.candidate_quantiles)
            ),
            confidence_quantiles=(
                None
                if args.confidence_quantiles is None
                else tuple(args.confidence_quantiles)
            ),
            cf_mode=args.cf_mode.replace("-", "_"),
            tabicl_joint_permutations=args.tabicl_joint_permutations,
            max_validity_steps=args.max_validity_steps,
            allow_revisits=args.allow_revisits,
            joint_shortlist_size=args.joint_shortlist_size,
            max_extra_actions=args.max_extra_actions,
            min_joint_log_gain=args.min_joint_log_gain,
            n_counterfactuals=args.n_counterfactuals,
            diversity_beam_width=args.diversity_beam_width,
            diversity_candidate_pool_size=args.diversity_candidate_pool_size,
            diversity_max_extra_actions=args.diversity_max_extra_actions,
            diversity_max_gower_ratio=args.diversity_max_gower_ratio,
            diversity_max_gower_increase=args.diversity_max_gower_increase,
            validation_fraction=args.validation_fraction,
            test_selection=args.test_selection,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
