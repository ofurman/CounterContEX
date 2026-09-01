#  Copyright (c) Prior Labs GmbH 2026.
"""Thin CounterContEx compatibility shim over the generic benchmark runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.zeroshot_cf.benchmark_protocol import (
    DATASETS,
    DEFAULT_MAX_TEST,
    DEFAULT_VALIDATION_FRACTION,
)
from experiments.zeroshot_cf.benchmark_protocol import (
    prepare_benchmark_context as prepare_benchmark_context,
)
from experiments.zeroshot_cf.generator import DEFAULT_TEMPERATURE
from experiments.zeroshot_cf.orchestration.compat_cli import (
    aggregate_legacy_method,
    legacy_run_spec,
    run_legacy_dataset,
)
from experiments.zeroshot_cf.retained_config import TAU

DEFAULT_N_ESTIMATORS = 1
DEFAULT_MAX_VALIDITY_STEPS = 100
DEFAULT_JOINT_SHORTLIST_SIZE = 16
DEFAULT_MAX_EXTRA_ACTIONS = 1
DEFAULT_MIN_JOINT_LOG_GAIN = 0.0
DEFAULT_TABICL_JOINT_PERMUTATIONS = 1
DEFAULT_N_COUNTERFACTUALS = 3
DEFAULT_DIVERSITY_BEAM_WIDTH = 8
DEFAULT_DIVERSITY_CANDIDATE_POOL_SIZE = 16
DEFAULT_DIVERSITY_MAX_EXTRA_ACTIONS = 2
DEFAULT_DIVERSITY_MAX_GOWER_RATIO = 1.5
DEFAULT_DIVERSITY_MAX_GOWER_INCREASE = 0.02
DEFAULT_CANDIDATE_QUANTILES = tuple(i / 10 for i in range(1, 10))
DEFAULT_CONFIDENCE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
RESULTS_DIR = Path(__file__).parent / "results" / "athena" / "exp9_dicoflex"


def _spec(  # noqa: PLR0913
    dataset_name: str,
    *,
    max_test: int,
    n_estimators: int,
    temperature: float,
    tau: float,
    candidate_quantiles: tuple[float, ...],
    confidence_quantiles: tuple[float, ...] | None,
    cf_mode: str,
    tabicl_joint_permutations: int,
    max_validity_steps: int,
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
    drop_heloc_all_minus9: bool,
):
    normalized_mode = cf_mode.replace("-", "_")
    return legacy_run_spec(
        dataset_name,
        "countercontex",
        method_variant="tabicl_sparse" if normalized_mode == "sparse" else "default",
        method_params={
            "search": {
                "tau": tau,
                "candidate_quantiles": candidate_quantiles,
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
                "confidence_quantiles": confidence_quantiles,
                "tabicl_joint_permutations": tabicl_joint_permutations,
            },
        },
        n_counterfactuals=n_counterfactuals,
        max_test=max_test,
        validation_fraction=validation_fraction,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
        probability_threshold=tau,
    )


def run_dataset(  # noqa: PLR0913
    dataset_name: str,
    *,
    max_test: int = DEFAULT_MAX_TEST,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    temperature: float = DEFAULT_TEMPERATURE,
    tau: float = TAU,
    candidate_quantiles: tuple[float, ...] = DEFAULT_CANDIDATE_QUANTILES,
    confidence_quantiles: tuple[float, ...] | None = DEFAULT_CONFIDENCE_QUANTILES,
    cf_mode: str = "sparse",
    tabicl_joint_permutations: int = DEFAULT_TABICL_JOINT_PERMUTATIONS,
    max_validity_steps: int = DEFAULT_MAX_VALIDITY_STEPS,
    allow_revisits: bool = True,
    joint_shortlist_size: int = DEFAULT_JOINT_SHORTLIST_SIZE,
    max_extra_actions: int = DEFAULT_MAX_EXTRA_ACTIONS,
    min_joint_log_gain: float = DEFAULT_MIN_JOINT_LOG_GAIN,
    n_counterfactuals: int = DEFAULT_N_COUNTERFACTUALS,
    diversity_beam_width: int = DEFAULT_DIVERSITY_BEAM_WIDTH,
    diversity_candidate_pool_size: int = DEFAULT_DIVERSITY_CANDIDATE_POOL_SIZE,
    diversity_max_extra_actions: int = DEFAULT_DIVERSITY_MAX_EXTRA_ACTIONS,
    diversity_max_gower_ratio: float = DEFAULT_DIVERSITY_MAX_GOWER_RATIO,
    diversity_max_gower_increase: float = DEFAULT_DIVERSITY_MAX_GOWER_INCREASE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    drop_heloc_all_minus9: bool = True,
    tabicl_cache_dir: Path | None = None,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Translate and run one CounterContEx case through the generic lifecycle."""
    spec = _spec(
        dataset_name,
        max_test=max_test,
        n_estimators=n_estimators,
        temperature=temperature,
        tau=tau,
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
        drop_heloc_all_minus9=drop_heloc_all_minus9,
    )
    return run_legacy_dataset(
        spec,
        results_dir=results_dir,
        tabicl_cache_dir=tabicl_cache_dir,
    )


def aggregate_results(results_dir: Path = RESULTS_DIR) -> Path:
    return aggregate_legacy_method(
        results_dir,
        "countercontex",
        DATASETS,
        "exp9_tabicl_all_metrics.csv",
    )


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "aggregate"], required=True)
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument(
        "--candidate-quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_CANDIDATE_QUANTILES,
    )
    parser.add_argument(
        "--confidence-quantiles",
        type=float,
        nargs="+",
        default=DEFAULT_CONFIDENCE_QUANTILES,
    )
    parser.add_argument(
        "--confidence-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument(
        "--cf-mode", choices=["sparse", "data-plausible"], default="sparse"
    )
    parser.add_argument(
        "--tabicl-joint-permutations",
        type=int,
        default=DEFAULT_TABICL_JOINT_PERMUTATIONS,
    )
    parser.add_argument(
        "--max-validity-steps", type=int, default=DEFAULT_MAX_VALIDITY_STEPS
    )
    parser.add_argument(
        "--allow-revisits", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--joint-shortlist-size", type=int, default=DEFAULT_JOINT_SHORTLIST_SIZE
    )
    parser.add_argument(
        "--max-extra-actions", type=int, default=DEFAULT_MAX_EXTRA_ACTIONS
    )
    parser.add_argument(
        "--min-joint-log-gain", type=float, default=DEFAULT_MIN_JOINT_LOG_GAIN
    )
    parser.add_argument(
        "--n-counterfactuals", type=int, default=DEFAULT_N_COUNTERFACTUALS
    )
    parser.add_argument(
        "--diversity-beam-width", type=int, default=DEFAULT_DIVERSITY_BEAM_WIDTH
    )
    parser.add_argument(
        "--diversity-candidate-pool-size",
        type=int,
        default=DEFAULT_DIVERSITY_CANDIDATE_POOL_SIZE,
    )
    parser.add_argument(
        "--diversity-max-extra-actions",
        type=int,
        default=DEFAULT_DIVERSITY_MAX_EXTRA_ACTIONS,
    )
    parser.add_argument(
        "--diversity-max-gower-ratio",
        type=float,
        default=DEFAULT_DIVERSITY_MAX_GOWER_RATIO,
    )
    parser.add_argument(
        "--diversity-max-gower-increase",
        type=float,
        default=DEFAULT_DIVERSITY_MAX_GOWER_INCREASE,
    )
    parser.add_argument(
        "--drop-heloc-all-minus9",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tabicl-cache-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    if args.dataset == "aggregate":
        aggregate_results(args.results_dir)
        return
    run_dataset(
        args.dataset,
        max_test=args.max_test,
        n_estimators=args.n_estimators,
        temperature=args.temperature,
        tau=args.tau,
        candidate_quantiles=tuple(args.candidate_quantiles),
        confidence_quantiles=(
            tuple(args.confidence_quantiles) if args.confidence_conditioning else None
        ),
        cf_mode=args.cf_mode,
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
        drop_heloc_all_minus9=args.drop_heloc_all_minus9,
        tabicl_cache_dir=args.tabicl_cache_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
