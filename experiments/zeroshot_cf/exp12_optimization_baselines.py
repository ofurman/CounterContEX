#  Copyright (c) Prior Labs GmbH 2026.
"""Thin optimization-baseline compatibility shim over the generic runner."""

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
from experiments.zeroshot_cf.methods.optimization import (
    growing_spheres_counterfactual as growing_spheres_counterfactual,
)
from experiments.zeroshot_cf.methods.optimization import (
    wachter_coordinate_counterfactual as wachter_coordinate_counterfactual,
)
from experiments.zeroshot_cf.orchestration.compat_cli import (
    legacy_run_spec,
    run_legacy_dataset,
    run_legacy_specs,
)
from experiments.zeroshot_cf.retained_config import TAU

METHODS = ("wachter", "growing_spheres")
RESULTS_DIR = Path(__file__).parent / "results" / "local" / "exp12_optimization"


def _spec(
    dataset_name: str,
    method: str,
    *,
    max_test: int,
    validation_fraction: float,
    drop_heloc_all_minus9: bool,
    random_state: int,
    sphere_candidates: int,
):
    if method not in METHODS:
        raise ValueError(f"Unsupported Exp12 method: {method!r}")
    params = {"n_candidates": sphere_candidates} if method == "growing_spheres" else {}
    return legacy_run_spec(
        dataset_name,
        method,
        method_params=params,
        max_test=max_test,
        validation_fraction=validation_fraction,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
        probability_threshold=TAU,
        seed=random_state,
    )


def run_dataset(
    dataset_name: str,
    method: str,
    *,
    max_test: int = DEFAULT_MAX_TEST,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    drop_heloc_all_minus9: bool = True,
    random_state: int = 42,
    sphere_candidates: int = 512,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run one optimization case through the generic lifecycle."""
    return run_legacy_dataset(
        _spec(
            dataset_name,
            method,
            max_test=max_test,
            validation_fraction=validation_fraction,
            drop_heloc_all_minus9=drop_heloc_all_minus9,
            random_state=random_state,
            sphere_candidates=sphere_candidates,
        ),
        results_dir=results_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--sphere-candidates", type=int, default=512)
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument(
        "--drop-heloc-all-minus9",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    methods = METHODS if args.method == "all" else (args.method,)
    specs = tuple(
        _spec(
            dataset,
            method,
            max_test=args.max_test,
            validation_fraction=args.validation_fraction,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
            random_state=42,
            sphere_candidates=args.sphere_candidates,
        )
        for dataset in datasets
        for method in methods
    )
    if len(specs) == 1:
        run_legacy_dataset(specs[0], results_dir=args.results_dir)
    else:
        run_legacy_specs(
            specs,
            results_dir=args.results_dir,
            aggregate_name="exp12_optimization_all_metrics.csv",
        )


if __name__ == "__main__":
    main()
