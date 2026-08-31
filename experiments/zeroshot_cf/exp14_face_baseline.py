#  Copyright (c) Prior Labs GmbH 2026.
"""Thin FACE-kNN compatibility shim over the generic benchmark runner."""

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
from experiments.zeroshot_cf.methods.face import FaceGraph as FaceGraph
from experiments.zeroshot_cf.methods.face import (
    _expanded_actionable_columns as _expanded_actionable_columns,
)
from experiments.zeroshot_cf.methods.face import (
    build_face_knn_graph as build_face_knn_graph,
)
from experiments.zeroshot_cf.methods.face import (
    face_counterfactual as face_counterfactual,
)
from experiments.zeroshot_cf.orchestration.compat_cli import (
    legacy_run_spec,
    run_legacy_dataset,
    run_legacy_specs,
)
from experiments.zeroshot_cf.retained_config import TAU

RESULTS_DIR = Path(__file__).parent / "results" / "local" / "exp14_face_knn"


def _spec(
    dataset_name: str,
    *,
    max_test: int,
    n_neighbors: int,
    density_power: float,
    tau: float,
    validation_fraction: float,
    drop_heloc_all_minus9: bool,
):
    return legacy_run_spec(
        dataset_name,
        "face",
        method_params={
            "n_neighbors": n_neighbors,
            "density_power": density_power,
            "tau": tau,
        },
        max_test=max_test,
        validation_fraction=validation_fraction,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
        probability_threshold=tau,
    )


def run_dataset(
    dataset_name: str,
    *,
    max_test: int = DEFAULT_MAX_TEST,
    n_neighbors: int = 100,
    density_power: float = 1.0,
    tau: float = TAU,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    drop_heloc_all_minus9: bool = True,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run one FACE case through the generic lifecycle."""
    return run_legacy_dataset(
        _spec(
            dataset_name,
            max_test=max_test,
            n_neighbors=n_neighbors,
            density_power=density_power,
            tau=tau,
            validation_fraction=validation_fraction,
            drop_heloc_all_minus9=drop_heloc_all_minus9,
        ),
        results_dir=results_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--n-neighbors", type=int, default=100)
    parser.add_argument("--density-power", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=TAU)
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
    specs = tuple(
        _spec(
            dataset,
            max_test=args.max_test,
            n_neighbors=args.n_neighbors,
            density_power=args.density_power,
            tau=args.tau,
            validation_fraction=args.validation_fraction,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
        )
        for dataset in datasets
    )
    if len(specs) == 1:
        run_legacy_dataset(specs[0], results_dir=args.results_dir)
    else:
        run_legacy_specs(
            specs,
            results_dir=args.results_dir,
            aggregate_name="exp14_face_knn_all_metrics.csv",
        )


if __name__ == "__main__":
    main()
