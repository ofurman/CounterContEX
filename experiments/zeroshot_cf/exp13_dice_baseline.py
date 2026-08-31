#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: T201

"""Official DiCE genetic baseline under the fixed Exp9 protocol.

The adapter exposes each one-hot group as one categorical variable to DiCE and
decodes proposals back to the repository's model matrix. Consequently, DiCE
cannot create malformed partial one-hot actions. Features excluded by the
dataset actionability metadata are omitted from ``features_to_vary``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.benchmark_protocol import (
    DATASETS,
    DEFAULT_MAX_TEST,
    DEFAULT_PROTOCOL_SEED,
    DEFAULT_VALIDATION_FRACTION,
    aggregate_metrics_path,
    build_common_result_row,
    dataset_result_paths,
    mean_on_valid,
    prepare_benchmark_context,
    write_result_table,
)
from experiments.zeroshot_cf.core.validation import target_probabilities
from experiments.zeroshot_cf.data import (
    DatasetBundle,
)
from experiments.zeroshot_cf.methods.dice import (
    OUTCOME as OUTCOME,
)
from experiments.zeroshot_cf.methods.dice import (
    DiceClassifierAdapter as DiceClassifierAdapter,
)
from experiments.zeroshot_cf.methods.dice import (
    DiceMixedAdapter as DiceMixedAdapter,
)
from experiments.zeroshot_cf.methods.dice import (
    generate_dice_counterfactuals as generate_dice_counterfactuals,
)
from experiments.zeroshot_cf.retained_config import TAU
from experiments.zeroshot_cf.runner_compat import (
    evaluate_result,
    generation_request,
    legacy_candidate_matrix,
    legacy_common_metrics,
    method_context,
    point_diagnostics,
    write_legacy_outputs_with_timing,
)

RESULTS_DIR = Path(__file__).parent / "results" / "local" / "exp13_dice"


def _features_to_vary(
    bundle: DatasetBundle,
    codec: DiceMixedAdapter,
    scalar_actionable: list[int],
    grouped_actionable: list[OneHotActionGroup],
) -> list[str]:
    scalar_set = set(scalar_actionable)
    group_set = {group.name for group in grouped_actionable}
    return [
        *(
            name
            for name, column in zip(
                codec.scalar_names, codec.scalar_columns, strict=True
            )
            if column in scalar_set
        ),
        *(group.name for group in codec.groups if group.name in group_set),
    ]


def run_dataset(  # noqa: PLR0913
    dataset_name: str,
    *,
    max_test: int = DEFAULT_MAX_TEST,
    max_iterations: int = 200,
    search_restarts: int = 1,
    stopping_threshold: float = TAU,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    drop_heloc_all_minus9: bool = True,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run official DiCE-genetic with the fixed Exp9 data and classifier."""
    from experiments.zeroshot_cf.methods.dice import DiceConfig, DiceMethod
    from experiments.zeroshot_cf.metrics_harness import print_metrics

    total_started = time.perf_counter()
    context = prepare_benchmark_context(
        dataset_name,
        max_test=max_test,
        validation_fraction=validation_fraction,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
    )
    X_test = context.X_test
    y_test = context.y_test
    prepare_started = time.perf_counter()
    prepared = DiceMethod(
        DiceConfig(
            max_iterations=max_iterations,
            search_restarts=search_restarts,
            stopping_threshold=stopping_threshold,
        )
    ).prepare(method_context(context))
    runtime_prepare = time.perf_counter() - prepare_started
    vary = prepared.features_to_vary

    generation_started = time.perf_counter()
    result = prepared.generate(
        generation_request(context, seed=DEFAULT_PROTOCOL_SEED)
    )
    runtime_generation = time.perf_counter() - generation_started
    evaluation_started = time.perf_counter()
    report = evaluate_result(
        context,
        result,
        probability_threshold=stopping_threshold,
    )
    runtime_evaluation = time.perf_counter() - evaluation_started
    X_cf = legacy_candidate_matrix(result)
    X_cf_raw = np.asarray(result.artifacts["method.raw_candidates"])
    point_info = point_diagnostics(result)
    y_cf_pred = np.asarray(context.disc_model.predict(X_cf), dtype=int)
    probabilities = target_probabilities(context.disc_model, X_cf, context.y_target)
    valid = y_cf_pred == context.y_target
    common_metrics = legacy_common_metrics(report)
    print_metrics(common_metrics, prefix=f"{dataset_name}/DiCE-genetic")
    changed_columns = (X_cf != X_test).sum(axis=1)
    raw_changed_columns = (X_cf_raw != X_test).sum(axis=1)
    raw_l2 = np.linalg.norm(X_cf_raw - X_test, axis=1)
    all_l2 = np.linalg.norm(X_cf[valid] - X_test[valid], axis=1)
    found = np.asarray([info["found"] for info in point_info], dtype=bool)
    row: dict[str, Any] = build_common_result_row(
        context,
        method="dice_genetic_atomic_pruned",
        cf_per_factual=1,
        extra_fields={
        "backend": "sklearn",
        "search": "genetic",
        "initialization": "kdtree",
        "max_iterations": max_iterations,
        "search_restarts": search_restarts,
        "stopping_threshold": stopping_threshold,
        "proximity_weight": 0.2,
        "sparsity_weight": 0.2,
        "categorical_penalty": 0.1,
        "categorical_actions": "compact_atomic_groups",
        "posthoc_action_pruning": True,
        "posthoc_scalar_contraction": True,
        "features_to_vary_count": len(vary),
        "prepare_s": runtime_prepare,
        "generate_s": runtime_generation,
        "evaluate_s": runtime_evaluation,
        "runtime_generation_s": round(runtime_generation, 3),
        "dice_found_fraction": float(found.mean()),
        "dice_returned_fraction": float(
            np.mean([info["returned"] for info in point_info])
        ),
        **common_metrics,
        "sparsity_exact": float((X_test != X_cf).mean()),
        "true_actionability": common_metrics["actionability"],
        "proximity_all_features_euclidean": (
            float(all_l2.mean()) if len(all_l2) else float("nan")
        ),
        "failure_rate": float((~valid).mean()),
        "l0_count_mean": mean_on_valid(changed_columns, valid),
        "raw_l0_count_mean": float(raw_changed_columns.mean()),
        "raw_proximity_all_features_euclidean": float(raw_l2.mean()),
        "factual_oob_fraction": float(
            (((X_test < 0.0) | (X_test > 1.0)).any(axis=1)).mean()
        ),
        "cf_oob_fraction": float((((X_cf < 0.0) | (X_cf > 1.0)).any(axis=1)).mean()),
        },
    )
    row["runtime_total_s"] = round(time.perf_counter() - total_started, 3)

    paths = dataset_result_paths(results_dir, "exp13_dice_genetic", dataset_name)
    write_legacy_outputs_with_timing(
        paths,
        row,
        [
            {
                "point": index,
                "factual_label": int(y_test[index]),
                "factual_prediction": int(context.y_pred[index]),
                "target": int(context.y_target[index]),
                "cf_prediction": int(y_cf_pred[index]),
                "valid": bool(valid[index]),
                "dice_found": bool(info["found"]),
                "dice_returned": bool(info["returned"]),
                "valid_candidates": int(info["valid_candidates"]),
                "search_attempts": int(info["attempts"]),
                "target_probability": float(probabilities[index]),
                "changed_columns": int(changed_columns[index]),
                "raw_changed_columns": int(raw_changed_columns[index]),
                "raw_proximity_l2": float(raw_l2[index]),
                "runtime_s": float(info["runtime_s"]),
            }
            for index, info in enumerate(point_info)
        ],
        arrays={
            "X_test": X_test,
            "y_test": y_test,
            "X_cf": X_cf,
            "y_pred": context.y_pred,
            "y_target": context.y_target,
            "y_cf_pred": y_cf_pred,
            "X_cf_raw": X_cf_raw,
        },
        total_started=total_started,
    )
    return row


def main() -> None:
    """Run DiCE-genetic locally on one or all Exp9 datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--search-restarts", type=int, default=1)
    parser.add_argument("--stopping-threshold", type=float, default=TAU)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--drop-heloc-all-minus9",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    rows = [
        run_dataset(
            dataset,
            max_test=args.max_test,
            max_iterations=args.max_iterations,
            search_restarts=args.search_restarts,
            stopping_threshold=args.stopping_threshold,
            validation_fraction=args.validation_fraction,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
            results_dir=args.results_dir,
        )
        for dataset in datasets
    ]
    if len(rows) > 1:
        write_result_table(
            aggregate_metrics_path(args.results_dir, "exp13_dice_genetic"), rows
        )


if __name__ == "__main__":
    main()
