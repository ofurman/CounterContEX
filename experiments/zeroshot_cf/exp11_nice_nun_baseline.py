#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: T201

"""NICE-style nearest-unlike-neighbour baseline for the Exp9 setting.

The baseline uses the same fixed split, target classifier, factual subset,
actionability constraints, and metrics as Exp9. For each factual it finds the
nearest training row predicted as the desired class, then greedily copies one
actionable scalar or one complete one-hot group at a time. Before validity,
the action with the largest target-probability gain is selected. If several
one-step actions are valid, the lowest-LOF action is selected, matching the
valid-candidate preference used by the TabICL experiment.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.baseline_common import ActionUnit, build_action_units
from experiments.zeroshot_cf.benchmark_protocol import (
    DATASETS,
    DEFAULT_MAX_TEST,
    DEFAULT_SPARSITY_EPS,
    DEFAULT_VALIDATION_FRACTION,
    TARGET_CLASSIFIER_LABELS,
    aggregate_metrics_path,
    build_common_result_row,
    dataset_result_paths,
    mean_on_valid,
    prepare_benchmark_context,
    write_dataset_outputs,
    write_result_table,
)
from experiments.zeroshot_cf.retained_config import TAU

if TYPE_CHECKING:
    from sklearn.neighbors import LocalOutlierFactor

RESULTS_DIR = Path(__file__).parent / "results" / "local" / "exp11_nice_nun"


def nearest_unlike_prototypes(
    X_train: np.ndarray,
    train_predictions: np.ndarray,
    X_test: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the nearest classifier-target training row for each factual."""
    from sklearn.neighbors import NearestNeighbors

    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    train_predictions = np.asarray(train_predictions, dtype=int)
    targets = np.asarray(targets, dtype=int)
    prototypes = np.empty_like(X_test)
    prototype_indices = np.empty(len(X_test), dtype=int)
    distances = np.empty(len(X_test), dtype=np.float64)

    for target in np.unique(targets):
        factual_rows = np.flatnonzero(targets == target)
        pool_indices = np.flatnonzero(train_predictions == target)
        if len(pool_indices) == 0:
            raise ValueError(f"No training rows predicted as target class {target}")
        neighbours = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        neighbours.fit(X_train[pool_indices])
        target_distances, local_indices = neighbours.kneighbors(X_test[factual_rows])
        selected = pool_indices[local_indices[:, 0]]
        prototypes[factual_rows] = X_train[selected]
        prototype_indices[factual_rows] = selected
        distances[factual_rows] = target_distances[:, 0]

    return prototypes, prototype_indices, distances


def greedy_nice_counterfactual(
    disc_model: Any,
    factual: np.ndarray,
    prototype: np.ndarray,
    target: int,
    action_units: Sequence[ActionUnit],
    *,
    plausibility_model: LocalOutlierFactor | None = None,
    tau: float = TAU,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Copy prototype actions greedily until the target prediction is reached."""
    factual = np.asarray(factual, dtype=np.float64)
    prototype = np.asarray(prototype, dtype=np.float64)
    current = factual.copy()
    remaining = [
        unit
        for unit in action_units
        if not np.array_equal(
            current[list(unit.columns)], prototype[list(unit.columns)]
        )
    ]
    selected_units: list[str] = []
    changed_columns: set[int] = set()
    target_probability = float(
        disc_model.predict_proba(current.reshape(1, -1))[0, target]
    )

    while remaining:
        trials: list[np.ndarray] = []
        for unit in remaining:
            trial = current.copy()
            trial[list(unit.columns)] = prototype[list(unit.columns)]
            trials.append(trial)
        trial_matrix = np.stack(trials)
        probabilities = np.asarray(disc_model.predict_proba(trial_matrix))[:, target]
        predictions = np.asarray(disc_model.predict(trial_matrix), dtype=int)
        valid = (predictions == target) & (probabilities >= tau)

        if valid.any() and plausibility_model is not None:
            eligible = np.flatnonzero(valid)
            lof_scores = -np.asarray(
                plausibility_model.score_samples(trial_matrix[eligible]),
                dtype=np.float64,
            )
            best = int(eligible[np.argmin(lof_scores)])
        elif valid.any():
            eligible = np.flatnonzero(valid)
            best = int(eligible[np.argmax(probabilities[eligible])])
        else:
            best = int(np.argmax(probabilities))

        unit = remaining.pop(best)
        current = trial_matrix[best]
        target_probability = float(probabilities[best])
        selected_units.append(unit.name)
        for column in unit.columns:
            if factual[column] != current[column]:
                changed_columns.add(column)
            else:
                changed_columns.discard(column)

        if bool(valid[best]):
            break

    prediction = int(disc_model.predict(current.reshape(1, -1))[0])
    return current, {
        "valid": prediction == target and target_probability >= tau,
        "prediction": prediction,
        "target_probability": target_probability,
        "steps": len(selected_units),
        "changed_columns": len(changed_columns),
        "selected_units": selected_units,
    }


def run_dataset(
    dataset_name: str,
    *,
    max_test: int = DEFAULT_MAX_TEST,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    drop_heloc_all_minus9: bool = True,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run one NICE-NUN baseline case in the fixed Exp9 setting."""
    from experiments.zeroshot_cf.metrics_harness import (
        compute_dicoflex_common_metrics,
        print_metrics,
    )
    from sklearn.neighbors import LocalOutlierFactor

    total_started = time.perf_counter()
    context = prepare_benchmark_context(
        dataset_name,
        max_test=max_test,
        validation_fraction=validation_fraction,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
    )
    bundle = context.bundle
    X_test = context.X_test
    y_test = context.y_test
    action_units = build_action_units(
        list(context.scalar_actionable),
        list(context.grouped_actionable),
    )

    train_predictions = np.asarray(context.disc_model.predict(bundle.X_train), dtype=int)
    plausibility_model = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(
        bundle.X_train
    )

    generation_started = time.perf_counter()
    prototypes, prototype_indices, prototype_distances = nearest_unlike_prototypes(
        bundle.X_train,
        train_predictions,
        X_test,
        context.y_target,
    )
    X_cf = np.empty_like(X_test)
    point_info: list[dict[str, Any]] = []
    for factual, prototype, target in zip(
        X_test,
        prototypes,
        context.y_target,
        strict=True,
    ):
        counterfactual, info = greedy_nice_counterfactual(
            context.disc_model,
            factual,
            prototype,
            int(target),
            action_units,
            plausibility_model=plausibility_model,
        )
        X_cf[len(point_info)] = counterfactual
        point_info.append(info)
    runtime_generation = time.perf_counter() - generation_started

    common_metrics = compute_dicoflex_common_metrics(
        context.disc_model,
        X_cf,
        X_test,
        bundle.X_train,
        context.y_target,
        bundle.numerical_features_indices,
        list(context.immutable_idx),
        categorical_groups=context.categorical_groups,
        sparsity_eps=DEFAULT_SPARSITY_EPS,
    )
    print_metrics(common_metrics, prefix=f"{dataset_name}/NICE-NUN")
    valid = np.asarray([info["valid"] for info in point_info], dtype=bool)
    changed_counts = np.asarray(
        [info["changed_columns"] for info in point_info], dtype=float
    )
    steps = np.asarray([info["steps"] for info in point_info], dtype=float)
    all_l2 = (
        np.linalg.norm(X_cf[valid] - X_test[valid], axis=1)
        if valid.any()
        else np.empty(0, dtype=float)
    )
    row: dict[str, Any] = build_common_result_row(
        context,
        method="nice_nun_greedy_lof",
        cf_per_factual=1,
        extra_fields={
        "prototype_pool_labels": TARGET_CLASSIFIER_LABELS,
        "prototype_metric": "euclidean",
        "valid_candidate_selection": "lof",
        "categorical_actions": "atomic_one_hot_groups",
        "runtime_generation_s": round(runtime_generation, 3),
        **common_metrics,
        "sparsity_exact": float((X_test != X_cf).mean()),
        "true_actionability": common_metrics["actionability"],
        "proximity_all_features_euclidean": (
            float(all_l2.mean()) if len(all_l2) else float("nan")
        ),
        "failure_rate": float((~valid).mean()),
        "l0_count_mean": mean_on_valid(changed_counts, valid),
        "steps_mean": mean_on_valid(steps, valid),
        "prototype_distance_mean": float(prototype_distances.mean()),
        "factual_oob_fraction": float(
            (((X_test < 0.0) | (X_test > 1.0)).any(axis=1)).mean()
        ),
        "cf_oob_fraction": float((((X_cf < 0.0) | (X_cf > 1.0)).any(axis=1)).mean()),
        },
    )
    row["runtime_total_s"] = round(time.perf_counter() - total_started, 3)

    point_rows = [
        {
            "point": index,
            "factual_label": int(y_test[index]),
            "factual_prediction": int(context.y_pred[index]),
            "target": int(context.y_target[index]),
            "cf_prediction": int(info["prediction"]),
            "valid": bool(info["valid"]),
            "target_probability": float(info["target_probability"]),
            "changed_columns": int(info["changed_columns"]),
            "steps": int(info["steps"]),
            "prototype_index": int(prototype_indices[index]),
            "prototype_distance": float(prototype_distances[index]),
        }
        for index, info in enumerate(point_info)
    ]
    write_dataset_outputs(
        dataset_result_paths(results_dir, "exp11_nice_nun", dataset_name),
        row,
        point_rows,
        arrays={
            "X_test": X_test,
            "y_test": y_test,
            "X_cf": X_cf,
            "y_pred": context.y_pred,
            "y_target": context.y_target,
            "y_cf_pred": context.disc_model.predict(X_cf),
            "prototypes": prototypes,
            "prototype_indices": prototype_indices,
        },
    )
    return row


def main() -> None:
    """Run one dataset or all four fixed-protocol datasets locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
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
            validation_fraction=args.validation_fraction,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
            results_dir=args.results_dir,
        )
        for dataset in datasets
    ]
    if args.dataset == "all":
        write_result_table(aggregate_metrics_path(args.results_dir, "exp11_nice_nun"), rows)


if __name__ == "__main__":
    main()
