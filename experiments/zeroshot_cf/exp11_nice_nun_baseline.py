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
import csv
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.data import (
    OneHotActionGroup,
    get_grouped_categorical_action_space,
    load_dataset,
)
from experiments.zeroshot_cf.discriminator import train_discriminator
from experiments.zeroshot_cf.exp8_tabicl_cf import _select_test_rows
from experiments.zeroshot_cf.exp9_dicoflex_benchmark import (
    DATASETS,
    DEFAULT_MAX_TEST,
    DEFAULT_VALIDATION_FRACTION,
)
from experiments.zeroshot_cf.metrics_harness import (
    compute_dicoflex_common_metrics,
    print_metrics,
)
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

RESULTS_DIR = Path(__file__).parent / "results" / "local" / "exp11_nice_nun"


@dataclass(frozen=True)
class ActionUnit:
    """One scalar intervention or one atomic one-hot intervention."""

    name: str
    columns: tuple[int, ...]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty result table")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def _action_units(
    scalar_actionable: Sequence[int],
    grouped_actionable: Sequence[OneHotActionGroup],
) -> list[ActionUnit]:
    units = [
        ActionUnit(f"feature_{column}", (int(column),)) for column in scalar_actionable
    ]
    units.extend(
        ActionUnit(group.name, tuple(group.columns)) for group in grouped_actionable
    )
    return units


def nearest_unlike_prototypes(
    X_train: np.ndarray,
    train_predictions: np.ndarray,
    X_test: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the nearest classifier-target training row for each factual."""
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
    tau: float = 0.5,
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
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported Exp11 dataset: {dataset_name!r}")

    total_started = time.perf_counter()
    bundle = load_dataset(
        dataset_name,
        drop_heloc_all_minus9=(
            drop_heloc_all_minus9 if dataset_name == "heloc" else False
        ),
        validation_fraction=validation_fraction,
    )
    limit = None if max_test < 0 else max_test
    X_test, y_test = _select_test_rows(
        bundle.X_test,
        bundle.y_test,
        limit,
        "stratified",
    )
    scalar_actionable, grouped_actionable, immutable_idx = (
        get_grouped_categorical_action_space(bundle)
    )
    action_units = _action_units(scalar_actionable, grouped_actionable)

    discriminator_tag = (
        f"{dataset_name}_drop_all_minus9"
        if bundle.preprocessing_variant == "drop_heloc_all_minus9"
        else dataset_name
    )
    discriminator_tag = f"{discriminator_tag}_{bundle.split_variant}"
    X_disc_eval = bundle.X_val if bundle.X_val is not None else X_test
    y_disc_eval = bundle.y_val if bundle.y_val is not None else y_test
    disc_model = train_discriminator(
        bundle.X_train,
        bundle.y_train,
        X_disc_eval,
        y_disc_eval,
        discriminator_tag,
    )
    y_pred = np.asarray(disc_model.predict(X_test), dtype=int)
    y_target = 1 - y_pred
    train_predictions = np.asarray(disc_model.predict(bundle.X_train), dtype=int)
    plausibility_model = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(
        bundle.X_train
    )

    generation_started = time.perf_counter()
    prototypes, prototype_indices, prototype_distances = nearest_unlike_prototypes(
        bundle.X_train,
        train_predictions,
        X_test,
        y_target,
    )
    X_cf = np.empty_like(X_test)
    point_info: list[dict[str, Any]] = []
    for factual, prototype, target in zip(
        X_test,
        prototypes,
        y_target,
        strict=True,
    ):
        counterfactual, info = greedy_nice_counterfactual(
            disc_model,
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
        disc_model,
        X_cf,
        X_test,
        bundle.X_train,
        y_target,
        bundle.numerical_features_indices,
        immutable_idx,
        sparsity_eps=0.05,
    )
    print_metrics(common_metrics, prefix=f"{dataset_name}/NICE-NUN")
    valid = np.asarray([info["valid"] for info in point_info], dtype=bool)
    changed_counts = np.asarray(
        [info["changed_columns"] for info in point_info], dtype=float
    )
    steps = np.asarray([info["steps"] for info in point_info], dtype=float)
    all_l2 = np.linalg.norm(X_cf[valid] - X_test[valid], axis=1)
    validation_accuracy = (
        float("nan")
        if bundle.X_val is None or bundle.y_val is None
        else float((disc_model.predict(bundle.X_val) == bundle.y_val).mean())
    )
    row: dict[str, Any] = {
        "dataset": dataset_name,
        "method": "nice_nun_greedy_lof",
        "split_variant": bundle.split_variant,
        "split_seed": 42,
        "test_selection": "stratified",
        "n_train": len(bundle.X_train),
        "n_validation": 0 if bundle.X_val is None else len(bundle.X_val),
        "n_test_pool": len(bundle.X_test),
        "n_test": len(X_test),
        "cf_per_factual": 1,
        "target_classifier_validation_accuracy": validation_accuracy,
        "target_classifier_test_accuracy": float((y_pred == y_test).mean()),
        "prototype_pool_labels": "target_classifier",
        "prototype_metric": "euclidean",
        "valid_candidate_selection": "lof",
        "categorical_actions": "atomic_one_hot_groups",
        "preprocessing_variant": bundle.preprocessing_variant,
        "n_dropped_rows": bundle.n_dropped_rows,
        "runtime_generation_s": round(runtime_generation, 3),
        **common_metrics,
        "sparsity_exact": float((X_test != X_cf).mean()),
        "true_actionability": common_metrics["actionability"],
        "proximity_all_features_euclidean": float(all_l2.mean()),
        "failure_rate": float((~valid).mean()),
        "l0_count_mean": float(changed_counts[valid].mean()),
        "steps_mean": float(steps[valid].mean()),
        "prototype_distance_mean": float(prototype_distances.mean()),
        "factual_oob_fraction": float(
            (((X_test < 0.0) | (X_test > 1.0)).any(axis=1)).mean()
        ),
        "cf_oob_fraction": float((((X_cf < 0.0) | (X_cf > 1.0)).any(axis=1)).mean()),
    }
    row["runtime_total_s"] = round(time.perf_counter() - total_started, 3)

    prefix = results_dir / f"exp11_nice_nun_{dataset_name}"
    _write_csv(prefix.with_name(f"{prefix.name}_metrics.csv"), [row])
    point_rows = [
        {
            "point": index,
            "factual_label": int(y_test[index]),
            "factual_prediction": int(y_pred[index]),
            "target": int(y_target[index]),
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
    _write_csv(prefix.with_name(f"{prefix.name}_points.csv"), point_rows)
    np.savez_compressed(
        prefix.with_name(f"{prefix.name}_arrays.npz"),
        X_test=X_test,
        y_test=y_test,
        X_cf=X_cf,
        y_pred=y_pred,
        y_target=y_target,
        y_cf_pred=disc_model.predict(X_cf),
        prototypes=prototypes,
        prototype_indices=prototype_indices,
    )
    return row


def main() -> None:
    """Run one dataset or all five datasets locally."""
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
        _write_csv(args.results_dir / "exp11_nice_nun_all_metrics.csv", rows)


if __name__ == "__main__":
    main()
