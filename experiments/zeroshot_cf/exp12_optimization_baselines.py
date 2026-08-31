#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: T201

"""Model-agnostic optimization baselines for the fixed Exp9 setting.

The implementations adapt two established counterfactual strategies to the
repository's mixed-data action space:

* ``wachter`` minimizes a Wachter-style distance-plus-classification-loss
  objective through black-box coordinate search.
* ``growing_spheres`` samples expanding neighbourhoods until it finds target
  predictions, then removes unnecessary actions and contracts scalar edits.

Both methods use only ``predict`` and ``predict_proba`` from the target model.
One-hot groups are always changed atomically and immutable features are never
offered to either optimizer. Dataset splitting, factual selection, target
labels, and metrics are identical to Exp9 and Exp11.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.baseline_common import (
    ActionUnit,
    build_action_units,
    contract_scalar_actions,
    prune_counterfactual_actions,
)
from experiments.zeroshot_cf.benchmark_protocol import (
    DATASETS,
    DEFAULT_MAX_TEST,
    DEFAULT_SPARSITY_EPS,
    DEFAULT_VALIDATION_FRACTION,
    aggregate_metrics_path,
    build_common_result_row,
    dataset_result_paths,
    mean_on_valid,
    prepare_benchmark_context,
    write_dataset_outputs,
    write_result_table,
)
from experiments.zeroshot_cf.retained_config import TAU

METHODS = ("wachter", "growing_spheres")
RESULTS_DIR = Path(__file__).parent / "results" / "local" / "exp12_optimization"
WACHTER_QUANTILES = tuple(i / 20 for i in range(1, 20))


def _is_valid(
    disc_model: Any,
    rows: np.ndarray,
    target: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.atleast_2d(rows)
    probabilities = np.asarray(disc_model.predict_proba(matrix))[:, target]
    predictions = np.asarray(disc_model.predict(matrix), dtype=int)
    return (predictions == target) & (probabilities >= tau), probabilities


def _action_sparsity(
    rows: np.ndarray,
    factual: np.ndarray,
    scalar_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
) -> np.ndarray:
    matrix = np.atleast_2d(rows)
    counts = np.zeros(len(matrix), dtype=int)
    if scalar_columns:
        columns = np.asarray(scalar_columns, dtype=int)
        counts += np.count_nonzero(
            ~np.isclose(matrix[:, columns], factual[columns]),
            axis=1,
        )
    for group in categorical_groups:
        columns = list(group.columns)
        counts += np.argmax(matrix[:, columns], axis=1) != int(
            np.argmax(factual[columns])
        )
    return counts


def _changed_units(
    factual: np.ndarray,
    candidate: np.ndarray,
    action_units: Sequence[ActionUnit],
) -> list[ActionUnit]:
    return [
        unit
        for unit in action_units
        if not np.allclose(
            factual[list(unit.columns)],
            candidate[list(unit.columns)],
        )
    ]


def _coordinate_trials(
    current: np.ndarray,
    scalar_values: Mapping[int, np.ndarray],
    categorical_groups: Sequence[OneHotActionGroup],
) -> np.ndarray:
    trials: list[np.ndarray] = []
    for column, values in scalar_values.items():
        for value in values:
            if np.isclose(value, current[column]):
                continue
            trial = current.copy()
            trial[column] = value
            trials.append(trial)
    for group in categorical_groups:
        columns = list(group.columns)
        current_category = int(np.argmax(current[columns]))
        for category, column in enumerate(columns):
            if category == current_category:
                continue
            trial = current.copy()
            trial[columns] = 0.0
            trial[column] = 1.0
            trials.append(trial)
    return np.stack(trials) if trials else np.empty((0, len(current)))


def wachter_coordinate_counterfactual(
    disc_model: Any,
    factual: np.ndarray,
    target: int,
    scalar_values: Mapping[int, np.ndarray],
    categorical_groups: Sequence[OneHotActionGroup],
    action_units: Sequence[ActionUnit],
    *,
    tau: float = TAU,
    loss_weights: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    max_steps_per_weight: int = 12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Black-box mixed-data coordinate minimization of a Wachter objective."""
    factual = np.asarray(factual, dtype=np.float64)
    best_valid: np.ndarray | None = None
    best_valid_key = (np.inf, np.inf)
    best_probability = float(
        disc_model.predict_proba(factual.reshape(1, -1))[0, target]
    )
    best_probability_row = factual.copy()
    evaluations = 1

    for loss_weight in loss_weights:
        current = factual.copy()
        for _ in range(max_steps_per_weight):
            trials = _coordinate_trials(current, scalar_values, categorical_groups)
            if not len(trials):
                break
            evaluations += len(trials)
            valid, probabilities = _is_valid(disc_model, trials, target, tau)
            distances = np.abs(trials - factual).sum(axis=1)
            losses = np.maximum(0.0, tau - probabilities) ** 2
            objective = distances + float(loss_weight) * losses
            current_valid, current_probability = _is_valid(
                disc_model, current, target, tau
            )
            current_distance = float(np.abs(current - factual).sum())
            current_objective = (
                current_distance
                + float(loss_weight)
                * max(0.0, tau - float(current_probability[0])) ** 2
            )
            best = int(np.argmin(objective))
            if float(objective[best]) >= current_objective - 1e-12:
                break
            current = trials[best]

            probability_best = int(np.argmax(probabilities))
            if probabilities[probability_best] > best_probability:
                best_probability = float(probabilities[probability_best])
                best_probability_row = trials[probability_best].copy()

            if valid.any():
                eligible = np.flatnonzero(valid)
                sparsity = _action_sparsity(
                    trials[eligible],
                    factual,
                    tuple(scalar_values),
                    categorical_groups,
                )
                l2 = np.linalg.norm(trials[eligible] - factual, axis=1)
                order = np.lexsort((l2, sparsity))
                candidate = trials[int(eligible[order[0]])]
                key = (int(sparsity[order[0]]), float(l2[order[0]]))
                if key < best_valid_key:
                    best_valid = candidate.copy()
                    best_valid_key = key
            if bool(current_valid[0]):
                break

    candidate = best_probability_row if best_valid is None else best_valid
    if best_valid is not None:
        candidate = prune_counterfactual_actions(
            disc_model, factual, candidate, target, action_units, tau=tau
        )
        candidate = contract_scalar_actions(
            disc_model,
            factual,
            candidate,
            target,
            tuple(scalar_values),
            tau=tau,
        )
    valid, probabilities = _is_valid(disc_model, candidate, target, tau)
    return candidate, {
        "valid": bool(valid[0]),
        "target_probability": float(probabilities[0]),
        "evaluations": evaluations,
    }


def _sample_sphere_candidates(
    factual: np.ndarray,
    scalar_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    rng: np.random.Generator,
    n_candidates: int,
    radius: float,
) -> np.ndarray:
    trials = np.repeat(factual.reshape(1, -1), n_candidates, axis=0)
    if scalar_columns:
        columns = np.asarray(scalar_columns, dtype=int)
        directions = rng.normal(size=(n_candidates, len(columns)))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= np.maximum(norms, 1e-12)
        radial = radius * rng.uniform(0.5, 1.0, n_candidates) ** (
            1.0 / max(1, len(columns))
        )
        trials[:, columns] = np.clip(
            factual[columns] + directions * radial[:, None],
            0.0,
            1.0,
        )
    category_probability = min(0.5, max(0.02, radius / 3.0))
    for group in categorical_groups:
        columns = list(group.columns)
        factual_category = int(np.argmax(factual[columns]))
        change = rng.random(n_candidates) < category_probability
        alternatives = rng.integers(0, len(columns) - 1, n_candidates)
        alternatives += alternatives >= factual_category
        selected = np.flatnonzero(change)
        if len(selected):
            trials[np.ix_(selected, columns)] = 0.0
            trials[selected, np.asarray(columns)[alternatives[selected]]] = 1.0
    return trials


def growing_spheres_counterfactual(
    disc_model: Any,
    factual: np.ndarray,
    target: int,
    scalar_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    action_units: Sequence[ActionUnit],
    *,
    tau: float = TAU,
    n_candidates: int = 512,
    initial_radius: float = 0.05,
    radius_multiplier: float = 1.5,
    max_shells: int = 11,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mixed-data Growing Spheres with pruning and scalar contraction."""
    factual = np.asarray(factual, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    best_probability = float(
        disc_model.predict_proba(factual.reshape(1, -1))[0, target]
    )
    best_probability_row = factual.copy()
    candidate: np.ndarray | None = None
    evaluations = 1
    radius = initial_radius

    for _ in range(max_shells):
        trials = _sample_sphere_candidates(
            factual,
            scalar_columns,
            categorical_groups,
            rng,
            n_candidates,
            radius,
        )
        evaluations += len(trials)
        valid, probabilities = _is_valid(disc_model, trials, target, tau)
        probability_best = int(np.argmax(probabilities))
        if probabilities[probability_best] > best_probability:
            best_probability = float(probabilities[probability_best])
            best_probability_row = trials[probability_best].copy()
        if valid.any():
            eligible = np.flatnonzero(valid)
            sparsity = _action_sparsity(
                trials[eligible], factual, scalar_columns, categorical_groups
            )
            distances = np.linalg.norm(trials[eligible] - factual, axis=1)
            order = np.lexsort((distances, sparsity))
            candidate = trials[int(eligible[order[0]])]
            break
        radius *= radius_multiplier

    final_candidate: np.ndarray
    if candidate is None:
        final_candidate = best_probability_row
    else:
        final_candidate = prune_counterfactual_actions(
            disc_model, factual, candidate, target, action_units, tau=tau
        )
        final_candidate = contract_scalar_actions(
            disc_model,
            factual,
            final_candidate,
            target,
            scalar_columns,
            tau=tau,
        )
    valid, probabilities = _is_valid(disc_model, final_candidate, target, tau)
    return final_candidate, {
        "valid": bool(valid[0]),
        "target_probability": float(probabilities[0]),
        "evaluations": evaluations,
        "final_radius": radius,
    }


def run_dataset(  # noqa: PLR0913
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
    """Run one optimization baseline under the fixed Exp9 protocol."""
    from experiments.zeroshot_cf.metrics_harness import (
        compute_dicoflex_common_metrics,
        print_metrics,
    )

    if method not in METHODS:
        raise ValueError(f"Unsupported Exp12 method: {method!r}")

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
    scalar_values = {
        int(column): np.unique(
            np.quantile(bundle.X_train[:, column], WACHTER_QUANTILES)
        )
        for column in context.scalar_actionable
    }

    generation_started = time.perf_counter()
    X_cf = np.empty_like(X_test)
    point_info: list[dict[str, Any]] = []
    for index, (factual, target) in enumerate(
        zip(X_test, context.y_target, strict=True)
    ):
        if method == "wachter":
            candidate, info = wachter_coordinate_counterfactual(
                context.disc_model,
                factual,
                int(target),
                scalar_values,
                list(context.grouped_actionable),
                action_units,
            )
        else:
            candidate, info = growing_spheres_counterfactual(
                context.disc_model,
                factual,
                int(target),
                list(context.scalar_actionable),
                list(context.grouped_actionable),
                action_units,
                n_candidates=sphere_candidates,
                random_state=random_state + index,
            )
        X_cf[index] = candidate
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
    print_metrics(common_metrics, prefix=f"{dataset_name}/{method}")
    valid = np.asarray([info["valid"] for info in point_info], dtype=bool)
    changed_columns = (X_cf != X_test).sum(axis=1)
    action_counts = np.asarray(
        [
            _action_sparsity(
                X_cf[i],
                X_test[i],
                list(context.scalar_actionable),
                list(context.grouped_actionable),
            )[0]
            for i in range(len(X_test))
        ],
        dtype=float,
    )
    all_l2 = (
        np.linalg.norm(X_cf[valid] - X_test[valid], axis=1)
        if valid.any()
        else np.empty(0, dtype=float)
    )
    evaluations = np.asarray([info["evaluations"] for info in point_info], dtype=float)
    row: dict[str, Any] = build_common_result_row(
        context,
        method=method,
        cf_per_factual=1,
        extra_fields={
            "model_access": "predict_and_predict_proba",
            "categorical_actions": "atomic_one_hot_groups",
            "posthoc_action_pruning": True,
            "posthoc_scalar_contraction": True,
            "sphere_candidates": (
                sphere_candidates if method == "growing_spheres" else 0
            ),
            "runtime_generation_s": round(runtime_generation, 3),
            **common_metrics,
            "sparsity_exact": float((X_test != X_cf).mean()),
            "true_actionability": common_metrics["actionability"],
            "proximity_all_features_euclidean": (
                float(all_l2.mean()) if len(all_l2) else float("nan")
            ),
            "failure_rate": float((~valid).mean()),
            "l0_count_mean": mean_on_valid(changed_columns, valid),
            "action_count_mean": mean_on_valid(action_counts, valid),
            "model_evaluations_mean": float(evaluations.mean()),
            "factual_oob_fraction": float(
                (((X_test < 0.0) | (X_test > 1.0)).any(axis=1)).mean()
            ),
            "cf_oob_fraction": float(
                (((X_cf < 0.0) | (X_cf > 1.0)).any(axis=1)).mean()
            ),
        },
    )
    row["runtime_total_s"] = round(time.perf_counter() - total_started, 3)

    point_rows = [
        {
            "point": index,
            "factual_label": int(y_test[index]),
            "factual_prediction": int(context.y_pred[index]),
            "target": int(context.y_target[index]),
            "cf_prediction": int(context.disc_model.predict(X_cf[index : index + 1])[0]),
            "valid": bool(info["valid"]),
            "target_probability": float(info["target_probability"]),
            "changed_columns": int(changed_columns[index]),
            "action_count": int(action_counts[index]),
            "model_evaluations": int(info["evaluations"]),
        }
        for index, info in enumerate(point_info)
    ]
    write_dataset_outputs(
        dataset_result_paths(results_dir, f"exp12_{method}", dataset_name),
        row,
        point_rows,
        arrays={
            "X_test": X_test,
            "y_test": y_test,
            "X_cf": X_cf,
            "y_pred": context.y_pred,
            "y_target": context.y_target,
            "y_cf_pred": context.disc_model.predict(X_cf),
        },
    )
    return row


def main() -> None:
    """Run selected optimization baselines locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--sphere-candidates", type=int, default=512)
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
    methods = METHODS if args.method == "all" else (args.method,)
    rows = [
        run_dataset(
            dataset,
            method,
            max_test=args.max_test,
            validation_fraction=args.validation_fraction,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
            sphere_candidates=args.sphere_candidates,
            results_dir=args.results_dir,
        )
        for dataset in datasets
        for method in methods
    ]
    write_result_table(aggregate_metrics_path(args.results_dir, "exp12_optimization"), rows)


if __name__ == "__main__":
    main()
