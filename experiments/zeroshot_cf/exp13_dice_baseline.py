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
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import dice_ml
import numpy as np
import pandas as pd
from experiments.zeroshot_cf.data import (
    DatasetBundle,
    OneHotActionGroup,
    get_grouped_categorical_action_space,
    get_one_hot_groups,
    load_dataset,
)
from experiments.zeroshot_cf.discriminator import train_discriminator
from experiments.zeroshot_cf.exp8_tabicl_cf import _select_test_rows
from experiments.zeroshot_cf.exp9_dicoflex_benchmark import (
    DATASETS,
    DEFAULT_MAX_TEST,
    DEFAULT_VALIDATION_FRACTION,
)
from experiments.zeroshot_cf.exp11_nice_nun_baseline import _action_units
from experiments.zeroshot_cf.exp12_optimization_baselines import (
    contract_scalar_actions,
    prune_counterfactual_actions,
)
from experiments.zeroshot_cf.metrics_harness import (
    compute_dicoflex_common_metrics,
    print_metrics,
)
from raiutils.exceptions import UserConfigValidationException

RESULTS_DIR = Path(__file__).parent / "results" / "local" / "exp13_dice"
OUTCOME = "target_model_prediction"


@dataclass(frozen=True)
class DiceMixedAdapter:
    """Round-trip between repository one-hot matrices and compact DiCE frames."""

    n_features: int
    scalar_columns: tuple[int, ...]
    groups: tuple[OneHotActionGroup, ...]
    scalar_names: tuple[str, ...]

    @classmethod
    def from_bundle(cls, bundle: DatasetBundle) -> DiceMixedAdapter:
        groups = tuple(get_one_hot_groups(bundle))
        grouped = {column for group in groups for column in group.columns}
        scalar_columns = tuple(
            column
            for column in range(len(bundle.feature_names))
            if column not in grouped
        )
        scalar_names = tuple(bundle.feature_names[column] for column in scalar_columns)
        return cls(
            n_features=len(bundle.feature_names),
            scalar_columns=scalar_columns,
            groups=groups,
            scalar_names=scalar_names,
        )

    @property
    def feature_names(self) -> list[str]:
        return [*self.scalar_names, *(group.name for group in self.groups)]

    def encode(self, X: np.ndarray) -> pd.DataFrame:
        matrix = np.atleast_2d(np.asarray(X, dtype=np.float64))
        data: dict[str, Any] = {
            name: matrix[:, column]
            for name, column in zip(self.scalar_names, self.scalar_columns, strict=True)
        }
        for group in self.groups:
            columns = list(group.columns)
            if not np.allclose(matrix[:, columns].sum(axis=1), 1.0):
                raise ValueError(f"one-hot group {group.name!r} is invalid")
            data[group.name] = [
                str(category) for category in np.argmax(matrix[:, columns], axis=1)
            ]
        return pd.DataFrame(data, columns=self.feature_names)

    def decode(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        compact = (
            frame.loc[:, self.feature_names]
            if isinstance(frame, pd.DataFrame)
            else pd.DataFrame(frame, columns=self.feature_names)
        )
        matrix = np.zeros((len(compact), self.n_features), dtype=np.float64)
        for name, column in zip(self.scalar_names, self.scalar_columns, strict=True):
            matrix[:, column] = np.asarray(
                pd.to_numeric(compact[name]),
                dtype=np.float64,
            )
        for group in self.groups:
            categories = np.asarray(
                pd.to_numeric(compact[group.name]),
                dtype=np.int64,
            )
            if np.any((categories < 0) | (categories >= len(group.columns))):
                raise ValueError(f"category outside group {group.name!r}")
            matrix[np.arange(len(matrix)), np.asarray(group.columns)[categories]] = 1.0
        return matrix


class DiceClassifierAdapter:
    """Sklearn-like classifier accepting compact DiCE frames."""

    def __init__(self, classifier: Any, codec: DiceMixedAdapter) -> None:
        super().__init__()
        self.classifier = classifier
        self.codec = codec
        self.classes_ = np.asarray(getattr(classifier, "classes_", (0, 1)))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.predict_proba(self.codec.decode(X)))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.predict(self.codec.decode(X)))


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


def generate_dice_counterfactuals(
    explainer: Any,
    codec: DiceMixedAdapter,
    classifier: Any,
    X_test: np.ndarray,
    y_target: np.ndarray,
    features_to_vary: list[str],
    *,
    max_iterations: int = 200,
    search_restarts: int = 1,
    stopping_threshold: float = 0.5,
    random_state: int = 42,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate one CF per factual from DiCE's valid pre-sparsification set."""
    if search_restarts < 1:
        raise ValueError("search_restarts must be positive")
    if not 0.5 <= stopping_threshold < 1.0:
        raise ValueError("stopping_threshold must be in [0.5, 1.0)")
    X_cf = np.asarray(X_test, dtype=np.float64).copy()
    point_info: list[dict[str, Any]] = []
    queries = codec.encode(X_test)
    for index, target in enumerate(np.asarray(y_target, dtype=int)):
        started = time.perf_counter()
        returned = False
        valid_candidates = 0
        attempts_used = 0
        for attempt in range(search_restarts):
            attempts_used = attempt + 1
            attempt_seed = random_state + index + attempt * 100_003
            random.seed(attempt_seed)
            np.random.seed(attempt_seed)
            try:
                explainer.generate_counterfactuals(
                    queries.iloc[[index]],
                    total_CFs=1,
                    desired_class=int(target),
                    features_to_vary=features_to_vary,
                    stopping_threshold=stopping_threshold,
                    posthoc_sparsity_param=0.0,
                    posthoc_sparsity_algorithm="binary",
                    initialization="kdtree",
                    proximity_weight=0.2,
                    sparsity_weight=0.2,
                    categorical_penalty=0.1,
                    maxiterations=max_iterations,
                    verbose=False,
                )
                # DiCE rounds continuous values before exposing final_cfs_df
                # and may thereby move a marginal CF back across the boundary.
                # Recover the genetic solver's unrounded candidates instead.
                final_frame = cast(
                    pd.DataFrame | None,
                    explainer.label_decode_cfs(explainer.final_cfs),
                )
                attempt_returned = final_frame is not None and len(final_frame) > 0
            except UserConfigValidationException as error:
                if "No counterfactuals found" not in str(error):
                    raise
                final_frame = None
                attempt_returned = False
            returned = returned or attempt_returned
            if not attempt_returned or final_frame is None:
                continue

            candidates = codec.decode(final_frame)
            predictions = np.asarray(
                classifier.predict(candidates), dtype=int
            )
            valid_indices = np.flatnonzero(predictions == int(target))
            valid_candidates = len(valid_indices)
            if valid_candidates:
                candidates = candidates[valid_indices]
                compact_factual = queries.iloc[index].to_numpy()
                compact_candidates = final_frame.iloc[valid_indices][
                    codec.feature_names
                ].to_numpy()
                changed_actions = (compact_candidates != compact_factual).sum(axis=1)
                l2 = np.linalg.norm(candidates - X_test[index], axis=1)
                selected = np.lexsort((l2, changed_actions))[0]
                X_cf[index] = candidates[selected]
                break
        point_info.append(
            {
                "returned": bool(returned),
                "found": bool(valid_candidates),
                "valid_candidates": int(valid_candidates),
                "attempts": attempts_used,
                "runtime_s": time.perf_counter() - started,
            }
        )
    return X_cf, point_info


def run_dataset(  # noqa: PLR0913
    dataset_name: str,
    *,
    max_test: int = DEFAULT_MAX_TEST,
    max_iterations: int = 200,
    search_restarts: int = 1,
    stopping_threshold: float = 0.5,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    drop_heloc_all_minus9: bool = True,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run official DiCE-genetic with the fixed Exp9 data and classifier."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported Exp13 dataset: {dataset_name!r}")
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
        bundle.X_test, bundle.y_test, limit, "stratified"
    )
    scalar_actionable, grouped_actionable, immutable_idx = (
        get_grouped_categorical_action_space(bundle)
    )
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

    codec = DiceMixedAdapter.from_bundle(bundle)
    train_frame = codec.encode(bundle.X_train)
    train_frame[OUTCOME] = train_predictions
    data_interface = dice_ml.Data(
        dataframe=train_frame,
        continuous_features=list(codec.scalar_names),
        outcome_name=OUTCOME,
    )
    model_interface = dice_ml.Model(
        model=DiceClassifierAdapter(disc_model, codec),
        backend="sklearn",
        model_type="classifier",
    )
    explainer = dice_ml.Dice(data_interface, model_interface, method="genetic")
    vary = _features_to_vary(bundle, codec, scalar_actionable, grouped_actionable)

    generation_started = time.perf_counter()
    X_cf, point_info = generate_dice_counterfactuals(
        explainer,
        codec,
        disc_model,
        X_test,
        y_target,
        vary,
        max_iterations=max_iterations,
        search_restarts=search_restarts,
        stopping_threshold=stopping_threshold,
    )
    X_cf_raw = X_cf.copy()
    action_units = _action_units(scalar_actionable, grouped_actionable)
    for index, target in enumerate(y_target):
        raw_prediction = int(disc_model.predict(X_cf[index : index + 1])[0])
        if raw_prediction != int(target):
            continue
        X_cf[index] = prune_counterfactual_actions(
            disc_model,
            X_test[index],
            X_cf[index],
            int(target),
            action_units,
        )
        X_cf[index] = contract_scalar_actions(
            disc_model,
            X_test[index],
            X_cf[index],
            int(target),
            scalar_actionable,
        )
    runtime_generation = time.perf_counter() - generation_started
    y_cf_pred = np.asarray(disc_model.predict(X_cf), dtype=int)
    probabilities = np.asarray(disc_model.predict_proba(X_cf))[
        np.arange(len(X_cf)), y_target
    ]
    valid = y_cf_pred == y_target
    common_metrics = compute_dicoflex_common_metrics(
        disc_model,
        X_cf,
        X_test,
        bundle.X_train,
        y_target,
        bundle.numerical_features_indices,
        immutable_idx,
        categorical_groups=codec.groups,
        sparsity_eps=0.05,
    )
    print_metrics(common_metrics, prefix=f"{dataset_name}/DiCE-genetic")
    changed_columns = (X_cf != X_test).sum(axis=1)
    raw_changed_columns = (X_cf_raw != X_test).sum(axis=1)
    raw_l2 = np.linalg.norm(X_cf_raw - X_test, axis=1)
    all_l2 = np.linalg.norm(X_cf[valid] - X_test[valid], axis=1)
    found = np.asarray([info["found"] for info in point_info], dtype=bool)
    validation_accuracy = (
        float("nan")
        if bundle.X_val is None or bundle.y_val is None
        else float((disc_model.predict(bundle.X_val) == bundle.y_val).mean())
    )
    row: dict[str, Any] = {
        "dataset": dataset_name,
        "method": "dice_genetic_atomic_pruned",
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
        "preprocessing_variant": bundle.preprocessing_variant,
        "n_dropped_rows": bundle.n_dropped_rows,
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
        "l0_count_mean": (
            float(changed_columns[valid].mean()) if valid.any() else float("nan")
        ),
        "raw_l0_count_mean": float(raw_changed_columns.mean()),
        "raw_proximity_all_features_euclidean": float(raw_l2.mean()),
        "factual_oob_fraction": float(
            (((X_test < 0.0) | (X_test > 1.0)).any(axis=1)).mean()
        ),
        "cf_oob_fraction": float((((X_cf < 0.0) | (X_cf > 1.0)).any(axis=1)).mean()),
    }
    row["runtime_total_s"] = round(time.perf_counter() - total_started, 3)

    prefix = results_dir / f"exp13_dice_genetic_{dataset_name}"
    _write_csv(prefix.with_name(f"{prefix.name}_metrics.csv"), [row])
    _write_csv(
        prefix.with_name(f"{prefix.name}_points.csv"),
        [
            {
                "point": index,
                "factual_label": int(y_test[index]),
                "factual_prediction": int(y_pred[index]),
                "target": int(y_target[index]),
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
    )
    np.savez_compressed(
        prefix.with_name(f"{prefix.name}_arrays.npz"),
        X_test=X_test,
        y_test=y_test,
        X_cf=X_cf,
        y_pred=y_pred,
        y_target=y_target,
        y_cf_pred=y_cf_pred,
        X_cf_raw=X_cf_raw,
    )
    return row


def main() -> None:
    """Run DiCE-genetic locally on one or all Exp9 datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--search-restarts", type=int, default=1)
    parser.add_argument("--stopping-threshold", type=float, default=0.5)
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
        _write_csv(args.results_dir / "exp13_dice_genetic_all_metrics.csv", rows)


if __name__ == "__main__":
    main()
