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
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.baseline_common import (
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
from experiments.zeroshot_cf.data import (
    DatasetBundle,
    get_one_hot_groups,
)
from experiments.zeroshot_cf.retained_config import TAU

if TYPE_CHECKING:
    import pandas as pd

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
        import pandas as pd

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
        import pandas as pd

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
    stopping_threshold: float = TAU,
    random_state: int = 42,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate one CF per factual from DiCE's valid pre-sparsification set."""
    from raiutils.exceptions import UserConfigValidationException

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
                final_frame = explainer.label_decode_cfs(explainer.final_cfs)
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
    stopping_threshold: float = TAU,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    drop_heloc_all_minus9: bool = True,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run official DiCE-genetic with the fixed Exp9 data and classifier."""
    import dice_ml
    from experiments.zeroshot_cf.metrics_harness import (
        compute_dicoflex_common_metrics,
        print_metrics,
    )

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
    train_predictions = np.asarray(
        context.disc_model.predict(bundle.X_train),
        dtype=int,
    )

    codec = DiceMixedAdapter.from_bundle(bundle)
    train_frame = codec.encode(bundle.X_train)
    train_frame[OUTCOME] = train_predictions
    data_interface = dice_ml.Data(
        dataframe=train_frame,
        continuous_features=list(codec.scalar_names),
        outcome_name=OUTCOME,
    )
    model_interface = dice_ml.Model(
        model=DiceClassifierAdapter(context.disc_model, codec),
        backend="sklearn",
        model_type="classifier",
    )
    explainer = dice_ml.Dice(data_interface, model_interface, method="genetic")
    vary = _features_to_vary(
        bundle,
        codec,
        list(context.scalar_actionable),
        list(context.grouped_actionable),
    )

    generation_started = time.perf_counter()
    X_cf, point_info = generate_dice_counterfactuals(
        explainer,
        codec,
        context.disc_model,
        X_test,
        context.y_target,
        vary,
        max_iterations=max_iterations,
        search_restarts=search_restarts,
        stopping_threshold=stopping_threshold,
    )
    X_cf_raw = X_cf.copy()
    action_units = build_action_units(
        list(context.scalar_actionable),
        list(context.grouped_actionable),
    )
    for index, target in enumerate(context.y_target):
        raw_prediction = int(context.disc_model.predict(X_cf[index : index + 1])[0])
        if raw_prediction != int(target):
            continue
        X_cf[index] = prune_counterfactual_actions(
            context.disc_model,
            X_test[index],
            X_cf[index],
            int(target),
            action_units,
            tau=stopping_threshold,
        )
        X_cf[index] = contract_scalar_actions(
            context.disc_model,
            X_test[index],
            X_cf[index],
            int(target),
            list(context.scalar_actionable),
            tau=stopping_threshold,
        )
    runtime_generation = time.perf_counter() - generation_started
    y_cf_pred = np.asarray(context.disc_model.predict(X_cf), dtype=int)
    probabilities = np.asarray(context.disc_model.predict_proba(X_cf))[
        np.arange(len(X_cf)), context.y_target
    ]
    valid = y_cf_pred == context.y_target
    common_metrics = compute_dicoflex_common_metrics(
        context.disc_model,
        X_cf,
        X_test,
        bundle.X_train,
        context.y_target,
        bundle.numerical_features_indices,
        list(context.immutable_idx),
        categorical_groups=codec.groups,
        sparsity_eps=DEFAULT_SPARSITY_EPS,
    )
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

    write_dataset_outputs(
        dataset_result_paths(results_dir, "exp13_dice_genetic", dataset_name),
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
        write_result_table(aggregate_metrics_path(args.results_dir, "exp13_dice_genetic"), rows)


if __name__ == "__main__":
    main()
