#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: T201

"""Single-split TabICL benchmark on the suitable DiCoFlex datasets.

Each invocation evaluates exactly one dataset so the five runs can be
scheduled independently on Athena. Adult is intentionally excluded: its very
wide categorical representation is not a good fit for the current iterative
conditional-imputation search. HELOC is included as the established reference.

The benchmark uses one fixed 64/16/20 train/validation/test split, selects up
to 1,000 held-out factuals with a fixed stratified sample, and generates one
counterfactual per factual. It reports the method-independent subset of the
DiCoFlex metrics alongside the existing project diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.exp4_greedy_cf import TAU
from experiments.zeroshot_cf.exp8_tabicl_cf import (
    ATHENA_CONTEXT_SIZE,
    ATHENA_CONTEXT_STRATEGY,
    DEFAULT_TEMPERATURE,
    generate_tabicl_counterfactuals,
)
from experiments.zeroshot_cf.metrics_harness import (
    compute_dicoflex_common_metrics,
    print_metrics,
)
from sklearn.neighbors import LocalOutlierFactor

DATASETS = (
    "heloc",
    "bank_marketing",
    "give_me_some_credit",
    "lending_club",
    "credit_default",
)
DEFAULT_MAX_TEST = 1000
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_N_ESTIMATORS = 1
DEFAULT_MAX_VALIDITY_STEPS = 100
DEFAULT_MAX_REFINEMENT_STEPS = 2
DEFAULT_MIN_RELATIVE_LOF_GAIN = 0.05
DEFAULT_REFINEMENT_LOF_QUANTILE = 0.90
DEFAULT_CANDIDATE_QUANTILES = tuple(i / 20 for i in range(1, 20))
DEFAULT_CONFIDENCE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
RESULTS_DIR = Path(__file__).parent / "results" / "athena" / "exp9_dicoflex"


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


def _levels_text(values: tuple[float, ...]) -> str:
    return ";".join(f"{value:g}" for value in values)


def run_dataset(  # noqa: PLR0913
    dataset_name: str,
    *,
    max_test: int = DEFAULT_MAX_TEST,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    temperature: float = DEFAULT_TEMPERATURE,
    tau: float = TAU,
    candidate_quantiles: tuple[float, ...] = DEFAULT_CANDIDATE_QUANTILES,
    confidence_quantiles: tuple[float, ...] = DEFAULT_CONFIDENCE_QUANTILES,
    use_lof_refinement: bool = True,
    max_validity_steps: int = DEFAULT_MAX_VALIDITY_STEPS,
    allow_revisits: bool = True,
    max_refinement_steps: int = DEFAULT_MAX_REFINEMENT_STEPS,
    min_relative_lof_gain: float = DEFAULT_MIN_RELATIVE_LOF_GAIN,
    refinement_lof_quantile: float = DEFAULT_REFINEMENT_LOF_QUANTILE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    drop_heloc_all_minus9: bool = True,
    tabicl_cache_dir: Path | None = None,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run the fixed training-free benchmark configuration for one dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported Exp9 dataset: {dataset_name!r}")

    total_started = time.perf_counter()
    X_test, y_test, X_cf, info = generate_tabicl_counterfactuals(
        dataset_name,
        tau=tau,
        temperature=temperature,
        n_estimators=n_estimators,
        max_test=max_test,
        context_labels="disc",
        candidate_mode="batched",
        context_update="replace",
        point_estimate="mode",
        project_to_domain=True,
        candidate_quantiles=candidate_quantiles,
        confidence_quantiles=confidence_quantiles,
        use_lof_refinement=use_lof_refinement,
        max_validity_steps=max_validity_steps,
        allow_revisits=allow_revisits,
        max_refinement_steps=max_refinement_steps,
        min_relative_lof_gain=min_relative_lof_gain,
        refinement_lof_quantile=refinement_lof_quantile,
        validation_fraction=validation_fraction,
        test_selection="stratified",
        drop_heloc_all_minus9=(
            drop_heloc_all_minus9 if dataset_name == "heloc" else False
        ),
        cache_dir=tabicl_cache_dir,
    )

    bundle = info["bundle"]
    common_metrics = compute_dicoflex_common_metrics(
        info["disc_model"],
        X_cf,
        X_test,
        bundle.X_train,
        info["y_target"],
        bundle.numerical_features_indices,
        info["immutable_idx"],
        sparsity_eps=0.05,
    )
    print_metrics(common_metrics, prefix=f"{dataset_name}/DiCoFlex-common")

    lof_per_point = info["lof_per_point"]
    if lof_per_point is None:
        posthoc_lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(
            bundle.X_train
        )
        lof_per_point = -np.asarray(posthoc_lof.score_samples(X_cf), dtype=float)

    y_cf_pred = np.asarray(info["disc_model"].predict(X_cf), dtype=int)
    valid = y_cf_pred == info["y_target"]
    changed_counts = np.asarray(
        [len(columns) for columns in info["changed_per_point"]],
        dtype=float,
    )
    steps = np.asarray(info["steps_per_point"], dtype=float)
    validity_steps = np.asarray(info["validity_steps_per_point"], dtype=float)
    refinement_steps = np.asarray(info["refinement_steps_per_point"], dtype=float)
    initial_valid_records = [
        next(
            (
                step
                for step in history
                if isinstance(step, dict) and step.get("immediate_valid")
            ),
            None,
        )
        for history in info["history_per_point"]
    ]

    def history_value(record: dict[str, Any] | None, key: str) -> float:
        value = None if record is None else record.get(key)
        return float("nan") if value is None else float(value)

    def finite_mean(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if len(finite) else float("nan")

    initial_valid_lof = np.asarray(
        [history_value(record, "lof") for record in initial_valid_records],
        dtype=float,
    )
    initial_valid_sparsity = np.asarray(
        [history_value(record, "action_sparsity") for record in initial_valid_records],
        dtype=float,
    )
    initial_valid_proximity = np.asarray(
        [history_value(record, "proximity_l2") for record in initial_valid_records],
        dtype=float,
    )
    final_action_sparsity = np.asarray(
        [
            0.0
            if not history or not isinstance(history[-1], dict)
            else float(history[-1].get("action_sparsity", np.nan))
            for history in info["history_per_point"]
        ],
        dtype=float,
    )
    first_action_types = [
        (
            history[0].get("action_type", "numerical")
            if history and isinstance(history[0], dict)
            else "numerical"
        )
        for history in info["history_per_point"]
    ]
    project_l2 = (
        float(np.linalg.norm(X_cf[valid] - X_test[valid], axis=1).mean())
        if valid.any()
        else float("nan")
    )
    l0_count_mean = float(changed_counts[valid].mean()) if valid.any() else float("nan")
    steps_mean = float(steps[valid].mean()) if valid.any() else float("nan")
    validity_steps_mean = float(validity_steps.mean())

    validation_accuracy = float("nan")
    if bundle.X_val is not None and bundle.y_val is not None:
        validation_accuracy = float(
            (info["disc_model"].predict(bundle.X_val) == bundle.y_val).mean()
        )

    row: dict[str, Any] = {
        "dataset": dataset_name,
        "method": "tabicl_v2_greedy_icl_validity_gate_lof",
        "split_variant": bundle.split_variant,
        "split_seed": 42,
        "test_selection": "stratified",
        "n_train": len(bundle.X_train),
        "n_validation": 0 if bundle.X_val is None else len(bundle.X_val),
        "n_test_pool": len(bundle.X_test),
        "n_test": len(X_test),
        "cf_per_factual": 1,
        "target_classifier_validation_accuracy": validation_accuracy,
        "target_classifier_test_accuracy": float((info["y_pred"] == y_test).mean()),
        "context_strategy": ATHENA_CONTEXT_STRATEGY,
        "context_size": ATHENA_CONTEXT_SIZE,
        "context_labels": "target_classifier",
        "candidate_mode": "batched",
        "candidate_quantiles": _levels_text(candidate_quantiles),
        "confidence_quantiles": _levels_text(confidence_quantiles),
        "use_lof_refinement": use_lof_refinement,
        "max_validity_steps": max_validity_steps,
        "allow_revisits": allow_revisits,
        "categorical_proposal_count": info["categorical_proposal_count"],
        "categorical_confidence_batching": info[
            "categorical_confidence_batching"
        ],
        "conditional_estimator_cache": info["conditional_estimator_cache"],
        "tabicl_kv_cache": info["tabicl_kv_cache"],
        "max_refinement_steps": max_refinement_steps,
        "min_relative_lof_gain": min_relative_lof_gain,
        "refinement_lof_quantile": refinement_lof_quantile,
        "refinement_lof_threshold": info["refinement_lof_threshold"],
        "refinement_lof_threshold_source": info["refinement_lof_threshold_source"],
        "search_schedule": "probability_ascent_until_valid_then_lof_gate",
        "n_estimators": n_estimators,
        "temperature": temperature,
        "tau": tau,
        "preprocessing_variant": info["preprocessing_variant"],
        "n_dropped_rows": info["n_dropped_rows"],
        "runtime_generation_s": round(float(info["runtime_s"]), 3),
        **common_metrics,
        "sparsity_exact": float((X_test != X_cf).mean()),
        "true_actionability": common_metrics["actionability"],
        "proximity_all_features_euclidean": project_l2,
        "failure_rate": float((~valid).mean()),
        "l0_count_mean": l0_count_mean,
        "steps_mean": steps_mean,
        "validity_steps_mean": validity_steps_mean,
        "post_valid_refinement": use_lof_refinement,
        "refinement_steps_mean": float(refinement_steps.mean()),
        "refined_fraction": float((refinement_steps > 0).mean()),
        "initial_valid_lof_mean": finite_mean(initial_valid_lof),
        "initial_valid_action_sparsity_mean": finite_mean(initial_valid_sparsity),
        "initial_valid_proximity_l2_mean": finite_mean(initial_valid_proximity),
        "final_action_sparsity_mean": finite_mean(final_action_sparsity),
        "refinement_lof_reduction_mean": finite_mean(initial_valid_lof - lof_per_point),
        "categorical_first_fraction": float(
            np.mean(np.asarray(first_action_types) == "categorical")
        ),
        "factual_oob_fraction": float(
            (((X_test < 0.0) | (X_test > 1.0)).any(axis=1)).mean()
        ),
        "cf_oob_fraction": float((((X_cf < 0.0) | (X_cf > 1.0)).any(axis=1)).mean()),
    }
    row["runtime_total_s"] = round(time.perf_counter() - total_started, 3)

    prefix = results_dir / f"exp9_tabicl_{dataset_name}"
    _write_csv(prefix.with_name(f"{prefix.name}_metrics.csv"), [row])

    point_rows = [
        {
            "point": i,
            "factual_label": int(y_test[i]),
            "factual_prediction": int(info["y_pred"][i]),
            "target": int(info["y_target"][i]),
            "cf_prediction": int(y_cf_pred[i]),
            "valid": bool(y_cf_pred[i] == info["y_target"][i]),
            "target_probability": float(info["target_probability_per_point"][i]),
            "lof_score": float(lof_per_point[i]),
            "changed_columns": len(info["changed_per_point"][i]),
            "steps": int(info["steps_per_point"][i]),
            "validity_steps": int(info["validity_steps_per_point"][i]),
            "attempt_steps": len(info["attempt_history_per_point"][i]),
            "initial_valid_step": info["initial_valid_step_per_point"][i],
            "refinement_steps": int(info["refinement_steps_per_point"][i]),
            "initial_valid_lof": float(initial_valid_lof[i]),
            "initial_valid_action_sparsity": float(initial_valid_sparsity[i]),
            "initial_valid_proximity_l2": float(initial_valid_proximity[i]),
            "final_action_sparsity": float(final_action_sparsity[i]),
            "refinement_lof_reduction": float(initial_valid_lof[i] - lof_per_point[i]),
            "first_action_type": first_action_types[i],
        }
        for i in range(len(X_test))
    ]
    _write_csv(prefix.with_name(f"{prefix.name}_points.csv"), point_rows)
    arrays_path = prefix.with_name(f"{prefix.name}_arrays.npz")
    np.savez_compressed(
        arrays_path,
        X_test=X_test,
        y_test=y_test,
        X_cf=X_cf,
        y_pred=info["y_pred"],
        y_target=info["y_target"],
        y_cf_pred=y_cf_pred,
    )
    print(f"Wrote {arrays_path}")
    return row


def aggregate_results(results_dir: Path = RESULTS_DIR) -> Path:
    """Combine completed per-dataset result rows without rerunning models."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset_name in DATASETS:
        path = results_dir / f"exp9_tabicl_{dataset_name}_metrics.csv"
        if not path.exists():
            missing.append(dataset_name)
            continue
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if missing:
        raise FileNotFoundError(
            "Missing per-dataset Exp9 results for: " + ", ".join(missing)
        )
    output = results_dir / "exp9_tabicl_all_metrics.csv"
    _write_csv(output, rows)
    return output


def main() -> None:
    """Run one dataset or aggregate the five completed result rows."""
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
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="Fraction of the provisional 80%% train set used for validation.",
    )
    parser.add_argument(
        "--lof-refinement",
        dest="use_lof_refinement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable validity-preserving LOF refinement after the class flip.",
    )
    parser.add_argument(
        "--max-validity-steps",
        type=int,
        default=DEFAULT_MAX_VALIDITY_STEPS,
        help="Maximum committed probability-ascent actions before validity.",
    )
    parser.add_argument(
        "--allow-revisits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow action units to be proposed again after intervening edits.",
    )
    parser.add_argument(
        "--max-refinement-steps",
        type=int,
        default=DEFAULT_MAX_REFINEMENT_STEPS,
        help="Maximum validity-preserving LOF refinement actions.",
    )
    parser.add_argument(
        "--min-relative-lof-gain",
        type=float,
        default=DEFAULT_MIN_RELATIVE_LOF_GAIN,
        help="Minimum relative LOF reduction required per refinement action.",
    )
    parser.add_argument(
        "--refinement-lof-quantile",
        type=float,
        default=DEFAULT_REFINEMENT_LOF_QUANTILE,
        help="Validation-LOF quantile above which valid CFs are refined.",
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
        confidence_quantiles=tuple(args.confidence_quantiles),
        use_lof_refinement=args.use_lof_refinement,
        max_validity_steps=args.max_validity_steps,
        allow_revisits=args.allow_revisits,
        max_refinement_steps=args.max_refinement_steps,
        min_relative_lof_gain=args.min_relative_lof_gain,
        refinement_lof_quantile=args.refinement_lof_quantile,
        validation_fraction=args.validation_fraction,
        drop_heloc_all_minus9=args.drop_heloc_all_minus9,
        tabicl_cache_dir=args.tabicl_cache_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
