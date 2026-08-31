#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: T201

"""Single-split TabICL benchmark on the suitable DiCoFlex datasets.

Each invocation evaluates exactly one dataset so the four runs can be
scheduled independently on Athena. Adult is intentionally excluded: its very
wide categorical representation is not a good fit for the current iterative
conditional-imputation search. HELOC is included as the established reference.

The benchmark uses one fixed 64/16/20 train/validation/test split and selects
up to 1,000 held-out factuals with a fixed stratified sample. It returns a
configurable set of valid counterfactuals per factual and reports
method-independent DiCoFlex metrics, grouped mixed-data costs, diversity, and
runtime diagnostics.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.benchmark_protocol import (
    DATASETS,
    DEFAULT_MAX_TEST,
    DEFAULT_SPARSITY_EPS,
    DEFAULT_TEST_SELECTION,
    DEFAULT_VALIDATION_FRACTION,
    TARGET_CLASSIFIER_LABELS,
    aggregate_dataset_metrics,
    build_common_result_row,
    dataset_result_paths,
    mean_on_valid,
    prepare_benchmark_context,
    write_dataset_outputs,
)
from experiments.zeroshot_cf.data import get_one_hot_groups
from experiments.zeroshot_cf.generator import (
    ATHENA_CONTEXT_SIZE,
    ATHENA_CONTEXT_STRATEGY,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_TEMPERATURE,
)
from experiments.zeroshot_cf.metrics_harness import (
    compute_dicoflex_common_metrics,
    print_metrics,
)
from experiments.zeroshot_cf.mixed_distance import (
    grouped_gower_distance,
)
from experiments.zeroshot_cf.reporting import evaluate_diverse_counterfactual_sets
from experiments.zeroshot_cf.retained_config import TAU
from experiments.zeroshot_cf.tabicl_runtime import run_tabicl_benchmark
from sklearn.neighbors import LocalOutlierFactor

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


def _levels_text(values: tuple[float, ...] | None) -> str:
    if values is None:
        return "none"
    return ";".join(f"{value:g}" for value in values)


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
    """Run the fixed training-free benchmark configuration for one dataset."""
    total_started = time.perf_counter()
    context = prepare_benchmark_context(
        dataset_name,
        max_test=max_test,
        validation_fraction=validation_fraction,
        test_selection=DEFAULT_TEST_SELECTION,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
    )
    run = run_tabicl_benchmark(
        context,
        tau=tau,
        temperature=temperature,
        n_estimators=n_estimators,
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
        cache_dir=tabicl_cache_dir,
    )

    bundle = context.bundle
    X_test = context.X_test
    y_test = context.y_test
    X_cf = run.counterfactuals
    diagnostics = run.diagnostics
    categorical_groups = get_one_hot_groups(bundle)
    common_metrics = compute_dicoflex_common_metrics(
        context.disc_model,
        X_cf,
        X_test,
        bundle.X_train,
        context.y_target,
        bundle.numerical_features_indices,
        list(context.immutable_idx),
        categorical_groups=categorical_groups,
        sparsity_eps=DEFAULT_SPARSITY_EPS,
    )
    diverse_metrics = evaluate_diverse_counterfactual_sets(
        X_test=X_test,
        bundle=bundle,
        disc_model=context.disc_model,
        y_target=context.y_target,
        X_cf_sets=run.counterfactual_sets,
        counts=diagnostics.diverse_available_count_per_point,
        tau=tau,
    )
    print_metrics(common_metrics, prefix=f"{dataset_name}/DiCoFlex-common")

    posthoc_lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(bundle.X_train)
    lof_per_point = -np.asarray(posthoc_lof.score_samples(X_cf), dtype=float)

    y_cf_pred = np.asarray(context.disc_model.predict(X_cf), dtype=int)
    valid = y_cf_pred == context.y_target
    changed_column_counts = np.asarray(
        [len(columns) for columns in diagnostics.changed_per_point], dtype=float
    )
    steps = np.asarray(diagnostics.steps_per_point, dtype=float)
    validity_steps = np.asarray(diagnostics.validity_steps_per_point, dtype=float)
    refinement_steps = np.asarray(diagnostics.refinement_steps_per_point, dtype=float)
    initial_valid_records = [
        next(
            (
                step
                for step in history
                if isinstance(step, dict) and step.get("immediate_valid")
            ),
            None,
        )
        for history in diagnostics.history_per_point
    ]

    def history_value(record: dict[str, Any] | None, key: str) -> float:
        value = None if record is None else record.get(key)
        return float("nan") if value is None else float(value)

    def finite_mean(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if len(finite) else float("nan")

    initial_valid_sparsity = np.asarray(
        [history_value(record, "action_sparsity") for record in initial_valid_records],
        dtype=float,
    )
    initial_valid_gower = np.asarray(
        [history_value(record, "grouped_gower") for record in initial_valid_records],
        dtype=float,
    )
    final_action_sparsity = np.asarray(
        [
            0.0
            if not history or not isinstance(history[-1], dict)
            else float(history[-1].get("action_sparsity", np.nan))
            for history in diagnostics.history_per_point
        ],
        dtype=float,
    )
    first_action_types = [
        (
            history[0].get("action_type", "numerical")
            if history and isinstance(history[0], dict)
            else "numerical"
        )
        for history in diagnostics.history_per_point
    ]
    project_l2 = (
        float(np.linalg.norm(X_cf[valid] - X_test[valid], axis=1).mean())
        if valid.any()
        else float("nan")
    )
    grouped_gower_per_point = grouped_gower_distance(
        X_cf,
        X_test,
        bundle.numerical_features_indices,
        categorical_groups,
    )
    initial_action_counts = np.asarray(
        diagnostics.initial_sparse_action_count_per_point, dtype=float
    )
    final_action_counts = np.asarray(diagnostics.final_action_count_per_point, dtype=float)
    l0_count_mean = mean_on_valid(final_action_counts, valid)
    steps_mean = mean_on_valid(steps, valid)
    validity_steps_mean = float(validity_steps.mean())
    accepted_refinements = np.asarray(
        diagnostics.accepted_refinement_count_per_point, dtype=float
    )
    extra_actions = np.asarray(diagnostics.extra_actions_per_point, dtype=float)
    initial_joint_scores = np.asarray(
        diagnostics.initial_tabicl_joint_log_density_per_point, dtype=float
    )
    final_joint_scores = np.asarray(
        diagnostics.final_tabicl_joint_log_density_per_point, dtype=float
    )
    joint_score_gains = np.asarray(
        diagnostics.tabicl_joint_log_density_gain_per_point, dtype=float
    )
    joint_batch_counts = np.asarray(
        diagnostics.joint_scoring_batch_count_per_point, dtype=float
    )
    joint_rows_scored = np.asarray(diagnostics.joint_rows_scored_per_point, dtype=float)
    stopping_reasons = list(diagnostics.refinement_stopping_reason_per_point)
    stopping_reason_counts = ";".join(
        f"{reason}:{stopping_reasons.count(reason)}"
        for reason in sorted(set(stopping_reasons))
    )
    row: dict[str, Any] = build_common_result_row(
        context,
        method=(
            "tabicl_v2_diverse_dpp"
            if n_counterfactuals > 1
            else f"tabicl_v2_{cf_mode}"
        ),
        cf_per_factual=n_counterfactuals,
        extra_fields={
            "cf_mode": cf_mode,
            "context_strategy": ATHENA_CONTEXT_STRATEGY,
            "context_size": ATHENA_CONTEXT_SIZE,
            "context_labels": TARGET_CLASSIFIER_LABELS,
            "candidate_mode": "batched",
            "candidate_quantiles": _levels_text(candidate_quantiles),
            "confidence_quantiles": _levels_text(confidence_quantiles),
            "plausibility_backend": diagnostics.plausibility_backend,
            "tabicl_joint_permutations": tabicl_joint_permutations,
            "max_validity_steps": diagnostics.max_validity_steps,
            "allow_revisits": allow_revisits,
            "categorical_proposal_count": diagnostics.categorical_proposal_count,
            "categorical_confidence_batching": diagnostics.categorical_confidence_batching,
            "conditional_estimator_cache": diagnostics.conditional_estimator_cache,
            "tabicl_kv_cache": diagnostics.tabicl_kv_cache,
            "joint_shortlist_size": joint_shortlist_size,
            "max_extra_actions": max_extra_actions,
            "min_joint_log_gain": min_joint_log_gain,
            "diversity_beam_width": diversity_beam_width,
            "diversity_candidate_pool_size": diversity_candidate_pool_size,
            "diversity_max_extra_actions": diversity_max_extra_actions,
            "diversity_max_gower_ratio": diversity_max_gower_ratio,
            "diversity_max_gower_increase": diversity_max_gower_increase,
            "diversity_candidate_generation": "bounded_beam",
            "diversity_selector": "exact_fixed_size_dpp_map",
            "search_schedule": (
                "bounded_beam_then_exact_fixed_size_dpp_map"
                if n_counterfactuals > 1
                else (
                    "probability_ascent_until_valid_then_one_shot_joint_reranking"
                    if cf_mode == "data_plausible"
                    else "probability_ascent_until_valid_then_min_grouped_gower"
                )
            ),
            "valid_candidate_objective": (
                "quality_constrained_dpp"
                if n_counterfactuals > 1
                else "grouped_gower"
            ),
            "n_estimators": n_estimators,
            "temperature": temperature,
            "tau": tau,
            "runtime_generation_s": round(float(diagnostics.runtime_s), 3),
            "runtime_generation_per_factual_s": float(diagnostics.runtime_s)
            / len(X_test),
            "joint_scoring_runtime_s": float(
                np.asarray(diagnostics.joint_scoring_runtime_s_per_point).sum()
            ),
            "point_runtime_s_mean": float(np.asarray(diagnostics.point_runtime_s).mean()),
            **common_metrics,
            **diverse_metrics,
            "sparsity_exact": float((X_test != X_cf).mean()),
            "true_actionability": common_metrics["actionability"],
            "proximity_all_features_euclidean": project_l2,
            "failure_rate": float((~valid).mean()),
            "l0_count_mean": l0_count_mean,
            "steps_mean": steps_mean,
            "validity_steps_mean": validity_steps_mean,
            "post_valid_refinement": cf_mode == "data_plausible",
            "refinement_steps_mean": float(refinement_steps.mean()),
            "refined_fraction": float((refinement_steps > 0).mean()),
            "accepted_refinement_count_mean": float(accepted_refinements.mean()),
            "initial_tabicl_joint_log_density_mean": finite_mean(initial_joint_scores),
            "final_tabicl_joint_log_density_mean": finite_mean(final_joint_scores),
            "tabicl_joint_log_density_gain_mean": finite_mean(joint_score_gains),
            "joint_scoring_batch_count_mean": float(joint_batch_counts.mean()),
            "joint_rows_scored_mean": float(joint_rows_scored.mean()),
            "extra_actions_mean": float(extra_actions.mean()),
            "initial_sparse_action_count_mean": finite_mean(
                np.where(initial_action_counts >= 0, initial_action_counts, np.nan)
            ),
            "final_action_count_mean": float(final_action_counts.mean()),
            "refinement_stopping_reasons": stopping_reason_counts,
            "initial_valid_action_sparsity_mean": finite_mean(initial_valid_sparsity),
            "initial_valid_grouped_gower_mean": finite_mean(initial_valid_gower),
            "final_action_sparsity_mean": finite_mean(final_action_sparsity),
            "categorical_first_fraction": float(
                np.mean(np.asarray(first_action_types) == "categorical")
            ),
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
            "point": i,
            "factual_label": int(y_test[i]),
            "factual_prediction": int(context.y_pred[i]),
            "target": int(context.y_target[i]),
            "cf_prediction": int(y_cf_pred[i]),
            "valid": bool(y_cf_pred[i] == context.y_target[i]),
            "target_probability": float(diagnostics.target_probability_per_point[i]),
            "lof_score": float(lof_per_point[i]),
            "changed_columns": int(changed_column_counts[i]),
            "initial_sparse_action_count": int(initial_action_counts[i]),
            "final_action_count": int(final_action_counts[i]),
            "steps": int(diagnostics.steps_per_point[i]),
            "validity_steps": int(diagnostics.validity_steps_per_point[i]),
            "attempt_steps": len(diagnostics.attempt_history_per_point[i]),
            "initial_valid_step": diagnostics.initial_valid_step_per_point[i],
            "refinement_steps": int(diagnostics.refinement_steps_per_point[i]),
            "accepted_refinement_count": int(
                diagnostics.accepted_refinement_count_per_point[i]
            ),
            "initial_tabicl_joint_log_density": float(
                diagnostics.initial_tabicl_joint_log_density_per_point[i]
            ),
            "final_tabicl_joint_log_density": float(
                diagnostics.final_tabicl_joint_log_density_per_point[i]
            ),
            "tabicl_joint_log_density_gain": float(
                diagnostics.tabicl_joint_log_density_gain_per_point[i]
            ),
            "joint_scoring_batch_count": int(
                diagnostics.joint_scoring_batch_count_per_point[i]
            ),
            "joint_rows_scored": int(diagnostics.joint_rows_scored_per_point[i]),
            "extra_actions": int(diagnostics.extra_actions_per_point[i]),
            "refinement_stopping_reason": diagnostics.refinement_stopping_reason_per_point[i],
            "initial_valid_action_sparsity": float(initial_valid_sparsity[i]),
            "initial_valid_grouped_gower": float(initial_valid_gower[i]),
            "final_grouped_gower": float(grouped_gower_per_point[i]),
            "final_action_sparsity": float(final_action_sparsity[i]),
            "first_action_type": first_action_types[i],
            "point_runtime_s": float(diagnostics.point_runtime_s[i]),
            "joint_scoring_runtime_s": float(
                diagnostics.joint_scoring_runtime_s_per_point[i]
            ),
        }
        for i in range(len(X_test))
    ]
    write_dataset_outputs(
        dataset_result_paths(results_dir, "exp9_tabicl", dataset_name),
        row,
        point_rows,
        arrays={
            "X_test": X_test,
            "X_sparse": run.sparse_counterfactuals,
            "y_test": y_test,
            "X_cf": X_cf,
            "X_cf_sets": run.counterfactual_sets,
            "diverse_available_count": diagnostics.diverse_available_count_per_point,
            "y_pred": context.y_pred,
            "y_target": context.y_target,
            "y_cf_pred": y_cf_pred,
        },
    )
    return row


def aggregate_results(results_dir: Path = RESULTS_DIR) -> Path:
    """Combine completed per-dataset result rows without rerunning models."""
    return aggregate_dataset_metrics(results_dir, "exp9_tabicl")


def main() -> None:
    """Run one dataset or aggregate the completed result rows."""
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
        help=(
            "Central TabICL conditional-quantile proposal grid (default: 0.1...0.9)."
        ),
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
        help=(
            "Append target-class confidence to the TabICL proposal context "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="Fraction of the provisional 80%% train set used for validation.",
    )
    parser.add_argument(
        "--cf-mode",
        choices=["sparse", "data-plausible"],
        default="sparse",
        help="Counterfactual objective (default: sparse).",
    )
    parser.add_argument(
        "--tabicl-joint-permutations",
        type=int,
        default=DEFAULT_TABICL_JOINT_PERMUTATIONS,
        help="Feature-order permutations in TabICL joint-density scoring.",
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
        "--joint-shortlist-size",
        type=int,
        default=DEFAULT_JOINT_SHORTLIST_SIZE,
        help="Maximum alternatives in the one-shot whole-row scoring batch.",
    )
    parser.add_argument(
        "--max-extra-actions",
        type=int,
        default=DEFAULT_MAX_EXTRA_ACTIONS,
        help="Maximum action-count increase over the initial sparse CFE.",
    )
    parser.add_argument(
        "--min-joint-log-gain",
        type=float,
        default=DEFAULT_MIN_JOINT_LOG_GAIN,
        help="Minimum raw joint-log-density gain over the sparse CFE.",
    )
    parser.add_argument(
        "--n-counterfactuals",
        type=int,
        default=DEFAULT_N_COUNTERFACTUALS,
    )
    parser.add_argument(
        "--diversity-beam-width",
        type=int,
        default=DEFAULT_DIVERSITY_BEAM_WIDTH,
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
        cf_mode=args.cf_mode.replace("-", "_"),
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
