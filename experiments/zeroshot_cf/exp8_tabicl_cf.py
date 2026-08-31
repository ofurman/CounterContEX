#  Copyright (c) Prior Labs GmbH 2026.

"""Experiment 8 compatibility adapter for the retained TabICL generator."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.zeroshot_cf.candidate_domains import infer_feature_domains
from experiments.zeroshot_cf.generator import (
    ATHENA_CONTEXT_SIZE,
    ATHENA_CONTEXT_STRATEGY,
    DEFAULT_CATEGORICAL_PROPOSAL_COUNT,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_POINT_ESTIMATE,
    DEFAULT_TEMPERATURE,
    TabICLGeneratorConfig,
    TabICLGeneratorInputs,
    TabICLGeneratorPointBackend,
    empirical_confidence_grid,
    generate_counterfactual_batch,
    select_test_rows,
)
from experiments.zeroshot_cf.grouped_categorical import (
    CompactMixedSampler,
    ConditionedCategoryDistribution,
    GroupedCategoricalCodec,
)
from experiments.zeroshot_cf.reporting import (
    evaluate_counterfactuals,
    evaluate_diverse_sets,
)
from experiments.zeroshot_cf.retained_config import DATASET_PARAMS, TAU

RESULTS_DIR = Path(__file__).parent / "results"
CF_MODES = ("sparse", "data_plausible")
_select_test_rows = select_test_rows


def _resolve_max_test(dataset_name: str, max_test: int | None) -> int | None:
    if max_test is not None and max_test < 0:
        return None
    if max_test is not None:
        return max_test
    return DATASET_PARAMS.get(dataset_name, {"max_test": 50})["max_test"]


def _build_category_distribution(
    *,
    sampler_context: Any,
    categorical_codec: GroupedCategoricalCodec,
    target_class: int,
    confidence_grid: tuple[float, ...] | None,
) -> ConditionedCategoryDistribution:
    cache: dict[tuple[bytes, str], tuple[np.ndarray, np.ndarray]] = {}
    anchors = (None,) if confidence_grid is None else confidence_grid

    def conditioned_category_distribution(
        row: np.ndarray,
        group: Any,
        confidence: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (np.ascontiguousarray(row).tobytes(), group.name)
        if key not in cache:
            encoded_row = categorical_codec.encode_row(row)
            encoded_col = categorical_codec.encoded_column_for_group(group)
            fixed_confidences = (
                None
                if anchors == (None,)
                else np.asarray(anchors, dtype=np.float32)
            )
            categories, probability_grid = sampler_context.categorical_distribution(
                encoded_row.reshape(1, -1),
                encoded_col,
                fixed_target=target_class,
                fixed_confidence=fixed_confidences,
            )
            cache[key] = (
                np.asarray(categories, dtype=int),
                np.atleast_2d(np.asarray(probability_grid, dtype=np.float64)),
            )
        categories, probability_grid = cache[key]
        if confidence is None:
            anchor_index = 0
        else:
            matches = np.flatnonzero(np.isclose(np.asarray(anchors, dtype=float), confidence))
            if not len(matches):
                raise ValueError(f"unknown categorical confidence anchor: {confidence}")
            anchor_index = int(matches[0])
        return categories, probability_grid[anchor_index]

    return conditioned_category_distribution


def _build_point_backend_factory(
    X_train: np.ndarray,
    y_context: np.ndarray,
    *,
    grouped_actionable: Sequence[Any],
    categorical_codec: GroupedCategoricalCodec | None,
    context_probabilities: np.ndarray | None,
    confidence_quantiles: tuple[float, ...] | None,
    cf_mode: str,
    tabicl_joint_permutations: int,
    n_estimators: int,
    temperature: float,
    cache_dir: Path | None,
):
    from experiments.zeroshot_cf.tabicl_checkpoints import TABICL_DEVICE
    from experiments.zeroshot_cf.tabicl_joint_plausibility import TabICLJointScorer
    from experiments.zeroshot_cf.tabicl_sampler import TabICLConditionalDensitySampler

    categorical_features = (
        None if categorical_codec is None else categorical_codec.categorical_columns
    )
    sampler_context = TabICLConditionalDensitySampler(
        n_estimators=n_estimators,
        temperature=temperature,
        random_state=42,
        device=TABICL_DEVICE,
        cache_dir=cache_dir,
        numerical_point_estimate=DEFAULT_POINT_ESTIMATE,
        categorical_features=categorical_features,
    )
    proposal_sampler = (
        sampler_context
        if categorical_codec is None
        else CompactMixedSampler(sampler_context, categorical_codec)
    )
    X_sampler_train = (
        X_train if categorical_codec is None else categorical_codec.encode(X_train)
    )

    joint_sampler_context = None
    joint_sampler = None
    if cf_mode == "data_plausible":
        joint_sampler_context = TabICLConditionalDensitySampler(
            n_estimators=n_estimators,
            temperature=temperature,
            random_state=42,
            device=TABICL_DEVICE,
            cache_dir=cache_dir,
            numerical_point_estimate=DEFAULT_POINT_ESTIMATE,
            categorical_features=categorical_features,
        )
        joint_sampler = (
            joint_sampler_context
            if categorical_codec is None
            else CompactMixedSampler(joint_sampler_context, categorical_codec)
        )

    def point_backend_factory(
        factual: np.ndarray,
        target_class: int,
    ) -> TabICLGeneratorPointBackend:
        query = factual if categorical_codec is None else categorical_codec.encode_row(factual)
        confidence_context = (
            None
            if context_probabilities is None
            else context_probabilities[:, int(target_class)]
        )
        sampler_context.set_context(
            X_sampler_train,
            y_context=y_context,
            confidence_context=confidence_context,
            target_class=None,
            max_context=ATHENA_CONTEXT_SIZE,
            selection="knn",
            query=query,
        )
        confidence_grid = None
        if confidence_quantiles is not None:
            selected_confidences = sampler_context.selected_confidences_
            selected_labels = sampler_context.selected_labels_
            if selected_confidences is None or selected_labels is None:
                raise RuntimeError(
                    "confidence-conditioned context diagnostics are unavailable"
                )
            confidence_grid = empirical_confidence_grid(
                selected_confidences,
                selected_labels,
                int(target_class),
                confidence_quantiles,
            )

        joint_scorer = None
        if cf_mode == "data_plausible":
            if joint_sampler_context is None or joint_sampler is None:
                raise RuntimeError("TabICL joint scorer is unavailable")
            joint_sampler_context.set_context(
                X_sampler_train,
                y_context=y_context,
                confidence_context=None,
                target_class=None,
                max_context=ATHENA_CONTEXT_SIZE,
                selection="knn",
                query=query,
            )
            joint_scorer = TabICLJointScorer(
                sampler=joint_sampler,
                target_class=int(target_class),
                n_permutations=tabicl_joint_permutations,
            )

        category_distribution = None
        if categorical_codec is not None and grouped_actionable:
            category_distribution = _build_category_distribution(
                sampler_context=sampler_context,
                categorical_codec=categorical_codec,
                target_class=int(target_class),
                confidence_grid=confidence_grid,
            )

        estimator_params = getattr(sampler_context, "estimator_params", {})
        return TabICLGeneratorPointBackend(
            sampler=proposal_sampler,
            candidate_confidences=confidence_grid,
            category_distribution=category_distribution,
            joint_scorer=joint_scorer,
            metadata={
                "categorical_confidence_batching": True,
                "conditional_estimator_cache": True,
                "tabicl_kv_cache": bool(estimator_params.get("kv_cache", False)),
            },
        )

    return point_backend_factory


def _legacy_info_from_result(
    result,
    *,
    bundle: Any,
    y_pred: np.ndarray,
    actionable_idx: Sequence[int],
    immutable_idx: Sequence[int],
    disc_model: Any,
    n_estimators: int,
) -> dict[str, Any]:
    diagnostics = result.diagnostics
    diversity = diagnostics.diversity_config
    return {
        "bundle": bundle,
        "y_pred": np.asarray(y_pred, dtype=int),
        "y_target": result.targets.copy(),
        "actionable_idx": list(actionable_idx),
        "immutable_idx": list(immutable_idx),
        "disc_model": disc_model,
        "tau": diagnostics.tau,
        "temperature": diagnostics.temperature,
        "candidate_quantiles": diagnostics.candidate_quantiles,
        "confidence_quantiles": diagnostics.confidence_quantiles,
        "cf_mode": diagnostics.cf_mode,
        "plausibility_backend": diagnostics.plausibility_backend,
        "max_validity_steps": diagnostics.max_validity_steps,
        "allow_revisits": diagnostics.allow_revisits,
        "joint_shortlist_size": diagnostics.joint_shortlist_size,
        "max_extra_actions": diagnostics.max_extra_actions,
        "min_joint_log_gain": diagnostics.min_joint_log_gain,
        "n_counterfactuals": diagnostics.n_counterfactuals,
        "diversity_beam_width": diversity.beam_width,
        "diversity_candidate_pool_size": diversity.candidate_pool_size,
        "diversity_max_extra_actions": diversity.max_extra_actions,
        "diversity_max_gower_ratio": diversity.max_gower_ratio,
        "diversity_max_gower_increase": diversity.max_gower_increase,
        "diversity_candidate_generation": "bounded_beam",
        "diversity_selector": "exact_fixed_size_dpp_map",
        "categorical_proposal_count": diagnostics.categorical_proposal_count,
        "categorical_confidence_batching": diagnostics.categorical_confidence_batching,
        "conditional_estimator_cache": diagnostics.conditional_estimator_cache,
        "tabicl_kv_cache": diagnostics.tabicl_kv_cache,
        "test_selection": getattr(bundle, "test_selection", "first"),
        "split_variant": bundle.split_variant,
        "preprocessing_variant": bundle.preprocessing_variant,
        "n_dropped_rows": bundle.n_dropped_rows,
        "n_estimators": n_estimators,
        "runtime_s": diagnostics.runtime_s,
        "X_sparse": result.sparse_counterfactuals.copy(),
        "X_cf_sets": result.counterfactual_sets.copy(),
        "diverse_available_count_per_point": diagnostics.diverse_available_count_per_point.copy(),
        "diverse_candidate_pool_count_per_point": diagnostics.diverse_candidate_pool_count_per_point.copy(),
        "diverse_search_depth_per_point": diagnostics.diverse_search_depth_per_point.copy(),
        "diverse_histories_per_point": [
            [list(step) if isinstance(step, tuple) else step for step in histories]
            for histories in diagnostics.diverse_histories_per_point
        ],
        "point_runtime_s": diagnostics.point_runtime_s.copy(),
        "joint_scoring_runtime_s_per_point": diagnostics.joint_scoring_runtime_s_per_point.copy(),
        "changed_per_point": [list(changed) for changed in diagnostics.changed_per_point],
        "flipped_per_point": list(diagnostics.flipped_per_point),
        "steps_per_point": list(diagnostics.steps_per_point),
        "history_per_point": [list(history) for history in diagnostics.history_per_point],
        "attempt_history_per_point": [
            list(history) for history in diagnostics.attempt_history_per_point
        ],
        "validity_steps_per_point": list(diagnostics.validity_steps_per_point),
        "initial_valid_step_per_point": list(diagnostics.initial_valid_step_per_point),
        "refinement_steps_per_point": list(diagnostics.refinement_steps_per_point),
        "accepted_refinement_count_per_point": list(
            diagnostics.accepted_refinement_count_per_point
        ),
        "initial_sparse_action_count_per_point": diagnostics.initial_sparse_action_count_per_point.copy(),
        "final_action_count_per_point": diagnostics.final_action_count_per_point.copy(),
        "initial_tabicl_joint_log_density_per_point": diagnostics.initial_tabicl_joint_log_density_per_point.copy(),
        "final_tabicl_joint_log_density_per_point": diagnostics.final_tabicl_joint_log_density_per_point.copy(),
        "tabicl_joint_log_density_gain_per_point": diagnostics.tabicl_joint_log_density_gain_per_point.copy(),
        "joint_scoring_batch_count_per_point": diagnostics.joint_scoring_batch_count_per_point.copy(),
        "joint_rows_scored_per_point": diagnostics.joint_rows_scored_per_point.copy(),
        "extra_actions_per_point": diagnostics.extra_actions_per_point.copy(),
        "refinement_stopping_reason_per_point": list(
            diagnostics.refinement_stopping_reason_per_point
        ),
        "target_probability_per_point": diagnostics.target_probability_per_point.copy(),
    }


def generate_tabicl_counterfactuals(
    dataset_name: str,
    *,
    tau: float = TAU,
    temperature: float = DEFAULT_TEMPERATURE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_test: int | None = None,
    candidate_quantiles: tuple[float, ...] | None = None,
    confidence_quantiles: tuple[float, ...] | None = None,
    cf_mode: str = "sparse",
    tabicl_joint_permutations: int = 1,
    max_validity_steps: int | None = None,
    allow_revisits: bool = True,
    joint_shortlist_size: int = 16,
    max_extra_actions: int = 1,
    min_joint_log_gain: float = 0.0,
    n_counterfactuals: int = 1,
    diversity_beam_width: int = 8,
    diversity_candidate_pool_size: int = 16,
    diversity_max_extra_actions: int = 2,
    diversity_max_gower_ratio: float = 1.5,
    diversity_max_gower_increase: float = 0.02,
    validation_fraction: float = 0.0,
    test_selection: str = "first",
    drop_heloc_all_minus9: bool = False,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compatibility adapter that loads datasets and delegates to the stable API."""
    from experiments.zeroshot_cf.data import (
        get_grouped_categorical_action_space,
        get_one_hot_groups,
        load_dataset,
    )
    from experiments.zeroshot_cf.discriminator import train_discriminator
    from experiments.zeroshot_cf.diverse_search import DiverseBeamSearchConfig

    candidate_quantiles = (
        None
        if candidate_quantiles is None
        else tuple(float(value) for value in candidate_quantiles)
    )
    confidence_quantiles = (
        None
        if confidence_quantiles is None
        else tuple(float(value) for value in confidence_quantiles)
    )
    if cf_mode not in CF_MODES:
        raise ValueError(f"cf_mode must be one of {CF_MODES}, got {cf_mode!r}")

    limit = _resolve_max_test(dataset_name, max_test)
    bundle = load_dataset(
        dataset_name,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
        validation_fraction=validation_fraction,
    )
    bundle.test_selection = test_selection
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test, y_test = select_test_rows(
        bundle.X_test,
        bundle.y_test,
        limit,
        test_selection,
    )
    numerical_actionable_idx, grouped_actionable, immutable_idx = (
        get_grouped_categorical_action_space(bundle)
    )
    all_one_hot_groups = get_one_hot_groups(bundle)
    categorical_codec = (
        None
        if not all_one_hot_groups
        else GroupedCategoricalCodec.from_matrix(X_train, all_one_hot_groups)
    )
    actionable_idx = list(numerical_actionable_idx)
    for group in grouped_actionable:
        actionable_idx.extend(group.columns)
    effective_max_validity_steps = (
        len(numerical_actionable_idx) + len(grouped_actionable)
        if max_validity_steps is None
        else max_validity_steps
    )

    discriminator_cache_tag = (
        f"{dataset_name}_drop_all_minus9"
        if bundle.preprocessing_variant == "drop_heloc_all_minus9"
        else dataset_name
    )
    if bundle.X_val is not None:
        discriminator_cache_tag = f"{discriminator_cache_tag}_{bundle.split_variant}"
    X_disc_eval = bundle.X_val if bundle.X_val is not None else X_test
    y_disc_eval = bundle.y_val if bundle.y_val is not None else y_test
    disc_model = train_discriminator(
        X_train,
        y_train,
        X_disc_eval,
        y_disc_eval,
        discriminator_cache_tag,
    )
    y_pred = np.asarray(disc_model.predict(X_test), dtype=int)
    y_target = 1 - y_pred
    y_context = np.asarray(disc_model.predict(X_train), dtype=int)
    context_probabilities = (
        np.asarray(disc_model.predict_proba(X_train))
        if confidence_quantiles is not None
        else None
    )
    feature_domains = infer_feature_domains(X_train)

    print(f"\n=== Experiment 8 (TabICL): {dataset_name.upper()} ===")
    print(
        f"  selector=prob_ascent, context={ATHENA_CONTEXT_STRATEGY}"
        f"@{ATHENA_CONTEXT_SIZE}, labels=disc, "
        f"point_estimate={DEFAULT_POINT_ESTIMATE}, project_to_domain=True, "
        f"candidate_quantiles={candidate_quantiles}, "
        f"confidence_quantiles={confidence_quantiles}, cf_mode={cf_mode}, "
        f"joint_shortlist_size={joint_shortlist_size}, "
        f"max_extra_actions={max_extra_actions}, "
        f"min_joint_log_gain={min_joint_log_gain}, "
        f"max_validity_steps={effective_max_validity_steps}, "
        f"allow_revisits={allow_revisits}, "
        f"n_counterfactuals={n_counterfactuals}, "
        f"diversity_beam_width={diversity_beam_width}, "
        f"diversity_candidate_pool_size={diversity_candidate_pool_size}, "
        f"split={bundle.split_variant}, test_selection={test_selection}, "
        f"preprocessing={bundle.preprocessing_variant}, "
        f"n_dropped_rows={bundle.n_dropped_rows}, "
        f"temperature={temperature}, n_estimators={n_estimators}, "
        f"n_test={len(X_test)}"
    )
    print(
        f"  Features: {X_train.shape[1]} total, "
        f"{len(numerical_actionable_idx)} scalar actionable, "
        f"{len(grouped_actionable)} grouped categorical actionable, "
        f"{len(immutable_idx)} immutable"
    )

    config = TabICLGeneratorConfig(
        tau=tau,
        temperature=temperature,
        candidate_quantiles=candidate_quantiles,
        confidence_quantiles=confidence_quantiles,
        cf_mode=cf_mode,
        tabicl_joint_permutations=tabicl_joint_permutations,
        max_validity_steps=max_validity_steps,
        allow_revisits=allow_revisits,
        joint_shortlist_size=joint_shortlist_size,
        max_extra_actions=max_extra_actions,
        min_joint_log_gain=min_joint_log_gain,
        diversity_config=DiverseBeamSearchConfig(
            n_counterfactuals=n_counterfactuals,
            beam_width=diversity_beam_width,
            candidate_pool_size=diversity_candidate_pool_size,
            max_extra_actions=diversity_max_extra_actions,
            max_gower_ratio=diversity_max_gower_ratio,
            max_gower_increase=diversity_max_gower_increase,
        ),
        categorical_proposal_count=DEFAULT_CATEGORICAL_PROPOSAL_COUNT,
    )
    point_backend_factory = _build_point_backend_factory(
        X_train,
        y_context,
        grouped_actionable=grouped_actionable,
        categorical_codec=categorical_codec,
        context_probabilities=context_probabilities,
        confidence_quantiles=confidence_quantiles,
        cf_mode=cf_mode,
        tabicl_joint_permutations=tabicl_joint_permutations,
        n_estimators=n_estimators,
        temperature=temperature,
        cache_dir=cache_dir,
    )
    result = generate_counterfactual_batch(
        TabICLGeneratorInputs(
            factuals=X_test,
            targets=y_target,
            numerical_columns=tuple(numerical_actionable_idx),
            categorical_groups=tuple(grouped_actionable),
            immutable_idx=tuple(immutable_idx),
            feature_domains=feature_domains,
        ),
        discriminator=disc_model,
        config=config,
        point_backend_factory=point_backend_factory,
    )
    info = _legacy_info_from_result(
        result,
        bundle=bundle,
        y_pred=y_pred,
        actionable_idx=actionable_idx,
        immutable_idx=immutable_idx,
        disc_model=disc_model,
        n_estimators=n_estimators,
    )
    return X_test, y_test, result.counterfactuals, info


def run_and_report(
    dataset_name: str,
    **kwargs: Any,
) -> dict[str, float]:
    """Run one dataset, evaluate it, and write one backend-comparison row."""
    X_test, y_test, X_cf, info = generate_tabicl_counterfactuals(dataset_name, **kwargs)
    metrics = evaluate_counterfactuals(
        dataset_name,
        X_test,
        y_test,
        X_cf,
        info,
        results_dir=RESULTS_DIR,
        output_prefix="exp8_tabicl",
        write_csv=False,
    )
    metrics.update(evaluate_diverse_sets(X_test, info))
    initial_action_counts = np.asarray(
        info["initial_sparse_action_count_per_point"], dtype=float
    )
    reached_validity = initial_action_counts >= 0
    initial_action_count_mean = (
        float(initial_action_counts[reached_validity].mean())
        if np.any(reached_validity)
        else float("nan")
    )
    row: dict[str, Any] = {
        "dataset": dataset_name,
        "backend": "tabicl_v2",
        "selector": "prob_ascent",
        "valid_candidate_objective": "grouped_gower",
        "context_strategy": ATHENA_CONTEXT_STRATEGY,
        "context_size": ATHENA_CONTEXT_SIZE,
        "context_labels": "disc",
        "candidate_mode": "batched",
        "context_update": "replace",
        "point_estimate": DEFAULT_POINT_ESTIMATE,
        "project_to_domain": True,
        "candidate_quantiles": info["candidate_quantiles"],
        "confidence_quantiles": info["confidence_quantiles"],
        "cf_mode": info["cf_mode"],
        "plausibility_backend": info["plausibility_backend"],
        "max_validity_steps": info["max_validity_steps"],
        "allow_revisits": info["allow_revisits"],
        "joint_shortlist_size": info["joint_shortlist_size"],
        "max_extra_actions": info["max_extra_actions"],
        "min_joint_log_gain": info["min_joint_log_gain"],
        "n_counterfactuals": info["n_counterfactuals"],
        "diversity_beam_width": info["diversity_beam_width"],
        "diversity_candidate_pool_size": info["diversity_candidate_pool_size"],
        "diversity_max_extra_actions": info["diversity_max_extra_actions"],
        "diversity_max_gower_ratio": info["diversity_max_gower_ratio"],
        "diversity_max_gower_increase": info["diversity_max_gower_increase"],
        "joint_scoring_batch_count_mean": float(
            np.mean(info["joint_scoring_batch_count_per_point"])
        ),
        "joint_rows_scored_mean": float(np.mean(info["joint_rows_scored_per_point"])),
        "accepted_refinement_count_mean": float(
            np.mean(info["accepted_refinement_count_per_point"])
        ),
        "initial_sparse_action_count_mean": initial_action_count_mean,
        "final_action_count_mean": float(np.mean(info["final_action_count_per_point"])),
        "extra_actions_mean": float(np.mean(info["extra_actions_per_point"])),
        "categorical_proposal_count": info["categorical_proposal_count"],
        "categorical_confidence_batching": info["categorical_confidence_batching"],
        "conditional_estimator_cache": info["conditional_estimator_cache"],
        "tabicl_kv_cache": info["tabicl_kv_cache"],
        "split_variant": info["split_variant"],
        "test_selection": info["test_selection"],
        "n_estimators": info["n_estimators"],
        "temperature": info["temperature"],
        "n_test": len(X_test),
        "runtime_s": round(float(info["runtime_s"]), 2),
        **metrics,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"exp8_tabicl_{dataset_name}_metrics.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"\n  Wrote {output}")
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TabICL greedy counterfactuals with a fixed kNN context"
    )
    parser.add_argument(
        "--dataset",
        choices=["moons", "heloc", "german_credit", "all"],
        default="moons",
    )
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Default: moons=100, heloc=50; use -1 for the full test split.",
    )
    parser.add_argument(
        "--candidate-quantiles",
        type=float,
        nargs="+",
        default=None,
        metavar="Q",
        help=(
            "Generate deterministic conditional proposals per feature. Prefer "
            "a central grid, e.g. --candidate-quantiles 0.1 0.3 0.5 0.7 0.9."
        ),
    )
    parser.add_argument(
        "--confidence-quantiles",
        type=float,
        nargs="+",
        default=None,
        metavar="Q",
        help=(
            "Quantile levels of the selected context's target-class confidence "
            "distribution. The resulting empirical confidence values are appended "
            "to TabICL queries; requires --candidate-quantiles."
        ),
    )
    parser.add_argument(
        "--cf-mode",
        choices=["sparse", "data-plausible"],
        default="sparse",
        help=(
            "Sparse stops after selecting the lowest grouped-Gower valid "
            "proposal; data-plausible then performs one bounded TabICL "
            "complete-row reranking batch."
        ),
    )
    parser.add_argument("--tabicl-joint-permutations", type=int, default=1)
    parser.add_argument(
        "--max-validity-steps",
        type=int,
        default=None,
        help=(
            "Maximum committed probability-ascent actions before validity. "
            "Defaults to the number of actionable feature units."
        ),
    )
    parser.add_argument(
        "--allow-revisits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow an action unit to be proposed again after another edit.",
    )
    parser.add_argument(
        "--joint-shortlist-size",
        type=int,
        default=16,
        help="Maximum valid alternatives in the one-shot whole-row batch.",
    )
    parser.add_argument(
        "--max-extra-actions",
        type=int,
        default=1,
        help="Maximum final action-count increase over the sparse CFE.",
    )
    parser.add_argument(
        "--min-joint-log-gain",
        type=float,
        default=0.0,
        help="Minimum raw whole-row log-density gain over the sparse CFE.",
    )
    parser.add_argument(
        "--n-counterfactuals",
        type=int,
        default=1,
        help=(
            "Number of valid, unique counterfactuals requested per factual. "
            "Values above one enable diverse beam search."
        ),
    )
    parser.add_argument(
        "--diversity-beam-width",
        type=int,
        default=8,
        help="Number of partial counterfactual paths retained per search step.",
    )
    parser.add_argument(
        "--diversity-candidate-pool-size",
        type=int,
        default=16,
        help="Valid candidates retained before exact DPP selection.",
    )
    parser.add_argument(
        "--diversity-max-extra-actions",
        type=int,
        default=2,
        help="Maximum action-count increase over the closest valid candidate.",
    )
    parser.add_argument(
        "--diversity-max-gower-ratio",
        type=float,
        default=1.5,
        help=(
            "Multiplicative grouped-Gower quality limit relative to the "
            "closest valid candidate."
        ),
    )
    parser.add_argument(
        "--diversity-max-gower-increase",
        type=float,
        default=0.02,
        help="Additive grouped-Gower allowance beyond the ratio-based limit.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of the provisional 80%% train partition reserved for "
            "validation; 0.2 gives a fixed 64/16/20 split."
        ),
    )
    parser.add_argument(
        "--test-selection",
        choices=["first", "stratified"],
        default="first",
        help="How --max-test selects held-out factuals (default: first).",
    )
    parser.add_argument(
        "--drop-heloc-all-minus9",
        action="store_true",
        help=(
            "Before splitting HELOC, remove rows whose 23 predictors are all "
            "the -9 no-bureau-record sentinel."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    datasets = (
        ["moons", "heloc", "german_credit"] if args.dataset == "all" else [args.dataset]
    )
    for dataset_name in datasets:
        run_and_report(
            dataset_name,
            tau=args.tau,
            temperature=args.temperature,
            n_estimators=args.n_estimators,
            max_test=args.max_test,
            candidate_quantiles=(
                None
                if args.candidate_quantiles is None
                else tuple(args.candidate_quantiles)
            ),
            confidence_quantiles=(
                None
                if args.confidence_quantiles is None
                else tuple(args.confidence_quantiles)
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
            test_selection=args.test_selection,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
