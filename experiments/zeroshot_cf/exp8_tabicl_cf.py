#  Copyright (c) Prior Labs GmbH 2026.

"""Experiment 8: greedy counterfactuals with the TabICLv2 backend.

This runner intentionally does not repeat the context ablation. It fixes the
Athena winner for all comparison datasets:

* selector: ``prob_ascent``
* context: 512 nearest neighbours from both classes (``knn_both@512``)
* labels: predictions of the discriminator being explained (Athena Exp7)
* configurable greedy rounds; on mixed data, numerical proposals and atomic
  categorical swaps compete globally at every step

Numerical candidate interventions for each greedy step are expanded into one
matrix and imputed in one TabICL call. They are then scored together with every
legal whole-category swap. The overall counterfactual search remains iterative.
Context remains per-factual because the winning kNN context is query-specific.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.exp4_greedy_cf import (
    _DATASET_PARAMS,
    TAU,
    evaluate_and_report,
)

RESULTS_DIR = Path(__file__).parent / "results"
ATHENA_CONTEXT_SIZE = 512
ATHENA_CONTEXT_STRATEGY = "knn_both"
DEFAULT_TEMPERATURE = 1e-9  # deterministic point estimate / categorical mode
DEFAULT_N_ESTIMATORS = 4
DEFAULT_POINT_ESTIMATE = "mode"


def empirical_confidence_grid(
    confidences: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    quantile_levels: tuple[float, ...],
) -> tuple[float, ...]:
    """Derive query-confidence candidates from the selected target-class rows."""
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if levels.ndim != 1 or len(levels) == 0:
        raise ValueError("confidence quantile levels must be a non-empty sequence")
    if np.any((levels <= 0) | (levels >= 1)) or np.any(np.diff(levels) <= 0):
        raise ValueError(
            "confidence quantile levels must be strictly increasing inside (0, 1)"
        )
    scores = np.asarray(confidences, dtype=np.float64)
    context_labels = np.asarray(labels)
    target_scores = scores[context_labels == target_class]
    if len(target_scores) == 0:
        target_scores = scores
    values = np.quantile(target_scores, levels)
    return tuple(float(v) for v in np.unique(values))


def _resolve_max_test(dataset_name: str, max_test: int | None) -> int | None:
    if max_test is not None and max_test < 0:
        return None
    if max_test is not None:
        return max_test
    return _DATASET_PARAMS.get(dataset_name, {"max_test": 50})["max_test"]


def _select_test_rows(
    X_test: np.ndarray,
    y_test: np.ndarray,
    limit: int | None,
    selection: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic held-out evaluation subset."""
    if selection not in {"first", "stratified"}:
        raise ValueError("test_selection must be 'first' or 'stratified'")
    if limit is None or limit >= len(X_test):
        return X_test, y_test
    if limit <= 0:
        raise ValueError("max_test must be positive or -1 for the full test set")
    if selection == "first":
        return X_test[:limit], y_test[:limit]

    from sklearn.model_selection import train_test_split

    if limit < len(np.unique(y_test)):
        rng = np.random.default_rng(42)
        selected = np.sort(rng.choice(len(X_test), size=limit, replace=False))
        return X_test[selected], y_test[selected]

    selected, _ = train_test_split(
        np.arange(len(X_test)),
        train_size=limit,
        random_state=42,
        stratify=y_test,
    )
    selected.sort()
    return X_test[selected], y_test[selected]


def generate_tabicl_counterfactuals(
    dataset_name: str,
    *,
    tau: float = TAU,
    temperature: float = DEFAULT_TEMPERATURE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_test: int | None = None,
    context_labels: str = "disc",
    candidate_mode: str = "batched",
    context_update: str = "replace",
    point_estimate: str = DEFAULT_POINT_ESTIMATE,
    project_to_domain: bool = True,
    retain_best: bool = True,
    candidate_quantiles: tuple[float, ...] | None = None,
    confidence_quantiles: tuple[float, ...] | None = None,
    lof_first: bool = False,
    probability_slack: float = 0.0,
    max_rounds: int = 1,
    categorical_fallback: bool = False,
    validation_fraction: float = 0.0,
    test_selection: str = "first",
    drop_heloc_all_minus9: bool = False,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate TabICL counterfactuals under the fixed Athena configuration."""
    if context_labels not in {"disc", "data"}:
        raise ValueError("context_labels must be 'disc' or 'data'")
    if candidate_mode not in {"batched", "sequential"}:
        raise ValueError("candidate_mode must be 'batched' or 'sequential'")
    if context_update not in {"replace", "refit"}:
        raise ValueError("context_update must be 'replace' or 'refit'")
    if point_estimate not in {"median", "mode"}:
        raise ValueError("point_estimate must be 'median' or 'mode'")
    if candidate_quantiles is not None:
        candidate_quantiles = tuple(float(q) for q in candidate_quantiles)
        if candidate_mode != "batched":
            raise ValueError("candidate_quantiles require candidate_mode='batched'")
    if confidence_quantiles is not None:
        confidence_quantiles = tuple(float(q) for q in confidence_quantiles)
        if candidate_quantiles is None:
            raise ValueError("confidence_quantiles require candidate_quantiles")
    if lof_first and candidate_quantiles is None:
        raise ValueError("lof_first requires candidate_quantiles")
    if probability_slack < 0:
        raise ValueError("probability_slack must be non-negative")
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    if test_selection not in {"first", "stratified"}:
        raise ValueError("test_selection must be 'first' or 'stratified'")

    from experiments.zeroshot_cf.data import (
        get_actionable_immutable,
        get_grouped_categorical_action_space,
        get_one_hot_groups,
        load_dataset,
    )
    from experiments.zeroshot_cf.discriminator import train_discriminator
    from experiments.zeroshot_cf.greedy import (
        greedy_counterfactual,
        infer_feature_domains,
    )
    from experiments.zeroshot_cf.grouped_categorical import (
        CompactMixedSampler,
        GroupedCategoricalCodec,
        greedy_mixed_counterfactual,
    )
    from experiments.zeroshot_cf.tabicl_checkpoints import TABICL_DEVICE
    from experiments.zeroshot_cf.tabicl_sampler import (
        TabICLConditionalDensitySampler,
    )

    limit = _resolve_max_test(dataset_name, max_test)
    bundle = load_dataset(
        dataset_name,
        drop_heloc_all_minus9=drop_heloc_all_minus9,
        validation_fraction=validation_fraction,
    )
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test, y_test = _select_test_rows(
        bundle.X_test,
        bundle.y_test,
        limit,
        test_selection,
    )
    grouped_actionable = []
    all_one_hot_groups = []
    categorical_codec = None
    if categorical_fallback:
        (
            numerical_actionable_idx,
            grouped_actionable,
            immutable_idx,
        ) = get_grouped_categorical_action_space(bundle)
        all_one_hot_groups = get_one_hot_groups(bundle)
        if all_one_hot_groups:
            categorical_codec = GroupedCategoricalCodec.from_matrix(
                X_train,
                all_one_hot_groups,
            )
    else:
        numerical_actionable_idx, immutable_idx = get_actionable_immutable(
            dataset_name,
            bundle,
        )
    actionable_idx = list(numerical_actionable_idx)
    for group in grouped_actionable:
        actionable_idx.extend(group.columns)

    discriminator_cache_tag = (
        f"{dataset_name}_drop_all_minus9"
        if bundle.preprocessing_variant == "drop_heloc_all_minus9"
        else dataset_name
    )
    if bundle.X_val is not None:
        discriminator_cache_tag = (
            f"{discriminator_cache_tag}_{bundle.split_variant}"
        )
    X_disc_eval = bundle.X_val if bundle.X_val is not None else X_test
    y_disc_eval = bundle.y_val if bundle.y_val is not None else y_test
    disc_model = train_discriminator(
        X_train,
        y_train,
        X_disc_eval,
        y_disc_eval,
        discriminator_cache_tag,
    )
    y_pred = disc_model.predict(X_test)
    y_target = 1 - y_pred
    y_context = disc_model.predict(X_train) if context_labels == "disc" else y_train
    context_probabilities = (
        np.asarray(disc_model.predict_proba(X_train))
        if confidence_quantiles is not None
        else None
    )

    plausibility_model = None
    if lof_first:
        from sklearn.neighbors import LocalOutlierFactor

        plausibility_model = LocalOutlierFactor(n_neighbors=20, novelty=True)
        plausibility_model.fit(X_train)

    print(f"\n=== Experiment 8 (TabICL): {dataset_name.upper()} ===")
    print(
        f"  selector=prob_ascent, context={ATHENA_CONTEXT_STRATEGY}"
        f"@{ATHENA_CONTEXT_SIZE}, labels={context_labels}, "
        f"candidate_mode={candidate_mode}, context_update={context_update}, "
        f"point_estimate={point_estimate}, project_to_domain={project_to_domain}, "
        f"retain_best={retain_best}, candidate_quantiles={candidate_quantiles}, "
        f"confidence_quantiles={confidence_quantiles}, lof_first={lof_first}, "
        f"probability_slack={probability_slack}, "
        f"max_rounds={max_rounds}, "
        f"categorical_fallback={categorical_fallback}, "
        f"split={bundle.split_variant}, test_selection={test_selection}, "
        f"preprocessing={bundle.preprocessing_variant}, "
        f"n_dropped_rows={bundle.n_dropped_rows}, "
        f"temperature={temperature}, "
        f"n_estimators={n_estimators}, n_test={len(X_test)}"
    )
    print(
        f"  Features: {X_train.shape[1]} total, "
        f"{len(numerical_actionable_idx)} scalar actionable, "
        f"{len(grouped_actionable)} grouped categorical actionable, "
        f"{len(immutable_idx)} immutable"
    )

    sampler_context = TabICLConditionalDensitySampler(
        n_estimators=n_estimators,
        temperature=temperature,
        random_state=42,
        device=TABICL_DEVICE,
        cache_dir=cache_dir,
        context_update=context_update,
        numerical_point_estimate=point_estimate,
        categorical_features=(
            None
            if categorical_codec is None
            else categorical_codec.categorical_columns
        ),
    )
    if categorical_codec is None:
        X_sampler_train = X_train
        sampler = sampler_context
    else:
        X_sampler_train = categorical_codec.encode(X_train)
        sampler = CompactMixedSampler(sampler_context, categorical_codec)
    feature_domains = infer_feature_domains(X_train) if project_to_domain else None

    X_cf = X_test.copy()
    changed_per_point: list[list[int]] = [[] for _ in range(len(X_test))]
    flipped_per_point = [False] * len(X_test)
    steps_per_point = [0] * len(X_test)
    history_per_point: list[list[tuple]] = [[] for _ in range(len(X_test))]
    attempt_history_per_point: list[list[tuple]] = [
        [] for _ in range(len(X_test))
    ]
    selection_history_per_point: list[list[dict]] = [
        [] for _ in range(len(X_test))
    ]
    confidence_grid_per_point: list[tuple[float, ...] | None] = [
        None for _ in range(len(X_test))
    ]
    categorical_history_per_point: list[list[dict]] = [
        [] for _ in range(len(X_test))
    ]
    rounds_per_point = [0] * len(X_test)
    round_history_per_point: list[list[dict]] = [
        [] for _ in range(len(X_test))
    ]

    started = time.perf_counter()
    for i, (x, target) in enumerate(zip(X_test, y_target, strict=True)):
        # Athena winner: both-class pool, per-factual 512-row kNN context.
        sampler_query = (
            x if categorical_codec is None else categorical_codec.encode_row(x)
        )
        sampler_context.set_context(
            X_sampler_train,
            y_context=y_context,
            confidence_context=(
                None
                if context_probabilities is None
                else context_probabilities[:, int(target)]
            ),
            target_class=None,
            max_context=ATHENA_CONTEXT_SIZE,
            selection="knn",
            query=sampler_query,
        )
        confidence_grid = None
        if confidence_quantiles is not None:
            confidence_grid = empirical_confidence_grid(
                sampler_context.selected_confidences_,
                sampler_context.selected_labels_,
                int(target),
                confidence_quantiles,
            )
        target_class = int(target)
        point_confidence_grid = confidence_grid

        def numerical_pass(
            row: np.ndarray,
            rounds: int,
            *,
            require_improvement: bool = False,
        ) -> tuple[np.ndarray, list[int], dict[str, Any]]:
            return greedy_counterfactual(
                sampler,
                disc_model,
                row,
                target_class,
                numerical_actionable_idx,
                "prob_ascent",
                tau=tau,
                budget=len(numerical_actionable_idx),
                temperature=temperature,
                batch_candidates=candidate_mode == "batched",
                feature_domains=feature_domains,
                retain_best=retain_best,
                candidate_quantiles=candidate_quantiles,
                candidate_confidences=point_confidence_grid,
                plausibility_model=plausibility_model,
                validity_first=lof_first,
                probability_slack=probability_slack,
                max_rounds=rounds,
                require_improvement=require_improvement,
            )

        if categorical_codec is not None and grouped_actionable:

            def category_distribution(
                row: np.ndarray,
                group: Any,
            ) -> tuple[np.ndarray, np.ndarray]:
                encoded_row = categorical_codec.encode_row(row)
                encoded_col = categorical_codec.encoded_column_for_group(group)
                fixed_confidence = (
                    None
                    if point_confidence_grid is None
                    else point_confidence_grid[len(point_confidence_grid) // 2]
                )
                return sampler_context.categorical_distribution(
                    encoded_row.reshape(1, -1),
                    encoded_col,
                    fixed_target=target_class,
                    fixed_confidence=fixed_confidence,
                )

            x_cf, changed, greedy_info = greedy_mixed_counterfactual(
                sampler,
                disc_model,
                x,
                target_class,
                numerical_actionable_idx,
                grouped_actionable,
                candidate_quantiles=candidate_quantiles,
                candidate_confidences=point_confidence_grid,
                feature_domains=feature_domains,
                plausibility_model=plausibility_model,
                validity_first=lof_first,
                probability_slack=probability_slack,
                max_rounds=max_rounds,
                tau=tau,
                temperature=temperature,
                category_distribution=category_distribution,
            )
        else:
            x_cf, changed, greedy_info = numerical_pass(
                x,
                max_rounds,
            )
            greedy_info["categorical_history"] = []
            greedy_info["round_history"] = []
        X_cf[i] = x_cf
        changed_per_point[i] = changed
        flipped_per_point[i] = greedy_info["flipped"]
        steps_per_point[i] = greedy_info["steps"]
        history_per_point[i] = greedy_info["history"]
        attempt_history_per_point[i] = greedy_info["attempt_history"]
        selection_history_per_point[i] = greedy_info["selection_history"]
        confidence_grid_per_point[i] = confidence_grid
        categorical_history_per_point[i] = greedy_info["categorical_history"]
        rounds_per_point[i] = greedy_info["rounds"]
        round_history_per_point[i] = greedy_info["round_history"]

        if i == 0:
            first_s = time.perf_counter() - started
            print(
                f"  [timing] first point: {first_s:.2f}s "
                f"(~{first_s * len(X_test) / 60:.1f} min linear estimate)"
            )

    runtime_s = time.perf_counter() - started
    lof_per_point = (
        None
        if plausibility_model is None
        else -np.asarray(plausibility_model.score_samples(X_cf), dtype=np.float64)
    )
    target_probability_per_point = np.asarray(disc_model.predict_proba(X_cf))[
        np.arange(len(X_cf)), y_target.astype(int)
    ]
    info: dict[str, Any] = {
        "bundle": bundle,
        "y_pred": y_pred,
        "y_target": y_target,
        "actionable_idx": actionable_idx,
        "immutable_idx": immutable_idx,
        "disc_model": disc_model,
        "selector": "prob_ascent",
        "context_type": ATHENA_CONTEXT_STRATEGY,
        "context_labels": context_labels,
        "tau": tau,
        "budget": len(numerical_actionable_idx),
        "temperature": temperature,
        "n_permutations": 0,
        "max_context": ATHENA_CONTEXT_SIZE,
        "candidate_mode": candidate_mode,
        "context_update": context_update,
        "point_estimate": point_estimate,
        "project_to_domain": project_to_domain,
        "retain_best": retain_best,
        "candidate_quantiles": candidate_quantiles,
        "confidence_quantiles": confidence_quantiles,
        "lof_first": lof_first,
        "probability_slack": probability_slack,
        "max_rounds": max_rounds,
        "categorical_fallback": categorical_fallback,
        "grouped_actionable": [group.name for group in grouped_actionable],
        "validation_fraction": validation_fraction,
        "test_selection": test_selection,
        "split_variant": bundle.split_variant,
        "drop_heloc_all_minus9": drop_heloc_all_minus9,
        "preprocessing_variant": bundle.preprocessing_variant,
        "n_dropped_rows": bundle.n_dropped_rows,
        "n_estimators": n_estimators,
        "runtime_s": runtime_s,
        "changed_per_point": changed_per_point,
        "flipped_per_point": flipped_per_point,
        "steps_per_point": steps_per_point,
        "history_per_point": history_per_point,
        "attempt_history_per_point": attempt_history_per_point,
        "selection_history_per_point": selection_history_per_point,
        "confidence_grid_per_point": confidence_grid_per_point,
        "categorical_history_per_point": categorical_history_per_point,
        "rounds_per_point": rounds_per_point,
        "round_history_per_point": round_history_per_point,
        "lof_per_point": lof_per_point,
        "target_probability_per_point": target_probability_per_point,
    }
    return X_test, y_test, X_cf, info


def run_and_report(
    dataset_name: str,
    **kwargs: Any,
) -> dict[str, float]:
    """Run one dataset, evaluate it, and write one backend-comparison row."""
    X_test, y_test, X_cf, info = generate_tabicl_counterfactuals(dataset_name, **kwargs)
    metrics = evaluate_and_report(
        dataset_name,
        X_test,
        y_test,
        X_cf,
        info,
        write_csv=False,
    )

    row: dict[str, Any] = {
        "dataset": dataset_name,
        "backend": "tabicl_v2",
        "selector": "prob_ascent",
        "context_strategy": ATHENA_CONTEXT_STRATEGY,
        "context_size": ATHENA_CONTEXT_SIZE,
        "context_labels": info["context_labels"],
        "candidate_mode": info["candidate_mode"],
        "context_update": info["context_update"],
        "point_estimate": info["point_estimate"],
        "project_to_domain": info["project_to_domain"],
        "retain_best": info["retain_best"],
        "candidate_quantiles": info["candidate_quantiles"],
        "confidence_quantiles": info["confidence_quantiles"],
        "lof_first": info["lof_first"],
        "probability_slack": info["probability_slack"],
        "categorical_fallback": info["categorical_fallback"],
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
    if info["lof_per_point"] is not None:
        diagnostics = {
            "dataset": dataset_name,
            "preprocessing_variant": info["preprocessing_variant"],
            "split_variant": info["split_variant"],
            "n_dropped_rows": info["n_dropped_rows"],
            "lof_per_point": info["lof_per_point"].tolist(),
            "y_pred": info["y_pred"].tolist(),
            "y_target": info["y_target"].tolist(),
            "target_probability_per_point": info[
                "target_probability_per_point"
            ].tolist(),
            "changed_per_point": info["changed_per_point"],
            "flipped_per_point": info["flipped_per_point"],
            "steps_per_point": info["steps_per_point"],
            "history_per_point": info["history_per_point"],
            "attempt_history_per_point": info["attempt_history_per_point"],
            "selection_history_per_point": info["selection_history_per_point"],
            "confidence_grid_per_point": info["confidence_grid_per_point"],
            "categorical_history_per_point": info[
                "categorical_history_per_point"
            ],
            "X_test": X_test.tolist(),
            "X_cf": X_cf.tolist(),
        }
        diagnostics_output = RESULTS_DIR / f"exp8_tabicl_{dataset_name}_diagnostics.json"
        with diagnostics_output.open("w") as handle:
            json.dump(diagnostics, handle, indent=2)
        print(f"  Wrote {diagnostics_output}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TabICL greedy counterfactuals at Athena's winning context"
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
        "--context-labels",
        choices=["disc", "data"],
        default="disc",
        help="Athena Exp7 used discriminator labels; 'data' reproduces Exp6.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["batched", "sequential"],
        default="batched",
        help="Use sequential only for the small equivalence/runtime baseline.",
    )
    parser.add_argument(
        "--context-update",
        choices=["replace", "refit"],
        default="replace",
        help=(
            "'replace' updates TabICL's stored context without reloading weights; "
            "'refit' calls the upstream fit() method for every factual and is "
            "intended only as a small correctness baseline."
        ),
    )
    parser.add_argument(
        "--point-estimate",
        choices=["median", "mode"],
        default=DEFAULT_POINT_ESTIMATE,
        help="Numerical TabICL point estimate; mode aligns with TabPFN near-MAP.",
    )
    parser.add_argument(
        "--candidate-quantiles",
        type=float,
        nargs="+",
        default=None,
        metavar="Q",
        help=(
            "Score deterministic conditional quantiles per feature, e.g. "
            "--candidate-quantiles 0.05 0.2 0.5 0.8 0.95."
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
        "--lof-first",
        action="store_true",
        help=(
            "Among immediately valid candidates choose minimum LOF; before a "
            "flip, use LOF among candidates within --probability-slack of the "
            "best target probability."
        ),
    )
    parser.add_argument(
        "--probability-slack",
        type=float,
        default=0.0,
        help="Pre-flip probability window in which LOF decides (default: 0).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1,
        help=(
            "Greedy passes over actionable features. Values above 1 allow "
            "earlier features to be revisited after later edits (default: 1)."
        ),
    )
    parser.add_argument(
        "--categorical-fallback",
        action="store_true",
        help=(
            "After numerical search fails, use TabICL categorical conditionals "
            "and atomic one-hot group swaps (mixed datasets only)."
        ),
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
    parser.add_argument(
        "--no-domain-projection",
        action="store_true",
        help="Disable training-range/support projection (diagnostic only).",
    )
    parser.add_argument(
        "--no-retain-best",
        action="store_true",
        help="Return the final failed state instead of its best probability state.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    datasets = (
        ["moons", "heloc", "german_credit"]
        if args.dataset == "all"
        else [args.dataset]
    )
    for dataset_name in datasets:
        run_and_report(
            dataset_name,
            tau=args.tau,
            temperature=args.temperature,
            n_estimators=args.n_estimators,
            max_test=args.max_test,
            context_labels=args.context_labels,
            candidate_mode=args.candidate_mode,
            context_update=args.context_update,
            point_estimate=args.point_estimate,
            project_to_domain=not args.no_domain_projection,
            retain_best=not args.no_retain_best,
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
            lof_first=args.lof_first,
            probability_slack=args.probability_slack,
            max_rounds=args.max_rounds,
            categorical_fallback=args.categorical_fallback,
            validation_fraction=args.validation_fraction,
            test_selection=args.test_selection,
            drop_heloc_all_minus9=args.drop_heloc_all_minus9,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
