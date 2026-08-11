#  Copyright (c) Prior Labs GmbH 2026.

"""Experiment 8: greedy counterfactuals with the TabICLv2 backend.

This runner intentionally does not repeat the context ablation. It fixes the
Athena winner for all comparison datasets:

* selector: ``prob_ascent``
* context: 512 nearest neighbours from both classes (``knn_both@512``)
* labels: predictions of the discriminator being explained (Athena Exp7)
* one greedy pass, with a feature changed at most once

Candidate interventions for each greedy step are expanded into one matrix and
imputed in one TabICL call. Context remains per-factual because the winning kNN
context is query-specific.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_TEMPERATURE = 1e-9  # deterministic TabICL median / categorical mode
DEFAULT_N_ESTIMATORS = 4


def _resolve_max_test(dataset_name: str, max_test: int | None) -> int | None:
    if max_test is not None and max_test < 0:
        return None
    if max_test is not None:
        return max_test
    return _DATASET_PARAMS.get(dataset_name, {"max_test": 50})["max_test"]


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
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate TabICL counterfactuals under the fixed Athena configuration."""
    if context_labels not in {"disc", "data"}:
        raise ValueError("context_labels must be 'disc' or 'data'")
    if candidate_mode not in {"batched", "sequential"}:
        raise ValueError("candidate_mode must be 'batched' or 'sequential'")
    if context_update not in {"replace", "refit"}:
        raise ValueError("context_update must be 'replace' or 'refit'")

    from experiments.zeroshot_cf.data import (
        get_actionable_immutable,
        load_dataset,
    )
    from experiments.zeroshot_cf.discriminator import train_discriminator
    from experiments.zeroshot_cf.greedy import greedy_counterfactual
    from experiments.zeroshot_cf.tabicl_checkpoints import TABICL_DEVICE
    from experiments.zeroshot_cf.tabicl_sampler import (
        TabICLConditionalDensitySampler,
    )

    limit = _resolve_max_test(dataset_name, max_test)
    bundle = load_dataset(dataset_name)
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test, y_test = bundle.X_test[:limit], bundle.y_test[:limit]
    actionable_idx, immutable_idx = get_actionable_immutable(dataset_name, bundle)

    disc_model = train_discriminator(X_train, y_train, X_test, y_test, dataset_name)
    y_pred = disc_model.predict(X_test)
    y_target = 1 - y_pred
    y_context = disc_model.predict(X_train) if context_labels == "disc" else y_train

    print(f"\n=== Experiment 8 (TabICL): {dataset_name.upper()} ===")
    print(
        f"  selector=prob_ascent, context={ATHENA_CONTEXT_STRATEGY}"
        f"@{ATHENA_CONTEXT_SIZE}, labels={context_labels}, "
        f"candidate_mode={candidate_mode}, context_update={context_update}, "
        f"temperature={temperature}, "
        f"n_estimators={n_estimators}, n_test={len(X_test)}"
    )
    print(
        f"  Features: {X_train.shape[1]} total, "
        f"{len(actionable_idx)} actionable, {len(immutable_idx)} immutable"
    )

    sampler = TabICLConditionalDensitySampler(
        n_estimators=n_estimators,
        temperature=temperature,
        random_state=42,
        device=TABICL_DEVICE,
        cache_dir=cache_dir,
        context_update=context_update,
    )

    X_cf = X_test.copy()
    changed_per_point: list[list[int]] = [[] for _ in range(len(X_test))]
    flipped_per_point = [False] * len(X_test)
    steps_per_point = [0] * len(X_test)
    history_per_point: list[list[tuple]] = [[] for _ in range(len(X_test))]

    started = time.perf_counter()
    for i, (x, target) in enumerate(zip(X_test, y_target)):
        # Athena winner: both-class pool, per-factual 512-row kNN context.
        sampler.set_context(
            X_train,
            y_context=y_context,
            target_class=None,
            max_context=ATHENA_CONTEXT_SIZE,
            selection="knn",
            query=x,
        )
        x_cf, changed, greedy_info = greedy_counterfactual(
            sampler,
            disc_model,
            x,
            int(target),
            actionable_idx,
            "prob_ascent",
            tau=tau,
            budget=len(actionable_idx),
            temperature=temperature,
            batch_candidates=candidate_mode == "batched",
        )
        X_cf[i] = x_cf
        changed_per_point[i] = changed
        flipped_per_point[i] = greedy_info["flipped"]
        steps_per_point[i] = greedy_info["steps"]
        history_per_point[i] = greedy_info["history"]

        if i == 0:
            first_s = time.perf_counter() - started
            print(
                f"  [timing] first point: {first_s:.2f}s "
                f"(~{first_s * len(X_test) / 60:.1f} min linear estimate)"
            )

    runtime_s = time.perf_counter() - started
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
        "budget": len(actionable_idx),
        "temperature": temperature,
        "n_permutations": 0,
        "max_context": ATHENA_CONTEXT_SIZE,
        "candidate_mode": candidate_mode,
        "context_update": context_update,
        "n_estimators": n_estimators,
        "runtime_s": runtime_s,
        "changed_per_point": changed_per_point,
        "flipped_per_point": flipped_per_point,
        "steps_per_point": steps_per_point,
        "history_per_point": history_per_point,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TabICL greedy counterfactuals at Athena's winning context"
    )
    parser.add_argument(
        "--dataset",
        choices=["moons", "heloc", "all"],
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
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    datasets = ["moons", "heloc"] if args.dataset == "all" else [args.dataset]
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
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
