"""Retained reporting helpers for the focused TabICL counterfactual suite."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from experiments.zeroshot_cf.data import get_grouped_categorical_action_space
from experiments.zeroshot_cf.diverse_search import (
    action_set_jaccard_distance,
    action_unit_signature,
)
from experiments.zeroshot_cf.metrics_harness import compute_metrics, print_metrics
from experiments.zeroshot_cf.mixed_distance import (
    action_unit_change_count,
    grouped_gower_distance,
)


def evaluate_counterfactuals(
    dataset_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_cf: np.ndarray,
    info: dict[str, Any],
    *,
    results_dir: Path,
    output_prefix: str,
    write_csv: bool = True,
) -> dict[str, float]:
    """Compute retained suite metrics without routing through legacy Exp4 code."""
    bundle = info["bundle"]
    immutable_idx = info["immutable_idx"]
    disc_model = info["disc_model"]
    y_target = info["y_target"]

    oob_mask = (X_cf < 0.0) | (X_cf > 1.0)
    frac_oob = float(oob_mask.any(axis=1).mean())
    print(
        f"\n  Out-of-[0,1] fraction of CFs: {frac_oob:.3f} "
        f"({int(oob_mask.any(axis=1).sum())}/{len(X_cf)} points)"
    )

    X_cf_clipped = np.clip(X_cf, 0.0, 1.0)
    if immutable_idx:
        columns = np.asarray(immutable_idx)
        max_dev = float(np.abs(X_cf[:, columns] - X_test[:, columns]).max())
        print(f"  Max immutable-column deviation: {max_dev:.2e}")
        assert max_dev < 1e-9, (
            f"Immutable columns drifted: max_dev={max_dev} — must be preserved exactly."
        )

    metrics = compute_metrics(
        disc_model=disc_model,
        X_cf=X_cf_clipped,
        X_test=X_test,
        X_train=bundle.X_train,
        y_test=y_test,
        y_target=y_target,
        immutable_idx=immutable_idx,
        X_cf_lof=X_cf,
    )
    metrics["frac_oob"] = frac_oob

    flipped = np.asarray(info["flipped_per_point"], dtype=bool)
    l0_all = np.array([len(changed) for changed in info["changed_per_point"]], dtype=float)
    steps_all = np.asarray(info["steps_per_point"], dtype=float)
    if flipped.any():
        l0_valid = l0_all[flipped]
        steps_valid = steps_all[flipped]
        metrics["l0_count_mean"] = float(l0_valid.mean())
        metrics["l0_count_median"] = float(np.median(l0_valid))
        metrics["l0_count_max"] = float(l0_valid.max())
        metrics["steps_mean"] = float(steps_valid.mean())
        metrics["steps_median"] = float(np.median(steps_valid))
        metrics["steps_max"] = float(steps_valid.max())
    else:
        for key in (
            "l0_count_mean",
            "l0_count_median",
            "l0_count_max",
            "steps_mean",
            "steps_median",
            "steps_max",
        ):
            metrics[key] = float("nan")
    metrics["failure_rate"] = float((~flipped).mean())
    metrics["n_actionable"] = int(len(info["actionable_idx"]))

    print_metrics(metrics, prefix=dataset_name)
    if write_csv:
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"{output_prefix}_{dataset_name}_metrics.csv"
        row = {"dataset": dataset_name, **metrics}
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        print(f"\n  Wrote {csv_path}")
    return metrics


def evaluate_diverse_sets(
    X_test: np.ndarray,
    info: dict[str, Any],
) -> dict[str, float]:
    """Evaluate validity, coverage, and pairwise diversity of returned sets."""
    return evaluate_diverse_counterfactual_sets(
        X_test=X_test,
        bundle=info["bundle"],
        disc_model=info["disc_model"],
        y_target=np.asarray(info["y_target"], dtype=int),
        X_cf_sets=np.asarray(info["X_cf_sets"]),
        counts=np.asarray(info["diverse_available_count_per_point"], dtype=int),
        tau=float(info["tau"]),
    )


def evaluate_diverse_counterfactual_sets(
    *,
    X_test: np.ndarray,
    bundle: Any,
    disc_model: Any,
    y_target: np.ndarray,
    X_cf_sets: np.ndarray,
    counts: np.ndarray,
    tau: float,
) -> dict[str, float]:
    """Evaluate validity, coverage, and pairwise diversity of returned sets."""
    requested = int(X_cf_sets.shape[1]) if X_cf_sets.ndim >= 2 else 0
    metrics = {
        "diverse_coverage_at_k": float(np.mean(counts >= requested)),
        "diverse_returned_count_mean": float(np.mean(counts)),
    }
    if requested == 1:
        return metrics

    numerical, groups, _ = get_grouped_categorical_action_space(bundle)
    classifier = disc_model
    targets = np.asarray(y_target, dtype=int)
    sets = np.asarray(X_cf_sets)
    validity: list[bool] = []
    action_distances: list[float] = []
    value_distances: list[float] = []
    factual_distances: list[float] = []
    action_counts: list[int] = []
    for index, count in enumerate(counts):
        rows = sets[index, :count]
        if not len(rows):
            continue
        probabilities = np.asarray(classifier.predict_proba(rows))
        predictions = np.asarray(classifier.predict(rows))
        validity.extend(
            (
                (predictions == targets[index])
                & (probabilities[:, targets[index]] >= tau)
            ).tolist()
        )
        factual_distances.extend(
            grouped_gower_distance(
                rows,
                X_test[index],
                numerical,
                groups,
            ).tolist()
        )
        action_counts.extend(
            action_unit_change_count(
                rows,
                X_test[index],
                numerical,
                groups,
            ).tolist()
        )
        signatures = [
            action_unit_signature(row, X_test[index], numerical, groups) for row in rows
        ]
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                action_distances.append(
                    action_set_jaccard_distance(signatures[left], signatures[right])
                )
                value_distances.append(
                    float(
                        grouped_gower_distance(
                            rows[left],
                            rows[right],
                            numerical,
                            groups,
                        )[0]
                    )
                )

    metrics.update(
        {
            "diverse_returned_validity": (
                float(np.mean(validity)) if validity else float("nan")
            ),
            "diverse_action_jaccard_mean": (
                float(np.mean(action_distances)) if action_distances else float("nan")
            ),
            "diverse_action_jaccard_min": (
                float(np.min(action_distances)) if action_distances else float("nan")
            ),
            "diverse_pairwise_gower_mean": (
                float(np.mean(value_distances)) if value_distances else float("nan")
            ),
            "diverse_pairwise_gower_min": (
                float(np.min(value_distances)) if value_distances else float("nan")
            ),
            "diverse_factual_gower_mean": (
                float(np.mean(factual_distances)) if factual_distances else float("nan")
            ),
            "diverse_action_count_mean": (
                float(np.mean(action_counts)) if action_counts else float("nan")
            ),
        }
    )
    return metrics
