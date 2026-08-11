#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: T201

"""Small real-model checks for the two TabICL speed optimizations.

This is deliberately not an experiment sweep.  It runs three configurations on
the same two or three factuals:

* batched candidate evaluation + direct context replacement (production path),
* sequential candidate evaluation + direct context replacement,
* batched candidate evaluation + upstream ``fit()`` on every context.

The first pair isolates candidate batching.  The second pair isolates direct
context replacement; its comparison starts at factual index 1 because every
sampler necessarily calls ``fit()`` for its first context.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.exp8_tabicl_cf import (
    DEFAULT_N_ESTIMATORS,
    DEFAULT_TEMPERATURE,
    generate_tabicl_counterfactuals,
)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results" / "tabicl_diagnostics"
RUN_CONFIGS = (
    ("batched_replace", "batched", "replace"),
    ("sequential_replace", "sequential", "replace"),
    ("batched_refit", "batched", "refit"),
)


def _as_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _as_builtin(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_as_builtin(item) for item in value]
    if isinstance(value, list):
        return [_as_builtin(item) for item in value]
    return value


def _run(
    dataset: str,
    *,
    candidate_mode: str,
    context_update: str,
    max_test: int,
    n_estimators: int,
    temperature: float,
    cache_dir: Path | None,
) -> dict[str, Any]:
    X_test, y_test, X_cf, info = generate_tabicl_counterfactuals(
        dataset,
        max_test=max_test,
        n_estimators=n_estimators,
        temperature=temperature,
        context_labels="disc",
        candidate_mode=candidate_mode,
        context_update=context_update,
        cache_dir=cache_dir,
    )
    y_target = np.asarray(info["y_target"], dtype=int)
    target_probability = info["disc_model"].predict_proba(X_cf)[
        np.arange(len(X_cf)), y_target
    ]
    return {
        "X_test": X_test,
        "y_test": y_test,
        "X_cf": X_cf,
        "y_target": y_target,
        "target_probability": target_probability,
        "changed": info["changed_per_point"],
        "flipped": info["flipped_per_point"],
        "steps": info["steps_per_point"],
        "history": info["history_per_point"],
        "runtime_s": float(info["runtime_s"]),
    }


def compare_runs(
    comparison: str,
    reference_name: str,
    reference: dict[str, Any],
    candidate_name: str,
    candidate: dict[str, Any],
    *,
    start_index: int = 0,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Compare paired real-model outputs, optionally skipping the first row."""
    ref_cf = np.asarray(reference["X_cf"])[start_index:]
    cand_cf = np.asarray(candidate["X_cf"])[start_index:]
    ref_prob = np.asarray(reference["target_probability"])[start_index:]
    cand_prob = np.asarray(candidate["target_probability"])[start_index:]

    cf_close = np.isclose(ref_cf, cand_cf, rtol=0.0, atol=atol, equal_nan=True)
    prob_close = np.isclose(
        ref_prob, cand_prob, rtol=0.0, atol=atol, equal_nan=True
    )
    finite_cf = np.abs(ref_cf - cand_cf)
    finite_prob = np.abs(ref_prob - cand_prob)

    return {
        "comparison": comparison,
        "reference": reference_name,
        "candidate": candidate_name,
        "start_index": start_index,
        "n_points": len(ref_cf),
        "atol": atol,
        "cf_allclose": bool(np.all(cf_close)),
        "cf_differing_cells": int(np.size(cf_close) - np.count_nonzero(cf_close)),
        "cf_max_abs_diff": float(np.nanmax(finite_cf)) if finite_cf.size else 0.0,
        "target_probability_allclose": bool(np.all(prob_close)),
        "target_probability_max_abs_diff": (
            float(np.nanmax(finite_prob)) if finite_prob.size else 0.0
        ),
        "changed_equal": (
            reference["changed"][start_index:] == candidate["changed"][start_index:]
        ),
        "flipped_equal": (
            reference["flipped"][start_index:] == candidate["flipped"][start_index:]
        ),
        "steps_equal": (
            reference["steps"][start_index:] == candidate["steps"][start_index:]
        ),
        "reference_runtime_s": round(reference["runtime_s"], 6),
        "candidate_runtime_s": round(candidate["runtime_s"], 6),
    }


def main() -> None:
    """Run the three paired diagnostics and persist detailed comparisons."""
    parser = argparse.ArgumentParser(
        description="Check real TabICL batching and context-update equivalence"
    )
    parser.add_argument("--dataset", choices=["moons", "heloc"], default="heloc")
    parser.add_argument(
        "--max-test",
        type=int,
        default=2,
        help="Use 2 to exercise one context update without an extensive run.",
    )
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--tabicl-cache-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    if args.max_test < 2:
        parser.error("--max-test must be at least 2 to exercise a context update")

    runs: dict[str, dict[str, Any]] = {}
    for name, candidate_mode, context_update in RUN_CONFIGS:
        print(
            f"\n######## diagnostic run: {name} "
            f"({candidate_mode=}, {context_update=}) ########"
        )
        runs[name] = _run(
            args.dataset,
            candidate_mode=candidate_mode,
            context_update=context_update,
            max_test=args.max_test,
            n_estimators=args.n_estimators,
            temperature=args.temperature,
            cache_dir=args.tabicl_cache_dir,
        )

    comparisons = [
        compare_runs(
            "candidate_batching",
            "batched_replace",
            runs["batched_replace"],
            "sequential_replace",
            runs["sequential_replace"],
            atol=args.atol,
        ),
        compare_runs(
            "context_update",
            "batched_replace",
            runs["batched_replace"],
            "batched_refit",
            runs["batched_refit"],
            start_index=1,
            atol=args.atol,
        ),
    ]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"exp8_tabicl_diagnostics_{args.dataset}"
    csv_path = args.results_dir / f"{stem}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    npz_path = args.results_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        X_test=runs["batched_replace"]["X_test"],
        y_test=runs["batched_replace"]["y_test"],
        y_target=runs["batched_replace"]["y_target"],
        **{f"X_cf_{name}": result["X_cf"] for name, result in runs.items()},
        **{
            f"target_probability_{name}": result["target_probability"]
            for name, result in runs.items()
        },
    )

    json_path = args.results_dir / f"{stem}.json"
    detail = {
        "dataset": args.dataset,
        "max_test": args.max_test,
        "n_estimators": args.n_estimators,
        "temperature": args.temperature,
        "comparisons": comparisons,
        "runs": {
            name: {
                "runtime_s": result["runtime_s"],
                "changed": result["changed"],
                "flipped": result["flipped"],
                "steps": result["steps"],
                "history": result["history"],
                "target_probability": result["target_probability"],
            }
            for name, result in runs.items()
        },
    }
    json_path.write_text(json.dumps(_as_builtin(detail), indent=2) + "\n")

    print("\nComparison verdicts:")
    for row in comparisons:
        passed = all(
            row[key]
            for key in (
                "cf_allclose",
                "target_probability_allclose",
                "changed_equal",
                "flipped_equal",
                "steps_equal",
            )
        )
        print(
            f"  {row['comparison']}: {'PASS' if passed else 'DIFFERENT'} "
            f"(max |CF diff|={row['cf_max_abs_diff']:.3g})"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
