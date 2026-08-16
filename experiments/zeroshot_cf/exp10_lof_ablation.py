#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: T201

"""Paired 50-point HELOC ablation of LOF candidate selection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from experiments.zeroshot_cf.exp9_dicoflex_benchmark import run_dataset

VARIANTS = {
    "lof_first": True,
    "probability_first": False,
}
RESULTS_DIR = Path(__file__).parent / "results" / "athena" / "exp10_lof_ablation"


def run_variant(
    variant: str,
    *,
    max_test: int = 50,
    n_estimators: int = 1,
    max_validity_steps: int = 100,
    tabicl_cache_dir: Path | None = None,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """Generate one side of the paired HELOC comparison."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown LOF variant: {variant!r}")
    return run_dataset(
        "heloc",
        max_test=max_test,
        n_estimators=n_estimators,
        _legacy_lof_refinement=VARIANTS[variant],
        max_validity_steps=max_validity_steps,
        allow_revisits=True,
        tabicl_cache_dir=tabicl_cache_dir,
        results_dir=results_dir / variant,
    )


def _read_one_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one result row in {path}, found {len(rows)}")
    return rows[0]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty comparison")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def aggregate_results(results_dir: Path = RESULTS_DIR) -> dict:
    """Create aggregate and paired comparisons from the two completed runs."""
    metrics_rows = []
    point_tables = {}
    arrays = {}
    for variant in VARIANTS:
        variant_dir = results_dir / variant
        prefix = variant_dir / "exp9_tabicl_heloc"
        metric = _read_one_row(prefix.with_name(f"{prefix.name}_metrics.csv"))
        metric = {"variant": variant, **metric}
        metrics_rows.append(metric)
        point_tables[variant] = _read_rows(
            prefix.with_name(f"{prefix.name}_points.csv")
        )
        arrays[variant] = np.load(prefix.with_name(f"{prefix.name}_arrays.npz"))

    lof_arrays = arrays["lof_first"]
    probability_arrays = arrays["probability_first"]
    if not np.array_equal(lof_arrays["X_test"], probability_arrays["X_test"]):
        raise ValueError("The two variants did not evaluate identical factuals")
    if not np.array_equal(lof_arrays["y_target"], probability_arrays["y_target"]):
        raise ValueError("The two variants did not use identical targets")

    lof_points = point_tables["lof_first"]
    probability_points = point_tables["probability_first"]
    if len(lof_points) != len(probability_points):
        raise ValueError("The two variants produced different point counts")

    paired_rows = []
    for index, (lof_point, probability_point) in enumerate(
        zip(lof_points, probability_points, strict=True)
    ):
        lof_score = float(lof_point["lof_score"])
        probability_score = float(probability_point["lof_score"])
        cf_distance = float(
            np.linalg.norm(
                lof_arrays["X_cf"][index] - probability_arrays["X_cf"][index]
            )
        )
        paired_rows.append(
            {
                "point": index,
                "target": int(lof_point["target"]),
                "valid_lof_first": lof_point["valid"],
                "valid_probability_first": probability_point["valid"],
                "lof_score_lof_first": lof_score,
                "lof_score_probability_first": probability_score,
                "lof_delta_on_minus_off": lof_score - probability_score,
                "cf_l2_between_variants": cf_distance,
                "same_counterfactual": bool(cf_distance <= 1e-12),
            }
        )

    lof_values = np.asarray(
        [row["lof_score_lof_first"] for row in paired_rows], dtype=float
    )
    probability_values = np.asarray(
        [row["lof_score_probability_first"] for row in paired_rows], dtype=float
    )
    lof_valid = np.asarray(
        [row["valid_lof_first"] == "True" for row in paired_rows], dtype=bool
    )
    probability_valid = np.asarray(
        [row["valid_probability_first"] == "True" for row in paired_rows],
        dtype=bool,
    )
    cf_different = np.asarray(
        [not row["same_counterfactual"] for row in paired_rows], dtype=bool
    )
    tolerance = 1e-12
    summary = {
        "n_test": len(paired_rows),
        "n_different_counterfactuals": int(cf_different.sum()),
        "validity_lof_first": float(lof_valid.mean()),
        "validity_probability_first": float(probability_valid.mean()),
        "mean_lof_lof_first": float(lof_values.mean()),
        "mean_lof_probability_first": float(probability_values.mean()),
        "median_lof_lof_first": float(np.median(lof_values)),
        "median_lof_probability_first": float(np.median(probability_values)),
        "mean_lof_delta_on_minus_off": float((lof_values - probability_values).mean()),
        "n_lof_first_more_plausible": int(
            (lof_values < probability_values - tolerance).sum()
        ),
        "n_probability_first_more_plausible": int(
            (probability_values < lof_values - tolerance).sum()
        ),
        "n_equal_lof": int(
            (np.abs(lof_values - probability_values) <= tolerance).sum()
        ),
        "n_valid_only_lof_first": int((lof_valid & ~probability_valid).sum()),
        "n_valid_only_probability_first": int((probability_valid & ~lof_valid).sum()),
    }

    _write_rows(results_dir / "exp10_lof_metrics.csv", metrics_rows)
    _write_rows(results_dir / "exp10_lof_paired.csv", paired_rows)
    _write_rows(results_dir / "exp10_lof_summary.csv", [summary])
    return summary


def main() -> None:
    """Run one ablation variant or aggregate both completed variants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=[*VARIANTS, "aggregate"],
        required=True,
    )
    parser.add_argument("--max-test", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--max-validity-steps", type=int, default=100)
    parser.add_argument("--tabicl-cache-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.variant == "aggregate":
        summary = aggregate_results(args.results_dir)
        for key, value in summary.items():
            print(f"{key}: {value}")
        return
    run_variant(
        args.variant,
        max_test=args.max_test,
        n_estimators=args.n_estimators,
        max_validity_steps=args.max_validity_steps,
        tabicl_cache_dir=args.tabicl_cache_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
