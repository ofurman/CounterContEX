"""Path-only builders for paper figures and tables."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from experiments.zeroshot_cf.analysis.core import (
    aggregate_seeds,
    load_published_cells,
    write_rows,
)
from experiments.zeroshot_cf.analysis.statistics import holm_wilcoxon
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config
from scipy.stats import rankdata, studentized_range

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _paths(output_dir: Path | str, stem: str) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{stem}.pdf", root / f"{stem}.csv"


def _numeric(rows: tuple[dict[str, Any], ...], name: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if name not in frame or not pd.api.types.is_numeric_dtype(frame[name]):
        raise ValueError(f"artifact metric {name!r} is unavailable")
    return frame[frame[name].notna()].copy()


def _save_scatter(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    figure: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for method, group in frame.groupby("method", sort=True):
        ordered = group.sort_values(x)
        axis.plot(ordered[x], ordered[y], marker="o", label=method)
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(figure)
    plt.close(fig)


def build_f3_critical_difference(
    output_root: Path | str, matrix_config: Path | str, output_dir: Path | str
) -> tuple[Path, Path]:
    cells = load_published_cells(output_root, matrix_config)
    metric = "primary_validity_returned_class"
    frame = _numeric(cells, metric)
    grouped = frame.groupby(
        ["dataset", "target_model", "method"], as_index=False
    )[metric].mean()
    ranks: dict[str, list[float]] = defaultdict(list)
    for _key, block in grouped.groupby(["dataset", "target_model"]):
        values = block[metric].to_numpy(dtype=float)
        for method, rank in zip(
            block["method"], rankdata(-values, method="average"), strict=True
        ):
            ranks[str(method)].append(float(rank))
    rows = tuple(
        {"method": method, "average_rank": float(np.mean(values)), "n": len(values)}
        for method, values in sorted(ranks.items())
    )
    figure, data = _paths(output_dir, "f3_critical_difference")
    write_rows(data, rows)
    n_blocks = min((row["n"] for row in rows), default=0)
    n_methods = len(rows)
    critical_difference = None
    if n_blocks and n_methods > 1:
        q_alpha = studentized_range.ppf(0.95, n_methods, math.inf) / math.sqrt(2)
        critical_difference = float(
            q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6 * n_blocks))
        )
    fig, axis = plt.subplots(figsize=(7.0, 2.8))
    for index, row in enumerate(rows):
        axis.scatter(row["average_rank"], index)
        axis.text(row["average_rank"], index + 0.12, row["method"], ha="center")
    if critical_difference is not None:
        axis.plot([1, 1 + critical_difference], [-0.6, -0.6], color="black")
        axis.text(
            1 + critical_difference / 2,
            -0.45,
            f"CD={critical_difference:.3f}",
            ha="center",
        )
    axis.set(xlabel="Average rank (lower is better)", yticks=[])
    fig.tight_layout()
    fig.savefig(figure)
    plt.close(fig)
    return figure, data


def build_f4_confidence_pareto(
    output_root: Path | str, matrix_config: Path | str, output_dir: Path | str
) -> tuple[Path, Path]:
    rows = aggregate_seeds(output_root, matrix_config)
    frame = _numeric(rows, "proximity_grouped_gower_mean")
    y = "validity_returned_threshold_mean"
    figure, data = _paths(output_dir, "f4_confidence_pareto")
    frame.to_csv(data, index=False)
    _save_scatter(
        frame,
        x="proximity_grouped_gower_mean",
        y=y,
        figure=figure,
        xlabel="Grouped-Gower proximity",
        ylabel="Threshold validity",
    )
    return figure, data


def build_f5_cost_quality(
    output_root: Path | str, matrix_config: Path | str, output_dir: Path | str
) -> tuple[Path, Path]:
    cells = load_published_cells(output_root, matrix_config)
    frame = _numeric(cells, "timing_total_s")
    grouped = frame.groupby("method", as_index=False).agg(
        timing_total_s=("timing_total_s", "mean"),
        validity_returned_threshold=("validity_returned_threshold", "mean"),
    )
    figure, data = _paths(output_dir, "f5_cost_quality")
    grouped.to_csv(data, index=False)
    _save_scatter(
        grouped,
        x="timing_total_s",
        y="validity_returned_threshold",
        figure=figure,
        xlabel="Total runtime (s)",
        ylabel="Threshold validity",
    )
    return figure, data


def build_f6_target_probability(
    output_root: Path | str, matrix_config: Path | str, output_dir: Path | str
) -> tuple[Path, Path]:
    cells = load_published_cells(output_root, matrix_config)
    rows: list[dict[str, Any]] = []
    for cell in cells:
        arrays = np.load(Path(cell["artifact_path"]) / "arrays.npz")
        finite_slots = np.argwhere(
            np.isfinite(arrays["common.target_probabilities"])
        )
        for point, rank in finite_slots:
            rows.append(
                {
                    "dataset": cell["dataset"],
                    "target_model": cell["target_model"],
                    "method": cell["method"],
                    "seed": cell["seed"],
                    "point": int(point),
                    "rank": int(rank),
                    "target_probability": float(
                        arrays["common.target_probabilities"][point, rank]
                    ),
                }
            )
    figure, data = _paths(output_dir, "f6_target_probability")
    write_rows(data, tuple(rows))
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    methods = sorted(frame["method"].unique())
    axis.boxplot(
        [
            frame.loc[frame["method"] == method, "target_probability"]
            for method in methods
        ],
        tick_labels=methods,
    )
    axis.set(ylabel="Target probability")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figure)
    plt.close(fig)
    return figure, data


def build_f7_qualitative_case(
    output_root: Path | str, matrix_config: Path | str, output_dir: Path | str
) -> tuple[Path, Path]:
    cells = load_published_cells(output_root, matrix_config)
    rows: list[dict[str, Any]] = []
    for cell in cells:
        arrays = np.load(Path(cell["artifact_path"]) / "arrays.npz")
        available = np.argwhere(arrays["common.available"])
        if not len(available):
            continue
        point, rank = available[0]
        for feature, value in enumerate(arrays["common.candidates"][point, rank]):
            rows.append(
                {
                    "dataset": cell["dataset"],
                    "method": cell["method"],
                    "point": int(point),
                    "rank": int(rank),
                    "feature": feature,
                    "candidate_value": float(value),
                }
            )
    figure, data = _paths(output_dir, "f7_qualitative_case")
    write_rows(data, tuple(rows))
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(8.0, 4.2))
    for method, group in frame.groupby("method", sort=True):
        axis.plot(group["feature"], group["candidate_value"], marker=".", label=method)
    axis.set(xlabel="Feature index", ylabel="Candidate value")
    axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(figure)
    plt.close(fig)
    return figure, data


_TABLE_METRICS = {
    "t1_main": (
        "primary_coverage",
        "primary_validity_returned_class",
        "primary_validity_returned_threshold",
        "proximity_grouped_gower",
        "action_unit_sparsity_mean",
    ),
    "t2_diversity": (
        "set_coverage_at_k",
        "set_validity_returned_class",
        "set_validity_returned_threshold",
        "proximity_grouped_gower",
        "set_action_jaccard_mean",
        "set_pairwise_gower_mean",
    ),
    "t3_backend": (
        "coverage",
        "validity_returned_threshold",
        "proximity_grouped_gower",
        "plausibility_gower_kth_neighbor_mean",
    ),
}


def _build_table(
    output_root: Path | str,
    matrix_config: Path | str,
    output_dir: Path | str,
    stem: str,
) -> tuple[Path, Path]:
    rows = aggregate_seeds(output_root, matrix_config)
    identity = [
        "dataset",
        "target_model",
        "method",
        "method_variant",
        "backend",
        "n_counterfactuals",
        "seed_n",
    ]
    columns = identity + [
        f"{metric}_{suffix}"
        for metric in _TABLE_METRICS[stem]
        for suffix in ("mean", "std", "n")
    ]
    frame = pd.DataFrame(rows).reindex(columns=columns)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = root / f"{stem}.csv", root / f"{stem}.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(frame.to_latex(index=False, float_format="%.4f"))
    return tex_path, csv_path


def build_t1_main(
    output_root: Path | str, matrix_config: Path | str, output_dir: Path | str
) -> tuple[Path, Path]:
    return _build_table(output_root, matrix_config, output_dir, "t1_main")


def build_t2_diversity(
    output_root: Path | str, matrix_config: Path | str, output_dir: Path | str
) -> tuple[Path, Path]:
    return _build_table(output_root, matrix_config, output_dir, "t2_diversity")


def build_t3_backend(
    output_root: Path | str, matrix_config: Path | str, output_dir: Path | str
) -> tuple[Path, Path]:
    return _build_table(output_root, matrix_config, output_dir, "t3_backend")


def build_significance(
    output_root: Path | str, matrix_config: Path | str, output_dir: Path | str
) -> Path:
    cells = load_published_cells(output_root, matrix_config)
    config = load_matrix_config(matrix_config)
    reference = "countercontex"
    if reference not in {run.method.name for run in config.runs}:
        raise ValueError("significance analysis requires countercontex")
    comparisons = {}
    identities = {}
    noise_floors = {}
    for metric in _TABLE_METRICS["t1_main"]:
        frame = _numeric(cells, metric)
        grid = frame.groupby(
            ["dataset", "target_model", "method"], as_index=False
        )[metric].mean()
        pivot = grid.pivot(
            index=["dataset", "target_model"], columns="method", values=metric
        )
        seed_spread = frame.groupby(
            ["dataset", "target_model", "method"]
        )[metric].std()
        noise_floor = float(seed_spread.fillna(0.0).max())
        for method in pivot.columns:
            if method == reference:
                continue
            key = f"{metric}:{method}"
            comparisons[key] = (
                pivot[reference].to_numpy(),
                pivot[method].to_numpy(),
            )
            identities[key] = (metric, method)
            noise_floors[key] = noise_floor
    results = holm_wilcoxon(comparisons, noise_floor=0.0)
    rows = []
    for result in results:
        metric, method = identities[result.comparison]
        row = {"metric": metric, **result.__dict__}
        row["comparison"] = method
        row["below_noise_floor"] = (
            abs(result.mean_difference) < noise_floors[result.comparison]
        )
        rows.append(row)
    output = Path(output_dir) / "significance.csv"
    write_rows(output, tuple(rows))
    return output


def build_all(
    output_root: Path | str, matrix_config: Path | str, output_dir: Path | str
) -> tuple[Path, ...]:
    products = [
        build_f3_critical_difference(output_root, matrix_config, output_dir),
        build_f4_confidence_pareto(output_root, matrix_config, output_dir),
        build_f5_cost_quality(output_root, matrix_config, output_dir),
        build_f6_target_probability(output_root, matrix_config, output_dir),
        build_f7_qualitative_case(output_root, matrix_config, output_dir),
        build_t1_main(output_root, matrix_config, output_dir),
        build_t2_diversity(output_root, matrix_config, output_dir),
        build_t3_backend(output_root, matrix_config, output_dir),
    ]
    paths = [path for pair in products for path in pair]
    paths.append(build_significance(output_root, matrix_config, output_dir))
    manifest = Path(output_dir) / "analysis_manifest.json"
    manifest.write_text(json.dumps([str(path) for path in paths], indent=2) + "\n")
    return tuple([*paths, manifest])
