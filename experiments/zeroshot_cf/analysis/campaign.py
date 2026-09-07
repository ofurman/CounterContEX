"""Build the final campaign paper products from published run artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from experiments.zeroshot_cf.analysis.builders import (
    build_f3_critical_difference,
    build_f4_confidence_campaign,
    build_f5_cost_quality,
    build_f6_target_probability,
    build_t1_main,
    build_t2_diversity,
    build_t3_backend,
)
from experiments.zeroshot_cf.analysis.core import load_published_cells, write_rows
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

_CASE_DATASETS = ("heloc", "adult_census")
_BASELINES = ("nice", "dice")


def _qualitative_rows(
    *,
    dataset: str,
    source_index: int,
    feature_names: Sequence[str],
    factual: np.ndarray,
    candidates: Mapping[str, Sequence[tuple[int, np.ndarray, float]]],
    inverse_transform: Callable[[np.ndarray], np.ndarray],
) -> tuple[dict[str, Any], ...]:
    """Return long-form factual/candidate values in dataset feature units."""
    ordered = [("factual", -1, np.asarray(factual), float("nan"))]
    ordered.extend(
        (method, rank, np.asarray(values), probability)
        for method, method_rows in candidates.items()
        for rank, values, probability in method_rows
    )
    transformed = np.asarray(
        inverse_transform(np.vstack([values for _, _, values, _ in ordered]))
    )
    factual_values = transformed[0]
    rows: list[dict[str, Any]] = []
    for (method, rank, _values, probability), candidate_values in zip(
        ordered, transformed, strict=True
    ):
        for feature, factual_value, candidate_value in zip(
            feature_names, factual_values, candidate_values, strict=True
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "source_index": source_index,
                    "method": method,
                    "rank": rank,
                    "target_probability": probability,
                    "feature": feature,
                    "factual_value": float(factual_value),
                    "candidate_value": float(candidate_value),
                    "changed": bool(
                        method != "factual"
                        and not np.isclose(
                            candidate_value, factual_value, rtol=0.0, atol=1e-7
                        )
                    ),
                }
            )
    return tuple(rows)


def _one_cell(
    cells: Sequence[Mapping[str, Any]], *, dataset: str, method: str
) -> Mapping[str, Any]:
    matches = [
        cell
        for cell in cells
        if cell["dataset"] == dataset
        and cell["method"] == method
        and cell["target_model"] == "retained_logistic_regression"
        and cell["seed"] == 42
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {dataset}/{method}/LR/seed-42 cell, got {len(matches)}"
        )
    return matches[0]


def _authenticate_case_artifacts(
    loaded: Any, artifact_paths: Sequence[Path]
) -> None:
    for artifact_path in artifact_paths:
        payload = json.loads((artifact_path / "manifest.json").read_text())
        if loaded.case.case_id != payload["report_metadata"]["case_id"]:
            raise ValueError(
                "reconstructed qualitative case does not match artifact case_id"
            )


def _load_case(run: Any, artifact_paths: Sequence[Path]) -> Any:
    # Lazy import keeps ordinary analyses free of dataset and model loading.
    from experiments.zeroshot_cf.orchestration.runner import _default_case_loader

    loaded = _default_case_loader(run)
    _authenticate_case_artifacts(loaded, artifact_paths)
    return loaded


def _artifact_candidates(
    cell: Mapping[str, Any], point: int
) -> tuple[tuple[int, np.ndarray, float], ...]:
    artifact = Path(cell["artifact_path"])
    point_table = pd.read_csv(artifact / "points.csv")
    if point not in set(point_table["point"]):
        raise ValueError("qualitative point is absent from points.csv")
    table = pd.read_csv(artifact / "candidates.csv")
    arrays = np.load(artifact / "arrays.npz")
    rows = table[(table["point"] == point) & table["available"]]
    result = []
    for row in rows.itertuples():
        rank = int(row.rank)
        probability = float(row.target_probability)
        stored = float(arrays["common.target_probabilities"][point, rank])
        if not np.isclose(probability, stored, rtol=0.0, atol=1e-15):
            raise ValueError("candidate table and array probabilities disagree")
        result.append((rank, arrays["common.candidates"][point, rank], probability))
    return tuple(result)


def build_f7_campaign(
    headline_output_root: Path | str,
    headline_matrix_config: Path | str,
    baseline_output_root: Path | str,
    baseline_matrix_config: Path | str,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Build matched HELOC/Adult cases for CounterContEx, NICE, and DiCE."""
    headline_config = load_matrix_config(headline_matrix_config)
    baseline_config = load_matrix_config(baseline_matrix_config)
    headline_cells = load_published_cells(headline_output_root, headline_matrix_config)
    baseline_cells = load_published_cells(baseline_output_root, baseline_matrix_config)
    all_rows: list[dict[str, Any]] = []
    for dataset in _CASE_DATASETS:
        headline_cell = _one_cell(
            headline_cells, dataset=dataset, method="countercontex"
        )
        baseline_by_method = {
            method: _one_cell(baseline_cells, dataset=dataset, method=method)
            for method in _BASELINES
        }
        headline_run = next(
            run
            for run in headline_config.runs
            if run.dataset.name == dataset and run.seed == 42
        )
        baseline_run = next(
            run
            for run in baseline_config.runs
            if run.dataset.name == dataset
            and run.target_model.name == "retained_logistic_regression"
            and run.seed == 42
        )
        headline_loaded = _load_case(
            headline_run, (Path(headline_cell["artifact_path"]),)
        )
        baseline_loaded = _load_case(
            baseline_run,
            tuple(
                Path(cell["artifact_path"])
                for cell in baseline_by_method.values()
            ),
        )
        headline_indices = {
            int(index): point
            for point, index in enumerate(headline_loaded.case.factuals.indices)
        }
        baseline_indices = {
            int(index): point
            for point, index in enumerate(baseline_loaded.case.factuals.indices)
        }
        selected = None
        for source_index in sorted(headline_indices.keys() & baseline_indices.keys()):
            headline_point = headline_indices[source_index]
            baseline_point = baseline_indices[source_index]
            candidate_map = {
                "countercontex": _artifact_candidates(headline_cell, headline_point),
                **{
                    method: _artifact_candidates(cell, baseline_point)
                    for method, cell in baseline_by_method.items()
                },
            }
            if len(candidate_map["countercontex"]) == 3 and all(
                candidate_map[method] for method in _BASELINES
            ):
                selected = source_index, headline_point, candidate_map
                break
        if selected is None:
            raise ValueError(f"no complete matched qualitative case for {dataset}")
        source_index, headline_point, candidate_map = selected
        adapter = headline_loaded.runtime_context["dataset_adapter"]
        all_rows.extend(
            _qualitative_rows(
                dataset=dataset,
                source_index=source_index,
                feature_names=headline_loaded.case.dataset.schema.names,
                factual=headline_loaded.case.factuals.values[headline_point],
                candidates=candidate_map,
                inverse_transform=adapter.inverse_transform,
            )
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = output / "f7_qualitative_case.csv"
    figure = output / "f7_qualitative_case.pdf"
    write_rows(data, tuple(all_rows))
    frame = pd.DataFrame(all_rows)
    with PdfPages(figure) as pdf:
        for dataset in _CASE_DATASETS:
            changed = frame[
                (frame["dataset"] == dataset) & frame["changed"]
            ][["method", "rank", "feature", "factual_value", "candidate_value"]]
            fig, axis = plt.subplots(figsize=(11.0, max(3.0, 0.28 * len(changed))))
            axis.axis("off")
            axis.set_title(f"{dataset}: changed features in original units")
            table = axis.table(
                cellText=changed.round(4).values,
                colLabels=changed.columns,
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.auto_set_column_width(col=list(range(len(changed.columns))))
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return figure, data


def build_campaign_paper(
    matrix_dir: Path | str, output_dir: Path | str
) -> tuple[Path, ...]:
    """Build the campaign's declared T1--T3 and F3--F7 product set."""
    matrices = Path(matrix_dir)
    output = Path(output_dir)
    configs = {
        name: matrices / f"campaign_{name}.yaml"
        for name in (
            "e1_main",
            "e2_diverse",
            "e4_confidence",
            "e7_cost",
            "e9_fmswap",
            "e10_headline",
        )
    }
    loaded = {name: load_matrix_config(path) for name, path in configs.items()}
    products = [
        build_t1_main(
            loaded["e1_main"].execution.output_root, configs["e1_main"], output
        ),
        build_t2_diversity(
            loaded["e2_diverse"].execution.output_root, configs["e2_diverse"], output
        ),
        build_t3_backend(
            loaded["e9_fmswap"].execution.output_root, configs["e9_fmswap"], output
        ),
        build_f3_critical_difference(
            loaded["e1_main"].execution.output_root, configs["e1_main"], output
        ),
        build_f4_confidence_campaign(
            loaded["e4_confidence"].execution.output_root,
            configs["e4_confidence"],
            loaded["e1_main"].execution.output_root,
            configs["e1_main"],
            output,
        ),
        build_f5_cost_quality(
            loaded["e7_cost"].execution.output_root, configs["e7_cost"], output
        ),
        build_f6_target_probability(
            loaded["e1_main"].execution.output_root, configs["e1_main"], output
        ),
        build_f7_campaign(
            loaded["e10_headline"].execution.output_root,
            configs["e10_headline"],
            loaded["e1_main"].execution.output_root,
            configs["e1_main"],
            output,
        ),
    ]
    paths = tuple(path for pair in products for path in pair)
    manifest = output / "analysis_manifest.json"
    manifest.write_text(json.dumps([path.name for path in paths], indent=2) + "\n")
    return (*paths, manifest)
