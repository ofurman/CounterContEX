"""Read-only robustness rescoring of published counterfactual artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from experiments.zeroshot_cf.analysis.core import load_published_cells
from experiments.zeroshot_cf.discriminator import (
    DEFAULT_LR_PARAMS,
    DEFAULT_MLP_PARAMS,
    DEFAULT_XGB_PARAMS,
)
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config
from experiments.zeroshot_cf.orchestration.runner import _default_case_loader
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

RETRAINING_SEEDS = (137, 271, 811)
PROBABILITY_BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000000001)
PROBABILITY_LABELS = (
    "[0.0,0.5)",
    "[0.5,0.6)",
    "[0.6,0.7)",
    "[0.7,0.8)",
    "[0.8,0.9)",
    "[0.9,1.0]",
)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _instrument(model_name: str, seed: int):
    if model_name == "retained_logistic_regression":
        return LogisticRegression(**{**DEFAULT_LR_PARAMS, "random_state": seed})
    if model_name == "retained_mlp":
        return MLPClassifier(**{**DEFAULT_MLP_PARAMS, "random_state": seed})
    if model_name == "retained_xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**{**DEFAULT_XGB_PARAMS, "random_state": seed})
    raise ValueError(f"unsupported robustness target model: {model_name!r}")


def _probability_bins(values: np.ndarray) -> np.ndarray:
    positions = np.digitize(values, PROBABILITY_BINS[1:-1], right=False)
    return np.asarray([PROBABILITY_LABELS[index] for index in positions])


def build_e8_robustness(
    output_root: Path | str,
    matrix_config: Path | str,
    output_dir: Path | str,
) -> tuple[Path, Path, Path]:
    """Retrain evaluation-only models and rescore immutable E1 candidates."""
    source_root = Path(output_root)
    destination = Path(output_dir)
    if destination == source_root or source_root in destination.parents:
        raise ValueError("robustness outputs must be outside the source artifact root")
    before = _tree_digest(source_root)
    cells = load_published_cells(source_root, matrix_config)
    config = load_matrix_config(matrix_config)
    specs = {
        (run.dataset.name, run.target_model.name): run for run in config.runs
    }
    grouped_cells: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        key = (str(cell["dataset"]), str(cell["target_model"]))
        grouped_cells.setdefault(key, []).append(cell)

    rows: list[dict[str, Any]] = []
    instrument_configs: list[dict[str, Any]] = []
    for key, members in sorted(grouped_cells.items()):
        dataset_name, model_name = key
        loaded = _default_case_loader(specs[key])
        case = loaded.case
        for retraining_seed in RETRAINING_SEEDS:
            estimator = _instrument(model_name, retraining_seed)
            estimator.fit(case.dataset.X_train, case.dataset.y_train)
            instrument_configs.append(
                {
                    "dataset": dataset_name,
                    "target_model": model_name,
                    "retraining_seed": retraining_seed,
                    "params": estimator.get_params(deep=False),
                }
            )
            for cell in members:
                artifact = Path(str(cell["artifact_path"]))
                arrays = np.load(artifact / "arrays.npz")
                candidates = arrays["common.candidates"]
                available = arrays["common.available"]
                probabilities = arrays["common.target_probabilities"]
                points = pd.read_csv(artifact / "points.csv")
                point_indices, ranks = np.nonzero(available)
                if not len(point_indices):
                    continue
                candidate_rows = candidates[point_indices, ranks]
                predictions = np.asarray(estimator.predict(candidate_rows))
                targets = points["target"].to_numpy()[point_indices]
                original = probabilities[point_indices, ranks]
                bins = _probability_bins(original)
                for index in range(len(candidate_rows)):
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "target_model": model_name,
                            "method": cell["method"],
                            "original_seed": cell["seed"],
                            "retraining_seed": retraining_seed,
                            "point": int(point_indices[index]),
                            "rank": int(ranks[index]),
                            "original_target_probability": float(original[index]),
                            "probability_bin": bins[index],
                            "retained_class_validity": bool(
                                predictions[index] == targets[index]
                            ),
                        }
                    )

    destination.mkdir(parents=True, exist_ok=True)
    candidates_path = destination / "e8_robustness_candidates.csv"
    table_path = destination / "e8_robustness.csv"
    manifest_path = destination / "e8_robustness_manifest.json"
    detail = pd.DataFrame(rows)
    detail.to_csv(candidates_path, index=False)
    table = (
        detail.groupby(
            ["dataset", "target_model", "method", "probability_bin"],
            observed=True,
            as_index=False,
        )
        .agg(
            retained_validity=("retained_class_validity", "mean"),
            n=("retained_class_validity", "size"),
        )
        .sort_values(["dataset", "target_model", "method", "probability_bin"])
    )
    table.to_csv(table_path, index=False)
    after = _tree_digest(source_root)
    if after != before:
        raise RuntimeError("E8 modified its source artifact tree")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "countercontex.robustness.v1",
                "source_root": str(source_root),
                "source_digest_before": before,
                "source_digest_after": after,
                "retraining_seeds": list(RETRAINING_SEEDS),
                "probability_bins": list(PROBABILITY_LABELS),
                "instruments": instrument_configs,
                "candidate_rows": len(detail),
                "table_rows": len(table),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    return table_path, candidates_path, manifest_path
