"""Helpers for the retained benchmark and focused test contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.zeroshot_cf.data import (
    get_actionable_immutable,
    get_grouped_categorical_action_space,
    get_one_hot_groups,
    load_dataset,
)
from experiments.zeroshot_cf.retained_config import TAU
from experiments.zeroshot_cf.vendor_setup import CEL_GIT_URL, CEL_REPO, PINNED_CEL_REVISION

ATHENA_CONTEXT_SIZE = 512
ATHENA_CONTEXT_STRATEGY = "gower_knn_both"
CF_MODES = ("sparse", "data_plausible")
DATASETS = ("heloc", "bank_marketing", "give_me_some_credit", "lending_club")
DEFAULT_GENERATOR_N_ESTIMATORS = 4
DEFAULT_POINT_ESTIMATE = "mode"
DEFAULT_TEMPERATURE = 1e-9
DEFAULT_CANDIDATE_QUANTILES = tuple(i / 10 for i in range(1, 10))
DEFAULT_CONFIDENCE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_DIVERSITY_BEAM_WIDTH = 8
DEFAULT_DIVERSITY_CANDIDATE_POOL_SIZE = 16
DEFAULT_DIVERSITY_MAX_EXTRA_ACTIONS = 2
DEFAULT_DIVERSITY_MAX_GOWER_INCREASE = 0.02
DEFAULT_DIVERSITY_MAX_GOWER_RATIO = 1.5
DEFAULT_JOINT_SHORTLIST_SIZE = 16
DEFAULT_MAX_EXTRA_ACTIONS = 1
DEFAULT_MAX_TEST = 1000
DEFAULT_MAX_VALIDITY_STEPS = 100
DEFAULT_MIN_JOINT_LOG_GAIN = 0.0
DEFAULT_N_COUNTERFACTUALS = 3
DEFAULT_N_ESTIMATORS = 1
DEFAULT_TABICL_JOINT_PERMUTATIONS = 1
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_DISC_TYPE = "lr"
DEFAULT_LR_PARAMS = {
    "max_iter": 1000,
    "random_state": 42,
    "C": 1.0,
}

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATASET_CONTRACT_FIXTURE = FIXTURES_DIR / "dataset_contract.json"


def read_dataset_contract_fixture() -> dict[str, Any]:
    return json.loads(DATASET_CONTRACT_FIXTURE.read_text())


def _relative(path: Path) -> str:
    absolute_path = path if path.is_absolute() else REPO_ROOT / path
    try:
        return str(absolute_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(absolute_path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return sum(1 for _ in reader)


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}


def _dataset_input_paths(dataset_name: str) -> tuple[Path, Path]:
    config_path = CEL_REPO / "config" / "datasets" / f"{dataset_name}.yaml"
    config = yaml.safe_load(config_path.read_text())
    raw_data_path = CEL_REPO / config["raw_data_path"]
    return config_path, raw_data_path


def _dataset_contract(dataset_name: str) -> dict[str, Any]:
    from cel.datasets.file_dataset import FileDataset

    config_path, raw_data_path = _dataset_input_paths(dataset_name)
    cel_dataset = FileDataset(config_path=config_path)
    usable_y = np.asarray(cel_dataset.y, dtype=np.int64)
    usable_row_count = len(cel_dataset.X)
    n_dropped_rows = 0
    preprocessing_variant = "original"
    if dataset_name == "heloc":
        all_minus9 = np.all(np.asarray(cel_dataset.X) == -9, axis=1)
        n_dropped_rows = int(all_minus9.sum())
        usable_y = usable_y[~all_minus9]
        usable_row_count -= n_dropped_rows
        preprocessing_variant = "drop_heloc_all_minus9"

    bundle = load_dataset(
        dataset_name,
        drop_heloc_all_minus9=(dataset_name == "heloc"),
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
    )
    one_hot_groups = get_one_hot_groups(bundle)
    if one_hot_groups:
        scalar_actionable, grouped_actionable, immutable_idx = (
            get_grouped_categorical_action_space(bundle)
        )
        actionable_idx = list(scalar_actionable)
        for group in grouped_actionable:
            actionable_idx.extend(group.columns)
    else:
        actionable_idx, immutable_idx = get_actionable_immutable(dataset_name, bundle)
    scaler = bundle.method_dataset.preprocessing_pipeline.get_step("minmax").scaler

    return {
        "config_path": _relative(config_path),
        "config_sha256": _sha256_file(config_path),
        "raw_data_path": _relative(raw_data_path),
        "raw_data_sha256": _sha256_file(raw_data_path),
        "raw_row_count": _row_count(raw_data_path),
        "cel_row_count": int(len(cel_dataset.X)),
        "usable_row_count": int(usable_row_count),
        "n_dropped_rows": int(n_dropped_rows),
        "split_variant": bundle.split_variant,
        "preprocessing_variant": preprocessing_variant,
        "feature_names": list(bundle.feature_names),
        "numerical_indices": list(bundle.numerical_features_indices),
        "categorical_indices": list(bundle.categorical_features_indices),
        "one_hot_groups": [
            {
                "name": group.name,
                "columns": list(group.columns),
                "feature_names": [bundle.feature_names[index] for index in group.columns],
            }
            for group in one_hot_groups
        ],
        "actionable_indices": list(actionable_idx),
        "immutable_indices": list(immutable_idx),
        "immutable_feature_names": [bundle.feature_names[index] for index in immutable_idx],
        "clean_label_counts": _label_counts(usable_y),
        "split_sizes": {
            "train": int(len(bundle.X_train)),
            "validation": 0 if bundle.X_val is None else int(len(bundle.X_val)),
            "test": int(len(bundle.X_test)),
        },
        "split_label_counts": {
            "train": _label_counts(bundle.y_train),
            "validation": {} if bundle.y_val is None else _label_counts(bundle.y_val),
            "test": _label_counts(bundle.y_test),
        },
        "scaler_min": np.asarray(scaler.data_min_, dtype=np.float64).round(12).tolist(),
        "scaler_max": np.asarray(scaler.data_max_, dtype=np.float64).round(12).tolist(),
        "array_hashes": {
            "X_train": _sha256_array(bundle.X_train),
            "y_train": _sha256_array(bundle.y_train),
            "X_val": _sha256_array(bundle.X_val),
            "y_val": _sha256_array(bundle.y_val),
            "X_test": _sha256_array(bundle.X_test),
            "y_test": _sha256_array(bundle.y_test),
        },
    }


def build_dataset_contract_fixture() -> dict[str, Any]:
    return {
        "fixture_version": 1,
        "cel": {
            "git_url": CEL_GIT_URL,
            "revision": PINNED_CEL_REVISION,
            "vendor_repo": _relative(CEL_REPO),
        },
        "generator_defaults": {
            "context_strategy": ATHENA_CONTEXT_STRATEGY,
            "context_size": ATHENA_CONTEXT_SIZE,
            "context_labels": "target_classifier",
            "target_policy": "flip_classifier_prediction",
            "candidate_mode": "batched",
            "point_estimate": DEFAULT_POINT_ESTIMATE,
            "temperature": DEFAULT_TEMPERATURE,
            "n_estimators": DEFAULT_GENERATOR_N_ESTIMATORS,
            "cf_modes": list(CF_MODES),
            "split_seed": 42,
            "test_selection_seed": 42,
        },
        "benchmark_defaults": {
            "datasets": list(DATASETS),
            "max_test": DEFAULT_MAX_TEST,
            "validation_fraction": DEFAULT_VALIDATION_FRACTION,
            "tau": TAU,
            "temperature": DEFAULT_TEMPERATURE,
            "n_estimators": DEFAULT_N_ESTIMATORS,
            "candidate_quantiles": list(DEFAULT_CANDIDATE_QUANTILES),
            "confidence_quantiles": list(DEFAULT_CONFIDENCE_QUANTILES),
            "max_validity_steps": DEFAULT_MAX_VALIDITY_STEPS,
            "allow_revisits": True,
            "joint_shortlist_size": DEFAULT_JOINT_SHORTLIST_SIZE,
            "max_extra_actions": DEFAULT_MAX_EXTRA_ACTIONS,
            "min_joint_log_gain": DEFAULT_MIN_JOINT_LOG_GAIN,
            "tabicl_joint_permutations": DEFAULT_TABICL_JOINT_PERMUTATIONS,
            "n_counterfactuals": DEFAULT_N_COUNTERFACTUALS,
            "diversity_beam_width": DEFAULT_DIVERSITY_BEAM_WIDTH,
            "diversity_candidate_pool_size": DEFAULT_DIVERSITY_CANDIDATE_POOL_SIZE,
            "diversity_max_extra_actions": DEFAULT_DIVERSITY_MAX_EXTRA_ACTIONS,
            "diversity_max_gower_ratio": DEFAULT_DIVERSITY_MAX_GOWER_RATIO,
            "diversity_max_gower_increase": DEFAULT_DIVERSITY_MAX_GOWER_INCREASE,
            "drop_heloc_all_minus9": True,
            "test_selection": "stratified",
        },
        "discriminator_defaults": {
            "disc_type": DEFAULT_DISC_TYPE,
            "lr_params": DEFAULT_LR_PARAMS,
        },
        "datasets": {
            dataset_name: _dataset_contract(dataset_name) for dataset_name in DATASETS
        },
    }
