"""Dataset loading for the zero-shot CF experiment.

Loads HELOC, MOONS, AUDIT, and German Credit via cel, applies MinMax scaling (fit on
train), and provides each dataset's actionable/immutable feature split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
from cel.datasets.file_dataset import FileDataset
from cel.datasets.method_dataset import MethodDataset
from cel.preprocessing.pipeline import PreprocessingPipeline
from cel.preprocessing.scalers import MinMaxScalingStep

CEL_REPO = Path(__file__).parent / "vendor" / "counterfactuals"
CONFIGS_DIR = Path(__file__).parent / "configs"


@dataclass
class DatasetBundle:
    """All data and metadata for one experiment dataset."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    numerical_features_indices: List[int]
    categorical_features_indices: List[int]
    method_dataset: MethodDataset  # for inverse_transform back to original space
    n_dropped_rows: int = 0
    preprocessing_variant: str = "original"

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.method_dataset.inverse_transform(X)


@dataclass(frozen=True)
class OneHotActionGroup:
    """One categorical action represented by an atomic one-hot column group."""

    name: str
    columns: Tuple[int, ...]


def get_one_hot_groups(dataset: DatasetBundle) -> List[OneHotActionGroup]:
    """Resolve every metadata-defined one-hot group to transformed columns."""
    raw_groups = getattr(
        dataset.method_dataset.file_dataset,
        "one_hot_feature_groups",
        {},
    )
    feature_to_idx = {name: i for i, name in enumerate(dataset.feature_names)}
    return [
        OneHotActionGroup(
            group_name,
            tuple(feature_to_idx[name] for name in feature_names),
        )
        for group_name, feature_names in raw_groups.items()
    ]


def get_grouped_categorical_action_space(
    dataset: DatasetBundle,
) -> Tuple[List[int], List[OneHotActionGroup], List[int]]:
    """Return scalar actions, atomic one-hot actions, and true immutables.

    A one-hot group is actionable only when *every* member column is declared
    actionable by the dataset metadata.  This prevents a partial dummy edit
    and, in German Credit, keeps ``personal_status_sex`` immutable while
    exposing the other categorical variables as whole-group interventions.

    The first return value intentionally contains scalar columns only.  It can
    therefore be passed to the existing numerical greedy search without ever
    producing malformed one-hot vectors.
    """
    method_dataset = dataset.method_dataset
    all_groups = get_one_hot_groups(dataset)
    if not all_groups:
        actionable, immutable = get_actionable_immutable(dataset.name, dataset)
        return actionable, [], immutable

    declared_actionable = set(method_dataset.actionable_features)
    grouped_columns: set[int] = set()
    actionable_groups: List[OneHotActionGroup] = []

    for group in all_groups:
        grouped_columns.update(group.columns)
        feature_names = [dataset.feature_names[i] for i in group.columns]
        if all(name in declared_actionable for name in feature_names):
            actionable_groups.append(group)

    scalar_actionable = [
        i
        for i, name in enumerate(dataset.feature_names)
        if i not in grouped_columns and name in declared_actionable
    ]
    actionable_columns = set(scalar_actionable)
    for group in actionable_groups:
        actionable_columns.update(group.columns)
    immutable = [
        i for i in range(len(dataset.feature_names)) if i not in actionable_columns
    ]
    return scalar_actionable, actionable_groups, immutable


def load_dataset(
    name: str,
    *,
    drop_heloc_all_minus9: bool = False,
) -> DatasetBundle:
    """Load a supported cel classification dataset, MinMax-scaled.

    Split is 80/20 stratified with random_state=42 (cel default).
    Scaling is fit on X_train only.

    When ``drop_heloc_all_minus9`` is enabled, completely unavailable HELOC
    bureau records are removed before splitting and scaling. Partial special
    codes are intentionally preserved for this controlled comparison.
    """
    config_path = CEL_REPO / "config" / "datasets" / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    file_dataset = FileDataset(config_path=config_path)
    n_dropped_rows = 0
    preprocessing_variant = "original"
    if name == "heloc" and drop_heloc_all_minus9:
        # FICO uses -9 as a symbolic "No Bureau Record or No Investigation"
        # code. Rows containing -9 in every predictor have no usable factual
        # profile and cannot support individualized counterfactual recourse.
        # Remove them before the train/test split and before MinMax scaling.
        all_minus9 = np.all(np.asarray(file_dataset.X) == -9, axis=1)
        n_dropped_rows = int(all_minus9.sum())
        keep = ~all_minus9
        file_dataset.X = file_dataset.X[keep]
        file_dataset.y = file_dataset.y[keep]
        file_dataset.raw_data = file_dataset.raw_data.loc[keep].reset_index(drop=True)
        preprocessing_variant = "drop_heloc_all_minus9"
    preprocessing = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
        ]
    )
    md = MethodDataset(file_dataset, preprocessing_pipeline=preprocessing)

    return DatasetBundle(
        name=name,
        X_train=md.X_train.astype(np.float64),
        X_test=md.X_test.astype(np.float64),
        y_train=md.y_train.astype(np.int64),
        y_test=md.y_test.astype(np.int64),
        feature_names=list(md.features),
        numerical_features_indices=list(md.numerical_features_indices),
        categorical_features_indices=list(md.categorical_features_indices),
        method_dataset=md,
        n_dropped_rows=n_dropped_rows,
        preprocessing_variant=preprocessing_variant,
    )


def get_actionable_immutable(
    name: str, dataset: DatasetBundle | None = None
) -> Tuple[List[int], List[int]]:
    """Return (actionable_idx, immutable_idx) in the scaled feature matrix column order.

    For 'heloc': uses configs/heloc_actionability.yaml (Decision #2).
    For 'german_credit': numerical features are actionable; one-hot categorical
    groups are fixed until grouped categorical interventions are implemented.
    For 'moons' and 'audit': all features are actionable, no immutables.

    Args:
        name: Dataset name ('heloc', 'moons', or 'audit').
        dataset: Optional pre-loaded DatasetBundle; if None the dataset is loaded
                 just to resolve feature names → column indices.

    Returns:
        Tuple of (actionable_column_indices, immutable_column_indices).
    """
    if name in {"moons", "audit"}:
        if dataset is None:
            dataset = load_dataset(name)
        n = len(dataset.feature_names)
        return list(range(n)), []

    if name == "german_credit":
        if dataset is None:
            dataset = load_dataset(name)
        actionable_idx = list(dataset.numerical_features_indices)
        immutable_idx = [
            i for i in range(len(dataset.feature_names)) if i not in actionable_idx
        ]
        return actionable_idx, immutable_idx

    if name == "heloc":
        cfg_path = CONFIGS_DIR / "heloc_actionability.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        immutable_names: List[str] = cfg["immutable_features"]

        if dataset is None:
            dataset = load_dataset("heloc")

        feature_names = dataset.feature_names
        immutable_idx = [feature_names.index(fn) for fn in immutable_names]
        actionable_idx = [
            i for i in range(len(feature_names)) if i not in immutable_idx
        ]
        return actionable_idx, immutable_idx

    raise ValueError(
        f"Unknown dataset: {name!r}. Supported: 'heloc', 'moons', 'audit', "
        "'german_credit'."
    )
