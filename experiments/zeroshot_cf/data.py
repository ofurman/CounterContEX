"""Dataset loading for the zero-shot CF experiment.

Loads HELOC, MOONS, AUDIT, and German Credit via cel, applies MinMax scaling (fit on
train), and provides each dataset's actionable/immutable feature split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.contracts import PreparedDataset
from experiments.zeroshot_cf.datasets.base import DatasetSpec
from experiments.zeroshot_cf.datasets.cel import CelDatasetProvider

if TYPE_CHECKING:
    from cel.datasets.method_dataset import MethodDataset

CEL_REPO = Path(__file__).parent / "vendor" / "counterfactuals"
CONFIGS_DIR = Path(__file__).parent / "configs"


@dataclass
class DatasetBundle:
    """Deprecated CEL compatibility view; use ``PreparedDataset`` in new code."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    numerical_features_indices: list[int]
    categorical_features_indices: list[int]
    method_dataset: MethodDataset  # for inverse_transform back to original space
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    split_variant: str = "train_test_80_20"
    n_dropped_rows: int = 0
    preprocessing_variant: str = "original"
    prepared: PreparedDataset | None = None

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.method_dataset.inverse_transform(X)


def get_one_hot_groups(dataset: DatasetBundle) -> list[OneHotActionGroup]:
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
) -> tuple[list[int], list[OneHotActionGroup], list[int]]:
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
    actionable_groups: list[OneHotActionGroup] = []

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


def select_test_rows(
    X_test: np.ndarray,
    y_test: np.ndarray,
    limit: int | None,
    selection: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility delegate for deterministic held-out factual selection."""
    from experiments.zeroshot_cf.datasets.benchmark import select_factual_indices

    selected = select_factual_indices(y_test, limit, selection, seed=42)
    return X_test[selected], y_test[selected]


def load_dataset(
    name: str,
    *,
    drop_heloc_all_minus9: bool = False,
    validation_fraction: float = 0.0,
) -> DatasetBundle:
    """Compatibility delegate to the provider-neutral CEL adapter.

    Split is 80/20 stratified with random_state=42 (cel default).
    Scaling is fit on X_train only.

    When ``drop_heloc_all_minus9`` is enabled, completely unavailable HELOC
    bureau records are removed before splitting and scaling. Partial special
    codes are intentionally preserved for this controlled comparison.

    ``validation_fraction`` is the fraction split from the provisional 80%
    training partition with a second fixed, stratified split. A value of 0.2
    therefore produces one reproducible 64%/16%/20% train/validation/test split.
    """
    adapter = CelDatasetProvider().prepare_adapter(
        DatasetSpec(
            name=name,
            validation_fraction=validation_fraction,
            drop_heloc_all_minus9=drop_heloc_all_minus9,
        )
    )
    return dataset_bundle_from_adapter(name, adapter)


def dataset_bundle_from_adapter(name: str, adapter) -> DatasetBundle:
    """Build the historical live dataset view from one prepared CEL adapter."""
    prepared = adapter.prepared
    has_validation = bool(len(prepared.X_validation))
    return DatasetBundle(
        name=name,
        X_train=prepared.X_train,
        X_test=prepared.X_test,
        y_train=prepared.y_train,
        y_test=prepared.y_test,
        feature_names=list(prepared.schema.names),
        numerical_features_indices=list(adapter.numerical_features_indices),
        categorical_features_indices=list(adapter.categorical_features_indices),
        method_dataset=adapter.method_dataset,
        X_val=prepared.X_validation if has_validation else None,
        y_val=prepared.y_validation if has_validation else None,
        split_variant=adapter.split_variant,
        n_dropped_rows=adapter.n_dropped_rows,
        preprocessing_variant=adapter.preprocessing_variant,
        prepared=prepared,
    )


def get_actionable_immutable(
    name: str, dataset: DatasetBundle | None = None
) -> tuple[list[int], list[int]]:
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
        immutable_names: list[str] = cfg["immutable_features"]

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
