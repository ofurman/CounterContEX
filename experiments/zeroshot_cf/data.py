"""Dataset loading for the zero-shot CF experiment.

Loads HELOC and MOONS via cel, applies MinMax scaling (fit on train), and also
provides a small native-categorical synthetic dataset for the discrete greedy-CF
sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Tuple

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
    method_dataset: MethodDataset | "_IdentityDataset"  # for inverse_transform

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.method_dataset.inverse_transform(X)


class _IdentityDataset(Protocol):
    def inverse_transform(self, X: np.ndarray) -> np.ndarray: ...


class _IdentityMethodDataset:
    """Minimal MethodDataset stand-in for already-interpretable synthetic data."""

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)


def _load_binary_cat() -> DatasetBundle:
    """Return a deterministic all-categorical binary dataset.

    The label is exactly the first categorical feature. Every feature is a
    semantic categorical column encoded as stable integer codes 0/1, with no
    scaling and no one-hot expansion. The construction repeats the full binary
    cube so train and test both contain complete observed support.
    """
    rng = np.random.default_rng(42)
    cube = np.array(
        [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)],
        dtype=np.float64,
    )
    X = np.tile(cube, (80, 1))
    y = X[:, 0].astype(np.int64)

    perm = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx = perm[:split]
    test_idx = perm[split:]

    return DatasetBundle(
        name="binary_cat",
        X_train=X[train_idx].astype(np.float64),
        X_test=X[test_idx].astype(np.float64),
        y_train=y[train_idx],
        y_test=y[test_idx],
        feature_names=["decision_code", "segment_code", "channel_code"],
        numerical_features_indices=[],
        categorical_features_indices=[0, 1, 2],
        method_dataset=_IdentityMethodDataset(),
    )


def load_dataset(name: str) -> DatasetBundle:
    """Load a cel dataset by name ('heloc' or 'moons'), MinMax-scaled.

    Split is 80/20 stratified with random_state=42 (cel default).
    Scaling is fit on X_train only.
    """
    if name == "binary_cat":
        return _load_binary_cat()

    config_path = CEL_REPO / "config" / "datasets" / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    file_dataset = FileDataset(config_path=config_path)
    preprocessing = PreprocessingPipeline([
        ("minmax", MinMaxScalingStep()),
    ])
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
    )


def get_actionable_immutable(
    name: str, dataset: DatasetBundle | None = None
) -> Tuple[List[int], List[int]]:
    """Return (actionable_idx, immutable_idx) in the scaled feature matrix column order.

    For datasets with configs/<name>_actionability.yaml, use that generic
    actionability split. For 'moons': both features are actionable, no
    immutables.

    Args:
        name: Dataset name ('heloc' or 'moons').
        dataset: Optional pre-loaded DatasetBundle; if None the dataset is loaded
                 just to resolve feature names → column indices.

    Returns:
        Tuple of (actionable_column_indices, immutable_column_indices).
    """
    if name == "moons":
        if dataset is None:
            dataset = load_dataset("moons")
        n = len(dataset.feature_names)
        return list(range(n)), []

    cfg_path = CONFIGS_DIR / f"{name}_actionability.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        if dataset is None:
            dataset = load_dataset(name)

        feature_names = dataset.feature_names
        immutable_names: List[str] = cfg.get("immutable_features", [])
        unknown_immutable = sorted(set(immutable_names) - set(feature_names))
        if unknown_immutable:
            raise ValueError(
                f"Unknown immutable features in {cfg_path}: {unknown_immutable}"
            )
        immutable_idx = [feature_names.index(fn) for fn in immutable_names]
        actionable_names = cfg.get("actionable_features")
        if actionable_names is None:
            actionable_idx = [
                i for i in range(len(feature_names)) if i not in immutable_idx
            ]
        else:
            unknown_actionable = sorted(set(actionable_names) - set(feature_names))
            if unknown_actionable:
                raise ValueError(
                    f"Unknown actionable features in {cfg_path}: {unknown_actionable}"
                )
            actionable_idx = [feature_names.index(fn) for fn in actionable_names]
        return actionable_idx, immutable_idx

    raise ValueError(
        f"Unknown dataset/actionability split: {name!r}. Add "
        f"{cfg_path.name} or use 'moons'."
    )
