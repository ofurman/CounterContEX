"""Loader for CETGFN-derived local datasets.

Datasets ported from ../CETGFN (CounterFlowNet) live under
``experiments/zeroshot_cf/datasets/<name>/`` as ``config.json``, ``train.csv``,
``val.csv`` (unused here), ``test.csv``, and the classifier checkpoints
``model.pt`` / ``flow.pt`` (``model.pkl`` too, where CETGFN produced one). The
CSVs and checkpoints are gitignored (see .gitignore); only ``config.json`` is
tracked.

Two feature-encoding modes, matching how CETGFN itself preprocesses each
dataset (``rgfn/gfns/counterfactual_gfn/preprocessing/{l2c,discretizer}.py``):

- ``DISCRETIZED_DATASETS`` (german, adult, admission, student): numerical
  columns are binned into ordinal codes using ``config.json``'s
  ``numerical_bins`` (the same fixed, right-closed bin edges rgfn's
  ``L2CDiscretizer`` uses), then MinMax-scaled to [0, 1] like every other
  feature.
- All other ported datasets: numerical columns are left continuous and only
  MinMax-scaled. ``numerical_bins`` was stripped from their ``config.json`` at
  port time, so there is no binarisation step for these.

Categorical columns are ordinal-encoded (fit on train) in both modes, then
MinMax-scaled the same way, so the final ``X`` is a single float64 matrix —
consistent with the cel-backed datasets in ``data.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

LOCAL_DATASETS_DIR = Path(__file__).parent / "datasets"

DISCRETIZED_DATASETS = frozenset({"german", "adult", "admission", "student"})

LOCAL_DATASET_NAMES = frozenset(
    {
        "adult",
        "adult_dicoflex",
        "bank",
        "default",
        "german",
        "gmc",
        "lending-club",
        "sba",
        "student",
        "admission",
    }
)


def _bin_index(value: float, edges: List[Tuple[float, float]]) -> int:
    """Right-closed interval lookup — matches rgfn's L2CDiscretizer._map_interval."""
    if value <= edges[0][0]:
        return 0
    if value > edges[-1][1]:
        return len(edges) - 1
    for i, (left, right) in enumerate(edges):
        if left < value <= right:
            return i
    return len(edges) - 1


@dataclass
class _ColumnCodec:
    """Encodes one raw CSV column to a [0, 1] float and back."""

    kind: str  # "numeric" | "discretized" | "categorical"
    lo: float
    hi: float
    bin_edges: List[Tuple[float, float]] = field(default_factory=list)
    bin_centers: List[float] = field(default_factory=list)
    encoder: OrdinalEncoder | None = None

    def encode(self, raw: pd.Series) -> np.ndarray:
        if self.kind == "categorical":
            codes = self.encoder.transform(raw.to_frame()).ravel()
        elif self.kind == "discretized":
            codes = np.array(
                [_bin_index(v, self.bin_edges) for v in raw.to_numpy(dtype=float)],
                dtype=float,
            )
        else:
            codes = raw.to_numpy(dtype=float)
        span = self.hi - self.lo
        return (codes - self.lo) / span if span > 0 else np.zeros_like(codes)

    def decode(self, scaled: np.ndarray) -> np.ndarray:
        """Undo MinMax scaling; discretized/categorical columns land on a
        representative numeric value (bin center / ordinal code), not the
        original label — matching the all-float contract DatasetBundle
        callers (e.g. exp4's write_examples) expect from inverse_transform.
        """
        codes = scaled * (self.hi - self.lo) + self.lo
        if self.kind == "discretized":
            idx = np.clip(np.rint(codes), 0, len(self.bin_centers) - 1).astype(int)
            return np.asarray(self.bin_centers, dtype=float)[idx]
        return codes


class _LocalInverseTransformer:
    """Duck-types cel's MethodDataset.inverse_transform(X) for DatasetBundle."""

    def __init__(self, codecs: List[_ColumnCodec]) -> None:
        self._codecs = codecs

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        out = np.empty_like(X, dtype=float)
        for j, codec in enumerate(self._codecs):
            out[:, j] = codec.decode(X[:, j])
        return out


def load_local_dataset(name: str):
    """Load a CETGFN-ported dataset by name. Returns a data.DatasetBundle.

    Uses ``train.csv`` / ``test.csv`` (``val.csv`` is not folded in — the rest
    of this experiment only consumes a train/test split). Categorical columns
    are ordinal-encoded and numerical columns are MinMax-scaled to [0, 1]
    (binned first for ``DISCRETIZED_DATASETS``), fit on the train split only.
    """
    from experiments.zeroshot_cf.data import DatasetBundle  # local import: avoid cycle

    if name not in LOCAL_DATASET_NAMES:
        raise FileNotFoundError(f"No local dataset {name!r} under {LOCAL_DATASETS_DIR}")

    ds_dir = LOCAL_DATASETS_DIR / name
    cfg = json.loads((ds_dir / "config.json").read_text())
    target_col = cfg["target_column"]
    numerical_cols = cfg["numerical_columns"]
    categorical_cols = cfg["categorical_columns"]
    numerical_bins = cfg.get("numerical_bins", {})
    discretize = name in DISCRETIZED_DATASETS

    train_df = pd.read_csv(ds_dir / "train.csv")
    test_df = pd.read_csv(ds_dir / "test.csv")

    feature_names = [c for c in train_df.columns if c != target_col]

    codecs: List[_ColumnCodec] = []
    X_train_cols, X_test_cols = [], []
    numerical_idx, categorical_idx = [], []

    for j, col in enumerate(feature_names):
        if col in categorical_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(train_df[[col]])
            codec = _ColumnCodec(
                kind="categorical", lo=0.0, hi=float(len(enc.categories_[0]) - 1), encoder=enc
            )
            categorical_idx.append(j)
        elif discretize and col in numerical_bins:
            edges = [tuple(e) for e in numerical_bins[col]]
            centers = [(lo + hi) / 2 for lo, hi in edges]
            codec = _ColumnCodec(
                kind="discretized",
                lo=0.0,
                hi=float(len(edges) - 1),
                bin_edges=edges,
                bin_centers=centers,
            )
            numerical_idx.append(j)
        else:
            train_vals = train_df[col].to_numpy(dtype=float)
            codec = _ColumnCodec(
                kind="numeric", lo=float(train_vals.min()), hi=float(train_vals.max())
            )
            numerical_idx.append(j)

        codecs.append(codec)
        X_train_cols.append(codec.encode(train_df[col]))
        X_test_cols.append(codec.encode(test_df[col]))

    X_train = np.column_stack(X_train_cols).astype(np.float64)
    X_test = np.column_stack(X_test_cols).astype(np.float64)

    target_enc = LabelEncoder()
    y_train = target_enc.fit_transform(train_df[target_col].astype(str)).astype(np.int64)
    y_test = target_enc.transform(test_df[target_col].astype(str)).astype(np.int64)

    return DatasetBundle(
        name=name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        numerical_features_indices=numerical_idx,
        categorical_features_indices=categorical_idx,
        method_dataset=_LocalInverseTransformer(codecs),
    )


def get_local_actionable_immutable(name: str, feature_names: List[str]):
    """Return (actionable_idx, immutable_idx) from config.json's `immutable` list."""
    cfg = json.loads((LOCAL_DATASETS_DIR / name / "config.json").read_text())
    immutable_names = cfg.get("immutable", [])
    immutable_idx = [feature_names.index(n) for n in immutable_names]
    actionable_idx = [i for i in range(len(feature_names)) if i not in immutable_idx]
    return actionable_idx, immutable_idx
