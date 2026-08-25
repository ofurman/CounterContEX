"""Validity oracle discriminator for the zero-shot CF experiment.

Uses sklearn LogisticRegression (binary) rather than cel's PyTorch LR.
The sklearn model is simpler to train (no DataLoader/epoch loop) and works
directly with float64 numpy arrays. Wrapped to expose the cel metrics
contract: `.predict(X_np) -> array` and `.eval()` (no-op).

Decision (recorded): sklearn LR chosen over cel's PyTorch LR because:
  - No DataLoader/epochs boilerplate needed.
  - Works directly with float64 numpy arrays.
  - Equivalent validation accuracy for this task.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression as _SklearnLR
from sklearn.neural_network import MLPClassifier as _SklearnMLP

MODELS_DIR = Path(__file__).parent / "models"


class DiscriminatorModel:
    """Sklearn classifier wrapped to satisfy the cel metrics contract."""

    def __init__(self, clf) -> None:
        self._clf = clf

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)

    def eval(self) -> "DiscriminatorModel":
        return self


def train_discriminator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str,
    disc_type: Literal["lr", "mlp"] = "lr",
    force_retrain: bool = False,
) -> DiscriminatorModel:
    """Train (or load from cache) a validity oracle for the given dataset.

    Args:
        X_train: Scaled training features.
        y_train: Training labels (int).
        X_test: Scaled test features.
        y_test: Test labels (int).
        dataset_name: Name tag used for the cache file.
        disc_type: 'lr' for logistic regression (default), 'mlp' for MLP.
        force_retrain: If True, retrain even if a cached model exists.

    Returns:
        Trained DiscriminatorModel.
    """
    models_dir = Path(os.environ.get("ZEROSHOT_CF_MODELS_DIR", str(MODELS_DIR)))
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_path = models_dir / f"disc_{dataset_name}_{disc_type}.pkl"

    if cache_path.exists() and not force_retrain:
        print(f"[discriminator] Loading cached model from {cache_path}")
        return joblib.load(cache_path)

    if disc_type == "lr":
        clf = _SklearnLR(max_iter=1000, random_state=42, C=1.0)
    elif disc_type == "mlp":
        clf = _SklearnMLP(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
    else:
        raise ValueError(f"Unknown disc_type: {disc_type!r}")

    print(
        f"[discriminator] Training {disc_type.upper()} on {dataset_name} "
        f"({len(X_train)} train, {len(X_test)} test) ..."
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"[discriminator]  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    model = DiscriminatorModel(clf)
    joblib.dump(model, cache_path)
    print(f"[discriminator] Saved to {cache_path}")
    return model
