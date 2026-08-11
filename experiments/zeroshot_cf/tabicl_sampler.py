#  Copyright (c) Prior Labs GmbH 2026.

"""TabICL conditional-density adapter for greedy counterfactual generation.

The adapter preserves the existing Y-as-an-appended-column construction while
using :class:`tabicl.TabICLUnsupervised` for masked feature imputation.  Its
``sample_candidates`` method expands all candidate interventions for one
factual point and imputes them in one call, which is the principal fast path
used by the TabICL runner.

Only the appended label column is categorical. The experiment data is already
MinMax-scaled, so auto-detecting low-cardinality scaled columns and casting them
to integer class labels would destroy their original support.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.tabicl_checkpoints import require_checkpoints

ModelFactory = Callable[..., Any]


def _knn_indices(X: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Return sorted indices of the k Euclidean-nearest context rows."""
    q = np.asarray(query, dtype=X.dtype).reshape(-1)
    diff = X - q[None, :]
    dist2 = np.einsum("ij,ij->i", diff, diff)
    nearest = np.argpartition(dist2, k - 1)[:k]
    return np.sort(nearest)


def _select_context(
    X_context: np.ndarray,
    y_context: np.ndarray | None,
    *,
    target_class: int | None,
    max_context: int | None,
    selection: str,
    query: np.ndarray | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply the same class filtering and random/kNN selection as TabPFN."""
    if selection not in {"random", "knn"}:
        raise ValueError(f"selection must be 'random' or 'knn', got {selection!r}")
    if selection == "knn" and query is None:
        raise ValueError("query is required when selection='knn'")
    if max_context is not None and max_context <= 0:
        raise ValueError("max_context must be positive when provided")

    X = np.asarray(X_context, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X_context must be 2D, got shape {X.shape}")
    y = None if y_context is None else np.asarray(y_context)
    if y is not None and len(y) != len(X):
        raise ValueError("X_context and y_context must contain the same rows")

    if target_class is not None:
        if y is None:
            raise ValueError("y_context required when target_class is given")
        keep = y == target_class
        X = X[keep]
        y = y[keep]

    if len(X) == 0:
        raise ValueError("context selection produced an empty context")

    if max_context is not None and len(X) > max_context:
        if selection == "knn":
            idx = _knn_indices(X, np.asarray(query), max_context)
        else:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(X), size=max_context, replace=False)
            idx.sort()
        X = X[idx]
        if y is not None:
            y = y[idx]

    return X, y


def _local_tabicl_model_factory(
    *,
    classifier_path: Path,
    regressor_path: Path,
    **kwargs: Any,
):
    """Build TabICLUnsupervised with separate explicit local checkpoints.

    TabICL 2.1.1 forwards one shared ``estimator_params`` mapping to both inner
    estimators, which cannot express two different ``model_path`` values. This
    narrow subclass only overrides initial shared-weight loading; downstream
    conditional estimators still use the upstream implementation unchanged.
    """
    try:
        from tabicl import TabICLClassifier, TabICLUnsupervised
    except ImportError as exc:
        raise RuntimeError(
            "TabICL is not installed. Install experiments/zeroshot_cf/requirements.txt."
        ) from exc

    class _LocalCheckpointTabICLUnsupervised(TabICLUnsupervised):
        def _load_shared_model(self, estimator_cls):
            path = (
                classifier_path if estimator_cls is TabICLClassifier else regressor_path
            )
            estimator_kwargs = {
                **self._estimator_kwargs,
                "model_path": path,
                "allow_auto_download": False,
            }
            estimator = estimator_cls(**estimator_kwargs)
            estimator._resolve_device()
            estimator._load_model()
            estimator.model_.to(estimator.device_)
            return estimator.model_

    return _LocalCheckpointTabICLUnsupervised(**kwargs)


class TabICLConditionalDensitySampler:
    """Class-conditional masked imputation backed by TabICL.

    Parameters intentionally mirror the subset of ``ConditionalDensitySampler``
    used by the winning ``prob_ascent`` greedy search. ``model_factory`` exists
    for lightweight unit tests; normal runs load the two repo-local checkpoints.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 4,
        temperature: float = 1e-9,
        random_state: int = 0,
        device: str | None = None,
        batch_size: int | None = 8,
        cache_dir: Path | None = None,
        estimator_params: dict[str, Any] | None = None,
        model_factory: ModelFactory | None = None,
        context_update: str = "replace",
    ) -> None:
        if context_update not in {"replace", "refit"}:
            raise ValueError(
                "context_update must be 'replace' or 'refit', "
                f"got {context_update!r}"
            )
        self.n_estimators = n_estimators
        self.temperature = temperature
        self.random_state = random_state
        self.device = None if device in {None, "auto"} else device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.estimator_params = dict(estimator_params or {})
        self._model_factory = model_factory
        self.context_update = context_update

        self.model: Any | None = None
        self._model_initialized = False
        self._n_original_features: int | None = None
        self._fitted = False
        self.selected_context_: np.ndarray | None = None
        self.selected_labels_: np.ndarray | None = None

    def _build_model(self, y_idx: int):
        kwargs = {
            "n_estimators": self.n_estimators,
            "categorical_features": [y_idx],
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "device": self.device,
            "estimator_params": self.estimator_params,
        }
        if self._model_factory is not None:
            return self._model_factory(**kwargs)

        classifier_path, regressor_path = require_checkpoints(self.cache_dir)
        return _local_tabicl_model_factory(
            classifier_path=classifier_path,
            regressor_path=regressor_path,
            **kwargs,
        )

    @staticmethod
    def _replace_fitted_context(model: Any, X_aug: np.ndarray, y_idx: int) -> None:
        """Replace query-specific context without reloading shared checkpoints.

        Upstream ``TabICLUnsupervised.fit`` stores these attributes and loads the
        same classifier/regressor weights. kNN changes only the stored context,
        so updating that fitted state avoids a checkpoint reload for every
        factual point.
        """
        model.X_ = np.asarray(X_aug, dtype=np.float32).copy()
        model.n_features_in_ = X_aug.shape[1]
        model.categorical_features_ = [y_idx]
        model.categories_ = {
            y_idx: np.unique(X_aug[:, y_idx][~np.isnan(X_aug[:, y_idx])]).astype(int)
        }
        model.numerical_features_ = [j for j in range(X_aug.shape[1]) if j != y_idx]

    def set_context(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray | None = None,
        target_class: int | None = None,
        max_context: int | None = None,
        selection: str = "random",
        query: np.ndarray | None = None,
    ) -> "TabICLConditionalDensitySampler":
        """Select context rows, append Y, and prepare TabICL for imputation."""
        X, y = _select_context(
            X_context,
            y_context,
            target_class=target_class,
            max_context=max_context,
            selection=selection,
            query=query,
            random_state=self.random_state,
        )
        if y is None:
            raise ValueError("y_context is required for class-conditional sampling")

        n_features = X.shape[1]
        if (
            self._n_original_features is not None
            and self._n_original_features != n_features
        ):
            raise ValueError(
                "The number of features cannot change between context updates"
            )
        self._n_original_features = n_features
        y_idx = n_features
        X_aug = np.column_stack([X, np.asarray(y, dtype=np.float32)]).astype(
            np.float32, copy=False
        )

        if self.model is None:
            self.model = self._build_model(y_idx)
        if not self._model_initialized or self.context_update == "refit":
            self.model.fit(X_aug)
            self._model_initialized = True
        else:
            self._replace_fitted_context(self.model, X_aug, y_idx)

        self.selected_context_ = X.copy()
        self.selected_labels_ = np.asarray(y).copy()
        self._fitted = True
        return self

    def _augmented_candidate_rows(
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        fixed_target: int,
    ) -> np.ndarray:
        if not self._fitted or self._n_original_features is None:
            raise RuntimeError("Call set_context() before sampling features.")

        X = np.asarray(X_query, dtype=np.float32)
        if X.ndim != 2 or X.shape != (1, self._n_original_features):
            raise ValueError(
                "candidate expansion expects one query row with shape "
                f"(1, {self._n_original_features}), got {X.shape}"
            )
        candidates = np.asarray(candidate_cols, dtype=int)
        if candidates.ndim != 1 or len(candidates) == 0:
            raise ValueError("candidate_cols must be a non-empty 1D sequence")
        if len(np.unique(candidates)) != len(candidates):
            raise ValueError("candidate_cols must not contain duplicates")
        if np.any(candidates < 0) or np.any(candidates >= self._n_original_features):
            raise IndexError("candidate feature index is out of bounds")

        rows = np.repeat(X, len(candidates), axis=0)
        rows[np.arange(len(candidates)), candidates] = np.nan
        target = np.full((len(rows), 1), float(fixed_target), dtype=np.float32)
        return np.concatenate([rows, target], axis=1)

    def sample_candidates(
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        sample_temperature: float | None = None,
        fixed_target: int,
    ) -> np.ndarray:
        """Impute every candidate intervention for one point in one model call."""
        candidates = np.asarray(candidate_cols, dtype=int)
        X_aug = self._augmented_candidate_rows(X_query, candidates, fixed_target)
        temperature = (
            self.temperature if sample_temperature is None else sample_temperature
        )
        filled = np.asarray(
            self.model.impute(
                X_aug,
                temperature=float(temperature),
                n_iterations=1,
            )
        )
        return filled[np.arange(len(candidates)), candidates].astype(np.float64)

    def sample_feature(
        self,
        X_query: np.ndarray,
        target_col: int,
        n_samples: int = 1,
        sample_temperature: float | None = None,
        fixed_target: int | None = None,
    ) -> np.ndarray:
        """Compatibility method matching the existing greedy sampler contract."""
        if fixed_target is None:
            raise ValueError("fixed_target is required for TabICL sampling")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        if n_samples == 1:
            return self.sample_candidates(
                X_query,
                [target_col],
                sample_temperature=sample_temperature,
                fixed_target=fixed_target,
            )

        original_state = self.model.random_state
        draws = []
        try:
            for offset in range(n_samples):
                self.model.random_state = self.random_state + offset
                draws.append(
                    self.sample_candidates(
                        X_query,
                        [target_col],
                        sample_temperature=sample_temperature,
                        fixed_target=fixed_target,
                    )
                )
        finally:
            self.model.random_state = original_state
        return np.stack(draws, axis=0)

    def impute_masked(
        self,
        X_query: np.ndarray,
        mask_cols: Sequence[int],
        fixed_target: int | None = None,
        dag: dict[int, list[int]] | None = None,
    ) -> np.ndarray:
        """Compatibility path for masking the same columns in every query row."""
        if not self._fitted or self._n_original_features is None:
            raise RuntimeError("Call set_context() before impute_masked().")
        if fixed_target is None:
            raise ValueError("fixed_target is required for TabICL imputation")
        if dag is not None:
            raise NotImplementedError(
                "TabICLUnsupervised does not expose DAG imputation"
            )

        original = np.asarray(X_query, dtype=np.float64)
        X = np.asarray(X_query, dtype=np.float32).copy()
        X[:, list(mask_cols)] = np.nan
        target = np.full((len(X), 1), float(fixed_target), dtype=np.float32)
        X_aug = np.concatenate([X, target], axis=1)
        filled = np.asarray(
            self.model.impute(
                X_aug,
                temperature=float(self.temperature),
                n_iterations=1,
            )
        )[:, : self._n_original_features].astype(np.float64)

        masked = set(mask_cols)
        observed = [j for j in range(original.shape[1]) if j not in masked]
        filled[:, observed] = original[:, observed]
        return filled
