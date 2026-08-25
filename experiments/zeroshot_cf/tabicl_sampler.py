#  Copyright (c) Prior Labs GmbH 2026.

"""TabICL conditional-density adapter for greedy counterfactual generation.

The adapter preserves the existing Y-as-an-appended-column construction while
using :class:`tabicl.TabICLUnsupervised` for masked feature imputation.  Its
``sample_candidates`` method expands all candidate interventions for one
factual point and imputes them in one call, which is the principal fast path
used by the TabICL runner.

By default only the appended label column is categorical. Callers using a
compact mixed-data representation may explicitly identify additional
categorical columns; auto-detection remains disabled because casting arbitrary
low-cardinality scaled columns to class labels would destroy their support.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, overload
from typing_extensions import override

import numpy as np
import torch
from experiments.zeroshot_cf.mixed_distance import compact_gower_distance
from experiments.zeroshot_cf.tabicl_checkpoints import require_checkpoints
from tabicl import TabICLClassifier, TabICLRegressor
from tabicl._model.quantile_dist import QuantileDistribution
from tabicl._sklearn.preprocessing import Shuffler
from tabicl._unsupervised.unsupervised import TabICLUnsupervised

ModelFactory = Callable[..., Any]


def quantile_mode(dist: Any) -> np.ndarray:
    """Return the mode of TabICL's interior piecewise-quantile density.

    Density within each interpolated quantile interval is the inverse of that
    interval's slope. Select the midpoint of the densest interval, breaking
    ties toward the median. Restricting the estimate to the predicted quantile
    knots avoids treating extrapolated tail-boundary artefacts as modes. This
    requires no additional foundation-model forward pass.
    """
    with torch.no_grad():
        slopes = dist.slopes
        valid = torch.isfinite(slopes) & (slopes >= 0)
        log_density = torch.where(
            valid,
            -torch.log(torch.clamp(slopes, min=dist.tol)),
            torch.full_like(slopes, -torch.inf),
        )
        max_density = log_density.max(dim=-1, keepdim=True).values
        tied = torch.isclose(log_density, max_density, rtol=1e-5, atol=1e-7)
        interval_alpha = (dist.alpha_lo + dist.alpha_hi) / 2
        distance_from_median = (interval_alpha - 0.5).abs()
        tie_distance = torch.where(
            tied,
            distance_from_median,
            torch.full_like(log_density, torch.inf),
        )
        best = tie_distance.argmin(dim=-1)
        interval_value = (dist.q_lo + dist.q_hi) / 2
        mode = interval_value.gather(-1, best.unsqueeze(-1)).squeeze(-1)

        finite = torch.isfinite(max_density.squeeze(-1))
        median_alpha = torch.tensor(
            0.5,
            device=dist.quantiles.device,
            dtype=dist.quantiles.dtype,
        )
        median = dist.icdf(median_alpha)
        mode = torch.where(finite, mode, median)
    return mode.cpu().numpy()


def _knn_indices(
    X: np.ndarray,
    query: np.ndarray,
    k: int,
    categorical_features: Sequence[int] = (),
) -> np.ndarray:
    """Return sorted indices of the k Gower-nearest context rows."""
    distances = compact_gower_distance(X, query, categorical_features)
    nearest = np.argpartition(distances, k - 1)[:k]
    return np.sort(nearest)


@overload
def _select_context(
    X_context: np.ndarray,
    y_context: np.ndarray | None,
    *,
    target_class: int | None,
    max_context: int | None,
    selection: str,
    query: np.ndarray | None,
    random_state: int,
    categorical_features: Sequence[int] = (),
    return_indices: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray | None]: ...


@overload
def _select_context(
    X_context: np.ndarray,
    y_context: np.ndarray | None,
    *,
    target_class: int | None,
    max_context: int | None,
    selection: str,
    query: np.ndarray | None,
    random_state: int,
    categorical_features: Sequence[int] = (),
    return_indices: Literal[True],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]: ...


def _select_context(
    X_context: np.ndarray,
    y_context: np.ndarray | None,
    *,
    target_class: int | None,
    max_context: int | None,
    selection: str,
    query: np.ndarray | None,
    random_state: int,
    categorical_features: Sequence[int] = (),
    return_indices: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray | None]
    | tuple[np.ndarray, np.ndarray | None, np.ndarray]
):
    """Apply the same class filtering and random/kNN selection as TabPFN."""
    if selection not in {"random", "knn"}:
        raise ValueError(f"selection must be 'random' or 'knn', got {selection!r}")
    if selection == "knn" and query is None:
        raise ValueError("query is required when selection='knn'")
    if max_context is not None and max_context <= 0:
        raise ValueError("max_context must be positive when provided")

    X = np.asarray(X_context, dtype=np.float32)
    selected_indices = np.arange(len(X))
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
        selected_indices = selected_indices[keep]

    if len(X) == 0:
        raise ValueError("context selection produced an empty context")

    if max_context is not None and len(X) > max_context:
        if selection == "knn":
            idx = _knn_indices(
                X,
                np.asarray(query),
                max_context,
                categorical_features,
            )
        else:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(X), size=max_context, replace=False)
            idx.sort()
        X = X[idx]
        selected_indices = selected_indices[idx]
        if y is not None:
            y = y[idx]

    if return_indices:
        return X, y, selected_indices
    return X, y


def _local_tabicl_model_factory(
    *,
    classifier_path: Path,
    regressor_path: Path,
    numerical_point_estimate: str,
    **kwargs: Any,
):
    """Build TabICLUnsupervised with separate explicit local checkpoints.

    TabICL 2.1.1 forwards one shared ``estimator_params`` mapping to both inner
    estimators, which cannot express two different ``model_path`` values. This
    narrow subclass keeps the upstream estimator implementation while adding
    explicit local loading and per-context full-conditional memoization.
    """

    class _LocalCheckpointTabICLUnsupervised(TabICLUnsupervised):
        _numerical_quantile_grid: ClassVar[np.ndarray | None] = None
        _conditional_estimator_cache: (
            dict[
                tuple[int, tuple[int, ...], bytes, bytes],
                tuple[TabICLClassifier | TabICLRegressor, bool],
            ]
            | None
        ) = None

        @override
        def _load_shared_model(
            self,
            estimator_cls: type[TabICLClassifier] | type[TabICLRegressor],
        ) -> torch.nn.Module:
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

        @staticmethod
        @override
        def _sample_numerical(
            dist: QuantileDistribution,
            temperature: float,
            rng: np.random.Generator,
        ) -> np.ndarray:
            quantile_grid = _LocalCheckpointTabICLUnsupervised._numerical_quantile_grid
            if quantile_grid is not None:
                batch_rows = dist.quantiles.shape[0]
                if batch_rows % len(quantile_grid) != 0:
                    raise ValueError(
                        "the numerical distribution batch must contain a "
                        "whole number of quantile grids"
                    )
                quantile_grid = np.tile(
                    quantile_grid,
                    batch_rows // len(quantile_grid),
                )
                alphas = torch.as_tensor(
                    quantile_grid,
                    device=dist.quantiles.device,
                    dtype=dist.quantiles.dtype,
                ).unsqueeze(-1)
                return dist.icdf(alphas).squeeze(-1).cpu().numpy()
            if numerical_point_estimate == "mode" and temperature <= 1e-8:
                return quantile_mode(dist)
            return TabICLUnsupervised._sample_numerical(dist, temperature, rng)

        @override
        def _fit_conditional_estimator(
            self,
            col_idx: int,
            X_train: np.ndarray,
            y_train: np.ndarray,
        ) -> tuple[TabICLClassifier | TabICLRegressor, bool]:
            """Reuse a fitted full-conditional estimator within one context."""
            cache = getattr(self, "_conditional_estimator_cache", None)
            if cache is None:
                return super()._fit_conditional_estimator(
                    col_idx,
                    X_train,
                    y_train,
                )
            key = (
                int(col_idx),
                X_train.shape,
                np.ascontiguousarray(X_train).tobytes(),
                np.ascontiguousarray(y_train).tobytes(),
            )
            if key not in cache:
                cache[key] = super()._fit_conditional_estimator(
                    col_idx,
                    X_train,
                    y_train,
                )
            return cache[key]

        def log_score_samples(
            self,
            X: np.ndarray,
            n_permutations: int = 1,
        ) -> np.ndarray:
            """Return TabICL's chain-rule joint log-density without exponentiating."""
            rows = np.asarray(X, dtype=np.float32)
            if rows.ndim != 2 or rows.shape[1] != self.n_features_in_:
                raise ValueError(
                    "rows must be a 2D matrix with the fitted feature count"
                )
            if n_permutations < 1:
                raise ValueError("n_permutations must be positive")
            rng = np.random.default_rng(self.random_state)
            permutations = Shuffler(
                self.n_features_in_,
                random_state=self.random_state,
            ).shuffle(n_permutations)
            log_densities = [
                self._compute_log_density(rows, permutation, rng)
                for permutation in permutations
            ]
            return np.mean(log_densities, axis=0)

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
        numerical_point_estimate: str = "median",
        categorical_features: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if context_update not in {"replace", "refit"}:
            raise ValueError(
                f"context_update must be 'replace' or 'refit', got {context_update!r}"
            )
        if numerical_point_estimate not in {"median", "mode"}:
            raise ValueError(
                "numerical_point_estimate must be 'median' or 'mode', "
                f"got {numerical_point_estimate!r}"
            )
        self.n_estimators = n_estimators
        self.temperature = temperature
        self.random_state = random_state
        self.device = None if device in {None, "auto"} else device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.estimator_params = dict(estimator_params or {})
        self.estimator_params.setdefault("kv_cache", True)
        self._model_factory = model_factory
        self.context_update = context_update
        self.numerical_point_estimate = numerical_point_estimate
        self.categorical_features = tuple(int(j) for j in (categorical_features or ()))
        if len(set(self.categorical_features)) != len(self.categorical_features):
            raise ValueError("categorical_features must be unique")
        if any(j < 0 for j in self.categorical_features):
            raise ValueError("categorical_features must be non-negative")

        self.model: Any | None = None
        self._model_initialized = False
        self._n_original_features: int | None = None
        self._fitted = False
        self.selected_context_: np.ndarray | None = None
        self.selected_labels_: np.ndarray | None = None
        self.selected_confidences_: np.ndarray | None = None
        self._uses_confidence = False

    def _require_model(self) -> Any:
        """Return the initialized backend model or raise a lifecycle error."""
        if self.model is None:
            raise RuntimeError("Call set_context() before using the TabICL model.")
        return self.model

    def _build_model(self, y_idx: int):
        if any(j >= y_idx for j in self.categorical_features):
            raise IndexError("categorical feature index is out of bounds")
        kwargs = {
            "n_estimators": self.n_estimators,
            "categorical_features": [*self.categorical_features, y_idx],
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
            numerical_point_estimate=self.numerical_point_estimate,
            **kwargs,
        )

    @staticmethod
    def _replace_fitted_context(
        model: Any,
        X_aug: np.ndarray,
        categorical_features: Sequence[int],
        y_idx: int,
    ) -> None:
        """Replace query-specific context without reloading shared checkpoints.

        Upstream ``TabICLUnsupervised.fit`` stores these attributes and loads the
        same classifier/regressor weights. kNN changes only the stored context,
        so updating that fitted state avoids a checkpoint reload for every
        factual point.
        """
        model.X_ = np.asarray(X_aug, dtype=np.float32).copy()
        model.n_features_in_ = X_aug.shape[1]
        model.categorical_features_ = [*categorical_features, y_idx]
        model.categories_ = {
            j: np.unique(X_aug[:, j][~np.isnan(X_aug[:, j])]).astype(int)
            for j in model.categorical_features_
        }
        model.numerical_features_ = [
            j for j in range(X_aug.shape[1]) if j not in model.categorical_features_
        ]

    def set_context(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray | None = None,
        confidence_context: np.ndarray | None = None,
        target_class: int | None = None,
        max_context: int | None = None,
        selection: str = "random",
        query: np.ndarray | None = None,
    ) -> "TabICLConditionalDensitySampler":
        """Select context rows, append Y, and prepare TabICL for imputation."""
        X, y, selected_indices = _select_context(
            X_context,
            y_context,
            target_class=target_class,
            max_context=max_context,
            selection=selection,
            query=query,
            random_state=self.random_state,
            categorical_features=self.categorical_features,
            return_indices=True,
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
        confidence = None
        if confidence_context is not None:
            confidence_all = np.asarray(confidence_context, dtype=np.float32)
            if confidence_all.ndim != 1 or len(confidence_all) != len(X_context):
                raise ValueError(
                    "confidence_context must be a 1D array aligned with X_context"
                )
            if not np.all(np.isfinite(confidence_all)):
                raise ValueError("confidence_context must contain only finite values")
            confidence = confidence_all[selected_indices]

        uses_confidence = confidence is not None
        if self._model_initialized and uses_confidence != self._uses_confidence:
            raise ValueError(
                "confidence conditioning cannot be enabled or disabled after "
                "the TabICL model has been initialized"
            )
        self._uses_confidence = uses_confidence

        columns = [X, np.asarray(y, dtype=np.float32)]
        if confidence is not None:
            columns.append(confidence)
        X_aug = np.column_stack(columns).astype(np.float32, copy=False)

        if self.model is None:
            self.model = self._build_model(y_idx)
        model = self._require_model()
        if not self._model_initialized or self.context_update == "refit":
            model.fit(X_aug)
            self._model_initialized = True
        else:
            self._replace_fitted_context(
                model,
                X_aug,
                self.categorical_features,
                y_idx,
            )
        # Conditional estimators and their TabICL KV representations are valid
        # only for this factual's selected context. Reuse them across greedy
        # iterations, then discard them when the next context is installed.
        model._conditional_estimator_cache = {}

        self.selected_context_ = X.copy()
        self.selected_labels_ = np.asarray(y).copy()
        self.selected_confidences_ = (
            None if confidence is None else np.asarray(confidence).copy()
        )
        self._fitted = True
        return self

    def _augmented_candidate_rows(  # noqa: C901, PLR0912
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        fixed_target: int,
        *,
        allow_duplicate_cols: bool = False,
        fixed_confidence: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._fitted or self._n_original_features is None:
            raise RuntimeError("Call set_context() before sampling features.")

        X = np.asarray(X_query, dtype=np.float32)
        if X.ndim != 2 or X.shape[1:] != (self._n_original_features,):
            raise ValueError(
                "candidate expansion expects query rows with shape "
                f"(n, {self._n_original_features}), got {X.shape}"
            )
        candidates = np.asarray(candidate_cols, dtype=int)
        if candidates.ndim != 1 or len(candidates) == 0:
            raise ValueError("candidate_cols must be a non-empty 1D sequence")
        if not allow_duplicate_cols and len(np.unique(candidates)) != len(candidates):
            raise ValueError("candidate_cols must not contain duplicates")
        if np.any(candidates < 0) or np.any(candidates >= self._n_original_features):
            raise IndexError("candidate feature index is out of bounds")

        if len(X) == 1:
            rows = np.repeat(X, len(candidates), axis=0)
        elif len(X) == len(candidates):
            rows = X.copy()
        else:
            raise ValueError(
                "provide either one shared query row or one query row per "
                "candidate column"
            )
        rows[np.arange(len(candidates)), candidates] = np.nan
        target = np.full((len(rows), 1), float(fixed_target), dtype=np.float32)
        augmented = [rows, target]
        if self._uses_confidence:
            if fixed_confidence is None:
                raise ValueError(
                    "fixed_confidence is required when confidence conditioning "
                    "is enabled"
                )
            confidence = np.asarray(fixed_confidence, dtype=np.float32)
            if confidence.ndim == 0:
                confidence = np.full(len(rows), float(confidence), dtype=np.float32)
            if confidence.shape != (len(rows),):
                raise ValueError(
                    "fixed_confidence must be scalar or contain one value per "
                    "candidate row"
                )
            augmented.append(confidence.reshape(-1, 1))
        elif fixed_confidence is not None:
            raise ValueError(
                "fixed_confidence requires confidence_context in set_context()"
            )
        return np.concatenate(augmented, axis=1)

    def score_joint_rows(
        self,
        X_rows: np.ndarray,
        *,
        fixed_target: int,
        fixed_confidence: float | Sequence[float] | np.ndarray | None = None,
        n_permutations: int = 1,
    ) -> np.ndarray:
        """Score complete rows with TabICL's augmented joint log-density.

        The fitted context contains ``[X, Y]`` and, when enabled, the target
        classifier confidence. Candidate rows are augmented with the requested
        target class and their actual target-class confidence before TabICL's
        built-in chain-rule density is evaluated.
        """
        if not self._fitted or self._n_original_features is None or self.model is None:
            raise RuntimeError("Call set_context() before scoring rows.")
        rows = np.asarray(X_rows, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != self._n_original_features:
            raise ValueError(
                "row scoring expects a 2D matrix with shape "
                f"(n, {self._n_original_features}), got {rows.shape}"
            )
        target = np.full((len(rows), 1), float(fixed_target), dtype=np.float32)
        augmented = [rows, target]
        if self._uses_confidence:
            if fixed_confidence is None:
                raise ValueError(
                    "fixed_confidence is required when confidence conditioning "
                    "is enabled"
                )
            confidence = np.asarray(fixed_confidence, dtype=np.float32)
            if confidence.ndim == 0:
                confidence = np.full(len(rows), float(confidence), dtype=np.float32)
            if confidence.shape != (len(rows),):
                raise ValueError(
                    "fixed_confidence must be scalar or contain one value per row"
                )
            augmented.append(confidence.reshape(-1, 1))
        complete_rows = np.concatenate(augmented, axis=1)
        return np.asarray(
            self.model.log_score_samples(
                complete_rows,
                n_permutations=n_permutations,
            ),
            dtype=np.float64,
        )

    def sample_candidates(
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        sample_temperature: float | None = None,
        fixed_target: int,
        fixed_confidence: float | None = None,
    ) -> np.ndarray:
        """Impute every candidate intervention for one point in one model call."""
        candidates = np.asarray(candidate_cols, dtype=int)
        X_aug = self._augmented_candidate_rows(
            X_query,
            candidates.tolist(),
            fixed_target,
            fixed_confidence=fixed_confidence,
        )
        temperature = (
            self.temperature if sample_temperature is None else sample_temperature
        )
        model = self._require_model()
        filled = np.asarray(
            model.impute(
                X_aug,
                temperature=float(temperature),
                n_iterations=1,
            )
        )
        return filled[np.arange(len(candidates)), candidates].astype(np.float64)

    def sample_candidates_batch(
        self,
        X_queries: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        sample_temperature: float | None = None,
        fixed_target: int,
        fixed_confidence: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Impute one candidate column in each query row in one model call."""
        candidates = np.asarray(candidate_cols, dtype=int)
        queries = np.asarray(X_queries)
        if queries.ndim != 2 or len(queries) != len(candidates):
            raise ValueError("X_queries must contain one row per candidate column")
        X_aug = self._augmented_candidate_rows(
            queries,
            candidates.tolist(),
            fixed_target,
            allow_duplicate_cols=True,
            fixed_confidence=fixed_confidence,
        )
        temperature = (
            self.temperature if sample_temperature is None else sample_temperature
        )
        filled = np.asarray(
            self._require_model().impute(
                X_aug,
                temperature=float(temperature),
                n_iterations=1,
            )
        )
        return filled[np.arange(len(candidates)), candidates].astype(np.float64)

    def categorical_distribution(
        self,
        X_query: np.ndarray,
        target_col: int,
        *,
        fixed_target: int,
        fixed_confidence: float | Sequence[float] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict ``p(X[target_col] | X[-target_col], target)`` with TabICL.

        This is the categorical counterpart of the numerical quantile call. It
        returns the complete learned category support and probabilities rather
        than drawing one category, allowing the counterfactual search to retain
        full coverage while recording TabICL's conditional preference. A
        confidence sequence is evaluated as one prediction batch and returns a
        probability matrix with one row per confidence value.
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Call set_context() before predicting categories.")
        if target_col not in self.categorical_features:
            raise ValueError(f"column {target_col} is not categorical")

        confidence = (
            None
            if fixed_confidence is None
            else np.asarray(fixed_confidence, dtype=np.float32)
        )
        is_batched = confidence is not None and confidence.ndim == 1
        n_queries = len(confidence) if confidence is not None and is_batched else 1
        X_aug = self._augmented_candidate_rows(
            X_query,
            [target_col] * n_queries,
            fixed_target,
            allow_duplicate_cols=n_queries > 1,
            fixed_confidence=fixed_confidence,
        )
        model = self._require_model()
        train_mask = ~np.isnan(model.X_[:, target_col])
        conditioning = [j for j in range(model.n_features_in_) if j != target_col]
        rng = np.random.default_rng(self.random_state)
        X_train_cond, y_train_cond, X_test_cond = model._prepare_conditional_data(
            tgt_idx=target_col,
            cond_features=conditioning,
            train_mask=train_mask,
            X_test=X_aug,
            rng=rng,
        )
        estimator, is_categorical = model._fit_conditional_estimator(
            target_col,
            X_train_cond,
            y_train_cond,
        )
        if not is_categorical:
            raise RuntimeError(f"column {target_col} was not routed as categorical")
        probabilities = np.asarray(estimator.predict_proba(X_test_cond))
        categories = np.asarray(estimator.classes_, dtype=int)
        probabilities = probabilities.astype(np.float64, copy=False)
        if is_batched:
            return categories, probabilities
        return categories, probabilities[0]

    def sample_candidate_grid(
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        quantiles: Sequence[float],
        fixed_target: int,
        confidences: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Return deterministic conditional quantiles for every candidate.

        Each candidate feature is represented by one row per requested
        quantile. During TabICL's per-column imputation, those rows share one
        conditional-estimator fit and one batched prediction. The returned
        matrix is feature-major with shape ``(n_candidates, n_quantiles)``.
        """
        candidates = np.asarray(candidate_cols, dtype=int)
        if len(np.unique(candidates)) != len(candidates):
            raise ValueError("candidate_cols must not contain duplicates")
        query = np.asarray(X_query)
        if query.ndim != 2 or len(query) != 1:
            raise ValueError("sample_candidate_grid expects one query row")
        queries = np.repeat(query, len(candidates), axis=0)
        result = self.sample_candidate_grid_batch(
            queries,
            candidates,
            quantiles=quantiles,
            fixed_target=fixed_target,
            confidences=confidences,
        )
        if confidences is None:
            return result[:, 0, :]
        return result

    def sample_candidate_grid_batch(  # noqa: C901, PLR0912
        self,
        X_queries: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        quantiles: Sequence[float],
        fixed_target: int,
        confidences: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Return quantile grids for query/feature pairs in one model call.

        Unlike :meth:`sample_candidate_grid`, each candidate column is paired
        with its own query row. This is the fast path used to expand an entire
        counterfactual beam level at once.
        """
        candidates = np.asarray(candidate_cols, dtype=int)
        queries = np.asarray(X_queries)
        alphas = np.asarray(quantiles, dtype=np.float64)
        if queries.ndim != 2 or len(queries) != len(candidates):
            raise ValueError("X_queries must contain one row per candidate column")
        if candidates.ndim != 1 or len(candidates) == 0:
            raise ValueError("candidate_cols must be a non-empty 1D sequence")
        if alphas.ndim != 1 or len(alphas) == 0:
            raise ValueError("quantiles must be a non-empty 1D sequence")
        if not np.all(np.isfinite(alphas)) or np.any((alphas <= 0) | (alphas >= 1)):
            raise ValueError("quantiles must be finite values strictly between 0 and 1")
        if np.any(np.diff(alphas) <= 0):
            raise ValueError("quantiles must be strictly increasing and unique")

        confidence_values = None
        n_confidences = 1
        if confidences is not None:
            confidence_values = np.asarray(confidences, dtype=np.float64)
            if not self._uses_confidence:
                raise ValueError(
                    "confidences require confidence_context in set_context()"
                )
            if confidence_values.ndim != 1 or len(confidence_values) == 0:
                raise ValueError("confidences must be a non-empty 1D sequence")
            if not np.all(np.isfinite(confidence_values)):
                raise ValueError("confidences must contain only finite values")
            n_confidences = len(confidence_values)
        elif self._uses_confidence:
            raise ValueError(
                "confidences are required when confidence conditioning is enabled"
            )

        expansion = n_confidences * len(alphas)
        expanded_queries = np.repeat(queries, expansion, axis=0)
        expanded_candidates = np.repeat(candidates, expansion)
        expanded_confidences = None
        if confidence_values is not None:
            expanded_confidences = np.tile(
                np.repeat(confidence_values, len(alphas)),
                len(candidates),
            )
        X_aug = self._augmented_candidate_rows(
            expanded_queries,
            expanded_candidates.tolist(),
            fixed_target,
            allow_duplicate_cols=True,
            fixed_confidence=expanded_confidences,
        )

        model = self._require_model()
        model_type = type(model)
        sentinel = object()
        previous = getattr(model_type, "_numerical_quantile_grid", sentinel)
        model_type._numerical_quantile_grid = np.tile(
            alphas,
            n_confidences,
        ).astype(np.float32)
        try:
            filled = np.asarray(
                model.impute(
                    X_aug,
                    temperature=float(self.temperature),
                    n_iterations=1,
                )
            )
        finally:
            if previous is sentinel:
                delattr(model_type, "_numerical_quantile_grid")
            else:
                model_type._numerical_quantile_grid = previous

        values = filled[
            np.arange(len(expanded_candidates)),
            expanded_candidates,
        ]
        return values.astype(np.float64).reshape(
            len(candidates),
            n_confidences,
            len(alphas),
        )

    def sample_feature(
        self,
        X_query: np.ndarray,
        target_col: int,
        n_samples: int = 1,
        sample_temperature: float | None = None,
        fixed_target: int | None = None,
        fixed_confidence: float | None = None,
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
                fixed_confidence=fixed_confidence,
            )

        model = self._require_model()
        original_state = model.random_state
        draws = []
        try:
            for offset in range(n_samples):
                model.random_state = self.random_state + offset
                draws.append(
                    self.sample_candidates(
                        X_query,
                        [target_col],
                        sample_temperature=sample_temperature,
                        fixed_target=fixed_target,
                        fixed_confidence=fixed_confidence,
                    )
                )
        finally:
            model.random_state = original_state
        return np.stack(draws, axis=0)

    def impute_masked(
        self,
        X_query: np.ndarray,
        mask_cols: Sequence[int],
        fixed_target: int | None = None,
        fixed_confidence: float | None = None,
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
        augmented = [X, target]
        if self._uses_confidence:
            if fixed_confidence is None:
                raise ValueError(
                    "fixed_confidence is required when confidence conditioning "
                    "is enabled"
                )
            confidence = np.full(
                (len(X), 1),
                float(fixed_confidence),
                dtype=np.float32,
            )
            augmented.append(confidence)
        elif fixed_confidence is not None:
            raise ValueError(
                "fixed_confidence requires confidence_context in set_context()"
            )
        X_aug = np.concatenate(augmented, axis=1)
        model = self._require_model()
        filled = np.asarray(
            model.impute(
                X_aug,
                temperature=float(self.temperature),
                n_iterations=1,
            )
        )[:, : self._n_original_features].astype(np.float64)

        masked = set(mask_cols)
        observed = [j for j in range(original.shape[1]) if j not in masked]
        filled[:, observed] = original[:, observed]
        return filled
