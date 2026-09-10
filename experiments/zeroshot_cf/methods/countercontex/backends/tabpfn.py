"""TabPFN v2 conditional-feature proposal backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.contracts import MethodContext
from experiments.zeroshot_cf.methods.countercontex.backends.base import (
    CategoryProposals,
    NumericalDistribution,
    ProposalCapabilities,
)

TABPFN_BACKEND_IMPLEMENTATION_VERSION = "tabpfn-v2-proposal-v1"

EstimatorFactory = Callable[[], Any]


def _default_factories(
    *,
    classifier_path: Path,
    regressor_path: Path,
    device: str,
    n_estimators: int,
    seed: int,
) -> tuple[EstimatorFactory, EstimatorFactory]:
    """Build lazy, exact-checkpoint TabPFN v2 estimator factories."""
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError as error:  # pragma: no cover - exercised by optional import gate
        raise RuntimeError("tabpfn is required for the TabPFN backend") from error

    common = {
        "n_estimators": n_estimators,
        "device": device,
        "random_state": seed,
        "show_progress_bar": False,
    }

    def classifier():
        return TabPFNClassifier(model_path=classifier_path, **common)

    def regressor():
        return TabPFNRegressor(model_path=regressor_path, **common)

    return classifier, regressor


def _query_rows(rows: np.ndarray, count: int) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    if len(matrix) == 1:
        return np.repeat(matrix, count, axis=0)
    if len(matrix) != count:
        raise ValueError("proposal rows must contain one row or one row per column")
    return matrix


@dataclass
class TabPFNProposalSession:
    """Lazy feature-conditional estimators for one factual-specific context."""

    reference: np.ndarray
    reference_labels: np.ndarray
    target: int
    categorical_groups: tuple[OneHotActionGroup, ...]
    classifier_factory: EstimatorFactory
    regressor_factory: EstimatorFactory
    confidence_anchors: tuple[float, ...] | None = None
    diagnostics: Mapping[str, bool] = field(
        default_factory=lambda: {
            "categorical_confidence_batching": False,
            "conditional_estimator_cache": True,
            "tabicl_kv_cache": False,
        }
    )
    _regressors: dict[int, Any] = field(default_factory=dict, init=False)
    _classifiers: dict[tuple[int, ...], Any] = field(default_factory=dict, init=False)

    def _design(self, rows: np.ndarray, excluded: Sequence[int]) -> np.ndarray:
        kept = np.delete(np.asarray(rows, dtype=np.float64), tuple(excluded), axis=1)
        labels = np.full((len(kept), 1), self.target, dtype=np.float64)
        return np.column_stack((kept, labels))

    def _reference_design(self, excluded: Sequence[int]) -> np.ndarray:
        kept = np.delete(self.reference, tuple(excluded), axis=1)
        return np.column_stack((kept, self.reference_labels))

    def _regressor(self, column: int):
        if column not in self._regressors:
            estimator = self.regressor_factory()
            estimator.fit(
                self._reference_design((column,)), self.reference[:, column]
            )
            self._regressors[column] = estimator
        return self._regressors[column]

    def propose_numerical(
        self,
        rows: np.ndarray,
        columns: Sequence[int],
        *,
        quantiles: Sequence[float] | None,
        confidence: float | None,
        temperature: float,
    ) -> np.ndarray:
        del temperature
        if confidence is not None:
            raise ValueError("TabPFN backend does not support confidence conditioning")
        positions = tuple(int(column) for column in columns)
        if not positions:
            raise ValueError("numerical proposal columns must be non-empty")
        queries = _query_rows(rows, len(positions))
        proposals = []
        for row, column in zip(queries, positions, strict=True):
            estimator = self._regressor(column)
            design = self._design(row.reshape(1, -1), (column,))
            if quantiles is None:
                value = np.asarray(
                    estimator.predict(design, output_type="mode"), dtype=np.float64
                )[0]
                proposals.append(value)
            else:
                values = estimator.predict(
                    design,
                    output_type="quantiles",
                    quantiles=list(quantiles),
                )
                proposals.append(
                    np.asarray([value[0] for value in values], dtype=np.float64)
                )
        return np.asarray(proposals, dtype=np.float64)

    def propose_numerical_batch(
        self,
        rows: np.ndarray,
        columns: Sequence[int],
        *,
        quantiles: Sequence[float] | None,
        confidences: float | Sequence[float] | np.ndarray | None,
        temperature: float,
    ) -> np.ndarray:
        if confidences is not None:
            raise ValueError("TabPFN backend does not support confidence conditioning")
        values = self.propose_numerical(
            rows,
            columns,
            quantiles=quantiles,
            confidence=None,
            temperature=temperature,
        )
        return values[:, None, :] if quantiles is not None else values

    def numerical_distribution_batch(
        self,
        rows: np.ndarray,
        columns: Sequence[int],
        *,
        quantiles: Sequence[float],
        confidences: float | Sequence[float] | np.ndarray | None,
    ) -> NumericalDistribution:
        del rows, columns, quantiles, confidences
        raise ValueError("TabPFN backend does not support numerical distributions")

    def _multiclass_probabilities(
        self, row: np.ndarray, group: OneHotActionGroup, targets: np.ndarray
    ) -> np.ndarray:
        key = tuple(group.columns)
        estimator = self._classifiers.get(key)
        if estimator is None:
            estimator = self.classifier_factory()
            estimator.fit(self._reference_design(group.columns), targets)
            self._classifiers[key] = estimator
        design = self._design(row.reshape(1, -1), group.columns)
        predicted = np.asarray(estimator.predict_proba(design)[0], dtype=np.float64)
        probabilities = np.zeros(len(group.columns), dtype=np.float64)
        probabilities[np.asarray(estimator.classes_, dtype=int)] = predicted
        return probabilities

    def _one_vs_rest_probabilities(
        self, row: np.ndarray, group: OneHotActionGroup, targets: np.ndarray
    ) -> np.ndarray:
        design = self._design(row.reshape(1, -1), group.columns)
        reference_design = self._reference_design(group.columns)
        probabilities = np.zeros(len(group.columns), dtype=np.float64)
        for category in range(len(group.columns)):
            binary = (targets == category).astype(int)
            if len(np.unique(binary)) == 1:
                probabilities[category] = float(binary[0])
                continue
            key = (*group.columns, category)
            estimator = self._classifiers.get(key)
            if estimator is None:
                estimator = self.classifier_factory()
                estimator.fit(reference_design, binary)
                self._classifiers[key] = estimator
            classes = np.asarray(estimator.classes_, dtype=int)
            predicted = np.asarray(
                estimator.predict_proba(design)[0], dtype=np.float64
            )
            positive = np.flatnonzero(classes == 1)
            probabilities[category] = (
                float(predicted[positive[0]]) if len(positive) else 0.0
            )
        return probabilities

    def categorical_distribution(
        self,
        row: np.ndarray,
        group: OneHotActionGroup,
        *,
        confidence: float | None,
    ) -> CategoryProposals:
        if confidence is not None:
            raise ValueError("TabPFN backend does not support confidence conditioning")
        values = self.reference[:, group.columns]
        if not np.allclose(values.sum(axis=1), 1.0):
            raise ValueError(f"reference group {group.name!r} is not one-hot")
        targets = np.argmax(values, axis=1)
        observed = np.unique(targets)
        if len(observed) == 1:
            probabilities = np.zeros(len(group.columns), dtype=np.float64)
            probabilities[int(observed[0])] = 1.0
        elif len(observed) <= 10:
            probabilities = self._multiclass_probabilities(row, group, targets)
        else:
            probabilities = self._one_vs_rest_probabilities(row, group, targets)
        total = float(probabilities.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("TabPFN categorical probabilities have no finite mass")
        return CategoryProposals(
            np.arange(len(group.columns), dtype=np.int64), probabilities / total
        )

    def score_joint(self, rows: np.ndarray, target: int) -> np.ndarray:
        del rows, target
        raise ValueError("TabPFN backend does not support joint scoring")


@dataclass(frozen=True)
class PreparedTabPFNBackend:
    reference: np.ndarray
    reference_labels: np.ndarray
    categorical_groups: tuple[OneHotActionGroup, ...]
    context_size: int
    classifier_factory: EstimatorFactory | None
    regressor_factory: EstimatorFactory | None
    classifier_path: Path | None
    regressor_path: Path | None
    device: str
    n_estimators: int
    backend_id: str = "tabpfn"
    capabilities: ProposalCapabilities = ProposalCapabilities(
        numerical_proposals=True,
        categorical_distribution=True,
    )

    def for_factual(
        self, factual: np.ndarray, target: int, *, seed: int
    ) -> TabPFNProposalSession:
        query = np.asarray(factual, dtype=np.float64)
        distances = np.sum((self.reference - query) ** 2, axis=1)
        order = np.argsort(distances, kind="stable")[: self.context_size]
        classifier_factory = self.classifier_factory
        regressor_factory = self.regressor_factory
        if classifier_factory is None or regressor_factory is None:
            if self.classifier_path is None or self.regressor_path is None:
                raise RuntimeError("TabPFN checkpoint paths were not prepared")
            classifier_factory, regressor_factory = _default_factories(
                classifier_path=self.classifier_path,
                regressor_path=self.regressor_path,
                device=self.device,
                n_estimators=self.n_estimators,
                seed=seed,
            )
        return TabPFNProposalSession(
            reference=self.reference[order],
            reference_labels=self.reference_labels[order],
            target=int(target),
            categorical_groups=self.categorical_groups,
            classifier_factory=classifier_factory,
            regressor_factory=regressor_factory,
        )


@dataclass(frozen=True)
class TabPFNBackend:
    """Config-bound TabPFN v2 backend with injectable estimator factories."""

    context_size: int
    context_labels: str
    classifier_path: Path | None = None
    regressor_path: Path | None = None
    device: str = "auto"
    n_estimators: int = 1
    classifier_factory: EstimatorFactory | None = None
    regressor_factory: EstimatorFactory | None = None
    backend_id: str = "tabpfn"
    capabilities: ProposalCapabilities = ProposalCapabilities(
        numerical_proposals=True,
        categorical_distribution=True,
    )

    def prepare(self, context: MethodContext) -> PreparedTabPFNBackend:
        reference = np.asarray(context.X_reference, dtype=np.float64)
        predictions = np.asarray(context.oracle.predict(reference)).reshape(-1)
        if self.context_labels == "true":
            if context.y_reference is None:
                raise ValueError(
                    "true context labels require training reference labels"
                )
            labels = np.asarray(context.y_reference).reshape(-1)
        else:
            labels = predictions
        if self.classifier_factory is None or self.regressor_factory is None:
            if self.classifier_path is None or self.regressor_path is None:
                raise ValueError("TabPFN backend requires both local checkpoint paths")
        return PreparedTabPFNBackend(
            reference=reference.copy(),
            reference_labels=np.asarray(labels).copy(),
            categorical_groups=context.feature_schema.categorical_groups,
            context_size=min(self.context_size, len(reference)),
            classifier_factory=self.classifier_factory,
            regressor_factory=self.regressor_factory,
            classifier_path=self.classifier_path,
            regressor_path=self.regressor_path,
            device=self.device,
            n_estimators=self.n_estimators,
        )
