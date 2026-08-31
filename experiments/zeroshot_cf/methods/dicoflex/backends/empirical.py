"""Deterministic empirical proposal backend for runnable DiCoFlex ablations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.contracts import MethodContext
from experiments.zeroshot_cf.methods.dicoflex.backends.base import (
    CategoryProposals,
    ProposalCapabilities,
)

EMPIRICAL_BACKEND_IMPLEMENTATION_VERSION = "empirical-reference-v1"


@dataclass(frozen=True)
class EmpiricalProposalSession:
    """Target-class empirical proposals for one factual."""

    reference: np.ndarray
    confidence_anchors: tuple[float, ...] | None = None
    diagnostics: Mapping[str, bool] = field(
        default_factory=lambda: {
            "categorical_confidence_batching": False,
            "conditional_estimator_cache": False,
            "tabicl_kv_cache": False,
        }
    )

    def propose_numerical(
        self,
        rows: np.ndarray,
        columns: Sequence[int],
        *,
        quantiles: Sequence[float] | None,
        confidence: float | None,
        temperature: float,
    ) -> np.ndarray:
        del rows, temperature
        if confidence is not None:
            raise ValueError(
                "empirical backend does not support confidence conditioning"
            )
        positions = np.asarray(tuple(int(column) for column in columns), dtype=int)
        if positions.ndim != 1 or len(positions) == 0:
            raise ValueError("numerical proposal columns must be non-empty")
        if np.any(positions < 0) or np.any(positions >= self.reference.shape[1]):
            raise IndexError("numerical proposal column is out of bounds")
        values = self.reference[:, positions]
        if quantiles is None:
            return np.asarray(np.median(values, axis=0), dtype=np.float64)
        levels = np.asarray(
            tuple(float(value) for value in quantiles), dtype=np.float64
        )
        if (
            levels.ndim != 1
            or len(levels) == 0
            or np.any((levels <= 0) | (levels >= 1))
        ):
            raise ValueError("empirical quantiles must lie strictly inside (0, 1)")
        # np.quantile is quantile-major; the search contract is feature-major.
        return np.asarray(np.quantile(values, levels, axis=0).T, dtype=np.float64)

    def categorical_distribution(
        self,
        row: np.ndarray,
        group: OneHotActionGroup,
        *,
        confidence: float | None,
    ) -> CategoryProposals:
        del row
        if confidence is not None:
            raise ValueError(
                "empirical backend does not support confidence conditioning"
            )
        group_values = self.reference[:, group.columns]
        if not np.allclose(group_values.sum(axis=1), 1.0):
            raise ValueError(f"reference group {group.name!r} is not one-hot")
        categories = np.arange(len(group.columns), dtype=np.int64)
        counts = np.bincount(
            np.argmax(group_values, axis=1), minlength=len(group.columns)
        ).astype(np.float64)
        # Unit smoothing retains complete category support deterministically.
        probabilities = (counts + 1.0) / (counts.sum() + len(categories))
        return CategoryProposals(categories, probabilities)

    def score_joint(self, rows: np.ndarray, target: int) -> np.ndarray:
        del rows, target
        raise ValueError("empirical backend does not support joint scoring")


@dataclass(frozen=True)
class PreparedEmpiricalBackend:
    reference: np.ndarray
    reference_predictions: np.ndarray
    backend_id: str = "empirical"
    capabilities: ProposalCapabilities = ProposalCapabilities(
        numerical_proposals=True,
        categorical_distribution=True,
    )

    def for_factual(
        self,
        factual: np.ndarray,
        target: int,
        *,
        seed: int,
    ) -> EmpiricalProposalSession:
        del factual, seed
        target_reference = self.reference[self.reference_predictions == int(target)]
        if len(target_reference) == 0:
            target_reference = self.reference
        return EmpiricalProposalSession(target_reference)


@dataclass(frozen=True)
class EmpiricalBackend:
    """Prepare deterministic proposals from the method reference matrix."""

    backend_id: str = "empirical"
    capabilities: ProposalCapabilities = ProposalCapabilities(
        numerical_proposals=True,
        categorical_distribution=True,
    )

    def prepare(self, context: MethodContext) -> PreparedEmpiricalBackend:
        reference = np.asarray(context.X_reference, dtype=np.float64)
        predictions = np.asarray(context.oracle.predict(reference)).reshape(-1)
        if reference.ndim != 2 or len(predictions) != len(reference):
            raise ValueError("empirical backend requires aligned reference predictions")
        return PreparedEmpiricalBackend(reference.copy(), predictions.copy())
