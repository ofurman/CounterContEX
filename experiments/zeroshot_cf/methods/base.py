"""Shared contracts and helpers for counterfactual methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from experiments.zeroshot_cf.core.contracts import (
    GenerationRequest,
    GenerationResult,
    MethodContext,
)


@dataclass(frozen=True)
class MethodCapabilities:
    """Static requirements and supported behavior of one method."""

    supports_categorical: bool
    enforces_actionability: bool
    supports_multiple_counterfactuals: bool
    requires_probabilities: bool
    optional_dependencies: tuple[str, ...] = ()


@runtime_checkable
class CounterfactualMethod(Protocol):
    """Configuration plus dataset-level method preparation."""

    method_id: str
    capabilities: MethodCapabilities

    def config_dict(self) -> Mapping[str, Any]: ...

    def prepare(self, context: MethodContext) -> PreparedMethod: ...


@runtime_checkable
class PreparedMethod(Protocol):
    """Prepared method that generates candidates for one request."""

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


def require_single_counterfactual(request: GenerationRequest) -> None:
    if request.n_counterfactuals != 1:
        raise ValueError("method supports exactly one counterfactual per factual")


def canonical_single_result(
    raw_candidates: np.ndarray,
    available: np.ndarray,
    *,
    point_diagnostics: tuple[Mapping[str, Any], ...],
    run_diagnostics: Mapping[str, Any],
    extra_artifacts: Mapping[str, np.ndarray] | None = None,
) -> GenerationResult:
    """Represent genuine returns canonically and retain failures as best effort."""
    raw = np.asarray(raw_candidates, dtype=np.float64)
    mask = np.asarray(available, dtype=bool).reshape(-1)
    if raw.ndim != 2 or len(raw) != len(mask):
        raise ValueError("raw candidates and availability must have equal row counts")
    candidates = raw[:, None, :].copy()
    candidates[~mask, 0, :] = np.nan
    artifacts: dict[str, np.ndarray] = dict(extra_artifacts or {})
    if (~mask).any():
        artifacts["method.best_effort"] = raw.copy()
    return GenerationResult(
        candidates=candidates,
        available=mask[:, None],
        point_diagnostics=point_diagnostics,
        run_diagnostics=run_diagnostics,
        artifacts=artifacts,
    )
