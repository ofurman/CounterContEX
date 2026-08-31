"""Portable proposal contracts consumed by DiCoFlex search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.contracts import MethodContext


@dataclass(frozen=True)
class ProposalCapabilities:
    """Proposal semantics a backend explicitly guarantees."""

    numerical_proposals: bool = True
    confidence_conditioning: bool = False
    categorical_distribution: bool = False
    joint_scoring: bool = False


@dataclass(frozen=True)
class CategoryProposals:
    """Complete category support and conditional probabilities for one group."""

    categories: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        categories = np.asarray(self.categories, dtype=np.int64)
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if categories.ndim != 1 or probabilities.shape != categories.shape:
            raise ValueError(
                "category proposals require equally sized one-dimensional arrays"
            )
        if len(categories) == 0 or len(np.unique(categories)) != len(categories):
            raise ValueError("category proposals require unique non-empty support")
        if np.any(categories < 0):
            raise ValueError("category proposal indices must be non-negative")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError(
                "category proposal probabilities must be finite and non-negative"
            )
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("category proposal probabilities must sum to one")
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "probabilities", probabilities)


@runtime_checkable
class ProposalSession(Protocol):
    """Factual-specific proposal and optional scoring operations."""

    confidence_anchors: tuple[float, ...] | None
    diagnostics: Mapping[str, bool]

    def propose_numerical(
        self,
        rows: np.ndarray,
        columns: Sequence[int],
        *,
        quantiles: Sequence[float] | None,
        confidence: float | None,
        temperature: float,
    ) -> np.ndarray: ...

    def categorical_distribution(
        self,
        row: np.ndarray,
        group: OneHotActionGroup,
        *,
        confidence: float | None,
    ) -> CategoryProposals: ...

    def score_joint(self, rows: np.ndarray, target: int) -> np.ndarray: ...


@runtime_checkable
class PreparedBackend(Protocol):
    """Dataset-level proposal state shared by factual sessions."""

    backend_id: str
    capabilities: ProposalCapabilities

    def for_factual(
        self,
        factual: np.ndarray,
        target: int,
        *,
        seed: int,
    ) -> ProposalSession: ...


@runtime_checkable
class ProposalBackend(Protocol):
    """Config-bound proposal backend prepared against one method context."""

    backend_id: str
    capabilities: ProposalCapabilities

    def prepare(self, context: MethodContext) -> PreparedBackend: ...


def validate_backend_capabilities(
    capabilities: ProposalCapabilities,
    *,
    needs_confidence: bool,
    needs_categorical: bool,
    needs_joint: bool,
) -> None:
    """Reject unsupported search/backend combinations before generation."""
    missing: list[str] = []
    if not capabilities.numerical_proposals:
        missing.append("numerical proposals")
    if needs_confidence and not capabilities.confidence_conditioning:
        missing.append("confidence conditioning")
    if needs_categorical and not capabilities.categorical_distribution:
        missing.append("categorical distributions")
    if needs_joint and not capabilities.joint_scoring:
        missing.append("joint scoring")
    if missing:
        raise ValueError(
            "proposal backend does not support required capabilities: "
            + ", ".join(missing)
        )


# Architecture documents used the longer name before the implementation was
# materialized. Keep it as a typed compatibility alias.
PreparedProposalBackend = PreparedBackend
