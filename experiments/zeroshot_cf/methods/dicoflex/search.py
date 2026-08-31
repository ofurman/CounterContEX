"""Backend-neutral adapter from DiCoFlex sessions to the retained search core."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from experiments.zeroshot_cf.generator import (
    DiscriminatorProtocol,
    TabICLGeneratorInputs,
    TabICLGeneratorPointBackend,
    TabICLGeneratorResult,
    generate_counterfactual_batch,
)
from experiments.zeroshot_cf.methods.dicoflex.backends.base import (
    PreparedBackend,
    ProposalSession,
    validate_backend_capabilities,
)
from experiments.zeroshot_cf.methods.dicoflex.config import DiCoFlexConfig


class _SessionSampler:
    """Legacy sampler surface implemented only through a proposal session."""

    def __init__(self, session: ProposalSession) -> None:
        self._session = session

    def sample_candidates(
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        sample_temperature: float,
        fixed_target: int,
        fixed_confidence: float | None = None,
    ) -> np.ndarray:
        del fixed_target
        return self._session.propose_numerical(
            X_query,
            candidate_cols,
            quantiles=None,
            confidence=fixed_confidence,
            temperature=sample_temperature,
        )

    def sample_candidate_grid(
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        quantiles: Sequence[float],
        fixed_target: int,
        confidences: Sequence[float] | None = None,
    ) -> np.ndarray:
        del fixed_target
        anchors: tuple[float | None, ...] = (
            (None,) if confidences is None else tuple(float(v) for v in confidences)
        )
        grids = [
            np.asarray(
                self._session.propose_numerical(
                    X_query,
                    candidate_cols,
                    quantiles=quantiles,
                    confidence=anchor,
                    temperature=0.0,
                ),
                dtype=np.float64,
            )
            for anchor in anchors
        ]
        return grids[0] if confidences is None else np.stack(grids, axis=1)

    def sample_candidates_batch(
        self,
        X_queries: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        sample_temperature: float,
        fixed_target: int,
        fixed_confidence: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        del fixed_target
        return np.asarray(
            self._session.propose_numerical_batch(
                X_queries,
                candidate_cols,
                quantiles=None,
                confidences=fixed_confidence,
                temperature=sample_temperature,
            ),
            dtype=np.float64,
        )

    def sample_candidate_grid_batch(
        self,
        X_queries: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        quantiles: Sequence[float],
        fixed_target: int,
        confidences: Sequence[float] | None = None,
    ) -> np.ndarray:
        del fixed_target
        return np.asarray(
            self._session.propose_numerical_batch(
                X_queries,
                candidate_cols,
                quantiles=quantiles,
                confidences=confidences,
                temperature=0.0,
            ),
            dtype=np.float64,
        )


@dataclass
class _SessionJointScorer:
    session: ProposalSession
    batch_count: int = 0
    row_count: int = 0

    def score_rows(self, rows: np.ndarray, target_class: int):
        scores = np.asarray(
            self.session.score_joint(rows, int(target_class)), dtype=np.float64
        )
        matrix = np.atleast_2d(np.asarray(rows))
        if scores.shape != (len(matrix),) or not np.all(np.isfinite(scores)):
            raise ValueError("proposal backend returned invalid joint scores")
        self.batch_count += 1
        self.row_count += len(matrix)
        # Retained search reads this stable value object by attribute.
        return _JointScoreBatch(scores)


@dataclass(frozen=True)
class _JointScoreBatch:
    joint_log_density: np.ndarray


def _point_backend(
    session: ProposalSession,
    *,
    use_categorical_distribution: bool,
    use_joint_scoring: bool,
) -> TabICLGeneratorPointBackend:
    category_distribution = None
    if use_categorical_distribution:

        def category_distribution(row, group, confidence):
            proposals = session.categorical_distribution(
                row,
                group,
                confidence=confidence,
            )
            return proposals.categories, proposals.probabilities

    joint_scorer = _SessionJointScorer(session) if use_joint_scoring else None
    return TabICLGeneratorPointBackend(
        sampler=_SessionSampler(session),
        candidate_confidences=session.confidence_anchors,
        category_distribution=category_distribution,
        joint_scorer=joint_scorer,
        metadata=dict(session.diagnostics),
    )


def generate_with_backend(
    inputs: TabICLGeneratorInputs,
    *,
    discriminator: DiscriminatorProtocol,
    config: DiCoFlexConfig,
    backend: PreparedBackend,
    seed: int,
    n_counterfactuals: int,
) -> TabICLGeneratorResult:
    """Run retained search using only the portable proposal-backend contract."""
    validate_backend_capabilities(
        backend.capabilities,
        needs_confidence=config.foundation.confidence_quantiles is not None,
        needs_categorical=bool(inputs.categorical_groups),
        needs_joint=config.search.cf_mode == "data_plausible",
    )

    def point_backend_factory(
        factual: np.ndarray,
        target: int,
    ) -> TabICLGeneratorPointBackend:
        session = backend.for_factual(factual, target, seed=seed)
        return _point_backend(
            session,
            use_categorical_distribution=bool(inputs.categorical_groups),
            use_joint_scoring=config.search.cf_mode == "data_plausible",
        )

    return generate_counterfactual_batch(
        inputs,
        discriminator=discriminator,
        config=config.generator_config(n_counterfactuals),
        point_backend_factory=point_backend_factory,
    )
