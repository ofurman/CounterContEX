"""TabICL proposal adapter owned entirely by CounterContEx."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.contracts import MethodContext
from experiments.zeroshot_cf.generator import (
    ATHENA_CONTEXT_SIZE,
    DEFAULT_POINT_ESTIMATE,
    empirical_confidence_grid,
)
from experiments.zeroshot_cf.grouped_categorical import (
    CompactMixedSampler,
    ConditionedCategoryDistribution,
    GroupedCategoricalCodec,
)
from experiments.zeroshot_cf.methods.countercontex.backends.base import (
    CategoryProposals,
    ProposalCapabilities,
)
from experiments.zeroshot_cf.methods.countercontex.config import CounterContExConfig

TABICL_BACKEND_IMPLEMENTATION_VERSION = "tabicl-proposal-v1"


@dataclass(frozen=True)
class CounterContExBackendInputs:
    """Portable data required to prepare the foundation backend."""

    X_reference: np.ndarray
    categorical_groups: tuple[OneHotActionGroup, ...]
    actionable_groups: tuple[OneHotActionGroup, ...]
    oracle: Any


@dataclass(frozen=True)
class CounterContExBackendRuntime:
    """Lazily imported runtime types and device selection."""

    device: str
    sampler_type: type
    joint_scorer_type: type


def load_backend_runtime() -> CounterContExBackendRuntime:
    """Load optional TabICL runtime code during method preparation."""
    from experiments.zeroshot_cf.tabicl_checkpoints import TABICL_DEVICE
    from experiments.zeroshot_cf.tabicl_joint_plausibility import TabICLJointScorer
    from experiments.zeroshot_cf.tabicl_sampler import TabICLConditionalDensitySampler

    return CounterContExBackendRuntime(
        device=TABICL_DEVICE,
        sampler_type=TabICLConditionalDensitySampler,
        joint_scorer_type=TabICLJointScorer,
    )


def _build_category_distribution(
    *,
    sampler_context: Any,
    categorical_codec: GroupedCategoricalCodec,
    target_class: int,
    confidence_grid: tuple[float, ...] | None,
) -> ConditionedCategoryDistribution:
    cache: dict[tuple[bytes, str], tuple[np.ndarray, np.ndarray]] = {}
    anchors = (None,) if confidence_grid is None else confidence_grid

    def conditioned_category_distribution(
        row: np.ndarray,
        group: Any,
        confidence: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (np.ascontiguousarray(row).tobytes(), group.name)
        if key not in cache:
            encoded_row = categorical_codec.encode_row(row)
            encoded_col = categorical_codec.encoded_column_for_group(group)
            fixed_confidences = (
                None if anchors == (None,) else np.asarray(anchors, dtype=np.float32)
            )
            categories, probability_grid = sampler_context.categorical_distribution(
                encoded_row.reshape(1, -1),
                encoded_col,
                fixed_target=target_class,
                fixed_confidence=fixed_confidences,
            )
            cache[key] = (
                np.asarray(categories, dtype=int),
                np.atleast_2d(np.asarray(probability_grid, dtype=np.float64)),
            )
        categories, probability_grid = cache[key]
        if confidence is None:
            anchor_index = 0
        else:
            matches = np.flatnonzero(
                np.isclose(np.asarray(anchors, dtype=float), confidence)
            )
            if not len(matches):
                raise ValueError(f"unknown categorical confidence anchor: {confidence}")
            anchor_index = int(matches[0])
        return categories, probability_grid[anchor_index]

    return conditioned_category_distribution


@dataclass(frozen=True)
class _TabICLPointState:
    sampler: Any
    candidate_confidences: tuple[float, ...] | None
    category_distribution: ConditionedCategoryDistribution | None
    joint_scorer: Any | None
    metadata: Mapping[str, bool]


@dataclass(frozen=True)
class TabICLProposalSession:
    """Portable proposal operations over one factual-specific TabICL context."""

    state: _TabICLPointState
    target: int

    @property
    def confidence_anchors(self) -> tuple[float, ...] | None:
        return self.state.candidate_confidences

    @property
    def diagnostics(self) -> Mapping[str, bool]:
        return self.state.metadata

    def propose_numerical(
        self,
        rows: np.ndarray,
        columns: Sequence[int],
        *,
        quantiles: Sequence[float] | None,
        confidence: float | None,
        temperature: float,
    ) -> np.ndarray:
        if quantiles is None:
            return np.asarray(
                self.state.sampler.sample_candidates(
                    rows,
                    columns,
                    sample_temperature=temperature,
                    fixed_target=self.target,
                    fixed_confidence=confidence,
                ),
                dtype=np.float64,
            )
        values = np.asarray(
            self.state.sampler.sample_candidate_grid(
                rows,
                columns,
                quantiles=quantiles,
                fixed_target=self.target,
                confidences=None if confidence is None else (confidence,),
            ),
            dtype=np.float64,
        )
        if confidence is not None:
            values = values[:, 0, :]
        return values

    def propose_numerical_batch(
        self,
        rows: np.ndarray,
        columns: Sequence[int],
        *,
        quantiles: Sequence[float] | None,
        confidences: float | Sequence[float] | np.ndarray | None,
        temperature: float,
    ) -> np.ndarray:
        if quantiles is None:
            return np.asarray(
                self.state.sampler.sample_candidates_batch(
                    rows,
                    columns,
                    sample_temperature=temperature,
                    fixed_target=self.target,
                    fixed_confidence=confidences,
                ),
                dtype=np.float64,
            )
        return np.asarray(
            self.state.sampler.sample_candidate_grid_batch(
                rows,
                columns,
                quantiles=quantiles,
                fixed_target=self.target,
                confidences=confidences,
            ),
            dtype=np.float64,
        )

    def categorical_distribution(
        self,
        row: np.ndarray,
        group: OneHotActionGroup,
        *,
        confidence: float | None,
    ) -> CategoryProposals:
        if self.state.category_distribution is None:
            raise ValueError("TabICL categorical distributions were not prepared")
        categories, probabilities = self.state.category_distribution(
            row, group, confidence
        )
        return CategoryProposals(categories, probabilities)

    def score_joint(self, rows: np.ndarray, target: int) -> np.ndarray:
        if self.state.joint_scorer is None:
            raise ValueError("TabICL joint scoring was not prepared")
        return np.asarray(
            self.state.joint_scorer.score_rows(rows, target).joint_log_density,
            dtype=np.float64,
        )


@dataclass(frozen=True)
class PreparedTabICLBackend:
    """Dataset-level backend state shared by generation requests."""

    inputs: CounterContExBackendInputs
    config: CounterContExConfig
    runtime: CounterContExBackendRuntime
    categorical_codec: GroupedCategoricalCodec | None
    X_sampler_reference: np.ndarray
    reference_predictions: np.ndarray
    reference_probabilities: np.ndarray | None
    oracle_classes: np.ndarray
    backend_id: str = "tabicl"
    capabilities: ProposalCapabilities = ProposalCapabilities(
        numerical_proposals=True,
        confidence_conditioning=True,
        categorical_distribution=True,
        joint_scoring=True,
    )
    _factories: dict[int, Any] = field(default_factory=dict, compare=False, repr=False)

    def point_backend_factory(self, *, seed: int):
        """Build request-scoped samplers whose stochastic state uses ``seed``."""
        foundation = self.config.foundation
        search = self.config.search
        categorical_features = (
            None
            if self.categorical_codec is None
            else self.categorical_codec.categorical_columns
        )

        def make_sampler_context():
            return self.runtime.sampler_type(
                n_estimators=foundation.n_estimators,
                temperature=foundation.temperature,
                random_state=seed,
                device=self.runtime.device,
                cache_dir=foundation.cache_dir,
                numerical_point_estimate=DEFAULT_POINT_ESTIMATE,
                categorical_features=categorical_features,
            )

        proposal_context = make_sampler_context()
        proposal_sampler = (
            proposal_context
            if self.categorical_codec is None
            else CompactMixedSampler(proposal_context, self.categorical_codec)
        )
        joint_context = None
        joint_sampler = None
        if search.cf_mode == "data_plausible":
            joint_context = make_sampler_context()
            joint_sampler = (
                joint_context
                if self.categorical_codec is None
                else CompactMixedSampler(joint_context, self.categorical_codec)
            )

        def point_backend_factory(
            factual: np.ndarray,
            target_class: int,
        ) -> _TabICLPointState:
            query = (
                factual
                if self.categorical_codec is None
                else self.categorical_codec.encode_row(factual)
            )
            confidence_context = None
            if self.reference_probabilities is not None:
                positions = np.flatnonzero(self.oracle_classes == target_class)
                if len(positions) != 1:
                    raise ValueError(
                        f"target class {target_class} is absent from classifier classes"
                    )
                confidence_context = self.reference_probabilities[:, int(positions[0])]
            proposal_context.set_context(
                self.X_sampler_reference,
                y_context=self.reference_predictions,
                confidence_context=confidence_context,
                target_class=None,
                max_context=ATHENA_CONTEXT_SIZE,
                selection="knn",
                query=query,
            )
            confidence_grid = None
            if foundation.confidence_quantiles is not None:
                selected_confidences = proposal_context.selected_confidences_
                selected_labels = proposal_context.selected_labels_
                if selected_confidences is None or selected_labels is None:
                    raise RuntimeError(
                        "confidence-conditioned context diagnostics are unavailable"
                    )
                confidence_grid = empirical_confidence_grid(
                    selected_confidences,
                    selected_labels,
                    int(target_class),
                    foundation.confidence_quantiles,
                )

            joint_scorer = None
            if search.cf_mode == "data_plausible":
                if joint_context is None or joint_sampler is None:
                    raise RuntimeError("joint scorer is unavailable")
                joint_context.set_context(
                    self.X_sampler_reference,
                    y_context=self.reference_predictions,
                    confidence_context=None,
                    target_class=None,
                    max_context=ATHENA_CONTEXT_SIZE,
                    selection="knn",
                    query=query,
                )
                joint_scorer = self.runtime.joint_scorer_type(
                    sampler=joint_sampler,
                    target_class=int(target_class),
                    n_permutations=foundation.tabicl_joint_permutations,
                )

            category_distribution = None
            if self.categorical_codec is not None and self.inputs.actionable_groups:
                category_distribution = _build_category_distribution(
                    sampler_context=proposal_context,
                    categorical_codec=self.categorical_codec,
                    target_class=int(target_class),
                    confidence_grid=confidence_grid,
                )

            estimator_params = getattr(proposal_context, "estimator_params", {})
            return _TabICLPointState(
                sampler=proposal_sampler,
                candidate_confidences=confidence_grid,
                category_distribution=category_distribution,
                joint_scorer=joint_scorer,
                metadata={
                    "categorical_confidence_batching": True,
                    "conditional_estimator_cache": True,
                    "tabicl_kv_cache": bool(estimator_params.get("kv_cache", False)),
                },
            )

        return point_backend_factory

    def for_factual(
        self,
        factual: np.ndarray,
        target: int,
        *,
        seed: int,
    ) -> TabICLProposalSession:
        """Prepare neighbor context and anchors inside the TabICL adapter."""
        if seed not in self._factories:
            self._factories[seed] = self.point_backend_factory(seed=seed)
        state = self._factories[seed](np.asarray(factual), int(target))
        return TabICLProposalSession(state, int(target))


@dataclass(frozen=True)
class TabICLBackend:
    """Config-bound TabICL proposal backend."""

    config: CounterContExConfig
    runtime: CounterContExBackendRuntime | None = None
    backend_id: str = "tabicl"
    capabilities: ProposalCapabilities = ProposalCapabilities(
        numerical_proposals=True,
        confidence_conditioning=True,
        categorical_distribution=True,
        joint_scoring=True,
    )

    def prepare(self, context: MethodContext) -> PreparedTabICLBackend:
        schema = context.feature_schema
        return prepare_backend(
            CounterContExBackendInputs(
                X_reference=context.X_reference,
                categorical_groups=schema.categorical_groups,
                actionable_groups=schema.actionable_groups,
                oracle=context.oracle,
            ),
            self.config,
            runtime=self.runtime,
        )


def prepare_backend(
    inputs: CounterContExBackendInputs,
    config: CounterContExConfig,
    *,
    runtime: CounterContExBackendRuntime | None = None,
) -> PreparedTabICLBackend:
    """Prepare encoded reference data and lazy runtime dependencies."""
    runtime = load_backend_runtime() if runtime is None else runtime
    categorical_codec = (
        None
        if not inputs.categorical_groups
        else GroupedCategoricalCodec.from_matrix(
            inputs.X_reference,
            inputs.categorical_groups,
        )
    )
    X_sampler_reference = (
        np.asarray(inputs.X_reference)
        if categorical_codec is None
        else categorical_codec.encode(inputs.X_reference)
    )
    predictions = np.asarray(inputs.oracle.predict(inputs.X_reference)).reshape(-1)
    probabilities = (
        np.asarray(inputs.oracle.predict_proba(inputs.X_reference))
        if config.foundation.confidence_quantiles is not None
        else None
    )
    classes = np.asarray(
        getattr(
            inputs.oracle,
            "classes_",
            np.arange(probabilities.shape[1] if probabilities is not None else 0),
        )
    )
    return PreparedTabICLBackend(
        inputs=inputs,
        config=config,
        runtime=runtime,
        categorical_codec=categorical_codec,
        X_sampler_reference=X_sampler_reference,
        reference_predictions=predictions,
        reference_probabilities=probabilities,
        oracle_classes=classes,
    )
