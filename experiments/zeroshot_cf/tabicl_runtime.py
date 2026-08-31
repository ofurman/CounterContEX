"""Neutral runtime adapter from the benchmark protocol to the stable TabICL API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.zeroshot_cf.benchmark_protocol import BenchmarkDatasetContext
from experiments.zeroshot_cf.candidate_domains import infer_feature_domains
from experiments.zeroshot_cf.diverse_search import DiverseBeamSearchConfig
from experiments.zeroshot_cf.generator import (
    ATHENA_CONTEXT_SIZE,
    ATHENA_CONTEXT_STRATEGY,
    DEFAULT_CATEGORICAL_PROPOSAL_COUNT,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_POINT_ESTIMATE,
    DEFAULT_TEMPERATURE,
    CF_MODES,
    TabICLGeneratorConfig,
    TabICLGeneratorInputs,
    TabICLGeneratorPointBackend,
    TabICLGeneratorResult,
    empirical_confidence_grid,
    generate_counterfactual_batch,
)
from experiments.zeroshot_cf.grouped_categorical import (
    CompactMixedSampler,
    ConditionedCategoryDistribution,
    GroupedCategoricalCodec,
)
from experiments.zeroshot_cf.retained_config import TAU


@dataclass(frozen=True)
class TabICLBenchmarkRun:
    """One benchmark run executed through the stable TabICL generator API."""

    context: BenchmarkDatasetContext
    n_estimators: int
    result: TabICLGeneratorResult

    @property
    def counterfactuals(self) -> np.ndarray:
        return self.result.counterfactuals

    @property
    def sparse_counterfactuals(self) -> np.ndarray:
        return self.result.sparse_counterfactuals

    @property
    def counterfactual_sets(self) -> np.ndarray:
        return self.result.counterfactual_sets

    @property
    def diagnostics(self):
        return self.result.diagnostics


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
                None
                if anchors == (None,)
                else np.asarray(anchors, dtype=np.float32)
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
            matches = np.flatnonzero(np.isclose(np.asarray(anchors, dtype=float), confidence))
            if not len(matches):
                raise ValueError(f"unknown categorical confidence anchor: {confidence}")
            anchor_index = int(matches[0])
        return categories, probability_grid[anchor_index]

    return conditioned_category_distribution


def _build_point_backend_factory(
    context: BenchmarkDatasetContext,
    *,
    confidence_quantiles: tuple[float, ...] | None,
    cf_mode: str,
    tabicl_joint_permutations: int,
    n_estimators: int,
    temperature: float,
    cache_dir: Path | None,
):
    from experiments.zeroshot_cf.tabicl_checkpoints import TABICL_DEVICE
    from experiments.zeroshot_cf.tabicl_joint_plausibility import TabICLJointScorer
    from experiments.zeroshot_cf.tabicl_sampler import TabICLConditionalDensitySampler

    categorical_codec = (
        None
        if not context.categorical_groups
        else GroupedCategoricalCodec.from_matrix(
            context.bundle.X_train,
            context.categorical_groups,
        )
    )
    categorical_features = (
        None if categorical_codec is None else categorical_codec.categorical_columns
    )
    sampler_context = TabICLConditionalDensitySampler(
        n_estimators=n_estimators,
        temperature=temperature,
        random_state=42,
        device=TABICL_DEVICE,
        cache_dir=cache_dir,
        numerical_point_estimate=DEFAULT_POINT_ESTIMATE,
        categorical_features=categorical_features,
    )
    proposal_sampler = (
        sampler_context
        if categorical_codec is None
        else CompactMixedSampler(sampler_context, categorical_codec)
    )
    X_sampler_train = (
        context.bundle.X_train
        if categorical_codec is None
        else categorical_codec.encode(context.bundle.X_train)
    )
    y_context = np.asarray(
        context.disc_model.predict(context.bundle.X_train),
        dtype=int,
    )
    context_probabilities = (
        np.asarray(context.disc_model.predict_proba(context.bundle.X_train))
        if confidence_quantiles is not None
        else None
    )

    joint_sampler_context = None
    joint_sampler = None
    if cf_mode == "data_plausible":
        joint_sampler_context = TabICLConditionalDensitySampler(
            n_estimators=n_estimators,
            temperature=temperature,
            random_state=42,
            device=TABICL_DEVICE,
            cache_dir=cache_dir,
            numerical_point_estimate=DEFAULT_POINT_ESTIMATE,
            categorical_features=categorical_features,
        )
        joint_sampler = (
            joint_sampler_context
            if categorical_codec is None
            else CompactMixedSampler(joint_sampler_context, categorical_codec)
        )

    def point_backend_factory(
        factual: np.ndarray,
        target_class: int,
    ) -> TabICLGeneratorPointBackend:
        query = factual if categorical_codec is None else categorical_codec.encode_row(factual)
        confidence_context = (
            None
            if context_probabilities is None
            else context_probabilities[:, int(target_class)]
        )
        sampler_context.set_context(
            X_sampler_train,
            y_context=y_context,
            confidence_context=confidence_context,
            target_class=None,
            max_context=ATHENA_CONTEXT_SIZE,
            selection="knn",
            query=query,
        )
        confidence_grid = None
        if confidence_quantiles is not None:
            selected_confidences = sampler_context.selected_confidences_
            selected_labels = sampler_context.selected_labels_
            if selected_confidences is None or selected_labels is None:
                raise RuntimeError(
                    "confidence-conditioned context diagnostics are unavailable"
                )
            confidence_grid = empirical_confidence_grid(
                selected_confidences,
                selected_labels,
                int(target_class),
                confidence_quantiles,
            )

        joint_scorer = None
        if cf_mode == "data_plausible":
            if joint_sampler_context is None or joint_sampler is None:
                raise RuntimeError("TabICL joint scorer is unavailable")
            joint_sampler_context.set_context(
                X_sampler_train,
                y_context=y_context,
                confidence_context=None,
                target_class=None,
                max_context=ATHENA_CONTEXT_SIZE,
                selection="knn",
                query=query,
            )
            joint_scorer = TabICLJointScorer(
                sampler=joint_sampler,
                target_class=int(target_class),
                n_permutations=tabicl_joint_permutations,
            )

        category_distribution = None
        if categorical_codec is not None and context.grouped_actionable:
            category_distribution = _build_category_distribution(
                sampler_context=sampler_context,
                categorical_codec=categorical_codec,
                target_class=int(target_class),
                confidence_grid=confidence_grid,
            )

        estimator_params = getattr(sampler_context, "estimator_params", {})
        return TabICLGeneratorPointBackend(
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


def run_tabicl_benchmark(
    context: BenchmarkDatasetContext,
    *,
    tau: float = TAU,
    temperature: float = DEFAULT_TEMPERATURE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    candidate_quantiles: Sequence[float] | None = None,
    confidence_quantiles: Sequence[float] | None = None,
    cf_mode: str = "sparse",
    tabicl_joint_permutations: int = 1,
    max_validity_steps: int | None = None,
    allow_revisits: bool = True,
    joint_shortlist_size: int = 16,
    max_extra_actions: int = 1,
    min_joint_log_gain: float = 0.0,
    n_counterfactuals: int = 3,
    diversity_beam_width: int = 8,
    diversity_candidate_pool_size: int = 16,
    diversity_max_extra_actions: int = 2,
    diversity_max_gower_ratio: float = 1.5,
    diversity_max_gower_increase: float = 0.02,
    cache_dir: Path | None = None,
) -> TabICLBenchmarkRun:
    """Run the benchmark protocol through the stable TabICL generator API."""
    candidate_quantiles = (
        None
        if candidate_quantiles is None
        else tuple(float(value) for value in candidate_quantiles)
    )
    confidence_quantiles = (
        None
        if confidence_quantiles is None
        else tuple(float(value) for value in confidence_quantiles)
    )
    if cf_mode not in CF_MODES:
        raise ValueError(f"cf_mode must be one of {CF_MODES}, got {cf_mode!r}")

    point_backend_factory = _build_point_backend_factory(
        context,
        confidence_quantiles=confidence_quantiles,
        cf_mode=cf_mode,
        tabicl_joint_permutations=tabicl_joint_permutations,
        n_estimators=n_estimators,
        temperature=temperature,
        cache_dir=cache_dir,
    )
    result = generate_counterfactual_batch(
        TabICLGeneratorInputs(
            factuals=context.X_test,
            targets=context.y_target,
            numerical_columns=context.scalar_actionable,
            categorical_groups=context.grouped_actionable,
            immutable_idx=context.immutable_idx,
            feature_domains=infer_feature_domains(context.bundle.X_train),
        ),
        discriminator=context.disc_model,
        config=TabICLGeneratorConfig(
            tau=tau,
            temperature=temperature,
            candidate_quantiles=candidate_quantiles,
            confidence_quantiles=confidence_quantiles,
            cf_mode=cf_mode,
            tabicl_joint_permutations=tabicl_joint_permutations,
            max_validity_steps=max_validity_steps,
            allow_revisits=allow_revisits,
            joint_shortlist_size=joint_shortlist_size,
            max_extra_actions=max_extra_actions,
            min_joint_log_gain=min_joint_log_gain,
            diversity_config=DiverseBeamSearchConfig(
                n_counterfactuals=n_counterfactuals,
                beam_width=diversity_beam_width,
                candidate_pool_size=diversity_candidate_pool_size,
                max_extra_actions=diversity_max_extra_actions,
                max_gower_ratio=diversity_max_gower_ratio,
                max_gower_increase=diversity_max_gower_increase,
            ),
            categorical_proposal_count=DEFAULT_CATEGORICAL_PROPOSAL_COUNT,
        ),
        point_backend_factory=point_backend_factory,
    )
    return TabICLBenchmarkRun(
        context=context,
        n_estimators=n_estimators,
        result=result,
    )
