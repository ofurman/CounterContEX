"""Neutral runtime adapter from the benchmark protocol to the stable TabICL API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from experiments.zeroshot_cf.benchmark_protocol import BenchmarkDatasetContext
from experiments.zeroshot_cf.candidate_domains import infer_feature_domains
from experiments.zeroshot_cf.diverse_search import DiverseBeamSearchConfig
from experiments.zeroshot_cf.generator import (
    CF_MODES,
    DEFAULT_CATEGORICAL_PROPOSAL_COUNT,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_TEMPERATURE,
    TabICLGeneratorConfig,
    TabICLGeneratorInputs,
    TabICLGeneratorResult,
    generate_counterfactual_batch,
)
from experiments.zeroshot_cf.methods.dicoflex.backend import (
    DiCoFlexBackendInputs,
    prepare_backend,
)
from experiments.zeroshot_cf.methods.dicoflex.config import (
    DiCoFlexConfig,
    DiCoFlexFoundationConfig,
    DiCoFlexSearchConfig,
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


def _build_point_backend_factory(
    context: BenchmarkDatasetContext,
    *,
    confidence_quantiles: tuple[float, ...] | None,
    cf_mode: str,
    tabicl_joint_permutations: int,
    n_estimators: int,
    temperature: float,
    cache_dir: Path | None,
    seed: int = 42,
):
    # This compatibility boundary intentionally keeps seed 42 as its default.
    config = DiCoFlexConfig(
        search=DiCoFlexSearchConfig(
            candidate_quantiles=(0.5,) if confidence_quantiles is not None else None,
            cf_mode=cf_mode,
        ),
        foundation=DiCoFlexFoundationConfig(
            n_estimators=n_estimators,
            temperature=temperature,
            confidence_quantiles=confidence_quantiles,
            tabicl_joint_permutations=tabicl_joint_permutations,
            cache_dir=cache_dir,
        ),
    )
    backend = prepare_backend(
        DiCoFlexBackendInputs(
            X_reference=context.bundle.X_train,
            categorical_groups=tuple(context.categorical_groups),
            actionable_groups=tuple(context.grouped_actionable),
            oracle=context.disc_model,
        ),
        config,
    )
    return backend.point_backend_factory(seed=seed)


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
