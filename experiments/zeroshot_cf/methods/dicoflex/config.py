"""Typed configuration for the retained DiCoFlex search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from experiments.zeroshot_cf.diverse_search import DiverseBeamSearchConfig
from experiments.zeroshot_cf.generator import (
    CF_MODES,
    DEFAULT_CATEGORICAL_PROPOSAL_COUNT,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_TEMPERATURE,
    TabICLGeneratorConfig,
)
from experiments.zeroshot_cf.retained_config import TAU


@dataclass(frozen=True)
class DiCoFlexSearchConfig:
    """Settings for greedy validity search and optional refinement."""

    tau: float = TAU
    candidate_quantiles: tuple[float, ...] | None = None
    cf_mode: str = "sparse"
    max_validity_steps: int | None = None
    allow_revisits: bool = True
    joint_shortlist_size: int = 16
    max_extra_actions: int = 1
    min_joint_log_gain: float = 0.0
    categorical_proposal_count: int = DEFAULT_CATEGORICAL_PROPOSAL_COUNT

    def __post_init__(self) -> None:
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be between zero and one")
        if self.cf_mode not in CF_MODES:
            raise ValueError(f"cf_mode must be one of {CF_MODES}, got {self.cf_mode!r}")
        if self.max_validity_steps is not None and self.max_validity_steps < 1:
            raise ValueError("max_validity_steps must be at least 1")
        if self.joint_shortlist_size < 1:
            raise ValueError("joint_shortlist_size must be at least 1")
        if self.max_extra_actions < 0:
            raise ValueError("max_extra_actions must be non-negative")
        if self.min_joint_log_gain < 0:
            raise ValueError("min_joint_log_gain must be non-negative")
        if self.categorical_proposal_count < 1:
            raise ValueError("categorical_proposal_count must be at least 1")


@dataclass(frozen=True)
class DiCoFlexDiversityConfig:
    """Settings for bounded beam generation and fixed-size DPP selection."""

    beam_width: int = 8
    candidate_pool_size: int = 16
    max_extra_actions: int = 2
    max_gower_ratio: float = 1.5
    max_gower_increase: float = 0.02

    def build(self, n_counterfactuals: int) -> DiverseBeamSearchConfig:
        return DiverseBeamSearchConfig(
            n_counterfactuals=n_counterfactuals,
            beam_width=self.beam_width,
            candidate_pool_size=self.candidate_pool_size,
            max_extra_actions=self.max_extra_actions,
            max_gower_ratio=self.max_gower_ratio,
            max_gower_increase=self.max_gower_increase,
        )

    def __post_init__(self) -> None:
        # Reuse the retained implementation's validation without owning k here.
        self.build(1)


@dataclass(frozen=True)
class DiCoFlexFoundationConfig:
    """Settings for the TabICL proposal and joint-density runtime."""

    n_estimators: int = DEFAULT_N_ESTIMATORS
    temperature: float = DEFAULT_TEMPERATURE
    confidence_quantiles: tuple[float, ...] | None = None
    tabicl_joint_permutations: int = 1
    cache_dir: Path | None = None
    backend: str = "tabicl"

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("foundation backend must be non-empty")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.tabicl_joint_permutations < 1:
            raise ValueError("tabicl_joint_permutations must be positive")


@dataclass(frozen=True)
class DiCoFlexConfig:
    """Complete serializable method configuration."""

    search: DiCoFlexSearchConfig = field(default_factory=DiCoFlexSearchConfig)
    diversity: DiCoFlexDiversityConfig = field(default_factory=DiCoFlexDiversityConfig)
    foundation: DiCoFlexFoundationConfig = field(
        default_factory=DiCoFlexFoundationConfig
    )

    def __post_init__(self) -> None:
        if (
            self.foundation.confidence_quantiles is not None
            and self.search.candidate_quantiles is None
        ):
            raise ValueError("confidence_quantiles require candidate_quantiles")

    def generator_config(self, n_counterfactuals: int) -> TabICLGeneratorConfig:
        """Translate benchmark-facing settings to the retained search config."""
        return TabICLGeneratorConfig(
            tau=self.search.tau,
            temperature=self.foundation.temperature,
            candidate_quantiles=self.search.candidate_quantiles,
            confidence_quantiles=self.foundation.confidence_quantiles,
            cf_mode=self.search.cf_mode,
            tabicl_joint_permutations=self.foundation.tabicl_joint_permutations,
            max_validity_steps=self.search.max_validity_steps,
            allow_revisits=self.search.allow_revisits,
            joint_shortlist_size=self.search.joint_shortlist_size,
            max_extra_actions=self.search.max_extra_actions,
            min_joint_log_gain=self.search.min_joint_log_gain,
            diversity_config=self.diversity.build(n_counterfactuals),
            categorical_proposal_count=self.search.categorical_proposal_count,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe nested representation."""
        values = asdict(self)
        cache_dir = self.foundation.cache_dir
        values["foundation"]["cache_dir"] = (
            None if cache_dir is None else str(cache_dir)
        )
        return values
