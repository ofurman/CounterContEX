"""Typed configuration for the retained CounterContEx search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.diverse_search import DiverseBeamSearchConfig
from experiments.zeroshot_cf.generator import (
    CF_MODES,
    DEFAULT_CATEGORICAL_PROPOSAL_COUNT,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_TEMPERATURE,
    TabICLGeneratorConfig,
)
from experiments.zeroshot_cf.retained_config import TAU

NUMERICAL_DECODERS = ("historical", "mode", "grid", "iid", "top-k", "top-p")
CATEGORICAL_DECODERS = ("historical", "greedy", "iid", "top-k", "top-p")
PROPOSAL_RNG_SCHEME = "sha256-canonical-v1"
PROJECTION_ACCOUNTING_POLICY = "project-first-no-refill-v1"
GUIDANCE_POLICIES = ("none", "pi-beta", "classifier-top-b")
GUIDANCE_RESAMPLING_POLICY = "sir-with-replacement-v1"
GUIDANCE_COMMON_RNG_SCHEME = "identical-state-action-crn-v1"


@dataclass(frozen=True)
class CounterContExSearchConfig:
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
    numerical_decoder: str = "historical"
    numerical_proposal_budget: int | None = None
    categorical_decoder: str = "historical"
    categorical_proposal_budget: int | None = None
    truncation_k: int = 5
    truncation_p: float = 0.9
    strict_proposal_budget: bool = False
    proposal_rng_scheme: str = PROPOSAL_RNG_SCHEME
    proposal_accounting_policy: str = PROJECTION_ACCOUNTING_POLICY
    guidance_policy: str = "none"
    guidance_beta: float = 0.0
    guidance_pool_size: int = 64
    guidance_epsilon: float = 1e-12
    guidance_with_replacement: bool = True
    guidance_score_cache: bool = True
    guidance_resampling_policy: str = GUIDANCE_RESAMPLING_POLICY
    guidance_common_rng_scheme: str = GUIDANCE_COMMON_RNG_SCHEME

    def __post_init__(self) -> None:
        if self.candidate_quantiles is not None:
            quantiles = tuple(float(value) for value in self.candidate_quantiles)
            if (
                not quantiles
                or any(not np.isfinite(value) for value in quantiles)
                or any(not 0.0 < value < 1.0 for value in quantiles)
                or any(
                    right <= left
                    for left, right in zip(quantiles, quantiles[1:], strict=False)
                )
            ):
                raise ValueError(
                    "candidate_quantiles must be strictly increasing inside (0, 1)"
                )
            object.__setattr__(self, "candidate_quantiles", quantiles)
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
        if self.numerical_decoder not in NUMERICAL_DECODERS:
            raise ValueError(
                f"numerical_decoder must be one of {NUMERICAL_DECODERS}"
            )
        if self.categorical_decoder not in CATEGORICAL_DECODERS:
            raise ValueError(
                f"categorical_decoder must be one of {CATEGORICAL_DECODERS}"
            )
        for name, value in (
            ("numerical_proposal_budget", self.numerical_proposal_budget),
            ("categorical_proposal_budget", self.categorical_proposal_budget),
        ):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.truncation_k, int) or isinstance(
            self.truncation_k, bool
        ) or self.truncation_k < 1:
            raise ValueError("truncation_k must be a positive integer")
        if not 0.0 < self.truncation_p <= 1.0:
            raise ValueError("truncation_p must lie in (0, 1]")
        if self.proposal_rng_scheme != PROPOSAL_RNG_SCHEME:
            raise ValueError(
                f"proposal_rng_scheme must be {PROPOSAL_RNG_SCHEME!r}"
            )
        if self.proposal_accounting_policy != PROJECTION_ACCOUNTING_POLICY:
            raise ValueError(
                "proposal_accounting_policy must be "
                f"{PROJECTION_ACCOUNTING_POLICY!r}"
            )
        if self.guidance_policy not in GUIDANCE_POLICIES:
            raise ValueError(f"guidance_policy must be one of {GUIDANCE_POLICIES}")
        if not np.isfinite(self.guidance_beta) or self.guidance_beta < 0.0:
            raise ValueError("guidance_beta must be finite and non-negative")
        if (
            not isinstance(self.guidance_pool_size, int)
            or isinstance(self.guidance_pool_size, bool)
            or self.guidance_pool_size < 1
        ):
            raise ValueError("guidance_pool_size must be a positive integer")
        if not np.isfinite(self.guidance_epsilon) or not (
            0.0 < self.guidance_epsilon <= 1.0
        ):
            raise ValueError("guidance_epsilon must lie in (0, 1]")
        if not self.guidance_with_replacement:
            raise ValueError("classifier guidance requires with-replacement sampling")
        if not self.guidance_score_cache:
            raise ValueError("classifier guidance requires score caching")
        if self.guidance_resampling_policy != GUIDANCE_RESAMPLING_POLICY:
            raise ValueError(
                f"guidance_resampling_policy must be {GUIDANCE_RESAMPLING_POLICY!r}"
            )
        if self.guidance_common_rng_scheme != GUIDANCE_COMMON_RNG_SCHEME:
            raise ValueError(
                f"guidance_common_rng_scheme must be {GUIDANCE_COMMON_RNG_SCHEME!r}"
            )
        if self.guidance_policy == "none" and (
            self.guidance_beta != 0.0
            or self.guidance_pool_size != 64
            or self.guidance_epsilon != 1e-12
        ):
            raise ValueError(
                "guidance beta, pool size, and epsilon require a guidance policy"
            )

        decoders_are_historical = (
            self.numerical_decoder == "historical"
            and self.categorical_decoder == "historical"
        )
        if not self.strict_proposal_budget:
            if self.guidance_policy != "none":
                raise ValueError("classifier guidance requires strict proposal mode")
            if not decoders_are_historical:
                raise ValueError(
                    "non-historical proposal decoders require "
                    "strict_proposal_budget=True"
                )
            if (
                self.numerical_proposal_budget is not None
                or self.categorical_proposal_budget is not None
            ):
                raise ValueError(
                    "proposal budgets require strict_proposal_budget=True"
                )
            return
        if decoders_are_historical or "historical" in {
            self.numerical_decoder,
            self.categorical_decoder,
        }:
            raise ValueError(
                "strict proposal mode requires explicit numerical and categorical "
                "decoders"
            )
        if self.numerical_proposal_budget is None:
            raise ValueError("strict proposal mode requires numerical_proposal_budget")
        if self.categorical_proposal_budget is None:
            raise ValueError(
                "strict proposal mode requires categorical_proposal_budget"
            )
        if self.numerical_decoder == "mode" and self.numerical_proposal_budget != 1:
            raise ValueError("mode is a one-proposal numerical reference")
        if self.numerical_decoder == "top-k" and self.truncation_k > 256:
            raise ValueError("numerical top-k cannot retain more than 256 bins")
        if self.numerical_decoder == "grid":
            if self.candidate_quantiles is None:
                raise ValueError("grid decoder requires candidate_quantiles")
            if len(self.candidate_quantiles) != self.numerical_proposal_budget:
                raise ValueError(
                    "grid candidate_quantiles must match numerical_proposal_budget"
                )
        elif self.candidate_quantiles is not None:
            raise ValueError("candidate_quantiles apply only to the grid decoder")
        if (
            self.categorical_decoder == "greedy"
            and self.categorical_proposal_budget != 1
        ):
            raise ValueError("greedy is a one-proposal categorical reference")
        if self.categorical_proposal_count != DEFAULT_CATEGORICAL_PROPOSAL_COUNT:
            raise ValueError(
                "strict categorical decoding uses categorical_proposal_budget, not "
                "categorical_proposal_count"
            )
        if self.guidance_policy != "none":
            if self.numerical_decoder != "iid" or self.categorical_decoder != "iid":
                raise ValueError("classifier guidance requires IID q pool decoders")
            if self.guidance_pool_size < max(
                self.numerical_proposal_budget,
                self.categorical_proposal_budget,
            ):
                raise ValueError(
                    "guidance_pool_size must be at least proposal budget B"
                )
            if self.guidance_policy == "classifier-top-b" and self.guidance_beta != 0.0:
                raise ValueError("classifier-top-b does not use guidance_beta")


@dataclass(frozen=True)
class CounterContExDiversityConfig:
    """Settings for bounded beam generation and fixed-size DPP selection."""

    beam_width: int = 8
    candidate_pool_size: int = 16
    max_extra_actions: int = 2
    max_gower_ratio: float = 1.5
    max_gower_increase: float = 0.02
    selection_strategy: str = "dpp"

    def build(
        self, n_counterfactuals: int, *, selection_seed: int = 0
    ) -> DiverseBeamSearchConfig:
        return DiverseBeamSearchConfig(
            n_counterfactuals=n_counterfactuals,
            beam_width=self.beam_width,
            candidate_pool_size=self.candidate_pool_size,
            max_extra_actions=self.max_extra_actions,
            max_gower_ratio=self.max_gower_ratio,
            max_gower_increase=self.max_gower_increase,
            selection_strategy=self.selection_strategy,
            selection_seed=selection_seed,
        )

    def __post_init__(self) -> None:
        # Reuse the retained implementation's validation without owning k here.
        self.build(1)


@dataclass(frozen=True)
class CounterContExFoundationConfig:
    """Settings for the TabICL proposal and joint-density runtime."""

    n_estimators: int = DEFAULT_N_ESTIMATORS
    temperature: float = DEFAULT_TEMPERATURE
    confidence_quantiles: tuple[float, ...] | None = None
    tabicl_joint_permutations: int = 1
    cache_dir: Path | None = None
    backend: str = "tabicl"
    context_size: int = 512
    context_labels: str = "predictions"

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("foundation backend must be non-empty")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.tabicl_joint_permutations < 1:
            raise ValueError("tabicl_joint_permutations must be positive")
        if self.context_size < 1:
            raise ValueError("context_size must be positive")
        if self.context_labels not in {"predictions", "true"}:
            raise ValueError("context_labels must be 'predictions' or 'true'")


@dataclass(frozen=True)
class CounterContExConfig:
    """Complete serializable method configuration."""

    search: CounterContExSearchConfig = field(default_factory=CounterContExSearchConfig)
    diversity: CounterContExDiversityConfig = field(
        default_factory=CounterContExDiversityConfig
    )
    foundation: CounterContExFoundationConfig = field(
        default_factory=CounterContExFoundationConfig
    )

    def __post_init__(self) -> None:
        if (
            self.foundation.confidence_quantiles is not None
            and self.search.candidate_quantiles is None
        ):
            raise ValueError("confidence_quantiles require candidate_quantiles")
        if self.search.strict_proposal_budget:
            if self.foundation.confidence_quantiles is not None:
                raise ValueError(
                    "strict proposal experiments disable confidence conditioning"
                )
            if self.search.cf_mode != "sparse":
                raise ValueError("strict proposal experiments require sparse search")
            if self.search.numerical_decoder in {
                "iid",
                "top-k",
                "top-p",
            } and not np.isclose(self.foundation.temperature, 1.0):
                raise ValueError(
                    "full-distribution numerical sampling requires temperature=1.0"
                )

    def generator_config(
        self, n_counterfactuals: int, *, seed: int = 0
    ) -> TabICLGeneratorConfig:
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
            diversity_config=self.diversity.build(
                n_counterfactuals, selection_seed=seed
            ),
            categorical_proposal_count=self.search.categorical_proposal_count,
            strict_proposal_budget=self.search.strict_proposal_budget,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe nested representation."""
        values = asdict(self)
        cache_dir = self.foundation.cache_dir
        values["foundation"]["cache_dir"] = (
            None if cache_dir is None else str(cache_dir)
        )
        return values
