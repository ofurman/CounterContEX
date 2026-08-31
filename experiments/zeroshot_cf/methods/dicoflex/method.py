"""Benchmark-facing lifecycle for the retained DiCoFlex generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from experiments.zeroshot_cf.core.contracts import (
    GenerationRequest,
    GenerationResult,
    MethodContext,
)
from experiments.zeroshot_cf.generator import (
    TabICLGeneratorInputs,
    TabICLGeneratorResult,
    generate_counterfactual_batch,
)
from experiments.zeroshot_cf.methods.base import MethodCapabilities
from experiments.zeroshot_cf.methods.dicoflex.backend import (
    DiCoFlexBackendInputs,
    PreparedDiCoFlexBackend,
    prepare_backend,
)
from experiments.zeroshot_cf.methods.dicoflex.config import DiCoFlexConfig


def adapt_generator_result(
    result: TabICLGeneratorResult,
    *,
    seed: int,
) -> GenerationResult:
    """Adapt retained search output without turning failures into padding."""
    diagnostics = result.diagnostics
    candidates = np.asarray(result.counterfactual_sets, dtype=np.float64).copy()
    counts = np.asarray(
        diagnostics.diverse_available_count_per_point,
        dtype=np.int64,
    )
    if candidates.ndim != 3 or len(counts) != len(candidates):
        raise ValueError("retained counterfactual sets have inconsistent dimensions")
    if np.any(counts < 0) or np.any(counts > candidates.shape[1]):
        raise ValueError("retained available counts are outside the candidate rank")
    available = np.arange(candidates.shape[1])[None, :] < counts[:, None]
    candidates[~available] = np.nan

    point_diagnostics: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        point_diagnostics.append(
            {
                "returned_count": int(count),
                "flipped": bool(diagnostics.flipped_per_point[index]),
                "changed_columns": list(diagnostics.changed_per_point[index]),
                "steps": int(diagnostics.steps_per_point[index]),
                "validity_steps": int(diagnostics.validity_steps_per_point[index]),
                "refinement_steps": int(diagnostics.refinement_steps_per_point[index]),
                "accepted_refinement_count": int(
                    diagnostics.accepted_refinement_count_per_point[index]
                ),
                "candidate_pool_count": int(
                    diagnostics.diverse_candidate_pool_count_per_point[index]
                ),
                "search_depth": int(diagnostics.diverse_search_depth_per_point[index]),
                "target_probability": float(
                    diagnostics.target_probability_per_point[index]
                ),
                "point_runtime_s": float(diagnostics.point_runtime_s[index]),
                "joint_scoring_runtime_s": float(
                    diagnostics.joint_scoring_runtime_s_per_point[index]
                ),
            }
        )

    artifacts = {
        "method.best_effort": np.asarray(result.counterfactuals, dtype=np.float64),
        "method.sparse_counterfactuals": np.asarray(
            result.sparse_counterfactuals, dtype=np.float64
        ),
        "method.available_count": counts,
    }
    return GenerationResult(
        candidates=candidates,
        available=available,
        point_diagnostics=tuple(point_diagnostics),
        run_diagnostics={
            "seed": seed,
            "proposal_backend": "conditional_density",
            "joint_scoring": (
                "one_shot" if diagnostics.cf_mode == "data_plausible" else "disabled"
            ),
            "cache": {
                "conditional_estimator": diagnostics.conditional_estimator_cache,
                "key_value": diagnostics.tabicl_kv_cache,
            },
            "runtime_s": float(diagnostics.runtime_s),
        },
        artifacts=artifacts,
    )


@dataclass(frozen=True)
class DiCoFlexMethod:
    config: DiCoFlexConfig = DiCoFlexConfig()
    method_id = "dicoflex"
    capabilities = MethodCapabilities(
        supports_categorical=True,
        enforces_actionability=True,
        supports_multiple_counterfactuals=True,
        requires_probabilities=True,
        optional_dependencies=("tabicl",),
    )

    def config_dict(self) -> dict[str, Any]:
        return self.config.as_dict()

    def prepare(self, context: MethodContext) -> PreparedDiCoFlexMethod:
        inputs = DiCoFlexBackendInputs(
            X_reference=context.X_reference,
            categorical_groups=context.feature_schema.categorical_groups,
            actionable_groups=context.feature_schema.actionable_groups,
            oracle=context.oracle,
        )
        return PreparedDiCoFlexMethod(
            context=context,
            config=self.config,
            backend=prepare_backend(inputs, self.config),
        )


@dataclass(frozen=True)
class PreparedDiCoFlexMethod:
    context: MethodContext
    config: DiCoFlexConfig
    backend: PreparedDiCoFlexBackend

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.factuals.shape[1] != self.context.X_reference.shape[1]:
            raise ValueError("request feature width does not match method context")
        schema = self.context.feature_schema
        retained = generate_counterfactual_batch(
            TabICLGeneratorInputs(
                factuals=request.factuals,
                targets=request.targets,
                numerical_columns=schema.actionable_scalars,
                categorical_groups=schema.actionable_groups,
                immutable_idx=schema.immutable,
                feature_domains=(
                    schema.domains.lower,
                    schema.domains.upper,
                    dict(schema.domains.discrete),
                ),
            ),
            discriminator=self.context.oracle,
            config=self.config.generator_config(request.n_counterfactuals),
            point_backend_factory=self.backend.point_backend_factory(seed=request.seed),
        )
        return adapt_generator_result(retained, seed=request.seed)
