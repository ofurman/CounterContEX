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
)
from experiments.zeroshot_cf.methods.base import MethodCapabilities, json_diagnostic
from experiments.zeroshot_cf.methods.dicoflex.backend import (
    DiCoFlexBackendInputs,
    prepare_backend,
)
from experiments.zeroshot_cf.methods.dicoflex.backends.base import (
    PreparedBackend,
    ProposalBackend,
    validate_backend_capabilities,
)
from experiments.zeroshot_cf.methods.dicoflex.backends.empirical import EmpiricalBackend
from experiments.zeroshot_cf.methods.dicoflex.config import DiCoFlexConfig
from experiments.zeroshot_cf.methods.dicoflex.search import generate_with_backend


def adapt_generator_result(
    result: TabICLGeneratorResult,
    *,
    seed: int,
    proposal_backend: str = "conditional_density",
    actionable_idx: tuple[int, ...] = (),
    immutable_idx: tuple[int, ...] = (),
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

    def diagnostic_value(name: str, index: int, default: Any) -> Any:
        values = getattr(diagnostics, name, None)
        return default if values is None else values[index]

    def history_float(record: dict[str, Any] | None, key: str) -> float:
        value = None if record is None else record.get(key)
        return float("nan") if value is None else float(value)

    point_diagnostics: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        history = diagnostic_value("history_per_point", index, ())
        attempt_history = diagnostic_value("attempt_history_per_point", index, ())
        diverse_histories = diagnostic_value("diverse_histories_per_point", index, ())
        initial_valid_record = next(
            (
                step
                for step in history
                if isinstance(step, dict) and step.get("immediate_valid")
            ),
            None,
        )
        final_record = history[-1] if history and isinstance(history[-1], dict) else {}
        point_diagnostics.append(
            json_diagnostic(
                {
                    "returned_count": int(count),
                    "flipped": bool(diagnostics.flipped_per_point[index]),
                    "changed_columns": list(diagnostics.changed_per_point[index]),
                    "steps": int(diagnostics.steps_per_point[index]),
                    "validity_steps": int(diagnostics.validity_steps_per_point[index]),
                    "refinement_steps": int(
                        diagnostics.refinement_steps_per_point[index]
                    ),
                    "accepted_refinement_count": int(
                        diagnostics.accepted_refinement_count_per_point[index]
                    ),
                    "history": json_diagnostic(history),
                    "attempt_history": json_diagnostic(attempt_history),
                    "diverse_histories": json_diagnostic(diverse_histories),
                    "attempt_steps": len(attempt_history),
                    "initial_valid_step": diagnostic_value(
                        "initial_valid_step_per_point", index, None
                    ),
                    "initial_sparse_action_count": int(
                        diagnostic_value(
                            "initial_sparse_action_count_per_point", index, -1
                        )
                    ),
                    "final_action_count": int(
                        diagnostic_value(
                            "final_action_count_per_point",
                            index,
                            len(diagnostics.changed_per_point[index]),
                        )
                    ),
                    "initial_tabicl_joint_log_density": float(
                        diagnostic_value(
                            "initial_tabicl_joint_log_density_per_point", index, np.nan
                        )
                    ),
                    "final_tabicl_joint_log_density": float(
                        diagnostic_value(
                            "final_tabicl_joint_log_density_per_point", index, np.nan
                        )
                    ),
                    "tabicl_joint_log_density_gain": float(
                        diagnostic_value(
                            "tabicl_joint_log_density_gain_per_point", index, np.nan
                        )
                    ),
                    "joint_scoring_batch_count": int(
                        diagnostic_value(
                            "joint_scoring_batch_count_per_point", index, 0
                        )
                    ),
                    "joint_rows_scored": int(
                        diagnostic_value("joint_rows_scored_per_point", index, 0)
                    ),
                    "extra_actions": int(
                        diagnostic_value("extra_actions_per_point", index, 0)
                    ),
                    "refinement_stopping_reason": str(
                        diagnostic_value(
                            "refinement_stopping_reason_per_point", index, "not_started"
                        )
                    ),
                    "initial_valid_action_sparsity": history_float(
                        initial_valid_record, "action_sparsity"
                    ),
                    "initial_valid_grouped_gower": history_float(
                        initial_valid_record, "grouped_gower"
                    ),
                    "final_action_sparsity": history_float(
                        final_record if history else {"action_sparsity": 0.0},
                        "action_sparsity",
                    ),
                    "first_action_type": (
                        history[0].get("action_type", "numerical")
                        if history and isinstance(history[0], dict)
                        else "numerical"
                    ),
                    "candidate_pool_count": int(
                        diagnostics.diverse_candidate_pool_count_per_point[index]
                    ),
                    "search_depth": int(
                        diagnostics.diverse_search_depth_per_point[index]
                    ),
                    "target_probability": float(
                        diagnostics.target_probability_per_point[index]
                    ),
                    "point_runtime_s": float(diagnostics.point_runtime_s[index]),
                    "joint_scoring_runtime_s": float(
                        diagnostics.joint_scoring_runtime_s_per_point[index]
                    ),
                }
            )
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
            "proposal_backend": proposal_backend,
            "actionable_idx": [int(column) for column in actionable_idx],
            "immutable_idx": [int(column) for column in immutable_idx],
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
    proposal_backend: ProposalBackend | None = None
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
        if self.proposal_backend is None:
            if self.config.foundation.backend == "tabicl":
                inputs = DiCoFlexBackendInputs(
                    X_reference=context.X_reference,
                    categorical_groups=context.feature_schema.categorical_groups,
                    actionable_groups=context.feature_schema.actionable_groups,
                    oracle=context.oracle,
                )
                backend = prepare_backend(inputs, self.config)
            elif self.config.foundation.backend == "empirical":
                backend = EmpiricalBackend().prepare(context)
            else:
                raise ValueError(
                    f"unknown DiCoFlex proposal backend: "
                    f"{self.config.foundation.backend!r}"
                )
        else:
            if self.config.foundation.backend not in {
                "injected",
                self.proposal_backend.backend_id,
            }:
                raise ValueError(
                    "foundation backend does not match the injected proposal backend"
                )
            backend = self.proposal_backend.prepare(context)
        validate_backend_capabilities(
            backend.capabilities,
            needs_confidence=self.config.foundation.confidence_quantiles is not None,
            needs_categorical=bool(context.feature_schema.actionable_groups),
            needs_joint=self.config.search.cf_mode == "data_plausible",
        )
        return PreparedDiCoFlexMethod(
            context=context,
            config=self.config,
            backend=backend,
        )


@dataclass(frozen=True)
class PreparedDiCoFlexMethod:
    context: MethodContext
    config: DiCoFlexConfig
    backend: PreparedBackend

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.factuals.shape[1] != self.context.X_reference.shape[1]:
            raise ValueError("request feature width does not match method context")
        schema = self.context.feature_schema
        retained = generate_with_backend(
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
            config=self.config,
            backend=self.backend,
            seed=request.seed,
            n_counterfactuals=request.n_counterfactuals,
        )
        proposal_backend = (
            "conditional_density"
            if self.backend.backend_id == "tabicl"
            else self.backend.backend_id
        )
        return adapt_generator_result(
            retained,
            seed=request.seed,
            proposal_backend=proposal_backend,
            actionable_idx=(
                tuple(schema.actionable_scalars)
                + tuple(
                    column
                    for group in schema.actionable_groups
                    for column in group.columns
                )
            ),
            immutable_idx=tuple(schema.immutable),
        )
