"""Wachter coordinate-search and Growing Spheres method adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.baseline_common import (
    ActionUnit,
    build_action_units,
    contract_scalar_actions,
    prune_counterfactual_actions,
)
from experiments.zeroshot_cf.core.contracts import GenerationRequest, MethodContext
from experiments.zeroshot_cf.core.validation import target_probabilities
from experiments.zeroshot_cf.methods.base import (
    MethodCapabilities,
    canonical_single_result,
    require_single_counterfactual,
)
from experiments.zeroshot_cf.retained_config import TAU

WACHTER_QUANTILES = tuple(i / 20 for i in range(1, 20))


def _is_valid(
    disc_model: Any,
    rows: np.ndarray,
    target: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.atleast_2d(rows)
    probabilities = target_probabilities(
        disc_model,
        matrix,
        np.full(len(matrix), target),
    )
    predictions = np.asarray(disc_model.predict(matrix), dtype=int)
    return (predictions == target) & (probabilities >= tau), probabilities


def _action_sparsity(
    rows: np.ndarray,
    factual: np.ndarray,
    scalar_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
) -> np.ndarray:
    matrix = np.atleast_2d(rows)
    counts = np.zeros(len(matrix), dtype=int)
    if scalar_columns:
        columns = np.asarray(scalar_columns, dtype=int)
        counts += np.count_nonzero(
            ~np.isclose(matrix[:, columns], factual[columns]),
            axis=1,
        )
    for group in categorical_groups:
        columns = list(group.columns)
        counts += np.argmax(matrix[:, columns], axis=1) != int(
            np.argmax(factual[columns])
        )
    return counts


def _changed_units(
    factual: np.ndarray,
    candidate: np.ndarray,
    action_units: Sequence[ActionUnit],
) -> list[ActionUnit]:
    return [
        unit
        for unit in action_units
        if not np.allclose(
            factual[list(unit.columns)],
            candidate[list(unit.columns)],
        )
    ]


def _coordinate_trials(
    current: np.ndarray,
    scalar_values: Mapping[int, np.ndarray],
    categorical_groups: Sequence[OneHotActionGroup],
) -> np.ndarray:
    trials: list[np.ndarray] = []
    for column, values in scalar_values.items():
        for value in values:
            if np.isclose(value, current[column]):
                continue
            trial = current.copy()
            trial[column] = value
            trials.append(trial)
    for group in categorical_groups:
        columns = list(group.columns)
        current_category = int(np.argmax(current[columns]))
        for category, column in enumerate(columns):
            if category == current_category:
                continue
            trial = current.copy()
            trial[columns] = 0.0
            trial[column] = 1.0
            trials.append(trial)
    return np.stack(trials) if trials else np.empty((0, len(current)))


def wachter_coordinate_counterfactual(
    disc_model: Any,
    factual: np.ndarray,
    target: int,
    scalar_values: Mapping[int, np.ndarray],
    categorical_groups: Sequence[OneHotActionGroup],
    action_units: Sequence[ActionUnit],
    *,
    tau: float = TAU,
    loss_weights: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    max_steps_per_weight: int = 12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Black-box mixed-data coordinate minimization of a Wachter objective."""
    factual = np.asarray(factual, dtype=np.float64)
    best_valid: np.ndarray | None = None
    best_valid_key = (np.inf, np.inf)
    best_probability = float(
        target_probabilities(disc_model, factual.reshape(1, -1), np.array([target]))[0]
    )
    best_probability_row = factual.copy()
    evaluations = 1

    for loss_weight in loss_weights:
        current = factual.copy()
        for _ in range(max_steps_per_weight):
            trials = _coordinate_trials(current, scalar_values, categorical_groups)
            if not len(trials):
                break
            evaluations += len(trials)
            valid, probabilities = _is_valid(disc_model, trials, target, tau)
            distances = np.abs(trials - factual).sum(axis=1)
            losses = np.maximum(0.0, tau - probabilities) ** 2
            objective = distances + float(loss_weight) * losses
            current_valid, current_probability = _is_valid(
                disc_model, current, target, tau
            )
            current_distance = float(np.abs(current - factual).sum())
            current_objective = (
                current_distance
                + float(loss_weight)
                * max(0.0, tau - float(current_probability[0])) ** 2
            )
            best = int(np.argmin(objective))
            if float(objective[best]) >= current_objective - 1e-12:
                break
            current = trials[best]

            probability_best = int(np.argmax(probabilities))
            if probabilities[probability_best] > best_probability:
                best_probability = float(probabilities[probability_best])
                best_probability_row = trials[probability_best].copy()

            if valid.any():
                eligible = np.flatnonzero(valid)
                sparsity = _action_sparsity(
                    trials[eligible],
                    factual,
                    tuple(scalar_values),
                    categorical_groups,
                )
                l2 = np.linalg.norm(trials[eligible] - factual, axis=1)
                order = np.lexsort((l2, sparsity))
                candidate = trials[int(eligible[order[0]])]
                key = (int(sparsity[order[0]]), float(l2[order[0]]))
                if key < best_valid_key:
                    best_valid = candidate.copy()
                    best_valid_key = key
            if bool(current_valid[0]):
                break

    candidate = best_probability_row if best_valid is None else best_valid
    if best_valid is not None:
        candidate = prune_counterfactual_actions(
            disc_model, factual, candidate, target, action_units, tau=tau
        )
        candidate = contract_scalar_actions(
            disc_model,
            factual,
            candidate,
            target,
            tuple(scalar_values),
            tau=tau,
        )
    valid, probabilities = _is_valid(disc_model, candidate, target, tau)
    return candidate, {
        "valid": bool(valid[0]),
        "target_probability": float(probabilities[0]),
        "evaluations": evaluations,
    }


def _sample_sphere_candidates(
    factual: np.ndarray,
    scalar_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    rng: np.random.Generator,
    n_candidates: int,
    radius: float,
) -> np.ndarray:
    trials = np.repeat(factual.reshape(1, -1), n_candidates, axis=0)
    if scalar_columns:
        columns = np.asarray(scalar_columns, dtype=int)
        directions = rng.normal(size=(n_candidates, len(columns)))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= np.maximum(norms, 1e-12)
        radial = radius * rng.uniform(0.5, 1.0, n_candidates) ** (
            1.0 / max(1, len(columns))
        )
        trials[:, columns] = np.clip(
            factual[columns] + directions * radial[:, None],
            0.0,
            1.0,
        )
    category_probability = min(0.5, max(0.02, radius / 3.0))
    for group in categorical_groups:
        columns = list(group.columns)
        factual_category = int(np.argmax(factual[columns]))
        change = rng.random(n_candidates) < category_probability
        alternatives = rng.integers(0, len(columns) - 1, n_candidates)
        alternatives += alternatives >= factual_category
        selected = np.flatnonzero(change)
        if len(selected):
            trials[np.ix_(selected, columns)] = 0.0
            trials[selected, np.asarray(columns)[alternatives[selected]]] = 1.0
    return trials


def growing_spheres_counterfactual(
    disc_model: Any,
    factual: np.ndarray,
    target: int,
    scalar_columns: Sequence[int],
    categorical_groups: Sequence[OneHotActionGroup],
    action_units: Sequence[ActionUnit],
    *,
    tau: float = TAU,
    n_candidates: int = 512,
    initial_radius: float = 0.05,
    radius_multiplier: float = 1.5,
    max_shells: int = 11,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mixed-data Growing Spheres with pruning and scalar contraction."""
    factual = np.asarray(factual, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    best_probability = float(
        target_probabilities(disc_model, factual.reshape(1, -1), np.array([target]))[0]
    )
    best_probability_row = factual.copy()
    candidate: np.ndarray | None = None
    evaluations = 1
    radius = initial_radius

    for _ in range(max_shells):
        trials = _sample_sphere_candidates(
            factual,
            scalar_columns,
            categorical_groups,
            rng,
            n_candidates,
            radius,
        )
        evaluations += len(trials)
        valid, probabilities = _is_valid(disc_model, trials, target, tau)
        probability_best = int(np.argmax(probabilities))
        if probabilities[probability_best] > best_probability:
            best_probability = float(probabilities[probability_best])
            best_probability_row = trials[probability_best].copy()
        if valid.any():
            eligible = np.flatnonzero(valid)
            sparsity = _action_sparsity(
                trials[eligible], factual, scalar_columns, categorical_groups
            )
            distances = np.linalg.norm(trials[eligible] - factual, axis=1)
            order = np.lexsort((distances, sparsity))
            candidate = trials[int(eligible[order[0]])]
            break
        radius *= radius_multiplier

    final_candidate: np.ndarray
    if candidate is None:
        final_candidate = best_probability_row
    else:
        final_candidate = prune_counterfactual_actions(
            disc_model, factual, candidate, target, action_units, tau=tau
        )
        final_candidate = contract_scalar_actions(
            disc_model,
            factual,
            final_candidate,
            target,
            scalar_columns,
            tau=tau,
        )
    valid, probabilities = _is_valid(disc_model, final_candidate, target, tau)
    return final_candidate, {
        "valid": bool(valid[0]),
        "target_probability": float(probabilities[0]),
        "evaluations": evaluations,
        "final_radius": radius,
    }


_CAPABILITIES = MethodCapabilities(
    supports_categorical=True,
    enforces_actionability=True,
    supports_multiple_counterfactuals=False,
    requires_probabilities=True,
    optional_dependencies=(),
)


@dataclass(frozen=True)
class WachterConfig:
    tau: float = TAU
    loss_weights: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    max_steps_per_weight: int = 12
    quantiles: tuple[float, ...] = WACHTER_QUANTILES

    def __post_init__(self) -> None:
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be between zero and one")
        if not self.loss_weights or any(weight <= 0 for weight in self.loss_weights):
            raise ValueError("loss_weights must be positive")
        if self.max_steps_per_weight <= 0:
            raise ValueError("max_steps_per_weight must be positive")
        if not self.quantiles or any(not 0 < value < 1 for value in self.quantiles):
            raise ValueError("quantiles must lie strictly between zero and one")


@dataclass(frozen=True)
class WachterMethod:
    config: WachterConfig = WachterConfig()
    method_id = "wachter"
    capabilities = _CAPABILITIES

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def prepare(self, context: MethodContext) -> PreparedWachterMethod:
        scalar_values = {
            int(column): np.unique(
                np.quantile(context.X_reference[:, column], self.config.quantiles)
            )
            for column in context.feature_schema.actionable_scalars
        }
        return PreparedWachterMethod(context, self.config, scalar_values)


@dataclass(frozen=True)
class PreparedWachterMethod:
    context: MethodContext
    config: WachterConfig
    scalar_values: dict[int, np.ndarray]

    def generate(self, request: GenerationRequest):
        require_single_counterfactual(request)
        if request.factuals.shape[1] != self.context.X_reference.shape[1]:
            raise ValueError("request feature width does not match method context")
        action_units = build_action_units(
            list(self.context.feature_schema.actionable_scalars),
            list(self.context.feature_schema.actionable_groups),
        )
        raw = np.empty_like(request.factuals)
        available = np.zeros(len(request.factuals), dtype=bool)
        diagnostics: list[dict[str, Any]] = []
        for index, (factual, target) in enumerate(
            zip(request.factuals, request.targets, strict=True)
        ):
            candidate, info = wachter_coordinate_counterfactual(
                self.context.oracle,
                factual,
                target.item() if isinstance(target, np.generic) else target,
                self.scalar_values,
                self.context.feature_schema.actionable_groups,
                action_units,
                tau=self.config.tau,
                loss_weights=self.config.loss_weights,
                max_steps_per_weight=self.config.max_steps_per_weight,
            )
            raw[index] = candidate
            available[index] = bool(info["valid"])
            diagnostics.append(info)
        return canonical_single_result(
            raw,
            available,
            point_diagnostics=tuple(diagnostics),
            run_diagnostics={"seed": request.seed},
        )


@dataclass(frozen=True)
class GrowingSpheresConfig:
    tau: float = TAU
    n_candidates: int = 512
    initial_radius: float = 0.05
    radius_multiplier: float = 1.5
    max_shells: int = 11

    def __post_init__(self) -> None:
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be between zero and one")
        if self.n_candidates <= 0 or self.max_shells <= 0:
            raise ValueError("candidate and shell counts must be positive")
        if self.initial_radius <= 0 or self.radius_multiplier <= 1:
            raise ValueError("radii must be positive and expanding")


@dataclass(frozen=True)
class GrowingSpheresMethod:
    config: GrowingSpheresConfig = GrowingSpheresConfig()
    method_id = "growing_spheres"
    capabilities = _CAPABILITIES

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def prepare(self, context: MethodContext) -> PreparedGrowingSpheresMethod:
        return PreparedGrowingSpheresMethod(context, self.config)


@dataclass(frozen=True)
class PreparedGrowingSpheresMethod:
    context: MethodContext
    config: GrowingSpheresConfig

    def generate(self, request: GenerationRequest):
        require_single_counterfactual(request)
        if request.factuals.shape[1] != self.context.X_reference.shape[1]:
            raise ValueError("request feature width does not match method context")
        action_units = build_action_units(
            list(self.context.feature_schema.actionable_scalars),
            list(self.context.feature_schema.actionable_groups),
        )
        raw = np.empty_like(request.factuals)
        available = np.zeros(len(request.factuals), dtype=bool)
        diagnostics: list[dict[str, Any]] = []
        for index, (factual, target) in enumerate(
            zip(request.factuals, request.targets, strict=True)
        ):
            point_seed = request.seed + index
            candidate, info = growing_spheres_counterfactual(
                self.context.oracle,
                factual,
                target.item() if isinstance(target, np.generic) else target,
                self.context.feature_schema.actionable_scalars,
                self.context.feature_schema.actionable_groups,
                action_units,
                tau=self.config.tau,
                n_candidates=self.config.n_candidates,
                initial_radius=self.config.initial_radius,
                radius_multiplier=self.config.radius_multiplier,
                max_shells=self.config.max_shells,
                random_state=point_seed,
            )
            raw[index] = candidate
            available[index] = bool(info["valid"])
            diagnostics.append({**info, "seed": point_seed})
        return canonical_single_result(
            raw,
            available,
            point_diagnostics=tuple(diagnostics),
            run_diagnostics={"seed": request.seed},
        )
