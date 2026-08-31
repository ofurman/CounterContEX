"""NICE nearest-unlike-neighbour method adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from experiments.zeroshot_cf.baseline_common import ActionUnit, build_action_units
from experiments.zeroshot_cf.core.contracts import GenerationRequest, MethodContext
from experiments.zeroshot_cf.core.validation import target_probabilities
from experiments.zeroshot_cf.methods.base import (
    MethodCapabilities,
    canonical_single_result,
    require_single_counterfactual,
)
from experiments.zeroshot_cf.retained_config import TAU

if TYPE_CHECKING:
    from sklearn.neighbors import LocalOutlierFactor


def nearest_unlike_prototypes(
    X_train: np.ndarray,
    train_predictions: np.ndarray,
    X_test: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the nearest classifier-target training row for each factual."""
    from sklearn.neighbors import NearestNeighbors

    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    train_predictions = np.asarray(train_predictions, dtype=int)
    targets = np.asarray(targets, dtype=int)
    prototypes = np.empty_like(X_test)
    prototype_indices = np.empty(len(X_test), dtype=int)
    distances = np.empty(len(X_test), dtype=np.float64)

    for target in np.unique(targets):
        factual_rows = np.flatnonzero(targets == target)
        pool_indices = np.flatnonzero(train_predictions == target)
        if len(pool_indices) == 0:
            raise ValueError(f"No training rows predicted as target class {target}")
        neighbours = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        neighbours.fit(X_train[pool_indices])
        target_distances, local_indices = neighbours.kneighbors(X_test[factual_rows])
        selected = pool_indices[local_indices[:, 0]]
        prototypes[factual_rows] = X_train[selected]
        prototype_indices[factual_rows] = selected
        distances[factual_rows] = target_distances[:, 0]

    return prototypes, prototype_indices, distances


def greedy_nice_counterfactual(
    disc_model: Any,
    factual: np.ndarray,
    prototype: np.ndarray,
    target: int,
    action_units: Sequence[ActionUnit],
    *,
    plausibility_model: LocalOutlierFactor | None = None,
    tau: float = TAU,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Copy prototype actions greedily until the target prediction is reached."""
    factual = np.asarray(factual, dtype=np.float64)
    prototype = np.asarray(prototype, dtype=np.float64)
    current = factual.copy()
    remaining = [
        unit
        for unit in action_units
        if not np.array_equal(
            current[list(unit.columns)], prototype[list(unit.columns)]
        )
    ]
    selected_units: list[str] = []
    changed_columns: set[int] = set()
    target_probability = float(
        target_probabilities(disc_model, current.reshape(1, -1), np.array([target]))[0]
    )

    while remaining:
        trials: list[np.ndarray] = []
        for unit in remaining:
            trial = current.copy()
            trial[list(unit.columns)] = prototype[list(unit.columns)]
            trials.append(trial)
        trial_matrix = np.stack(trials)
        probabilities = target_probabilities(
            disc_model,
            trial_matrix,
            np.full(len(trial_matrix), target),
        )
        predictions = np.asarray(disc_model.predict(trial_matrix), dtype=int)
        valid = (predictions == target) & (probabilities >= tau)

        if valid.any() and plausibility_model is not None:
            eligible = np.flatnonzero(valid)
            lof_scores = -np.asarray(
                plausibility_model.score_samples(trial_matrix[eligible]),
                dtype=np.float64,
            )
            best = int(eligible[np.argmin(lof_scores)])
        elif valid.any():
            eligible = np.flatnonzero(valid)
            best = int(eligible[np.argmax(probabilities[eligible])])
        else:
            best = int(np.argmax(probabilities))

        unit = remaining.pop(best)
        current = trial_matrix[best]
        target_probability = float(probabilities[best])
        selected_units.append(unit.name)
        for column in unit.columns:
            if factual[column] != current[column]:
                changed_columns.add(column)
            else:
                changed_columns.discard(column)

        if bool(valid[best]):
            break

    prediction = int(disc_model.predict(current.reshape(1, -1))[0])
    return current, {
        "valid": prediction == target and target_probability >= tau,
        "prediction": prediction,
        "target_probability": target_probability,
        "steps": len(selected_units),
        "changed_columns": len(changed_columns),
        "selected_units": selected_units,
    }


@dataclass(frozen=True)
class NiceConfig:
    tau: float = TAU
    lof_n_neighbors: int = 20

    def __post_init__(self) -> None:
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be between zero and one")
        if self.lof_n_neighbors <= 0:
            raise ValueError("lof_n_neighbors must be positive")


@dataclass(frozen=True)
class NiceMethod:
    config: NiceConfig = NiceConfig()
    method_id = "nice_nun_greedy_lof"
    capabilities = MethodCapabilities(
        supports_categorical=True,
        enforces_actionability=True,
        supports_multiple_counterfactuals=False,
        requires_probabilities=True,
        optional_dependencies=("scikit-learn",),
    )

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def prepare(self, context: MethodContext) -> PreparedNiceMethod:
        from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

        predictions = np.asarray(context.oracle.predict(context.X_reference)).reshape(
            -1
        )
        pools: dict[Any, tuple[np.ndarray, Any]] = {}
        for target in np.unique(predictions):
            indices = np.flatnonzero(predictions == target)
            neighbours = NearestNeighbors(
                n_neighbors=1,
                metric="euclidean",
                n_jobs=-1,
            ).fit(context.X_reference[indices])
            pools[target.item()] = (indices, neighbours)
        lof = LocalOutlierFactor(
            n_neighbors=min(self.config.lof_n_neighbors, len(context.X_reference) - 1),
            novelty=True,
        ).fit(context.X_reference)
        return PreparedNiceMethod(context, self.config, pools, lof)


@dataclass(frozen=True)
class PreparedNiceMethod:
    context: MethodContext
    config: NiceConfig
    target_pools: dict[Any, tuple[np.ndarray, Any]]
    plausibility_model: Any

    def generate(self, request: GenerationRequest):
        require_single_counterfactual(request)
        if request.factuals.shape[1] != self.context.X_reference.shape[1]:
            raise ValueError("request feature width does not match method context")
        action_units = build_action_units(
            list(self.context.feature_schema.actionable_scalars),
            list(self.context.feature_schema.actionable_groups),
        )
        raw = np.empty_like(request.factuals)
        prototypes = np.empty_like(request.factuals)
        prototype_indices = np.empty(len(request.factuals), dtype=np.int64)
        prototype_distances = np.empty(len(request.factuals), dtype=np.float64)
        diagnostics: list[dict[str, Any]] = []
        available = np.zeros(len(request.factuals), dtype=bool)
        for index, (factual, target) in enumerate(
            zip(request.factuals, request.targets, strict=True)
        ):
            target_key = target.item() if isinstance(target, np.generic) else target
            if target_key not in self.target_pools:
                raise ValueError(
                    f"No reference rows predicted as target class {target_key}"
                )
            pool_indices, neighbours = self.target_pools[target_key]
            distances, local_indices = neighbours.kneighbors(factual.reshape(1, -1))
            prototype_index = int(pool_indices[int(local_indices[0, 0])])
            prototype = self.context.X_reference[prototype_index]
            candidate, info = greedy_nice_counterfactual(
                self.context.oracle,
                factual,
                prototype,
                target_key,
                action_units,
                plausibility_model=self.plausibility_model,
                tau=self.config.tau,
            )
            raw[index] = candidate
            prototypes[index] = prototype
            prototype_indices[index] = prototype_index
            prototype_distances[index] = float(distances[0, 0])
            available[index] = bool(info["valid"])
            diagnostics.append(
                {
                    **info,
                    "prototype_index": prototype_index,
                    "prototype_distance": float(distances[0, 0]),
                }
            )
        return canonical_single_result(
            raw,
            available,
            point_diagnostics=tuple(diagnostics),
            run_diagnostics={"seed": request.seed},
            extra_artifacts={
                "method.prototypes": prototypes,
                "method.prototype_indices": prototype_indices,
                "method.prototype_distances": prototype_distances,
            },
        )
