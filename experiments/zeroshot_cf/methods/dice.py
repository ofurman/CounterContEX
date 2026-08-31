"""Lazy DiCE genetic method adapter with mixed-data codec ownership."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.baseline_common import (
    build_action_units,
    contract_scalar_actions,
    prune_counterfactual_actions,
)
from experiments.zeroshot_cf.core.contracts import GenerationRequest, MethodContext
from experiments.zeroshot_cf.methods.base import (
    MethodCapabilities,
    canonical_single_result,
    require_single_counterfactual,
)
from experiments.zeroshot_cf.retained_config import TAU

if TYPE_CHECKING:
    import pandas as pd
    from experiments.zeroshot_cf.data import DatasetBundle


OUTCOME = "target_model_prediction"


@dataclass(frozen=True)
class DiceMixedAdapter:
    """Round-trip between repository one-hot matrices and compact DiCE frames."""

    n_features: int
    scalar_columns: tuple[int, ...]
    groups: tuple[OneHotActionGroup, ...]
    scalar_names: tuple[str, ...]

    @classmethod
    def from_bundle(cls, bundle: DatasetBundle) -> DiceMixedAdapter:
        from experiments.zeroshot_cf.data import get_one_hot_groups

        groups = tuple(get_one_hot_groups(bundle))
        grouped = {column for group in groups for column in group.columns}
        scalar_columns = tuple(
            column
            for column in range(len(bundle.feature_names))
            if column not in grouped
        )
        scalar_names = tuple(bundle.feature_names[column] for column in scalar_columns)
        return cls(
            n_features=len(bundle.feature_names),
            scalar_columns=scalar_columns,
            groups=groups,
            scalar_names=scalar_names,
        )

    @property
    def feature_names(self) -> list[str]:
        return [*self.scalar_names, *(group.name for group in self.groups)]

    def encode(self, X: np.ndarray) -> pd.DataFrame:
        import pandas as pd

        matrix = np.atleast_2d(np.asarray(X, dtype=np.float64))
        data: dict[str, Any] = {
            name: matrix[:, column]
            for name, column in zip(self.scalar_names, self.scalar_columns, strict=True)
        }
        for group in self.groups:
            columns = list(group.columns)
            if not np.allclose(matrix[:, columns].sum(axis=1), 1.0):
                raise ValueError(f"one-hot group {group.name!r} is invalid")
            data[group.name] = [
                str(category) for category in np.argmax(matrix[:, columns], axis=1)
            ]
        return pd.DataFrame(data, columns=self.feature_names)

    def decode(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        import pandas as pd

        compact = (
            frame.loc[:, self.feature_names]
            if isinstance(frame, pd.DataFrame)
            else pd.DataFrame(frame, columns=self.feature_names)
        )
        matrix = np.zeros((len(compact), self.n_features), dtype=np.float64)
        for name, column in zip(self.scalar_names, self.scalar_columns, strict=True):
            matrix[:, column] = np.asarray(
                pd.to_numeric(compact[name]),
                dtype=np.float64,
            )
        for group in self.groups:
            categories = np.asarray(
                pd.to_numeric(compact[group.name]),
                dtype=np.int64,
            )
            if np.any((categories < 0) | (categories >= len(group.columns))):
                raise ValueError(f"category outside group {group.name!r}")
            matrix[np.arange(len(matrix)), np.asarray(group.columns)[categories]] = 1.0
        return matrix


class DiceClassifierAdapter:
    """Sklearn-like classifier accepting compact DiCE frames."""

    def __init__(self, classifier: Any, codec: DiceMixedAdapter) -> None:
        super().__init__()
        self.classifier = classifier
        self.codec = codec
        self.classes_ = np.asarray(getattr(classifier, "classes_", (0, 1)))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.predict_proba(self.codec.decode(X)))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.predict(self.codec.decode(X)))


def generate_dice_counterfactuals(
    explainer: Any,
    codec: DiceMixedAdapter,
    classifier: Any,
    X_test: np.ndarray,
    y_target: np.ndarray,
    features_to_vary: list[str],
    *,
    max_iterations: int = 200,
    search_restarts: int = 1,
    stopping_threshold: float = TAU,
    random_state: int = 42,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate one CF per factual from DiCE's valid pre-sparsification set."""
    from raiutils.exceptions import UserConfigValidationException

    if search_restarts < 1:
        raise ValueError("search_restarts must be positive")
    if not 0.5 <= stopping_threshold < 1.0:
        raise ValueError("stopping_threshold must be in [0.5, 1.0)")
    X_cf = np.asarray(X_test, dtype=np.float64).copy()
    point_info: list[dict[str, Any]] = []
    queries = codec.encode(X_test)
    for index, target in enumerate(np.asarray(y_target, dtype=int)):
        started = time.perf_counter()
        returned = False
        valid_candidates = 0
        attempts_used = 0
        for attempt in range(search_restarts):
            attempts_used = attempt + 1
            attempt_seed = random_state + index + attempt * 100_003
            random.seed(attempt_seed)
            np.random.seed(attempt_seed)
            try:
                explainer.generate_counterfactuals(
                    queries.iloc[[index]],
                    total_CFs=1,
                    desired_class=int(target),
                    features_to_vary=features_to_vary,
                    stopping_threshold=stopping_threshold,
                    posthoc_sparsity_param=0.0,
                    posthoc_sparsity_algorithm="binary",
                    initialization="kdtree",
                    proximity_weight=0.2,
                    sparsity_weight=0.2,
                    categorical_penalty=0.1,
                    maxiterations=max_iterations,
                    verbose=False,
                )
                # DiCE rounds continuous values before exposing final_cfs_df
                # and may thereby move a marginal CF back across the boundary.
                # Recover the genetic solver's unrounded candidates instead.
                final_frame = explainer.label_decode_cfs(explainer.final_cfs)
                attempt_returned = final_frame is not None and len(final_frame) > 0
            except UserConfigValidationException as error:
                if "No counterfactuals found" not in str(error):
                    raise
                final_frame = None
                attempt_returned = False
            returned = returned or attempt_returned
            if not attempt_returned or final_frame is None:
                continue

            candidates = codec.decode(final_frame)
            predictions = np.asarray(
                classifier.predict(candidates), dtype=int
            )
            valid_indices = np.flatnonzero(predictions == int(target))
            valid_candidates = len(valid_indices)
            if valid_candidates:
                candidates = candidates[valid_indices]
                compact_factual = queries.iloc[index].to_numpy()
                compact_candidates = final_frame.iloc[valid_indices][
                    codec.feature_names
                ].to_numpy()
                changed_actions = (compact_candidates != compact_factual).sum(axis=1)
                l2 = np.linalg.norm(candidates - X_test[index], axis=1)
                selected = np.lexsort((l2, changed_actions))[0]
                X_cf[index] = candidates[selected]
                break
        point_info.append(
            {
                "returned": bool(returned),
                "found": bool(valid_candidates),
                "valid_candidates": int(valid_candidates),
                "attempts": attempts_used,
                "runtime_s": time.perf_counter() - started,
            }
        )
    return X_cf, point_info


@dataclass(frozen=True)
class DiceConfig:
    max_iterations: int = 200
    search_restarts: int = 1
    stopping_threshold: float = TAU

    def __post_init__(self) -> None:
        if self.max_iterations <= 0 or self.search_restarts <= 0:
            raise ValueError("iteration and restart counts must be positive")
        if not 0.5 <= self.stopping_threshold < 1:
            raise ValueError("stopping_threshold must be in [0.5, 1.0)")


@dataclass(frozen=True)
class DiceMethod:
    config: DiceConfig = DiceConfig()
    method_id = "dice_genetic_atomic_pruned"
    capabilities = MethodCapabilities(
        supports_categorical=True,
        enforces_actionability=True,
        supports_multiple_counterfactuals=False,
        requires_probabilities=True,
        optional_dependencies=("dice-ml", "raiutils", "pandas"),
    )

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def prepare(self, context: MethodContext) -> PreparedDiceMethod:
        import dice_ml

        schema = context.feature_schema
        groups = tuple(schema.categorical_groups)
        grouped_columns = {column for group in groups for column in group.columns}
        scalar_columns = tuple(
            column
            for column in range(len(schema.names))
            if column not in grouped_columns
        )
        codec = DiceMixedAdapter(
            n_features=len(schema.names),
            scalar_columns=scalar_columns,
            groups=groups,
            scalar_names=tuple(schema.names[column] for column in scalar_columns),
        )
        train_frame = codec.encode(context.X_reference)
        train_frame[OUTCOME] = np.asarray(context.oracle.predict(context.X_reference))
        data_interface = dice_ml.Data(
            dataframe=train_frame,
            continuous_features=list(codec.scalar_names),
            outcome_name=OUTCOME,
        )
        model_interface = dice_ml.Model(
            model=DiceClassifierAdapter(context.oracle, codec),
            backend="sklearn",
            model_type="classifier",
        )
        explainer = dice_ml.Dice(data_interface, model_interface, method="genetic")
        actionable_scalars = set(schema.actionable_scalars)
        actionable_groups = {group.name for group in schema.actionable_groups}
        features_to_vary = [
            *(
                name
                for name, column in zip(
                    codec.scalar_names, codec.scalar_columns, strict=True
                )
                if column in actionable_scalars
            ),
            *(group.name for group in codec.groups if group.name in actionable_groups),
        ]
        return PreparedDiceMethod(
            context,
            self.config,
            codec,
            explainer,
            tuple(features_to_vary),
        )


@dataclass(frozen=True)
class PreparedDiceMethod:
    context: MethodContext
    config: DiceConfig
    codec: DiceMixedAdapter
    explainer: Any
    features_to_vary: tuple[str, ...]

    def generate(self, request: GenerationRequest):
        require_single_counterfactual(request)
        if request.factuals.shape[1] != self.context.X_reference.shape[1]:
            raise ValueError("request feature width does not match method context")
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            raw, diagnostics = generate_dice_counterfactuals(
                self.explainer,
                self.codec,
                self.context.oracle,
                request.factuals,
                request.targets,
                list(self.features_to_vary),
                max_iterations=self.config.max_iterations,
                search_restarts=self.config.search_restarts,
                stopping_threshold=self.config.stopping_threshold,
                random_state=request.seed,
            )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        raw_unpruned = raw.copy()
        action_units = build_action_units(
            list(self.context.feature_schema.actionable_scalars),
            list(self.context.feature_schema.actionable_groups),
        )
        available = np.zeros(len(raw), dtype=bool)
        for index, target in enumerate(request.targets):
            target_value = target.item() if isinstance(target, np.generic) else target
            if not diagnostics[index]["found"]:
                continue
            raw[index] = prune_counterfactual_actions(
                self.context.oracle,
                request.factuals[index],
                raw[index],
                target_value,
                action_units,
                tau=self.config.stopping_threshold,
            )
            raw[index] = contract_scalar_actions(
                self.context.oracle,
                request.factuals[index],
                raw[index],
                target_value,
                self.context.feature_schema.actionable_scalars,
                tau=self.config.stopping_threshold,
            )
            available[index] = bool(
                np.asarray(self.context.oracle.predict(raw[index : index + 1]))[0]
                == target_value
            )
        return canonical_single_result(
            raw,
            available,
            point_diagnostics=tuple(diagnostics),
            run_diagnostics={
                "seed": request.seed,
                "features_to_vary": list(self.features_to_vary),
            },
            extra_artifacts={"method.raw_candidates": raw_unpruned},
        )
