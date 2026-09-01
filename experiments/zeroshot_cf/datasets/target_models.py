"""Fixed target-classifier families owned by benchmark-case construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal


@dataclass(frozen=True)
class TargetModelEntry:
    name: str
    discriminator_type: Literal["lr", "mlp", "xgb"]
    model_kind: str
    declared_params: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "declared_params", MappingProxyType(dict(self.declared_params))
        )


class TargetModelRegistry:
    def __init__(self, entries: tuple[TargetModelEntry, ...] = ()) -> None:
        self._entries: dict[str, TargetModelEntry] = {}
        for entry in entries:
            if entry.name in self._entries:
                raise ValueError(f"duplicate target model registry name: {entry.name}")
            self._entries[entry.name] = entry

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def resolve(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
    ) -> TargetModelEntry:
        try:
            entry = self._entries[name]
        except KeyError as error:
            raise KeyError(f"unknown target model: {name}") from error
        supplied = dict(entry.declared_params if params is None else params)
        expected = dict(entry.declared_params)
        if supplied != expected:
            raise ValueError(
                f"target model {name} requires fixed params {expected}, got {supplied}"
            )
        return entry


DEFAULT_TARGET_MODEL_REGISTRY = TargetModelRegistry(
    (
        TargetModelEntry(
            "retained_logistic_regression",
            "lr",
            "logistic_regression",
            {"C": 1.0, "max_iter": 1000, "seed": 42},
        ),
        TargetModelEntry(
            "retained_mlp",
            "mlp",
            "mlp",
            {"hidden_layer_sizes": [64, 32], "max_iter": 300, "seed": 42},
        ),
        TargetModelEntry(
            "retained_xgboost",
            "xgb",
            "xgboost",
            {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "seed": 42,
            },
        ),
    )
)
