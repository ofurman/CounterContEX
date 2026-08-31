"""Explicit lazy registry for benchmark-facing counterfactual methods."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegistryEntry:
    name: str
    module: str
    method_class: str
    config_class: str
    implementation_version: str
    factory: Callable[[Mapping[str, Any]], Any] | None = None
    supported_variants: tuple[str, ...] = ("default",)
    variant_resolver: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = (
        None
    )


class MethodRegistry:
    def __init__(self, entries: tuple[RegistryEntry, ...] = ()) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: RegistryEntry) -> None:
        if entry.name in self._entries:
            raise ValueError(f"duplicate method registry name: {entry.name}")
        if (
            not entry.supported_variants
            or any(not variant for variant in entry.supported_variants)
            or len(set(entry.supported_variants)) != len(entry.supported_variants)
        ):
            raise ValueError("method registry variants must be unique and non-empty")
        self._entries[entry.name] = entry

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def entry(self, name: str) -> RegistryEntry:
        try:
            return self._entries[name]
        except KeyError as error:
            raise KeyError(f"unknown counterfactual method: {name}") from error

    def create(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        variant: str = "default",
    ):
        entry = self.entry(name)
        if variant not in entry.supported_variants:
            raise ValueError(
                f"unsupported variant for {name}: {variant!r}; "
                f"expected one of {entry.supported_variants}"
            )
        values = dict(params or {})
        if entry.variant_resolver is not None:
            values = dict(entry.variant_resolver(variant, values))
        if entry.factory is not None:
            return entry.factory(values)
        module = importlib.import_module(entry.module)
        config_type = getattr(module, entry.config_class)
        method_type = getattr(module, entry.method_class)
        allowed = set(inspect.signature(config_type).parameters)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown parameters for {name}: {sorted(unknown)}")
        try:
            config = config_type(**values)
        except TypeError as error:
            raise ValueError(f"invalid parameters for {name}: {error}") from error
        return method_type(config)


def _dicoflex_factory(values: Mapping[str, Any]):
    module = importlib.import_module("experiments.zeroshot_cf.methods.dicoflex")
    unknown = set(values) - {"search", "diversity", "foundation"}
    if unknown:
        raise ValueError(f"unknown parameters for dicoflex: {sorted(unknown)}")
    try:
        config = module.DiCoFlexConfig(
            search=module.DiCoFlexSearchConfig(**dict(values.get("search", {}))),
            diversity=module.DiCoFlexDiversityConfig(
                **dict(values.get("diversity", {}))
            ),
            foundation=module.DiCoFlexFoundationConfig(
                **dict(values.get("foundation", {}))
            ),
        )
    except TypeError as error:
        raise ValueError(f"invalid parameters for dicoflex: {error}") from error
    return module.DiCoFlexMethod(config)


def _dicoflex_variant(
    variant: str,
    values: Mapping[str, Any],
) -> Mapping[str, Any]:
    if variant != "tabicl_sparse":
        return values
    resolved = dict(values)
    search = dict(resolved.get("search", {}))
    if search.get("cf_mode", "sparse") != "sparse":
        raise ValueError("dicoflex tabicl_sparse variant requires cf_mode='sparse'")
    search["cf_mode"] = "sparse"
    resolved["search"] = search
    return resolved


DEFAULT_METHOD_REGISTRY = MethodRegistry(
    (
        RegistryEntry(
            "dicoflex",
            "experiments.zeroshot_cf.methods.dicoflex",
            "DiCoFlexMethod",
            "DiCoFlexConfig",
            "dicoflex-v1",
            _dicoflex_factory,
            ("default", "tabicl_sparse"),
            _dicoflex_variant,
        ),
        RegistryEntry(
            "nice",
            "experiments.zeroshot_cf.methods.nice",
            "NiceMethod",
            "NiceConfig",
            "nice-v1",
        ),
        RegistryEntry(
            "wachter",
            "experiments.zeroshot_cf.methods.optimization",
            "WachterMethod",
            "WachterConfig",
            "wachter-v1",
        ),
        RegistryEntry(
            "growing_spheres",
            "experiments.zeroshot_cf.methods.optimization",
            "GrowingSpheresMethod",
            "GrowingSpheresConfig",
            "growing-spheres-v1",
        ),
        RegistryEntry(
            "dice",
            "experiments.zeroshot_cf.methods.dice",
            "DiceMethod",
            "DiceConfig",
            "dice-v1",
        ),
        RegistryEntry(
            "face",
            "experiments.zeroshot_cf.methods.face",
            "FaceMethod",
            "FaceConfig",
            "face-v1",
        ),
    )
)
