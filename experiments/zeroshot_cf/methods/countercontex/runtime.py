"""Method-owned execution and identity policies for CounterContEx backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.zeroshot_cf.methods.registry import ResolvedMethodRuntime


@dataclass(frozen=True)
class BackendRuntimePolicy:
    implementation_version: str
    resolve: Callable[
        [dict[str, Any], Mapping[str, Path], str | None], ResolvedMethodRuntime
    ]


def _empirical_runtime(
    params: dict[str, Any],
    cache_paths: Mapping[str, Path],
    device: str | None,
) -> ResolvedMethodRuntime:
    del cache_paths, device
    from experiments.zeroshot_cf.methods.countercontex.backends.empirical import (
        EMPIRICAL_BACKEND_IMPLEMENTATION_VERSION,
    )

    return ResolvedMethodRuntime(
        params,
        backend_implementation=EMPIRICAL_BACKEND_IMPLEMENTATION_VERSION,
    )


def _tabicl_runtime(
    params: dict[str, Any],
    cache_paths: Mapping[str, Path],
    device: str | None,
) -> ResolvedMethodRuntime:
    from experiments.zeroshot_cf import tabicl_checkpoints
    from experiments.zeroshot_cf.methods.countercontex.backends.tabicl import (
        TABICL_BACKEND_HISTORICAL_IMPLEMENTATION_VERSION,
        TABICL_BACKEND_IMPLEMENTATION_VERSION,
    )

    cache_dir = cache_paths.get("tabicl")
    paths = tabicl_checkpoints.require_checkpoints(cache_dir)
    foundation = dict(params.get("foundation", {}))
    if cache_dir is not None:
        foundation["cache_dir"] = cache_dir
        params["foundation"] = foundation

    @contextmanager
    def activate():
        if device is None:
            with nullcontext():
                yield
            return
        previous = tabicl_checkpoints.TABICL_DEVICE
        try:
            tabicl_checkpoints.TABICL_DEVICE = device
            yield
        finally:
            tabicl_checkpoints.TABICL_DEVICE = previous

    strict_distribution = bool(
        dict(params.get("search", {})).get("strict_proposal_budget", False)
    )
    return ResolvedMethodRuntime(
        params,
        backend_implementation=(
            TABICL_BACKEND_IMPLEMENTATION_VERSION
            if strict_distribution
            else TABICL_BACKEND_HISTORICAL_IMPLEMENTATION_VERSION
        ),
        checkpoint_content_ids={
            path.name: tabicl_checkpoints._CHECKPOINT_SHA256[path.name]
            for path in paths
        },
        activate=activate,
    )


def _tabpfn_runtime(
    params: dict[str, Any],
    cache_paths: Mapping[str, Path],
    device: str | None,
) -> ResolvedMethodRuntime:
    from experiments.zeroshot_cf import tabpfn_checkpoints
    from experiments.zeroshot_cf.methods.countercontex.backends.tabpfn import (
        TABPFN_BACKEND_IMPLEMENTATION_VERSION,
    )

    cache_dir = cache_paths.get("tabpfn")
    paths = tabpfn_checkpoints.require_checkpoints(cache_dir)
    foundation = dict(params.get("foundation", {}))
    if cache_dir is not None:
        foundation["cache_dir"] = cache_dir
        params["foundation"] = foundation

    @contextmanager
    def activate():
        if device is None:
            with nullcontext():
                yield
            return
        previous = tabpfn_checkpoints.TABPFN_DEVICE
        try:
            tabpfn_checkpoints.TABPFN_DEVICE = device
            yield
        finally:
            tabpfn_checkpoints.TABPFN_DEVICE = previous

    return ResolvedMethodRuntime(
        params,
        backend_implementation=TABPFN_BACKEND_IMPLEMENTATION_VERSION,
        checkpoint_content_ids={
            path.name: tabpfn_checkpoints._CHECKPOINT_SHA256[path.name]
            for path in paths
        },
        activate=activate,
    )


_BACKEND_POLICIES = {
    "tabicl": BackendRuntimePolicy("tabicl-proposal-v2-distributions", _tabicl_runtime),
    "tabpfn": BackendRuntimePolicy("tabpfn-v2-proposal-v1", _tabpfn_runtime),
    "empirical": BackendRuntimePolicy("empirical-reference-v1", _empirical_runtime),
}


def resolve_runtime(
    values: Mapping[str, Any],
    cache_paths: Mapping[str, Path],
    device: str | None,
) -> ResolvedMethodRuntime:
    params = deepcopy(dict(values))
    foundation = params.get("foundation", {})
    if not isinstance(foundation, Mapping):
        raise ValueError("countercontex foundation params must be a mapping")
    backend = foundation.get("backend", "tabicl")
    try:
        policy = _BACKEND_POLICIES[str(backend)]
    except KeyError as error:
        raise ValueError(
            f"unknown CounterContEx proposal backend: {backend!r}"
        ) from error
    resolved = policy.resolve(params, cache_paths, device)
    compatible_implementation = (
        resolved.backend_implementation
        in {"tabicl-proposal-v1", policy.implementation_version}
        if str(backend) == "tabicl"
        else resolved.backend_implementation == policy.implementation_version
    )
    if not compatible_implementation:
        raise RuntimeError("CounterContEx backend runtime identity is inconsistent")
    return resolved
