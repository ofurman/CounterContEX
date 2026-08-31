"""Compatibility imports for the retained TabICL backend path."""

from experiments.zeroshot_cf.methods.dicoflex.backends.tabicl import (
    DiCoFlexBackendInputs,
    DiCoFlexBackendRuntime,
    PreparedTabICLBackend,
    TabICLBackend,
    load_backend_runtime,
    prepare_backend,
)

PreparedDiCoFlexBackend = PreparedTabICLBackend

__all__ = [
    "DiCoFlexBackendInputs",
    "DiCoFlexBackendRuntime",
    "PreparedDiCoFlexBackend",
    "PreparedTabICLBackend",
    "TabICLBackend",
    "load_backend_runtime",
    "prepare_backend",
]
