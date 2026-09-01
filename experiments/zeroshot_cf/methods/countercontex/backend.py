"""Compatibility imports for the retained TabICL backend path."""

from experiments.zeroshot_cf.methods.countercontex.backends.tabicl import (
    CounterContExBackendInputs,
    CounterContExBackendRuntime,
    PreparedTabICLBackend,
    TabICLBackend,
    load_backend_runtime,
    prepare_backend,
)

PreparedCounterContExBackend = PreparedTabICLBackend

__all__ = [
    "CounterContExBackendInputs",
    "CounterContExBackendRuntime",
    "PreparedCounterContExBackend",
    "PreparedTabICLBackend",
    "TabICLBackend",
    "load_backend_runtime",
    "prepare_backend",
]
