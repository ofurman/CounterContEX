"""Strict and lazy behavior of the explicit method registry."""

from __future__ import annotations

import subprocess
import sys

import pytest
from experiments.zeroshot_cf.methods.registry import (
    DEFAULT_METHOD_REGISTRY,
    MethodRegistry,
    RegistryEntry,
)


def test_registry_lists_all_methods_and_rejects_duplicates_and_unknown_names() -> None:
    assert DEFAULT_METHOD_REGISTRY.names() == (
        "dice",
        "dicoflex",
        "face",
        "growing_spheres",
        "nice",
        "wachter",
    )
    entry = RegistryEntry("fake", "fake", "Fake", "Config", "v1", lambda p: p)
    registry = MethodRegistry((entry,))
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(entry)
    with pytest.raises(KeyError, match="unknown"):
        registry.create("missing")


def test_registry_rejects_unknown_or_invalid_method_parameters() -> None:
    with pytest.raises(ValueError, match="unknown parameters"):
        DEFAULT_METHOD_REGISTRY.create("nice", {"not_a_setting": 1})
    with pytest.raises(ValueError, match="positive"):
        DEFAULT_METHOD_REGISTRY.create("nice", {"lof_n_neighbors": 0})
    with pytest.raises(ValueError, match="unknown parameters"):
        DEFAULT_METHOD_REGISTRY.create("dicoflex", {"mystery": {}})
    with pytest.raises(ValueError, match="unsupported variant"):
        DEFAULT_METHOD_REGISTRY.create("nice", variant="tuned")
    with pytest.raises(ValueError, match="unsupported variant"):
        DEFAULT_METHOD_REGISTRY.create("dicoflex", variant="data_plausible")
    assert DEFAULT_METHOD_REGISTRY.entry("dicoflex").supported_variants == (
        "default",
        "tabicl_sparse",
    )
    sparse = DEFAULT_METHOD_REGISTRY.create("dicoflex", variant="tabicl_sparse")
    assert sparse.config.search.cf_mode == "sparse"
    with pytest.raises(ValueError, match="requires cf_mode='sparse'"):
        DEFAULT_METHOD_REGISTRY.create(
            "dicoflex",
            {"search": {"cf_mode": "data_plausible"}},
            variant="tabicl_sparse",
        )


def test_registry_listing_and_cli_help_do_not_import_optional_runtimes() -> None:
    code = """
import sys
from experiments.zeroshot_cf.methods.registry import DEFAULT_METHOD_REGISTRY
from experiments.zeroshot_cf.cli import build_parser, main
assert DEFAULT_METHOD_REGISTRY.names()
build_parser().format_help()
main(['list-methods'])
for forbidden in ('dice_ml', 'raiutils', 'tabicl', 'torch',
                  'experiments.zeroshot_cf.methods.dice',
                  'experiments.zeroshot_cf.methods.face',
                  'experiments.zeroshot_cf.methods.dicoflex',
                  'experiments.zeroshot_cf.tabicl_checkpoints',
                  'experiments.zeroshot_cf.tabicl_sampler'):
    assert forbidden not in sys.modules, forbidden
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
