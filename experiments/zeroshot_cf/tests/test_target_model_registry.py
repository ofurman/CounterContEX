"""Dataset-owned target-model registry contracts."""

from __future__ import annotations

import pytest
from experiments.zeroshot_cf.datasets.target_models import (
    DEFAULT_TARGET_MODEL_REGISTRY,
)


def test_default_target_model_registry_exposes_three_fixed_families() -> None:
    assert DEFAULT_TARGET_MODEL_REGISTRY.names() == (
        "retained_logistic_regression",
        "retained_mlp",
        "retained_xgboost",
    )

    resolved = tuple(
        DEFAULT_TARGET_MODEL_REGISTRY.resolve(name)
        for name in DEFAULT_TARGET_MODEL_REGISTRY.names()
    )

    assert {model.discriminator_type for model in resolved} == {"lr", "mlp", "xgb"}
    assert {model.model_kind for model in resolved} == {
        "logistic_regression",
        "mlp",
        "xgboost",
    }
    assert all(model.declared_params["seed"] == 42 for model in resolved)


def test_target_model_registry_rejects_unknown_or_nonfixed_parameters() -> None:
    with pytest.raises(KeyError, match="unknown target model"):
        DEFAULT_TARGET_MODEL_REGISTRY.resolve("unknown")
    with pytest.raises(ValueError, match="fixed params"):
        DEFAULT_TARGET_MODEL_REGISTRY.resolve(
            "retained_logistic_regression",
            {"C": 2.0, "max_iter": 1000, "seed": 42},
        )
