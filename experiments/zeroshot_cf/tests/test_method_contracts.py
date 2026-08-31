"""Shared contract tests for reference counterfactual method wrappers."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.contracts import (
    FeatureDomains,
    FeatureSchema,
    GenerationRequest,
    MethodContext,
)
from experiments.zeroshot_cf.methods.base import (
    CounterfactualMethod,
    PreparedMethod,
)
from experiments.zeroshot_cf.methods.dice import DiceMethod
from experiments.zeroshot_cf.methods.face import FaceConfig, FaceMethod
from experiments.zeroshot_cf.methods.nice import NiceMethod
from experiments.zeroshot_cf.methods.optimization import (
    GrowingSpheresConfig,
    GrowingSpheresMethod,
    WachterMethod,
)


class _ReversedClassifier:
    classes_ = np.array([7, 2])

    def predict_proba(self, X):
        matrix = np.asarray(X)
        target = np.clip(0.05 + 0.35 * matrix[:, 0] + 0.65 * matrix[:, 2], 0, 1)
        return np.column_stack([1 - target, target])

    def predict(self, X):
        target = self.predict_proba(X)[:, 1]
        return np.where(target >= 0.5, 2, 7)


class _ImmutableClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probability = np.where(np.asarray(X)[:, 3] >= 0.5, 0.9, 0.1)
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return (np.asarray(X)[:, 3] >= 0.5).astype(int)


def _context() -> MethodContext:
    group = OneHotActionGroup("kind", (1, 2))
    schema = FeatureSchema(
        names=("amount", "kind_a", "kind_b", "fixed"),
        numerical=(0, 3),
        categorical_groups=(group,),
        actionable_scalars=(0,),
        actionable_groups=(group,),
        immutable=(3,),
        domains=FeatureDomains(np.zeros(4), np.ones(4), {}),
    )
    reference = np.array(
        [
            [0.0, 1.0, 0.0, 0.2],
            [0.2, 1.0, 0.0, 0.4],
            [0.4, 1.0, 0.0, 0.6],
            [0.3, 0.0, 1.0, 0.2],
            [0.7, 0.0, 1.0, 0.5],
            [1.0, 0.0, 1.0, 0.8],
        ]
    )
    return MethodContext(reference, schema, _ReversedClassifier())


def _request(seed=17):
    return GenerationRequest(
        factuals=np.array(
            [
                [0.0, 1.0, 0.0, 0.25],
                [0.1, 1.0, 0.0, 0.75],
            ]
        ),
        targets=np.array([2, 2]),
        n_counterfactuals=1,
        seed=seed,
    )


def _failure_context() -> MethodContext:
    context = _context()
    return MethodContext(
        context.X_reference,
        context.feature_schema,
        _ImmutableClassifier(),
    )


@pytest.mark.parametrize(
    "method",
    [
        NiceMethod(),
        WachterMethod(),
        GrowingSpheresMethod(GrowingSpheresConfig(n_candidates=128, max_shells=12)),
        FaceMethod(FaceConfig(n_neighbors=3)),
    ],
    ids=lambda method: method.method_id,
)
def test_reference_methods_share_validated_actionable_result_contract(method):
    assert isinstance(method, CounterfactualMethod)
    assert method.config_dict()
    assert method.capabilities.supports_categorical
    assert method.capabilities.enforces_actionability
    assert not method.capabilities.supports_multiple_counterfactuals
    assert method.capabilities.requires_probabilities

    prepared = method.prepare(_context())
    assert isinstance(prepared, PreparedMethod)
    result = prepared.generate(_request())

    assert result.candidates.shape == (2, 1, 4)
    assert result.available.shape == (2, 1)
    assert result.available.all()
    assert all(name.startswith("method.") for name in result.artifacts)
    candidates = result.candidates[:, 0]
    np.testing.assert_array_equal(candidates[:, 3], _request().factuals[:, 3])
    np.testing.assert_allclose(candidates[:, 1:3].sum(axis=1), 1.0)
    np.testing.assert_array_equal(_context().oracle.predict(candidates), [2, 2])

    with pytest.raises(ValueError, match="exactly one"):
        prepared.generate(
            GenerationRequest(_request().factuals, _request().targets, 2, seed=17)
        )


def test_growing_spheres_uses_request_seed_deterministically():
    prepared = GrowingSpheresMethod(
        GrowingSpheresConfig(n_candidates=32, max_shells=8)
    ).prepare(_context())
    first = prepared.generate(_request(seed=9))
    repeated = prepared.generate(_request(seed=9))
    different = prepared.generate(_request(seed=10))
    np.testing.assert_equal(first.candidates, repeated.candidates)
    assert first.point_diagnostics == repeated.point_diagnostics
    assert [row["seed"] for row in first.point_diagnostics] == [9, 10]
    assert [row["seed"] for row in different.point_diagnostics] == [10, 11]


@pytest.mark.parametrize(
    "method",
    [
        NiceMethod(),
        WachterMethod(),
        GrowingSpheresMethod(GrowingSpheresConfig(n_candidates=16, max_shells=3)),
        FaceMethod(FaceConfig(n_neighbors=3)),
    ],
    ids=lambda method: method.method_id,
)
def test_genuine_failures_are_unavailable_with_namespaced_best_effort(method):
    request = GenerationRequest(
        np.array([[0.0, 1.0, 0.0, 0.1]]),
        np.array([1]),
        1,
        seed=5,
    )
    result = method.prepare(_failure_context()).generate(request)
    assert not result.available.any()
    assert np.isnan(result.candidates).all()
    best_effort = result.artifacts["method.best_effort"]
    assert np.isfinite(best_effort).all()
    assert best_effort[0, 3] == request.factuals[0, 3]
    json.dumps([dict(values) for values in result.point_diagnostics], allow_nan=False)


def test_dice_prepare_is_lazy_and_generate_restores_global_rng(monkeypatch):
    class _Data:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Model:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Dice:
        def __init__(self, data, model, method):
            self.data = data
            self.model = model
            self.method = method

    monkeypatch.setitem(
        sys.modules,
        "dice_ml",
        SimpleNamespace(Data=_Data, Model=_Model, Dice=_Dice),
    )
    from experiments.zeroshot_cf.methods import dice as dice_module

    propagated_seeds = []

    def fake_generate(*args, **kwargs):
        propagated_seeds.append(kwargs["random_state"])
        random.seed(999)
        np.random.seed(999)
        factuals = np.asarray(args[3])
        rows = factuals.copy()
        rows[:, 0] = 1.0
        rows[:, 1:3] = [0.0, 1.0]
        info = [
            {
                "returned": True,
                "found": True,
                "valid_candidates": 1,
                "attempts": 1,
                "runtime_s": 0.0,
            }
            for _ in rows
        ]
        return rows, info

    monkeypatch.setattr(dice_module, "generate_dice_counterfactuals", fake_generate)
    method = DiceMethod()
    assert isinstance(method, CounterfactualMethod)
    assert method.capabilities.supports_categorical
    assert method.capabilities.enforces_actionability
    assert not method.capabilities.supports_multiple_counterfactuals
    assert method.capabilities.requires_probabilities
    prepared = method.prepare(_context())
    assert isinstance(prepared, PreparedMethod)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    result = prepared.generate(_request(seed=23))
    repeated_with_new_seed = prepared.generate(_request(seed=29))

    assert random.getstate() == python_state
    restored = np.random.get_state()
    assert restored[0] == numpy_state[0]
    np.testing.assert_array_equal(restored[1], numpy_state[1])
    assert restored[2:] == numpy_state[2:]
    assert result.available.all()
    assert repeated_with_new_seed.available.all()
    assert propagated_seeds == [23, 29]
    assert "method.raw_candidates" in result.artifacts
    np.testing.assert_array_equal(result.candidates[:, 0, 3], _request().factuals[:, 3])
    np.testing.assert_array_equal(result.candidates[:, 0, 1:3].sum(axis=1), 1.0)
    np.testing.assert_array_equal(
        _context().oracle.predict(result.candidates[:, 0]), [2, 2]
    )
    with pytest.raises(ValueError, match="exactly one"):
        prepared.generate(
            GenerationRequest(_request().factuals, _request().targets, 2, seed=23)
        )

    def fake_missing(*args, **kwargs):
        factuals = np.asarray(args[3]).copy()
        return factuals, [
            {
                "returned": False,
                "found": False,
                "valid_candidates": 0,
                "attempts": 1,
                "runtime_s": 0.0,
            }
            for _ in factuals
        ]

    monkeypatch.setattr(dice_module, "generate_dice_counterfactuals", fake_missing)
    missing = prepared.generate(_request(seed=31))
    assert not missing.available.any()
    assert np.isnan(missing.candidates).all()
    np.testing.assert_array_equal(
        missing.artifacts["method.best_effort"], _request().factuals
    )
    np.testing.assert_array_equal(
        missing.artifacts["method.raw_candidates"], _request().factuals
    )


def test_method_modules_import_without_optional_runtime_initialization():
    code = """
import sys
for name in (
    'experiments.zeroshot_cf.methods.base',
    'experiments.zeroshot_cf.methods.nice',
    'experiments.zeroshot_cf.methods.optimization',
    'experiments.zeroshot_cf.methods.dice',
    'experiments.zeroshot_cf.methods.face',
):
    __import__(name)
assert 'dice_ml' not in sys.modules
assert 'raiutils' not in sys.modules
for forbidden in (
    'experiments.zeroshot_cf.benchmark_protocol',
    'experiments.zeroshot_cf.methods.legacy',
    'experiments.zeroshot_cf.evaluation',
    'experiments.zeroshot_cf.orchestration',
    'pandas',
    'scipy',
    'sklearn',
):
    assert not any(
        loaded == forbidden or loaded.startswith(forbidden + '.')
        for loaded in sys.modules
    ), forbidden
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
