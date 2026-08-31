"""Contract tests for the benchmark-facing DiCoFlex method."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.zeroshot_cf.core.contracts import (
    FeatureDomains,
    FeatureSchema,
    GenerationRequest,
    MethodContext,
)
from experiments.zeroshot_cf.generator import (
    TabICLGeneratorResult,
    generate_counterfactual_batch,
)
from experiments.zeroshot_cf.methods.base import CounterfactualMethod, PreparedMethod
from experiments.zeroshot_cf.methods.dicoflex.backend import (
    DiCoFlexBackendInputs,
    DiCoFlexBackendRuntime,
    prepare_backend,
)
from experiments.zeroshot_cf.methods.dicoflex.backends.base import ProposalCapabilities
from experiments.zeroshot_cf.methods.dicoflex.config import (
    DiCoFlexConfig,
    DiCoFlexFoundationConfig,
    DiCoFlexSearchConfig,
)
from experiments.zeroshot_cf.methods.dicoflex.method import (
    DiCoFlexMethod,
    PreparedDiCoFlexMethod,
    adapt_generator_result,
)


class _Oracle:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return (np.asarray(X)[:, 0] >= 0.5).astype(int)

    def predict_proba(self, X):
        probability = np.clip(np.asarray(X)[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])


def _context() -> MethodContext:
    schema = FeatureSchema(
        names=("amount", "fixed"),
        numerical=(0, 1),
        categorical_groups=(),
        actionable_scalars=(0,),
        actionable_groups=(),
        immutable=(1,),
        domains=FeatureDomains(
            lower=np.zeros(2),
            upper=np.ones(2),
            discrete={},
        ),
    )
    return MethodContext(
        X_reference=np.array([[0.0, 0.1], [0.4, 0.2], [0.8, 0.3], [1.0, 0.4]]),
        feature_schema=schema,
        oracle=_Oracle(),
    )


def _retained_result(counts: tuple[int, ...], k: int) -> TabICLGeneratorResult:
    factuals = np.array([[0.1, 0.1], [0.2, 0.2]])[: len(counts)]
    sets = np.full((len(counts), k, 2), np.nan)
    for row, count in enumerate(counts):
        for rank in range(count):
            sets[row, rank] = [0.7 + rank * 0.1, factuals[row, 1]]
    primary = factuals.copy()
    for row, count in enumerate(counts):
        if count:
            primary[row] = sets[row, 0]
    diagnostics = SimpleNamespace(
        diverse_available_count_per_point=np.asarray(counts),
        diverse_candidate_pool_count_per_point=np.asarray(counts) + 1,
        diverse_search_depth_per_point=np.asarray(counts) + 2,
        flipped_per_point=tuple(count > 0 for count in counts),
        changed_per_point=tuple((0,) if count else () for count in counts),
        steps_per_point=tuple(1 if count else 0 for count in counts),
        validity_steps_per_point=tuple(1 if count else 0 for count in counts),
        refinement_steps_per_point=tuple(0 for _ in counts),
        accepted_refinement_count_per_point=tuple(0 for _ in counts),
        target_probability_per_point=np.asarray(
            [0.8 if count else 0.2 for count in counts]
        ),
        point_runtime_s=np.full(len(counts), 0.01),
        joint_scoring_runtime_s_per_point=np.zeros(len(counts)),
        cf_mode="sparse",
        conditional_estimator_cache=True,
        tabicl_kv_cache=False,
        runtime_s=0.1,
    )
    return TabICLGeneratorResult(
        factuals=factuals,
        targets=np.ones(len(counts), dtype=int),
        counterfactuals=primary,
        sparse_counterfactuals=primary.copy(),
        counterfactual_sets=sets,
        diagnostics=diagnostics,
    )


@pytest.mark.parametrize(
    ("counts", "k", "expected_available"),
    [
        ((1, 0), 1, [[True], [False]]),
        ((3, 1), 3, [[True, True, True], [True, False, False]]),
    ],
)
def test_adapter_preserves_genuine_k1_and_partial_k3_returns(
    counts,
    k,
    expected_available,
) -> None:
    retained = _retained_result(counts, k)

    result = adapt_generator_result(retained, seed=42)

    np.testing.assert_array_equal(result.available, expected_available)
    assert np.isnan(result.candidates[~result.available]).all()
    result.validate_for_factuals(retained.factuals)
    assert result.run_diagnostics["proposal_backend"] == "conditional_density"
    assert "tabicl_kv_cache" not in result.run_diagnostics


def test_seed_42_adapter_is_exactly_equivalent_to_legacy_available_slots() -> None:
    retained = _retained_result((2, 1), 3)

    canonical = adapt_generator_result(retained, seed=42)

    np.testing.assert_allclose(
        canonical.candidates[canonical.available],
        retained.counterfactual_sets[canonical.available],
    )
    np.testing.assert_allclose(
        canonical.artifacts["method.best_effort"],
        retained.counterfactuals,
    )
    assert canonical.run_diagnostics["seed"] == 42


def test_legacy_runtime_resolves_seed_42_without_changing_public_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.zeroshot_cf.tabicl_runtime as runtime

    captured: dict[str, int] = {}

    class _Backend:
        def point_backend_factory(self, *, seed):
            captured["seed"] = seed
            return object()

    monkeypatch.setattr(runtime, "prepare_backend", lambda *args, **kwargs: _Backend())
    context = SimpleNamespace(
        bundle=SimpleNamespace(X_train=np.zeros((2, 1))),
        categorical_groups=(),
        grouped_actionable=(),
        disc_model=object(),
    )

    runtime._build_point_backend_factory(
        context,
        confidence_quantiles=None,
        cf_mode="sparse",
        tabicl_joint_permutations=1,
        n_estimators=1,
        temperature=0.0,
        cache_dir=None,
    )

    assert captured["seed"] == 42
    assert "seed" not in inspect.signature(runtime.run_tabicl_benchmark).parameters
    assert tuple(inspect.signature(generate_counterfactual_batch).parameters) == (
        "inputs",
        "discriminator",
        "config",
        "point_backend_factory",
    )


def test_config_serialization_separates_search_diversity_and_foundation() -> None:
    config = DiCoFlexConfig(
        search=DiCoFlexSearchConfig(candidate_quantiles=(0.25, 0.75)),
        foundation=DiCoFlexFoundationConfig(
            confidence_quantiles=(0.5,),
            cache_dir=Path("cache/tabicl"),
        ),
    )

    serialized = config.as_dict()

    assert set(serialized) == {"search", "diversity", "foundation"}
    assert serialized["search"]["candidate_quantiles"] == (0.25, 0.75)
    assert serialized["foundation"]["cache_dir"] == "cache/tabicl"


def test_method_prepare_owns_portable_backend_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.zeroshot_cf.methods.dicoflex.method as method_module

    captured: dict[str, object] = {}
    backend = SimpleNamespace(
        capabilities=ProposalCapabilities(
            confidence_conditioning=True,
            categorical_distribution=True,
            joint_scoring=True,
        )
    )

    def fake_prepare_backend(inputs, config):
        captured["inputs"] = inputs
        captured["config"] = config
        return backend

    monkeypatch.setattr(method_module, "prepare_backend", fake_prepare_backend)
    method = DiCoFlexMethod()
    context = _context()

    prepared = method.prepare(context)

    assert isinstance(method, CounterfactualMethod)
    assert isinstance(prepared, PreparedMethod)
    assert prepared.backend is backend
    assert captured["inputs"].oracle is context.oracle
    assert captured["config"] is method.config


def test_method_passes_request_k_seed_and_portable_domains_to_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    captures: dict[str, object] = {}

    backend = SimpleNamespace(backend_id="tabicl")

    def fake_generate(
        inputs,
        *,
        discriminator,
        config,
        backend,
        seed,
        n_counterfactuals,
    ):
        captures["inputs"] = inputs
        captures["discriminator"] = discriminator
        captures["config"] = config
        captures["backend"] = backend
        captures["seed"] = seed
        captures["n_counterfactuals"] = n_counterfactuals
        return _retained_result((1, 1), 3)

    monkeypatch.setattr(
        "experiments.zeroshot_cf.methods.dicoflex.method.generate_with_backend",
        fake_generate,
    )
    request = GenerationRequest(
        factuals=np.array([[0.1, 0.1], [0.2, 0.2]]),
        targets=np.ones(2, dtype=int),
        n_counterfactuals=3,
        seed=137,
    )

    result = PreparedDiCoFlexMethod(context, DiCoFlexConfig(), backend).generate(
        request
    )

    assert captures["seed"] == 137
    assert captures["n_counterfactuals"] == 3
    assert captures["backend"] is backend
    lower, upper, supports = captures["inputs"].feature_domains
    np.testing.assert_array_equal(lower, [0.0, 0.0])
    np.testing.assert_array_equal(upper, [1.0, 1.0])
    assert supports == {}
    assert result.candidates.shape == (2, 3, 2)


def test_nondefault_seed_reaches_proposal_and_joint_sampler_contexts() -> None:
    constructed: list[object] = []

    class _Sampler:
        estimator_params = {"kv_cache": True}

        def __init__(self, **kwargs):
            self.random_state = kwargs["random_state"]
            constructed.append(self)

        def set_context(self, *args, **kwargs):
            self.context_random_state = self.random_state
            return self

    class _JointScorer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    context = _context()
    config = DiCoFlexConfig(
        search=DiCoFlexSearchConfig(cf_mode="data_plausible"),
    )
    backend = prepare_backend(
        DiCoFlexBackendInputs(
            X_reference=context.X_reference,
            categorical_groups=(),
            actionable_groups=(),
            oracle=context.oracle,
        ),
        config,
        runtime=DiCoFlexBackendRuntime(
            device="cpu",
            sampler_type=_Sampler,
            joint_scorer_type=_JointScorer,
        ),
    )

    factory = backend.point_backend_factory(seed=137)
    factory(np.array([0.1, 0.1]), 1)

    assert [sampler.random_state for sampler in constructed] == [137, 137]
    assert [sampler.context_random_state for sampler in constructed] == [137, 137]


def test_method_import_does_not_initialize_tabicl_or_checkpoint_runtime() -> None:
    code = """
import sys
import experiments.zeroshot_cf.methods.dicoflex.method
for forbidden in (
    'experiments.zeroshot_cf.tabicl_checkpoints',
    'experiments.zeroshot_cf.tabicl_sampler',
    'experiments.zeroshot_cf.tabicl_joint_plausibility',
):
    assert forbidden not in sys.modules, forbidden
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
