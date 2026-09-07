"""Conformance, rejection, and parity gates for CounterContEx proposal backends."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
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
from experiments.zeroshot_cf.generator import (
    TabICLGeneratorInputs,
    TabICLGeneratorPointBackend,
    generate_counterfactual_batch,
)
from experiments.zeroshot_cf.methods.countercontex.backends.base import (
    CategoryProposals,
    PreparedBackend,
    ProposalBackend,
    ProposalCapabilities,
    ProposalSession,
)
from experiments.zeroshot_cf.methods.countercontex.backends.empirical import (
    EmpiricalBackend,
    LocalEmpiricalBackend,
)
from experiments.zeroshot_cf.methods.countercontex.backends.tabicl import (
    TabICLProposalSession,
)
from experiments.zeroshot_cf.methods.countercontex.config import (
    CounterContExConfig,
    CounterContExFoundationConfig,
    CounterContExSearchConfig,
)
from experiments.zeroshot_cf.methods.countercontex.method import CounterContExMethod
from experiments.zeroshot_cf.methods.countercontex.runtime import resolve_runtime
from experiments.zeroshot_cf.methods.countercontex.search import (
    _SessionSampler,
    generate_with_backend,
)
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config


class _Oracle:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return (np.asarray(X)[:, 0] >= 0.5).astype(int)

    def predict_proba(self, X):
        probability = np.clip(np.asarray(X)[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - probability, probability))


class _LegacySampler:
    def sample_candidates(
        self,
        X_query,
        candidate_cols,
        *,
        sample_temperature,
        fixed_target,
        fixed_confidence=None,
    ):
        del X_query, sample_temperature, fixed_target, fixed_confidence
        return np.full(len(candidate_cols), 0.9)


@dataclass(frozen=True)
class _FakeSession:
    confidence_anchors: tuple[float, ...] | None = None
    diagnostics = {
        "categorical_confidence_batching": False,
        "conditional_estimator_cache": False,
        "tabicl_kv_cache": False,
    }

    def propose_numerical(
        self,
        rows,
        columns,
        *,
        quantiles,
        confidence,
        temperature,
    ):
        del rows, confidence, temperature
        if quantiles is None:
            return np.full(len(columns), 0.9)
        return np.full((len(columns), len(quantiles)), 0.9)

    def propose_numerical_batch(
        self,
        rows,
        columns,
        *,
        quantiles,
        confidences,
        temperature,
    ):
        del rows, confidences, temperature
        if quantiles is None:
            return np.full(len(columns), 0.9)
        return np.full((len(columns), 1, len(quantiles)), 0.9)

    def categorical_distribution(self, row, group, *, confidence):
        del row, confidence
        probability = np.full(len(group.columns), 1.0 / len(group.columns))
        return CategoryProposals(np.arange(len(group.columns)), probability)

    def score_joint(self, rows, target):
        del target
        return np.asarray(rows)[:, 0]


@dataclass(frozen=True)
class _FakeBackend:
    capabilities: ProposalCapabilities
    backend_id: str = "fake"

    def prepare(self, context):
        del context
        return self

    def for_factual(self, factual, target, *, seed):
        del factual, target, seed
        return _FakeSession()


@dataclass(frozen=True)
class _JointRejectingSession(_FakeSession):
    def score_joint(self, rows, target):
        del rows, target
        raise AssertionError("sparse mode must not activate joint scoring")


@dataclass(frozen=True)
class _JointCapableBackend(_FakeBackend):
    def for_factual(self, factual, target, *, seed):
        del factual, target, seed
        return _JointRejectingSession()


def _context() -> MethodContext:
    return MethodContext(
        X_reference=np.array([[0.0], [0.4], [0.8], [1.0]]),
        feature_schema=FeatureSchema(
            names=("amount",),
            numerical=(0,),
            categorical_groups=(),
            actionable_scalars=(0,),
            actionable_groups=(),
            immutable=(),
            domains=FeatureDomains(
                lower=np.zeros(1),
                upper=np.ones(1),
                discrete={},
            ),
        ),
        oracle=_Oracle(),
    )


def _categorical_context() -> MethodContext:
    group = OneHotActionGroup("segment", (1, 2))
    return MethodContext(
        X_reference=np.array([[0.0, 1.0, 0.0], [0.4, 1.0, 0.0], [0.8, 0.0, 1.0]]),
        feature_schema=FeatureSchema(
            names=("amount", "segment_a", "segment_b"),
            numerical=(0,),
            categorical_groups=(group,),
            actionable_scalars=(0,),
            actionable_groups=(group,),
            immutable=(),
            domains=FeatureDomains(
                lower=np.zeros(3),
                upper=np.ones(3),
                discrete={},
            ),
        ),
        oracle=_Oracle(),
    )


def _fake_config(*, search=None, confidence_quantiles=None) -> CounterContExConfig:
    return CounterContExConfig(
        search=search or CounterContExSearchConfig(),
        foundation=CounterContExFoundationConfig(
            backend="fake",
            confidence_quantiles=confidence_quantiles,
        ),
    )


def test_fake_backend_conforms_to_explicit_protocols() -> None:
    backend = _FakeBackend(
        ProposalCapabilities(
            confidence_conditioning=True,
            categorical_distribution=True,
            joint_scoring=True,
        )
    )
    prepared = backend.prepare(_context())
    session = prepared.for_factual(np.array([0.1]), 1, seed=137)

    assert isinstance(backend, ProposalBackend)
    assert isinstance(prepared, PreparedBackend)
    assert isinstance(session, ProposalSession)
    np.testing.assert_array_equal(
        session.propose_numerical(
            np.array([[0.1]]),
            [0],
            quantiles=(0.25, 0.75),
            confidence=None,
            temperature=0.0,
        ),
        [[0.9, 0.9]],
    )
    np.testing.assert_array_equal(session.score_joint(np.array([[0.2]]), 1), [0.2])


def test_tabicl_beam_grid_uses_one_native_batch_call() -> None:
    calls = []

    class NativeSampler:
        def sample_candidate_grid_batch(self, rows, columns, **kwargs):
            calls.append((np.asarray(rows).copy(), tuple(columns), kwargs))
            return np.full((len(columns), 2, 3), 0.9)

    session = TabICLProposalSession(
        SimpleNamespace(
            sampler=NativeSampler(),
            candidate_confidences=(0.4, 0.8),
            metadata={},
        ),
        target=1,
    )
    sampler = _SessionSampler(session)

    values = sampler.sample_candidate_grid_batch(
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        [0, 1],
        quantiles=(0.25, 0.5, 0.75),
        fixed_target=1,
        confidences=(0.4, 0.8),
    )

    assert values.shape == (2, 2, 3)
    assert len(calls) == 1


def test_empirical_backend_is_deterministic_and_conforms() -> None:
    backend = EmpiricalBackend()
    prepared = backend.prepare(_categorical_context())
    first = prepared.for_factual(np.array([0.1, 1.0, 0.0]), 1, seed=17)
    second = prepared.for_factual(np.array([0.1, 1.0, 0.0]), 1, seed=999)

    assert isinstance(backend, ProposalBackend)
    assert isinstance(prepared, PreparedBackend)
    assert isinstance(first, ProposalSession)
    first_values = first.propose_numerical(
        np.array([[0.1, 1.0, 0.0]]),
        [0],
        quantiles=(0.25, 0.75),
        confidence=None,
        temperature=0.0,
    )
    second_values = second.propose_numerical(
        np.array([[0.1, 1.0, 0.0]]),
        [0],
        quantiles=(0.25, 0.75),
        confidence=None,
        temperature=1.0,
    )
    np.testing.assert_array_equal(first_values, second_values)
    proposals = first.categorical_distribution(
        np.array([0.1, 1.0, 0.0]),
        OneHotActionGroup("segment", (1, 2)),
        confidence=None,
    )
    np.testing.assert_array_equal(proposals.categories, [0, 1])
    np.testing.assert_allclose(proposals.probabilities, [1 / 3, 2 / 3])


def test_local_empirical_backend_is_factual_local_and_seed_independent() -> None:
    class MarkerOracle:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return (np.asarray(X)[:, 1] > 0.005).astype(int)

        def predict_proba(self, X):
            target = self.predict(X).astype(float)
            return np.column_stack((1.0 - target, target))

    local_non_targets = np.tile([0.2, 0.0], (502, 1))
    local_targets = np.tile([0.2, 0.01], (10, 1))
    distant_targets = np.tile([0.9, 0.01], (100, 1))
    reference = np.vstack((local_non_targets, local_targets, distant_targets))
    context = MethodContext(
        X_reference=reference,
        feature_schema=FeatureSchema(
            names=("amount", "marker"),
            numerical=(0, 1),
            categorical_groups=(),
            actionable_scalars=(0, 1),
            actionable_groups=(),
            immutable=(),
            domains=FeatureDomains(
                lower=np.zeros(2),
                upper=np.ones(2),
                discrete={},
            ),
        ),
        oracle=MarkerOracle(),
    )
    prepared = LocalEmpiricalBackend().prepare(context)

    near = prepared.for_factual(np.array([0.2, 0.0]), 1, seed=17)
    repeated = prepared.for_factual(np.array([0.2, 0.0]), 1, seed=999)
    distant = prepared.for_factual(np.array([0.9, 0.0]), 1, seed=17)
    proposals = [
        session.propose_numerical(
            np.array([[0.0, 0.0]]),
            [0],
            quantiles=None,
            confidence=None,
            temperature=temperature,
        )
        for session, temperature in ((near, 0.0), (repeated, 1.0), (distant, 0.0))
    ]

    np.testing.assert_array_equal(proposals[0], proposals[1])
    np.testing.assert_allclose(proposals[0], [0.2])
    np.testing.assert_allclose(proposals[2], [0.9])


def test_local_empirical_filters_target_before_categorical_frequencies() -> None:
    class MarkerOracle:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return (np.asarray(X)[:, 0] > 0.5).astype(int)

        def predict_proba(self, X):
            target = self.predict(X).astype(float)
            return np.column_stack((1.0 - target, target))

    group = OneHotActionGroup("segment", (1, 2))
    context = MethodContext(
        X_reference=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.1, 1.0, 0.0],
                [0.8, 1.0, 0.0],
                [0.8, 0.0, 1.0],
                [0.9, 0.0, 1.0],
            ]
        ),
        feature_schema=FeatureSchema(
            names=("marker", "segment_a", "segment_b"),
            numerical=(0,),
            categorical_groups=(group,),
            actionable_scalars=(0,),
            actionable_groups=(group,),
            immutable=(),
            domains=FeatureDomains(
                lower=np.zeros(3),
                upper=np.ones(3),
                discrete={},
            ),
        ),
        oracle=MarkerOracle(),
    )

    proposals = (
        LocalEmpiricalBackend()
        .prepare(context)
        .for_factual(np.array([0.0, 1.0, 0.0]), 1, seed=3)
        .categorical_distribution(np.array([0.0, 1.0, 0.0]), group, confidence=None)
    )

    np.testing.assert_array_equal(proposals.categories, [0, 1])
    np.testing.assert_allclose(proposals.probabilities, [2 / 5, 3 / 5])


def test_local_empirical_neighborhood_counts_one_hot_group_once() -> None:
    class Oracle:
        classes_ = np.array([0, 1])

        def predict(self, X):
            matrix = np.asarray(X)
            return ((matrix[:, 0] + matrix[:, 1] > 0.5) | (matrix[:, 3] == 1)).astype(
                int
            )

        def predict_proba(self, X):
            target = self.predict(X).astype(float)
            return np.column_stack((1.0 - target, target))

    group = OneHotActionGroup("segment", (2, 3))
    reference = np.vstack(
        (
            np.tile([0.0, 0.0, 1.0, 0.0], (511, 1)),
            [0.75, 0.75, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
    )
    context = MethodContext(
        X_reference=reference,
        feature_schema=FeatureSchema(
            names=("first", "second", "segment_a", "segment_b"),
            numerical=(0, 1),
            categorical_groups=(group,),
            actionable_scalars=(0, 1),
            actionable_groups=(group,),
            immutable=(),
            domains=FeatureDomains(
                lower=np.zeros(4),
                upper=np.ones(4),
                discrete={},
            ),
        ),
        oracle=Oracle(),
    )

    session = (
        LocalEmpiricalBackend()
        .prepare(context)
        .for_factual(np.array([0.0, 0.0, 1.0, 0.0]), 1, seed=1)
    )

    np.testing.assert_array_equal(session.reference, [[0.0, 0.0, 0.0, 1.0]])


def test_local_empirical_falls_back_to_global_target_rows() -> None:
    reference = np.concatenate((np.zeros((512, 1)), np.ones((1, 1))))
    prepared = LocalEmpiricalBackend().prepare(
        MethodContext(
            X_reference=reference,
            feature_schema=_context().feature_schema,
            oracle=_Oracle(),
        )
    )

    session = prepared.for_factual(np.array([0.0]), 1, seed=1)

    assert session.diagnostics["local_target_fallback"]
    np.testing.assert_array_equal(
        session.propose_numerical(
            np.array([[0.0]]),
            [0],
            quantiles=None,
            confidence=None,
            temperature=0.0,
        ),
        [1.0],
    )


def test_local_empirical_rejects_target_absent_from_reference() -> None:
    prepared = LocalEmpiricalBackend().prepare(_context())

    with pytest.raises(ValueError, match="no reference rows for target class 2"):
        prepared.for_factual(np.array([0.0]), 2, seed=1)


def test_local_empirical_has_distinct_runtime_identity() -> None:
    runtime = resolve_runtime({"foundation": {"backend": "empirical_local"}}, {}, None)

    assert runtime.backend_implementation == "empirical-local-reference-v1"


def test_empirical_backend_runs_complete_countercontex_sparse_search() -> None:
    method = CounterContExMethod(
        CounterContExConfig(
            foundation=CounterContExFoundationConfig(backend="empirical")
        )
    )
    prepared = method.prepare(_context())

    result = prepared.generate(
        GenerationRequest(
            factuals=np.array([[0.1]]),
            targets=np.array([1]),
            n_counterfactuals=1,
            seed=137,
        )
    )

    np.testing.assert_array_equal(result.available, [[True]])
    np.testing.assert_allclose(result.candidates, [[[0.9]]])
    assert result.run_diagnostics["proposal_backend"] == "empirical"


@pytest.mark.parametrize(
    "config",
    [
        CounterContExConfig(
            search=CounterContExSearchConfig(candidate_quantiles=(0.5,)),
            foundation=CounterContExFoundationConfig(
                backend="empirical", confidence_quantiles=(0.5,)
            ),
        ),
        CounterContExConfig(
            search=CounterContExSearchConfig(cf_mode="data_plausible"),
            foundation=CounterContExFoundationConfig(backend="empirical"),
        ),
    ],
)
def test_empirical_backend_rejects_unsupported_search_requests(config) -> None:
    with pytest.raises(ValueError, match="confidence conditioning|joint scoring"):
        CounterContExMethod(config).prepare(_context())


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            CounterContExConfig(
                search=CounterContExSearchConfig(candidate_quantiles=(0.5,)),
                foundation=CounterContExFoundationConfig(
                    backend="empirical_local", confidence_quantiles=(0.5,)
                ),
            ),
            "confidence conditioning",
        ),
        (
            CounterContExConfig(
                search=CounterContExSearchConfig(cf_mode="data_plausible"),
                foundation=CounterContExFoundationConfig(backend="empirical_local"),
            ),
            "joint scoring",
        ),
    ],
)
def test_local_empirical_rejects_unsupported_search_requests(config, message) -> None:
    with pytest.raises(ValueError, match=message):
        CounterContExMethod(config).prepare(_context())


@pytest.mark.parametrize(
    ("capabilities", "config", "message"),
    [
        (
            ProposalCapabilities(numerical_proposals=False),
            _fake_config(),
            "numerical proposals",
        ),
        (
            ProposalCapabilities(),
            _fake_config(
                search=CounterContExSearchConfig(candidate_quantiles=(0.5,)),
                confidence_quantiles=(0.5,),
            ),
            "confidence conditioning",
        ),
        (
            ProposalCapabilities(),
            _fake_config(search=CounterContExSearchConfig(cf_mode="data_plausible")),
            "joint scoring",
        ),
    ],
)
def test_unsupported_backend_capabilities_fail_during_prepare(
    capabilities,
    config,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        CounterContExMethod(config, _FakeBackend(capabilities)).prepare(_context())


def test_categorical_search_requires_declared_distribution_support() -> None:
    with pytest.raises(ValueError, match="categorical distributions"):
        method = CounterContExMethod(
            _fake_config(), _FakeBackend(ProposalCapabilities())
        )
        method.prepare(_categorical_context())


def test_fake_backend_adapter_preserves_legacy_candidate_arrays() -> None:
    inputs = TabICLGeneratorInputs(
        factuals=np.array([[0.1]]),
        targets=np.array([1]),
        numerical_columns=(0,),
        categorical_groups=(),
        feature_domains=(np.zeros(1), np.ones(1), {}),
    )
    config = _fake_config()
    legacy = generate_counterfactual_batch(
        inputs,
        discriminator=_Oracle(),
        config=config.generator_config(1),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=_LegacySampler()
        ),
    )
    portable = generate_with_backend(
        inputs,
        discriminator=_Oracle(),
        config=config,
        backend=_FakeBackend(ProposalCapabilities()),
        seed=42,
        n_counterfactuals=1,
    )

    assert (
        legacy.counterfactual_sets.tobytes() == portable.counterfactual_sets.tobytes()
    )
    assert legacy.counterfactuals.tobytes() == portable.counterfactuals.tobytes()


def test_sparse_search_does_not_activate_joint_scoring_capability() -> None:
    inputs = TabICLGeneratorInputs(
        factuals=np.array([[0.1]]),
        targets=np.array([1]),
        numerical_columns=(0,),
        categorical_groups=(),
        feature_domains=(np.zeros(1), np.ones(1), {}),
    )
    backend = _JointCapableBackend(ProposalCapabilities(joint_scoring=True))

    result = generate_with_backend(
        inputs,
        discriminator=_Oracle(),
        config=_fake_config(),
        backend=backend,
        seed=42,
        n_counterfactuals=1,
    )

    np.testing.assert_array_equal(result.counterfactual_sets, [[[0.9]]])


def test_search_contract_has_no_dynamic_capability_probes() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "methods/countercontex/search.py",
        "methods/countercontex/backends/base.py",
    ):
        tree = ast.parse((root / relative).read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"hasattr", "getattr"}
        ]
        any_names = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "Any"
        ]
        assert calls == []
        assert any_names == []


def test_portable_layers_do_not_import_foundation_or_concrete_methods() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = (
        "tabicl",
        "tabpfn",
        "tabfm",
        ".methods.",
        "exp8_",
        "exp9_",
        "exp11_",
        "exp12_",
        "exp13_",
        "exp14_",
    )
    for package in ("datasets", "evaluation"):
        for source in (root / package).glob("*.py"):
            tree = ast.parse(source.read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module.lower())
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name.lower() for alias in node.names)
            assert not any(
                token in imported for token in forbidden for imported in imports
            ), source


def test_tracked_ablation_and_full_reference_matrices_resolve_exact_axes() -> None:
    root = Path(__file__).resolve().parents[1]
    config_root = root / "configs" / "matrices"
    ablation = load_matrix_config(config_root / "countercontex_ablation_example.yaml")
    reference = load_matrix_config(config_root / "full_reference.yaml")

    assert len(ablation.runs) == 24
    assert {run.dataset.name for run in ablation.runs} == {
        "heloc",
        "bank_marketing",
    }
    assert {run.seed for run in ablation.runs} == {17, 42}
    assert {run.method.params["foundation"]["backend"] for run in ablation.runs} == {
        "tabicl",
        "empirical",
    }
    assert {run.method.n_counterfactuals for run in ablation.runs} == {1, 3}
    assert {run.method.params["search"]["cf_mode"] for run in ablation.runs} == {
        "sparse",
        "data_plausible",
    }
    backend_pair = [
        run
        for run in ablation.runs
        if run.dataset.name == "heloc"
        and run.seed == 17
        and run.method.params.get("search", {}).get("cf_mode") == "sparse"
        and run.method.params.get("search", {}).get("max_validity_steps") == 50
        and run.method.params["foundation"]["backend"] in {"tabicl", "empirical"}
    ]
    assert len(backend_pair) == 2
    normalized = []
    for run in backend_pair:
        params = {
            section: dict(values) for section, values in run.method.params.items()
        }
        params["foundation"]["backend"] = "normalized"
        normalized.append(params)
    assert normalized[0] == normalized[1]

    assert len(reference.runs) == 24
    assert reference.execution.legacy_export
    assert reference.execution.output_root == Path(
        "experiments/zeroshot_cf/results/local/architecture_full_reference"
    )
    assert {run.dataset.name for run in reference.runs} == {
        "heloc",
        "bank_marketing",
        "give_me_some_credit",
        "lending_club",
    }
    assert {run.method.name for run in reference.runs} == {
        "countercontex",
        "nice",
        "wachter",
        "growing_spheres",
        "dice",
        "face",
    }
    assert {run.seed for run in reference.runs} == {42}
    assert {run.protocol.max_test for run in reference.runs} == {1000}
    countercontex = next(
        run for run in reference.runs if run.method.name == "countercontex"
    )
    assert countercontex.method.n_counterfactuals == 3
    assert countercontex.method.params["foundation"]["backend"] == "tabicl"
