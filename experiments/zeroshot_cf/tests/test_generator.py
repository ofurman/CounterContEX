"""Public-API tests for the retained TabICL generator boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import experiments.zeroshot_cf.exp8_tabicl_cf as exp8

from experiments.zeroshot_cf.diverse_search import DiverseBeamSearchConfig
from experiments.zeroshot_cf.generator import (
    TabICLGeneratorConfig,
    TabICLGeneratorInputs,
    TabICLGeneratorPointBackend,
    generate_counterfactual_batch,
)
from experiments.zeroshot_cf.tabicl_joint_plausibility import TabICLJointScoreBatch


class _LinearDisc:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        matrix = np.asarray(X, dtype=float)
        p1 = np.clip(0.1 + 0.5 * matrix[:, 0] + 0.05 * matrix[:, 1], 0.0, 0.99)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _SingleFeatureDisc:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        matrix = np.asarray(X, dtype=float)
        p1 = np.clip(0.1 + 0.6 * matrix[:, 0], 0.0, 0.99)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _ConfidenceGridSampler:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def sample_candidate_grid(
        self,
        _query,
        columns,
        *,
        quantiles,
        fixed_target,
        confidences=None,
    ):
        self.calls.append(
            {
                "columns": tuple(columns),
                "quantiles": tuple(float(value) for value in quantiles),
                "confidences": None
                if confidences is None
                else tuple(float(value) for value in confidences),
                "fixed_target": int(fixed_target),
            }
        )
        assert tuple(columns) == (0,)
        return np.array([[[0.2, 0.4], [0.7, 0.9]]], dtype=float)


class _StatefulRevisitSampler:
    def sample_candidate_grid(
        self,
        query,
        columns,
        *,
        quantiles,
        fixed_target,
        confidences=None,
    ):
        del quantiles, fixed_target, confidences
        row = np.asarray(query, dtype=float)[0]
        values = []
        for column in columns:
            if column == 0:
                values.append([1.0] if row[0] >= 0.59 else [0.6])
            else:
                values.append([0.2])
        return np.asarray(values, dtype=float)


class _TwoFeatureSampler:
    def sample_candidate_grid(
        self,
        _query,
        columns,
        *,
        quantiles,
        fixed_target,
        confidences=None,
    ):
        del quantiles, fixed_target, confidences
        return np.ones((len(columns), 1), dtype=float)


class _AcceptanceScorer:
    def score_rows(self, rows, target_class):
        assert target_class == 1
        return TabICLJointScoreBatch(
            joint_log_density=np.array([-9.95, -9.45], dtype=float)
        )


class _RejectionScorer:
    def score_rows(self, rows, target_class):
        assert target_class == 1
        return TabICLJointScoreBatch(
            joint_log_density=np.array([-9.95, -9.96], dtype=float)
        )


class _SingleFeatureSampler:
    def sample_candidate_grid(
        self,
        _query,
        columns,
        *,
        quantiles,
        fixed_target,
        confidences=None,
    ):
        del quantiles, fixed_target, confidences
        return np.ones((len(columns), 1), dtype=float)


def test_public_generator_forwards_confidence_quantiles_and_preserves_immutable() -> None:
    sampler = _ConfidenceGridSampler()
    result = generate_counterfactual_batch(
        TabICLGeneratorInputs(
            factuals=np.array([[0.0, 5.0]], dtype=float),
            targets=np.array([1], dtype=int),
            numerical_columns=(0,),
            categorical_groups=(),
            immutable_idx=(1,),
        ),
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.25, 0.75),
            confidence_quantiles=(0.25, 0.75),
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=sampler,
            candidate_confidences=(0.55, 0.85),
            metadata={
                "categorical_confidence_batching": True,
                "conditional_estimator_cache": True,
                "tabicl_kv_cache": True,
            },
        ),
    )

    np.testing.assert_allclose(result.counterfactuals, [[0.7, 5.0]])
    assert sampler.calls == [
        {
            "columns": (0,),
            "quantiles": (0.25, 0.75),
            "confidences": (0.55, 0.85),
            "fixed_target": 1,
        }
    ]
    assert result.diagnostics.flipped_per_point == (True,)
    np.testing.assert_allclose(result.diagnostics.target_probability_per_point, [0.52])


def test_public_generator_revisit_control_changes_whether_validity_is_reached() -> None:
    factuals = np.array([[0.0, 0.0]], dtype=float)
    targets = np.array([1], dtype=int)
    inputs = TabICLGeneratorInputs(
        factuals=factuals,
        targets=targets,
        numerical_columns=(0, 1),
        categorical_groups=(),
    )

    no_revisit = generate_counterfactual_batch(
        inputs,
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.5,),
            max_validity_steps=2,
            allow_revisits=False,
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=_StatefulRevisitSampler()
        ),
    )
    with_revisit = generate_counterfactual_batch(
        inputs,
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.5,),
            max_validity_steps=2,
            allow_revisits=True,
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=_StatefulRevisitSampler()
        ),
    )

    assert no_revisit.diagnostics.flipped_per_point == (False,)
    np.testing.assert_allclose(no_revisit.counterfactuals, [[0.6, 0.0]])
    assert with_revisit.diagnostics.flipped_per_point == (True,)
    np.testing.assert_allclose(with_revisit.counterfactuals, [[1.0, 0.0]])


def test_public_generator_caps_pre_validity_steps() -> None:
    result = generate_counterfactual_batch(
        TabICLGeneratorInputs(
            factuals=np.array([[0.0, 0.0]], dtype=float),
            targets=np.array([1], dtype=int),
            numerical_columns=(0, 1),
            categorical_groups=(),
        ),
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.5,),
            max_validity_steps=1,
            allow_revisits=True,
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=_StatefulRevisitSampler()
        ),
    )

    assert result.diagnostics.flipped_per_point == (False,)
    assert result.diagnostics.validity_steps_per_point == (1,)
    np.testing.assert_allclose(result.counterfactuals, [[0.6, 0.0]])


@pytest.mark.parametrize(
    ("scorer", "expected", "accepted", "reason"),
    [
        (_AcceptanceScorer(), [1.0, 1.0], 1, "one_shot_accepted"),
        (_RejectionScorer(), [1.0, 0.0], 0, "no_improving_candidate"),
    ],
)
def test_public_generator_exposes_refinement_acceptance_and_rejection(
    scorer,
    expected,
    accepted,
    reason,
) -> None:
    result = generate_counterfactual_batch(
        TabICLGeneratorInputs(
            factuals=np.array([[0.0, 0.0]], dtype=float),
            targets=np.array([1], dtype=int),
            numerical_columns=(0, 1),
            categorical_groups=(),
        ),
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.5,),
            cf_mode="data_plausible",
            max_extra_actions=1,
            min_joint_log_gain=0.0,
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=_TwoFeatureSampler(),
            joint_scorer=scorer,
        ),
    )

    np.testing.assert_allclose(result.counterfactuals, [expected])
    assert result.diagnostics.accepted_refinement_count_per_point == (accepted,)
    assert result.diagnostics.refinement_stopping_reason_per_point == (reason,)


def test_public_generator_never_pads_diverse_results_with_invalid_rows() -> None:
    result = generate_counterfactual_batch(
        TabICLGeneratorInputs(
            factuals=np.array([[0.0]], dtype=float),
            targets=np.array([1], dtype=int),
            numerical_columns=(0,),
            categorical_groups=(),
        ),
        discriminator=_SingleFeatureDisc(),
        config=TabICLGeneratorConfig(
            candidate_quantiles=(0.5,),
            diversity_config=DiverseBeamSearchConfig(
                n_counterfactuals=3,
                beam_width=3,
                candidate_pool_size=3,
            ),
        ),
        point_backend_factory=lambda factual, target: TabICLGeneratorPointBackend(
            sampler=_SingleFeatureSampler()
        ),
    )

    assert result.diagnostics.diverse_available_count_per_point[0] == 1
    np.testing.assert_allclose(result.counterfactual_sets[0, 0], [1.0])
    assert np.isnan(result.counterfactual_sets[0, 1:]).all()


def test_exp8_adapter_smoke_uses_stable_generator_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = SimpleNamespace(
        X_train=np.array([[0.0], [1.0]], dtype=float),
        y_train=np.array([0, 1], dtype=int),
        X_test=np.array([[0.0], [0.2]], dtype=float),
        y_test=np.array([0, 1], dtype=int),
        X_val=None,
        y_val=None,
        split_variant="train_test",
        preprocessing_variant="baseline",
        n_dropped_rows=0,
    )

    class _Disc:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            values = np.clip(0.1 + 0.6 * np.asarray(X, dtype=float)[:, 0], 0.0, 0.99)
            return np.column_stack([1.0 - values, values])

    sampler = _SingleFeatureSampler()

    monkeypatch.setattr(
        "experiments.zeroshot_cf.data.load_dataset",
        lambda dataset_name, **kwargs: bundle,
    )
    monkeypatch.setattr(
        "experiments.zeroshot_cf.data.get_grouped_categorical_action_space",
        lambda loaded_bundle: ([0], [], []),
    )
    monkeypatch.setattr(
        "experiments.zeroshot_cf.data.get_one_hot_groups",
        lambda loaded_bundle: [],
    )
    monkeypatch.setattr(
        "experiments.zeroshot_cf.discriminator.train_discriminator",
        lambda *args, **kwargs: _Disc(),
    )
    monkeypatch.setattr(
        exp8,
        "_build_point_backend_factory",
        lambda *args, **kwargs: (
            lambda factual, target: TabICLGeneratorPointBackend(
                sampler=sampler,
                metadata={"tabicl_kv_cache": False},
            )
        ),
    )

    X_test, y_test, X_cf, info = exp8.generate_tabicl_counterfactuals(
        "heloc",
        candidate_quantiles=(0.5,),
        max_test=1,
    )

    np.testing.assert_allclose(X_test, [[0.0]])
    np.testing.assert_allclose(X_cf, [[1.0]])
    np.testing.assert_array_equal(y_test, [0])
    assert info["plausibility_backend"] == "proposal_support"
    assert "diverse_available_count_per_point" in info
    assert info["diverse_available_count_per_point"][0] == 1


def test_exp8_help_stays_runnable() -> None:
    with pytest.raises(SystemExit) as excinfo:
        exp8.main(["--help"])
    assert excinfo.value.code == 0
