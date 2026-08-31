"""Synthetic provenance tests for the portable data and case contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.core.contracts import (
    BenchmarkCase,
    DatasetProvenance,
    FactualSelection,
    FeatureDomains,
    FeatureSchema,
    GenerationRequest,
    PreparedDataset,
)
from experiments.zeroshot_cf.core.validation import target_probabilities
from experiments.zeroshot_cf.datasets.benchmark import (
    build_benchmark_case,
    method_context,
    select_factual_indices,
)
from experiments.zeroshot_cf.datasets.cel import _dataset_fingerprint


class _ReversedLabelPredictor:
    classes_ = np.array([7, 2])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(X[:, 0] >= 0.5, 7, 2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probability_for_seven = 0.2 + 0.6 * X[:, 0]
        return np.column_stack([probability_for_seven, 1 - probability_for_seven])


class _FittedPredictor(_ReversedLabelPredictor):
    def __init__(self, *, scale: float, coefficient: float) -> None:
        self.scale = scale
        self.coef_ = np.array([[coefficient]], dtype=np.float64)

    def get_params(self, deep: bool = True) -> dict[str, float]:
        del deep
        return {"scale": self.scale}


def _dataset() -> PreparedDataset:
    domains = FeatureDomains(
        lower=np.zeros(3),
        upper=np.ones(3),
        discrete={1: np.array([0.0, 1.0]), 2: np.array([0.0, 1.0])},
    )
    group = OneHotActionGroup("kind", (1, 2))
    schema = FeatureSchema(
        names=("amount", "kind_a", "kind_b"),
        numerical=(0,),
        categorical_groups=(group,),
        actionable_scalars=(0,),
        actionable_groups=(group,),
        immutable=(),
        domains=domains,
    )
    return PreparedDataset(
        name="synthetic",
        X_train=np.array([[0.0, 1, 0], [1.0, 0, 1], [0.3, 1, 0], [0.7, 0, 1]]),
        y_train=np.array([2, 7, 2, 7]),
        X_validation=np.array([[0.2, 1, 0], [0.8, 0, 1]]),
        y_validation=np.array([2, 7]),
        X_test=np.array([[0.1, 1, 0], [0.9, 0, 1], [0.2, 1, 0], [0.8, 0, 1]]),
        y_test=np.array([2, 7, 2, 7]),
        schema=schema,
        provenance=DatasetProvenance(
            provider="fixture",
            source_revision="v1",
            source_hashes={"rows": "abc"},
            preprocessing_id="identity",
            split_id="fixed",
            fingerprint="fixture-fingerprint",
        ),
    )


def test_portable_dataset_owns_read_only_arrays_and_has_no_cel_handle() -> None:
    dataset = _dataset()
    assert not hasattr(dataset, "method_dataset")
    assert all(
        not array.flags.writeable
        for array in (
            dataset.X_train,
            dataset.y_train,
            dataset.X_validation,
            dataset.y_validation,
            dataset.X_test,
            dataset.y_test,
            dataset.schema.domains.lower,
            dataset.schema.domains.discrete[1],
        )
    )
    with pytest.raises(ValueError):
        dataset.X_train[0, 0] = 5
    with pytest.raises(TypeError):
        dataset.provenance.source_hashes["rows"] = "changed"
    with pytest.raises(FrozenInstanceError):
        dataset.name = "changed"


def test_schema_rejects_invalid_type_and_action_partitions() -> None:
    domains = FeatureDomains(np.zeros(3), np.ones(3), {})
    group = OneHotActionGroup("kind", (1, 2))
    with pytest.raises(ValueError, match="disjoint"):
        FeatureSchema(
            names=("a", "b", "c"),
            numerical=(0, 1),
            categorical_groups=(group,),
            actionable_scalars=(0,),
            actionable_groups=(group,),
            immutable=(),
            domains=domains,
        )
    with pytest.raises(ValueError, match="partition"):
        FeatureSchema(
            names=("a", "b", "c"),
            numerical=(0,),
            categorical_groups=(group,),
            actionable_scalars=(0,),
            actionable_groups=(),
            immutable=(1,),
            domains=domains,
        )


def test_factual_selection_keeps_unique_source_indices_and_is_deterministic() -> None:
    labels = np.array([2, 7] * 10)
    first = select_factual_indices(labels, 8, seed=42)
    second = select_factual_indices(labels, 8, seed=42)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == len(first)
    assert not first.flags.writeable


def test_case_targets_and_probability_columns_follow_predictor_classes() -> None:
    dataset = _dataset()
    oracle = _ReversedLabelPredictor()
    case = build_benchmark_case(
        dataset,
        oracle,
        max_test=4,
        target_model={"kind": "fixture", "revision": "one"},
    )
    np.testing.assert_array_equal(case.factuals.indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(case.factual_predictions, [2, 7, 2, 7])
    np.testing.assert_array_equal(case.targets, [7, 2, 7, 2])
    np.testing.assert_allclose(
        target_probabilities(case.oracle, case.factuals.values, case.targets),
        [0.26, 0.26, 0.32, 0.32],
    )
    assert (
        case.case_id
        == build_benchmark_case(
            dataset,
            oracle,
            max_test=4,
            target_model={"kind": "fixture", "revision": "one"},
        ).case_id
    )
    assert (
        case.case_id
        != build_benchmark_case(
            dataset,
            oracle,
            max_test=4,
            target_model={"kind": "fixture", "revision": "two"},
        ).case_id
    )
    with pytest.raises(TypeError):
        case.protocol["target_model"]["revision"] = "changed"
    context = method_context(case)
    assert not hasattr(context, "true_labels")
    assert not hasattr(context, "targets")


def test_case_requires_factuals_to_bind_exactly_to_source_test_rows() -> None:
    dataset = _dataset()
    oracle = _ReversedLabelPredictor()
    case = build_benchmark_case(
        dataset,
        oracle,
        max_test=4,
        target_model={"kind": "fixture", "revision": "one"},
    )

    mismatched_values = case.factuals.values.copy()
    mismatched_values[0, 0] += 0.01
    with pytest.raises(ValueError, match="values must exactly match"):
        replace(
            case,
            factuals=FactualSelection(
                case.factuals.indices,
                mismatched_values,
                case.factuals.true_labels,
            ),
        )

    mismatched_labels = case.factuals.true_labels.copy()
    mismatched_labels[0] = 7
    with pytest.raises(ValueError, match="true labels must exactly match"):
        replace(
            case,
            factuals=FactualSelection(
                case.factuals.indices,
                case.factuals.values,
                mismatched_labels,
            ),
        )

    with pytest.raises(ValueError, match="non-negative and unique"):
        FactualSelection(
            indices=np.array([0, 0]),
            values=dataset.X_test[[0, 0]],
            true_labels=dataset.y_test[[0, 0]],
        )

    with pytest.raises(ValueError, match="must refer to dataset test rows"):
        replace(
            case,
            factuals=FactualSelection(
                indices=np.array([len(dataset.X_test)]),
                values=dataset.X_test[[0]],
                true_labels=dataset.y_test[[0]],
            ),
            factual_predictions=np.array([2]),
            targets=np.array([7]),
        )

    X_with_nan = dataset.X_test.copy()
    X_with_nan[0, 0] = np.nan
    nan_dataset = replace(dataset, X_test=X_with_nan)
    BenchmarkCase(
        case_id="nan-aware-binding",
        dataset=nan_dataset,
        factuals=FactualSelection(
            indices=np.array([0]),
            values=np.array([[np.nan, 1.0, 0.0]]),
            true_labels=np.array([2]),
        ),
        oracle=oracle,
        factual_predictions=np.array([2]),
        targets=np.array([7]),
        protocol={"target_policy": "opposite_classifier_prediction"},
    )


def test_case_identity_covers_selection_model_config_and_fitted_content() -> None:
    dataset = _dataset()

    def build(
        model: _FittedPredictor,
        *,
        max_test: int | None = 4,
        selection: str = "stratified",
        seed: int = 42,
    ) -> BenchmarkCase:
        return build_benchmark_case(
            dataset,
            model,
            max_test=max_test,
            test_selection=selection,
            seed=seed,
            target_model={"kind": "fitted-fixture"},
        )

    baseline = build(_FittedPredictor(scale=1.0, coefficient=0.25))
    assert baseline.protocol["target_policy"] == "opposite_classifier_prediction"
    assert (
        baseline.case_id != build(_FittedPredictor(scale=2.0, coefficient=0.25)).case_id
    )
    assert (
        baseline.case_id != build(_FittedPredictor(scale=1.0, coefficient=0.75)).case_id
    )
    assert (
        baseline.case_id
        != build(_FittedPredictor(scale=1.0, coefficient=0.25), max_test=None).case_id
    )
    assert (
        baseline.case_id
        != build(
            _FittedPredictor(scale=1.0, coefficient=0.25), selection="first"
        ).case_id
    )
    assert (
        baseline.case_id
        != build(_FittedPredictor(scale=1.0, coefficient=0.25), seed=7).case_id
    )
    with pytest.raises(ValueError, match="non-empty resolved model config"):
        build_benchmark_case(dataset, _ReversedLabelPredictor(), target_model={})


def test_dataset_fingerprint_is_sensitive_to_every_provenance_layer() -> None:
    dataset = _dataset()
    arrays = {
        "X_train": dataset.X_train,
        "y_train": dataset.y_train,
        "X_validation": dataset.X_validation,
        "y_validation": dataset.y_validation,
        "X_test": dataset.X_test,
        "y_test": dataset.y_test,
    }
    arguments = {
        "name": dataset.name,
        "source_revision": "v1",
        "source_hashes": {
            "config": "config-hash",
            "raw_data": "data-hash",
            "actionability": "actionability-hash",
        },
        "preprocessing_id": "minmax_train_only:original",
        "split_id": "train_val_test:seed=42",
        "split_seed": 42,
        "arrays": arrays,
        "schema": dataset.schema,
    }
    baseline = _dataset_fingerprint(**arguments)

    changed_arrays = dict(arrays)
    changed_arrays["X_test"] = dataset.X_test.copy()
    changed_arrays["X_test"][0, 0] += 0.01

    group = dataset.schema.categorical_groups[0]
    actionability_schema = replace(
        dataset.schema,
        actionable_scalars=(),
        actionable_groups=(group,),
        immutable=(0,),
    )
    renamed_schema = replace(
        dataset.schema,
        names=("renamed_amount", "kind_a", "kind_b"),
    )
    changed_domains = FeatureDomains(
        lower=np.array([-0.1, 0.0, 0.0]),
        upper=dataset.schema.domains.upper,
        discrete=dataset.schema.domains.discrete,
    )
    domain_schema = replace(dataset.schema, domains=changed_domains)

    mutations = (
        {"arrays": changed_arrays},
        {"schema": renamed_schema},
        {"schema": actionability_schema},
        {"schema": domain_schema},
        {"source_hashes": {**arguments["source_hashes"], "actionability": "changed"}},
        {"source_revision": "v2"},
        {"preprocessing_id": "standardize:original"},
        {"split_id": "train_val_test:seed=7"},
        {"split_seed": 7},
    )
    for mutation in mutations:
        assert _dataset_fingerprint(**{**arguments, **mutation}) != baseline


def test_generation_request_is_immutable_and_shape_checked() -> None:
    factuals = np.array([[0.1, 1.0, 0.0]])
    request = GenerationRequest(factuals, np.array([7]), n_counterfactuals=2, seed=42)
    factuals[0, 0] = 0.9
    assert request.factuals[0, 0] == pytest.approx(0.1)
    assert not request.factuals.flags.writeable
    with pytest.raises(ValueError, match="equal row counts"):
        GenerationRequest(np.ones((2, 3)), np.array([7]), 1, 42)
