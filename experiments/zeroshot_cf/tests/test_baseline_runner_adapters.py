"""One-factual retained-runner checks through the new method lifecycle."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.benchmark_protocol import BenchmarkDatasetContext
from experiments.zeroshot_cf.core.contracts import (
    BenchmarkCase,
    DatasetProvenance,
    FactualSelection,
    FeatureDomains,
    FeatureSchema,
    PreparedDataset,
)
from experiments.zeroshot_cf.data import DatasetBundle

FIXTURE = (
    Path(__file__).parent / "fixtures" / "architecture_v1" / "compatibility.json"
)
COMPATIBILITY = json.loads(FIXTURE.read_text())

# The first four methods must reduce to the single categorical action: changing
# kind_a to kind_b raises p(target) from 0.05 to 0.70, so the scalar edit is
# unnecessary. FACE selects observed training row 3 and therefore also copies
# amount=0.3. These expectations follow directly from _Classifier and the fixed
# action space rather than from an adapter-produced result.
EXPECTED_CANDIDATES = {
    "nice": np.array([[0.0, 0.0, 1.0, 0.25]]),
    "wachter": np.array([[0.0, 0.0, 1.0, 0.25]]),
    "growing_spheres": np.array([[0.0, 0.0, 1.0, 0.25]]),
    "dice": np.array([[0.0, 0.0, 1.0, 0.25]]),
    "face": np.array([[0.3, 0.0, 1.0, 0.25]]),
}
EXPECTED_COMMON = {
    "nice": (0.5, 1.0, 1 / 3, 0.0, np.sqrt(2.0), 2),
    "wachter": (0.5, 1.0, 1 / 3, 0.0, np.sqrt(2.0), 2),
    "growing_spheres": (0.5, 1.0, 1 / 3, 0.0, np.sqrt(2.0), 2),
    "dice": (0.5, 1.0, 1 / 3, 0.0, np.sqrt(2.0), 2),
    "face": (0.75, 2.0, 13 / 30, 0.3, np.sqrt(2.09), 3),
}
EXPECTED_POINTS = {
    "nice": {"target_probability": 0.70, "changed_columns": 2},
    "wachter": {"target_probability": 0.70, "changed_columns": 2},
    "growing_spheres": {"target_probability": 0.70, "changed_columns": 2},
    "dice": {"target_probability": 0.70, "changed_columns": 2},
    "face": {"target_probability": 0.805, "changed_columns": 3},
}


class _Classifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        matrix = np.asarray(X)
        target = np.clip(0.05 + 0.35 * matrix[:, 0] + 0.65 * matrix[:, 2], 0, 1)
        return np.column_stack([1 - target, target])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _runner_context() -> BenchmarkDatasetContext:
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
    X_train = np.array(
        [
            [0.0, 1.0, 0.0, 0.2],
            [0.2, 1.0, 0.0, 0.4],
            [0.4, 1.0, 0.0, 0.6],
            [0.3, 0.0, 1.0, 0.2],
            [0.7, 0.0, 1.0, 0.5],
            [1.0, 0.0, 1.0, 0.8],
        ]
    )
    y_train = _Classifier().predict(X_train)
    X_validation = np.array([[0.1, 1.0, 0.0, 0.3]])
    y_validation = np.array([0])
    X_test = np.array([[0.0, 1.0, 0.0, 0.25]])
    y_test = np.array([0])
    prepared = PreparedDataset(
        name="fixture",
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        schema=schema,
        provenance=DatasetProvenance(
            provider="fixture",
            source_revision="v1",
            source_hashes={"fixture": "tracked"},
            preprocessing_id="identity",
            split_id="fixed",
            fingerprint="baseline-runner-fixture",
        ),
    )
    oracle = _Classifier()
    case = BenchmarkCase(
        case_id="baseline-runner-case",
        dataset=prepared,
        factuals=FactualSelection(np.array([0]), X_test, y_test),
        oracle=oracle,
        factual_predictions=np.array([0]),
        targets=np.array([1]),
        protocol={"target_policy": "fixture"},
    )
    method_dataset = SimpleNamespace(
        inverse_transform=lambda X: X,
        file_dataset=SimpleNamespace(one_hot_feature_groups={}),
        actionable_features=list(schema.names),
    )
    bundle = DatasetBundle(
        name="fixture",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(schema.names),
        numerical_features_indices=[0, 3],
        categorical_features_indices=[1, 2],
        method_dataset=method_dataset,
        X_val=X_validation,
        y_val=y_validation,
        split_variant="fixture_split",
        preprocessing_variant="identity",
        prepared=prepared,
    )
    return BenchmarkDatasetContext(
        dataset_name="fixture",
        bundle=bundle,
        X_test=X_test,
        y_test=y_test,
        disc_model=oracle,
        y_pred=np.array([0]),
        y_target=np.array([1]),
        scalar_actionable=(0,),
        grouped_actionable=(group,),
        immutable_idx=(3,),
        categorical_groups=(group,),
        benchmark_case=case,
    )


def _fake_dice(monkeypatch):
    class _Interface:
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
        SimpleNamespace(Data=_Interface, Model=_Interface, Dice=_Dice),
    )
    from experiments.zeroshot_cf.methods import dice as dice_module

    def fake_generate(*args, **kwargs):
        factuals = np.asarray(args[3])
        candidate = factuals.copy()
        candidate[:, 0] = 1.0
        candidate[:, 1:3] = [0.0, 1.0]
        return candidate, [
            {
                "returned": True,
                "found": True,
                "valid_candidates": 1,
                "attempts": 1,
                "runtime_s": 0.0,
            }
        ]

    monkeypatch.setattr(dice_module, "generate_dice_counterfactuals", fake_generate)


@pytest.mark.parametrize(
    ("module_name", "method_name", "stem", "kwargs"),
    [
        ("exp11_nice_nun_baseline", "nice", "exp11_nice_nun", {}),
        (
            "exp12_optimization_baselines",
            "wachter",
            "exp12_wachter",
            {"method": "wachter"},
        ),
        (
            "exp12_optimization_baselines",
            "growing_spheres",
            "exp12_growing_spheres",
            {"method": "growing_spheres", "sphere_candidates": 64},
        ),
        (
            "exp13_dice_baseline",
            "dice",
            "exp13_dice_genetic",
            {},
        ),
        (
            "exp14_face_baseline",
            "face",
            "exp14_face_knn",
            {"n_neighbors": 3},
        ),
    ],
)
def test_retained_one_factual_runners_preserve_v1_inputs(
    tmp_path,
    monkeypatch,
    module_name,
    method_name,
    stem,
    kwargs,
):
    module = __import__(
        f"experiments.zeroshot_cf.{module_name}", fromlist=["run_dataset"]
    )
    context = _runner_context()
    monkeypatch.setattr(module, "prepare_benchmark_context", lambda *a, **k: context)
    if method_name == "dice":
        _fake_dice(monkeypatch)

    row = module.run_dataset("fixture", results_dir=tmp_path, **kwargs)
    metrics_path = tmp_path / f"{stem}_fixture_metrics.csv"
    points_path = tmp_path / f"{stem}_fixture_points.csv"
    arrays_path = tmp_path / f"{stem}_fixture_arrays.npz"
    assert metrics_path.is_file() and points_path.is_file() and arrays_path.is_file()
    assert all(
        name in row
        for name in ("prepare_s", "generate_s", "evaluate_s", "write_s", "total_s")
    )
    required_summary = set(COMPATIBILITY["common_summary_columns"])
    assert required_summary <= set(row)
    assert row["coverage"] == 1.0
    assert row["validity"] == 1.0
    assert row["actionability"] == 1.0
    assert row["true_actionability"] == 1.0
    assert row["failure_rate"] == 0.0
    (
        sparsity,
        action_sparsity,
        grouped_gower,
        continuous_distance,
        all_feature_l2,
        changed_columns,
    ) = EXPECTED_COMMON[method_name]
    assert row["sparsity"] == pytest.approx(sparsity)
    assert row["action_unit_sparsity_mean"] == pytest.approx(action_sparsity)
    assert row["proximity_grouped_gower"] == pytest.approx(grouped_gower)
    assert row["proximity_continuous_manhattan"] == pytest.approx(
        continuous_distance
    )
    assert row["proximity_continuous_euclidean"] == pytest.approx(
        continuous_distance
    )
    assert row["proximity_all_features_euclidean"] == pytest.approx(all_feature_l2)
    assert row["l0_count_mean"] == pytest.approx(changed_columns)

    with metrics_path.open(newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert len(persisted) == 1
    assert required_summary <= set(persisted[0])
    assert all(name in persisted[0] for name in ("write_s", "total_s"))
    assert float(persisted[0]["sparsity"]) == pytest.approx(sparsity)
    assert float(persisted[0]["proximity_grouped_gower"]) == pytest.approx(
        grouped_gower
    )
    with points_path.open(newline="") as handle:
        point_rows = list(csv.DictReader(handle))
    required_points = set(COMPATIBILITY["common_point_columns"])
    assert len(point_rows) == 1 and required_points <= set(point_rows[0])
    point = point_rows[0]
    assert point["point"] == "0"
    assert point["factual_label"] == "0"
    assert point["factual_prediction"] == "0"
    assert point["target"] == "1"
    assert point["cf_prediction"] == "1"
    assert point["valid"] == "True"
    assert float(point["target_probability"]) == pytest.approx(
        EXPECTED_POINTS[method_name]["target_probability"]
    )
    assert int(point["changed_columns"]) == EXPECTED_POINTS[method_name][
        "changed_columns"
    ]

    expected_keys = set(
        COMPATIBILITY["methods"][method_name]["required_npz_keys"]
    )
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == expected_keys
        np.testing.assert_array_equal(
            arrays["X_cf"], EXPECTED_CANDIDATES[method_name]
        )
        np.testing.assert_array_equal(arrays["X_test"], context.X_test)
        np.testing.assert_array_equal(arrays["y_target"], [1])
        np.testing.assert_array_equal(arrays["y_cf_pred"], [1])
