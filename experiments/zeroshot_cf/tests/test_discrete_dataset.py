"""Stage 7 tests for the native-categorical discrete dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset
from experiments.zeroshot_cf.sampler import ConditionalDensitySampler


DATASET = "binary_cat"


def test_binary_cat_loads_native_categorical_columns():
    bundle = load_dataset(DATASET)

    assert bundle.X_train.shape[0] > 0
    assert bundle.X_test.shape[0] > 0
    assert bundle.categorical_features_indices == [0, 1, 2]
    assert bundle.numerical_features_indices == []
    assert bundle.feature_names == ["decision_code", "segment_code", "channel_code"]
    assert not any("__" in name or name.endswith("_0") for name in bundle.feature_names)

    for j in bundle.categorical_features_indices:
        train_values = bundle.X_train[:, j]
        assert np.allclose(train_values, train_values.astype(int))
        support = sorted(np.unique(train_values).astype(int).tolist())
        assert support == list(range(len(support)))


def test_binary_cat_generic_actionability_config():
    bundle = load_dataset(DATASET)
    actionable_idx, immutable_idx = get_actionable_immutable(DATASET, bundle)

    assert actionable_idx == [0, 2]
    assert immutable_idx == [1]


def test_binary_cat_sampler_routes_categorical_column(models):
    clf, reg = models
    bundle = load_dataset(DATASET)
    sampler = ConditionalDensitySampler(
        clf=clf,
        reg=reg,
        append_target=True,
        n_permutations=1,
        temperature=1e-9,
        random_state=7,
        categorical_features_indices=bundle.categorical_features_indices,
    )
    sampler.set_context(
        bundle.X_train,
        y_context=bundle.y_train,
        target_class=None,
        max_context=64,
    )

    dist = sampler.predictive_distribution(
        bundle.X_test[:3],
        target_col=0,
        fixed_target=1,
    )
    assert set(dist) == {"proba", "classes"}
    assert dist["proba"].shape[0] == 3
    assert set(np.asarray(dist["classes"]).astype(int).tolist()).issubset({0, 1})

    committed = sampler.sample_feature(
        bundle.X_test[:8],
        target_col=0,
        n_samples=1,
        fixed_target=1,
    )
    assert set(np.asarray(committed).astype(int).tolist()).issubset({0, 1})


def test_exp4_binary_cat_smoke_writes_metrics_csv(monkeypatch):
    from experiments.zeroshot_cf import exp4_greedy_cf as exp4

    monkeypatch.setattr(exp4, "N_ESTIMATORS", 2)
    metrics = exp4.run_dataset(
        DATASET,
        selector="prob_ascent",
        budget=4,
        n_permutations=1,
        max_context=64,
        max_test=4,
    )

    for key in (
        "validity",
        "l0_count_mean",
        "proximity_l2_jaccard",
        "frac_oob",
        "true_actionability",
    ):
        assert key in metrics

    csv_path = Path(exp4.RESULTS_DIR) / f"exp4_greedy_{DATASET}_metrics.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["dataset"] == DATASET
    assert rows[0]["validity"] != ""
