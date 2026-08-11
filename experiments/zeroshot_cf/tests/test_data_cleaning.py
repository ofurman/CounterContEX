#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for explicit HELOC sentinel-row cleaning."""

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset


def test_heloc_all_minus9_rows_are_removed_before_split_and_scaling() -> None:
    """Remove all-sentinel HELOC rows before splitting and MinMax scaling."""
    original = load_dataset("heloc")
    cleaned = load_dataset("heloc", drop_heloc_all_minus9=True)

    original_size = len(original.X_train) + len(original.X_test)
    cleaned_size = len(cleaned.X_train) + len(cleaned.X_test)

    assert cleaned.n_dropped_rows == 588
    assert cleaned_size == original_size - cleaned.n_dropped_rows
    assert cleaned.preprocessing_variant == "drop_heloc_all_minus9"
    assert not np.any(np.all(cleaned.method_dataset.X_train_raw == -9, axis=1))
    assert not np.any(np.all(cleaned.method_dataset.X_test_raw == -9, axis=1))


def test_heloc_cleaning_is_explicit_and_disabled_by_default() -> None:
    """Keep historical dataset behavior unless the cleaning flag is enabled."""
    original = load_dataset("heloc")

    assert original.n_dropped_rows == 0
    assert original.preprocessing_variant == "original"
    assert np.any(np.all(original.method_dataset.X_train_raw == -9, axis=1))


def test_german_credit_uses_continuous_only_action_space() -> None:
    """Keep one-hot groups fixed until grouped categorical edits are supported."""
    bundle = load_dataset("german_credit")
    actionable, immutable = get_actionable_immutable("german_credit", bundle)

    assert actionable == bundle.numerical_features_indices
    assert len(actionable) == 7
    assert len(immutable) == 50
    assert set(actionable).isdisjoint(immutable)
    assert sorted([*actionable, *immutable]) == list(range(len(bundle.feature_names)))
