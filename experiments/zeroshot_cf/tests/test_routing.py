"""Stage 8 tests for classifier/regressor routing overrides."""

from __future__ import annotations

from experiments.zeroshot_cf.data import load_dataset
from experiments.zeroshot_cf.exp4_greedy_cf import parse_force_numeric_cols
from experiments.zeroshot_cf.sampler import ConditionalDensitySampler


def _fit_sampler(models, bundle, force_numeric_cols=None):
    clf, reg = models
    sampler = ConditionalDensitySampler(
        clf=clf,
        reg=reg,
        append_target=True,
        n_permutations=1,
        temperature=1e-9,
        random_state=11,
        categorical_features_indices=bundle.categorical_features_indices,
        force_numeric_cols=force_numeric_cols,
    )
    sampler.set_context(
        bundle.X_train,
        y_context=bundle.y_train,
        target_class=None,
        max_context=256,
    )
    return sampler


def _first_classifier_routed_col(sampler, n_original_features: int) -> int:
    for j in range(n_original_features):
        if sampler.model.use_classifier_(j, sampler.model.X_[:, j]):
            return j
    raise AssertionError("HELOC test precondition broken: no classifier-routed column")


def test_force_numeric_flips_heloc_lowcard_column_to_regressor(models):
    bundle = load_dataset("heloc")
    baseline = _fit_sampler(models, bundle)
    col = _first_classifier_routed_col(baseline, bundle.X_train.shape[1])

    baseline_dist = baseline.predictive_distribution(
        bundle.X_test[:2],
        target_col=col,
        fixed_target=1,
    )
    assert set(baseline_dist) == {"proba", "classes"}

    forced = _fit_sampler(models, bundle, force_numeric_cols=[col])
    assert not forced.model.use_classifier_(col, forced.model.X_[:, col])

    forced_dist = forced.predictive_distribution(
        bundle.X_test[:2],
        target_col=col,
        fixed_target=1,
    )
    assert set(forced_dist) == {"logits", "criterion"}


def test_force_numeric_none_preserves_current_routing(models):
    bundle = load_dataset("heloc")
    baseline = _fit_sampler(models, bundle, force_numeric_cols=None)
    explicit_none = _fit_sampler(models, bundle, force_numeric_cols=[])

    baseline_routing = [
        baseline.model.use_classifier_(j, baseline.model.X_[:, j])
        for j in range(bundle.X_train.shape[1])
    ]
    explicit_none_routing = [
        explicit_none.model.use_classifier_(j, explicit_none.model.X_[:, j])
        for j in range(bundle.X_train.shape[1])
    ]

    assert explicit_none_routing == baseline_routing


def test_parse_force_numeric_cols_names_indices_and_all():
    bundle = load_dataset("heloc")
    assert parse_force_numeric_cols("none", bundle) == []
    assert parse_force_numeric_cols(None, bundle) == []
    assert parse_force_numeric_cols("0", bundle) == [0]
    assert parse_force_numeric_cols(bundle.feature_names[1], bundle) == [1]
    assert parse_force_numeric_cols(f"0,{bundle.feature_names[1]},0", bundle) == [0, 1]
    assert parse_force_numeric_cols("all", bundle) == list(range(len(bundle.feature_names)))


def test_force_numeric_keeps_nonforced_auto_categoricals(models):
    bundle = load_dataset("heloc")
    baseline = _fit_sampler(models, bundle)
    classifier_cols = [
        j
        for j in range(bundle.X_train.shape[1])
        if baseline.model.use_classifier_(j, baseline.model.X_[:, j])
    ]
    if len(classifier_cols) < 2:
        return

    forced = _fit_sampler(models, bundle, force_numeric_cols=[classifier_cols[0]])
    assert not forced.model.use_classifier_(
        classifier_cols[0], forced.model.X_[:, classifier_cols[0]]
    )
    assert forced.model.use_classifier_(
        classifier_cols[1], forced.model.X_[:, classifier_cols[1]]
    )
