"""Tests for the Exp6 context-ablation driver (Stage 4, Step 4).

Two layers:
  - Pure / cheap assertions on the grid bookkeeping (strategy→spec mapping,
    selector→strategy skip rule, CSV column shape) — no model needed.
  - One end-to-end driver run on MOONS with a TINY ``--max-test`` and reduced
    sizes (monkeypatched) that asserts the emitted row count, that
    ``effective_size <= size`` and ``effective_size <= pool_size``, and that the
    ``class_scope`` / ``selection`` columns match the strategy name. Uses the
    shared ``models`` fixture from tests/conftest.py only indirectly (the driver
    loads its own models, but the fixture pins the offline checkpoints are
    importable before we attempt the slow run).
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from experiments.zeroshot_cf import exp6_context_ablation as exp6

# ---------------------------------------------------------------------------
# (a) grid bookkeeping — no model required
# ---------------------------------------------------------------------------

def test_strategy_spec_matches_names():
    """Each strategy maps to the (class_scope, selection) its name encodes."""
    assert exp6.STRATEGY_SPEC["random_target"] == ("target", "random")
    assert exp6.STRATEGY_SPEC["random_both"] == ("both", "random")
    assert exp6.STRATEGY_SPEC["knn_target"] == ("target", "knn")
    assert exp6.STRATEGY_SPEC["knn_both"] == ("both", "knn")
    assert exp6.SIZES == [256, 512, 1024, 2048]


def test_prob_ascent_runs_all_strategies():
    """prob_ascent is compatible with all four strategies → 16 cells."""
    strategies = exp6._strategies_for_selector("prob_ascent")
    assert set(strategies) == set(exp6.STRATEGIES)
    assert len(exp6.SIZES) * len(strategies) == 16


def test_class_divergence_skips_target_strategies():
    """class_divergence needs a both-classes pool → *_target cells skipped, 8 left."""
    strategies = exp6._strategies_for_selector("class_divergence")
    assert all(exp6.STRATEGY_SPEC[s][0] == "both" for s in strategies)
    assert set(strategies) == {"random_both", "knn_both"}
    assert len(exp6.SIZES) * len(strategies) == 8


def test_parse_strategies_supports_subset():
    """Strategy subset parsing preserves order and removes duplicates."""
    strategies = exp6._parse_strategies("knn_both,random_target,knn_both", "prob_ascent")
    assert strategies == ["knn_both", "random_target"]


def test_parse_strategies_rejects_class_divergence_target_scope():
    """class_divergence still cannot run target-only strategy shards."""
    with pytest.raises(ValueError, match="incompatible"):
        exp6._parse_strategies("random_target", "class_divergence")


def test_csv_columns_cover_metric_spec():
    """The CSV column list includes every grids.md metric + run identifier."""
    required = {
        "selector", "size", "effective_size", "strategy", "class_scope",
        "selection", "n_test", "validity", "l0_count_mean", "l0_count_median",
        "l0_count_max", "steps_mean", "steps_median", "steps_max",
        "failure_rate", "lof_scores_cf", "sparsity", "true_actionability",
        "proximity_l2_jaccard", "frac_oob", "runtime_s",
    }
    assert required <= set(exp6.CSV_COLUMNS)


# ---------------------------------------------------------------------------
# (b) end-to-end driver run on MOONS — row count, column shape, invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "selector,expected_rows",
    [("prob_ascent", 16), ("class_divergence", 8)],
)
def test_driver_emits_expected_rows(models, monkeypatch, tmp_path, selector, expected_rows):
    """Run the full grid on MOONS with tiny max-test + reduced sizes and assert
    the row count, the effective_size invariants, and column/strategy agreement.

    Sizes are reduced (not the shipped [256,512,1024,2048]) purely to keep the
    grid cheap; the row-count and column-shape assertions are the point and are
    independent of the actual size values.
    """
    # Reduce sizes for speed; keep four distinct levels so the grid shape holds.
    monkeypatch.setattr(exp6, "SIZES", [8, 16, 32, 64])
    # Redirect outputs to a temp dir so we don't clobber real results.
    monkeypatch.setattr(exp6, "RESULTS_DIR", tmp_path)

    rows = exp6.run_dataset_ablation(
        "moons",
        selector=selector,
        n_permutations=2,
        max_test=2,
    )

    assert len(rows) == expected_rows

    # CSV was written with exactly the declared columns, in order.
    csv_path = tmp_path / "exp6_context_moons.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == exp6.CSV_COLUMNS
        csv_rows = list(reader)
    assert len(csv_rows) == expected_rows

    pool_total = None  # both-scope pool = full train set; derived from rows below
    for r in rows:
        size = int(r["size"])
        eff = int(r["effective_size"])
        # effective_size <= size (capped) and positive.
        assert 0 < eff <= size, f"effective_size {eff} not in (0, {size}]"
        # class_scope / selection columns agree with the strategy name.
        scope, sel = exp6.STRATEGY_SPEC[r["strategy"]]
        assert r["class_scope"] == scope
        assert r["selection"] == sel
        # true_actionability is 1.0 by construction (immutables preserved).
        assert float(r["true_actionability"]) == pytest.approx(1.0)

    # effective_size <= pool_size: for MOONS the both-scope pool is the whole
    # train set and target-scope a single class — in all cases effective_size
    # must not exceed the relevant pool. We re-derive the pools from the bundle.
    from experiments.zeroshot_cf.data import load_dataset
    bundle = load_dataset("moons")
    y_train = bundle.y_train
    pool_both = int(len(y_train))
    pool_by_class = {int(c): int((y_train == c).sum()) for c in np.unique(y_train)}
    min_target_pool = min(pool_by_class.values())
    for r in rows:
        eff = int(r["effective_size"])
        if r["class_scope"] == "both":
            assert eff <= pool_both
        else:
            # target scope: effective_size for some class; bounded by the
            # smallest class pool is too strict, so bound by the largest.
            assert eff <= max(pool_by_class.values())
            # and never exceeds the per-cell size cap (already checked) —
            # additionally it must be <= at least one real class pool.
            assert eff <= max(min_target_pool, max(pool_by_class.values()))
