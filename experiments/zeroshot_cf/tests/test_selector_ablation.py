"""Unit tests for the Exp5 selector-ablation driver (Stage 2).

These tests exercise the driver's orchestration, CSV writing, per-metric verdict
and downstream-selector choice **without** running the real TabPFN generation
path (which is covered by the Stage-1 greedy tests). The expensive
``generate_counterfactuals`` / ``evaluate_and_report`` calls are replaced with
cheap stubs that honour the real contract (context scope derived from selector,
metrics dict shape), so the test is deterministic and fast.

Covers:
  - the per-dataset CSV has exactly one row per selector, in order;
  - context_scope is ``all_classes`` for class_divergence and ``target_only``
    for prob_ascent (Decision #6 wiring, routed correctly by the driver);
  - every documented CSV column is present;
  - _verdict_for_dataset picks the right per-metric winner;
  - _choose_downstream_selector prefers prob_ascent by default and switches to
    class_divergence only when it clearly wins HELOC plausibility without a
    validity cost.
"""

from __future__ import annotations

import csv

import numpy as np

import experiments.zeroshot_cf.exp5_selector_ablation as exp5
from experiments.zeroshot_cf.exp5_selector_ablation import (
    CSV_COLUMNS,
    _choose_downstream_selector,
    _verdict_for_dataset,
)


def _fake_generate(dataset_name, *, selector, **kwargs):
    """Mimic the real generate_counterfactuals contract cheaply: context scope is
    derived from the selector exactly as exp4 does."""
    context_type = "all_classes" if selector == "class_divergence" else "target_only"
    n = 6
    X_test = np.zeros((n, 2))
    y_test = np.zeros(n)
    X_cf = np.zeros((n, 2))
    info = {"context_type": context_type, "selector": selector, "_n": n}
    return X_test, y_test, X_cf, info


def _fake_evaluate(dataset_name, X_test, y_test, X_cf, info, write_csv=True):
    # Distinct metric values per selector so the verdict logic is exercised.
    if info["selector"] == "prob_ascent":
        return {
            "validity": 0.9, "l0_count_mean": 1.5, "steps_mean": 1.5,
            "steps_median": 1.0, "steps_max": 3.0, "failure_rate": 0.1,
            "lof_scores_cf": 1.2, "true_actionability": 1.0,
            "proximity_l2_jaccard": 0.3, "frac_oob": 0.05,
        }
    return {
        "validity": 0.8, "l0_count_mean": 2.0, "steps_mean": 2.5,
        "steps_median": 2.0, "steps_max": 4.0, "failure_rate": 0.2,
        "lof_scores_cf": 1.0, "true_actionability": 1.0,
        "proximity_l2_jaccard": 0.4, "frac_oob": 0.02,
    }


def test_run_dataset_ablation_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(exp5, "generate_counterfactuals", _fake_generate)
    monkeypatch.setattr(exp5, "evaluate_and_report", _fake_evaluate)
    monkeypatch.setattr(exp5, "RESULTS_DIR", tmp_path)

    rows = exp5.run_dataset_ablation("moons", max_test=6)

    # Two rows, one per selector, in order.
    assert [r["selector"] for r in rows] == ["prob_ascent", "class_divergence"]

    csv_path = tmp_path / "exp5_selector_moons.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 2
    assert [r["selector"] for r in csv_rows] == ["prob_ascent", "class_divergence"]

    # Context-scope wiring (Decision #6), routed to the correct selector row.
    scope = {r["selector"]: r["context_scope"] for r in csv_rows}
    assert scope["prob_ascent"] == "target_only"
    assert scope["class_divergence"] == "all_classes"

    # Every documented column present.
    assert set(csv_rows[0].keys()) == set(CSV_COLUMNS)
    assert csv_rows[0]["n_test"] == "6"


def test_verdict_per_metric_winner():
    rows = [
        {"selector": "prob_ascent", "validity": "0.9", "l0_count_mean": "1.5",
         "steps_mean": "1.5", "frac_oob": "0.05", "lof_scores_cf": "1.2"},
        {"selector": "class_divergence", "validity": "0.8", "l0_count_mean": "2.0",
         "steps_mean": "2.5", "frac_oob": "0.02", "lof_scores_cf": "1.0"},
    ]
    v = _verdict_for_dataset(rows)
    assert v["validity"] == "prob_ascent"        # higher better
    assert v["l0_count_mean"] == "prob_ascent"   # lower better
    assert v["steps_mean"] == "prob_ascent"      # lower better
    assert v["frac_oob"] == "class_divergence"   # lower better
    assert v["lof_scores_cf"] == "class_divergence"  # lower better


def test_choose_downstream_default_prob_ascent():
    # class_divergence loses validity → keep prob_ascent default.
    dataset_rows = {
        "heloc": [
            {"selector": "prob_ascent", "validity": "0.6", "frac_oob": "0.3"},
            {"selector": "class_divergence", "validity": "0.5", "frac_oob": "0.1"},
        ]
    }
    sel, _ = _choose_downstream_selector(dataset_rows)
    assert sel == "prob_ascent"


def test_choose_downstream_switches_to_class_divergence():
    # class_divergence wins plausibility without losing validity → switch.
    dataset_rows = {
        "heloc": [
            {"selector": "prob_ascent", "validity": "0.55", "frac_oob": "0.3"},
            {"selector": "class_divergence", "validity": "0.56", "frac_oob": "0.05"},
        ]
    }
    sel, rationale = _choose_downstream_selector(dataset_rows)
    assert sel == "class_divergence"
    assert "Decision #6" in rationale
