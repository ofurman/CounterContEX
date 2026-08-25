"""Metrics harness for the zero-shot CF experiment.

Computes the 5-metric evaluation subset defined in the plan, plus our own
`true_actionability` metric (immutable columns unchanged).

We compute metrics directly rather than routing through MetricsOrchestrator
because (a) the orchestrator unconditionally calls gen_model.eval(), (b) the
registered proximity_l2_jaccard metric computes 0*NaN for empty categorical
features. Direct computation is cleaner.

Metrics computed:
  - validity          : fraction where disc_model(X_cf) == y_target
                        (CF lands on the intended target class)
  - lof_scores_cf     : mean (-LOF score) of X_cf vs training distribution
  - sparsity          : mean fraction of features changed
  - actionability     : cel's metric — fraction of CFs identical to factuals
                        (mislabeled in cel; measures "no change", not constraint compliance)
  - true_actionability: fraction of CFs where immutable columns are exactly preserved
  - proximity_l2_jaccard: mean per-instance L2 distance on *valid* CFs
                           (pure L2 for all-continuous datasets per plan note)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.mixed_distance import (
    action_unit_change_count,
    grouped_gower_distance,
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def compute_metrics(
    disc_model: Any,
    X_cf: np.ndarray,
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_test: np.ndarray,
    y_target: np.ndarray,
    immutable_idx: Optional[List[int]] = None,
    lof_n_neighbors: int = 20,
    X_cf_lof: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute the 6-metric evaluation suite for a set of counterfactuals.

    Args:
        disc_model: Validity oracle with `.predict(X) -> np.ndarray`.
        X_cf: Counterfactual instances, shape (n, d). Used for validity, sparsity,
              proximity, and true_actionability. Should be the post-clipping array.
        X_test: Factual (original) instances, shape (n, d).
        X_train: Training set for LOF fitting, shape (m, d).
        y_test: True labels of the factual instances, shape (n,).
        y_target: Generation target class per instance, shape (n,). Validity is
                  defined as disc_model(X_cf) == y_target (CF lands on target class),
                  NOT as != y_test — these differ on misclassified factuals.
        immutable_idx: Column indices that must be unchanged. None or [] means
                       all features are actionable (true_actionability trivially = 1.0).
        lof_n_neighbors: Number of neighbours for LocalOutlierFactor.
        X_cf_lof: Optional unclipped X_cf used *only* for LOF computation. When OOB
                  rows have been clipped to [0,1] corners the LOF distances degenerate;
                  passing the unclipped array here preserves the true geometry. Defaults
                  to X_cf when not provided.

    Returns:
        Dict with keys: validity, lof_scores_cf, sparsity, actionability,
        true_actionability, proximity_l2_jaccard.
    """
    y_cf_pred = disc_model.predict(X_cf)
    if not isinstance(y_cf_pred, np.ndarray):
        y_cf_pred = np.array(y_cf_pred)
    y_target_arr = np.asarray(y_target).squeeze()

    # validity: fraction whose predicted label matches the intended target class
    valid_mask = y_cf_pred == y_target_arr
    validity = float(valid_mask.mean())

    # sparsity: mean fraction of feature values that changed
    sparsity = float((X_test != X_cf).mean())

    # actionability (cel mislabeled metric): fraction of CFs identical to factuals
    actionability = float(np.all(X_test == X_cf, axis=1).mean())

    # lof_scores_cf: mean negative LOF score (lower = more plausible)
    # Use X_cf_lof (unclipped) if provided — avoids degenerate LOF when many rows
    # are clipped to [0,1] corners, which collapses inter-point distances.
    X_for_lof = X_cf_lof if X_cf_lof is not None else X_cf
    lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, novelty=True)
    lof.fit(X_train)
    lof_scores_cf = float((-lof.score_samples(X_for_lof)).mean())

    # proximity_l2_jaccard: mean per-instance L2 on valid CFs
    # For all-continuous datasets this is pure Euclidean (no categorical part).
    n_valid = int(valid_mask.sum())
    if n_valid > 0:
        diffs = np.linalg.norm(X_cf[valid_mask] - X_test[valid_mask], axis=1)
        proximity_l2_jaccard = float(diffs.mean())
    else:
        proximity_l2_jaccard = float("nan")

    # true_actionability: immutable columns must be exactly unchanged
    if immutable_idx:
        immut = np.asarray(immutable_idx)
        preserved = np.all(X_cf[:, immut] == X_test[:, immut], axis=1)
        true_actionability = float(preserved.mean())
    else:
        true_actionability = 1.0  # no immutable features → trivially satisfied

    return {
        "validity": validity,
        "lof_scores_cf": lof_scores_cf,
        "sparsity": sparsity,
        "actionability": actionability,
        "true_actionability": true_actionability,
        "proximity_l2_jaccard": proximity_l2_jaccard,
    }


def compute_dicoflex_common_metrics(
    disc_model: Any,
    X_cf: np.ndarray,
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_target: np.ndarray,
    numerical_idx: List[int],
    immutable_idx: Optional[List[int]] = None,
    *,
    categorical_groups: Sequence[OneHotActionGroup] = (),
    sparsity_eps: float = 0.05,
    lof_n_neighbors: int = 20,
    isolation_forest_estimators: int = 100,
) -> Dict[str, float]:
    """Compute the method-independent metrics reported by DiCoFlex.

    DiCoFlex also reports generator likelihood metrics. Those are deliberately
    omitted because they are model-specific and TabICL does not expose a
    comparable joint counterfactual log density. Distances are evaluated only
    on valid counterfactuals, matching ``CFMetrics.feature_distance``. In
    addition to DiCoFlex's continuous-only distances, the returned grouped
    Gower metric assigns one contribution to each original categorical group.
    """
    X_cf = np.asarray(X_cf, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    target = np.asarray(y_target).reshape(-1)
    if X_cf.shape != X_test.shape:
        raise ValueError("X_cf and X_test must have the same shape")
    if X_cf.ndim != 2 or X_train.ndim != 2:
        raise ValueError("X_cf, X_test, and X_train must be 2D")
    if X_cf.shape[1] != X_train.shape[1]:
        raise ValueError("X_cf and X_train must have the same columns")
    if target.shape != (len(X_cf),):
        raise ValueError("y_target must contain one label per counterfactual")
    if not 0 <= sparsity_eps:
        raise ValueError("sparsity_eps must be non-negative")

    covered = ~np.isnan(X_cf).any(axis=1)
    coverage = float(covered.mean())
    if not covered.all():
        raise ValueError(
            "DiCoFlex common metrics require complete counterfactuals; "
            f"coverage was {coverage:.4f}"
        )

    y_cf_pred = np.asarray(disc_model.predict(X_cf)).reshape(-1)
    valid = y_cf_pred == target
    validity = float(valid.mean())
    sparsity = float((np.abs(X_test - X_cf) > sparsity_eps).mean())

    immutable = np.asarray(immutable_idx or [], dtype=int)
    actionability = (
        1.0
        if len(immutable) == 0
        else float(np.all(X_test[:, immutable] == X_cf[:, immutable], axis=1).mean())
    )

    numerical = [int(column) for column in numerical_idx]
    mixed_gower = grouped_gower_distance(
        X_cf,
        X_test,
        numerical,
        categorical_groups,
    )
    action_counts = action_unit_change_count(
        X_cf,
        X_test,
        numerical,
        categorical_groups,
        numerical_tolerance=sparsity_eps,
    )
    if valid.any() and len(numerical) > 0:
        continuous_diff = X_cf[valid][:, numerical] - X_test[valid][:, numerical]
        proximity_manhattan = float(np.abs(continuous_diff).sum(axis=1).mean())
        proximity_euclidean = float(
            np.linalg.norm(continuous_diff, axis=1).mean()
        )
    else:
        proximity_manhattan = float("nan")
        proximity_euclidean = float("nan")

    lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, novelty=True).fit(X_train)
    lof_scores_cf = float((-lof.score_samples(X_cf)).mean())
    lof_scores_test = float((-lof.score_samples(X_test)).mean())

    isolation_forest = IsolationForest(
        n_estimators=isolation_forest_estimators,
        random_state=42,
    ).fit(X_train)
    isolation_forest_scores_cf = float(
        isolation_forest.decision_function(X_cf).mean()
    )
    isolation_forest_scores_test = float(
        isolation_forest.decision_function(X_test).mean()
    )

    return {
        "coverage": coverage,
        "validity": validity,
        "actionability": actionability,
        "sparsity": sparsity,
        "action_unit_sparsity_mean": float(action_counts.mean()),
        "proximity_grouped_gower": (
            float(mixed_gower[valid].mean()) if valid.any() else float("nan")
        ),
        "proximity_continuous_manhattan": proximity_manhattan,
        "proximity_continuous_euclidean": proximity_euclidean,
        "lof_scores_cf": lof_scores_cf,
        "lof_scores_test": lof_scores_test,
        "isolation_forest_scores_cf": isolation_forest_scores_cf,
        "isolation_forest_scores_test": isolation_forest_scores_test,
    }


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Pretty-print a metrics dict."""
    pad = f"[{prefix}] " if prefix else ""
    print(f"{pad}Metrics:")
    for k, v in metrics.items():
        print(f"  {k:30s} {v:.4f}")
