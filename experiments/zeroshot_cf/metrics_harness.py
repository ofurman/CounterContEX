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
                        (mislabeled in cel; measures "no change", not constraint
                        compliance)
  - true_actionability: fraction of CFs where immutable columns are exactly preserved
  - proximity_l2_jaccard: mean per-instance L2 distance on *valid* CFs
                           (pure L2 for all-continuous datasets per plan note)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from experiments.zeroshot_cf.action_space import OneHotActionGroup
from experiments.zeroshot_cf.evaluation.metrics import compute_legacy_common_metrics
from sklearn.neighbors import LocalOutlierFactor


def compute_metrics(
    disc_model: Any,
    X_cf: np.ndarray,
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_test: np.ndarray,
    y_target: np.ndarray,
    immutable_idx: list[int] | None = None,
    lof_n_neighbors: int = 20,
    X_cf_lof: np.ndarray | None = None,
) -> dict[str, float]:
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
    numerical_idx: list[int],
    immutable_idx: list[int] | None = None,
    *,
    categorical_groups: Sequence[OneHotActionGroup] = (),
    sparsity_eps: float = 0.05,
    lof_n_neighbors: int = 20,
    isolation_forest_estimators: int = 100,
) -> dict[str, float]:
    """Compute the method-independent metrics reported by DiCoFlex.

    DiCoFlex also reports generator likelihood metrics. Those are deliberately
    omitted because they are model-specific and TabICL does not expose a
    comparable joint counterfactual log density. Distances are evaluated only
    on valid counterfactuals, matching ``CFMetrics.feature_distance``. In
    addition to DiCoFlex's continuous-only distances, the returned grouped
    Gower metric assigns one contribution to each original categorical group.
    """
    return compute_legacy_common_metrics(
        disc_model,
        X_cf,
        X_test,
        X_train,
        y_target,
        numerical_idx,
        immutable_idx or (),
        categorical_groups=categorical_groups,
        sparsity_eps=sparsity_eps,
        lof_n_neighbors=lof_n_neighbors,
        isolation_forest_estimators=isolation_forest_estimators,
    )


def print_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    """Pretty-print a metrics dict."""
    pad = f"[{prefix}] " if prefix else ""
    print(f"{pad}Metrics:")
    for k, v in metrics.items():
        print(f"  {k:30s} {v:.4f}")
