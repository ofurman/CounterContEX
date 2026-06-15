"""Metrics harness for the zero-shot CF experiment.

Computes the 5-metric evaluation subset defined in the plan, plus our own
`true_actionability` metric (immutable columns unchanged).

We compute metrics directly rather than routing through MetricsOrchestrator
because (a) the orchestrator unconditionally calls gen_model.eval(), (b) the
registered proximity_l2_jaccard metric computes 0*NaN for empty categorical
features. Direct computation is cleaner.

Metrics computed:
  - validity          : fraction where disc_model(X_cf) != y_test
  - lof_scores_cf     : mean (-LOF score) of X_cf vs training distribution
  - sparsity          : mean fraction of features changed
  - actionability     : cel's metric — fraction of CFs identical to factuals
                        (mislabeled in cel; measures "no change", not constraint compliance)
  - true_actionability: fraction of CFs where immutable columns are exactly preserved
  - proximity_l2_jaccard: mean per-instance L2 distance on *valid* CFs
                           (pure L2 for all-continuous datasets per plan note)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def compute_metrics(
    disc_model,
    X_cf: np.ndarray,
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_test: np.ndarray,
    immutable_idx: Optional[List[int]] = None,
    lof_n_neighbors: int = 20,
) -> Dict[str, float]:
    """Compute the 6-metric evaluation suite for a set of counterfactuals.

    Args:
        disc_model: Validity oracle with `.predict(X) -> np.ndarray`.
        X_cf: Counterfactual instances, shape (n, d).
        X_test: Factual (original) instances, shape (n, d).
        X_train: Training set for LOF fitting, shape (m, d).
        y_test: True labels of the factual instances, shape (n,).
        immutable_idx: Column indices that must be unchanged. None or [] means
                       all features are actionable (true_actionability trivially = 1.0).
        lof_n_neighbors: Number of neighbours for LocalOutlierFactor.

    Returns:
        Dict with keys: validity, lof_scores_cf, sparsity, actionability,
        true_actionability, proximity_l2_jaccard.
    """
    y_cf_pred = disc_model.predict(X_cf)
    if not isinstance(y_cf_pred, np.ndarray):
        y_cf_pred = np.array(y_cf_pred)
    y_test_arr = np.asarray(y_test).squeeze()

    # validity: fraction whose predicted label changed
    valid_mask = y_cf_pred != y_test_arr
    validity = float(valid_mask.mean())

    # sparsity: mean fraction of feature values that changed
    sparsity = float((X_test != X_cf).mean())

    # actionability (cel mislabeled metric): fraction of CFs identical to factuals
    actionability = float(np.all(X_test == X_cf, axis=1).mean())

    # lof_scores_cf: mean negative LOF score (lower = more plausible)
    lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, novelty=True)
    lof.fit(X_train)
    lof_scores_cf = float((-lof.score_samples(X_cf)).mean())

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


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Pretty-print a metrics dict."""
    pad = f"[{prefix}] " if prefix else ""
    print(f"{pad}Metrics:")
    for k, v in metrics.items():
        print(f"  {k:30s} {v:.4f}")
