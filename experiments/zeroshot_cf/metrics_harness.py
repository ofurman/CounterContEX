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

Standard-CF additions (all computed on the same arrays, additive — existing keys
are unchanged):
  - proximity_l1      : mean per-instance L1 (Manhattan) on valid CFs
  - l0_count_mean     : mean number of features changed by more than `change_tol`
  - l0_count_valid    : same, restricted to valid CFs
  - sparsity_tol      : tolerance-based fraction of features changed
  - n_valid / n_total : sample sizes behind the valid-only metrics
  - proximity_l2_continuous / cat_change_rate / n_categorical_features:
        for datasets with one-hot categorical columns, the continuous L2 part and
        the fraction of *decoded* categorical features whose category flipped.

Why the tolerance matters: `sparsity` (and cel's `actionability`) use exact float
equality. Counterfactuals produced by continuous generation differ from the factual
in essentially every cell, so those two metrics saturate (1.0 and 0.0) and carry no
information. `l0_count_mean` / `sparsity_tol` are the meaningful sparsity readouts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def _onehot_groups(
    feature_names: Optional[List[str]], categorical_idx: List[int]
) -> Dict[str, List[int]]:
    """Group one-hot columns back to their source feature.

    cel's one-hot encoder emits ``<feature>__<value>`` column names, so the prefix
    before ``__`` recovers the original categorical feature (e.g. ``race__1.0`` …
    ``race__8.0`` → ``race``). Columns without the separator form their own group.
    """
    groups: Dict[str, List[int]] = {}
    for i in categorical_idx:
        name = feature_names[i] if feature_names is not None else str(i)
        base = name.split("__")[0]
        groups.setdefault(base, []).append(i)
    return groups


def compute_metrics(
    disc_model,
    X_cf: np.ndarray,
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_test: np.ndarray,
    y_target: np.ndarray,
    immutable_idx: Optional[List[int]] = None,
    lof_n_neighbors: int = 20,
    X_cf_lof: Optional[np.ndarray] = None,
    change_tol: float = 1e-3,
    categorical_idx: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
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

    # ---- Standard CF-evaluation additions -------------------------------------
    # Exact `!=` is meaningless for continuously generated CFs (every cell differs
    # by some epsilon, so `sparsity` saturates at 1.0). The tolerance-based
    # variants below are the informative sparsity measures.
    changed_tol = np.abs(X_cf - X_test) > change_tol
    l0_count_mean = float(changed_tol.sum(axis=1).mean())
    sparsity_tol = float(changed_tol.mean())

    if n_valid > 0:
        vdiff = X_cf[valid_mask] - X_test[valid_mask]
        proximity_l1 = float(np.abs(vdiff).sum(axis=1).mean())
        l0_count_valid = float(changed_tol[valid_mask].sum(axis=1).mean())
    else:
        proximity_l1 = float("nan")
        l0_count_valid = float("nan")

    out = {
        "validity": validity,
        "lof_scores_cf": lof_scores_cf,
        "sparsity": sparsity,
        "actionability": actionability,
        "true_actionability": true_actionability,
        "proximity_l2_jaccard": proximity_l2_jaccard,
        "proximity_l1": proximity_l1,
        "l0_count_mean": l0_count_mean,
        "l0_count_valid": l0_count_valid,
        "sparsity_tol": sparsity_tol,
        "n_valid": float(n_valid),
        "n_total": float(len(X_cf)),
    }

    # ---- Categorical-aware distances (datasets with one-hot columns) ----------
    # Pure L2 over one-hot columns is not the standard CF distance; split the
    # continuous part (L2) from the categorical part (fraction of *decoded*
    # categorical features whose category changed).
    cat_idx = list(categorical_idx) if categorical_idx else []
    if cat_idx:
        cont_idx = [i for i in range(X_cf.shape[1]) if i not in set(cat_idx)]
        groups = _onehot_groups(feature_names, cat_idx)
        if n_valid > 0:
            if cont_idx:
                out["proximity_l2_continuous"] = float(
                    np.linalg.norm(
                        X_cf[np.ix_(valid_mask, cont_idx)]
                        - X_test[np.ix_(valid_mask, cont_idx)],
                        axis=1,
                    ).mean()
                )
            flips = [
                np.argmax(X_cf[np.ix_(valid_mask, cols)], axis=1)
                != np.argmax(X_test[np.ix_(valid_mask, cols)], axis=1)
                for cols in groups.values()
            ]
            out["cat_change_rate"] = float(np.mean(np.stack(flips, axis=1)))
        else:
            out["proximity_l2_continuous"] = float("nan")
            out["cat_change_rate"] = float("nan")
        out["n_categorical_features"] = float(len(groups))

    return out


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Pretty-print a metrics dict."""
    pad = f"[{prefix}] " if prefix else ""
    print(f"{pad}Metrics:")
    for k, v in metrics.items():
        print(f"  {k:30s} {v:.4f}")
