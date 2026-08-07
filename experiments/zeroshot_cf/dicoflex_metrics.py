"""DiCoFlex counterfactual metrics — a numpy-only port of ../CETGFN's
`DiCoFlexCounterfactualMetrics.compute_metrics`
(rgfn/trainer/metrics/counterfactual_metrics.py). Used for the "rest" (not
L2C-discretized) ported datasets: adult_dicoflex, bank, default, gmc,
lending-club, sba.

CETGFN's version works in its own one-hot/ordinal dataset representation
(`dataset.num_ord_transform`, `dataset.transform`, `dataset.predict_proba`).
This port operates directly on the already-ordinal-encoded, MinMax-scaled
[0, 1] float matrices this project's local_data.py produces (categoricals are
ordinal-, not one-hot-, encoded — the same representation CETGFN's
`num_ord_transform` targets), with `num_indices`/`cat_indices` taken from
DatasetBundle.numerical_features_indices/categorical_features_indices, and the
LOF/IsolationForest plausibility scores fit directly on that representation
instead of a separate one-hot block (functionally equivalent — both compare
distances in a consistent, fixed feature space).

Masking follows CETGFN exactly: `dicoflex_validity` and `l2c_diversity_weight_fast`
are computed over ALL attempts (including failed/non-flipping ones); every
other metric is computed only over the subset where the counterfactual's
predicted class actually differs from the original (y_orig != y_cf).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from experiments.zeroshot_cf.l2c_metrics import (
    compute_sparsity_rate_np,
    compute_validity_np,
    harmonic_mean,
    l2c_diversity,
)


def compute_clf_probability(y_prob: np.ndarray) -> float:
    if y_prob.size == 0:
        return 0.0
    return float(np.mean(np.max(y_prob, axis=1)))


def compute_sparsity_changed(X_orig: np.ndarray, X_cf: np.ndarray, columns: List[int]) -> float:
    """Fraction of feature cells CHANGED (DiCoFlex's own convention — the
    opposite of l2c_metrics.compute_sparsity_rate_np, which counts UNCHANGED)."""
    if X_orig.size == 0 or len(columns) == 0:
        return 0.0
    return float((X_orig[:, columns] != X_cf[:, columns]).mean())


def compute_proximity_l1(X_orig: np.ndarray, X_cf: np.ndarray, columns: List[int]) -> float:
    if X_orig.size == 0 or len(columns) == 0:
        return 0.0
    return float(np.abs(X_orig[:, columns] - X_cf[:, columns]).mean())


def compute_proximity_l2(X_orig: np.ndarray, X_cf: np.ndarray, columns: List[int]) -> float:
    if X_orig.size == 0 or len(columns) == 0:
        return 0.0
    diff = X_orig[:, columns] - X_cf[:, columns]
    return float(np.sqrt(np.mean(diff**2)))


def compute_eps_sparsity(
    X_orig: np.ndarray, X_cf: np.ndarray, columns: List[int], thr: float = 0.05
) -> float:
    if X_orig.size == 0 or len(columns) == 0:
        return 0.0
    eps = 1e-8
    diff = np.abs(X_orig[:, columns] - X_cf[:, columns]) / (np.abs(X_orig[:, columns]) + eps)
    return float((diff > thr).mean())


def compute_mad(X: np.ndarray, feature_indices: List[int]) -> np.ndarray:
    if len(feature_indices) == 0:
        return np.array([])
    X_subset = X[:, feature_indices]
    median_per_feature = np.median(X_subset, axis=0)
    return np.median(np.abs(X_subset - median_per_feature), axis=0)


def compute_eps_sparsity_mad(
    X_orig: np.ndarray,
    X_cf: np.ndarray,
    columns: List[int],
    mad_values: np.ndarray,
    mad_threshold: float = 0.1,
) -> float:
    if X_orig.size == 0 or len(columns) == 0:
        return 0.0
    Xo = X_orig[:, columns]
    Xc = X_cf[:, columns]
    mad_subset = np.where(mad_values < 1e-8, 1.0, mad_values)
    diff = np.abs(Xo - Xc)
    return float((diff > mad_threshold * mad_subset).mean())


def compute_l1_mad_np(
    X_orig: np.ndarray, X_cf: np.ndarray, mad_values: np.ndarray, feature_indices: List[int]
) -> float:
    if X_orig.size == 0 or len(feature_indices) == 0 or mad_values.size == 0:
        return 0.0
    Xo = X_orig[:, feature_indices]
    Xc = X_cf[:, feature_indices]
    mad_subset = np.where(mad_values < 1e-8, 1.0, mad_values)
    return float(np.mean(np.abs(Xo - Xc) / mad_subset))


def compute_lof_score(X_train: np.ndarray, X_cf: np.ndarray) -> float:
    if X_cf.size == 0:
        return 0.0
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(X_train)
    return float(np.median(np.log(-lof.score_samples(X_cf) + 1e-8)))


def compute_iso_forest_score(X_train: np.ndarray, X_cf: np.ndarray) -> float:
    if X_cf.size == 0:
        return 0.0
    iso = IsolationForest(random_state=42)
    iso.fit(X_train)
    return float(np.median(iso.decision_function(X_cf)))


def _pdist(X: np.ndarray, metric: str) -> np.ndarray:
    from scipy.spatial.distance import pdist

    return pdist(X, metric=metric)


def compute_pairwise_diversity_mixed(
    X_orig: np.ndarray, X_cf: np.ndarray, indices_num: List[int], indices_cat: List[int]
) -> float:
    if X_orig.size == 0:
        return 0.0
    groups: Dict[bytes, List[np.ndarray]] = {}
    for orig_row, cf_row in zip(X_orig, X_cf):
        groups.setdefault(orig_row.tobytes(), []).append(cf_row.astype(np.float32))

    n_features = len(indices_num) + len(indices_cat)
    group_diversities = []
    for cf_group in groups.values():
        K = len(cf_group)
        if K < 2:
            continue
        X_cf_group = np.vstack(cf_group)
        num_pairs = K * (K - 1) // 2
        d_cont = (
            _pdist(X_cf_group[:, indices_num], "euclidean")
            if indices_num
            else np.zeros(num_pairs)
        )
        d_cat = (
            _pdist(X_cf_group[:, indices_cat], "hamming") * len(indices_cat)
            if indices_cat
            else np.zeros(num_pairs)
        )
        group_diversities.append(np.mean((d_cont + d_cat) / n_features))

    return float(np.mean(group_diversities)) if group_diversities else 0.0


def compute_pairwise_diversity_mixed_l1_mad_hamming(
    X_orig: np.ndarray,
    X_cf: np.ndarray,
    indices_num: List[int],
    indices_cat: List[int],
    mad_values: np.ndarray,
) -> float:
    if X_orig.size == 0:
        return 0.0
    n_num, n_cat = len(indices_num), len(indices_cat)
    groups: Dict[bytes, List[np.ndarray]] = {}
    for orig_row, cf_row in zip(X_orig, X_cf):
        groups.setdefault(orig_row.tobytes(), []).append(cf_row.astype(np.float32))

    mad_safe = np.where(mad_values < 1e-8, 1.0, mad_values) if mad_values.size else mad_values

    group_diversities = []
    for cf_group in groups.values():
        K = len(cf_group)
        if K < 2:
            continue
        X_cf_group = np.vstack(cf_group)
        diversity = 0.0
        if n_num > 0 and mad_values.size > 0:
            X_num = X_cf_group[:, indices_num]
            diff = np.abs(X_num[:, None, :] - X_num[None, :, :]) / mad_safe
            l1_distances = np.sum(diff, axis=2)
            triu = np.triu_indices(K, k=1)
            diversity += np.mean(l1_distances[triu]) / n_num
        if n_cat > 0:
            d_cat = _pdist(X_cf_group[:, indices_cat], "hamming")
            diversity += np.mean(d_cat) / n_cat
        group_diversities.append(diversity)

    return float(np.mean(group_diversities)) if group_diversities else 0.0


def compute_dicoflex_metrics(
    X_orig: np.ndarray,
    X_cf: np.ndarray,
    y_orig: np.ndarray,
    y_cf: np.ndarray,
    y_cf_proba_full: np.ndarray,
    num_indices: List[int],
    cat_indices: List[int],
    X_train: np.ndarray,
) -> Dict[str, float]:
    """Reproduces DiCoFlexCounterfactualMetrics.compute_metrics's keys.

    X_orig/X_cf/y_orig/y_cf/y_cf_proba_full should include every generation
    ATTEMPT (failures as a no-op CF, X_cf == X_orig). `l2c_diversity_weight_fast`
    and `dicoflex_validity` use all attempts; every other metric is computed
    only where y_orig != y_cf, matching CETGFN's own masking order.
    """
    validity = compute_validity_np(y_orig, y_cf)
    l2c_div = l2c_diversity(X_orig, X_cf)  # unmasked, like CETGFN

    mask = y_orig != y_cf
    X_orig_m, X_cf_m = X_orig[mask], X_cf[mask]
    y_cf_proba_m = y_cf_proba_full[mask]

    clf_prob = compute_clf_probability(y_cf_proba_m)
    sparsity_cat = compute_sparsity_changed(X_orig_m, X_cf_m, cat_indices)
    prox_l1 = compute_proximity_l1(X_orig_m, X_cf_m, num_indices)
    prox_l2 = compute_proximity_l2(X_orig_m, X_cf_m, num_indices)

    lof_score = compute_lof_score(X_train, X_cf_m)
    iso_score = compute_iso_forest_score(X_train, X_cf_m)

    pairwise_dist = compute_pairwise_diversity_mixed(X_orig_m, X_cf_m, num_indices, cat_indices)
    eps_sparsity = compute_eps_sparsity(X_orig_m, X_cf_m, num_indices)

    if num_indices:
        mad_values = compute_mad(X_orig_m, num_indices)
        eps_sparsity_mad = compute_eps_sparsity_mad(X_orig_m, X_cf_m, num_indices, mad_values)
        l1_mad = compute_l1_mad_np(X_orig_m, X_cf_m, mad_values, num_indices)
        pairwise_l1_mad_hamming = compute_pairwise_diversity_mixed_l1_mad_hamming(
            X_orig_m, X_cf_m, num_indices, cat_indices, mad_values
        )
    else:
        eps_sparsity_mad = 0.0
        l1_mad = 0.0
        pairwise_l1_mad_hamming = 0.0

    cols = list(range(X_orig.shape[1]))
    sparsity_score = compute_sparsity_rate_np(X_orig_m, X_cf_m)  # l2c-style, over `cols` == all

    hm_spar_div = harmonic_mean(sparsity_score, pairwise_dist)
    hm_spar_div_mad = harmonic_mean(sparsity_score, pairwise_l1_mad_hamming)

    return {
        "dicoflex_validity": 100 * validity,
        "dicoflex_clf_prob": 100 * clf_prob,
        "dicoflex_proximity_l1_num": prox_l1,
        "dicoflex_proximity_l2_num": prox_l2,
        "dicoflex_sparsity_cat": 100 * sparsity_cat,
        "dicoflex_eps_sparsity": 100 * eps_sparsity,
        "dicoflex_eps_sparsity_mad": 100 * eps_sparsity_mad,
        "dicoflex_lof_score": lof_score,
        "l2c_diversity_weight_fast": 100 * l2c_div,
        "l2c_sparsity": 100 * sparsity_score,
        "iso_forest_score": iso_score,
        "dicoflex_pairwise_distance": 100 * pairwise_dist,
        "dicoflex_l1_mad": l1_mad,
        "dicoflex_pairwise_l1_mad_hamming": 100 * pairwise_l1_mad_hamming,
        "hmean_spar_diver": 100 * hm_spar_div,
        "hmean_spar_diver_mad": 100 * hm_spar_div_mad,
    }
