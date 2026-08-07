"""L2C counterfactual metrics — a numpy-only port of ../CETGFN's
`L2CCounterfactualMetrics.compute_metrics` (rgfn/trainer/metrics/counterfactual_metrics.py,
wired to configs/l2c_counterfactual.gin): l2c_validity, l2c_sparsity,
l2c_diversity_weight_fast, l2c_hmean_sparsity_diversity.

Shared by dice_baseline.py and exp2_counterfactuals.py's DiCE/TabPFN CF sets so
both report the same metric definitions without depending on cel or rgfn.

CETGFN's `diversity()` casts feature values to int (its features are GFN
bin/category indices); this port compares scaled float values for exact
equality instead, which is equivalent for finite-valued (e.g. L2C-discretized)
features and is the more general definition.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_validity_np(y_orig: np.ndarray, y_cf: np.ndarray) -> float:
    """Fraction of ALL attempted counterfactuals whose predicted class differs
    from the original — including generation failures (counted as invalid)."""
    if y_orig.size == 0:
        return 0.0
    return float(np.mean(y_orig != y_cf))


def compute_sparsity_rate_np(X_orig: np.ndarray, X_cf: np.ndarray) -> float:
    """Fraction of feature cells left UNCHANGED, over valid CFs only (higher = sparser edit)."""
    if X_orig.size == 0:
        return 0.0
    return float((X_orig == X_cf).mean())


def l2c_diversity(X_orig: np.ndarray, X_cf: np.ndarray) -> float:
    """Pairwise mismatch rate among CFs sharing the same original query point,
    averaged across groups and weighted by each group's pair count — mirrors
    CETGFN's diversity(), generalized from int-cast categorical codes to exact
    float equality. Computed over ALL attempts (valid + invalid), like CETGFN.
    Groups with a single CF (no repeats for that query point) don't contribute."""
    groups: Dict[bytes, List[np.ndarray]] = {}
    for orig, cf in zip(X_orig, X_cf):
        groups.setdefault(orig.tobytes(), []).append(cf)

    total_sum_dist = 0.0
    total_pairs = 0
    for cfs in groups.values():
        K = len(cfs)
        if K < 2:
            continue
        cfs_arr = np.vstack(cfs)
        D = cfs_arr.shape[1]
        num_pairs = K * (K - 1) // 2
        mismatch_total = 0
        for j in range(D):
            _, counts = np.unique(cfs_arr[:, j], return_counts=True)
            equal_pairs = int(np.sum(counts * (counts - 1) // 2))
            mismatch_total += num_pairs - equal_pairs
        avg_group = mismatch_total / (num_pairs * D)
        total_sum_dist += avg_group * num_pairs
        total_pairs += num_pairs

    return float(total_sum_dist / total_pairs) if total_pairs > 0 else 0.0


def harmonic_mean(a: float, b: float) -> float:
    return float((2.0 * a * b) / (a + b)) if (a + b) > 0.0 else 0.0


def compute_l2c_metrics(
    X_orig: np.ndarray, X_cf: np.ndarray, y_orig: np.ndarray, y_cf: np.ndarray
) -> Dict[str, float]:
    """Reproduces L2CCounterfactualMetrics.compute_metrics's four keys.

    X_orig/X_cf/y_orig/y_cf should include every generation ATTEMPT (one row
    per (query point, repeat) pair) — including failures, represented as a
    no-op CF (X_cf == X_orig, so y_cf == y_orig). Validity and diversity are
    computed over all attempts; sparsity (and therefore the harmonic mean) only
    over the subset where y_orig != y_cf, matching CETGFN's own ordering.
    """
    validity = compute_validity_np(y_orig, y_cf)
    diversity = l2c_diversity(X_orig, X_cf)  # unmasked, like CETGFN

    mask = y_orig != y_cf
    sparsity = compute_sparsity_rate_np(X_orig[mask], X_cf[mask])
    hmean = harmonic_mean(sparsity, diversity)

    return {
        "l2c_validity": 100 * validity,
        "l2c_sparsity": 100 * sparsity,
        "l2c_diversity_weight_fast": 100 * diversity,
        "l2c_hmean_sparsity_diversity": 100 * hmean,
    }
