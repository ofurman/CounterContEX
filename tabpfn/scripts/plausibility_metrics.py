"""
Plausibility Metrics for Counterfactual Evaluation
====================================================
Measures whether generated counterfactuals are realistic beyond
just being close (proximity) and label-flipped (validity).

Metrics:
- Distributional fidelity: KS test, Jensen-Shannon divergence
- Correlation structure preservation: Frobenius norm of correlation diff
- Manifold consistency: Local Outlier Factor
- Mahalanobis distance from factual distribution
- Actionability / sparsity analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import LocalOutlierFactor


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DistributionalFidelity:
    """Per-feature KS test statistics and JS divergence."""

    ks_stats: List[float] = field(default_factory=list)  # per-feature KS statistic
    ks_pvalues: List[float] = field(default_factory=list)  # per-feature KS p-value
    ks_mean: float = 0.0
    ks_max: float = 0.0
    js_divergences: List[float] = field(default_factory=list)  # per-feature JS div
    js_mean: float = 0.0
    js_max: float = 0.0
    num_features: int = 0


@dataclass
class CorrelationPreservation:
    """Measures how well inter-feature correlations are preserved."""

    frobenius_norm: float = 0.0  # ||Corr(x_f) - Corr(x_cf)||_F
    max_abs_diff: float = 0.0  # max element-wise |diff|
    mean_abs_diff: float = 0.0  # mean element-wise |diff|
    num_features: int = 0


@dataclass
class ManifoldConsistency:
    """LOF-based manifold consistency check."""

    outlier_fraction: float = 0.0  # fraction of CFs flagged as outliers
    mean_lof_factual: float = 0.0  # mean LOF score for factual points
    mean_lof_counterfactual: float = 0.0  # mean LOF score for CF points
    num_factual: int = 0
    num_counterfactual: int = 0


@dataclass
class MahalanobisMetrics:
    """Mahalanobis distance of CFs from factual distribution."""

    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    std: float = 0.0
    num_points: int = 0


@dataclass
class ActionabilityMetrics:
    """Analysis of how many features are changed per counterfactual."""

    mean_features_changed: float = 0.0
    std_features_changed: float = 0.0
    frac_1_feature: float = 0.0  # fraction with exactly 1 feature changed
    frac_2_features: float = 0.0  # fraction with exactly 2 features changed
    frac_3plus_features: float = 0.0  # fraction with 3+ features changed
    # Conditional validity: validity rate stratified by # features changed
    validity_by_num_changed: Dict[int, float] = field(default_factory=dict)
    count_by_num_changed: Dict[int, int] = field(default_factory=dict)
    num_pairs: int = 0


@dataclass
class PlausibilityReport:
    """Aggregated plausibility metrics."""

    distributional: DistributionalFidelity = field(
        default_factory=DistributionalFidelity
    )
    correlation: CorrelationPreservation = field(
        default_factory=CorrelationPreservation
    )
    manifold: ManifoldConsistency = field(default_factory=ManifoldConsistency)
    mahalanobis: MahalanobisMetrics = field(default_factory=MahalanobisMetrics)
    actionability: ActionabilityMetrics = field(default_factory=ActionabilityMetrics)


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------


def compute_distributional_fidelity(
    x_factual: np.ndarray,
    x_counterfactual: np.ndarray,
    num_bins: int = 50,
) -> DistributionalFidelity:
    """Compare marginal distributions of factual vs counterfactual features.

    Args:
        x_factual: (N, num_features) factual data
        x_counterfactual: (N, num_features) counterfactual data
        num_bins: number of bins for JS divergence histogram
    """
    num_features = x_factual.shape[1]
    ks_stats = []
    ks_pvalues = []
    js_divs = []

    for f in range(num_features):
        feat_f = x_factual[:, f]
        feat_cf = x_counterfactual[:, f]

        # KS test
        stat, pval = stats.ks_2samp(feat_f, feat_cf)
        ks_stats.append(float(stat))
        ks_pvalues.append(float(pval))

        # JS divergence via histograms
        combined_min = min(feat_f.min(), feat_cf.min())
        combined_max = max(feat_f.max(), feat_cf.max())
        bins = np.linspace(combined_min, combined_max, num_bins + 1)

        hist_f, _ = np.histogram(feat_f, bins=bins, density=True)
        hist_cf, _ = np.histogram(feat_cf, bins=bins, density=True)

        # Add small epsilon to avoid zero bins
        eps = 1e-10
        hist_f = hist_f + eps
        hist_cf = hist_cf + eps
        # Normalize to proper distributions
        hist_f = hist_f / hist_f.sum()
        hist_cf = hist_cf / hist_cf.sum()

        js_div = float(jensenshannon(hist_f, hist_cf) ** 2)  # squared = divergence
        js_divs.append(js_div)

    return DistributionalFidelity(
        ks_stats=ks_stats,
        ks_pvalues=ks_pvalues,
        ks_mean=float(np.mean(ks_stats)),
        ks_max=float(np.max(ks_stats)),
        js_divergences=js_divs,
        js_mean=float(np.mean(js_divs)),
        js_max=float(np.max(js_divs)),
        num_features=num_features,
    )


def compute_correlation_preservation(
    x_factual: np.ndarray,
    x_counterfactual: np.ndarray,
) -> CorrelationPreservation:
    """Measure whether inter-feature correlations are maintained.

    Args:
        x_factual: (N, num_features) factual data
        x_counterfactual: (N, num_features) counterfactual data
    """
    num_features = x_factual.shape[1]

    if num_features < 2:
        return CorrelationPreservation(num_features=num_features)

    corr_f = np.corrcoef(x_factual, rowvar=False)
    corr_cf = np.corrcoef(x_counterfactual, rowvar=False)

    # Replace NaN with 0 (can happen with constant features)
    corr_f = np.nan_to_num(corr_f, nan=0.0)
    corr_cf = np.nan_to_num(corr_cf, nan=0.0)

    diff = corr_f - corr_cf
    frobenius = float(np.linalg.norm(diff, "fro"))

    # Off-diagonal elements only
    mask = ~np.eye(num_features, dtype=bool)
    off_diag_abs = np.abs(diff[mask])

    return CorrelationPreservation(
        frobenius_norm=frobenius,
        max_abs_diff=float(off_diag_abs.max()) if off_diag_abs.size > 0 else 0.0,
        mean_abs_diff=float(off_diag_abs.mean()) if off_diag_abs.size > 0 else 0.0,
        num_features=num_features,
    )


def compute_manifold_consistency(
    x_factual: np.ndarray,
    x_counterfactual: np.ndarray,
    lof_threshold: float = -1.5,
    n_neighbors: int = 20,
    max_samples: int = 10_000,
) -> ManifoldConsistency:
    """Check if counterfactuals lie on the factual data manifold using LOF.

    Args:
        x_factual: (N, num_features) factual data
        x_counterfactual: (M, num_features) counterfactual data
        lof_threshold: LOF score threshold for outlier detection
        n_neighbors: number of neighbors for LOF
        max_samples: subsample if data exceeds this (LOF is O(n^2))
    """
    n_f = x_factual.shape[0]
    n_cf = x_counterfactual.shape[0]

    # Subsample for computational feasibility
    if n_f > max_samples:
        idx = np.random.choice(n_f, max_samples, replace=False)
        x_factual_sub = x_factual[idx]
    else:
        x_factual_sub = x_factual

    if n_cf > max_samples:
        idx = np.random.choice(n_cf, max_samples, replace=False)
        x_cf_sub = x_counterfactual[idx]
    else:
        x_cf_sub = x_counterfactual

    # Fit LOF on factual data
    lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(x_factual_sub) - 1), novelty=True)
    lof.fit(x_factual_sub)

    # Score factual points (for baseline comparison)
    scores_f = lof.score_samples(x_factual_sub)
    # Score counterfactual points
    scores_cf = lof.score_samples(x_cf_sub)

    outlier_fraction = float((scores_cf < lof_threshold).mean())

    return ManifoldConsistency(
        outlier_fraction=outlier_fraction,
        mean_lof_factual=float(scores_f.mean()),
        mean_lof_counterfactual=float(scores_cf.mean()),
        num_factual=len(x_factual_sub),
        num_counterfactual=len(x_cf_sub),
    )


def compute_mahalanobis(
    x_factual: np.ndarray,
    x_counterfactual: np.ndarray,
) -> MahalanobisMetrics:
    """Compute Mahalanobis distance of CFs from the factual distribution.

    Args:
        x_factual: (N, num_features) factual data
        x_counterfactual: (M, num_features) counterfactual data
    """
    mean_f = x_factual.mean(axis=0)
    cov_f = np.cov(x_factual, rowvar=False)

    # Regularize covariance for numerical stability
    cov_f += np.eye(cov_f.shape[0]) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov_f)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_f)

    diff = x_counterfactual - mean_f
    # Mahalanobis: sqrt(diff @ cov_inv @ diff.T) per point
    mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)
    mahal = np.sqrt(np.maximum(mahal_sq, 0.0))

    return MahalanobisMetrics(
        mean=float(mahal.mean()),
        median=float(np.median(mahal)),
        p95=float(np.percentile(mahal, 95)),
        std=float(mahal.std()),
        num_points=len(mahal),
    )


def compute_actionability(
    x_factual: np.ndarray,
    x_counterfactual: np.ndarray,
    label_flipped: np.ndarray,
    change_threshold: float = 1e-8,
) -> ActionabilityMetrics:
    """Analyze how many features change per counterfactual and conditional validity.

    Args:
        x_factual: (N, num_features) factual data
        x_counterfactual: (N, num_features) counterfactual data
        label_flipped: (N,) boolean array indicating valid (label-flipped) CFs
        change_threshold: absolute difference threshold to count a feature as changed
    """
    n = x_factual.shape[0]
    diff = np.abs(x_counterfactual - x_factual)
    num_changed = (diff > change_threshold).sum(axis=1)

    frac_1 = float((num_changed == 1).mean())
    frac_2 = float((num_changed == 2).mean())
    frac_3plus = float((num_changed >= 3).mean())

    # Conditional validity by number of features changed
    validity_by_num = {}
    count_by_num = {}
    unique_counts = np.unique(num_changed)
    for nc in unique_counts:
        mask = num_changed == nc
        count = int(mask.sum())
        if count > 0:
            validity = float(label_flipped[mask].mean())
            validity_by_num[int(nc)] = validity
            count_by_num[int(nc)] = count

    return ActionabilityMetrics(
        mean_features_changed=float(num_changed.mean()),
        std_features_changed=float(num_changed.std()),
        frac_1_feature=frac_1,
        frac_2_features=frac_2,
        frac_3plus_features=frac_3plus,
        validity_by_num_changed=validity_by_num,
        count_by_num_changed=count_by_num,
        num_pairs=n,
    )


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def compute_plausibility(
    x_factual: torch.Tensor,
    x_counterfactual: torch.Tensor,
    label_flipped: torch.Tensor,
    max_lof_samples: int = 10_000,
) -> PlausibilityReport:
    """Compute all plausibility metrics for a set of factual-counterfactual pairs.

    Args:
        x_factual: (N, num_features) factual feature values
        x_counterfactual: (N, num_features) counterfactual feature values
        label_flipped: (N,) boolean tensor indicating valid (label-flipped) CFs
        max_lof_samples: max samples for LOF computation (performance)

    Returns:
        PlausibilityReport with all metric groups populated
    """
    # Convert to numpy
    x_f_np = x_factual.detach().cpu().numpy().astype(np.float64)
    x_cf_np = x_counterfactual.detach().cpu().numpy().astype(np.float64)
    flipped_np = label_flipped.detach().cpu().numpy().astype(bool)

    distributional = compute_distributional_fidelity(x_f_np, x_cf_np)
    correlation = compute_correlation_preservation(x_f_np, x_cf_np)
    manifold = compute_manifold_consistency(
        x_f_np, x_cf_np, max_samples=max_lof_samples
    )
    mahalanobis = compute_mahalanobis(x_f_np, x_cf_np)
    actionability = compute_actionability(x_f_np, x_cf_np, flipped_np)

    return PlausibilityReport(
        distributional=distributional,
        correlation=correlation,
        manifold=manifold,
        mahalanobis=mahalanobis,
        actionability=actionability,
    )
