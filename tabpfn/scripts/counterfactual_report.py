"""
Counterfactual Generation Report (Expanded)
=============================================
Generates factual-counterfactual pairs using TabPFN's SCM prior across
all perturbation strategies and computes key metrics:

- Dataset Statistics:  feature distributions, correlations, class balance,
                       target distribution, feature ranges, inter-feature
                       dependence structure of the generated SCM data
- Validity:            fraction of pairs where the class label flipped
- Proximity (all):     distance metrics across ALL generated pairs
- Proximity (valid):   distance metrics across VALID (label-flipped) pairs only
- Timing:              wall-clock time per batch and per sample

Reports results per strategy, per configuration variant, and aggregated.
"""

import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, ".")
from tabpfn.priors.counterfactual import (
    CounterfactualSCMGenerator,
    CounterfactualBatch,
    PerturbationStrategy,
    get_default_counterfactual_config,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProximityMetrics:
    l1_mean: float = 0.0
    l1_std: float = 0.0
    l1_median: float = 0.0
    l2_mean: float = 0.0
    l2_std: float = 0.0
    l2_median: float = 0.0
    linf_mean: float = 0.0
    linf_std: float = 0.0
    linf_median: float = 0.0
    cosine_sim_mean: float = 0.0
    cosine_sim_std: float = 0.0
    cosine_sim_median: float = 0.0
    per_feature_mad_mean: float = 0.0
    per_feature_mad_std: float = 0.0
    sparsity_mean: float = 0.0
    sparsity_std: float = 0.0
    num_pairs: int = 0


@dataclass
class DatasetStatistics:
    # Per-feature statistics (averaged across features)
    feat_mean_mean: float = 0.0
    feat_mean_std: float = 0.0
    feat_std_mean: float = 0.0
    feat_std_std: float = 0.0
    feat_min_mean: float = 0.0
    feat_max_mean: float = 0.0
    feat_skew_mean: float = 0.0
    feat_skew_std: float = 0.0
    feat_kurtosis_mean: float = 0.0
    feat_kurtosis_std: float = 0.0
    # Inter-feature correlations
    mean_abs_correlation: float = 0.0
    max_abs_correlation: float = 0.0
    median_abs_correlation: float = 0.0
    # Feature-target correlations
    feat_target_corr_mean: float = 0.0
    feat_target_corr_std: float = 0.0
    feat_target_corr_max: float = 0.0
    # Class balance
    class_0_ratio: float = 0.0
    class_1_ratio: float = 0.0
    num_classes: int = 0
    class_entropy: float = 0.0
    # Target distribution (continuous)
    target_mean: float = 0.0
    target_std: float = 0.0
    target_min: float = 0.0
    target_max: float = 0.0
    target_skew: float = 0.0
    target_kurtosis: float = 0.0
    # Dataset shape
    num_samples: int = 0
    num_features: int = 0
    num_batches: int = 0
    # Feature ranges
    feat_range_mean: float = 0.0
    feat_range_std: float = 0.0
    feat_iqr_mean: float = 0.0
    feat_iqr_std: float = 0.0


@dataclass
class StrategyReport:
    strategy: str
    num_samples: int = 0
    validity_rate: float = 0.0
    num_valid: int = 0
    num_invalid: int = 0
    proximity_all: ProximityMetrics = field(default_factory=ProximityMetrics)
    proximity_valid: ProximityMetrics = field(default_factory=ProximityMetrics)
    dataset_stats: DatasetStatistics = field(default_factory=DatasetStatistics)
    generation_time_s: float = 0.0
    time_per_sample_ms: float = 0.0
    num_features_perturbed_mean: float = 0.0
    num_features_perturbed_std: float = 0.0
    # Valid-only feature perturbation stats
    valid_num_features_perturbed_mean: float = 0.0
    valid_num_features_perturbed_std: float = 0.0


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _skewness(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Compute skewness along a dimension."""
    m = x.mean(dim=dim, keepdim=True)
    s = x.std(dim=dim, keepdim=True).clamp(min=1e-8)
    return ((x - m) / s).pow(3).mean(dim=dim)


def _kurtosis(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Compute excess kurtosis along a dimension."""
    m = x.mean(dim=dim, keepdim=True)
    s = x.std(dim=dim, keepdim=True).clamp(min=1e-8)
    return ((x - m) / s).pow(4).mean(dim=dim) - 3.0


def compute_proximity(
    x_f: torch.Tensor, x_cf: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> ProximityMetrics:
    """Compute proximity metrics between factual and counterfactual points.

    Args:
        x_f:  (N, features) factual points
        x_cf: (N, features) counterfactual points
        mask: optional (N,) boolean mask to select a subset of pairs
    """
    if mask is not None:
        if mask.sum() == 0:
            return ProximityMetrics(num_pairs=0)
        x_f = x_f[mask]
        x_cf = x_cf[mask]

    diff = x_cf - x_f
    n = x_f.shape[0]

    l1 = diff.abs().sum(dim=-1)
    l2 = diff.norm(p=2, dim=-1)
    linf = diff.abs().max(dim=-1).values
    cos_sim = torch.nn.functional.cosine_similarity(x_f, x_cf, dim=-1)
    cos_sim = torch.where(torch.isnan(cos_sim), torch.ones_like(cos_sim), cos_sim)
    per_feat_mad = diff.abs().mean(dim=0)
    changed = (diff.abs() > 1e-8).float().mean(dim=-1)

    return ProximityMetrics(
        l1_mean=l1.mean().item(),
        l1_std=l1.std().item() if n > 1 else 0.0,
        l1_median=l1.median().item(),
        l2_mean=l2.mean().item(),
        l2_std=l2.std().item() if n > 1 else 0.0,
        l2_median=l2.median().item(),
        linf_mean=linf.mean().item(),
        linf_std=linf.std().item() if n > 1 else 0.0,
        linf_median=linf.median().item(),
        cosine_sim_mean=cos_sim.mean().item(),
        cosine_sim_std=cos_sim.std().item() if n > 1 else 0.0,
        cosine_sim_median=cos_sim.median().item(),
        per_feature_mad_mean=per_feat_mad.mean().item(),
        per_feature_mad_std=per_feat_mad.std().item(),
        sparsity_mean=changed.mean().item(),
        sparsity_std=changed.std().item() if n > 1 else 0.0,
        num_pairs=n,
    )


def compute_dataset_statistics(batch: CounterfactualBatch) -> DatasetStatistics:
    """Compute comprehensive statistics on the generated factual dataset."""
    # Flatten: (seq_len, batch, feat) -> (N, feat)
    x = batch.x_factual.reshape(-1, batch.x_factual.shape[-1])
    y_cont = batch.y_factual.reshape(-1)
    y_cls = batch.y_factual_class.reshape(-1)

    n, nf = x.shape

    # --- Per-feature statistics ---
    feat_mean = x.mean(dim=0)
    feat_std = x.std(dim=0)
    feat_min = x.min(dim=0).values
    feat_max = x.max(dim=0).values
    feat_skew = _skewness(x, dim=0)
    feat_kurt = _kurtosis(x, dim=0)
    feat_range = feat_max - feat_min
    q25 = torch.quantile(x, 0.25, dim=0)
    q75 = torch.quantile(x, 0.75, dim=0)
    feat_iqr = q75 - q25

    # --- Inter-feature correlation ---
    # Standardize
    x_std = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp(
        min=1e-8
    )
    corr_matrix = (x_std.T @ x_std) / (n - 1)
    # Mask diagonal
    mask_diag = ~torch.eye(nf, dtype=torch.bool, device=x.device)
    off_diag = corr_matrix[mask_diag].abs()

    # --- Feature-target correlation ---
    y_std = (y_cont - y_cont.mean()) / y_cont.std().clamp(min=1e-8)
    feat_target_corr = (x_std.T @ y_std.unsqueeze(-1)).squeeze(-1) / (n - 1)
    feat_target_corr_abs = feat_target_corr.abs()

    # --- Class balance ---
    unique_classes = torch.unique(y_cls)
    num_classes = len(unique_classes)
    class_counts = torch.stack([(y_cls == c).float().sum() for c in unique_classes])
    class_ratios = class_counts / n
    # Entropy
    log_ratios = torch.log2(class_ratios.clamp(min=1e-10))
    entropy = -(class_ratios * log_ratios).sum().item()

    # --- Target distribution ---
    target_skew = _skewness(y_cont.unsqueeze(-1), dim=0).squeeze()
    target_kurt = _kurtosis(y_cont.unsqueeze(-1), dim=0).squeeze()

    return DatasetStatistics(
        feat_mean_mean=feat_mean.mean().item(),
        feat_mean_std=feat_mean.std().item(),
        feat_std_mean=feat_std.mean().item(),
        feat_std_std=feat_std.std().item(),
        feat_min_mean=feat_min.mean().item(),
        feat_max_mean=feat_max.mean().item(),
        feat_skew_mean=feat_skew.mean().item(),
        feat_skew_std=feat_skew.std().item(),
        feat_kurtosis_mean=feat_kurt.mean().item(),
        feat_kurtosis_std=feat_kurt.std().item(),
        mean_abs_correlation=off_diag.mean().item() if off_diag.numel() > 0 else 0.0,
        max_abs_correlation=off_diag.max().item() if off_diag.numel() > 0 else 0.0,
        median_abs_correlation=off_diag.median().item()
        if off_diag.numel() > 0
        else 0.0,
        feat_target_corr_mean=feat_target_corr_abs.mean().item(),
        feat_target_corr_std=feat_target_corr_abs.std().item(),
        feat_target_corr_max=feat_target_corr_abs.max().item(),
        class_0_ratio=class_ratios[0].item() if num_classes >= 1 else 0.0,
        class_1_ratio=class_ratios[1].item() if num_classes >= 2 else 0.0,
        num_classes=num_classes,
        class_entropy=entropy,
        target_mean=y_cont.mean().item(),
        target_std=y_cont.std().item(),
        target_min=y_cont.min().item(),
        target_max=y_cont.max().item(),
        target_skew=target_skew.item(),
        target_kurtosis=target_kurt.item(),
        num_samples=n,
        num_features=nf,
        num_batches=batch.x_factual.shape[1],
        feat_range_mean=feat_range.mean().item(),
        feat_range_std=feat_range.std().item(),
        feat_iqr_mean=feat_iqr.mean().item(),
        feat_iqr_std=feat_iqr.std().item(),
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

STRATEGIES = [s.value for s in PerturbationStrategy]

CONFIG_VARIANTS = {
    "default": {},
    "high_perturbation": {
        "perturbation_prob": 0.7,
        "perturbation_magnitude": 2.0,
        "fixed_magnitude_k": 3.0,
    },
    "low_perturbation": {
        "perturbation_prob": 0.15,
        "perturbation_magnitude": 0.3,
        "fixed_magnitude_k": 0.5,
    },
    "deep_scm_6layers": {
        "num_layers": 6,
        "prior_mlp_hidden_dim": 128,
    },
    "shallow_scm_2layers": {
        "num_layers": 2,
        "prior_mlp_hidden_dim": 32,
    },
    "many_features_20": {},
}


def run_single_experiment(
    strategy: str,
    config_overrides: Dict,
    batch_size: int = 16,
    seq_len: int = 200,
    num_features: int = 10,
    num_repeats: int = 3,
) -> StrategyReport:
    """Run one experiment and compute all metrics including valid-only."""
    config = get_default_counterfactual_config()
    config["perturbation_strategy"] = strategy
    config.update(config_overrides)

    gen = CounterfactualSCMGenerator(config, device="cpu")

    # Accumulators across repeats
    all_x_f, all_x_cf = [], []
    all_flipped = []
    all_mask = []
    all_batches = []
    all_times = []
    total_samples = 0

    for _ in range(num_repeats):
        t0 = time.perf_counter()
        batch = gen.generate_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
        )
        elapsed = time.perf_counter() - t0
        all_times.append(elapsed)

        n = batch_size * seq_len
        total_samples += n

        all_x_f.append(batch.x_factual.reshape(-1, num_features))
        all_x_cf.append(batch.x_counterfactual.reshape(-1, num_features))
        all_flipped.append(batch.label_flipped.reshape(-1))
        all_mask.append(batch.intervention_mask.reshape(-1, num_features))
        all_batches.append(batch)

    # Concatenate all data
    x_f_all = torch.cat(all_x_f, dim=0)
    x_cf_all = torch.cat(all_x_cf, dim=0)
    flipped_all = torch.cat(all_flipped, dim=0)
    mask_all = torch.cat(all_mask, dim=0)

    num_valid = int(flipped_all.sum().item())
    num_invalid = total_samples - num_valid
    validity_rate = num_valid / total_samples if total_samples > 0 else 0.0

    # --- Proximity: ALL pairs ---
    prox_all = compute_proximity(x_f_all, x_cf_all)

    # --- Proximity: VALID (label-flipped) pairs only ---
    prox_valid = compute_proximity(x_f_all, x_cf_all, mask=flipped_all)

    # --- Dataset statistics (from last batch for representative stats) ---
    ds_stats = compute_dataset_statistics(all_batches[-1])

    # --- Features perturbed: all vs valid ---
    n_perturbed_all = mask_all.float().sum(dim=-1)
    fp_mean = n_perturbed_all.mean().item()
    fp_std = n_perturbed_all.std().item()

    if num_valid > 0:
        n_perturbed_valid = mask_all[flipped_all].float().sum(dim=-1)
        vfp_mean = n_perturbed_valid.mean().item()
        vfp_std = n_perturbed_valid.std().item()
    else:
        vfp_mean, vfp_std = 0.0, 0.0

    total_time = sum(all_times)
    return StrategyReport(
        strategy=strategy,
        num_samples=total_samples,
        validity_rate=validity_rate,
        num_valid=num_valid,
        num_invalid=num_invalid,
        proximity_all=prox_all,
        proximity_valid=prox_valid,
        dataset_stats=ds_stats,
        generation_time_s=total_time,
        time_per_sample_ms=(total_time / total_samples) * 1000,
        num_features_perturbed_mean=fp_mean,
        num_features_perturbed_std=fp_std,
        valid_num_features_perturbed_mean=vfp_mean,
        valid_num_features_perturbed_std=vfp_std,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_table(
    headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None
) -> str:
    if col_widths is None:
        col_widths = [
            max(len(h), max((len(r[i]) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]
    lines = []
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-+-".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(" | ".join(str(r).ljust(w) for r, w in zip(row, col_widths)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    batch_size: int = 16,
    seq_len: int = 200,
    num_features: int = 10,
    num_repeats: int = 3,
) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("COUNTERFACTUAL GENERATION REPORT (EXPANDED)")
    lines.append("TabPFN SCM Prior - Factual/Counterfactual Pair Statistics")
    lines.append("=" * 100)
    lines.append("")
    lines.append(
        f"Configuration: batch_size={batch_size}, seq_len={seq_len}, "
        f"num_features={num_features}, repeats={num_repeats}"
    )
    lines.append(f"Total samples per experiment: {batch_size * seq_len * num_repeats}")
    lines.append("")

    # ===================================================================
    # SECTION 1: Strategy comparison (default config)
    # ===================================================================
    lines.append("-" * 100)
    lines.append("SECTION 1: STRATEGY COMPARISON (default config)")
    lines.append("-" * 100)
    lines.append("")

    strategy_reports: Dict[str, StrategyReport] = {}
    for strategy in STRATEGIES:
        print(f"  Running strategy: {strategy} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        report = run_single_experiment(
            strategy=strategy,
            config_overrides={},
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            num_repeats=num_repeats,
        )
        print(f"done ({time.perf_counter() - t0:.1f}s)")
        strategy_reports[strategy] = report

    # --- 1.1 Validity ---
    lines.append("1.1 Validity (label flip rate)")
    lines.append("")
    headers = ["Strategy", "Validity Rate", "# Valid", "# Invalid", "Total"]
    rows = []
    for s in STRATEGIES:
        r = strategy_reports[s]
        rows.append(
            [
                s,
                f"{r.validity_rate:.4f}",
                str(r.num_valid),
                str(r.num_invalid),
                str(r.num_samples),
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # --- 1.2 Proximity: ALL pairs ---
    lines.append("1.2 Proximity Metrics - ALL pairs (mean +/- std [median])")
    lines.append("")
    headers = ["Strategy", "L1", "L2", "Linf", "Cosine Sim", "Sparsity"]
    rows = []
    for s in STRATEGIES:
        p = strategy_reports[s].proximity_all
        rows.append(
            [
                s,
                f"{p.l1_mean:.3f} +/- {p.l1_std:.3f} [{p.l1_median:.3f}]",
                f"{p.l2_mean:.3f} +/- {p.l2_std:.3f} [{p.l2_median:.3f}]",
                f"{p.linf_mean:.3f} +/- {p.linf_std:.3f} [{p.linf_median:.3f}]",
                f"{p.cosine_sim_mean:.4f} +/- {p.cosine_sim_std:.4f}",
                f"{p.sparsity_mean:.3f} +/- {p.sparsity_std:.3f}",
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # --- 1.3 Proximity: VALID (label-flipped) pairs only ---
    lines.append("1.3 Proximity Metrics - VALID pairs only (label flipped)")
    lines.append(
        "    (computed exclusively on counterfactuals that changed the predicted class)"
    )
    lines.append("")
    headers = ["Strategy", "# Valid", "L1", "L2", "Linf", "Cosine Sim", "Sparsity"]
    rows = []
    for s in STRATEGIES:
        r = strategy_reports[s]
        p = r.proximity_valid
        if p.num_pairs == 0:
            rows.append([s, "0", "N/A", "N/A", "N/A", "N/A", "N/A"])
        else:
            rows.append(
                [
                    s,
                    str(p.num_pairs),
                    f"{p.l1_mean:.3f} +/- {p.l1_std:.3f} [{p.l1_median:.3f}]",
                    f"{p.l2_mean:.3f} +/- {p.l2_std:.3f} [{p.l2_median:.3f}]",
                    f"{p.linf_mean:.3f} +/- {p.linf_std:.3f} [{p.linf_median:.3f}]",
                    f"{p.cosine_sim_mean:.4f} +/- {p.cosine_sim_std:.4f}",
                    f"{p.sparsity_mean:.3f} +/- {p.sparsity_std:.3f}",
                ]
            )
    lines.append(format_table(headers, rows))
    lines.append("")

    # --- 1.4 Comparison: ALL vs VALID proximity ---
    lines.append("1.4 Proximity Comparison: ALL vs VALID (L2 distance)")
    lines.append("")
    headers = [
        "Strategy",
        "L2 (all)",
        "L2 (valid)",
        "Ratio valid/all",
        "Cosine (all)",
        "Cosine (valid)",
    ]
    rows = []
    for s in STRATEGIES:
        r = strategy_reports[s]
        pa = r.proximity_all
        pv = r.proximity_valid
        if pv.num_pairs == 0:
            rows.append(
                [
                    s,
                    f"{pa.l2_mean:.3f}",
                    "N/A",
                    "N/A",
                    f"{pa.cosine_sim_mean:.4f}",
                    "N/A",
                ]
            )
        else:
            ratio = pv.l2_mean / pa.l2_mean if pa.l2_mean > 0 else 0
            rows.append(
                [
                    s,
                    f"{pa.l2_mean:.3f}",
                    f"{pv.l2_mean:.3f}",
                    f"{ratio:.3f}",
                    f"{pa.cosine_sim_mean:.4f}",
                    f"{pv.cosine_sim_mean:.4f}",
                ]
            )
    lines.append(format_table(headers, rows))
    lines.append("")

    # --- 1.5 Per-feature MAD ---
    lines.append("1.5 Per-Feature Mean Absolute Deviation")
    lines.append("")
    headers = [
        "Strategy",
        "MAD (all)",
        "MAD (valid)",
        "# Feat Perturbed (all)",
        "# Feat Perturbed (valid)",
    ]
    rows = []
    for s in STRATEGIES:
        r = strategy_reports[s]
        pv = r.proximity_valid
        mad_v = f"{pv.per_feature_mad_mean:.4f}" if pv.num_pairs > 0 else "N/A"
        vfp = (
            f"{r.valid_num_features_perturbed_mean:.2f} +/- {r.valid_num_features_perturbed_std:.2f}"
            if r.num_valid > 0
            else "N/A"
        )
        rows.append(
            [
                s,
                f"{r.proximity_all.per_feature_mad_mean:.4f}",
                mad_v,
                f"{r.num_features_perturbed_mean:.2f} +/- {r.num_features_perturbed_std:.2f}",
                vfp,
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # --- 1.6 Timing ---
    lines.append("1.6 Timing")
    lines.append("")
    headers = ["Strategy", "Total Time (s)", "Time/Sample (ms)"]
    rows = []
    for s in STRATEGIES:
        r = strategy_reports[s]
        rows.append([s, f"{r.generation_time_s:.3f}", f"{r.time_per_sample_ms:.4f}"])
    lines.append(format_table(headers, rows))
    lines.append("")

    # ===================================================================
    # SECTION 2: Dataset Statistics
    # ===================================================================
    lines.append("-" * 100)
    lines.append("SECTION 2: DATASET STATISTICS (generated factual data)")
    lines.append("-" * 100)
    lines.append("")
    lines.append(
        "Statistics computed on the factual (pre-perturbation) data generated by the SCM prior."
    )
    lines.append(
        "Values shown are from the default config with each perturbation strategy."
    )
    lines.append("")

    # 2.1 Feature distribution summary
    lines.append("2.1 Feature Distribution Summary")
    lines.append("")
    headers = [
        "Strategy",
        "Feat Mean",
        "Feat Std",
        "Feat Range",
        "Feat IQR",
        "Skewness",
        "Kurtosis",
    ]
    rows = []
    for s in STRATEGIES:
        ds = strategy_reports[s].dataset_stats
        rows.append(
            [
                s,
                f"{ds.feat_mean_mean:.3f} +/- {ds.feat_mean_std:.3f}",
                f"{ds.feat_std_mean:.3f} +/- {ds.feat_std_std:.3f}",
                f"{ds.feat_range_mean:.3f} +/- {ds.feat_range_std:.3f}",
                f"{ds.feat_iqr_mean:.3f} +/- {ds.feat_iqr_std:.3f}",
                f"{ds.feat_skew_mean:.3f} +/- {ds.feat_skew_std:.3f}",
                f"{ds.feat_kurtosis_mean:.3f} +/- {ds.feat_kurtosis_std:.3f}",
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # 2.2 Inter-feature correlation
    lines.append("2.2 Inter-Feature Correlation Structure")
    lines.append("")
    headers = ["Strategy", "Mean |corr|", "Median |corr|", "Max |corr|"]
    rows = []
    for s in STRATEGIES:
        ds = strategy_reports[s].dataset_stats
        rows.append(
            [
                s,
                f"{ds.mean_abs_correlation:.4f}",
                f"{ds.median_abs_correlation:.4f}",
                f"{ds.max_abs_correlation:.4f}",
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # 2.3 Feature-target correlation
    lines.append("2.3 Feature-Target Correlation")
    lines.append("")
    headers = ["Strategy", "Mean |corr|", "Std |corr|", "Max |corr|"]
    rows = []
    for s in STRATEGIES:
        ds = strategy_reports[s].dataset_stats
        rows.append(
            [
                s,
                f"{ds.feat_target_corr_mean:.4f}",
                f"{ds.feat_target_corr_std:.4f}",
                f"{ds.feat_target_corr_max:.4f}",
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # 2.4 Class balance and target distribution
    lines.append("2.4 Class Balance & Target Distribution")
    lines.append("")
    headers = [
        "Strategy",
        "# Classes",
        "Class 0 Ratio",
        "Class 1 Ratio",
        "Entropy",
        "Target Mean",
        "Target Std",
        "Target Skew",
    ]
    rows = []
    for s in STRATEGIES:
        ds = strategy_reports[s].dataset_stats
        rows.append(
            [
                s,
                str(ds.num_classes),
                f"{ds.class_0_ratio:.4f}",
                f"{ds.class_1_ratio:.4f}",
                f"{ds.class_entropy:.4f}",
                f"{ds.target_mean:.3f}",
                f"{ds.target_std:.3f}",
                f"{ds.target_skew:.3f}",
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # 2.5 Dataset shape
    lines.append("2.5 Dataset Shape")
    lines.append("")
    ds = strategy_reports[STRATEGIES[0]].dataset_stats
    lines.append(f"  Samples per batch element:  {seq_len}")
    lines.append(f"  Batch elements (SCMs):      {ds.num_batches}")
    lines.append(f"  Features:                   {ds.num_features}")
    lines.append(f"  Total samples (last batch): {ds.num_samples}")
    lines.append("")

    # ===================================================================
    # SECTION 3: Configuration Variant Comparison
    # ===================================================================
    lines.append("-" * 100)
    lines.append(
        "SECTION 3: CONFIGURATION VARIANT COMPARISON (strategy=fixed_magnitude)"
    )
    lines.append("-" * 100)
    lines.append("")

    variant_reports: Dict[str, StrategyReport] = {}
    for variant_name, overrides in CONFIG_VARIANTS.items():
        nf = 20 if variant_name == "many_features_20" else num_features
        print(f"  Running variant: {variant_name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        report = run_single_experiment(
            strategy="fixed_magnitude",
            config_overrides=overrides,
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=nf,
            num_repeats=num_repeats,
        )
        print(f"done ({time.perf_counter() - t0:.1f}s)")
        variant_reports[variant_name] = report

    # 3.1 Overview
    lines.append("3.1 Validity & Proximity Overview")
    lines.append("")
    headers = [
        "Variant",
        "Validity",
        "# Valid",
        "L2 (all)",
        "L2 (valid)",
        "Cosine (all)",
        "Cosine (valid)",
        "Time/Sample (ms)",
    ]
    rows = []
    for vname in CONFIG_VARIANTS:
        r = variant_reports[vname]
        pa = r.proximity_all
        pv = r.proximity_valid
        l2v = f"{pv.l2_mean:.3f}" if pv.num_pairs > 0 else "N/A"
        cosv = f"{pv.cosine_sim_mean:.4f}" if pv.num_pairs > 0 else "N/A"
        rows.append(
            [
                vname,
                f"{r.validity_rate:.4f}",
                str(r.num_valid),
                f"{pa.l2_mean:.3f}",
                l2v,
                f"{pa.cosine_sim_mean:.4f}",
                cosv,
                f"{r.time_per_sample_ms:.4f}",
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # 3.2 Dataset stats across variants
    lines.append("3.2 Dataset Statistics Across Variants")
    lines.append("")
    headers = [
        "Variant",
        "Mean |feat-corr|",
        "Max |feat-corr|",
        "Mean |tgt-corr|",
        "Class Entropy",
        "Feat Std",
    ]
    rows = []
    for vname in CONFIG_VARIANTS:
        ds = variant_reports[vname].dataset_stats
        rows.append(
            [
                vname,
                f"{ds.mean_abs_correlation:.4f}",
                f"{ds.max_abs_correlation:.4f}",
                f"{ds.feat_target_corr_mean:.4f}",
                f"{ds.class_entropy:.4f}",
                f"{ds.feat_std_mean:.3f}",
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # ===================================================================
    # SECTION 4: Aggregate Summary
    # ===================================================================
    lines.append("-" * 100)
    lines.append("SECTION 4: AGGREGATE SUMMARY")
    lines.append("-" * 100)
    lines.append("")

    best_valid = max(strategy_reports.values(), key=lambda r: r.validity_rate)
    best_prox = min(strategy_reports.values(), key=lambda r: r.proximity_all.l2_mean)
    best_cos = max(
        strategy_reports.values(), key=lambda r: r.proximity_all.cosine_sim_mean
    )
    fastest = min(strategy_reports.values(), key=lambda r: r.time_per_sample_ms)
    most_sparse = min(
        strategy_reports.values(), key=lambda r: r.proximity_all.sparsity_mean
    )

    # Best valid-only proximity (among those with valid pairs)
    valid_strategies = {s: r for s, r in strategy_reports.items() if r.num_valid > 0}
    best_valid_prox = (
        min(valid_strategies.values(), key=lambda r: r.proximity_valid.l2_mean)
        if valid_strategies
        else None
    )

    lines.append("  Overall Bests (across strategies, default config):")
    lines.append("")
    lines.append(
        f"    Highest validity rate:       {best_valid.strategy} ({best_valid.validity_rate:.4f}, {best_valid.num_valid} valid pairs)"
    )
    lines.append(
        f"    Lowest L2 (all pairs):       {best_prox.strategy} (L2={best_prox.proximity_all.l2_mean:.4f})"
    )
    if best_valid_prox:
        lines.append(
            f"    Lowest L2 (valid only):      {best_valid_prox.strategy} (L2={best_valid_prox.proximity_valid.l2_mean:.4f}, n={best_valid_prox.proximity_valid.num_pairs})"
        )
    lines.append(
        f"    Highest cosine sim (all):    {best_cos.strategy} ({best_cos.proximity_all.cosine_sim_mean:.4f})"
    )
    lines.append(
        f"    Fastest generation:          {fastest.strategy} ({fastest.time_per_sample_ms:.4f} ms/sample)"
    )
    lines.append(
        f"    Most sparse perturbations:   {most_sparse.strategy} (sparsity={most_sparse.proximity_all.sparsity_mean:.4f})"
    )
    lines.append("")

    # Validity vs Proximity tradeoff
    lines.append("  Validity vs Proximity Tradeoff:")
    lines.append("  " + "-" * 80)
    lines.append(
        f"  {'Strategy':25s} {'Validity':>10s}  {'L2(all)':>10s}  {'L2(valid)':>10s}  {'Cos(valid)':>10s}"
    )
    lines.append("  " + "-" * 80)
    for s in STRATEGIES:
        r = strategy_reports[s]
        pa = r.proximity_all
        pv = r.proximity_valid
        l2v = f"{pv.l2_mean:.3f}" if pv.num_pairs > 0 else "N/A"
        cosv = f"{pv.cosine_sim_mean:.4f}" if pv.num_pairs > 0 else "N/A"
        bar = "#" * int(r.validity_rate * 100)
        lines.append(
            f"  {s:25s} {r.validity_rate:>10.4f}  {pa.l2_mean:>10.3f}  {l2v:>10s}  {cosv:>10s}  |{bar}"
        )
    lines.append("")

    lines.append("=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating expanded counterfactual report...")
    print("")

    report_text = generate_report(
        batch_size=16,
        seq_len=200,
        num_features=10,
        num_repeats=3,
    )

    print("")
    print(report_text)

    output_path = "counterfactual_report.txt"
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {output_path}")
