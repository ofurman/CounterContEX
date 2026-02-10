"""
Counterfactual Generation Report
=================================
Generates factual-counterfactual pairs using TabPFN's SCM prior across
all perturbation strategies and computes key metrics:

- Validity:   fraction of pairs where the class label flipped
- Proximity:  distance between factual and counterfactual points
              (L1, L2, Linf, cosine similarity, per-feature MAD)
- Timing:     wall-clock time per batch and per sample

Reports results per strategy, per configuration variant, and aggregated.
"""

import sys
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

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
# Metric computation
# ---------------------------------------------------------------------------


@dataclass
class ProximityMetrics:
    l1_mean: float = 0.0
    l1_std: float = 0.0
    l2_mean: float = 0.0
    l2_std: float = 0.0
    linf_mean: float = 0.0
    linf_std: float = 0.0
    cosine_sim_mean: float = 0.0
    cosine_sim_std: float = 0.0
    per_feature_mad_mean: float = 0.0  # mean absolute deviation averaged over features
    per_feature_mad_std: float = 0.0
    sparsity_mean: float = 0.0  # fraction of features that actually changed
    sparsity_std: float = 0.0


@dataclass
class StrategyReport:
    strategy: str
    num_samples: int = 0
    validity_rate: float = 0.0  # fraction of label flips
    proximity: ProximityMetrics = field(default_factory=ProximityMetrics)
    generation_time_s: float = 0.0  # total wall-clock seconds
    time_per_sample_ms: float = 0.0  # milliseconds per sample
    num_features_perturbed_mean: float = 0.0
    num_features_perturbed_std: float = 0.0


def compute_proximity(batch: CounterfactualBatch) -> ProximityMetrics:
    """Compute proximity metrics between factual and counterfactual points."""
    # Flatten batch dims: (seq_len, batch, features) -> (N, features)
    x_f = batch.x_factual.reshape(-1, batch.x_factual.shape[-1])
    x_cf = batch.x_counterfactual.reshape(-1, batch.x_counterfactual.shape[-1])
    diff = x_cf - x_f

    # L1 (Manhattan) distance per sample
    l1 = diff.abs().sum(dim=-1)
    # L2 (Euclidean) distance per sample
    l2 = diff.norm(p=2, dim=-1)
    # Linf (Chebyshev) distance per sample
    linf = diff.abs().max(dim=-1).values
    # Cosine similarity per sample
    cos_sim = torch.nn.functional.cosine_similarity(x_f, x_cf, dim=-1)
    # Handle degenerate case where both are zero
    cos_sim = torch.where(torch.isnan(cos_sim), torch.ones_like(cos_sim), cos_sim)
    # Per-feature mean absolute deviation
    per_feat_mad = diff.abs().mean(dim=0)  # (features,)
    # Sparsity: fraction of features that changed per sample
    changed = (diff.abs() > 1e-8).float().mean(dim=-1)

    return ProximityMetrics(
        l1_mean=l1.mean().item(),
        l1_std=l1.std().item(),
        l2_mean=l2.mean().item(),
        l2_std=l2.std().item(),
        linf_mean=linf.mean().item(),
        linf_std=linf.std().item(),
        cosine_sim_mean=cos_sim.mean().item(),
        cosine_sim_std=cos_sim.std().item(),
        per_feature_mad_mean=per_feat_mad.mean().item(),
        per_feature_mad_std=per_feat_mad.std().item(),
        sparsity_mean=changed.mean().item(),
        sparsity_std=changed.std().item(),
    )


def compute_validity(batch: CounterfactualBatch) -> float:
    """Compute validity = fraction of pairs where the class label flipped."""
    return batch.label_flipped.float().mean().item()


def compute_features_perturbed(batch: CounterfactualBatch):
    """Compute mean/std of number of features directly intervened on per sample."""
    n_perturbed = batch.intervention_mask.float().sum(dim=-1).reshape(-1)
    return n_perturbed.mean().item(), n_perturbed.std().item()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

STRATEGIES = [s.value for s in PerturbationStrategy]

# Configuration variants to test
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
    "many_features_20": {
        # Will override num_features in generate_batch
    },
}


def run_single_experiment(
    strategy: str,
    config_overrides: Dict,
    batch_size: int = 16,
    seq_len: int = 200,
    num_features: int = 10,
    num_repeats: int = 3,
) -> StrategyReport:
    """Run one experiment: strategy + config variant, repeated for stability."""
    config = get_default_counterfactual_config()
    config["perturbation_strategy"] = strategy
    config.update(config_overrides)

    gen = CounterfactualSCMGenerator(config, device="cpu")

    all_validity = []
    all_proximity = []
    all_times = []
    all_feat_perturbed = []
    total_samples = 0

    for _ in range(num_repeats):
        t0 = time.perf_counter()
        batch = gen.generate_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
        )
        elapsed = time.perf_counter() - t0

        n = batch_size * seq_len
        total_samples += n
        all_times.append(elapsed)
        all_validity.append(compute_validity(batch))
        all_proximity.append(compute_proximity(batch))
        fp_mean, fp_std = compute_features_perturbed(batch)
        all_feat_perturbed.append((fp_mean, fp_std))

    # Aggregate proximity across repeats
    def avg_field(field_name):
        return float(np.mean([getattr(p, field_name) for p in all_proximity]))

    prox = ProximityMetrics(
        l1_mean=avg_field("l1_mean"),
        l1_std=avg_field("l1_std"),
        l2_mean=avg_field("l2_mean"),
        l2_std=avg_field("l2_std"),
        linf_mean=avg_field("linf_mean"),
        linf_std=avg_field("linf_std"),
        cosine_sim_mean=avg_field("cosine_sim_mean"),
        cosine_sim_std=avg_field("cosine_sim_std"),
        per_feature_mad_mean=avg_field("per_feature_mad_mean"),
        per_feature_mad_std=avg_field("per_feature_mad_std"),
        sparsity_mean=avg_field("sparsity_mean"),
        sparsity_std=avg_field("sparsity_std"),
    )

    total_time = sum(all_times)
    return StrategyReport(
        strategy=strategy,
        num_samples=total_samples,
        validity_rate=float(np.mean(all_validity)),
        proximity=prox,
        generation_time_s=total_time,
        time_per_sample_ms=(total_time / total_samples) * 1000,
        num_features_perturbed_mean=float(
            np.mean([fp[0] for fp in all_feat_perturbed])
        ),
        num_features_perturbed_std=float(np.mean([fp[1] for fp in all_feat_perturbed])),
    )


def format_table(
    headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None
) -> str:
    """Format a simple ASCII table."""
    if col_widths is None:
        col_widths = [
            max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)
        ]
    lines = []
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-+-".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(" | ".join(str(r).ljust(w) for r, w in zip(row, col_widths)))
    return "\n".join(lines)


def generate_report(
    batch_size: int = 16,
    seq_len: int = 200,
    num_features: int = 10,
    num_repeats: int = 3,
) -> str:
    """Generate the full report as a formatted string."""
    lines = []
    lines.append("=" * 90)
    lines.append("COUNTERFACTUAL GENERATION REPORT")
    lines.append("TabPFN SCM Prior - Factual/Counterfactual Pair Statistics")
    lines.append("=" * 90)
    lines.append("")
    lines.append(
        f"Configuration: batch_size={batch_size}, seq_len={seq_len}, "
        f"num_features={num_features}, repeats={num_repeats}"
    )
    lines.append(f"Total samples per experiment: {batch_size * seq_len * num_repeats}")
    lines.append("")

    all_reports: Dict[str, Dict[str, StrategyReport]] = {}

    # -----------------------------------------------------------------------
    # Section 1: Strategy comparison (default config)
    # -----------------------------------------------------------------------
    lines.append("-" * 90)
    lines.append("SECTION 1: STRATEGY COMPARISON (default config)")
    lines.append("-" * 90)
    lines.append("")

    strategy_reports = {}
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

    all_reports["default"] = strategy_reports

    # Validity table
    lines.append("1.1 Validity (label flip rate)")
    lines.append("")
    headers = ["Strategy", "Validity Rate", "Samples"]
    rows = []
    for s in STRATEGIES:
        r = strategy_reports[s]
        rows.append([s, f"{r.validity_rate:.4f}", str(r.num_samples)])
    lines.append(format_table(headers, rows))
    lines.append("")

    # Proximity table
    lines.append("1.2 Proximity Metrics (mean +/- std)")
    lines.append("")
    headers = ["Strategy", "L1", "L2", "Linf", "Cosine Sim", "Sparsity"]
    rows = []
    for s in STRATEGIES:
        r = strategy_reports[s]
        p = r.proximity
        rows.append(
            [
                s,
                f"{p.l1_mean:.3f} +/- {p.l1_std:.3f}",
                f"{p.l2_mean:.3f} +/- {p.l2_std:.3f}",
                f"{p.linf_mean:.3f} +/- {p.linf_std:.3f}",
                f"{p.cosine_sim_mean:.4f} +/- {p.cosine_sim_std:.4f}",
                f"{p.sparsity_mean:.3f} +/- {p.sparsity_std:.3f}",
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # Per-feature MAD
    lines.append("1.3 Per-Feature Mean Absolute Deviation")
    lines.append("")
    headers = ["Strategy", "MAD (mean)", "MAD (std)", "# Features Perturbed"]
    rows = []
    for s in STRATEGIES:
        r = strategy_reports[s]
        rows.append(
            [
                s,
                f"{r.proximity.per_feature_mad_mean:.4f}",
                f"{r.proximity.per_feature_mad_std:.4f}",
                f"{r.num_features_perturbed_mean:.2f} +/- {r.num_features_perturbed_std:.2f}",
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # Timing table
    lines.append("1.4 Timing")
    lines.append("")
    headers = ["Strategy", "Total Time (s)", "Time/Sample (ms)"]
    rows = []
    for s in STRATEGIES:
        r = strategy_reports[s]
        rows.append([s, f"{r.generation_time_s:.3f}", f"{r.time_per_sample_ms:.4f}"])
    lines.append(format_table(headers, rows))
    lines.append("")

    # -----------------------------------------------------------------------
    # Section 2: Configuration variant comparison (fixed strategy = fixed_magnitude)
    # -----------------------------------------------------------------------
    lines.append("-" * 90)
    lines.append(
        "SECTION 2: CONFIGURATION VARIANT COMPARISON (strategy=fixed_magnitude)"
    )
    lines.append("-" * 90)
    lines.append("")

    variant_reports = {}
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

    headers = [
        "Variant",
        "Validity",
        "L2 dist",
        "Cosine Sim",
        "Sparsity",
        "Time/Sample (ms)",
    ]
    rows = []
    for vname in CONFIG_VARIANTS:
        r = variant_reports[vname]
        p = r.proximity
        rows.append(
            [
                vname,
                f"{r.validity_rate:.4f}",
                f"{p.l2_mean:.3f} +/- {p.l2_std:.3f}",
                f"{p.cosine_sim_mean:.4f}",
                f"{p.sparsity_mean:.3f}",
                f"{r.time_per_sample_ms:.4f}",
            ]
        )
    lines.append(format_table(headers, rows))
    lines.append("")

    # -----------------------------------------------------------------------
    # Section 3: Aggregate summary
    # -----------------------------------------------------------------------
    lines.append("-" * 90)
    lines.append("SECTION 3: AGGREGATE SUMMARY")
    lines.append("-" * 90)
    lines.append("")

    # Best strategy by validity
    best_valid = max(strategy_reports.values(), key=lambda r: r.validity_rate)
    lines.append(
        f"  Highest validity rate:  {best_valid.strategy} ({best_valid.validity_rate:.4f})"
    )

    # Best strategy by proximity (lowest L2)
    best_prox = min(strategy_reports.values(), key=lambda r: r.proximity.l2_mean)
    lines.append(
        f"  Lowest L2 distance:     {best_prox.strategy} ({best_prox.proximity.l2_mean:.4f})"
    )

    # Best strategy by proximity (highest cosine sim)
    best_cos = max(strategy_reports.values(), key=lambda r: r.proximity.cosine_sim_mean)
    lines.append(
        f"  Highest cosine sim:     {best_cos.strategy} ({best_cos.proximity.cosine_sim_mean:.4f})"
    )

    # Fastest strategy
    fastest = min(strategy_reports.values(), key=lambda r: r.time_per_sample_ms)
    lines.append(
        f"  Fastest generation:     {fastest.strategy} ({fastest.time_per_sample_ms:.4f} ms/sample)"
    )

    # Most sparse perturbation
    most_sparse = min(
        strategy_reports.values(), key=lambda r: r.proximity.sparsity_mean
    )
    lines.append(
        f"  Most sparse changes:    {most_sparse.strategy} (sparsity={most_sparse.proximity.sparsity_mean:.4f})"
    )

    lines.append("")

    # Validity vs Proximity tradeoff
    lines.append("  Validity vs Proximity tradeoff (per strategy):")
    lines.append("  " + "-" * 60)
    for s in STRATEGIES:
        r = strategy_reports[s]
        bar_v = "#" * int(r.validity_rate * 40)
        bar_p = "#" * min(40, int(r.proximity.l2_mean * 4))
        lines.append(f"  {s:25s} validity: {r.validity_rate:.3f} |{bar_v}")
        lines.append(f"  {'':25s} L2 dist:  {r.proximity.l2_mean:.3f} |{bar_p}")
        lines.append("")

    lines.append("=" * 90)
    lines.append("END OF REPORT")
    lines.append("=" * 90)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating counterfactual report...")
    print("")

    report_text = generate_report(
        batch_size=16,
        seq_len=200,
        num_features=10,
        num_repeats=3,
    )

    print("")
    print(report_text)

    # Also save to file
    output_path = "counterfactual_report.txt"
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {output_path}")
