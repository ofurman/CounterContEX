"""
Unified Counterfactual Evaluation Script
==========================================
Single entry-point that runs all evaluations (validity, proximity,
plausibility, scalability) and produces a comprehensive markdown report.

Usage:
    python tabpfn/scripts/evaluate_counterfactuals.py --scale small
    python tabpfn/scripts/evaluate_counterfactuals.py --scale medium --output docs/evaluation_report.md
    python tabpfn/scripts/evaluate_counterfactuals.py --scale large --device cuda
"""

import argparse
import math
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, ".")
from tabpfn.priors.counterfactual import (
    CounterfactualSCMGenerator,
    CounterfactualBatch,
    PerturbationStrategy,
    get_default_counterfactual_config,
)
from tabpfn.scripts.counterfactual_report import (
    compute_proximity,
    compute_dataset_statistics,
    ProximityMetrics,
    DatasetStatistics,
    StrategyReport,
    run_single_experiment,
    STRATEGIES,
)
from tabpfn.scripts.plausibility_metrics import (
    compute_plausibility,
    PlausibilityReport,
)
from tabpfn.scripts.scalability_benchmark import (
    run_scalability_benchmark,
    format_report as format_scalability_report,
    TimingResult,
    StrategyComparison,
    SCALE_POINTS,
)


# ---------------------------------------------------------------------------
# Scale presets
# ---------------------------------------------------------------------------

SCALE_PRESETS = {
    "small": {
        "quality_batch_size": 8,
        "quality_seq_len": 125,   # ~1K points per strategy
        "quality_repeats": 1,
        "scale_points": [1_000, 10_000],
        "scale_repeats": 1,
    },
    "medium": {
        "quality_batch_size": 16,
        "quality_seq_len": 200,   # ~10K points (16*200*3 repeats)
        "quality_repeats": 3,
        "scale_points": [1_000, 10_000, 100_000],
        "scale_repeats": 3,
    },
    "large": {
        "quality_batch_size": 32,
        "quality_seq_len": 300,   # ~48K points (32*300*5 repeats)
        "quality_repeats": 5,
        "scale_points": [1_000, 10_000, 100_000, 1_000_000],
        "scale_repeats": 5,
    },
}


# ---------------------------------------------------------------------------
# Quality evaluation (validity + proximity + plausibility)
# ---------------------------------------------------------------------------


def run_quality_evaluation(
    strategies: List[str],
    batch_size: int,
    seq_len: int,
    num_features: int,
    num_repeats: int,
    device: str = "cpu",
) -> Dict[str, Tuple[StrategyReport, PlausibilityReport]]:
    """Run quality evaluation for each strategy.

    Returns dict mapping strategy name to (StrategyReport, PlausibilityReport).
    """
    results = {}

    for strategy in strategies:
        print(f"  Quality eval: {strategy} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        # Generate data using existing infrastructure
        config = get_default_counterfactual_config()
        config["perturbation_strategy"] = strategy
        gen = CounterfactualSCMGenerator(config, device=device)

        all_x_f, all_x_cf, all_flipped, all_mask, all_batches = [], [], [], [], []
        all_times = []
        total_samples = 0

        for _ in range(num_repeats):
            t_rep = time.perf_counter()
            with torch.no_grad():
                batch = gen.generate_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_features=num_features,
                )
            all_times.append(time.perf_counter() - t_rep)

            n = batch_size * seq_len
            total_samples += n
            all_x_f.append(batch.x_factual.reshape(-1, num_features))
            all_x_cf.append(batch.x_counterfactual.reshape(-1, num_features))
            all_flipped.append(batch.label_flipped.reshape(-1))
            all_mask.append(batch.intervention_mask.reshape(-1, num_features))
            all_batches.append(batch)

        x_f = torch.cat(all_x_f, dim=0)
        x_cf = torch.cat(all_x_cf, dim=0)
        flipped = torch.cat(all_flipped, dim=0)
        mask = torch.cat(all_mask, dim=0)

        num_valid = int(flipped.sum().item())
        validity_rate = num_valid / total_samples if total_samples > 0 else 0.0

        # Proximity
        prox_all = compute_proximity(x_f, x_cf)
        prox_valid = compute_proximity(x_f, x_cf, mask=flipped)

        # Dataset stats
        ds_stats = compute_dataset_statistics(all_batches[-1])

        # Feature perturbation stats
        n_perturbed_all = mask.float().sum(dim=-1)
        fp_mean = n_perturbed_all.mean().item()
        fp_std = n_perturbed_all.std().item()
        if num_valid > 0:
            n_perturbed_valid = mask[flipped].float().sum(dim=-1)
            vfp_mean = n_perturbed_valid.mean().item()
            vfp_std = n_perturbed_valid.std().item()
        else:
            vfp_mean, vfp_std = 0.0, 0.0

        total_time = sum(all_times)
        strategy_report = StrategyReport(
            strategy=strategy,
            num_samples=total_samples,
            validity_rate=validity_rate,
            num_valid=num_valid,
            num_invalid=total_samples - num_valid,
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

        # Plausibility
        plaus = compute_plausibility(x_f, x_cf, flipped)

        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s, validity={validity_rate:.3f})")
        results[strategy] = (strategy_report, plaus)

    return results


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Generate a markdown table."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def generate_markdown_report(
    quality_results: Dict[str, Tuple[StrategyReport, PlausibilityReport]],
    scaling_results: List[TimingResult],
    strategy_comp: StrategyComparison,
    preset: str,
    device: str,
    num_features: int,
) -> str:
    """Generate the full markdown evaluation report."""
    lines = []
    strategies = list(quality_results.keys())

    # --- Header ---
    lines.append("# Counterfactual Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Scale preset: `{preset}` | Device: `{device}` | Features: {num_features}")
    lines.append("")

    # --- 1. Executive Summary ---
    lines.append("## 1. Executive Summary")
    lines.append("")

    # Find best strategy for each metric
    best_validity = max(quality_results.items(), key=lambda kv: kv[1][0].validity_rate)
    best_prox = min(quality_results.items(), key=lambda kv: kv[1][0].proximity_all.l2_mean)
    best_plaus_ks = min(quality_results.items(), key=lambda kv: kv[1][1].distributional.ks_mean)
    fastest = min(quality_results.items(), key=lambda kv: kv[1][0].time_per_sample_ms)

    lines.append(f"- **Highest validity rate**: `{best_validity[0]}` ({best_validity[1][0].validity_rate:.4f})")
    lines.append(f"- **Best proximity (L2)**: `{best_prox[0]}` (L2={best_prox[1][0].proximity_all.l2_mean:.4f})")
    lines.append(f"- **Best distributional fidelity (KS)**: `{best_plaus_ks[0]}` (mean KS={best_plaus_ks[1][1].distributional.ks_mean:.4f})")
    lines.append(f"- **Fastest generation**: `{fastest[0]}` ({fastest[1][0].time_per_sample_ms:.4f} ms/sample)")
    lines.append("")

    # Summary table
    headers = ["Strategy", "Validity", "L2 (all)", "KS (mean)", "Corr Frob", "LOF Outlier %", "Time/Sample (ms)"]
    rows = []
    for s in strategies:
        sr, pr = quality_results[s]
        rows.append([
            s,
            f"{sr.validity_rate:.4f}",
            f"{sr.proximity_all.l2_mean:.4f}",
            f"{pr.distributional.ks_mean:.4f}",
            f"{pr.correlation.frobenius_norm:.4f}",
            f"{pr.manifold.outlier_fraction:.4f}",
            f"{sr.time_per_sample_ms:.4f}",
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # --- 2. Validity Analysis ---
    lines.append("## 2. Validity Analysis")
    lines.append("")
    lines.append("Validity measures the fraction of counterfactual pairs where the class label flipped.")
    lines.append("")

    headers = ["Strategy", "Validity Rate", "# Valid", "# Invalid", "Total"]
    rows = []
    for s in strategies:
        sr, _ = quality_results[s]
        rows.append([
            s,
            f"{sr.validity_rate:.4f}",
            str(sr.num_valid),
            str(sr.num_invalid),
            str(sr.num_samples),
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # --- 3. Proximity Analysis ---
    lines.append("## 3. Proximity Analysis")
    lines.append("")

    # 3.1 All pairs
    lines.append("### 3.1 All Pairs")
    lines.append("")
    headers = ["Strategy", "L1", "L2", "Linf", "Cosine Sim", "Sparsity"]
    rows = []
    for s in strategies:
        p = quality_results[s][0].proximity_all
        rows.append([
            s,
            f"{p.l1_mean:.3f} +/- {p.l1_std:.3f}",
            f"{p.l2_mean:.3f} +/- {p.l2_std:.3f}",
            f"{p.linf_mean:.3f} +/- {p.linf_std:.3f}",
            f"{p.cosine_sim_mean:.4f}",
            f"{p.sparsity_mean:.3f}",
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # 3.2 Valid-only pairs
    lines.append("### 3.2 Valid Pairs Only (label flipped)")
    lines.append("")
    headers = ["Strategy", "# Valid", "L1", "L2", "Linf", "Cosine Sim", "Sparsity"]
    rows = []
    for s in strategies:
        sr, _ = quality_results[s]
        p = sr.proximity_valid
        if p.num_pairs == 0:
            rows.append([s, "0", "N/A", "N/A", "N/A", "N/A", "N/A"])
        else:
            rows.append([
                s,
                str(p.num_pairs),
                f"{p.l1_mean:.3f} +/- {p.l1_std:.3f}",
                f"{p.l2_mean:.3f} +/- {p.l2_std:.3f}",
                f"{p.linf_mean:.3f} +/- {p.linf_std:.3f}",
                f"{p.cosine_sim_mean:.4f}",
                f"{p.sparsity_mean:.3f}",
            ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # 3.3 Comparison
    lines.append("### 3.3 All vs Valid Comparison (L2)")
    lines.append("")
    headers = ["Strategy", "L2 (all)", "L2 (valid)", "Ratio (valid/all)"]
    rows = []
    for s in strategies:
        sr, _ = quality_results[s]
        pa = sr.proximity_all
        pv = sr.proximity_valid
        if pv.num_pairs == 0:
            rows.append([s, f"{pa.l2_mean:.3f}", "N/A", "N/A"])
        else:
            ratio = pv.l2_mean / pa.l2_mean if pa.l2_mean > 0 else 0
            rows.append([s, f"{pa.l2_mean:.3f}", f"{pv.l2_mean:.3f}", f"{ratio:.3f}"])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # --- 4. Plausibility Analysis ---
    lines.append("## 4. Plausibility Analysis")
    lines.append("")

    # 4.1 Distributional fidelity
    lines.append("### 4.1 Distributional Fidelity")
    lines.append("")
    lines.append("Kolmogorov-Smirnov test statistics and Jensen-Shannon divergence between")
    lines.append("factual and counterfactual feature distributions.")
    lines.append("")
    headers = ["Strategy", "KS Mean", "KS Max", "JS Mean", "JS Max"]
    rows = []
    for s in strategies:
        _, pr = quality_results[s]
        d = pr.distributional
        rows.append([
            s,
            f"{d.ks_mean:.4f}",
            f"{d.ks_max:.4f}",
            f"{d.js_mean:.6f}",
            f"{d.js_max:.6f}",
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # 4.2 Correlation preservation
    lines.append("### 4.2 Correlation Preservation")
    lines.append("")
    lines.append("Measures how well inter-feature correlations are maintained in counterfactuals.")
    lines.append("")
    headers = ["Strategy", "Frobenius Norm", "Max Abs Diff", "Mean Abs Diff"]
    rows = []
    for s in strategies:
        _, pr = quality_results[s]
        c = pr.correlation
        rows.append([
            s,
            f"{c.frobenius_norm:.4f}",
            f"{c.max_abs_diff:.4f}",
            f"{c.mean_abs_diff:.4f}",
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # 4.3 Manifold consistency
    lines.append("### 4.3 Manifold Consistency (LOF)")
    lines.append("")
    lines.append("Local Outlier Factor analysis: checks if counterfactuals lie on the factual data manifold.")
    lines.append("")
    headers = ["Strategy", "Outlier Fraction", "Mean LOF (factual)", "Mean LOF (CF)"]
    rows = []
    for s in strategies:
        _, pr = quality_results[s]
        m = pr.manifold
        rows.append([
            s,
            f"{m.outlier_fraction:.4f}",
            f"{m.mean_lof_factual:.4f}",
            f"{m.mean_lof_counterfactual:.4f}",
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # 4.4 Mahalanobis distance
    lines.append("### 4.4 Mahalanobis Distance")
    lines.append("")
    lines.append("Covariance-aware distance of counterfactuals from the factual distribution center.")
    lines.append("")
    headers = ["Strategy", "Mean", "Median", "95th Percentile", "Std"]
    rows = []
    for s in strategies:
        _, pr = quality_results[s]
        mh = pr.mahalanobis
        rows.append([
            s,
            f"{mh.mean:.4f}",
            f"{mh.median:.4f}",
            f"{mh.p95:.4f}",
            f"{mh.std:.4f}",
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # 4.5 Actionability
    lines.append("### 4.5 Actionability / Sparsity Analysis")
    lines.append("")
    lines.append("Distribution of number of features changed per counterfactual.")
    lines.append("")
    headers = ["Strategy", "Mean # Changed", "1 Feature", "2 Features", "3+ Features"]
    rows = []
    for s in strategies:
        _, pr = quality_results[s]
        a = pr.actionability
        rows.append([
            s,
            f"{a.mean_features_changed:.2f}",
            f"{a.frac_1_feature:.4f}",
            f"{a.frac_2_features:.4f}",
            f"{a.frac_3plus_features:.4f}",
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # Conditional validity table
    lines.append("**Conditional validity (validity rate by # features changed):**")
    lines.append("")
    # Collect all unique change counts across strategies
    all_change_counts = set()
    for s in strategies:
        _, pr = quality_results[s]
        all_change_counts.update(pr.actionability.validity_by_num_changed.keys())
    sorted_counts = sorted(all_change_counts)

    if sorted_counts:
        headers = ["Strategy"] + [f"{nc} feat" for nc in sorted_counts]
        rows = []
        for s in strategies:
            _, pr = quality_results[s]
            a = pr.actionability
            row = [s]
            for nc in sorted_counts:
                if nc in a.validity_by_num_changed:
                    count = a.count_by_num_changed.get(nc, 0)
                    row.append(f"{a.validity_by_num_changed[nc]:.3f} (n={count})")
                else:
                    row.append("N/A")
            rows.append(row)
        lines.append(_md_table(headers, rows))
        lines.append("")

    # --- 5. Scalability Analysis ---
    lines.append("## 5. Scalability Analysis")
    lines.append("")

    # 5.1 Generation time vs dataset size
    lines.append("### 5.1 Generation Time vs Dataset Size")
    lines.append("")

    # Filter to main scaling results (fixed_magnitude)
    scale_results = [
        r for r in scaling_results
        if r.strategy == "fixed_magnitude"
        and r.scaling_strategy in ("few_scms", "many_scms", "balanced")
        and r.error is None
    ]

    if scale_results:
        headers = ["Points", "Scaling", "batch_size", "seq_len", "Time (s)", "Throughput (pts/s)", "Memory (MB)"]
        rows = []
        for r in scale_results:
            rows.append([
                f"{r.total_points:,}",
                r.scaling_strategy,
                str(r.batch_size),
                str(r.seq_len),
                f"{r.wall_time_mean:.3f} +/- {r.wall_time_std:.3f}",
                f"{r.throughput:,.0f}",
                f"{r.peak_memory_mb:.1f}",
            ])
        lines.append(_md_table(headers, rows))
    else:
        lines.append("*No scaling results available.*")
    lines.append("")

    # 5.2 Per-strategy cost comparison
    lines.append("### 5.2 Per-Strategy Cost Comparison")
    lines.append(f"")
    lines.append(f"Measured at {strategy_comp.scale:,} points (balanced scaling).")
    lines.append("")

    if strategy_comp.results:
        headers = ["Strategy", "Time (s)", "Throughput (pts/s)", "Memory (MB)"]
        rows = []
        base_r = strategy_comp.results.get("fixed_magnitude")
        for strat, r in strategy_comp.results.items():
            if r.error:
                rows.append([strat, "ERROR", "N/A", "N/A"])
            else:
                rel = ""
                if base_r and not base_r.error and base_r.wall_time_mean > 0:
                    ratio = r.wall_time_mean / base_r.wall_time_mean
                    rel = f" ({ratio:.2f}x)"
                rows.append([
                    strat,
                    f"{r.wall_time_mean:.3f}{rel}",
                    f"{r.throughput:,.0f}",
                    f"{r.peak_memory_mb:.1f}",
                ])
        lines.append(_md_table(headers, rows))
    else:
        lines.append("*No strategy comparison results available.*")
    lines.append("")

    # 5.3 Projected times
    lines.append("### 5.3 Projected Generation Times")
    lines.append("")

    balanced_results = [
        r for r in scale_results
        if r.scaling_strategy == "balanced" and not r.error
    ]
    if balanced_results:
        headers = ["Measured Points", "Throughput (pts/s)", "Projected 10K (s)", "Projected 100K (s)", "Projected 1M (s)"]
        rows = []
        for r in balanced_results:
            tp = r.throughput
            if tp > 0:
                rows.append([
                    f"{r.total_points:,}",
                    f"{tp:,.0f}",
                    f"{10_000/tp:.2f}",
                    f"{100_000/tp:.2f}",
                    f"{1_000_000/tp:.1f}",
                ])
        lines.append(_md_table(headers, rows))
    else:
        lines.append("*No balanced scaling data for projections.*")
    lines.append("")

    # 5.4 1M bounds
    bound_results = [
        r for r in scaling_results
        if r.total_points >= 900_000
        and r.scaling_strategy == "balanced"
        and r.strategy in ("fixed_magnitude", "gradient_guided")
    ]
    if bound_results:
        lines.append("### 5.4 1M Point Strategy Bounds")
        lines.append("")
        headers = ["Strategy", "Time (s)", "Throughput (pts/s)", "Memory (MB)"]
        rows = []
        for r in bound_results:
            if r.error:
                rows.append([r.strategy, f"ERROR: {r.error}", "N/A", "N/A"])
            else:
                rows.append([
                    r.strategy,
                    f"{r.wall_time_mean:.2f}",
                    f"{r.throughput:,.0f}",
                    f"{r.peak_memory_mb:.1f}",
                ])
        lines.append(_md_table(headers, rows))
        lines.append("")

    # --- 6. Methodology Notes ---
    lines.append("## 6. Methodology Notes")
    lines.append("")
    lines.append("### Data Generation")
    lines.append("- Each batch element uses a separate random SCM (Structural Causal Model)")
    lines.append("- Counterfactuals are generated by perturbing input features according to the perturbation strategy")
    lines.append("- Label validity is determined by whether the SCM-predicted class changes after perturbation")
    lines.append("")
    lines.append("### Metrics")
    lines.append("- **Validity**: Fraction of counterfactual pairs where the predicted class label flipped")
    lines.append("- **Proximity**: Distance between factual and counterfactual points (L1, L2, Linf, cosine similarity)")
    lines.append("- **Sparsity**: Fraction of features changed per counterfactual")
    lines.append("- **Distributional fidelity**: KS test and JS divergence between factual/CF marginal distributions")
    lines.append("- **Correlation preservation**: Frobenius norm of difference between factual/CF correlation matrices")
    lines.append("- **Manifold consistency**: LOF-based outlier detection of CFs relative to factual data manifold")
    lines.append("- **Mahalanobis distance**: Covariance-aware distance of CFs from factual distribution center")
    lines.append("- **Actionability**: Distribution of number of features changed, conditional validity by change count")
    lines.append("")
    lines.append("### Scalability")
    lines.append("- **few_scms**: batch_size=1, seq_len=N (tests per-SCM scaling)")
    lines.append("- **many_scms**: batch_size=N/200, seq_len=200 (tests SCM sampling overhead)")
    lines.append("- **balanced**: sqrt-based split (realistic usage pattern)")
    lines.append("- Memory measured via `tracemalloc` (CPU) or `torch.cuda.max_memory_allocated` (GPU)")
    lines.append("- Warmup batch excluded from timing")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Unified counterfactual evaluation script"
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="small",
        help="Evaluation scale preset (small ~2min, medium ~15min, large ~60+min)",
    )
    parser.add_argument(
        "--output",
        default="docs/evaluation_report.md",
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Subset of strategies to evaluate (default: all)",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    preset = SCALE_PRESETS[args.scale]
    strategies = args.strategies or STRATEGIES

    print("=" * 70)
    print("Unified Counterfactual Evaluation")
    print("=" * 70)
    print(f"Scale: {args.scale} | Device: {device} | Features: {args.num_features}")
    print(f"Strategies: {strategies}")
    print("")

    # --- Phase 1: Quality evaluation ---
    print("-" * 70)
    print("Phase 1: Quality Evaluation (validity + proximity + plausibility)")
    print("-" * 70)

    quality_results = run_quality_evaluation(
        strategies=strategies,
        batch_size=preset["quality_batch_size"],
        seq_len=preset["quality_seq_len"],
        num_features=args.num_features,
        num_repeats=preset["quality_repeats"],
        device=device,
    )
    print("")

    # --- Phase 2: Scalability evaluation ---
    print("-" * 70)
    print("Phase 2: Scalability Evaluation")
    print("-" * 70)

    scaling_results, strategy_comp = run_scalability_benchmark(
        scale_points=preset["scale_points"],
        device=device,
        num_features=args.num_features,
        num_repeats=preset["scale_repeats"],
    )
    print("")

    # --- Generate report ---
    print("-" * 70)
    print("Generating markdown report...")
    print("-" * 70)

    report = generate_markdown_report(
        quality_results=quality_results,
        scaling_results=scaling_results,
        strategy_comp=strategy_comp,
        preset=args.scale,
        device=device,
        num_features=args.num_features,
    )

    with open(args.output, "w") as f:
        f.write(report)
    print(f"Report saved to {args.output}")
    print("")

    # Print summary
    print("=" * 70)
    print("Evaluation complete!")
    print(f"Full report: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
