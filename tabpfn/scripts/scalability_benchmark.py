"""
Scalability Benchmark for Counterfactual Generation
=====================================================
Measures wall-clock time and peak memory for generating counterfactual
points at various scales (1K to 1M), with different scaling strategies
(few_scms, many_scms, balanced) and perturbation strategies.

Output: structured report table printed to stdout and saved to
docs/scalability_report.txt.
"""

import math
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, ".")
from tabpfn.priors.counterfactual import (
    CounterfactualSCMGenerator,
    PerturbationStrategy,
    get_default_counterfactual_config,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    """Timing and memory result for a single benchmark configuration."""

    total_points: int = 0
    batch_size: int = 0
    seq_len: int = 0
    strategy: str = ""
    scaling_strategy: str = ""
    device: str = "cpu"
    wall_time_mean: float = 0.0
    wall_time_std: float = 0.0
    throughput: float = 0.0  # samples/second
    peak_memory_mb: float = 0.0
    num_repeats: int = 0
    error: Optional[str] = None


@dataclass
class StrategyComparison:
    """Comparison of perturbation strategies at a fixed scale."""

    scale: int = 0
    results: Dict[str, TimingResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


SCALE_POINTS = [1_000, 10_000, 100_000, 1_000_000]

ALL_STRATEGIES = [s.value for s in PerturbationStrategy]


def compute_scaling_params(
    total_points: int, scaling_strategy: str
) -> Tuple[int, int]:
    """Compute (batch_size, seq_len) for a given total point count and strategy.

    Args:
        total_points: target total number of points (batch_size * seq_len)
        scaling_strategy: one of 'few_scms', 'many_scms', 'balanced'

    Returns:
        (batch_size, seq_len) tuple
    """
    if scaling_strategy == "few_scms":
        # One SCM, many points
        return 1, total_points
    elif scaling_strategy == "many_scms":
        # Many SCMs, 200 points each
        seq_len = 200
        batch_size = max(1, total_points // seq_len)
        return batch_size, seq_len
    elif scaling_strategy == "balanced":
        # Balanced: sqrt-based split
        # batch_size * seq_len = total_points
        # Aim for batch_size ≈ sqrt(total_points / 200), seq_len ≈ sqrt(total_points * 200)
        # but ensure product = total_points
        seq_len_target = int(math.sqrt(total_points * 200))
        seq_len_target = max(1, seq_len_target)
        batch_size = max(1, total_points // seq_len_target)
        seq_len = total_points // batch_size
        return batch_size, seq_len
    else:
        raise ValueError(f"Unknown scaling strategy: {scaling_strategy}")


def run_benchmark_single(
    total_points: int,
    scaling_strategy: str,
    perturbation_strategy: str,
    device: str = "cpu",
    num_features: int = 10,
    num_repeats: int = 3,
    chunk_size: int = 10_000,
) -> TimingResult:
    """Run a single benchmark configuration.

    For large point counts, generates in chunks to avoid OOM.

    Args:
        total_points: target number of points
        scaling_strategy: how to split into batch_size/seq_len
        perturbation_strategy: perturbation strategy name
        device: 'cpu' or 'cuda'
        num_features: number of features
        num_repeats: number of timing repeats
        chunk_size: max points per generation call (for memory)
    """
    batch_size, seq_len = compute_scaling_params(total_points, scaling_strategy)
    actual_points = batch_size * seq_len

    config = get_default_counterfactual_config()
    config["perturbation_strategy"] = perturbation_strategy
    gen = CounterfactualSCMGenerator(config, device=device)

    times = []

    for rep in range(num_repeats):
        # Memory tracking
        tracemalloc.start()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()

        # If points fit in one call, do it directly; otherwise chunk
        if actual_points <= chunk_size or scaling_strategy == "few_scms":
            # For few_scms we can't easily chunk (single SCM)
            try:
                with torch.no_grad():
                    gen.generate_batch(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_features=num_features,
                    )
            except Exception as e:
                tracemalloc.stop()
                return TimingResult(
                    total_points=actual_points,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    strategy=perturbation_strategy,
                    scaling_strategy=scaling_strategy,
                    device=device,
                    num_repeats=num_repeats,
                    error=str(e),
                )
        else:
            # Chunk by batch_size dimension
            remaining_batches = batch_size
            while remaining_batches > 0:
                chunk_bs = min(remaining_batches, max(1, chunk_size // seq_len))
                with torch.no_grad():
                    gen.generate_batch(
                        batch_size=chunk_bs,
                        seq_len=seq_len,
                        num_features=num_features,
                    )
                remaining_batches -= chunk_bs

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        # Memory measurement
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    peak_memory_mb = peak_mem / (1024 * 1024)
    if device == "cuda" and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    mean_time = sum(times) / len(times)
    std_time = (
        (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
        if len(times) > 1
        else 0.0
    )

    return TimingResult(
        total_points=actual_points,
        batch_size=batch_size,
        seq_len=seq_len,
        strategy=perturbation_strategy,
        scaling_strategy=scaling_strategy,
        device=device,
        wall_time_mean=mean_time,
        wall_time_std=std_time,
        throughput=actual_points / mean_time if mean_time > 0 else 0.0,
        peak_memory_mb=peak_memory_mb,
        num_repeats=num_repeats,
    )


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def warmup(device: str = "cpu") -> None:
    """Run a small batch to exclude JIT/import overhead from timing."""
    config = get_default_counterfactual_config()
    gen = CounterfactualSCMGenerator(config, device=device)
    with torch.no_grad():
        gen.generate_batch(batch_size=2, seq_len=50, num_features=5)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_scalability_benchmark(
    scale_points: Optional[List[int]] = None,
    scaling_strategies: Optional[List[str]] = None,
    device: str = "cpu",
    num_features: int = 10,
    num_repeats: int = 3,
    chunk_size: int = 10_000,
) -> Tuple[List[TimingResult], StrategyComparison]:
    """Run the full scalability benchmark suite.

    Args:
        scale_points: list of total point counts to benchmark
        scaling_strategies: list of scaling strategies
        device: 'cpu' or 'cuda'
        num_features: number of features
        num_repeats: number of timing repeats per config
        chunk_size: max points per generation call

    Returns:
        (scaling_results, strategy_comparison) tuple
    """
    if scale_points is None:
        scale_points = SCALE_POINTS
    if scaling_strategies is None:
        scaling_strategies = ["few_scms", "many_scms", "balanced"]

    print("Warming up...", flush=True)
    warmup(device)
    print("Warmup complete.\n", flush=True)

    # --- Part 1: Scaling benchmark ---
    # Use fixed_magnitude as the default strategy for scaling tests
    scaling_results: List[TimingResult] = []
    default_strategy = "fixed_magnitude"

    for n_points in scale_points:
        for ss in scaling_strategies:
            bs, sl = compute_scaling_params(n_points, ss)
            actual = bs * sl
            print(
                f"  Scaling: {actual:>10,} points | {ss:12s} | "
                f"bs={bs}, sl={sl} | {default_strategy} ...",
                end=" ",
                flush=True,
            )
            result = run_benchmark_single(
                total_points=n_points,
                scaling_strategy=ss,
                perturbation_strategy=default_strategy,
                device=device,
                num_features=num_features,
                num_repeats=num_repeats,
                chunk_size=chunk_size,
            )
            if result.error:
                print(f"ERROR: {result.error}")
            else:
                print(
                    f"{result.wall_time_mean:.2f}s "
                    f"({result.throughput:,.0f} pts/s, "
                    f"{result.peak_memory_mb:.1f} MB)"
                )
            scaling_results.append(result)

    # --- Part 2: Strategy comparison at 10K ---
    print("\n  Strategy comparison at 10K points:")
    comparison_scale = 10_000
    strategy_comp = StrategyComparison(scale=comparison_scale)

    for strat in ALL_STRATEGIES:
        print(f"    {strat:25s} ...", end=" ", flush=True)
        result = run_benchmark_single(
            total_points=comparison_scale,
            scaling_strategy="balanced",
            perturbation_strategy=strat,
            device=device,
            num_features=num_features,
            num_repeats=num_repeats,
            chunk_size=chunk_size,
        )
        if result.error:
            print(f"ERROR: {result.error}")
        else:
            print(f"{result.wall_time_mean:.2f}s ({result.throughput:,.0f} pts/s)")
        strategy_comp.results[strat] = result

    # --- Part 3: 1M with cheapest and most expensive ---
    print("\n  1M point strategy bounds (fixed_magnitude vs gradient_guided):")
    bound_results: List[TimingResult] = []
    for strat in ["fixed_magnitude", "gradient_guided"]:
        print(f"    {strat:25s} ...", end=" ", flush=True)
        result = run_benchmark_single(
            total_points=1_000_000,
            scaling_strategy="balanced",
            perturbation_strategy=strat,
            device=device,
            num_features=num_features,
            num_repeats=1,  # Only 1 repeat for 1M
            chunk_size=chunk_size,
        )
        if result.error:
            print(f"ERROR: {result.error}")
        else:
            print(f"{result.wall_time_mean:.2f}s ({result.throughput:,.0f} pts/s)")
        bound_results.append(result)

    # Append bound results to scaling_results for reporting
    scaling_results.extend(bound_results)

    return scaling_results, strategy_comp


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(
    scaling_results: List[TimingResult],
    strategy_comp: StrategyComparison,
    device: str = "cpu",
) -> str:
    """Format benchmark results into a text report."""
    lines = []
    lines.append("=" * 100)
    lines.append("SCALABILITY BENCHMARK REPORT")
    lines.append("CounterContEX - Counterfactual Generation Performance")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Device: {device}")
    if device == "cuda" and torch.cuda.is_available():
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    lines.append("")

    # --- Section 1: Scaling results ---
    lines.append("-" * 100)
    lines.append("SECTION 1: SCALING BENCHMARK (strategy=fixed_magnitude)")
    lines.append("-" * 100)
    lines.append("")

    # Filter to just the fixed_magnitude scaling results (not the bound results)
    scale_results_only = [
        r for r in scaling_results
        if r.strategy == "fixed_magnitude"
        and r.scaling_strategy in ("few_scms", "many_scms", "balanced")
    ]

    headers = [
        "Points", "Strategy", "batch_size", "seq_len",
        "Time (s)", "Throughput (pts/s)", "Memory (MB)", "Status",
    ]
    col_w = [12, 12, 12, 12, 18, 18, 12, 10]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_w))
    lines.append(header_line)
    lines.append("-+-".join("-" * w for w in col_w))

    for r in scale_results_only:
        status = "OK" if r.error is None else "FAIL"
        time_str = f"{r.wall_time_mean:.3f} +/- {r.wall_time_std:.3f}" if not r.error else "N/A"
        tp_str = f"{r.throughput:,.0f}" if not r.error else "N/A"
        mem_str = f"{r.peak_memory_mb:.1f}" if not r.error else "N/A"
        row = [
            f"{r.total_points:,}",
            r.scaling_strategy,
            str(r.batch_size),
            str(r.seq_len),
            time_str,
            tp_str,
            mem_str,
            status,
        ]
        lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    lines.append("")

    # --- Section 2: Strategy comparison ---
    lines.append("-" * 100)
    lines.append(f"SECTION 2: STRATEGY COMPARISON AT {strategy_comp.scale:,} POINTS")
    lines.append("-" * 100)
    lines.append("")

    headers2 = ["Strategy", "Time (s)", "Throughput (pts/s)", "Memory (MB)", "Status"]
    col_w2 = [25, 18, 18, 12, 10]

    header_line2 = " | ".join(h.ljust(w) for h, w in zip(headers2, col_w2))
    lines.append(header_line2)
    lines.append("-+-".join("-" * w for w in col_w2))

    for strat in ALL_STRATEGIES:
        r = strategy_comp.results.get(strat)
        if r is None:
            continue
        status = "OK" if r.error is None else "FAIL"
        time_str = f"{r.wall_time_mean:.3f} +/- {r.wall_time_std:.3f}" if not r.error else "N/A"
        tp_str = f"{r.throughput:,.0f}" if not r.error else "N/A"
        mem_str = f"{r.peak_memory_mb:.1f}" if not r.error else "N/A"
        row = [strat, time_str, tp_str, mem_str, status]
        lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, col_w2)))

    lines.append("")

    # Relative cost comparison
    base_strat = "fixed_magnitude"
    base_r = strategy_comp.results.get(base_strat)
    if base_r and not base_r.error and base_r.wall_time_mean > 0:
        lines.append("  Relative cost (vs fixed_magnitude):")
        for strat in ALL_STRATEGIES:
            r = strategy_comp.results.get(strat)
            if r and not r.error:
                ratio = r.wall_time_mean / base_r.wall_time_mean
                lines.append(f"    {strat:25s}  {ratio:.2f}x")
        lines.append("")

    # --- Section 3: 1M bounds ---
    lines.append("-" * 100)
    lines.append("SECTION 3: 1M POINT STRATEGY BOUNDS")
    lines.append("-" * 100)
    lines.append("")

    bound_results = [
        r for r in scaling_results
        if r.total_points >= 900_000  # ~1M
        and r.scaling_strategy == "balanced"
        and r.strategy in ("fixed_magnitude", "gradient_guided")
    ]

    for r in bound_results:
        status = "OK" if r.error is None else f"FAIL: {r.error}"
        if r.error:
            lines.append(f"  {r.strategy:25s}  {status}")
        else:
            lines.append(
                f"  {r.strategy:25s}  "
                f"{r.wall_time_mean:.2f}s  "
                f"{r.throughput:,.0f} pts/s  "
                f"{r.peak_memory_mb:.1f} MB"
            )
    lines.append("")

    # --- Section 4: Extrapolations ---
    lines.append("-" * 100)
    lines.append("SECTION 4: THROUGHPUT EXTRAPOLATIONS")
    lines.append("-" * 100)
    lines.append("")

    # Estimate throughput from balanced scaling results
    balanced_results = [
        r for r in scale_results_only
        if r.scaling_strategy == "balanced" and not r.error
    ]
    if len(balanced_results) >= 2:
        lines.append("  Based on balanced scaling (fixed_magnitude):")
        lines.append(f"  {'Points':>12s}  {'Measured Time':>14s}  {'Throughput':>14s}  {'Projected 1M':>14s}")
        for r in balanced_results:
            proj_1m = 1_000_000 / r.throughput if r.throughput > 0 else float("inf")
            lines.append(
                f"  {r.total_points:>12,}  "
                f"{r.wall_time_mean:>13.3f}s  "
                f"{r.throughput:>12,.0f}/s  "
                f"{proj_1m:>13.1f}s"
            )
    lines.append("")

    lines.append("=" * 100)
    lines.append("END OF SCALABILITY REPORT")
    lines.append("=" * 100)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scalability benchmark for counterfactual generation")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=SCALE_POINTS,
        help="Scale points to benchmark",
    )
    parser.add_argument("--output", default="docs/scalability_report.txt")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("=" * 60)
    print("Scalability Benchmark - Counterfactual Generation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Scales: {args.scales}")
    print(f"Repeats: {args.num_repeats}")
    print("")

    scaling_results, strategy_comp = run_scalability_benchmark(
        scale_points=args.scales,
        device=device,
        num_features=args.num_features,
        num_repeats=args.num_repeats,
    )

    report = format_report(scaling_results, strategy_comp, device=device)
    print("")
    print(report)

    with open(args.output, "w") as f:
        f.write(report)
    print(f"\nReport saved to {args.output}")
