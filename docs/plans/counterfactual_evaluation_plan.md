# Plan: Comprehensive Evaluation of SCM-based Counterfactual Generation

## Overview

Evaluate the counterfactual pairs produced by `CounterfactualSCMGenerator` across four dimensions: **validity**, **proximity**, **plausibility**, and **scalability**. The existing `counterfactual_report.py` covers validity and proximity at small scale with basic timing. This plan extends evaluation with plausibility metrics (currently absent), rigorous scalability benchmarking (1K-1M points), and a unified evaluation script that produces a single comprehensive report.

## Context

- **Existing coverage**: `counterfactual_report.py` measures validity (label flip rate), proximity (L1/L2/Linf/cosine/MAD/sparsity), dataset statistics, and basic timing. It runs at ~9,600 samples (16 batch x 200 seq x 3 repeats).
- **Gaps**: No plausibility evaluation (distributional fidelity, manifold consistency, actionability). No scalability benchmarking beyond a single batch size. No memory profiling. No GPU vs CPU comparison.
- **Key constraint**: Each batch element uses a *separate* random SCM. Scaling to 1M points means either many SCMs (large batch_size) or many samples per SCM (large seq_len), and these have different performance profiles.

---

## Stage 1: Plausibility Evaluation Module

**Goal**: Create a plausibility evaluation module that measures whether counterfactuals are realistic, not just close and valid.

**Rationale**: Validity (label flip) and proximity (distance) are necessary but insufficient. A counterfactual that flips the label with minimal distance could still be implausible (e.g., violating feature correlations, lying off the data manifold, or requiring changes to immutable features).

### Steps

1. **Create `tabpfn/scripts/plausibility_metrics.py`** with the following metrics:

   - **Distributional fidelity**: Compare marginal distributions of counterfactual features vs factual features using:
     - Per-feature Kolmogorov-Smirnov test statistic (scipy.stats.ks_2samp)
     - Jensen-Shannon divergence between factual and CF feature histograms
     - Overall: mean/max KS statistic across features

   - **Correlation structure preservation**: Measure whether inter-feature correlations are maintained:
     - Compute correlation matrix of factual data and CF data separately
     - Frobenius norm of the difference: `||Corr(x_f) - Corr(x_cf)||_F`
     - Max absolute element-wise difference in correlation matrices

   - **Manifold consistency (Local Outlier Factor)**: Check if counterfactuals lie on the factual data manifold:
     - Fit LOF on factual data, score counterfactual points
     - Report fraction of CFs flagged as outliers (at threshold -1.5)
     - Mean LOF score for factual vs counterfactual points

   - **Mahalanobis distance**: Measure how far CFs are from the factual distribution in a covariance-aware sense:
     - Fit covariance on factual data
     - Compute Mahalanobis distance for each CF point
     - Report mean, median, 95th percentile

   - **Actionability / sparsity analysis**:
     - Distribution of number of features changed per counterfactual
     - Fraction of CFs with only 1 feature changed vs 2 vs 3+
     - Conditional validity: validity rate stratified by number of features changed

2. **Create `tabpfn/scripts/plausibility_metrics.py`** as a reusable module (functions, not a standalone script) so it can be imported by the main evaluation script.

### Verification
- [ ] `python -c "from tabpfn.scripts.plausibility_metrics import compute_plausibility; print('OK')"`
- [ ] Unit test: generate a small batch, call each metric function, verify shapes and value ranges

### Commit
```
feat(eval): add plausibility metrics module (KS, correlation preservation, LOF, Mahalanobis)
```

---

## Stage 2: Scalability Benchmarking Script

**Goal**: Create a dedicated scalability benchmark that measures wall-clock time and peak memory for generating 1K, 10K, 100K, and 1M counterfactual points.

### Steps

1. **Create `tabpfn/scripts/scalability_benchmark.py`** with:

   - **Scale points**: `[1_000, 10_000, 100_000, 1_000_000]`

   - **Scaling strategies** (how to reach N total points = seq_len x batch_size):
     - `few_scms`: batch_size=1, seq_len=N (many points from one SCM) — tests per-SCM scaling
     - `many_scms`: batch_size=N/200, seq_len=200 (many SCMs, moderate points each) — tests SCM sampling overhead
     - `balanced`: batch_size=sqrt(N/200), seq_len=sqrt(N*200) (balanced) — realistic usage

   - **Measurements per configuration**:
     - Wall-clock time (time.perf_counter, 3 repeats, report mean/std)
     - Peak memory delta (tracemalloc or torch.cuda.max_memory_allocated for GPU)
     - Throughput: samples/second
     - Time breakdown: SCM sampling vs forward pass vs perturbation vs counterfactual pass (instrument key sections with timers)

   - **Device comparison**: Run on CPU. If CUDA available, also run on GPU and compare.

   - **Strategy comparison at scale**: Run all 5 perturbation strategies at 10K scale to compare relative costs. Only run 1M with `fixed_magnitude` (cheapest) and `gradient_guided` (most expensive) to bound the range.

   - **Output format**: Print a structured report table + save to `docs/scalability_report.txt`. Include extrapolation estimates for configurations not directly measured (e.g., if 1M is too slow for gradient_guided, extrapolate from 10K/100K trends).

2. **Key design decisions**:
   - Use `torch.no_grad()` everywhere (generation doesn't need gradients)
   - Warm up with one small batch before timing to exclude JIT/import overhead
   - For 1M points, use batch chunking if memory is an issue (generate in chunks of 10K, accumulate time)

### Verification
- [ ] Script runs to completion for 1K and 10K within 5 minutes
- [ ] Output report contains timing for all planned configurations
- [ ] Memory measurements are non-zero and plausible

### Commit
```
feat(eval): add scalability benchmark script (1K-1M points, CPU/GPU, memory profiling)
```

---

## Stage 3: Unified Evaluation Script

**Goal**: Create a single entry-point evaluation script that runs all evaluations (validity, proximity, plausibility, scalability) and produces a comprehensive markdown report.

### Steps

1. **Create `tabpfn/scripts/evaluate_counterfactuals.py`** that:
   - Imports from `counterfactual_report.py` (reuse `compute_proximity`, `compute_dataset_statistics`)
   - Imports from `plausibility_metrics.py` (Stage 1)
   - Imports from `scalability_benchmark.py` (Stage 2)
   - Accepts CLI arguments: `--scale` (small/medium/large), `--output` (report path), `--device`, `--strategies` (subset)
   - Runs evaluation in phases:
     1. **Quality evaluation** (validity + proximity + plausibility) at moderate scale (10K points per strategy)
     2. **Scalability evaluation** at configured scale points
   - Produces a markdown report at `docs/evaluation_report.md`

2. **Report structure**:
   ```
   # Counterfactual Evaluation Report
   ## 1. Executive Summary
   ## 2. Validity Analysis
     - Per-strategy validity rates
     - Validity vs perturbation probability sweep
   ## 3. Proximity Analysis
     - All pairs and valid-only pairs
     - Per-strategy comparison
   ## 4. Plausibility Analysis
     - Distributional fidelity (KS statistics)
     - Correlation preservation
     - Manifold consistency (LOF)
     - Mahalanobis distances
     - Actionability breakdown
   ## 5. Scalability Analysis
     - Generation time vs dataset size (1K-1M)
     - Throughput (samples/sec) at each scale
     - Memory usage
     - Per-strategy cost comparison
     - Projected times for 1K/10K/100K/1M
   ## 6. Methodology Notes
   ```

3. **Configuration presets**:
   - `--scale small`: 1K and 10K only, 1 repeat (fast CI check, ~2 min)
   - `--scale medium`: 1K, 10K, 100K, 3 repeats (~15 min)
   - `--scale large`: 1K, 10K, 100K, 1M, 5 repeats (~60+ min)

### Verification
- [ ] `python tabpfn/scripts/evaluate_counterfactuals.py --scale small` completes in <5 min
- [ ] Output markdown report has all 6 sections populated
- [ ] Numbers in report are consistent (e.g., valid-only proximity only computed on valid pairs)

### Commit
```
feat(eval): unified evaluation script with validity/proximity/plausibility/scalability
```

---

## Stage 4: Run Full Evaluation and Document Results

**Goal**: Execute the evaluation, capture results, and save the report.

### Steps

1. Run `python tabpfn/scripts/evaluate_counterfactuals.py --scale medium --output docs/evaluation_report.md`
2. Review the report for correctness and consistency
3. If any metric computation fails, fix and re-run
4. Add a brief interpretation section to the report with key findings

### Verification
- [ ] `docs/evaluation_report.md` exists and is >500 lines
- [ ] All metric tables have numeric values (no "N/A" for things that should have data)
- [ ] Scalability section shows clear trends (time should scale roughly linearly)

### Commit
```
docs(eval): add counterfactual evaluation report with full results
```

---

## Execution Protocol

### For AI Agent Execution
1. Read this plan fully before starting
2. Execute stages sequentially (1 → 2 → 3 → 4)
3. After each stage: run verification checks, update progress tracker, commit
4. If a stage fails verification, fix before moving to next stage
5. Do not modify existing files unless necessary for imports

### Dependencies
- Stage 2 is independent of Stage 1
- Stage 3 depends on Stage 1 and Stage 2
- Stage 4 depends on Stage 3

### Key Libraries
- `torch` (core computation)
- `scipy.stats` (KS test for plausibility) — already available via scikit-learn dependency
- `sklearn.neighbors.LocalOutlierFactor` (manifold consistency)
- `tracemalloc` (memory profiling, stdlib)
- `time` (wall-clock timing, stdlib)

---

## Progress Tracker

| Stage | Description | Status | Notes |
|-------|-------------|--------|-------|
| 1 | Plausibility metrics module | DONE | Added KS, JS, correlation preservation, LOF, Mahalanobis, actionability metrics |
| 2 | Scalability benchmark script | NOT_STARTED | |
| 3 | Unified evaluation script | NOT_STARTED | |
| 4 | Run evaluation and document | NOT_STARTED | |

Last stage completed: Stage 1 - Plausibility metrics module
Last updated by: plan-runner-agent
