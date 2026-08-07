# Zero-Shot Autoregressive Counterfactuals with TabPFN

Fully offline experiment testing whether TabPFN v2 used as a conditional density estimator
can generate actionable counterfactuals without retraining.

## Quick Start

### 1. One-time checkpoint staging (network required once)

```bash
uv run python -c "from experiments.zeroshot_cf.checkpoints import stage_checkpoints; stage_checkpoints()"
```

Checkpoints are saved to `experiments/zeroshot_cf/models/` (gitignored).

### 2. Verify offline environment

```bash
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/smoke_test.py
```

### 3. Run experiments

```bash
# Experiment 1: single-feature estimation (sanity check)
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/exp1_single_feature.py

# Experiment 2: counterfactual generation
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/exp2_counterfactuals.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TABPFN_DEVICE` | `auto` | Device for TabPFN inference (`auto`/`cpu`/`mps`) |
| `TABPFN_LOCAL_CACHE` | `experiments/zeroshot_cf/models/` | Path to staged checkpoints |
| `HF_HUB_OFFLINE` | unset | Set to `1` to enforce no network access |

## Dependencies

Install via:
```bash
uv pip install -r experiments/zeroshot_cf/requirements.txt
```

`cel` (counterfactuals evaluation library) is vendored in
`experiments/zeroshot_cf/vendor/counterfactuals/` due to Python 3.13 / TensorFlow
incompatibility and installed editable with `--no-deps`.

## Datasets & in-context rows

Both datasets use the `cel` 80/20 **stratified** split (`random_state=42`) and MinMax
scaling to `[0,1]` fit on the train split. TabPFN is never trained — the "context" is the
in-context conditioning set passed to `TabPFNUnsupervisedModel.fit()` at inference time.

| Dataset | Features | Train rows | Train per-class | Test rows (full split) | In-context rows used |
|---------|---------:|-----------:|-----------------|-----------------------:|---------------------:|
| MOONS | 2 | 800 | [397, 403] | 200 | 256 |
| HELOC | 23 | 8367 | [4367, 4000] | 2092 | 256 |

**How the in-context row count is determined** (`sampler.py:set_context`, `exp2:159-173`):

- The conditioning set is selected from the **training** split only (the test point being
  explained is never in context).
- With `--context-type target_only` (default), context = the training rows belonging to the
  **target (desired) class**; with `--context-type all_classes`, context = the full training set.
- It is then subsampled to `--max-context` (default **256**) rows, deterministically
  (`np.random.default_rng(random_state)`). Because every per-class training pool exceeds 256
  (MOONS ≥397, HELOC ≥4000), the in-context set is **256 rows in every current configuration**.
  Raise `--max-context` to use more (cost and MPS memory grow with context size).
- **Query (test) rows evaluated**: capped by `--max-test` (default MOONS 100, HELOC 50; use
  `-1` for the **full** stratified test split — MOONS 200, HELOC 2092).

## Local datasets (ported from CETGFN)

Ten additional datasets are ported from the sibling `CETGFN` (CounterFlowNet)
project into `experiments/zeroshot_cf/datasets/<name>/`: `adult`,
`adult_dicoflex`, `bank`, `default`, `german`, `gmc`, `lending-club`, `sba`,
`student`, `admission`. Each folder holds:

- `config.json` — tracked in git; numerical/categorical columns, target column,
  actionability (`immutable`/`increasing`/`decreasing`), and (for the
  discretized set below) fixed bin edges.
- `train.csv` / `val.csv` / `test.csv`, `model.pt`, `flow.pt` (and `model.pkl`
  where CETGFN produced one) — gitignored (large data/checkpoint files, same
  convention as `models/` and `vendor/`). The classifier/flow checkpoints are
  copied through for reference only; they are **not** loaded by this
  experiment — `discriminator.py` still trains its own sklearn oracle from the
  CSVs, same as for `heloc`/`moons`.

Loading is handled by `local_data.py` and dispatched transparently from
`data.load_dataset()` / `data.get_actionable_immutable()`, so these names work
anywhere `--dataset` is accepted (`exp4_greedy_cf.py`, `exp5_selector_ablation.py`,
`exp6_context_ablation.py`):

```bash
uv run python experiments/zeroshot_cf/exp4_greedy_cf.py --dataset german --selector prob_ascent
```

**Discretization**: `german`, `adult`, `admission`, and `student` keep
`numerical_bins` in their `config.json` and are binned into ordinal codes at
load time — the same fixed, hand-authored bin edges rgfn's `L2CDiscretizer`
uses for these four datasets upstream — before the usual MinMax scaling. The
other six datasets had `numerical_bins` stripped when ported and stay
continuous (MinMax only, no binarisation), matching how `heloc`/`moons` are
already handled here.

## DiCE baseline (comparison against CETGFN's L2C metrics)

`dice_baseline.py` runs Microsoft's DiCE (`dice-ml`, method="random") against
this experiment's own sklearn discriminator, then reports the same four
metrics `L2CCounterfactualMetrics` computes in ../CETGFN
(`rgfn/trainer/metrics/counterfactual_metrics.py`, wired to
`configs/l2c_counterfactual.gin`): validity, sparsity, diversity, and their
harmonic mean — reimplemented in pure numpy so no `cel`/`rgfn`/PyTorch
dependency is needed for this comparison.

```bash
uv run python experiments/zeroshot_cf/dice_baseline.py --dataset german --total-cfs 5
```

For `german` (a `DISCRETIZED_DATASETS` entry — see above), all features are
treated as categorical for DiCE (matching their finite, L2C-binned value
sets), and `features_to_vary` is restricted to the actionable (non-immutable)
columns. On the full 200-row test split (`--max-test 0`, `total_CFs=5`,
seed=42): validity 100.0, sparsity 89.6, diversity 13.9, harmonic mean 24.0
— every query found a valid flip; results land in
`results/dice_baseline_german_metrics.csv`.

## exp2 (TabPFN zero-shot) L2C comparison

`exp2_l2c_report.py` runs `exp2_counterfactuals.py`'s TabPFN zero-shot CF
generation `--n-repeats` times (each with a different sampler seed, via the
`base_seed` param added to `generate_counterfactuals`/`run_dataset`) and pools
the results to report the same `l2c_metrics.py` metrics as the DiCE baseline
above — exp2 normally emits exactly one CF per test point, so repeats are
needed for a non-degenerate `l2c_diversity_weight_fast`.

```bash
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/exp2_l2c_report.py \
    --dataset german --n-repeats 5 --max-test 10 --n-permutations 1
```

TabPFN's per-column autoregressive imputation is comparatively expensive
(unlike DiCE's baseline, which only needs a `.predict()` call per candidate) —
budget `--max-test`/`--n-repeats`/`--n-permutations` accordingly; a GPU build
of `torch` (`TABPFN_DEVICE=cuda`) is close to an order of magnitude faster
than CPU here.

german, 5 repeats x 10 queries, `n_permutations=1`, seed=42: validity 58.0,
sparsity 41.6, diversity 21.2, harmonic mean 28.1 (`results/exp2_l2c_german_metrics.csv`)
— vs. DiCE-random's validity 100.0 / sparsity 89.6 / diversity 13.9 / hmean
24.0 above. TabPFN's joint-imputation draw changes more of the 16 actionable
columns per edit and isn't guaranteed to flip the classifier's prediction
(unlike DiCE, which explicitly searches for a flip), so lower validity and
sparsity are expected; its per-column posterior draws also explore a bit more
per-query variation, hence the slightly higher diversity.

## Full benchmark (all 10 local datasets, classifier + metric-suite per dataset)

`run_full_benchmark.py` pairs each ported dataset with a classifier type and
metric suite (`BENCHMARK_CONFIG`): german/admission use logistic regression,
adult/student use an MLP, all four report `l2c_metrics.py`; the six
non-discretized datasets (adult_dicoflex, bank, default, gmc, lending-club,
sba) use an MLP and `dicoflex_metrics.py` (a numpy port of CETGFN's
`DiCoFlexCounterfactualMetrics`). Each query point gets `--n-repeats` (default
5) independent TabPFN generation passes, different sampler seed each time,
pooled before scoring — so `l2c_diversity_weight_fast` /
`dicoflex_pairwise_distance` are non-degenerate (confirmed on a german smoke
test: 18.5% diversity at n_repeats=5, vs. 0 at n_repeats=1). This makes the
benchmark ~5x more expensive than a single pass; use `--n-repeats 1` to fall
back to single-pass (diversity always 0, 5x cheaper).

Each dataset is also logged to Weights & Biases as its own run, named
`<dataset>-<disc_type>-<metric_suite>-mt<max_test>-nr<n_repeats>-seed<seed>`
(e.g. `german-lr-l2c-mt256-nr5-seed42`) so a run is identifiable at a glance —
config (dataset/disc_type/metric_suite/max_test/n_repeats/n_permutations/seed)
and all metrics are logged. Needs `wandb` (`uv sync --extra wandb`) and either
network access or `WANDB_MODE=offline` (sync later with `wandb sync`); disable
entirely with `--no-wandb`.

```bash
# Smoke test (fast) before a real run
uv run python experiments/zeroshot_cf/run_full_benchmark.py --max-test 1

# Full run — expensive (256 points x 5 repeats x 10 datasets); see
# slurm/README.md for per-dataset timing estimates and a SLURM array job to
# run it on a cluster instead of locally.
HF_HUB_OFFLINE=1 TABPFN_DEVICE=cuda uv run python experiments/zeroshot_cf/run_full_benchmark.py --max-test 256
```

## Results

Metric reports are committed to `experiments/zeroshot_cf/results/`.
Raw data (large arrays) are gitignored.

### Headline numbers (2026-06-15, post-review corrected)

| Dataset | Exp | Validity | LOF plausibility | True actionability |
|---------|-----|---------|-----------------|-------------------|
| MOONS | Exp 2 | **1.00** | **1.076** (excellent) | **1.000** |
| HELOC | Exp 2 | **0.52** | ~3.1B (structurally poor) | **1.000** |

- MOONS validity=1.0 far exceeds the ≥0.70 target; plausibility is excellent (LOF≈1.08, zero OOB).
- HELOC validity=0.52 barely meets the ≥0.50 target; plausibility is poor (72% of CFs extrapolate outside [0,1] before clipping — root cause: 17/23 features masked with only 7 observed values under sparse conditioning).
- True actionability = 1.0 on both datasets (immutable columns frozen by construction).

#### Full test-split (2026-06-16)

Re-run on the **full** stratified test split (MOONS n=200, HELOC n=2092) at the same config:
MOONS validity **0.995** (LOF 1.060, OOB 0.010); HELOC validity **0.538** (LOF 5.68e9, OOB
0.653, `n_failed=0`). Numbers are stable vs. the capped runs — the sparse-conditioning
extrapolation on HELOC is confirmed at scale. (HELOC `true_actionability`=0.9986 is a
clip-vs-MinMax metric artefact on ~3 boundary rows; the unclipped immutability assert passed.
See `results/REPORT.md §3`.) Use `--max-test -1` to reproduce the full split.

#### How `frac_oob` (out-of-bounds fraction) is computed

`frac_oob` is the **row-level extrapolation rate** of the generated counterfactuals
(`exp2_counterfactuals.py:264-267`):

```python
oob_mask = (X_cf < 0.0) | (X_cf > 1.0)          # per-cell, on the UNCLIPPED generated CFs
frac_oob = float(oob_mask.any(axis=1).mean())   # fraction of CF ROWS with ≥1 out-of-range cell
```

- All features are MinMax-scaled to `[0,1]` on the **train** split, so any generated value
  outside `[0,1]` means TabPFN extrapolated **beyond the training range** for that feature.
- A CF **row** is counted out-of-bounds if **at least one** of its feature values is `< 0` or
  `> 1`; `frac_oob` is the mean of that row-level indicator over all evaluated test points.
- It is measured on the **raw imputed array, before** the `np.clip(X_cf, 0, 1)` that is applied
  prior to computing validity / proximity / LOF — so it reports how often the model left the
  valid region, not how the clipped CF scores. High `frac_oob` (e.g. HELOC 0.72) is the direct
  signature of sparse-conditioning extrapolation: 17/23 features imputed from only ~7 observed
  values. (Rows whose autoregressive chain overflows `float32` are mapped to an out-of-range
  sentinel, so they too count as OOB; see `robust_impute`.)

See `results/REPORT.md` for the full analysis and recommended next steps.

### Run Experiment 2 with recommended configs

```bash
# MOONS recommended: t=0.5, all-class context (best proximity 0.643, validity 0.983)
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/exp2_counterfactuals.py \
    --dataset moons --temperature 0.5 --context-type all_classes

# HELOC (original config is best; refinement sweep shows temperature doesn't fix OOB)
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/exp2_counterfactuals.py --dataset heloc

# Refinement sweep
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/refine.py --dataset moons
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/refine.py --dataset heloc
```
