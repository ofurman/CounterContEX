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

## Results

Metric reports are committed to `experiments/zeroshot_cf/results/`.
Raw data (large arrays) are gitignored.

### Headline numbers (2026-06-15, post-review corrected)

| Dataset | Exp | Validity | LOF plausibility | True actionability |
|---------|-----|---------|-----------------|-------------------|
| MOONS | Exp 2 | **1.00** | **1.076** (excellent) | **1.000** |
| HELOC | Exp 2 | **0.52** | ~3.1B (structurally poor) | **1.000** |
| MOONS | Exp 4 (beam, either regime) | **1.00** | **0.98** (excellent) | 1.000 |
| HELOC | Exp 4 — Set 1 frozen immutables | 0.13 | 7.9e6 (off-manifold) | **1.000** |
| HELOC | Exp 4 — Set 2 from scratch | **1.00** | **1.01** (excellent, OOB=0) | 0.000 (immutables regenerated) |

> **Exp 4 (Stage 10)** runs a task-guided **beam search** in two regimes that differ only
> in whether immutables are masked. **Set 2 (from scratch — mask nothing)** generates every
> feature from `p(X|Y=target)` and reaches **validity 1.0, LOF 1.0, frac_oob 0%** on HELOC —
> strictly beating Exp 2 — but gives up actionability (immutables drift ~0.12). **Set 1
> (freeze immutables, actionable)** keeps `true_actionability=1.0` but validity collapses to
> 0.13 and plausibility degrades, because pinning HELOC's highly class-predictive immutables
> to the wrong-class values forces an off-manifold, invalid configuration. The finding:
> **actionability and validity+plausibility are in fundamental tension on HELOC.** See
> `results/REPORT.md §8`.

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

### Run Experiment 4 (beam search: frozen-immutable vs from-scratch)

```bash
# Both regimes (Set 1 frozen + Set 2 from scratch), both datasets
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/exp4_beam_search.py \
    --dataset all --set both

# Just the from-scratch regime on HELOC
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/exp4_beam_search.py \
    --dataset heloc --set fromscratch

# Proximity dial: raise --lambda-actionable to pull actionables toward the factual
# (lambda must be O(20-100) to overcome TabPFN's log-density scale)
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/exp4_beam_search.py \
    --dataset heloc --set fromscratch --lambda-actionable 20 --max-test 20
```

### Run Experiment 7 (beam-search hyperparameter sweep)

Exp 4's grid was produced with one never-varied configuration. Exp 7 varies five
axes around those defaults. Generation runs on PLGrid; scoring runs locally.

`--run-id` is what makes a sweep possible: without it every run overwrites the same
`arrays/exp4_<dataset>_<set>_cfs.npz` and rewrites `exp4_summary.md`. With it, all
artifacts move into a sweep namespace (`results/arrays/sweep/`, `results/sweep/`)
and the untagged Exp-4 outputs are left untouched. The full resolved config — plus
the code commit and Slurm job id — is stored inside each npz as `config_json`.

```bash
# One configuration (the --run-id slug may not contain underscores)
HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/exp4_beam_search.py \
    --dataset heloc --set frozen --max-test -1 --chunk-size 4096 \
    --run-id lam0 --lambda-actionable 0.0

# Tail candidate quantiles instead of the default interior grid. An explicit list
# overrides --n-candidates: branching becomes len(probs) + 1 (the extra is the mode).
    --run-id probs-tail --candidate-probs tail
    --run-id probs-custom --candidate-probs 0.05,0.5,0.95

# The whole sweep, on Helios — one job per cell, configs sequential within a job
CELL=heloc:frozen bash plgrid/submit.sh plgrid/30_beam_sweep.sbatch
CELL=law:frozen   bash plgrid/submit.sh plgrid/30_beam_sweep.sbatch

# Score every config-tagged array under both metric conventions, then render
uv run python experiments/zeroshot_cf/exp7_sweep_table.py
uv run python experiments/zeroshot_cf/exp7_report.py
```

`--chunk-size` is **not** a sweep axis and stays 4096 everywhere: TabPFN's
predictions depend on the composition of the predict batch, so varying it would
confound every other axis.
