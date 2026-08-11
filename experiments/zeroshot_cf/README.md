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

### 4. Run the TabICL backend at the Athena-winning context

TabICL uses the fixed `prob_ascent + knn_both@512` configuration selected by
the Athena v3 runs. It does not repeat the context grid. Candidate feature
interventions are batched into one TabICL imputation call per greedy step.

Stage the two TabICLv2 checkpoints once with network access:

```bash
uv run python -m experiments.zeroshot_cf.tabicl_checkpoints
```

This writes ordinary checkpoint files under
`experiments/zeroshot_cf/models/tabicl/`. Copy that directory to the same path
on Athena, or pass its transferred location with `--tabicl-cache-dir`.

Verify the transferred weights and one minimal real-model imputation offline:

```bash
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.tabicl_smoke_test
```

Then run fully offline:

```bash
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp8_tabicl_cf \
    --dataset moons

HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp8_tabicl_cf \
    --dataset heloc --max-test 50

HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp8_tabicl_cf \
    --dataset audit --max-test 50
```

The default context labels are predictions from the discriminator being
explained, matching the Athena Exp7 follow-up. Use `--context-labels data` only
to reproduce the earlier Exp6 ground-truth-label setup. For a small timing and
equivalence baseline, use `--candidate-mode sequential`; production runs use
the default `batched` mode.

### Two-dataset backend comparison (run on Athena)

The comparison fixes the already-selected `prob_ascent + knn_both@512`
configuration and runs only TabPFNv3 versus TabICLv2 on MOONS and HELOC.
Each backend/dataset writes a separate result file, so they can be
submitted as independent jobs:

```bash
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp8_backend_comparison \
    --dataset moons --backend tabicl \
    --tabicl-cache-dir experiments/zeroshot_cf/models/tabicl

HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp8_backend_comparison \
    --dataset heloc --backend tabpfn --tabpfn-cache-dir models
```

Use `--dataset all --backend all` only on an appropriately provisioned compute
node. For a smoke test, add `--max-test 1 --n-estimators 1`.

To check the two TabICL speed optimizations with the real checkpoint, run the
small paired diagnostic on a GPU node:

```bash
HF_HUB_OFFLINE=1 TABICL_DEVICE=cuda \
  .venv/bin/python -m experiments.zeroshot_cf.exp8_tabicl_diagnostics \
  --dataset heloc --max-test 2 --n-estimators 4 \
  --tabicl-cache-dir experiments/zeroshot_cf/models/tabicl \
  --results-dir experiments/zeroshot_cf/results/athena/tabicl_diagnostics
```

It compares batched versus sequential candidates and direct context replacement
versus upstream `fit()`, writing a verdict CSV plus detailed JSON/NPZ artifacts.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TABPFN_DEVICE` | `auto` | Device for TabPFN inference (`auto`/`cpu`/`mps`) |
| `TABPFN_LOCAL_CACHE` | `experiments/zeroshot_cf/models/` | Path to staged checkpoints |
| `TABPFN_V3_LOCAL_CACHE` | `models/` | Path to the Athena TabPFNv3 weight pair |
| `TABICL_DEVICE` | `auto` | Device for TabICL inference (`auto`/`cpu`/`mps`/`cuda`) |
| `TABICL_LOCAL_CACHE` | `experiments/zeroshot_cf/models/tabicl/` | Path to staged TabICLv2 checkpoints |
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

The datasets use the `cel` 80/20 **stratified** split (`random_state=42`) and MinMax
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
