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
