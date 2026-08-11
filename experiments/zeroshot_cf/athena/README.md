# Athena Exp6 v3 Sweep

Run the context-size ablation on Athena with local TabPFN v3 checkpoints.
Do not run this sweep on a laptop.

## One-Time Setup On Athena

Clone this repo and create the Python environment once:

```bash
cd "$PLG_GROUPS_STORAGE/plgmodalitiescfes/$USER/CounterContEX"
uv venv --python 3.13
uv pip install -e .
uv pip install "setuptools>=70,<81"
uv pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"
uv run python experiments/zeroshot_cf/vendor_setup.py
```

Stage the v3 checkpoint files in the repo-root `models/` directory:

```bash
mkdir -p models
# Expected files:
#   models/tabpfn3_binary.ckpt
#   models/tabpfn3_regressor.ckpt
```

If the checkpoint directory is somewhere else, set `TABPFN_LOCAL_CACHE` when
submitting.

## Submit The Sweep

From the repo root on Athena:

```bash
bash experiments/zeroshot_cf/athena/submit_exp6_v3_sweep.sh
```

The submit script uses:

- Slurm account: `plgmodalitiescfes-gpu-a100`
- Partition: `plgrid-gpu-a100`
- One GPU per array task
- Wall time: 47 hours per array task
- `TABPFN_MODEL_VERSION=v3`
- `TABPFN_DEVICE=cuda`

Before starting Exp6, each Slurm task prints GPU visibility with:

```bash
nvidia-smi
```

Each row in `exp6_v3_cases.tsv` becomes one Slurm array task. Outputs go to:

```text
experiments/zeroshot_cf/results/athena/<tag>/
```

## Edit The Sweep

Edit `exp6_v3_cases.tsv`. Columns are tab-separated:

```text
dataset  selector  sizes  strategies  max_test  n_permutations  temperature  tag
```

For the context-size question, the important column is `sizes`, for example:

```text
512,1024,4096,10000
```

Use `strategies` to shard the sweep into smaller Slurm tasks. For example,
one row with `random_target` runs only that strategy for all listed sizes.
Valid strategies are:

- `random_target`
- `random_both`
- `knn_target`
- `knn_both`

If `10000` is larger than the available training pool, the driver reports
`effective_size` capped at the available pool.

For `class_divergence`, target-only strategies are skipped by the existing Exp6
logic because that selector needs both classes in context.

## Exp7: Multi-Pass Budget × Classifier-Label Conditioning

Follow-up on the Exp6 v3 residual failures (`knn_both` only, sizes 512/1024,
HELOC + MOONS). Same environment/setup as above. Submit with:

```bash
bash experiments/zeroshot_cf/athena/submit_exp7_v3_sweep.sh
```

Edit `exp7_v3_cases.tsv`; tab-separated columns are:

```text
dataset  sizes  max_test  n_permutations  temperature  max_rounds  labels  tag
```

- `max_rounds`: greedy passes over the actionable columns. `1` reproduces the
  Exp6 single-pass budget; `3` lets the loop re-edit columns (re-conditioned
  on the current CF) with a strict-improvement guard in rounds >= 2.
- `labels`: `disc` conditions the context on `disc.predict(X_train)` (aligns
  the generator with the validity oracle); `data` keeps Exp6's ground-truth
  `y_train` conditioning.

The default case file runs both datasets at `max_rounds=3,labels=disc` (extra
budget + label alignment; the reported `validity_r1` column isolates the label
effect) and at `max_rounds=1,labels=disc` (Exp6 budget, label effect alone).
Outputs land in `experiments/zeroshot_cf/results/athena/<tag>/` as
`exp7_multipass_<dataset>.csv` + `exp7_summary.md`.

## Exp8: TabPFNv3 versus TabICLv2

Exp8 runs only the final configuration selected by Exp6/Exp7:
`prob_ascent + knn_both@512 + disc labels + one pass`. It does not rerun the
context or label grids. The four array tasks are MOONS and HELOC for each
backend. Both backends project proposed values to the observed training domain
and retain the best intermediate state after a failed search. TabICL numerical
imputation uses the densest interior interval of its predicted quantile
distribution instead of its conditional median.

Stage the TabICLv2 checkpoint pair under:

```text
experiments/zeroshot_cf/models/tabicl/
```

Install `tabicl==2.1.1` into `.venv` from a compute node, then submit from the
repository root:

```bash
bash experiments/zeroshot_cf/athena/submit_exp8_backend_comparison.sh
```

Edit `exp8_backend_cases.tsv` to change test counts or estimator counts. Each
task writes its metrics to
`experiments/zeroshot_cf/results/athena/<tag>/` and uses a private
discriminator cache to avoid cross-task writes.

### TabICL optimization diagnostics

Before rerunning the full comparison, use one small GPU allocation to verify
that candidate batching and direct context replacement preserve real-model
outputs:

```bash
HF_HUB_OFFLINE=1 \
TABICL_DEVICE=cuda \
ZEROSHOT_CF_MODELS_DIR="$PWD/experiments/zeroshot_cf/results/athena/tabicl_diagnostics/models" \
.venv/bin/python -m experiments.zeroshot_cf.exp8_tabicl_diagnostics \
  --dataset heloc \
  --max-test 2 \
  --n-estimators 4 \
  --tabicl-cache-dir experiments/zeroshot_cf/models/tabicl \
  --results-dir experiments/zeroshot_cf/results/athena/tabicl_diagnostics
```

The diagnostic runs exactly three configurations. Its CSV reports `PASS` inputs
as boolean equivalence columns; the JSON contains per-point histories and the
NPZ contains the three raw counterfactual arrays.
