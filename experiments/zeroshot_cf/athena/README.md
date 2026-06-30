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
- `TABPFN_MODEL_VERSION=v3`
- `TABPFN_DEVICE=cuda`

Before starting Exp6, each Slurm task runs:

```bash
nvidia-smi
python experiments/zeroshot_cf/athena/check_torch_gpu.py --require-cuda
```

The task exits before the expensive experiment if PyTorch cannot see CUDA.

Each row in `exp6_v3_cases.tsv` becomes one Slurm array task. Outputs go to:

```text
experiments/zeroshot_cf/results/athena/<tag>/
```

## Edit The Sweep

Edit `exp6_v3_cases.tsv`. Columns are tab-separated:

```text
dataset  selector  sizes  max_test  n_permutations  temperature  tag
```

For the context-size question, the important column is `sizes`, for example:

```text
256,512,1024,2048,4096
```

The Exp6 driver still runs all four strategies for each size:

- `random_target`
- `random_both`
- `knn_target`
- `knn_both`

For `class_divergence`, target-only strategies are skipped by the existing Exp6
logic because that selector needs both classes in context.
