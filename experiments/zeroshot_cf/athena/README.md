# TabICL experiments on Athena

This directory contains the Slurm launcher for the TabICL DiCoFlex benchmark.

## One-time setup

Clone the repository and create the shared Python environment:

```bash
cd "$PLG_GROUPS_STORAGE/plgmodalitiescfes/$USER/CounterContEX"
uv venv --python 3.13
uv pip install -r experiments/zeroshot_cf/requirements.txt
uv run python experiments/zeroshot_cf/vendor_setup.py
```

Stage the TabICLv2 checkpoint pair under:

```text
experiments/zeroshot_cf/models/tabicl/
```

You can prepare that directory once on a machine with network access:

```bash
uv run python -m experiments.zeroshot_cf.tabicl_checkpoints
```

Copy the resulting directory to Athena. Set `TABICL_CACHE_DIR` when submitting
if the checkpoint directory is stored elsewhere.

## Exp9: DiCoFlex datasets with TabICL

Exp9 evaluates TabICL on HELOC, Bank Marketing, Give Me Some Credit, and
Lending Club. Each dataset uses a fixed stratified 64/16/20
train/validation/test split with seed 42, up to 1,000 held-out factuals, and
three requested counterfactuals per factual by default.

The four rows in `exp9_dicoflex_cases.tsv` run as independent Slurm array
tasks:

```bash
bash experiments/zeroshot_cf/athena/submit_exp9_dicoflex.sh
```

Each task writes aggregate metrics, per-point diagnostics, and compressed
factual/counterfactual arrays under:

```text
experiments/zeroshot_cf/results/athena/exp9_dicoflex/
```

After all tasks finish, combine their metric rows without loading TabICL:

```bash
.venv/bin/python -m experiments.zeroshot_cf.exp9_dicoflex_benchmark \
  --dataset aggregate
```

For a direct MOONS or HELOC TabICL run outside the array launcher, use:

```bash
HF_HUB_OFFLINE=1 TABICL_DEVICE=cuda \
.venv/bin/python -m experiments.zeroshot_cf.exp8_tabicl_cf \
  --dataset heloc --max-test 10 \
  --cache-dir experiments/zeroshot_cf/models/tabicl
```
