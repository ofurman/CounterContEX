# Athena operations

This directory contains the retained Slurm launchers for the four-dataset Exp9
TabICL benchmark.

## One-time setup

```bash
cd "$PLG_GROUPS_STORAGE/plgmodalitiescfes/$USER/CounterContEX"
uv sync --project experiments/zeroshot_cf --python 3.12 --locked
uv run --project experiments/zeroshot_cf \
  python experiments/zeroshot_cf/vendor_setup.py
```

If `vendor_setup.py` reports a revision mismatch in an existing CEL checkout,
inspect it first, then rerun with `--repin` only when you intend to rewrite the
local vendor tree.

Stage the TabICLv2 checkpoint pair once on a machine with network access:

```bash
uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.tabicl_checkpoints
```

Copy the resulting `experiments/zeroshot_cf/models/tabicl/` directory to
Athena. The array job defaults to `HF_HUB_OFFLINE=1` and never downloads
weights at runtime.

## Four benchmark cases

`exp9_countercontex_cases.tsv` intentionally contains exactly four non-comment rows:

- `heloc`
- `bank_marketing`
- `give_me_some_credit`
- `lending_club`

Each task uses the fixed 64/16/20 train/validation/test split, seed 42,
classifier-prediction targets, up to 1,000 held-out factuals, and three
requested counterfactuals unless you override the launcher environment.

## Environment variables

- `PROJECT_DIR`: repo root on Athena. Defaults to `SLURM_SUBMIT_DIR`.
- `SUITE_DIR`: suite project root. Defaults to `$PROJECT_DIR/experiments/zeroshot_cf`.
- `CASE_FILE`: case TSV path.
- `RESULTS_DIR`: output root. Defaults to `$SUITE_DIR/results/athena/exp9_countercontex`.
- `TABICL_CACHE_DIR`: staged checkpoint directory. The job exports this value as `TABICL_LOCAL_CACHE`.
- `PYTHON_BIN`: explicit interpreter override. Otherwise the launcher prefers `$SUITE_DIR/.venv/bin/python`, then `uv run --project "$SUITE_DIR" python`.
- `HF_HUB_OFFLINE`: defaults to `1`.
- `TABICL_DEVICE`: defaults to `cuda`.
- `UV_CACHE_DIR`: optional shared uv cache directory. If unset and `SCRATCH` exists, the job uses `$SCRATCH/uv-cache`.
- `WALLTIME`: Slurm time limit passed by the submit script. It defaults to
  `10:00:00`, which covers the measured 7.64-hour CounterContEx/Lending Club
  reference cell with some operational margin.

The submit script exports the selected limit to the array task as
`COUNTERCONTEX_SLURM_WALLTIME`. The Exp9 compatibility shim passes that value
into the generic runner's execution metadata, so each canonical manifest
records the requested scheduler limit without making it part of scientific
run identity. Direct `sbatch` use retains the static ten-hour fallback in the
batch file.

## Submit and aggregate

```bash
bash experiments/zeroshot_cf/athena/submit_exp9_countercontex.sh
```

Override the limit when queue policy or an ablation requires it:

```bash
WALLTIME=12:00:00 \
  bash experiments/zeroshot_cf/athena/submit_exp9_countercontex.sh
```

Outputs land under `experiments/zeroshot_cf/results/athena/exp9_countercontex/`.
That tree is ignored by git.

After all four tasks finish, aggregate the metric rows without loading TabICL:

```bash
uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.exp9_countercontex_benchmark \
  --dataset aggregate
```

Each task writes a content-addressed canonical run directory and the retained
flat Exp9 CSV/NPZ files. A run is resumable only after its `COMPLETE` marker is
published. Aggregation validates the expected completed cells and rejects
missing, partial, duplicate, or identity-mismatched results.

The full six-method reference matrix is intentionally separate from the
four-task Exp9 launcher. Run it from a host with the staged checkpoints and
all optional baseline dependencies:

```bash
uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.cli matrix \
  --config experiments/zeroshot_cf/configs/matrices/full_reference.yaml \
  --resume

uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.cli aggregate \
  --config experiments/zeroshot_cf/configs/matrices/full_reference.yaml
```

The recorded reference workload took about 9.42 hours in total. Do not treat
quality-point variation as an architecture failure; record completeness,
availability and validity denominators, phase timings, and stable artifact
hashes. Report any missing cell with its exact run ID.

For a direct offline TabICL run on Athena outside the array launcher:

```bash
HF_HUB_OFFLINE=1 TABICL_DEVICE=cuda \
uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.exp8_tabicl_cf \
  --dataset heloc \
  --max-test 10 \
  --cache-dir experiments/zeroshot_cf/models/tabicl
```
