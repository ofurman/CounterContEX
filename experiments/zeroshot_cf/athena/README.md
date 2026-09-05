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

## Full-matrix, 3-seed benchmark (all methods x all datasets)

`full_reference_3seeds_array.sbatch` runs the complete reference matrix --
all six registered methods (`countercontex`, `nice`, `wachter`,
`growing_spheres`, `dice`, `face`) across all four datasets and three seeds
(`42`, `43`, `44`) -- as one Slurm array of independent single-cell tasks.

Unlike the four-task Exp9 launcher above, this is not a bespoke script: each
array task slices exactly one `(dataset, method, seed)` cell out of
[`configs/matrices/full_reference_3seeds.yaml`](../configs/matrices/full_reference_3seeds.yaml)
with `select_matrix_cell.py --index "$SLURM_ARRAY_TASK_ID"` and runs it
through the same generic `cli.py matrix` runner used locally. Index `N` in
the array always names the same cell as row `N` of
`cli.py matrix --config full_reference_3seeds.yaml --dry-run`, so the array
size (4 datasets x 6 methods x 3 seeds = 72) is derived from the config
itself rather than hardcoded.

### Same classifier checkpoint everywhere

Every cell for a given dataset -- any method, any seed -- must be scored
against the identical classifier. The config's `target_model` block never
varies across cells, and the default case loader trains and disk-caches that
classifier once per dataset at `$ZEROSHOT_CF_MODELS_DIR/disc_<dataset>_lr.pkl`
(`discriminator.py`), keyed only by dataset name. Because up to 3 array tasks
per dataset (one per seed) could otherwise race to train and overwrite that
file the first time they run, `submit_full_reference_3seeds.sh` first runs
`warm_classifier_cache.py` synchronously for all four datasets *before*
submitting the array, so every one of the 72 tasks only ever reads an
already-populated, shared checkpoint. Set `SKIP_WARM=1` to skip this step if
the cache directory is already warm.

### Submit and aggregate

```bash
bash experiments/zeroshot_cf/athena/submit_full_reference_3seeds.sh
```

Override the walltime per task (default `10:00:00`, sized for the heaviest
CounterContEx/TabICL cell) the same way as the Exp9 launcher:

```bash
WALLTIME=12:00:00 \
  bash experiments/zeroshot_cf/athena/submit_full_reference_3seeds.sh
```

Outputs land under
`experiments/zeroshot_cf/results/athena/full_reference_3seeds/`, content
addressed by run ID, so all 72 concurrent array tasks can safely write into
the same `output_root`. After every task finishes:

```bash
uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.cli aggregate \
  --config experiments/zeroshot_cf/configs/matrices/full_reference_3seeds.yaml
```

`aggregate` validates that all 72 expected cells completed and rejects
missing, partial, duplicate, or identity-mismatched results, reporting any
missing cell by its exact run ID.

### Metrics land in Weights & Biases automatically

`GenericRunner.run()` pushes every computed metric to wandb the moment each
cell's `EvaluationReport` is produced (`orchestration/runner.py::_wandb_log`)
-- summary metrics, the per-point and per-candidate tables, and raw
metric-array histograms -- for both a freshly computed cell and a
`--resume`d one whose report was only just loaded from disk. This is wired
into the config, not a separate script: any matrix config with a top-level
`wandb.project` (see `full_reference_3seeds.yaml`) gets this for free; a
config without that block skips wandb entirely.

Athena's GPU compute nodes have no outbound network, so the array script
exports `WANDB_MODE=offline` and `WANDB_DIR="$RESULTS_DIR/wandb"`: wandb
still logs every metric locally as each task finishes, it just defers the
upload. Once the array is done, sync everything from a node with network
access (the login node, after `wandb login` or with `WANDB_API_KEY` set):

```bash
wandb sync experiments/zeroshot_cf/results/athena/full_reference_3seeds/wandb
```

Each cell is a stable wandb run keyed by its own scientific `run_id`, grouped
by the config's `suite` name and tagged by dataset/method/seed, so re-syncing
(or re-running a resumed cell) updates the same run instead of duplicating
it. Running the matrix locally with network access needs no offline dance --
just unset `WANDB_MODE` (or leave it unset) and metrics upload live.

`wandb` was added as a dependency; if you already have a locked
`experiments/zeroshot_cf/uv.lock` from before this change, run
`uv lock --project experiments/zeroshot_cf` (needs network access) once
before `uv sync --locked` will succeed again.

For a direct offline TabICL run on Athena outside the array launcher:

```bash
HF_HUB_OFFLINE=1 TABICL_DEVICE=cuda \
uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.exp8_tabicl_cf \
  --dataset heloc \
  --max-test 10 \
  --cache-dir experiments/zeroshot_cf/models/tabicl
```
