# SLURM: full 256-point TabPFN benchmark

Runs `run_full_benchmark.py --max-test 256` (the german/adult/admission/student
`l2c` + adult_dicoflex/bank/default/gmc/lending-club/sba `dicoflex` config from
`BENCHMARK_CONFIG`) as a 10-task SLURM array — one dataset per task, so they
run in parallel across separate GPU allocations instead of one long serial job.

`--n-repeats 5` (default): each query point gets 5 independent CFs (different
sampler seed per repeat) so `l2c_diversity_weight_fast` /
`dicoflex_pairwise_distance` are non-degenerate — see "Diversity" below.
This makes the benchmark ~5x more expensive than a single pass.

This was deliberately **not** run locally at 256 points: per-point TabPFN cost
varies ~10x across datasets (dominated by actionable-column count), and the
worst case (`default`, 23 actionable columns) alone is estimated at ~3.75h at
n_repeats=5, even on an RTX 4070 Ti (~10.7h if all 10 datasets were run
serially on one GPU) — not something to burn a single interactive session on.
Use this directory to run it on a cluster instead, where the 10 datasets can
run in parallel as an array.

## 1. One-time setup (login node, needs network)

`run_benchmark_array.sbatch` activates a conda env with `conda activate
/net/tscratch/people/plgbrinpow/envs/ccex` and runs the benchmark with plain
`python` (not `uv run python` — `uv run` would ignore that activated env and
manage its own, so this must stay in sync with whatever env path the sbatch
activates). Create and populate it once:

```bash
module load Miniconda3
eval "$(conda shell.bash hook)"

conda create -y -p /net/tscratch/people/plgbrinpow/envs/ccex python=3.11
conda activate /net/tscratch/people/plgbrinpow/envs/ccex

cd /net/pr2/projects/plgrid/plggtabrep/brinpow/CounterContEX
python -m pip install --upgrade pip
python -m pip install -r experiments/zeroshot_cf/requirements.txt
```

Notes:
- `cel` (line 8 of `requirements.txt`, installed via `git+https://...`) isn't
  needed by anything `run_full_benchmark.py` touches — the local-dataset path
  is cel-free by design (see `data.py`'s lazy import). Delete that line first
  if the git install is slow/flaky on the cluster's network.
- On Linux, `pip install torch` pulls CUDA-enabled wheels directly from PyPI
  (unlike Windows, no special `--index-url` needed) — `tabpfn`'s own
  dependency resolution should get this right automatically.
- Then stage the TabPFN checkpoints + verify the datasets are present (still
  useful even with a conda env — it just calls into the activated `python`):
  ```bash
  bash experiments/zeroshot_cf/slurm/stage_offline_assets.sh
  ```
  That script itself still says `uv run python`/`uv pip install` — either
  swap those two lines for plain `python`/`python -m pip install` to match
  the conda env, or run its steps manually: stage checkpoints
  (`python -c "from experiments.zeroshot_cf.checkpoints import stage_checkpoints; stage_checkpoints()"`),
  confirm `experiments/zeroshot_cf/datasets/<name>/{config.json,train.csv,test.csv}`
  exist for all 10 datasets, and `wandb login` if you want W&B synced.

## 2. `run_benchmark_array.sbatch` is configured for PLGrid

`--partition=plgrid-gpu-a100`, `--account=plgtabanomdet-gpu-a100`, 1 A100
GPU/task, `--cpus-per-task=16`, `--mem=120G`, `--time=10:00:00`,
`--job-name=ccex`. `--time` is comfortably over the ~3.75h calibrated worst
case (`default`) even allowing for the A100s behaving differently from the
local RTX 4070 Ti this was calibrated on (see "Timing" below) — tighten it if
you want less queue-priority impact.

If you're on a different cluster, swap these `#SBATCH` values, and swap the
`conda activate .../envs/ccex` line + `python` invocation for whatever your
cluster uses (`uv run python`, `module load` + a different env, etc.).

## 3. Submit

```bash
sbatch experiments/zeroshot_cf/slurm/run_benchmark_array.sbatch      # all 10 datasets
sbatch --array=0 experiments/zeroshot_cf/slurm/run_benchmark_array.sbatch   # german only
```

Array index → dataset (must match `run_full_benchmark.BENCHMARK_CONFIG`'s key
order): `0=german 1=adult 2=admission 3=student 4=adult_dicoflex 5=bank
6=default 7=gmc 8=lending-club 9=sba`.

Results land in `experiments/zeroshot_cf/results/benchmark_l2c_metrics.csv`
and `benchmark_dicoflex_metrics.csv` (appended per task — safe to run tasks in
parallel, each process opens/appends/closes the CSV independently, though if
you need strict safety under concurrent writes, consider `--array=0-9%1` to
serialize).

## Timing (measured on a local RTX 4070 Ti, `--n-permutations 1`, single-pass 16-point calibration, x5 for n_repeats)

Cost scales with `n_points x n_actionable_columns x n_repeats`; batching
amortizes per-point cost significantly within a single repeat's pass (a
16-point batch runs several times faster per point than a 1-point batch), so
the per-repeat 256-point rate should be *at least* as good as this 16-point
calibration extrapolated linearly, not worse. The `x5 (n_repeats)` column is
that per-repeat estimate multiplied by 5 — **not** independently measured at
n_repeats=5, since a small (4-point) n_repeats=5 smoke test showed
meaningfully worse per-point batching (each repeat only gets `--max-test`
points, so small `--max-test` values batch poorly regardless of n_repeats):

| Dataset | Actionable cols | 16-pt time | s/point | 256-pt, 1 repeat | 256-pt, x5 (n_repeats) |
|---|---:|---:|---:|---:|---:|
| german | 16 | 89.3s | 5.58 | ~24 min | ~2.0h |
| adult | 8 | 15.6s | 0.98 | ~4 min | ~20 min |
| admission | 6 | 7.9s | 0.49 | ~2 min | ~10 min |
| student | 10 | 22.5s | 1.41 | ~6 min | ~30 min |
| adult_dicoflex | 12 | 17.1s | 1.07 | ~5 min | ~25 min |
| bank | 16 | 69.8s | 4.36 | ~19 min | ~1.6h |
| default | 23 | 168.0s | 10.50 | ~45 min | ~3.75h |
| gmc | 10 | 13.1s | 0.82 | ~4 min | ~20 min |
| lending-club | 12 | 21.8s | 1.36 | ~6 min | ~30 min |
| sba | 13 | 47.0s | 2.94 | ~13 min | ~1.1h |

Serial total (all 10 on one GPU, one after another): ~10.7h. As a SLURM array
with GPUs available in parallel, wall-clock is bounded by the slowest single
task (`default`, ~3.75h).

The current `--time=10:00:00` in the sbatch covers the worst case (`default`,
~3.75h estimated) with margin, applied uniformly across the array (SLURM
arrays don't support per-task time limits) — the cheaper datasets will just
finish early. If you want tighter individual budgets, submit each dataset
separately with `--array=<i> --time=<HH:MM:SS>`.

These numbers are extrapolated from one local GPU and a 16-point,
single-repeat sample — **recalibrate for real before trusting the `--time`
budget**, e.g. `--max-test 16 --n-repeats 5` on your actual cluster hardware,
since this table's n_repeats=5 column is arithmetic (x5), not measured.

## Diversity

`--n-repeats 5` (default) draws 5 independent CFs per query point (different
sampler seed each time) and pools them before scoring, so
`l2c_diversity_weight_fast` / `dicoflex_pairwise_distance` (and anything
harmonic-meaned with them) are non-degenerate — confirmed on a small german
smoke test (18.5% diversity, vs. 0 at n_repeats=1). Use `--n-repeats 1` to
fall back to a single pass (5x cheaper, diversity always 0).

## Weights & Biases

Each of the 10 array tasks logs its dataset's run to W&B as its own run,
named `<dataset>-<disc_type>-<metric_suite>-mt256-nr5-seed42` (e.g.
`german-lr-l2c-mt256-nr5-seed42`) — identifiable at a glance in the W&B UI,
with config (dataset/disc_type/metric_suite/max_test/n_repeats/n_permutations/seed)
and every metric logged.

The sbatch currently sets `WANDB_MODE=online` and a fixed
`WANDB_PROJECT=CounterContEX` — this assumes the PLGrid A100 compute nodes
have outbound network access (unlike the TabPFN-checkpoint download, which
this project always treats as login-node-only via `HF_HUB_OFFLINE=1`).
**Verify that assumption** — if a compute node turns out to be offline,
`wandb.init()` will hang/retry rather than fail fast. If so, switch back to
offline logging + sync afterward:

```bash
# in the sbatch, replace WANDB_MODE=online with:
export WANDB_MODE=offline
```
```bash
# then, from a login node afterward (needs network + `wandb login` once, see step 1):
wandb sync wandb/offline-run-*
```
(runs are written locally under `<repo_root>/wandb/offline-run-*` and each
prints its own `wandb sync <path>` command in the job's `.out` log either way).

`WANDB_PROJECT` is hardcoded to `CounterContEX` in the sbatch now rather than
overridable via `sbatch --export` — edit that line directly to change it. If
you don't want W&B at all, drop `--wandb-project "$WANDB_PROJECT"` from the
sbatch's `run_full_benchmark.py` call and add `--no-wandb` instead.
