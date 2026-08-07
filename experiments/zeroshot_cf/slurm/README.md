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

```bash
bash experiments/zeroshot_cf/slurm/stage_offline_assets.sh
```

This installs dependencies, downloads the TabPFN v2 checkpoints into
`experiments/zeroshot_cf/models/` (compute nodes are usually offline —
`run_full_benchmark.py` is run with `HF_HUB_OFFLINE=1`), checks that the 10
ported datasets are present under `experiments/zeroshot_cf/datasets/<name>/`
(gitignored — copy them from a machine that has them, e.g. `rsync -av`), runs
a 1-point offline CPU smoke test (with `--no-wandb`, since it's just checking
TabPFN/dataset readiness), and offers to run `wandb login` (only needed if you
want the array job's offline W&B runs synced to the cloud later).

## 2. Edit `run_benchmark_array.sbatch`

Fill in `--partition`/`--account` for your cluster, and the environment-setup
block (the script defaults to `uv run python ...`, matching this project's own
README convention — swap for `module load` / `conda activate` if your cluster
doesn't have `uv`). `WANDB_PROJECT` can be overridden via the environment
(`sbatch --export=WANDB_PROJECT=my-project run_benchmark_array.sbatch`) — see
"Weights & Biases" below.

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

The default `--time=06:00:00` in the sbatch covers the worst case (`default`,
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

The sbatch sets `WANDB_MODE=offline` (compute nodes are usually offline) —
runs are written locally under `<repo_root>/wandb/offline-run-*` and print
their own `wandb sync <path>` command in the job's `.out` log. From a login
node afterward (needs network + `wandb login` once, see step 1), sync
everything from a job at once:

```bash
wandb sync wandb/offline-run-*
```

Project defaults to `zeroshot-cf-benchmark`; override per-submission with
`sbatch --export=ALL,WANDB_PROJECT=my-project run_benchmark_array.sbatch`. If
you don't want W&B at all, drop `--wandb-project "$WANDB_PROJECT"` from the
sbatch's `run_full_benchmark.py` call and add `--no-wandb` instead.
