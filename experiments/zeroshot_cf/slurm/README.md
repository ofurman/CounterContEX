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

## The greedy (exp4) variant

`run_greedy_benchmark_array.sbatch` / `run_greedy_benchmark.py` is the same
10-dataset array, same `BENCHMARK_CONFIG`, same conda env/account/partition —
but generates CFs with `exp4_greedy_cf.py`'s classifier-in-the-loop greedy
search (`greedy.py`) instead of `exp2`'s one-shot joint imputation. This is
the method that actually puts the validity-oracle classifier in the
generation loop (picking, and stopping on, a flip) rather than only using it
before/after generation — see the module docstring in
`run_greedy_benchmark.py` for the full architectural comparison.

**It's also far more expensive per point.** exp4's default selector calls
TabPFN once per remaining candidate feature at *every* greedy step — worst
case (budget exhausted, no early flip) is roughly `n_actionable²/2`
unbatched TabPFN calls per point, vs. exp2's one batched call for the whole
test set. One-point local calibration (RTX 4070 Ti): `admission` (6
actionable) exhausted its full budget without flipping in ~25s;
`german` (16 actionable) flipped after only 2 of 16 steps in ~26s — cost and
success are inversely correlated, so the datasets exp2 already struggled
with (gmc, lending-club, bank, default) are plausibly both the slowest
*and* the least likely to flip even here.

Because of that, `run_greedy_benchmark_array.sbatch` defaults to
`--max-test 16` (not 256) and `--n-repeats 1` (greedy is near-deterministic
at its default near-MAP temperature, so repeats don't buy much diversity
signal), and its `--time=03:00:00` is an **extrapolation from two
single-point samples**, not a calibrated number — recalibrate on `default`
specifically (the likely worst case) before trusting it, and before ever
raising `--max-test` past 16.

Runs log to the same `CounterContEX` W&B project, named
`greedy-<dataset>-<disc_type>-<metric_suite>-<selector>-tau<tau>-mt<max_test>-nr<n_repeats>-seed<seed>`
(the `greedy-` prefix plus selector/tau keep these distinguishable from the
`run_full_benchmark.py` runs already in that project) and write to
`results/greedy_benchmark_{l2c,dicoflex}_metrics.csv`, with extra columns
(`l0_count_mean`, `steps_mean`, `failure_rate`) not present in the exp2
benchmark's output.

## Full test set (`--max-test -1`)

Both `run_full_benchmark.py` and `run_greedy_benchmark.py` accept
`--max-test -1` for the dataset's full test split instead of a fixed cap —
already supported by `exp2_counterfactuals.py`/`exp4_greedy_cf.py`'s own
`generate_counterfactuals`, just not exercised by the two array jobs above.
`run_benchmark_array_fulltest.sbatch` / `run_greedy_benchmark_array_fulltest.sbatch`
run this, same `BENCHMARK_CONFIG`/cluster setup as the two scripts above,
`--n-repeats 5` kept for the diversity metric.

**Test-set sizes vary ~100x across these datasets** — this is the single
biggest thing to know before submitting either script:

| Dataset | Test rows |
|---|---:|
| admission | 100 |
| german | 200 |
| student | 226 |
| sba | 1,159 |
| default | 3,000 |
| adult | 9,045 |
| adult_dicoflex, bank, gmc, lending-club | 10,000 |

### exp2 full-test-set cost (extrapolated from the 256-pt calibration table
above — `robust_impute` chunks internally at 256 rows, so this is chunk-count
x per-chunk-cost, not a blind linear guess)

At `--n-repeats 5`: admission ~4min, student ~27min, german ~1.6h, sba
~5.4h, gmc ~13.3h, adult ~12h, adult_dicoflex ~16.7h, lending-club ~20h,
default ~45h (~1.9 days), **bank ~63h (~2.6 days) — the likely worst case**.
Serial total (all 10 on one GPU) ≈ 7.4 days; as this array, wall-clock is
bounded by `bank` alone. `--time=72:00:00` in the sbatch may not be enough
for `bank` on some clusters — check your actual job/QOS time limit
(`sacctmgr show qos` or equivalent) before submitting the full array.

### exp4 full-test-set cost — **6 of 10 datasets are very likely infeasible,
not just slow**

Extrapolating exp4's ~n_actionable²/2 unbatched-calls-per-point worst case
(calibrated at ~0.8s/call from the admission smoke test) to full test-set
size, at `--n-repeats 5`:

| Dataset | Worst case | Feasible at `--time=72:00:00`? |
|---|---:|:---|
| admission | ~2.3h | yes |
| student | ~13.8h | yes |
| german | ~30.2h | yes |
| sba | ~117.2h (4.9 days) | no — needs `--time` raised |
| gmc | ~611.1h (25.5 days) | **no** |
| adult | ~361.8h (15.1 days) | **no** |
| lending-club | ~866.7h (36.1 days) | **no** |
| adult_dicoflex | ~866.7h (36.1 days) | **no** |
| default | ~920.0h (38.3 days) | **no** |
| bank | ~1511.1h (63.0 days) | **no** |

Real cost will be lower wherever points flip before exhausting budget (the
`german` calibration point flipped in 2 of 16 steps, well under worst case)
— but the datasets with near-zero exp2 validity are exactly the ones
plausible to hit worst case often, so treat this table as realistic, not
merely pessimistic. `run_greedy_benchmark_array_fulltest.sbatch`'s
`--time=72:00:00` only actually covers german/admission/student/sba (array
indices 0, 2, 3, 9) — the other six tasks will be **killed by the time
limit before producing a result**, not just run slowly. Before submitting
the full `--array=0-9`, either:

- Submit only the feasible subset: `sbatch --array=0,2,3,9 run_greedy_benchmark_array_fulltest.sbatch`
- Cap `--max-test` for the other six specifically (edit the sbatch, or run
  them separately with e.g. `--max-test 500`)
- Accept partial/killed results for those six as a best-effort exploration
