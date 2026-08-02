# Running the zero-shot CF beam search on PLGrid Helios (GH200)

Everything here targets **Helios only**. The `plgcountercontex` grant has no
Athena allocation (`hpc-grants` is empty there), so Athena is not an option.

## Quick start

```bash
# from the CounterContEX-beam worktree, on the workstation
DRY=1 bash plgrid/sync-to-plgrid.sh     # inspect
bash plgrid/sync-to-plgrid.sh

ssh helios
cd projects/countercontex
bash plgrid/submit.sh --test-only plgrid/00_setup_env.sbatch
bash plgrid/submit.sh plgrid/00_setup_env.sbatch     # ~15 min, builds the venv
# verify: sacct -j <id> --format=JobID,State,ExitCode  AND read logs/cx_setup-<id>.out
bash plgrid/submit.sh plgrid/10_smoke.sbatch         # 20 points, must print SMOKE PASS
bash plgrid/submit.sh plgrid/20_beam_run.sbatch      # the real run
```

A job ID is not evidence of anything. Check `State=COMPLETED` **and**
`ExitCode=0:0` **and** the log before escalating to the next stage.

## Verified cluster facts

Established live, 2026-07-26 → 2026-07-28. Do not re-derive.

| Thing | Value |
|---|---|
| Login | `ssh helios` → `plgllenkiewicz@login01.helios.cyfronet.pl` |
| Grant | `plgcountercontex`, active to 2027-03-23 |
| Allocation | `plgcountercontex-gpu-gh200`, 10 000 h granted, 932.94 h consumed |
| Partition | `plgrid-gpu-gh200`, 110 nodes, `gpu:4`/node, max walltime `2-00:00:00` |
| GPU request | `--gres=gpu:1` — no type prefix |
| Storage group | `plggcfsgenwro`, 250 GB, **shared by four people**, 32.5 GiB used |
| Group layout | `$PLG_GROUPS_STORAGE/<group>/<login>/<project>` — **not** `.../users/<login>/` |
| `$HOME` | 10 GiB quota, code only |
| `$SCRATCH` | `/net/scratch/hscra/plgrid/plgllenkiewicz`, 12 TiB, personal |
| Modules | `ML-bundle/25.10` (also 24.06a, 25.04) |
| Architecture | login nodes **x86_64**, GH200 compute nodes **aarch64** |

The grant holds *only* a gpu-gh200 allocation: `sbatch --test-only` rejects the
`plgrid`, `plgrid-long` and `cpu` partitions with "Invalid account or
account/partition combination".

`$SCRATCH` retention: files older than 30 days, or everything after 30 days
with no submitted job. Fine for caches and sandboxes, never the only copy of a
result.

### Compute nodes have outbound network — verified

This was the big open question and it is settled. Job **19930987**
(`cf-env-setup`, a sibling project on this same partition and grant) ran on an
aarch64 compute node and successfully did `pip install uv`,
`uv python install 3.11`, and a full `uv sync` that pulled **torch
2.11.0+cu130** and scikit-learn 1.8.0 from PyPI. It left a 7.5 GB uv cache
behind. PyPI is reachable from compute nodes.

So `00_setup_env.sbatch` installs from PyPI on the node, and the offline
machinery below is about *pinning* the model weights, not about working around
a firewall.

### torch does **not** come from ML-bundle

`ML-bundle/25.10` is a CUDA/compiler stack (foss, CUDA 12.9.1, cuDNN, NCCL,
magma, HDF5). Its only torch is a local aarch64 wheelhouse at
`/net/software/aarch64/el9/wheels/ML-bundle/25.10`, and every wheel there is
**cp313** — using it would pin the project to Python 3.13 and to torch 2.9/2.13.

We do not use it. PyPI ships CUDA-enabled aarch64 torch wheels for this
platform, `uv` fetches its own interpreter, and `uv.lock` is respected exactly.
`python -m venv --system-site-packages` to "inherit torch from the module" does
**not** work on Helios — the universal-diffsae setup job 20042983 failed in six
seconds doing exactly that.

`ML-bundle` also sets `OMP_NUM_THREADS=1`. `_common.sh` overrides it with
`$SLURM_CPUS_PER_TASK`, pinned so timings stay comparable between cells.

## TabPFN checkpoint staging — the thing most likely to break

**Where the weights come from.** They are already on the workstation, in
`experiments/zeroshot_cf/models/`, downloaded during earlier local runs:

| File | Size |
|---|---|
| `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` | 27.7 MiB |
| `tabpfn-v2-regressor.ckpt` | 42.3 MiB |

This directory is **gitignored**, so a `git clone` on the cluster produces a
checkout that cannot run. `sync-to-plgrid.sh` rsyncs the working tree and hard-
fails if either file is absent.

**Why v2 and not the default.** TabPFN 8.0.8 defaults to `ModelVersion.V3`.
v2.5, v2.6 and v3 live in *gated* HuggingFace repos (`Prior-Labs/tabpfn_2_5`,
`tabpfn_2_6`, `tabpfn_3`); `_download_model` routes those through
`tabpfn.browser_auth.ensure_license_accepted()`, an interactive browser license
flow that cannot complete on a compute node. The v2 repos
(`Prior-Labs/TabPFN-v2-clf`, `Prior-Labs/TabPFN-v2-reg`) are ungated.
`experiments/zeroshot_cf/checkpoints.py` pins v2 for exactly this reason.

**How `HF_HUB_OFFLINE=1` finds them.** It does not — nothing is looked up.
`_common.sh` exports `TABPFN_LOCAL_CACHE=$PROJECT/experiments/zeroshot_cf/models`.
`checkpoints.py` reads that, exports `TABPFN_MODEL_CACHE_DIR` from it (the real
setting, `TabPFNSettings.model_cache_dir` via `env_prefix="TABPFN_"`), and then
passes an **explicit absolute `model_path=`** to both `TabPFNClassifier` and
`TabPFNRegressor`. With an explicit path, `_resolve_model_version()` infers the
version from the filename — neither name contains `v2.5`/`v2.6`/`v3`, so it
falls through to `ModelVersion.V2`.

`HF_HUB_OFFLINE=1` is the safety net, not the mechanism: `tabpfn/base.py`
hard-codes `download_if_not_exists=True`, so a missing or misnamed checkpoint
would otherwise quietly try to reach huggingface.co. Offline turns that into an
immediate loud failure. `_common.sh` additionally refuses to start a job if
either `.ckpt` is missing, so the failure happens in the first second rather
than after the model has warmed up.

Both estimators are constructed by `get_models()` even though the beam search
only uses the regressor, so **both** checkpoints must be present.

## The other thing git will not give you: `cel`

`experiments/zeroshot_cf/data.py` imports `cel`, which is the vendored
`ofurman/counterfactuals` repo living at
`experiments/zeroshot_cf/vendor/counterfactuals/` — also gitignored. It carries
the datasets (`data/heloc.csv`, `data/law.csv`) and their configs, and
`FileDataset` resolves those relative to `cel/__file__`, not cwd.

`vendor_setup.py` normally bootstraps it with `git clone`. We rsync it instead.
It is installed **editable, `--no-deps`**: its declared dependencies pull
`alibi[tensorflow]`, `cvxpy`, `mlflow`, `dice-ml` and friends, none of which the
beam search touches. The minimal transitive set that *is* installed is
`cel-nflows torchdiffeq UMNN omegaconf hydra-core`, matching `vendor_setup.py`.

Do **not** copy the macOS `.venv` — it contains `mlx`, `mlx_metal` and
`*-darwin.so`, and the `cel` editable finder hard-codes the Mac absolute path.

## Storage budget

The shared 250 GB group quota is treated as precious.

| Location | Contents | Size |
|---|---|---|
| `$HOME/projects/countercontex` | code, vendored `cel`, `.ckpt` weights, results | ~150 MB |
| group storage `.../countercontex/envs/beam` | the venv (torch + CUDA libs) | ~4–5 GB |
| group storage `.../countercontex/results` | durable mirror of every result | < 5 MB |
| `$SCRATCH/countercontex/cache` | uv / pip / torch / HF caches | up to ~8 GB |
| `$SCRATCH/countercontex/smoke-<jobid>` | throwaway smoke sandboxes | ~150 MB each |

The uv cache is deliberately on `$SCRATCH` — it reached 7.5 GB for a sibling
project, and it is fully reproducible. Only the venv, which has to survive
`$SCRATCH`'s 30-day retention, sits in group storage.

## What actually needs computing

| Cell | Points | Status |
|---|---|---|
| Law / frozen | 444 | done locally |
| Law / from-scratch | 444 | done locally — Law has no immutable features, so it is the identical code path |
| HELOC / frozen | 2092 | done locally |
| **HELOC / from-scratch** | **2092** | **the only missing cell** |

`20_beam_run.sbatch` therefore defaults to `CELLS="heloc:fromscratch"`. The full
grid is one env var away.

## Resuming

Three independent mechanisms, in increasing order of bluntness.

1. **Per-cell skip.** `20_beam_run.sbatch` skips any cell whose
   `results/arrays/exp4_<dataset>_<tag>_cfs.npz` already exists. `FORCE=1`
   disables it. exp4 itself has *no* resume logic — it recomputes and
   overwrites every requested cell — so this guard is the only thing standing
   between a requeue and paying twice.
2. **Durable mirror.** Results are rsynced to
   `<group storage>/countercontex/results` after *every* cell, and pulled back
   at the start of every job. A job killed in cell 3 keeps cells 1 and 2.
3. **Rescoring without regenerating.** `recompute_metrics.py` reads every saved
   `.npz` and rewrites `exp4_recomputed_metrics.csv`. It needs `cel` and the
   discriminator but **not** TabPFN and not a GPU. Use it to add a metric or to
   salvage an interrupted run.

`exp4_summary.md` is rewritten from only the cells of the invocation that
produced it, so after a resumed run it under-reports. **Trust
`exp4_recomputed_metrics.csv`**, which `20_beam_run.sbatch` regenerates at the
end from everything on disk.

## Reproducibility caveats

- **Not chunk-invariant.** TabPFN's predictions depend on the composition of the
  predict batch. Measured: `chunk=40` vs `chunk=7` differed by ~1.0 on a one-hot
  column. `--chunk-size` is pinned to 4096 (one call per class) in every job and
  must stay identical across cells that appear in the same table. The
  `--chunk-size` help text in the script still claims chunk-invariance; it is
  stale, the module-level comment is correct.
- **Cluster numbers will not bit-match the local CSVs.** The finished cells were
  produced on MPS; these run on CUDA. Expect small numeric drift.
- **Python version.** The venv is built with `PY_VERSION=3.11` (set in
  `plg-config.sh`), the version empirically proven to resolve on aarch64 here,
  whereas the local `.venv` is 3.13. `uv.lock` pins exact package versions in
  both cases, so the difference is confined to the few markers that branch on
  the interpreter. Override `PY_VERSION=3.13` if exact parity matters more than
  wheel-availability risk.
- **Seeds are not CLI-configurable.** `BeamConfig(random_state=42 + target_cls)`;
  the discriminator and the 80/20 stratified split both use `random_state=42`.

## Files

| File | Purpose |
|---|---|
| `plg-config.sh` | account, partition, group, project, Python version. The only place grant IDs live. |
| `_common.sh` | paths, caches, module load, venv activation, TabPFN offline env, results mirror. Sourced by every job. |
| `submit.sh` | injects `--account`/`--partition` at submit time so no stale grant ID is ever committed in an `#SBATCH` header. |
| `sync-to-plgrid.sh` | working-tree → cluster. `DRY=1` to inspect. |
| `00_setup_env.sbatch` | builds the venv on a GPU node. |
| `10_smoke.sbatch` | 20-point run in a `$SCRATCH` sandbox. |
| `20_beam_run.sbatch` | the real run, with per-cell resume. |

`submit.sh` passes extra flags through, so dependency chaining works:

```bash
bash plgrid/submit.sh --dependency=afterok:12345 plgrid/20_beam_run.sbatch
```

## Gotchas that cost time

- The smoke **must not** run in place. exp4 derives its output directory from
  `__file__` and has no `--results-dir`, so an in-place 20-point smoke would
  overwrite the real `exp4_heloc_fromscratch_cfs.npz`. `10_smoke.sbatch` copies
  the repo to `$SCRATCH` first.
- Cells run **sequentially in one job**, not as a Slurm array. Parallel array
  tasks sharing a checkout would race on `exp4_summary.md` and on each other's
  metrics CSVs.
- `uv sync` prunes anything not in `uv.lock`, which includes `cel`. Re-run
  `00_setup_env.sbatch` (which reinstalls `cel` afterwards) rather than calling
  `uv sync` by hand.
- Never sync `.env` or any credential. `pydantic-settings` in TabPFN reads a
  `.env` file if one is present; `sync-to-plgrid.sh` excludes it explicitly.
