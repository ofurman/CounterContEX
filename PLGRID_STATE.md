# CounterContEX beam search on PLGrid — state as of 2026-07-30

Goal: run `experiments/zeroshot_cf/exp4_beam_search.py` (TabPFN beam-search
counterfactuals) on PLGrid Helios GH200, because there are no hours elsewhere.

Grant `plgcountercontex-gpu-gh200`, storage group `plggcfsgenwro`. Cluster
gotchas: `~/.claude/skills/plgrid-run/references/helios-field-notes.md`.

## Code state — IMPORTANT

The work lives in the worktree `/Users/lukasz/genwro/CounterContEX-beam`, branch
`lukasz/zeroshot-cf-beam-search`, and is **UNCOMMITTED**. Nothing is pushed. The
cluster copy was synced from the working tree, not from a git ref.

Modified/untracked and essential:
`beam_search.py` (vectorized prune/select), `exp4_beam_search.py` (chunking, array
persistence, Law wiring), `metrics_harness.py` (standard CF metrics), `data.py`,
`recompute_metrics.py` (new), plus `.gitignore` and result CSVs.

## Cluster state

`~/projects/countercontex/plgrid/` — written by a subagent, modelled on
`universal-diffsae/plgrid/`: `00_setup_env.sbatch`, `10_smoke.sbatch`,
`20_beam_run.sbatch`, `_common.sh`, `plg-config.sh`, `submit.sh`,
`sync-to-plgrid.sh`, `README.md`.

**Environment approach differs from universal-diffsae, deliberately:** uv +
CPython 3.11 + **PyPI torch** (which bundles its own CUDA runtime), rather than
the ML-bundle wheelhouse. That sidesteps the torchvision ABI problem entirely,
since torch and torchvision then come from the same source and match. Costs ~5×
the install footprint. Env currently 5.1 GB at
`$PLG_GROUPS_STORAGE/plggcfsgenwro/$USER/countercontex/envs/beam` with a working
`bin/python`.

**If FID or torchvision is ever needed in universal-diffsae, this is the template.**

## Job history

| Job | State | Note |
|---|---|---|
| 20043781 | TIMEOUT 01:00:07 | root cause below |
| 20141858 | queued (Priority) | setup, resubmitted with fixes, 2 h |
| 20141859 | queued (Dependency) | smoke, `afterok:20141858` |

**No beam-search run has ever executed on the cluster.** There are no cluster
results.

### Why 20043781 timed out

Not TabPFN, not the workload — it reached step 4 of 5 (vendored `ce-library`
installed) and was killed entering verification. `_common.sh` had:

```
UV_CACHE_DIR="${STORE}/cache/uv"    # STORE=$SCRATCH → /net/scratch/hscra
VENV="${HEAVY}/envs/beam"           # HEAVY=$PLG_GROUPS_STORAGE → /net/storage/pr3
```

Two different Lustre filesystems, so uv printed `Failed to hardlink files;
falling back to full copy` and byte-copied every package. Multi-GB `nvidia-*`
CUDA wheels then ate the hour.

**Fixed:** `UV_CACHE_DIR` → `${HEAVY}/cache/uv` (both now on `/net/storage/pr3`,
verified with `df -T`), walltime 1 h → 2 h. Should be much faster now: venv
mostly built, uv cache warm.

## Local results (from a prior session, not the cluster)

Saved arrays in `experiments/zeroshot_cf/results/arrays/`:

| cell | n | status |
|---|---|---|
| `exp4_law_frozen_cfs.npz` | 444 | full split |
| `exp4_heloc_frozen_cfs.npz` | 2092 | full split |
| `exp4_law_fromscratch_cfs.npz` | 20 | smoke only |
| HELOC from-scratch | — | **never run** |

Law has **no immutable features**, so `freeze_immutable` filters nothing and both
regimes are the identical code path — the 444-row frozen run covers Set 1 and
Set 2. **HELOC from-scratch is the only genuinely missing cell.**

Scored metrics:

| metric | HELOC frozen (2092) | Law (444) |
|---|---|---|
| validity | 0.376 | 1.000 |
| LOF | **6.53e6** | **8.72** |
| proximity L2 / L1 | 0.561 / 1.437 | 1.128 / 1.855 |
| L2 continuous only | — | 0.367 |
| categorical change rate | — | 0.235 |
| L0 (features changed) | 10.76 of 17 | 4.41 of 13 |
| frac_oob | 0.0014 | 0.000 |
| true_actionability | 0.9986 | — |

`recompute_metrics.py` scores any saved cell without regenerating — generation is
the expensive step (~0.65 s/CF on MPS).

## The open scientific question

**HELOC frozen has LOF 6.5×10⁶ against Law's 8.72 — a ~750 000× gap — and
validity 0.376 against 1.000.** That is not a tuning difference.

Untested hypothesis worth checking first: HELOC is the *only* dataset where
freezing does anything (Law has no immutables), and the frozen HELOC n=30 smoke
scored validity 1.000 / LOF 1.004 in the *from-scratch* regime. So the failure
may be specific to the **conditional** regime — holding 6 immutables at their
factual values may leave a conditional distribution that contains little or no
target-class mass, so no beam can find a valid, on-manifold counterfactual.

**This has not been investigated.** The saved HELOC arrays (all 2092 rows) can
answer it locally with no cluster time. Do that before spending allocation on the
full grid.

Also note from the prior session: `true_actionability` 0.9986 rather than 1.0 is a
**metric artifact**, not a constraint violation — the scaler is fit on train, 6 of
2092 test rows fall outside [0,1], and clipping before metrics nudges 3 rows'
immutables off their factual values. Unclipped immutables are exactly equal.

And frozen HELOC validity moved **0.133 (n=30) → 0.233 → 0.376 (n=2092)**, so any
n=30 number in the report should be treated as noise, not signal.

## Next commands

```bash
# check the resubmitted setup + smoke
ssh helios 'sacct -j 20141858,20141859 --format=JobID,JobName%12,State,Elapsed,ExitCode -X'
ssh helios 'tail -40 ~/projects/countercontex/logs/cx_setup-20141858.out'

# then the beam run (12 h walltime as configured)
ssh helios 'cd projects/countercontex && bash plgrid/submit.sh plgrid/20_beam_run.sbatch'
```

## Cautions

- **`--chunk-size` changes results.** TabPFN's predictions depend on batch
  composition: chunk=40 vs chunk=7 differed by ~1.0 on a one-hot column. Default
  is 4096 so a whole target class is one call, preserving earlier semantics. Hold
  it fixed across cells you compare.
- **Peak RSS 13 GB** at that chunk size — size `--mem` accordingly.
- **`HF_HUB_OFFLINE=1` with pre-staged TabPFN checkpoints** is required.
- `tabpfn_extensions` is absent from the local venv, so 4 test modules fail to
  import locally. Pre-existing, unrelated to the beam work.
- **Python 3.13 vs 3.11:** the cluster env uses 3.11 via uv. Do not "fix" it to
  match universal-diffsae's 3.13 — the wheelhouse torch is cp313 and that is
  precisely the path this project avoided.
- The `results/` directory is in a mixed state: some CSVs are from n=30/n=100
  smokes, `exp4_summary.md` is stale. Do not read it as current.
