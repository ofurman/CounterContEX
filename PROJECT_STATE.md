# CounterContEX beam-search CF — project state

Last updated 2026-08-02 (Exp 7 sweep). Branch `lukasz/zeroshot-cf-beam-search`,
checked out at the repo root `/Users/lukasz/genwro/CounterContEX`.

Zero-shot counterfactual generation: beam search over TabPFN's conditional
distribution, no CF-specific training. Evaluated against the metric set used in the
`counterfactuals` repo (`/Users/lukasz/genwro/counterfactuals`).

## Experiment grid

Two regimes that differ only in whether immutable features are held at their factual
values:

All four cells now exist on **one backend** (Helios GH200, CUDA), regenerated at
defaults as the Exp-7 `base` runs. The earlier local-MPS arrays are kept as
`results/arrays/exp4_*_cfs.npz`; the cluster ones are
`results/arrays/sweep/exp4_*__base_cfs.npz`.

| cell | n | status | generated on |
|---|---|---|---|
| heloc / frozen | 2092 | done, full split | GH200, job 20359809, 84 s |
| heloc / fromscratch | 2092 | done, full split | GH200, job 20359811, 85 s |
| law / frozen | 444 | done, full split | GH200, job 20359810, 75 s |
| law / fromscratch | 444 | done, full split | GH200, job 20359812, 74 s |

Grid complete. Law has **no immutable features** — `get_actionable_immutable('law')`
returns an empty immutable list — so `freeze_immutable` masks nothing.

**Correction (2026-08-02): the two Law regimes ARE numerically identical.** The
earlier entry here recorded them as differing (LOF 9.83 vs 8.72, L1 0.7314 vs 0.7445,
ε-sparsity 0.9047 vs 0.9234) and flagged the comparison as confounded with hardware.
That confound was the entire effect. Run on one backend at identical settings, the
two regimes agree to full float precision on every metric and their `X_cf` arrays are
**bitwise equal** (`max |diff| = 0.0`). Law has no immutables, so the two code paths
are genuinely the same computation. This closes what was open item 2.

Consequence worth keeping: TabPFN's outputs are **not bitwise portable across
backends**, and the shift is not negligible — HELOC frozen validity is 0.3762 on MPS
and 0.3877 on CUDA at identical settings. Never mix backends within one table.

## Results

Standard metrics, computed by `exp4_metrics_table.py` using the vendored `cel`
registry's own metric classes (`results/exp4_metrics_table.csv`):

| dataset | set | n | validity | coverage | L1 | L2 | sparsity | ε-spars | LOF | IsoForest |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heloc | fromscratch | 2092 | 1.0000 | 1.0000 | 2.6234 | 0.8291 | 1.0000 | 0.7668 | 1.0087 | 0.1125 |
| heloc | frozen | 2092 | 0.3762 | 1.0000 | 1.4372 | 0.5607 | 0.7392 | 0.4456 | 6528790.5138 | 0.0777 |
| law | fromscratch | 444 | 1.0000 | 1.0000 | 0.7314 | 0.6912 | 1.0000 | 0.9047 | 9.8291 | 0.1142 |
| law | frozen | 444 | 1.0000 | 1.0000 | 0.7445 | 0.7001 | 1.0000 | 0.9234 | 8.7217 | 0.1116 |

LOF here is the registry's mean over **all** rows; that is where HELOC frozen's 6.5e6
comes from (see finding 3). `sparsity` is exact-equality and saturates at 1.0 for
continuously-generated CFs, so it carries no information — use ε-sparsity.

Reference (dicoflex) formulas, valid-CFs-only with median-log LOF
(`results/exp4_reference_metrics.json`, see `results/exp4_reference_report.md`):

| cell | validity | prox. L1 cont. | ε-sparsity | LOF median-log |
|---|---:|---:|---:|---:|
| heloc / fromscratch | 1.000 | 0.1141 | 0.7668 | -0.0136 |
| heloc / frozen | 0.376 | 0.0625 | 0.4633 | 0.0345 |
| law / fromscratch | 1.000 | 0.1676 | 0.9047 | 2.5272 |
| law / frozen | 1.000 | 0.1866 | 0.9234 | 2.3555 |

### Cluster-defaults baseline — use this table, not the one above

Same code, same settings, all four cells on the GH200. This is the baseline every
Exp-7 sweep number is compared against (`results/exp7_sweep_metrics.csv`, `run_id`
= `base`), and the one to quote going forward:

| cell | n | validity | prox. L1 cont. | ε-sparsity | LOF median-log |
|---|---:|---:|---:|---:|---:|
| heloc / fromscratch | 2092 | 1.0000 | 0.1141 | 0.7668 | -0.0136 |
| heloc / frozen | 2092 | **0.3877** | 0.0630 | 0.4642 | 0.0349 |
| law / fromscratch | 444 | 1.0000 | 0.1676 | 0.9047 | 2.5272 |
| law / frozen | 444 | 1.0000 | 0.1676 | 0.9047 | 2.5272 |

## Experiment 7 — beam-search hyperparameter sweep

37 runs on the GH200 (0.25 GPU-hours total), five axes one-at-a-time around the
`BeamConfig` defaults plus six targeted combinations. Full write-up:
`results/exp7_sweep_report.html`; long-format table: `results/exp7_sweep_metrics.csv`.

Sensitivity on `heloc/frozen`, by validity span across the axis:

| axis | levels | min | max | best config | Δ vs base |
|---|---|---:|---:|---|---:|
| `lambda_actionable` | 0, 0.1, **1**, 3, 10 | 0.1902 | 0.4168 | λ=0 | +0.0291 |
| `max_context` | 128, **256**, 512, 1024, 2048, 4096 | 0.3451 | 0.4622 | 1024 | +0.0745 |
| `beam_width` | 4, **8**, 16, 32 | 0.3786 | 0.3944 | 4 | +0.0067 |
| `n_candidates` | 4, **6**, 10, 16 | 0.3819 | 0.3934 | 4 | +0.0057 |
| `candidate_probs` | **interior**, tail | 0.3877 | 0.3881 | tail | +0.0005 |

Best configuration overall: `ctx1024-lam0` (max_context 1024, λ=0) at **0.4909**,
+0.1032 over the cluster baseline.

**These are test-split numbers.** The sweep is a sensitivity analysis; selecting a
configuration by this table and reporting its validity as a headline would be tuning
on test. Any specific winner needs a held-out re-evaluation before it carries a
published number.

## Findings

### 1. Perfect validity is only reached by dropping the constraint

HELOC from-scratch hits validity 1.000, but by rewriting essentially the whole row:
`l0_count_mean` 17.96 of 23, `true_actionability` 0.0, `immutable_drift_mean` 0.112.
It changes all six immutables, so it is a sample from the target class rather than an
actionable counterfactual. Law's 1.000 does not test the constraint at all (no
immutables). The one cell that tests constrained flipping — HELOC frozen — reaches
**0.376**.

### 2. The 62% of misses were all reachable — this is a search failure

The discriminator is sklearn `LogisticRegression`, so the ceiling is closed-form. Pin
the 6 immutables at their factual values, let the 17 actionable features range over
[0,1], and take the extreme achievable logit (each free feature to 1 or 0 by the sign
of its weight):

```
target REACHABLE in principle        2092  (1.0000)
target actually reached (validity)    787  (0.3762)
reachable but MISSED                 1305
hard infeasible                         0  (0.0000)
```

**Zero rows are infeasible.** The hypothesis that freezing immutables leaves a
conditional with no target-class mass is dead in its geometric form.

The misses are not near-misses either. Median margin runs from -0.903 at the factual
to +20.373 at the best achievable point, and the generated CF travels a median **0.3%**
of that (mean 1.9%). **496 of the 1305 moved the wrong way** — margin worse than the
factual's. Successes and failures start from nearly the same place (median factual
margin -0.695 vs -0.903), so the outcome is not explained by failures being harder.

Caveat: beam search does not optimise the LR logit — it samples TabPFN's conditional to
stay on-manifold, with LR only as the scoring oracle. What this rules out is the
infeasibility explanation. What remains open is whether TabPFN's conditional given the
6 pinned features simply has no target-side mass, or the selector is not pushing toward
validity. The 496 backwards moves point at the second.

### 3. The LOF 6.5e6 result was an artifact

Previously treated as the project's main open question (a ~750,000x gap against Law).
Under valid-only + median-log it is **0.0345** against from-scratch's **-0.0136**. The
gap does not exist. Two compounding causes, both confirmed:

| aggregation over HELOC frozen | value |
|---|---|
| mean, all 2092 CFs | 6.53e6 |
| mean, 787 valid CFs | 1.06 |
| median-log, 787 valid CFs | 0.0345 |

Exactly 115 rows have LOF > 1e3 (max 1.19e8), and they are byte-identical to HELOC's
115-row duplicate block — the **all-zeros row**, the MinMax image of the `-9` "no
record" code, which appears 473 times in `X_train`. A query point on a 473-fold
duplicated training point has k-NN distance ~0, so the LOF ratio diverges numerically.
Freezing pins those rows there; from-scratch lets them escape.

Frozen validity of 0.376 is unaffected and stands.

### 5. More search makes validity worse (Exp 7)

Validity on `heloc/frozen` is **strictly decreasing** in beam width:

| beam_width | 4 | 8 (default) | 16 | 32 |
|---|---:|---:|---:|---:|
| validity | 0.3944 | 0.3877 | 0.3834 | 0.3786 |

Widening the beam is the one change that unambiguously buys more search, and it costs
validity every time. The mechanism follows from where validity enters: beams are
ranked at every step by cumulative `log-density − λ·proximity`, and whether a partial
path will flip the class is not consulted until the terminal rerank among completed
beams. A wider beam fills with paths that score better on density and proximity,
crowding out the lower-scoring paths that would have reached the target.

This resolves the open half of finding 2. That finding could not separate "TabPFN's
conditional given the pinned immutables has no target-side mass" from "the selector
is not pushing toward validity". A monotone *penalty* for more search points at the
second. **The selector is the thing to fix**, not the search budget.

### 6. max_context has an interior optimum; the default is on the wrong side

| max_context | 128 | 256 (default) | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| validity | 0.3451 | 0.3877 | 0.4087 | **0.4622** | 0.3948 | 0.4187 |

Rises to a peak at 1024 and falls back. The peak is interior, so it is a real optimum
rather than an artifact of where the grid was truncated — the OFAT grid originally
stopped at 1024 and read as "monotone, not yet saturated"; extending it in the
targeted-combo round corrected that. The single cheapest available improvement is
raising `max_context` from 256 to 1024.

Also: on `law/frozen`, `max_context=128` **crashes** inside TabPFN
(`assert self.num_bars > 1`). A 128-row context subsample can leave one of Law's rare
one-hot categories constant, and the bar distribution degenerates. Not a bug in this
code — a genuine infeasibility of that level on that dataset.

### 7. Tail candidate quantiles do nothing — a clean negative result

The hypothesis (from the Exp-7 goal) was that the default interior quantile grid hugs
the mode, holding each step to a tiny move, and that tail quantiles would allow the
larger moves finding 2 says are needed. They do not: 0.3877 → 0.3881, i.e. +0.0005.
**Step size is not the binding constraint.**

### 8. The 115 all-zeros HELOC rows are one query point, never flipped

They are byte-identical, so deterministic generation gives them an identical
counterfactual (verified byte-for-byte): one query point counted 115 times, not 115
independent attempts. Across all 21 `heloc/frozen` configurations, the validity
achieved among those rows is **0.0000** — no configuration ever flips it.

Excluding them raises validity by a near-constant +0.011 to +0.027 and leaves the
ranking of configurations **exactly unchanged** (Spearman ρ = 1.0000). They therefore
change no conclusion of the sweep, and both numbers are carried in the CSV
(`validity_target`, `validity_excl_allzero`). This does not settle open item 3 —
whether a missing-data sentinel belongs in an evaluation set is a decision about the
benchmark, not about the search.

### 9. Law's perfect validity is a property of the default λ, not of the method

`law/frozen` holds validity 1.000 across most of the sweep, but not unconditionally:
λ=3 gives 0.8378 and λ=10 gives 0.2297. Separately, `n_candidates=10` collapses to
**0.5113** while both neighbouring levels (4 and 16) stay at 1.000 — reproducible
(generation is deterministic) and unexplained. The axes are not smooth; a
configuration cannot be assumed safe because the levels around it are.

### 4. Relabelling makes one validity definition meaningless

`exp4_beam_search.py:144` sets `y_target = 1 - disc_model.predict(X_test)` — the target
is the flip of the model's own prediction. The `cel` registry defines validity as
`mean(y_cf_pred != y_test)` against the *true* label, which under relabelling reduces
algebraically to the discriminator's accuracy:

| dataset | disc. accuracy | registry `validity` |
|---|---|---|
| heloc | 0.7232 | 0.7232 |
| law | 0.7815 | 0.7815 |

Identical to four decimals. **Use `y_cf_pred == y_target`.** The reference repo's own
pipelines target `1 - y_test`, where the two coincide; the divergence is specific to
this project's protocol.

## Metric tooling

Four scorers, all reading `results/arrays/*.npz`, none needing a GPU:

| script | purpose |
|---|---|
| `recompute_metrics.py` | project's own harness, rescoring saved cells |
| `cel_standard_metrics.py` | vendored `cel` registry metrics, unmodified |
| `reference_metrics.py` | dicoflex Table-1 formulas, valid-only, median-log LOF |
| `exp4_metrics_table.py` | the standard table: L1, L2, validity, LOF, IsoForest, sparsity, ε-sparsity |
| `exp7_sweep_table.py` | scores every config-tagged sweep array under **both** conventions → one long CSV |
| `exp7_report.py` | renders `results/exp7_sweep_report.html` from that CSV (no hand-typed numbers) |

`eps_sparsity` is emitted by **both** scorers under **different formulas** — the
registry averages over all rows, the reference over valid rows only (0.4454 vs 0.4642
on cluster heloc/frozen). `exp7_sweep_table.py` keeps the reference version under the
bare name and namespaces the registry's as `registry__eps_sparsity`; any other
name collision between the two scorers raises rather than silently keeping one.

### Running a sweep

`exp4_beam_search.py --run-id <slug>` moves every artifact into a sweep namespace
(`results/arrays/sweep/`, `results/sweep/`) and embeds the full resolved config, the
code commit and the Slurm job id inside the npz as `config_json`. Omitting `--run-id`
reproduces the original Exp-4 filenames exactly. `--candidate-probs` and
`--n-estimators` are now on the CLI. Driver: `plgrid/30_beam_sweep.sbatch` (one job
per cell, configs sequential within the job, resume guard on the config-tagged npz).
Pull back with `plgrid/pull-from-plgrid.sh` (verifies by SHA-256).

`reference_metrics.py` ports `cel/metrics/dicoflex_metrics.py` @ `b9715ef` on
`origin/ofurman/CFN_baselines`, which is not in that repo's working tree and had to be
read out of git history. Three of its formulas differ from the generic registry
(relative ε-sparsity, one-hot columns not groups, euclidean+hamming diversity).

Two reference columns are not reportable for this method and are guarded in code:
`sparsity_categorical` saturates at 1.0 on Law because beam search emits a continuous
relaxation (0% of categorical cells are exactly 0 or 1, some rows carry two columns
near 1 in one group), and `pairwise_diversity_mixed` needs K>1 CFs per factual where
this emits one.

## Cluster

Helios GH200, grant `plgcountercontex-gpu-gh200` (~9060 h of 10000 remaining, active to
2027-03-23), storage `plggcfsgenwro` at 81 of **500** GiB.

Environment is built and working: `envs/beam`, 5.1 GB, uv + CPython 3.11 + PyPI torch
(deliberately not the ML-bundle wheelhouse — sidesteps the torchvision ABI mismatch).
Setup (20141858) and smoke (20141859) both COMPLETED. Storage checked 2026-08-02:
`plggcfsgenwro` at 83 of 500 GiB, `$HOME` 279 MiB of 10 GiB.

The GH200 is far faster than the estimate this work was budgeted against: full-split
`heloc/frozen` is **~85 s**, not the ~15 min assumed. The whole 37-run Exp-7 sweep
cost **0.25 GPU-hours**. Budget generously; it will not be the constraint.

Two traps found the hard way in `30_beam_sweep.sbatch`, both now fixed and worth not
relearning:

- **`srun` consumes stdin.** A `while read` loop fed by a here-string lost its
  remaining input to the first `srun`: the sweep ran one config, printed "sweep
  complete" and exited 0. Read the config list on FD 3 and give `srun </dev/null`.
  The job now also asserts it visited every config.
- **One bad config killed the job.** Under `set -e` a crashing config aborted the
  remaining ones. Failures are now recorded and skipped, with a non-zero exit at the
  end so a partial sweep cannot read as COMPLETED in `sacct`.

Two maintenance items, neither blocking:

- `envs/beam/bin/python` is a **symlink into `$SCRATCH`** (`UV_PYTHON_INSTALL_DIR` is on
  `$STORE` in `_common.sh`). With 30-day retention the group-storage venv gets a
  dangling interpreter around late August. Verified resolving 2026-08-02.
- The warm 5.1 GB uv cache still sits on `$SCRATCH` while `UV_CACHE_DIR` now points at
  `$HEAVY`. Harmless while the venv is complete; see the corrected root cause in
  `PLGRID_STATE.md`.

## Open issues

**The 115 all-zeros HELOC rows are a missing-data code, not records.** They are 5.5% of
the test split, they break LOF outright, and they affect every metric computed on
HELOC. Decide whether they belong in the evaluation set. Finding 8 narrows the
question: they are one query point counted 115 times, are never flipped by any
configuration, and shift validity by a near-constant offset that leaves the sweep's
conclusions and config ranking untouched.

**Provenance.** Cluster runs are rsynced from the working tree, not from a git ref.
`sync-to-plgrid.sh` now stamps the working-tree state into `.plgrid-commit`, which
travels with the sync and is embedded in every generated npz (`-dirty` suffix when
there are uncommitted edits). Still push the branch before syncing — the stamp
records *which* tree, not that the tree is fetchable.

**`--chunk-size` changes results.** TabPFN predictions depend on batch composition
(chunk=40 vs chunk=7 differed by ~1.0 on a one-hot column). Pinned at 4096 so a whole
target class is one call. Hold it fixed across any cells compared in one table.

**Local test suite.** `test_beam_search.py` 9/9 and `test_sweep_config.py` 24/24.
Four modules fail to *import* (`tabpfn_extensions` absent locally) and
`test_context_ablation.py` has 2 failures — both pre-existing and unrelated
(reproduced at `6d4c117`, before any Exp-7 change).

## Next

1. **Fix the selector, not the search.** Findings 5 and 7 localise the failure: more
   beam width hurts, larger candidate steps do nothing, and every missed flip is
   reachable in closed form. Validity enters only at the terminal rerank, so the beam
   spends its whole budget on `log-density − λ·proximity`. The concrete change is to
   put validity (or the discriminator margin) into the *per-step* score, not just the
   final rerank — e.g. rank partial beams by expected margin improvement, or keep a
   validity-diverse subset at each prune. This is the main open work.
2. ~~Rerun `law/frozen` on the cluster to separate the regime difference from the
   MPS-vs-CUDA confound~~ — **done**. All four cells are on the GH200; the two Law
   regimes are bitwise identical and the earlier difference was entirely the backend.
3. Decide the all-zeros row question before any number goes in a paper table. Finding
   8 narrows it: they change no conclusion of the sweep (ranking identical, Spearman
   ρ = 1.0000) and both numbers are in the CSV, so this is now a benchmark-definition
   decision rather than a blocker on the results.
4. If a swept configuration is to be reported as a result rather than as sensitivity,
   it needs a held-out re-evaluation — everything in Exp 7 is scored on the test split.
5. Explain the `law/frozen` `n_candidates=10` collapse to 0.5113 between two levels
   that both hold 1.000 (finding 9), or at minimum bound how often such
   discontinuities occur.
