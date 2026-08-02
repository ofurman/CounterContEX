# CounterContEX beam-search CF — project state

Last updated 2026-08-02. Branch `lukasz/zeroshot-cf-beam-search`, worktree
`/Users/lukasz/genwro/CounterContEX-beam`.

Zero-shot counterfactual generation: beam search over TabPFN's conditional
distribution, no CF-specific training. Evaluated against the metric set used in the
`counterfactuals` repo (`/Users/lukasz/genwro/counterfactuals`).

## Experiment grid

Two regimes that differ only in whether immutable features are held at their factual
values:

| cell | n | status | generated on |
|---|---|---|---|
| heloc / frozen | 2092 | done, full split | local (MPS) |
| heloc / fromscratch | 2092 | done, full split | local (MPS) |
| law / frozen | 444 | done, full split | local (MPS) |
| law / fromscratch | 444 | done, full split | Helios GH200, job 20356027, 86 s |

Grid complete. Law has **no immutable features** — `get_actionable_immutable('law')`
returns an empty immutable list — so `freeze_immutable` masks nothing.

**The two Law regimes were expected to be numerically identical and are not.** They
agree on validity (1.000 both) but differ throughout: LOF 9.83 vs 8.72, L1 0.7314 vs
0.7445, ε-sparsity 0.9047 vs 0.9234. The earlier `exp4_law_fromscratch_metrics.csv`
that matched `law_frozen` to 14 decimals was a stale artifact, not evidence.

The difference is **confounded with hardware** and cannot currently be attributed to
the regime: `law/frozen` was generated locally on MPS, `law/fromscratch` on a GH200
under CUDA, and TabPFN's outputs are not bitwise portable across backends. Rerunning
`law/frozen` on the cluster would separate the two. Until then, treat cross-hardware
rows in the same table with caution — this also applies to comparing the HELOC cells
(local) against `law/fromscratch` (cluster).

The informative comparison — HELOC frozen vs from-scratch — is unaffected: both were
generated locally on the same backend.

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
Setup (20141858) and smoke (20141859) both COMPLETED.

Two maintenance items, neither blocking:

- `envs/beam/bin/python` is a **symlink into `$SCRATCH`** (`UV_PYTHON_INSTALL_DIR` is on
  `$STORE` in `_common.sh`). With 30-day retention the group-storage venv gets a
  dangling interpreter around late August.
- The warm 5.1 GB uv cache still sits on `$SCRATCH` while `UV_CACHE_DIR` now points at
  `$HEAVY`. Harmless while the venv is complete; see the corrected root cause in
  `PLGRID_STATE.md`.

## Open issues

**The 115 all-zeros HELOC rows are a missing-data code, not records.** They are 5.5% of
the test split, they break LOF outright, and they affect every metric computed on
HELOC. Decide whether they belong in the evaluation set.

**Provenance.** Cluster runs are rsynced from the working tree, not from a git ref, so a
cluster result is not reproducible from a commit unless the branch is pushed first.

**`--chunk-size` changes results.** TabPFN predictions depend on batch composition
(chunk=40 vs chunk=7 differed by ~1.0 on a one-hot column). Pinned at 4096 so a whole
target class is one call. Hold it fixed across any cells compared in one table.

**Local test suite.** `test_beam_search.py` passes 9/9. Four other modules fail to
import because `tabpfn_extensions` is absent locally — pre-existing, unrelated.

## Next

1. Diagnose the frozen-HELOC search failure — finding 2 says the headroom is real and
   every missed flip was reachable.
2. Rerun `law/frozen` on the cluster to separate the regime difference from the
   MPS-vs-CUDA confound, and ideally move all four cells onto one backend.
3. Decide the all-zeros row question before any number goes in a paper table.
