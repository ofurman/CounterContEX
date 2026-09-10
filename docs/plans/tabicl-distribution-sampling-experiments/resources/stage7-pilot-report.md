# Stage 7 validation-pilot report

Date: 2026-09-10

These are engineering and mechanism results on 20 deterministically target-stratified
**validation** factuals per dataset. They are not held-out performance evidence and did not
select or remove any predeclared arm. All runs used the pinned TabICL checkpoints on
`ofurman@gx10-bdc5`. Source chronology was: E3 replay at `3c20063`; E12 cells 0-8 at `553aab6`
and cells 9-35 at `2509e7b`; E11 and E13 at `2509e7b`. The final `2509e7b` worktree reverified
all E12 bundles.

The user had approved the remote pilot scope in the task, but the plan-required repository
authorization entry was created at 06:23 CEST, after the first remote checkpoint/GPU artifact
at 05:47 CEST. Therefore the frozen pre-execution-record GATE did not pass and Stage 7 remains
BLOCKED on B-3 even though all scientific and technical artifacts below are complete.

## Authentication and execution

- The real-adapter E3 replay reproduced availability and candidates exactly (`max_abs_diff=0`).
  Its three HELOC cases had `1530/1530` raw conditional values different from the empirical
  backend and `313` differences erased by projection. The replay `COMPLETE` SHA-256 is
  `b6ed40ba1d5b7f13cc470fd35d0f05db7f0a79ad6756c55529323626e9610f43`.
- E12: all `36` resolved cells and `720` per-factual bundles passed their internal SHA-256
  inventories and a second exact matrix-driven verification. The sorted E12 child-inventory
  digest is `2da321ff11e69bd2932bd181b5a534d98b60fc38d443a6b754c40b48b1f3054d`.
- E11: all `144` cells passed strict aggregation and external inventory verification. Inventory
  SHA-256: `a65bdf43016ad35f85fe3d56746e257fa241d16400400c06018412fb4065659c`.
- E13: all `60` cells passed strict aggregation and external inventory verification. Inventory
  SHA-256: `5e12097eb370040cd4cbd4a8f6445aa7d48c627233fe9c7c173cf0dcebf658f0`.
- A fresh keyed-RNG witness reproduced twelve seed-42 draws byte-for-byte and changed them under
  seed 101. Draw-vector SHA-256 values were `8e6a176f...f3c4a0c` and
  `60761ef7...97d3c3`.

## E11: q representation and decoding

For quality metrics, sampling seeds are first averaged inside each factual, factuals are then
averaged inside each dataset, and the four datasets receive equal weight. Repeated factuals
across seeds are Monte Carlo repeats, not independent observations. `thr` is threshold-valid
success per requested slot and `cov` is coverage. `Gower` is evaluated only for returned
target-class candidates; `NN-5` and `actions` use all returned candidates. Runtime is the
separate operational mean of total lifecycle seconds per 20-factual cell.

| Numerical / categorical policy | thr | cov | Gower | NN-5 | actions | runtime s |
|---|---:|---:|---:|---:|---:|---:|
| mode-1 / iid-9 | .1625 | .7167 | .0602 | .0326 | 1.405 | 26.47 |
| grid-9 / iid-9 | .0750 | .7792 | .0661 | .0352 | 1.450 | 21.39 |
| iid-9 / iid-9 | .1042 | .8125 | .0575 | .0364 | 1.402 | 18.26 |
| top-k-9 / iid-9 | .1375 | .7333 | .0594 | .0324 | 1.362 | 50.75 |
| top-p-9 / iid-9 | .1042 | .7750 | .0625 | .0358 | 1.449 | 38.02 |
| grid-49 / iid-9 | .0750 | .8208 | .0570 | .0355 | 1.348 | 18.69 |
| iid-49 / iid-9 | .0875 | .8375 | .0516 | .0381 | 1.303 | 17.80 |
| top-k-49 / iid-9 | .1417 | .7333 | .0554 | .0353 | 1.319 | 53.54 |
| top-p-49 / iid-9 | .0750 | .7917 | .0651 | .0352 | 1.442 | 37.44 |
| iid-9 / greedy-1 | .0708 | .7792 | .0472 | .0352 | 1.297 | 21.31 |
| iid-9 / top-k-9 | .1042 | .8125 | .0568 | .0364 | 1.402 | 17.96 |
| iid-9 / top-p-9 | .0917 | .7917 | .0524 | .0375 | 1.373 | 19.78 |

For the predeclared comparisons, IID improved `thr` over the grid by `+0.0292` at B=9 and
`+0.0125` at B=49. Coverage changed by `+0.0333` and `+0.0167`, while grouped Gower improved by
`0.0086` and `0.0054`; NN-5 worsened by `0.0012` and `0.0026`. The effect was heterogeneous:
Give Me Some Credit and HELOC had zero threshold success in every arm, so this pilot does not
establish a general decoder advantage. At B=9, the entire mean threshold gain came from Bank
Marketing (`.0500 -> .1667`), with Lending Club unchanged at `.2500`; at B=49, Bank Marketing
was unchanged at `.0667` and Lending Club moved from `.2333` to `.2833`. Truncated numerical
policies consumed about `146k-225k`
TabICL rows per cell on average, versus about `4.5k-21.8k` for the IID/grid primary arms.

Across all E11 cells, M was not applicable and B was the declared numerical/categorical budget
shown in the table. The run recorded `1,972,789` raw and projected proposals, of which `642,253`
were no-ops and `202,055` duplicates, leaving `1,128,481` unique proposed rows. It made `13,926`
classifier calls over `1,131,361` rows and `148,287` TabICL calls over `9,512,053` rows.
The k=1 `search_depth` was always zero; validity search used a mean `3.483` steps, maximum `17`
(`2,252` sparse-valid and `628` validity-not-reached factual-runs).

## E12: q-to-classifier pushforward

Sampling seeds were averaged inside each factual before the following estimates. `D` is the
predeclared conditional absolute probability-change contrast: categorical minus numerical,
with equal action-unit weights. `n paired` is the number of factuals out of 20 for which both
feature types are estimable. Both component columns use exactly this paired subset, so their
difference equals `D`; numerical effects alone were estimable for all 20 factuals. HELOC has no
categorical action units and therefore has no paired contrast.

| Classifier | Dataset | n paired | categorical | numerical | D |
|---|---|---:|---:|---:|---:|
| LR | Bank Marketing | 20 | .0470 | .0581 | -.0111 |
| LR | Give Me Some Credit | 6 | .1508 | .0110 | +.1398 |
| LR | Lending Club | 19 | .1179 | .0050 | +.1129 |
| MLP | Bank Marketing | 20 | .0614 | .0641 | -.0027 |
| MLP | Give Me Some Credit | 5 | .1976 | .0101 | +.1875 |
| MLP | Lending Club | 20 | .1129 | .0018 | +.1111 |
| XGBoost | Bank Marketing | 20 | .0375 | .0552 | -.0177 |
| XGBoost | Give Me Some Credit | 7 | .1067 | .0340 | +.0728 |
| XGBoost | Lending Club | 19 | .0868 | .0096 | +.0772 |

The equal-dataset mean D was `+.0805` for LR, `+.0986` for MLP, and `+.0441` for XGBoost; each
classifier was positive on two of three mixed datasets and negative on Bank Marketing. The
target-stratified hierarchical bootstrap supported positive D most clearly for LR target 0
(`+.1170`, 95% interval `[+.0954,+.1509]`) and XGBoost target 0 (`+.0656`,
`[+.0419,+.0947]`); other target/classifier intervals were heterogeneous or crossed zero. High
categorical no-op mass and limited non-factual support made many Give Me Some Credit factuals
not estimable. Thus the pilot supports the categorical-impact hypothesis conditionally, not as a
universal property of feature type or classifier.

## E13: classifier-guided pi-beta sampling

Quality metrics use the same factual-first, equal-dataset aggregation and candidate-population
denominators as E11. ESS first averages action keys inside each factual before the same
seed/factual/dataset hierarchy. Runtime is an operational per-cell mean.

| Policy | thr | cov | Gower | NN-5 | actions | ESS | runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| beta=0 | .1083 | .8250 | .0589 | .0359 | 1.409 | 64.00 | 19.11 |
| beta=.5 | .0958 | .8292 | .0550 | .0370 | 1.358 | 63.73 | 19.49 |
| beta=1 | .1042 | .8250 | .0485 | .0363 | 1.278 | 63.18 | 18.90 |
| beta=2 | .1042 | .8292 | .0513 | .0373 | 1.300 | 62.08 | 18.33 |
| classifier-top-B | .2667 | .8625 | .0509 | .0385 | 1.269 | n/a | 16.27 |

The predeclared beta=1 contrast changed threshold success by `-0.0042` versus beta=0, with no
coverage change. It improved grouped Gower by `0.0104` and reduced action units by `0.131`, but
worsened NN-5 by `0.0005`. The weak ESS change shows that `p_clf^beta` only mildly reweighted
most M=64 pools. Dataset threshold values for beta=0 versus beta=1 were `.1833 -> .1500` on Bank
Marketing, `.0167 -> 0` on Give Me Some Credit, `0 -> 0` on HELOC, and `.2333 -> .2667` on
Lending Club. At the exactly matched factual state, all `3300` action keys had identical q-pool
IDs, classifier scores, and resampling uniforms across beta arms. The selected mean target
probability still increased monotonically from `.26027` (beta=0) to `.26416` (beta=1) and
`.26720` (beta=2), confirming that the mechanism works even though adaptive final validity did
not improve. The operational classifier-top-B comparator was much stronger on this validation
pilot (`+0.1583` threshold success versus beta=0), but it is not a pi-beta sample and remains a
secondary report rather than a selected confirmation arm.

Every pi-beta expansion used M=64 and B=9; classifier-top-B used the same finite pool as an
operational comparator. Across all E13 cells, the run recorded `341,312` raw and projected
proposals, `123,513` no-ops, `34,615` duplicates, and `183,184` unique proposed rows. Guidance
traces contain `2,584,896` raw pool atoms and `1,315,205` post-projection unique pool rows. The
run made `30,241` classifier calls over `1,263,998` rows and `40,389` TabICL calls over
`1,766,085` rows. The k=1 `search_depth` was always zero; validity search used a mean `3.010`
steps, maximum `14` (`1,001` sparse-valid and `199` validity-not-reached factual-runs).

## Timings and confirmation estimate

Canonical E11 lifecycle totals were prepare `1.05 s`, generate `4059.04 s`, evaluate `8.90 s`,
write `11.57 s`, and total `4096.91 s`. E13 totals were prepare `0.45 s`, generate `1086.28 s`,
evaluate `3.90 s`, write `6.64 s`, and total `1105.33 s`. E12's proposal-diagnostic format does
not instrument those generic lifecycle phases; its final resume executed 27 new cells and
reverified all 36 in `495 s`, while preserving the earlier nine complete cells. That wall-clock
measurement cannot be decomposed into prepare/generate/evaluate/write and is marked unavailable.

Scaling preparation by the number of confirmation seeds and the remaining lifecycle phases by
both seed count and the 20-to-100 factual ratio projects about `9.48` aggregate GPU-hours for E11
and `2.56` for E13. E12 projects `0.14-0.31` GPU-hours depending on how closely its cost scales
with the reduction from 576 pilot integration/MC levels to 256 confirmation levels. The central
sum is about `12.2` GPU-hours; use a planning range of **10-15 aggregate GPU-hours** on this host.
This is an **E11-E13 k=1 confirmation** estimate and excludes the separately gated k=3
replication.

## Failures retained

- The historical replay initially lacked authenticated legacy-case reconstruction. Commits
  `cc12a29`, `fef5e29`, `3c20063`, and `553aab6` corrected the replay path without changing
  historical artifacts.
- The first Bank Marketing E12 cell exposed that categorical support is factual-specific after
  kNN context selection. Commit `2509e7b` records the actual learned support per factual and
  preserves the nine completed HELOC cells; the failed partial cell was never published.
- One detached launch lacked `/home/ofurman/.local/bin` in `PATH`, and the first E13 command used
  `classifier_guided` instead of the tracked `classifier_guidance` filename. Both attempts failed
  before creating canonical E13 runs; the corrected in-scope command resumed into the declared
  root. The operational log retains both failures.

Stage 8 remains outside this authorization. It requires a separate decision that cites the
measured `10-15` GPU-hour confirmation range; no test-partition or k=3 run has been launched.
