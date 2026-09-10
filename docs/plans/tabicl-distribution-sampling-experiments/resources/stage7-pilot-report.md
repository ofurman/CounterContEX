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

The expanded priority-metric tables below were recomputed without changing canonical outputs.
Coverage and validity came from `common.*` arrays; Gower, action counts, LOF, and Isolation
Forest came from authenticated candidate arrays. Numerical L2 and categorical Hamming counts
were derived from those same canonical candidates and the identity-matched benchmark factuals
and feature schema. Seeds were averaged inside factuals before dataset-equal aggregation.

## E11: q representation and decoding

For quality metrics, sampling seeds are first averaged inside each factual, factuals are then
averaged inside each dataset, and the four datasets receive equal weight. Repeated factuals
across seeds are Monte Carlo repeats, not independent observations. `class val` and `thr val`
use returned candidates; `thr/slot` includes unavailable requested slots. `Gower`, numerical
L2, and categorical proximity use returned target-class candidates. Categorical proximity is
the Hamming count of changed atomic one-hot groups. Sparsity is the count of changed original
feature/action units and uses all returned candidates. Numerical changes use the configured
`.05` tolerance for sparsity. Lower is better for all proximity and sparsity columns.

| Numerical / categorical policy | coverage | class val | thr val | thr/slot | Gower | numerical L2 | categorical proximity | sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode-1 / iid-9 | .7167 | 1.0000 | .2004 | .1625 | .0602 | .1944 | .5245 | 1.405 |
| grid-9 / iid-9 | .7792 | 1.0000 | .1018 | .0750 | .0661 | .1910 | .6053 | 1.450 |
| iid-9 / iid-9 | .8125 | 1.0000 | .1250 | .1042 | .0575 | .1902 | .5021 | 1.402 |
| top-k-9 / iid-9 | .7333 | 1.0000 | .1726 | .1375 | .0594 | .1889 | .5152 | 1.362 |
| top-p-9 / iid-9 | .7750 | 1.0000 | .1310 | .1042 | .0625 | .1954 | .5768 | 1.449 |
| grid-49 / iid-9 | .8208 | 1.0000 | .0944 | .0750 | .0570 | .1754 | .5096 | 1.348 |
| iid-49 / iid-9 | .8375 | 1.0000 | .1104 | .0875 | .0516 | .1760 | .4495 | 1.303 |
| top-k-49 / iid-9 | .7333 | 1.0000 | .1713 | .1417 | .0554 | .1883 | .4787 | 1.319 |
| top-p-49 / iid-9 | .7917 | 1.0000 | .0958 | .0750 | .0651 | .1820 | .6090 | 1.442 |
| iid-9 / greedy-1 | .7792 | 1.0000 | .0840 | .0708 | .0472 | .1989 | .3550 | 1.297 |
| iid-9 / top-k-9 | .8125 | 1.0000 | .1250 | .1042 | .0568 | .1904 | .4938 | 1.402 |
| iid-9 / top-p-9 | .7917 | 1.0000 | .1083 | .0917 | .0524 | .1970 | .4313 | 1.373 |

LOF, Isolation Forest, diversity, neighbour support, and runtime use all returned candidates.
Lower LOF means less outlying; higher Isolation Forest means more inlying. Pairwise diversity
is `N/A` because E11 returns one final counterfactual per factual (`k=1`). Runtime is the
operational mean total seconds per 20-factual cell.

| Numerical / categorical policy | LOF | Isolation Forest | NN-5 | pairwise diversity | runtime s |
|---|---:|---:|---:|---:|---:|
| mode-1 / iid-9 | 1.3164 | .0716 | .0326 | N/A | 26.47 |
| grid-9 / iid-9 | 1.3825 | .0656 | .0352 | N/A | 21.39 |
| iid-9 / iid-9 | 1.4072 | .0635 | .0364 | N/A | 18.26 |
| top-k-9 / iid-9 | 1.3118 | .0713 | .0324 | N/A | 50.75 |
| top-p-9 / iid-9 | 1.6795 | .0671 | .0358 | N/A | 38.02 |
| grid-49 / iid-9 | 1.3956 | .0631 | .0355 | N/A | 18.69 |
| iid-49 / iid-9 | 1.3703 | .0607 | .0381 | N/A | 17.80 |
| top-k-49 / iid-9 | 1.9415 | .0682 | .0353 | N/A | 53.54 |
| top-p-49 / iid-9 | 1.3594 | .0659 | .0352 | N/A | 37.44 |
| iid-9 / greedy-1 | 1.4008 | .0633 | .0352 | N/A | 21.31 |
| iid-9 / top-k-9 | 1.3942 | .0635 | .0364 | N/A | 17.96 |
| iid-9 / top-p-9 | 1.4123 | .0615 | .0375 | N/A | 19.78 |

For the predeclared comparisons, IID improved `thr` over the grid by `+0.0292` at B=9 and
`+0.0125` at B=49. Coverage changed by `+0.0333` and `+0.0167`, while grouped Gower improved by
`0.0086` and `0.0054`; NN-5 worsened by `0.0012` and `0.0026`. IID reduced categorical
proximity by `0.1031/0.0601` groups and sparsity by `0.0471/0.0445` units at B=9/49. Numerical
L2 improved by only `0.0008` at B=9 and worsened by `0.0006` at B=49. LOF worsened by `0.0248`
at B=9 but improved by `0.0252` at B=49; Isolation Forest worsened by `0.0021/0.0025`. The
effect was heterogeneous:
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

E12 traces feature proposals through a classifier at a fixed factual state; it does not run the
counterfactual search or return a final counterfactual set. Consequently, final-CF coverage,
validity, Gower, numerical L2, categorical proximity, sparsity, LOF, Isolation Forest, and
pairwise diversity are all `N/A`, not zero. The applicable quality outcome is the change in
target probability reported below. The final resume took `495 s` for 27 new cells plus exact
verification of all 36 cells; the diagnostic does not separate prepare/generate/evaluate/write
phases, so comparable per-cell lifecycle time is `NOT MEASURED`.

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
seed/factual/dataset hierarchy. Metric definitions and orientations are identical to E11.

| Policy | coverage | class val | thr val | thr/slot | Gower | numerical L2 | categorical proximity | sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| beta=0 | .8250 | 1.0000 | .1306 | .1083 | .0589 | .1880 | .5194 | 1.409 |
| beta=.5 | .8292 | 1.0000 | .1208 | .0958 | .0550 | .1896 | .4778 | 1.358 |
| beta=1 | .8250 | 1.0000 | .1264 | .1042 | .0485 | .1858 | .3986 | 1.278 |
| beta=2 | .8292 | 1.0000 | .1219 | .1042 | .0513 | .1906 | .4271 | 1.300 |
| classifier-top-B | .8625 | 1.0000 | .2784 | .2667 | .0509 | .2047 | .4082 | 1.269 |

Pairwise diversity is `N/A` because E13 also uses `k=1`. Runtime is the operational mean total
seconds per 20-factual cell.

| Policy | LOF | Isolation Forest | NN-5 | pairwise diversity | ESS | runtime s |
|---|---:|---:|---:|---:|---:|---:|
| beta=0 | 1.5993 | .0647 | .0359 | N/A | 64.00 | 19.11 |
| beta=.5 | 1.3635 | .0633 | .0370 | N/A | 63.73 | 19.49 |
| beta=1 | 1.3827 | .0635 | .0363 | N/A | 63.18 | 18.90 |
| beta=2 | 1.3915 | .0618 | .0373 | N/A | 62.08 | 18.33 |
| classifier-top-B | 1.4689 | .0598 | .0385 | N/A | N/A | 16.27 |

The predeclared beta=1 contrast changed threshold success by `-0.0042` versus beta=0, with no
coverage change. It improved grouped Gower by `0.0104` and reduced action units by `0.131`, but
worsened NN-5 by `0.0005`. Numerical L2 improved by `0.0023`, categorical proximity by `0.1208`
groups, and LOF by `0.2167`; Isolation Forest worsened by `0.0013`. The weak ESS change shows
that `p_clf^beta` only mildly reweighted
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
