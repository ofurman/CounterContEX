# E2b: Diversity-Budget Sweep

**Question.** Does loosening CounterContEx's diverse-search Gower budget
(`diversity.max_gower_ratio`) close the value-diversity gap to DiCE observed in E2, without
degrading coverage, validity, or proximity?

**Setup.** Six datasets (HELOC, Bank Marketing, Give Me Some Credit, Lending Club, Adult,
German Credit) x three seeds (17, 42, 101) x ratio in {2.5, 4.0}; 36 cells, k = 3,
logistic-regression target, TabICL backend, evaluation v2. Every other setting is
byte-identical to the E2 CounterContEx arm, which supplies the 1.5 point. The ratio bounds the
DPP candidate pool in `diverse_search.py`: a set member is eligible only if its Gower distance
from the factual is at most `ratio * anchor_gower + 0.02`, where the anchor is the closest valid
counterfactual found. Executed on Helios GH200 (job 21880998, 2026-09-04), 36/36 cells
complete, 7.13 h of cell time.

**Results** (six-dataset means; E2 rows quoted from the Stage 8 journal entry).

| Arm | Set coverage@k | Threshold validity | Proximity (Gower), lower better | Action Jaccard distance, higher better | Pairwise Gower, higher better | Pairwise / proximity |
|---|---|---|---|---|---|---|
| CounterContEx 1.5 (E2) | .9993 | .1345 | .0566 | .6011 | .0568 | 1.003 |
| CounterContEx 2.5 | .9993 | .1420 | .0580 | .6270 | .0607 | 1.046 |
| CounterContEx 4.0 | 1.0000 | .1491 | .0590 | .6523 | .0638 | 1.081 |
| DiCE (E2) | .6658 | .1086 | .0996 | .4672 | .0871 | .874 |

Seed-to-seed spread is zero on every set metric (deterministic proposals at temperature 1e-9),
so differences are exact for this protocol, not statistically tested. The last column is the
ratio of six-dataset means; the per-cell `set_pairwise_gower_ratio` averaged over datasets is
1.037 (2.5) and 1.064 (4.0).

**Findings.**

1. Both diversity axes rise monotonically with the budget: action-set Jaccard distance +.051
   and pairwise Gower +.007 from 1.5 to 4.0.
2. Guardrails hold. Proximity worsens by only +.0024 at 4.0, threshold validity improves, set
   coverage is unchanged, actionability stays 1.0, and the out-of-bounds fraction is flat
   (.0105, concentrated in German Credit and Lending Club).
3. The raw pairwise-Gower gap to DiCE narrows from -.030 to -.023 but does not close.
   Normalized by proximity, CounterContEx sets already spread more than DiCE's at every budget.
4. Runtime is unaffected: 3.79 h (2.5) versus 3.34 h (4.0); Lending Club dominates at about
   2.5 to 2.8 ks per cell.

**Validity across confidence thresholds.** The stored `common.target_probabilities` arrays
allow returned-candidate validity to be recomputed at any threshold without re-running anything;
the 0.7 column reproduces the published `validity_returned_threshold` exactly on every cell.
Six-dataset means over returned candidates (`class_success & p >= tau`):

| Arm | tau 0.5 | tau 0.6 | tau 0.7 | tau 0.8 | tau 0.9 |
|---|---|---|---|---|---|
| CounterContEx 2.5 | 1.0000 | .2640 | .1420 | .0660 | .0069 |
| CounterContEx 4.0 | 1.0000 | .2706 | .1491 | .0724 | .0076 |

Per dataset at tau 0.5 / 0.7 (ratio 2.5): Adult 1.000 / .120, Bank Marketing 1.000 / .144,
German Credit 1.000 / .028, Give Me Some Credit 1.000 / .403, HELOC 1.000 / .000, Lending Club
1.000 / .158. At tau 0.5 the value equals `validity_returned_class` by construction (binary
target, 0.5 decision boundary, no returned candidate below 0.5). The probability distribution
explains the drop: 59 % of returned candidates lie in [0.50, 0.55), the minimum is exactly 0.5000,
and the per-dataset median ranges from .5035 (HELOC, maximum .606) to .61 (Give Me Some Credit).
Sparse search stops at the first class crossing (generation tau 0.5), so threshold validity at
0.7 measures margin past the boundary, not correctness. Full table:
`results/campaign/e2b_budget_analysis/validity_by_tau.txt`.

**Interpretation.** The Gower gate is not the lever that limits value diversity. Widening it
only admits candidates the beam already found, and the small response suggests the candidate
pool itself (beam width 8, pool 16, at most two extra actions) is the binding constraint.
DiCE's larger raw spread is bought with sets that sit 1.7x farther from the factual and with one
third of factuals left without a full set.

**Caveats.** Descriptive comparison only; no inferential test. The 1.5 arm and the DiCE arm are
E2 artifacts on the DGX host `gx10-bdc5`, so the three-point table is not yet a single
analysis-layer product. Set metrics are computed over each method's own returned sets, so the
DiCE comparison mixes populations (backlog B-3). T2 does not label the two budget arms
(backlog B-4).

**Artifacts.** `experiments/zeroshot_cf/results/campaign/e2b_budget/` (36 runs,
`aggregate_summary.csv` SHA-256 `49c4ffa2cf866b04942ea3e684f92f65a8648867e0dda78ba061355d6e617f24`),
analysis in `results/campaign/e2b_budget_analysis/` (T2 SHA-256
`82276eb86a35ddfa1eea5a0768137bf8b9e37fe11a71bc4c6525562338959039`), job logs in
`results/campaign/launch/e2b_budget-21880998.{out,err}`, journal entry dated 2026-09-07.
