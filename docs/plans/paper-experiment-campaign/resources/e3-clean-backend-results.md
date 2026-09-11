# E3 (clean): Proposal-Backend Attribution

**Question.** Does CounterContEx benefit from TabICL's learned conditional proposals *beyond* the
locality a simple empirical estimator already supplies? The original `campaign_e3_backend.yaml`
run compared TabICL only against a **global** empirical backend, so any TabICL advantage there
confounds "learned conditioning" with "uses the factual's neighbourhood at all". This clean run
adds a factual-local empirical backend as the missing middle term and leaves the historical E3
untouched under its own identity.

**Setup.** Three backends -- `empirical` (global target-class quantiles), `empirical_local`
(factual-local quantiles, no checkpoint), and `tabicl` -- x four datasets (HELOC, Bank Marketing,
Give Me Some Credit, Lending Club) x three target models (logistic regression, MLP, XGBoost).
36 cells, k = 1, seed 42, 250 deterministic stratified factuals per block, sparse search,
generation threshold 0.5, confidence conditioning and joint scoring disabled, evaluation
`countercontex.evaluation.v2` at threshold 0.7. Method implementation `countercontex-v3`,
matrix `campaign_e3_clean_backend.yaml`. Executed on Helios GH200 (job 22122283, 2026-09-08),
36/36 cells complete in 0.78 h of cell time; repeated empirical seeds are deliberately omitted
because the fixed quantile-grid path is deterministic.

**Estimand.** The primary contrast is `tabicl - empirical_local`, paired over every matched
factual within each dataset/model block; positive favours TabICL. Two secondary contrasts
decompose it: `empirical_local - empirical` isolates factual-local context, and
`tabicl - empirical` is the full backend bundle. All 95 % intervals are paired over the 250
matched factuals in a block; there is no seed dimension, so they are within-block, not
seed-to-seed.

## Primary contrast: `tabicl - empirical_local`

Threshold validity (target class and p >= 0.7), per requested slot, per block. `n = 250` matched
factuals throughout.

| dataset | model | empirical_local | tabicl | delta | 95 % CI |
|---|---|---|---|---|---|
| Give Me Some Credit | MLP | .104 | .528 | **+.424** | [.360, .488] |
| Give Me Some Credit | logistic | .104 | .304 | **+.200** | [.148, .256] |
| Lending Club | logistic | .220 | .276 | +.056 | [.028, .088] |
| Bank Marketing | MLP | .044 | .052 | +.008 | [-.024, .040] |
| HELOC | all three | .000 | .000 | .000 | [.000, .000] |
| Lending Club | MLP | .196 | .192 | -.004 | [-.020, .012] |
| Bank Marketing | XGBoost | .028 | .016 | -.012 | [-.028, .000] |
| Lending Club | XGBoost | .168 | .156 | -.012 | [-.036, .012] |
| Bank Marketing | logistic | .104 | .056 | -.048 | [-.084, -.012] |
| Give Me Some Credit | XGBoost | .120 | .072 | -.048 | [-.080, -.016] |

Pooled over the twelve blocks the mean delta is **+.047** (range -.048 to +.424). The effect is
not broad: it is carried almost entirely by Give Me Some Credit under the two smooth classifiers
(LR, MLP), is small-positive on Lending Club LR, is exactly zero on all of HELOC, and is
negative on five blocks (Bank Marketing LR, both XGBoost credit blocks, and small negatives
elsewhere). HELOC is a floor, not a tie: no backend clears p >= 0.7 for any candidate, so the
zero contains no information about the backend.

## Coverage and class success

Every backend returns a candidate for essentially every factual and flips the decision boundary:
`empirical` coverage 1.000, `empirical_local` .995, `tabicl` .985 (means over blocks); class
validity per requested slot equals coverage in each case (k = 1, so every returned candidate that
exists also crosses 0.5). TabICL therefore buys its threshold-validity gains while losing a small
amount of raw coverage: the learned-conditional contrast is -.010 on coverage (range -.084 to 0)
and the full bundle -.015 (range -.112 to 0). The threshold gain and the coverage loss must be
read together.

## Proximity, sparsity, plausibility

Means over the twelve blocks, and the pooled contrast deltas (positive = the right-hand backend
scores higher):

| metric | empirical | empirical_local | tabicl | locality | learned_cond. | bundle |
|---|---|---|---|---|---|---|
| grouped-Gower (lower better) | .0459 | .0468 | .0536 | +.0010 | +.0071 | +.0081 |
| action units / CF | 1.056 | 1.166 | 1.456 | +.118 | +.309 | +.422 |
| sparsity (fraction) | .0437 | .0488 | .0624 | -- | -- | -- |
| LOF (higher = more outlying) | 1.467 | 1.448 | 1.460 | -.018 | +.013 | -.004 |
| Isolation Forest (higher = more inlying) | .0613 | .0627 | .0595 | +.001 | -.003 | -.002 |
| k-th neighbour Gower | .0386 | .0379 | .0380 | -.001 | +.000 | -.001 |
| out-of-bounds fraction | .013 | .013 | .013 | -- | -- | -- |
| actionability | 1.000 | 1.000 | 1.000 | -- | -- | -- |

Proximity metrics use the jointly-target-class-valid population (222-250 factuals per block);
the rest use all jointly-available candidates. TabICL's threshold-validity gains are bought with
consistently *farther* and *less sparse* counterfactuals: it changes about 1.46 action units
against the local empirical backend's 1.17 and the global backend's 1.06, and its grouped-Gower
is +.008 versus the global backend. The density diagnostics (LOF, Isolation Forest, k-th
neighbour) barely move in any direction, so there is no plausibility signal separating the
backends at this protocol.

## Findings

1. **Locality is cheap and small.** `empirical_local - empirical` adds only +.018 mean threshold
   validity, at a small coverage cost (-.005) and a modest sparsity increase. Most of what a
   neighbourhood buys is available without a checkpoint.
2. **Learned conditioning adds more, but narrowly.** `tabicl - empirical_local` is +.047 pooled,
   concentrated in Give Me Some Credit (LR, MLP). On seven of twelve blocks it is zero or
   negative. TabICL is not a uniform improvement over a local empirical estimator.
3. **Gains cost proximity and sparsity, not plausibility.** Wherever TabICL raises threshold
   validity it also moves counterfactuals farther from the factual and touches more features; the
   density diagnostics stay flat.
4. **Cost is asymmetric.** TabICL spends 227.9 s mean per cell against `empirical_local`'s 1.6 s
   and `empirical`'s 5.2 s -- roughly 140x the local empirical backend for a +.047 mean threshold
   gain that is real only on one dataset family.

## Interpretation

The clean decomposition sharpens the original E3 verdict. The historical run credited TabICL with
a threshold-validity edge over a global empirical backend, but roughly a third of that edge is
just locality, which any factual-local estimator reproduces for free. The learned-conditional
residual is genuine but dataset-specific: it appears on the smooth-classifier credit blocks and
vanishes or reverses elsewhere, and it is paid for in proximity, sparsity, and a large runtime
premium. On this evidence TabICL earns its place as a proposal backend only where the target
model is smooth and the data support it, not as a general default.

## Caveats

- Descriptive, single-seed, k = 1. The CIs are within-block paired intervals over 250 factuals,
  not seed-to-seed noise; the campaign's Stage 1 noise floor does not bound these v2 metrics.
- HELOC contributes no threshold-validity signal (universal zero at tau = 0.7); read it as a
  floor. See `resources/e2b-results.md` for the same boundary-hugging behaviour.
- This is a different experiment from the canonical E3 (`campaign/e3_backend`, k = 3, five seeds,
  logistic target only), which is unchanged and still lives only on the DGX node `gx10-bdc5`. Do
  not pool the two.
- The clean-E3 code (`analysis/e3.py`, the `empirical_local` backend, `campaign_e3_clean_backend.yaml`,
  method `countercontex-v3`) lives on branch `codex/e3-local-empirical` (commit `745bd60`), not on
  `origin/paper-experiment-campaign`. This report was built from the pulled artifacts; the current
  checkout (`e2-diversity-revision`) predates that code and cannot re-aggregate the tree.

## Artifacts

`experiments/zeroshot_cf/results/campaign/e3_clean_backend/` (36 runs; `aggregate_summary.csv`
SHA-256 `cb6523ca…07bd0bb`), analysis in `e3_clean_backend_analysis/`
(`e3_paired_contrasts.csv` SHA-256 `6d934c46…5d0b7c3`, `e3_raw_summaries.csv`
`4fa07d1a…c147aa9b`, `e3_paired_points.csv`, `e3_analysis_manifest.json`), job logs in
`results/campaign/launch/cx_e3-22122283.{out,err}`. Plan and reporting protocol:
`docs/plans/e3-clean-backend-ablation.md` on branch `codex/e3-local-empirical`.
