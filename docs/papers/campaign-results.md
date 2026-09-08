# CounterContEx paper campaign results

This document is the traceability source for the paper experiment campaign run
on the `gx10-bdc5` GB10. Scientific specifications are the tracked matrices in
`experiments/zeroshot_cf/configs/matrices/`; canonical payloads are under
`experiments/zeroshot_cf/results/campaign/`. Generated tables and figures are
committed in [`campaign-artifacts/`](campaign-artifacts/).

## Reading the metrics

| Metric | Concise meaning | Better |
|---|---|:---:|
| Coverage | Fraction of factuals with at least one returned candidate. | Higher |
| Set coverage@k | Fraction of factuals for which all `k` requested candidates were returned. | Higher |
| Class validity | Fraction of returned candidates predicted as the requested target class. | Higher |
| Threshold validity | Fraction of returned candidates that reach the target class and its probability threshold, normally .7. | Higher |
| Achieved probability | Mean target-class probability of returned candidates. | Higher |
| Grouped-Gower | Mixed-type distance from a factual to a target-class candidate; a one-hot group counts as one feature. | Lower |
| Action units | Mean number of changed numerical features or atomic categorical groups per returned candidate. | Lower |
| Actionability | Fraction of returned candidates that preserve every immutable feature. | Higher |
| Neighbour support | Grouped-Gower distance to the fifth-nearest training row. | Lower |
| Out-of-bounds fraction | Fraction of returned candidates with any normalized feature outside `[0, 1]`. | Lower |
| Action Jaccard distance | Dissimilarity between the changed-action sets within a returned set. | Higher diversity |
| Pairwise Gower | Mixed-type distance between candidates within a returned set. | Higher diversity |
| Robustness retention | Fraction still predicted as the target class after evaluation-only model retraining. | Higher |
| Detectability AUC | Orientation-independent linear-probe separation of real and counterfactual rows; .5 is indistinguishable and 1 is fully separable. | Lower |
| Runtime | Measured prepare, generate, evaluate, write, or total elapsed seconds. | Lower |

Coverage and validity use different denominators: unavailable slots reduce
coverage but do not enter returned-candidate validity. Per-requested-slot rates
do include those slots, while per-factual rates ask whether each factual has at
least one success. `Primary` means rank 0; `set` means all returned ranks.
Grouped-Gower and continuous proximity use target-class candidates, whereas
sparsity, actionability, bounds, LOF, Isolation Forest, and neighbour support
use every returned candidate. Larger stored LOF values are more outlying;
larger Isolation Forest values are more inlying. Detectability is reported only
with its arm counts and measurement status.

Unless a table says otherwise, values are macro means over the dataset,
classifier, and/or seed blocks named in the experiment configuration. `±` is
the sample standard deviation across the blocks, not a confidence interval.
`Delta` columns are paired first-arm-minus-second-arm differences. `—` means
that the quantity was not part of that reported comparison; it is not a zero and
was not imputed from another run.
Reported decimal values use two places, or four when their absolute value is
below `.01`. The linked artifacts retain full precision.

## Experiments

### E1 — main comparison

- Configuration: `campaign_e1_main.yaml`; six datasets × three target families
  × five seeds × six methods, 250 factuals, k=1 (540/540 complete).
- Artifacts: `results/campaign/e1_main_dice_v5/`; aggregate SHA-256
  `604eebe8...bcf991`; paper products `t1_main.*`,
  `f3_critical_difference.*`, `f6_target_probability.*`.

| Method | Primary coverage | Returned class validity | Returned threshold validity | Grouped-Gower | Action units |
|---|---:|---:|---:|---:|---:|
| CounterContEx | 1.00 ± .0085 | 1.00 ± 0 | .12 ± .16 | .06 ± .04 | 1.29 ± .46 |
| DiCE | .98 ± .04 | 1.00 ± 0 | .11 ± .10 | .09 ± .05 | 1.69 ± .56 |
| FACE | 1.00 ± 0 | 1.00 ± 0 | .20 ± .21 | .11 ± .06 | 4.36 ± 2.00 |
| Growing Spheres | 1.00 ± .0004 | 1.00 ± 0 | .12 ± .13 | .06 ± .03 | 1.31 ± .67 |
| NICE | .92 ± .16 | 1.00 ± 0 | .14 ± .14 | .06 ± .03 | 1.57 ± .55 |
| Wachter | .99 ± .04 | 1.00 ± 0 | .06 ± .07 | .06 ± .03 | 1.20 ± .34 |

| Method | Mean total s/cell | SD s/cell |
|---|---:|---:|
| CounterContEx | 469.81 | 463.59 |
| DiCE | 25.97 | 11.38 |
| FACE | 12.60 | 10.86 |
| Growing Spheres | 2.16 | 1.34 |
| NICE | 1.69 | 1.47 |
| Wachter | 3.85 | 1.61 |

The canonical E1 lifecycle totals were prepare 42.07 s, generate 45,856.78 s,
evaluate 520.19 s, write 5.47 s, and total 46,447.20 s (12.90
cell-hours). The 18 rows behind each method summary are available in
[`t1_main.csv`](campaign-artifacts/t1_main.csv); F6 contains 120,709 returned
candidates.

- Outcome: CounterContEx mean coverage/class validity/threshold validity/
  grouped-Gower/action units were `1.00/1/.12/.06/1.29` across
  18 dataset–classifier blocks. Against DiCE it significantly improved coverage
  (+.02, Holm p=.02), proximity (-.04, p=.0026), and action units
  (-.40, p=.0092); it also improved proximity and action units against
  FACE. The other 20 jointly corrected tests were not significant. All methods
  tied on returned-class validity, so F3’s rank tie is intentionally
  uninformative. F6 contains 120,709 returned candidates.

### E2 — diverse sets

- Configuration: `campaign_e2_diverse.yaml`; six datasets × five seeds ×
  CounterContEx/DiCE, 250 factuals, k=3 (60/60 complete).
- Artifacts: `results/campaign/e2_diverse/`; aggregate SHA-256
  `30ea0d5c...07a29f9`; paper product `t2_diversity.*`.

| Method | Set coverage@3 | Returned class validity | Returned threshold validity | Grouped-Gower | Action Jaccard distance | Pairwise Gower |
|---|---:|---:|---:|---:|---:|---:|
| CounterContEx | 1.00 | 1.00 | .13 | .06 | .60 | .06 |
| DiCE | .67 | 1.00 | .11 | .10 | .47 | .09 |

These are macro means over six dataset rows; each dataset row is a five-seed
mean. The full dataset-level means, standard deviations, and actual `n` are in
[`t2_diversity.csv`](campaign-artifacts/t2_diversity.csv). E2 used 24,659.79 s
in generation and 6.90 total cell-hours.

- Outcome: CounterContEx versus DiCE means for set coverage/class validity/
  threshold validity/grouped-Gower/action Jaccard/pairwise Gower were
  `1.00/1/.13/.06/.60/.06` versus
  `.67/1/.11/.10/.47/.09`. CounterContEx’s quality
  guardrails were better, but its sets were less diverse by both recorded
  orientations; no inferential E2 claim is made.

### E3 — TabICL versus empirical proposals

- Configuration: `campaign_e3_backend.yaml`; six datasets × three seeds × two
  backend arms, 250 factuals, k=3, shared sparse-search capabilities (36/36).
- Artifacts: `results/campaign/e3_backend/`; aggregate SHA-256
  `4683d367...960a49`.

The empirical baseline derives each requested quantile from reference rows that the classifier assigns to the target class.
It ranks categorical proposals by their target-class frequencies.
Every factual with the same target therefore gets the same proposals.
TabICL instead conditions each masked feature on the current row, target class, and local context.
Both arms use the same quantile grid and sparse search, so E3 isolates the proposal backend.

| Metric | TabICL − empirical | Stage 1 spread | Direction for TabICL |
|---|---:|---:|---|
| Coverage | -.0033 | 0 | Adverse |
| Returned class validity | 0 | 0 | Tie |
| Returned threshold validity | +.0057 | 0 | Favourable |
| Grouped-Gower | +.0079 | 0 | Adverse |
| Neighbour-support distance | -.0004 | NOT MEASURED | Favourable, without a Stage 1 noise verdict |
| Out-of-bounds fraction | +.0007 | — | Adverse |
| Action Jaccard distance | -.03 | — | Less diverse |
| Pairwise Gower | -.0002 | — | Less diverse |
| Actionability | 1.00 vs 1.00 | — | Tie |

| Dataset | Threshold-validity delta, TabICL − empirical |
|---|---:|
| Adult | -.05 |
| Bank | +.0029 |
| German | -.0028 |
| Give Me Some Credit | -.0067 |
| HELOC | -.0013 |
| Lending Club | +.09 |

| Backend | Mean total s/cell |
|---|---:|
| Empirical | 126.89 |
| TabICL | 1026.00 |

E3 used 20,637.62 s in generation and 5.76 total cell-hours. The Stage 1
spread is the narrow HELOC/k=1 repeatability reference, not a six-dataset k=3
uncertainty interval.

- Outcome: TabICL-minus-empirical coverage/threshold validity/grouped-Gower/
  neighbour-support differences were `-.0033/+.0057/+.0079/-.0004`.
  The threshold effect was Lending-Club-led, proximity worsened on five of six
  datasets, and runtime increased 8.09×. Neighbour-support noise relative to
  Stage 1 is NOT MEASURED because those v1 artifacts predate the metric.

### E4 — confidence and threshold Pareto

- Configuration: `campaign_e4_confidence.yaml`; four datasets × five seeds ×
  five generation thresholds × confidence off/on, 250 factuals (200/200).
- Artifacts: `results/campaign/e4_confidence/`; aggregate SHA-256
  `208776cd...59cc`; paper product `f4_confidence_pareto.*`, including the
  comparable E1 baseline slice.

The following table uses the evaluator's fixed threshold of .7. Each row is a
macro mean over four datasets and five seeds.

| Generation tau | Confidence conditioned | Achieved probability | Returned threshold validity@.7 | Coverage | Grouped-Gower | Mean total s/cell |
|---:|:---:|---:|---:|---:|---:|---:|
| .5 | No | .58 | .14 | .97 | .06 | 190.41 |
| .5 | Yes | .55 | .09 | .99 | .05 | 511.94 |
| .6 | No | .67 | .29 | .93 | .07 | 285.32 |
| .6 | Yes | .65 | .17 | .97 | .07 | 586.43 |
| .7 | No | .75 | 1.00 | .84 | .09 | 418.54 |
| .7 | Yes | .74 | 1.00 | .94 | .08 | 1003.32 |
| .8 | No | .84 | 1.00 | .70 | .11 | 542.53 |
| .8 | Yes | .83 | 1.00 | .85 | .10 | 1580.02 |
| .9 | No | .92 | 1.00 | .60 | .15 | 654.92 |
| .9 | Yes | .91 | 1.00 | .67 | .14 | 2751.19 |

The same candidates were also rescored at five evaluation thresholds. Entries
below are returned-candidate threshold-validity rates.

| Generation tau | Conditioned | Eval .5 | Eval .6 | Eval .7 | Eval .8 | Eval .9 |
|---:|:---:|---:|---:|---:|---:|---:|
| .5 | No | 1.00 | .32 | .14 | .04 | .0083 |
| .5 | Yes | 1.00 | .18 | .09 | .01 | .0041 |
| .6 | No | 1.00 | 1.00 | .29 | .10 | .01 |
| .6 | Yes | 1.00 | 1.00 | .17 | .05 | .0041 |
| .7 | No | 1.00 | 1.00 | 1.00 | .21 | .02 |
| .7 | Yes | 1.00 | 1.00 | 1.00 | .12 | .0093 |
| .8 | No | 1.00 | 1.00 | 1.00 | 1.00 | .09 |
| .8 | Yes | 1.00 | 1.00 | 1.00 | 1.00 | .05 |
| .9 | No | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| .9 | Yes | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

| E1 method, comparable four-dataset slice | Probability median | Probability mean | Fraction below .6 |
|---|---:|---:|---:|
| CounterContEx | .51 | .55 | .82 |
| DiCE | .50 | .57 | .72 |
| FACE | .54 | .60 | .71 |
| Growing Spheres | .50 | .55 | .78 |
| NICE | .53 | .59 | .71 |
| Wachter | .50 | .53 | .85 |

E4 consumed 47.36 total cell-hours, of which 47.28 were generation. The
1,600 cell/threshold rows are in
[`f4_confidence_pareto.csv`](campaign-artifacts/f4_confidence_pareto.csv).

- Outcome: achieved probability increased monotonically with generation
  threshold in all eight dataset/conditioning arms. From tau .5 to .9, the
  unconditioned arm moved from probability/coverage/Gower
  `.58/.97/.06` to `.92/.60/.15`; the conditioned arm moved
  from `.55/.99/.05` to `.91/.67/.14`. Conditioning did not
  raise confidence at fixed tau, but often preserved coverage and proximity at
  substantial runtime cost. E1 confirmed boundary hugging across all methods.

### E5 — search and diversity ablations

- Configuration: `campaign_e5_search.yaml`; six datasets × five seeds × seven
  one-axis variants, 100 factuals (210/210).
- Artifacts: `results/campaign/e5_search/`; aggregate SHA-256
  `e9875119...bfcb`.

| k=1 search arm | Coverage | Threshold validity | Grouped-Gower | Neighbour support delta | Mean total s/cell | Actionability |
|---|---:|---:|---:|---:|---:|---:|
| Mode only | .91 | .19 | .08 | — | 119.82 | 1.00 |
| Nine quantiles | .98 | .11 | .07 | Reference | 69.63 | 1.00 |
| Data-plausible refinement | .98 | .17 | .08 | -.0052 | 139.83 | 1.00 |
| Revisits off | .96 | .11 | .07 | — | 68.48 | 1.00 |

For neighbour-support distance, smaller is better; the data-plausible row's
`-.0052` is relative to the matched nine-quantile sparse arm.

| k=3 selector | Threshold validity | Action Jaccard distance | Pairwise Gower | Grouped-Gower delta vs DPP | Mean total s/cell | Actionability |
|---|---:|---:|---:|---:|---:|---:|
| DPP | .23 | .75 | .08 | Reference | 674.08 | 1.00 |
| Random | .23 | .60 | .07 | — | 614.44 | 1.00 |
| Greedy farthest | .26 | .66 | .09 | +.0049 | 591.24 | 1.00 |

| Joint-refinement diagnostic | Value |
|---|---:|
| Factuals with a scored sparse start | 2,945 |
| Fraction with positive internal-density gain | .99 |
| Mean internal-density gain | 4.27 |
| Action-unit delta vs matched sparse arm | +.37 |
| Grouped-Gower delta vs matched sparse arm | +.01 |

E5 consumed 18.98 total cell-hours.

- Outcome: nine quantiles improved coverage, proximity, and runtime over the
  mode-only arm but reduced threshold validity. Data-plausible refinement
  recovered .06 threshold validity and improved neighbour support by
  .0052, while adding .01 Gower and 70.20 seconds. Among k=3 selectors,
  greedy-farthest improved both diversity orientations over DPP. For the 2,945
  factuals with a scored sparse start, joint refinement raised internal TabICL
  density in 98.71% (mean gain 4.27), but added .37 action units versus the
  matched sparse arm.

### E6 — context ablation

- Configuration: `campaign_e6_context.yaml`; six datasets × five seeds × four
  context sizes × predicted/true training labels, 100 factuals (240/240).
- Artifacts: `results/campaign/e6_context/`; aggregate SHA-256
  `ba8bc038...329f`.

| Context labels | Context size | Coverage | Threshold validity | Grouped-Gower | Mean total s/cell | Actionability |
|---|---:|---:|---:|---:|---:|---:|
| Predicted | 64 | .89 | .26 | .12 | 177.09 | 1.00 |
| True training | 64 | .87 | .24 | .13 | 200.13 | 1.00 |
| Predicted | 128 | .88 | .24 | .10 | 231.80 | 1.00 |
| True training | 128 | .85 | .22 | .12 | 261.79 | 1.00 |
| Predicted | 256 | .90 | .21 | .10 | 311.19 | 1.00 |
| True training | 256 | .86 | .20 | .13 | 375.40 | 1.00 |
| Predicted | 512 | .91 | .20 | .08 | 449.49 | 1.00 |
| True training | 512 | .86 | .18 | .12 | 609.36 | 1.00 |

The true-training rows are reconstructed from the audited true-minus-predicted
paired deltas at each context size. E6 consumed 21.80 total cell-hours.

- Outcome: larger predicted-label contexts improved proximity and eventually
  coverage while reducing threshold validity and increasing runtime. True
  training labels were worse at every size on coverage, threshold validity,
  proximity, and runtime. No test or factual labels enter method context.

### E7 — cost-quality Pareto and freeze

- Configuration: `campaign_e7_cost.yaml`; four datasets × five seeds × four
  configurations, 250 factuals (80/80).
- Artifacts: `results/campaign/e7_cost/`; aggregate SHA-256
  `77aaf820...3b4b`; paper product `f5_cost_quality.*`.

| Configuration | k | Quantiles | Context | Confidence/joint scoring | Coverage | Threshold validity | Grouped-Gower | Mean total s/cell | Actionability |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| C1 | 1 | Mode only | 128 | No | .86 | .30 | .10 | 227.64 | 1.00 |
| C2 | 1 | 5 | 128 | No | .95 | .22 | .09 | 139.12 | 1.00 |
| C3 | 3 | 5 | 256 | No | .98 | .29 | .06 | 341.73 | 1.00 |
| C4 | 3 | 9 | 512 | Yes | 1.00 | .17 | .05 | 848.04 | 1.00 |

Each entry is a macro mean across E7's four datasets and five seeds.
Dataset-level runtime and threshold-validity means and standard deviations are
in [`f5_cost_quality.csv`](campaign-artifacts/f5_cost_quality.csv). E7 consumed
8.65 total cell-hours; C4's Lending Club mean was 2768.88 s/cell.

- Outcome: C1–C4 coverage/threshold validity/Gower/seconds were
  `.86/.30/.10/227.64`, `.95/.22/.09/139.12`,
  `.98/.29/.06/341.73`, and `1/.17/.05/848.04`.
  C3 (five quantiles, context 256, predicted labels, k=3, DPP) was frozen for
  E9/E10 before either result was inspected.

### E8 — robustness rescoring

- Configuration: no generation matrix; read-only rescoring of all 120,709 E1
  candidates with LR, MLP, and XGBoost instruments at seeds 137/271/811.
- Artifacts: `results/campaign/analysis/e8_robustness/`; source-tree digest
  `cf79ad2b...20`; table/detail/manifest SHA-256 values
  `c4beb207...1e3b`, `5c44865d...8163c`, `610f3c74...f9d9`.

| Original target-probability bin | Rescored rows | Retained class validity |
|---|---:|---:|
| [.5, .6) | 263,421 | .86 |
| [.6, .7) | 52,716 | .96 |
| [.7, .8) | 24,957 | .98 |
| [.8, .9) | 12,654 | 1.00 |
| [.9, 1] | 8,379 | 1.00 |

| Original method | Rescored rows | Retained class validity |
|---|---:|---:|
| CounterContEx | 61,455 | .90 |
| DiCE | 60,138 | .87 |
| FACE | 61,650 | .93 |
| Growing Spheres | 61,644 | .87 |
| NICE | 56,205 | .91 |
| Wachter | 61,035 | .87 |

| Evaluation instrument | Rescored rows | Retained class validity | Seed variation |
|---|---:|---:|---|
| Logistic regression | 121,242 | 1.00 | None under the fixed full-sample fit |
| MLP | 120,255 | .67 | Seeds 137/271/811 vary |
| XGBoost | 120,630 | 1.00 | None under the fixed full-sample fit |

The 362,127 rescored rows equal 120,709 E1 candidates times three retraining
seeds. Dataset-pooled retention ranged from .79 for Bank to .96 for Give
Me Some Credit. E8 was a read-only scoring pass and has no generic lifecycle
timing manifest.

- Outcome: pooled validity retention rose from .86 in probability bin
  `[.5,.6)` (n=263,421 rescored rows) to 1.00 in `[.9,1]` (n=8,379),
  supporting the predicted fragility of boundary-hugging candidates. LR and
  XGBoost instruments were deterministic under their fixed full-sample fits;
  only MLP supplied seed variation.

### E9 — foundation-model swap

- Configuration: `campaign_e9_fmswap.yaml`; Adult/HELOC × three seeds ×
  TabICL/TabPFN v2 under frozen C3 search (12/12).
- Artifacts: `results/campaign/e9_fmswap/`; aggregate SHA-256
  `abe833d8...9e808`; paper product `t3_backend.*`.

| Dataset | Backend | Coverage | Threshold validity | Grouped-Gower | Neighbour-support distance | Mean total s/cell |
|---|---|---:|---:|---:|---:|---:|
| Adult | TabICL | 1.00 ± 0 | .17 ± 0 | .10 ± 0 | .11 ± 0 | 699.05 |
| Adult | TabPFN v2 | 1.00 ± 0 | .16 ± .01 | .10 ± .0003 | .12 ± .0005 | 3148.14 |
| HELOC | TabICL | .93 ± 0 | 0 ± 0 | .02 ± 0 | .05 ± 0 | 405.07 |
| HELOC | TabPFN v2 | .91 ± .0061 | 0 ± 0 | .02 ± .0005 | .05 ± .0001 | 2892.58 |

Values are three-seed means ± sample standard deviations. The complete
dataset/backend rows are in
[`t3_backend.csv`](campaign-artifacts/t3_backend.csv). E9 consumed 5.95 total
cell-hours.

- Outcome: TabPFN-minus-TabICL threshold validity/Gower were
  `-.01/+.0049` on Adult and `0/-.0011` on HELOC; HELOC coverage fell
  .02. TabPFN was 4.50× and 7.14× slower. This demonstrates backend
  portability, not a quality gain.

### E10 — frozen 1,000-factual headline

- Configuration: `campaign_e10_headline.yaml`; six datasets, seed 42, logistic
  regression, frozen C3, up to 1,000 factuals, k=3 (6/6).
- Artifacts: `results/campaign/e10_headline/`; aggregate SHA-256
  `df781753...2941b`; run IDs and headline metrics are below. Paper product
  `f7_qualitative_case.*` uses matched HELOC source index 21 and Adult source
  index 10 with CounterContEx, NICE, and DiCE.

| Dataset | Run ID prefix | Factuals | k | Coverage | Class validity | Threshold validity | Gower | Neighbour support | Actionability | Total s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adult | `00cea69e` | 1,000 | 3 | 1.00 | 1.00 | .18 | .10 | .12 | 1.00 | 2367.93 |
| Bank | `9119b938` | 1,000 | 3 | 1.00 | 1.00 | .50 | .04 | .06 | 1.00 | 1806.81 |
| German | `40d888cb` | 120 | 3 | 1.00 | 1.00 | .0028 | .06 | .25 | 1.00 | 686.16 |
| GMSC | `993dcadb` | 1,000 | 3 | 1.00 | 1.00 | .45 | .09 | .05 | 1.00 | 452.27 |
| HELOC | `0eb68d5b` | 1,000 | 3 | .92 | 1.00 | 0 | .02 | .05 | 1.00 | 1733.82 |
| Lending Club | `8890a2c3` | 1,000 | 3 | 1.00 | 1.00 | .21 | .10 | .06 | 1.00 | 1051.55 |

German Credit has 120 test factuals after the retained deterministic split, so
its `max_test: 1000` cap does not fabricate additional rows.

E10 consumed 2.25 measured cell-hours: prepare .11 s, generate 8027.88 s,
evaluate 66.74 s, write .38 s, and total 8098.55 s. The configuration was
not revised after inspection.

## Compute accounting

The measured campaign lower bound is **133.11 cell-hours**, versus the
planning estimate of approximately 120 hours. It is the Stage 10 audited
124.91 hours through E7 (including Stage 1 feasibility and unique published
superseded E1 DiCE attempts), plus E9’s 5.95 and E10’s 2.25 hours. E8 is a
read-only scoring pass and has no generic lifecycle manifest; copied quarantine
artifacts are excluded. This is compute consumption summed from manifest
`total_s`, not elapsed wall time or an assertion that every phase used the GPU.

## Paper product provenance

`campaign-artifacts/analysis_manifest.json` lists exactly 17 products. T1 and
F3/F6 read E1; T2 reads E2; T3 reads E9; F4 reads E4 plus its E1 baseline;
F5 reads E7; and F7 reads E10 plus matched E1 NICE/DiCE rows. F7 authenticates
reconstructed factual selection through each stored `case_id` before applying
the pinned inverse transform. Setting `SOURCE_DATE_EPOCH=0` makes PDF metadata
deterministic; regeneration into a clean directory was byte-identical for all
17 files.
