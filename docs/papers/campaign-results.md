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

## Experiments

### E1 — main comparison

- Configuration: `campaign_e1_main.yaml`; six datasets × three target families
  × five seeds × six methods, 250 factuals, k=1 (540/540 complete).
- Artifacts: `results/campaign/e1_main_dice_v5/`; aggregate SHA-256
  `604eebe8...bcf991`; paper products `t1_main.*`,
  `f3_critical_difference.*`, `f6_target_probability.*`.

| Method | Primary coverage | Returned class validity | Returned threshold validity | Grouped-Gower | Action units |
|---|---:|---:|---:|---:|---:|
| CounterContEx | .997111 ± .008547 | 1.000000 ± 0 | .124030 ± .160885 | .055860 ± .035722 | 1.288793 ± .461191 |
| DiCE | .976493 ± .038741 | 1.000000 ± 0 | .111434 ± .096086 | .091708 ± .046491 | 1.687267 ± .555783 |
| FACE | 1.000000 ± 0 | 1.000000 ± 0 | .196667 ± .214299 | .110146 ± .063182 | 4.360259 ± 2.001345 |
| Growing Spheres | .999911 ± .000377 | 1.000000 ± 0 | .116507 ± .125890 | .060882 ± .032454 | 1.313916 ± .667387 |
| NICE | .916685 ± .164459 | 1.000000 ± 0 | .142359 ± .141368 | .055139 ± .032719 | 1.574491 ± .549634 |
| Wachter | .985833 ± .036554 | 1.000000 ± 0 | .056520 ± .067908 | .060144 ± .034712 | 1.200064 ± .336631 |

| Method | Mean total s/cell | SD s/cell |
|---|---:|---:|
| CounterContEx | 469.814 | 463.590 |
| DiCE | 25.966 | 11.375 |
| FACE | 12.595 | 10.862 |
| Growing Spheres | 2.164 | 1.338 |
| NICE | 1.690 | 1.467 |
| Wachter | 3.852 | 1.611 |

The canonical E1 lifecycle totals were prepare 42.072 s, generate 45,856.782 s,
evaluate 520.188 s, write 5.468 s, and total 46,447.199 s (12.9020
cell-hours). The 18 rows behind each method summary are available in
[`t1_main.csv`](campaign-artifacts/t1_main.csv); F6 contains 120,709 returned
candidates.

- Outcome: CounterContEx mean coverage/class validity/threshold validity/
  grouped-Gower/action units were `.997111/1/.124030/.055860/1.288793` across
  18 dataset–classifier blocks. Against DiCE it significantly improved coverage
  (+.020619, Holm p=.023501), proximity (-.035848, p=.002563), and action units
  (-.398474, p=.009232); it also improved proximity and action units against
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
| CounterContEx | .999333 | 1.000000 | .134516 | .056624 | .601143 | .056783 |
| DiCE | .665778 | 1.000000 | .108628 | .099626 | .467239 | .087085 |

These are macro means over six dataset rows; each dataset row is a five-seed
mean. The full dataset-level means, standard deviations, and actual `n` are in
[`t2_diversity.csv`](campaign-artifacts/t2_diversity.csv). E2 used 24,659.787 s
in generation and 6.90165 total cell-hours.

- Outcome: CounterContEx versus DiCE means for set coverage/class validity/
  threshold validity/grouped-Gower/action Jaccard/pairwise Gower were
  `.999333/1/.134516/.056624/.601143/.056783` versus
  `.665778/1/.108628/.099626/.467239/.087085`. CounterContEx’s quality
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
| Coverage | -.003333 | 0 | Adverse |
| Returned class validity | 0 | 0 | Tie |
| Returned threshold validity | +.005716 | 0 | Favourable |
| Grouped-Gower | +.007875 | 0 | Adverse |
| Neighbour-support distance | -.000432 | NOT MEASURED | Favourable, without a Stage 1 noise verdict |
| Out-of-bounds fraction | +.000667 | — | Adverse |
| Action Jaccard distance | -.028452 | — | Less diverse |
| Pairwise Gower | -.000249 | — | Less diverse |
| Actionability | 1.000000 vs 1.000000 | — | Tie |

| Dataset | Threshold-validity delta, TabICL − empirical |
|---|---:|
| Adult | -.047204 |
| Bank | +.002945 |
| German | -.002778 |
| Give Me Some Credit | -.006667 |
| HELOC | -.001333 |
| Lending Club | +.089333 |

| Backend | Mean total s/cell |
|---|---:|
| Empirical | 126.889 |
| TabICL | 1025.999 |

E3 used 20,637.625 s in generation and 5.76444 total cell-hours. The Stage 1
spread is the narrow HELOC/k=1 repeatability reference, not a six-dataset k=3
uncertainty interval.

- Outcome: TabICL-minus-empirical coverage/threshold validity/grouped-Gower/
  neighbour-support differences were `-.003333/+.005716/+.007875/-.000432`.
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
| .5 | No | .583196 | .138148 | .969 | .061883 | 190.411 |
| .5 | Yes | .551453 | .085243 | .988 | .054408 | 511.939 |
| .6 | No | .670470 | .289943 | .927 | .074386 | 285.316 |
| .6 | Yes | .646856 | .171770 | .972 | .066007 | 586.431 |
| .7 | No | .751825 | 1.000000 | .839 | .087523 | 418.545 |
| .7 | Yes | .735264 | 1.000000 | .940 | .080163 | 1003.324 |
| .8 | No | .835984 | 1.000000 | .695 | .111002 | 542.530 |
| .8 | Yes | .826365 | 1.000000 | .846 | .097235 | 1580.025 |
| .9 | No | .917131 | 1.000000 | .596 | .154544 | 654.918 |
| .9 | Yes | .914701 | 1.000000 | .666 | .139549 | 2751.191 |

The same candidates were also rescored at five evaluation thresholds. Entries
below are returned-candidate threshold-validity rates.

| Generation tau | Conditioned | Eval .5 | Eval .6 | Eval .7 | Eval .8 | Eval .9 |
|---:|:---:|---:|---:|---:|---:|---:|
| .5 | No | 1.000000 | .321295 | .138148 | .036709 | .008310 |
| .5 | Yes | 1.000000 | .183746 | .085243 | .014113 | .004057 |
| .6 | No | 1.000000 | 1.000000 | .289943 | .095879 | .010653 |
| .6 | Yes | 1.000000 | 1.000000 | .171770 | .054852 | .004090 |
| .7 | No | 1.000000 | 1.000000 | 1.000000 | .211936 | .024703 |
| .7 | Yes | 1.000000 | 1.000000 | 1.000000 | .123307 | .009332 |
| .8 | No | 1.000000 | 1.000000 | 1.000000 | 1.000000 | .089896 |
| .8 | Yes | 1.000000 | 1.000000 | 1.000000 | 1.000000 | .045989 |
| .9 | No | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| .9 | Yes | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

| E1 method, comparable four-dataset slice | Probability median | Probability mean | Fraction below .6 |
|---|---:|---:|---:|
| CounterContEx | .509904 | .547890 | .820821 |
| DiCE | .504055 | .568154 | .718377 |
| FACE | .536742 | .601131 | .707000 |
| Growing Spheres | .500023 | .554207 | .778200 |
| NICE | .533168 | .592495 | .710033 |
| Wachter | .500042 | .533893 | .852113 |

E4 consumed 47.3591 total cell-hours, of which 47.2846 were generation. The
1,600 cell/threshold rows are in
[`f4_confidence_pareto.csv`](campaign-artifacts/f4_confidence_pareto.csv).

- Outcome: achieved probability increased monotonically with generation
  threshold in all eight dataset/conditioning arms. From tau .5 to .9, the
  unconditioned arm moved from probability/coverage/Gower
  `.583196/.969/.061883` to `.917131/.596/.154544`; the conditioned arm moved
  from `.551453/.988/.054408` to `.914701/.666/.139549`. Conditioning did not
  raise confidence at fixed tau, but often preserved coverage and proximity at
  substantial runtime cost. E1 confirmed boundary hugging across all methods.

### E5 — search and diversity ablations

- Configuration: `campaign_e5_search.yaml`; six datasets × five seeds × seven
  one-axis variants, 100 factuals (210/210).
- Artifacts: `results/campaign/e5_search/`; aggregate SHA-256
  `e9875119...bfcb`.

| k=1 search arm | Coverage | Threshold validity | Grouped-Gower | Neighbour support delta | Mean total s/cell | Actionability |
|---|---:|---:|---:|---:|---:|---:|
| Mode only | .911667 | .192876 | .083567 | — | 119.825 | 1.000000 |
| Nine quantiles | .981667 | .113092 | .069148 | Reference | 69.633 | 1.000000 |
| Data-plausible refinement | .981667 | .170572 | .082331 | -.005199 | 139.833 | 1.000000 |
| Revisits off | .965000 | .113589 | .067435 | — | 68.483 | 1.000000 |

For neighbour-support distance, smaller is better; the data-plausible row's
`-.005199` is relative to the matched nine-quantile sparse arm.

| k=3 selector | Threshold validity | Action Jaccard distance | Pairwise Gower | Grouped-Gower delta vs DPP | Mean total s/cell | Actionability |
|---|---:|---:|---:|---:|---:|---:|
| DPP | .229431 | .746894 | .080104 | Reference | 674.080 | 1.000000 |
| Random | .228002 | .600300 | .066503 | — | 614.436 | 1.000000 |
| Greedy farthest | .260906 | .662453 | .085924 | +.004900 | 591.237 | 1.000000 |

| Joint-refinement diagnostic | Value |
|---|---:|
| Factuals with a scored sparse start | 2,945 |
| Fraction with positive internal-density gain | .9871 |
| Mean internal-density gain | 4.2683 |
| Action-unit delta vs matched sparse arm | +.3670 |
| Grouped-Gower delta vs matched sparse arm | +.013184 |

E5 consumed 18.97938 total cell-hours.

- Outcome: nine quantiles improved coverage, proximity, and runtime over the
  mode-only arm but reduced threshold validity. Data-plausible refinement
  recovered .057480 threshold validity and improved neighbour support by
  .005199, while adding .013184 Gower and 70.200 seconds. Among k=3 selectors,
  greedy-farthest improved both diversity orientations over DPP. For the 2,945
  factuals with a scored sparse start, joint refinement raised internal TabICL
  density in 98.71% (mean gain 4.2683), but added .3670 action units versus the
  matched sparse arm.

### E6 — context ablation

- Configuration: `campaign_e6_context.yaml`; six datasets × five seeds × four
  context sizes × predicted/true training labels, 100 factuals (240/240).
- Artifacts: `results/campaign/e6_context/`; aggregate SHA-256
  `ba8bc038...329f`.

| Context labels | Context size | Coverage | Threshold validity | Grouped-Gower | Mean total s/cell | Actionability |
|---|---:|---:|---:|---:|---:|---:|
| Predicted | 64 | .889333 | .264025 | .118627 | 177.088 | 1.000000 |
| True training | 64 | .870333 | .237338 | .127048 | 200.128 | 1.000000 |
| Predicted | 128 | .875667 | .238881 | .104258 | 231.795 | 1.000000 |
| True training | 128 | .849667 | .223822 | .123990 | 261.790 | 1.000000 |
| Predicted | 256 | .896000 | .212070 | .095112 | 311.188 | 1.000000 |
| True training | 256 | .863333 | .198569 | .127053 | 375.401 | 1.000000 |
| Predicted | 512 | .912667 | .196353 | .082489 | 449.494 | 1.000000 |
| True training | 512 | .861667 | .183253 | .121677 | 609.365 | 1.000000 |

The true-training rows are reconstructed from the audited true-minus-predicted
paired deltas at each context size. E6 consumed 21.80207 total cell-hours.

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
| C1 | 1 | Mode only | 128 | No | .864 | .296154 | .103595 | 227.637 | 1.000000 |
| C2 | 1 | 5 | 128 | No | .948 | .217709 | .089681 | 139.119 | 1.000000 |
| C3 | 3 | 5 | 256 | No | .983 | .291855 | .064201 | 341.726 | 1.000000 |
| C4 | 3 | 9 | 512 | Yes | 1.000 | .172385 | .053046 | 848.042 | 1.000000 |

Each entry is a macro mean across E7's four datasets and five seeds.
Dataset-level runtime and threshold-validity means and standard deviations are
in [`f5_cost_quality.csv`](campaign-artifacts/f5_cost_quality.csv). E7 consumed
8.64736 total cell-hours; C4's Lending Club mean was 2768.878 s/cell.

- Outcome: C1–C4 coverage/threshold validity/Gower/seconds were
  `.864/.296154/.103595/227.637`, `.948/.217709/.089681/139.119`,
  `.983/.291855/.064201/341.726`, and `1/.172385/.053046/848.042`.
  C3 (five quantiles, context 256, predicted labels, k=3, DPP) was frozen for
  E9/E10 before either result was inspected.

### E8 — robustness rescoring

- Configuration: no generation matrix; read-only rescoring of all 120,709 E1
  candidates with LR, MLP, and XGBoost instruments at seeds 137/271/811.
- Artifacts: `results/campaign/analysis/e8_robustness/`; source-tree digest
  `cf79ad2b...2008`; table/detail/manifest SHA-256 values
  `c4beb207...1e3b`, `5c44865d...8163c`, `610f3c74...f9d9`.

| Original target-probability bin | Rescored rows | Retained class validity |
|---|---:|---:|
| [.5, .6) | 263,421 | .861959 |
| [.6, .7) | 52,716 | .955004 |
| [.7, .8) | 24,957 | .979365 |
| [.8, .9) | 12,654 | .998815 |
| [.9, 1] | 8,379 | .999642 |

| Original method | Rescored rows | Retained class validity |
|---|---:|---:|
| CounterContEx | 61,455 | .899113 |
| DiCE | 60,138 | .871230 |
| FACE | 61,650 | .927981 |
| Growing Spheres | 61,644 | .874213 |
| NICE | 56,205 | .912641 |
| Wachter | 61,035 | .865323 |

| Evaluation instrument | Rescored rows | Retained class validity | Seed variation |
|---|---:|---:|---|
| Logistic regression | 121,242 | 1.000000 | None under the fixed full-sample fit |
| MLP | 120,255 | .673461 | Seeds 137/271/811 vary |
| XGBoost | 120,630 | 1.000000 | None under the fixed full-sample fit |

The 362,127 rescored rows equal 120,709 E1 candidates times three retraining
seeds. Dataset-pooled retention ranged from .794582 for Bank to .959727 for Give
Me Some Credit. E8 was a read-only scoring pass and has no generic lifecycle
timing manifest.

- Outcome: pooled validity retention rose from .861959 in probability bin
  `[.5,.6)` (n=263,421 rescored rows) to .999642 in `[.9,1]` (n=8,379),
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
| Adult | TabICL | 1.000000 ± 0 | .174089 ± 0 | .095318 ± 0 | .113645 ± 0 | 699.054 |
| Adult | TabPFN v2 | 1.000000 ± 0 | .162419 ± .010184 | .100210 ± .000344 | .116064 ± .000532 | 3148.137 |
| HELOC | TabICL | .932000 ± 0 | 0 ± 0 | .020393 ± 0 | .046465 ± 0 | 405.066 |
| HELOC | TabPFN v2 | .913333 ± .006110 | 0 ± 0 | .019247 ± .000465 | .046396 ± .000141 | 2892.578 |

Values are three-seed means ± sample standard deviations. The complete
dataset/backend rows are in
[`t3_backend.csv`](campaign-artifacts/t3_backend.csv). E9 consumed 5.95403 total
cell-hours.

- Outcome: TabPFN-minus-TabICL threshold validity/Gower were
  `-.011670/+.004892` on Adult and `0/-.001145` on HELOC; HELOC coverage fell
  .018667. TabPFN was 4.50× and 7.14× slower. This demonstrates backend
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
| Adult | `00cea69e` | 1,000 | 3 | 1.000 | 1.000 | .176272 | .097852 | .115369 | 1.000 | 2367.934 |
| Bank | `9119b938` | 1,000 | 3 | 1.000 | 1.000 | .496662 | .038434 | .061442 | 1.000 | 1806.811 |
| German | `40d888cb` | 120 | 3 | 1.000 | 1.000 | .002778 | .063975 | .250785 | 1.000 | 686.165 |
| GMSC | `993dcadb` | 1,000 | 3 | 1.000 | 1.000 | .449667 | .092073 | .049263 | 1.000 | 452.274 |
| HELOC | `0eb68d5b` | 1,000 | 3 | .925 | 1.000 | 0 | .021709 | .046502 | 1.000 | 1733.816 |
| Lending Club | `8890a2c3` | 1,000 | 3 | 1.000 | 1.000 | .206296 | .101839 | .057693 | 1.000 | 1051.546 |

German Credit has 120 test factuals after the retained deterministic split, so
its `max_test: 1000` cap does not fabricate additional rows.

E10 consumed 2.24960 measured cell-hours: prepare .108 s, generate 8027.883 s,
evaluate 66.745 s, write .376 s, and total 8098.546 s. The configuration was
not revised after inspection.

## Compute accounting

The measured campaign lower bound is **133.11459 cell-hours**, versus the
planning estimate of approximately 120 hours. It is the Stage 10 audited
124.91096 hours through E7 (including Stage 1 feasibility and unique published
superseded E1 DiCE attempts), plus E9’s 5.95403 and E10’s 2.24960 hours. E8 is a
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
