# CounterContEx paper campaign results

This document is the traceability source for the paper experiment campaign run
on the `gx10-bdc5` GB10. Scientific specifications are the tracked matrices in
`experiments/zeroshot_cf/configs/matrices/`; canonical payloads are under
`experiments/zeroshot_cf/results/campaign/`. Generated tables and figures are
committed in [`campaign-artifacts/`](campaign-artifacts/).

## Reading the metrics

Coverage divides factuals with at least one returned candidate by all factuals.
Returned-class and returned-threshold validity divide successes by available
candidates only. Per-requested-slot rates retain unavailable slots in their
denominator; per-factual rates count factuals with at least one success. Primary
metrics use rank 0 and set metrics use the full returned set. Grouped-Gower and
continuous proximity use returned target-class candidates; sparsity,
actionability, bounds, LOF, Isolation Forest, and neighbour support use every
available candidate. Detectability always travels with its measured arm counts
and status.

## Experiments

### E1 — main comparison

- Configuration: `campaign_e1_main.yaml`; six datasets × three target families
  × five seeds × six methods, 250 factuals, k=1 (540/540 complete).
- Artifacts: `results/campaign/e1_main_dice_v5/`; aggregate SHA-256
  `604eebe8...bcf991`; paper products `t1_main.*`,
  `f3_critical_difference.*`, `f6_target_probability.*`.
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
- Outcome: larger predicted-label contexts improved proximity and eventually
  coverage while reducing threshold validity and increasing runtime. True
  training labels were worse at every size on coverage, threshold validity,
  proximity, and runtime. No test or factual labels enter method context.

### E7 — cost-quality Pareto and freeze

- Configuration: `campaign_e7_cost.yaml`; four datasets × five seeds × four
  configurations, 250 factuals (80/80).
- Artifacts: `results/campaign/e7_cost/`; aggregate SHA-256
  `77aaf820...3b4b`; paper product `f5_cost_quality.*`.
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
- Outcome: TabPFN-minus-TabICL threshold validity/Gower were
  `-.011670/+.004892` on Adult and `0/-.001145` on HELOC; HELOC coverage fell
  .018667. TabPFN was 4.50× and 7.14× slower. This demonstrates backend
  portability, not a quality gain.

### E10 — frozen 1,000-factual headline

- Configuration: `campaign_e10_headline.yaml`; six datasets, seed 42, logistic
  regression, frozen C3, 1,000 factuals, k=3 (6/6).
- Artifacts: `results/campaign/e10_headline/`; aggregate SHA-256
  `df781753...2941b`; run IDs and headline metrics are below. Paper product
  `f7_qualitative_case.*` uses matched HELOC source index 21 and Adult source
  index 10 with CounterContEx, NICE, and DiCE.

| Dataset | Run ID prefix | Coverage | Class validity | Threshold validity | Gower | Neighbour support | Actionability | Total s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Adult | `00cea69e` | 1.000 | 1.000 | .176272 | .097852 | .115369 | 1.000 | 2367.934 |
| Bank | `9119b938` | 1.000 | 1.000 | .496662 | .038434 | .061442 | 1.000 | 1806.811 |
| German | `40d888cb` | 1.000 | 1.000 | .002778 | .063975 | .250785 | 1.000 | 686.165 |
| GMSC | `993dcadb` | 1.000 | 1.000 | .449667 | .092073 | .049263 | 1.000 | 452.274 |
| HELOC | `0eb68d5b` | .925 | 1.000 | 0 | .021709 | .046502 | 1.000 | 1733.816 |
| Lending Club | `8890a2c3` | 1.000 | 1.000 | .206296 | .101839 | .057693 | 1.000 | 1051.546 |

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
