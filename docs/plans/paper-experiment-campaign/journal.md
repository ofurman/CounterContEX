# Journal

Append-only. Newest entries at the bottom. Never rewrite an earlier entry.

One entry per invocation, in this shape:

```
## YYYY-MM-DD HH:MM -- Stage N: [Name] -- DONE
**Did**: [1-3 lines]
**Verification**: GATE lines passed. REPORT values: [metric]=[value]
**Provenance**: [per measured GATE: the input the value was read from, and the defect that would
turn it red] - [or `NOT MEASURED` for any that could not be produced from this run's own inputs]
**Problems**: [symptom -> root cause -> resolution -> inline/subagent] or "none"
**Commit**: `abc1234`
```

---

## 2026-09-01 22:25 -- Stage 1: Feasibility and noise floor -- DONE
**Did**: Confirmed the campaign branch was already published by `de65436`; provisioned the
locked GB10 environment, pinned CEL checkout, and verified TabICL checkpoints. Constructed
Adult and German Credit cases, ran their n=25/k=3 cost cells, ran five HELOC n=100/k=1 seeds,
repeated seed 42 under a second output root, and profiled n=25/k=3 HELOC versus Lending Club.
Adult has train/validation/test 19200/4800/6000, 191 features, 8 one-hot groups, 9 actionable
units, 83 immutable columns, test balance 4552/1448 and LR test accuracy 0.845. German Credit
has 384/96/120 rows, 57 features, 11 one-hot groups, 17 actionable units, 4 immutable columns,
test balance 60/60 and LR test accuracy 0.725.
**Verification**: GATE Adult's constructed `BenchmarkCase` has 9 actionable units and 83
immutable encoded columns. GATE seed-42 repeat produced run ID
`00c4a01d90256e5e6735e0c63a969bd05fc1f8b7d1993d772d3c5bc374705194` in both roots and
byte-identical `summary.csv` (SHA-256
`a68b944c1d6290735ad65f429edd42597b8ae65ca7adff5b11d54456ac915775`). REPORT HELOC
five-seed values: `proximity_grouped_gower` mean 0.0141824417063, sample std/range 0/0;
`action_unit_sparsity_mean` 1.32, 0/0; `validity_returned_class` 1.0, 0/0; coverage 1.0,
0/0. REPORT generation cost: Adult 37.194 s / 25 = 1.488 s/factual; German Credit 83.472 s /
25 = 3.339 s/factual. REPORT Lending Club: 21.767 s/factual versus HELOC 1.230; one depth-65
outlier used 498.785/544.167 s while trying and failing to fill the 16-row pool (13 returned to
the pool, k=3), so the pool-fill stopping rule causes the long tail; B-1 records the candidate
fix. Full values and artifact pointers are in `resources/compute-budget.md`.
**Provenance**: Adult/German gate values were read from `FeatureSchema` and arrays of cases
constructed from the pinned CEL revision; removing or failing to propagate YAML actionability
would make the Adult gate red. Determinism was read from the two published manifests and a
byte comparison of their independently written summaries; an ambient RNG path or unpinned
identity input would change the hash or run ID. All REPORT values were read from this run's
published manifests/summary files under `results/campaign/stage1/`; no literal or expected
fixture supplied a verdict.
**Problems**: Historical ignored `results/local/full_reference/` and
`architecture_full_reference/` are absent in the fresh GB10 clone, so the brief's prescribed
n=25 HELOC/Lending Club fallback was executed. Those paths remain designated read-only
historical evidence; their frozen baseline pointer is `docs/papers/positioning-draft.md` §1.4.
The long-tail fix changes scientific behavior, so it was deferred as B-1 rather than applied.
The full repository gate passed 265 tests. No Python file changed, so changed-package Ruff is
vacuously clean; an informational whole-suite Ruff run exposed 24 pre-existing violations in
untouched files and was not altered or treated as Stage 1 evidence.
An independent read-only provenance audit reconstructed Adult through the production loader
and matched case `48c17e...3182b5` and dataset fingerprint `86e311...159f5b`; it independently
compared the two HELOC artifacts and corroborated every new Stage 1 REPORT value. It marked the
older n=1000 timing table NOT MEASURED on this host because its ignored source tree is absent;
those historical values remain attributed only to the frozen positioning draft/prior run.
**Commit**: `HEAD` (this stage commit)


## 2026-09-01 22:55 -- Stage 2: Target-model registry -- DONE
**Did**: Added a dataset-owned fixed target-model registry for LR, MLP, and XGBoost; added the
lazy XGBoost discriminator arm and explicit locked dependency; replaced the orchestration
family check with lookup; preserved the historical LR identity and family-separated physical
caches; and added the backward-compatible `target_models` matrix axis. Documented the registry
and ignored campaign artifact root.
**Verification**: GATE the HELOC fitted model IDs are LR
`7cf029aad02cdbc1013f74b4829bc65b1b9aafa3247344e3e90a77d151a1ceb0`, MLP
`4e225ea3751c3f095274dac7b0cf542c5fbb1852527c08fa5056027a5a377dcf`, and XGBoost
`76500e835f5311d3e168bc7b0fdedf3e7971fb58fb8fe276e5b1ed783da9a65a`, backed by distinct
`disc_..._{lr,mlp,xgb}.pkl` files; LR exactly matches the Stage 1 manifest. GATE Wachter/XGBoost
run `5500593...b1381` returned one candidate with `coverage=1.0` and
`validity_returned_class=1.0`. GATE dry-run cells
`d0f6ff8...56eed` and `46be4bf...f6786` differ only in target-model fields. REPORT held-out
test accuracies (LR/MLP/XGB): HELOC .7408/.7322/.7484; Bank Marketing .9029/.9086/.9089;
Give Me Some Credit .7161/.7152/.7774; Lending Club .7787/.7803/.7813; Adult
.8450/.8442/.8597; German Credit .7250/.6667/.6583. REPORT all 15 baseline-family cells ran
and returned one target-class-valid candidate; D-4 records the table disposition.
**Provenance**: Model IDs were read from cases constructed through `_default_case_loader` and
the three physical cache files; omitting family from the effective discriminator cache filename
or hashing declared params without fitted state turns this gate red. Wachter validity was read
from the canonical result summary produced by the XGBoost case and method-blind evaluator; a
candidate checked against a reused LR oracle would change the manifest model identity and fail
the cache/model gate. Matrix equality was computed from resolved dry-run JSON after removing
only `cell_id` and `target_model`; failure to include target models in the Cartesian product
would leave only one row. Accuracy values came from predictions over each constructed case's
full held-out `X_test`/`y_test`, not validation logs or literals.
**Problems**: A real measurement found MLP identity initially depended on the memory address in
NumPy `RandomState.__repr__`; canonical serialization now hashes RNG state content, with a
regression test and repeated-cache-load witness. A cosmetic annotation edit also changed the
wrapper class source digest; its exact historical source was restored and a narrow existing-line
Ruff exemption preserves the Stage 1 LR fingerprint. Full suite: 270 passed; changed-file Ruff,
offline retained CLI, and the existing 24-cell matrix dry-run passed.
An independent provenance audit passed all three gates and corroborated all 15 COMPLETE
method-family cells. It confirmed three active cache paths with distinct hashes; three
byte-identical orphan files from the discarded doubled-suffix experiment remain ignored and
are not evidence. It also identified that cache loading does not validate training params or
implementation metadata; current files were freshly trained under this registry, and B-2
records the durable future-cache risk.
**Commit**: `HEAD` (this stage commit)

## 2026-09-01 23:20 -- Stage 3: Evaluation metrics v2 -- DONE
**Did**: Froze every paper-facing metric in the single
`countercontex.evaluation.v2` bump. Preserved the existing per-slot target-probability array;
added orientation-independent paired-fold detectability AUC with target-label-matched real rows,
explicit arm counts and `NOT_MEASURED`; added per-available-candidate fifth-neighbour
grouped-Gower support and its mean; documented populations and the joint-density decision D-5.
**Verification**: GATE the pre-change v1 deterministic `partial_k3` summary and the v2 summary
after removing only five new fields have identical canonical SHA-256
`0c211640ecfc1d17a1c2a87c0c1f207bf21e6ac2224bd0e2878a9079c0166f53`. GATE copied-real
detectability AUC is exactly 0.5, the +5 OOD fixture is 1.0, and an empty CF arm reports null
AUC/status `NOT_MEASURED` with count 0. REPORT real HELOC/NICE n=40 run
`f101d0354c6a7cf33b0574aa0c3974146e40226bd5d4bb1af3859180c09a2935`: detectability AUC
0.674556 with status `MEASURED` and 26 real/26 CF rows; fifth-neighbour grouped-Gower mean
0.044089; `common.target_probabilities` shape 40×1 with 26 finite returned slots; neighbour
array length 26. Both manifest evaluation identities are v2.
**Provenance**: The legacy hash was measured from production evaluator output at Stage 2 HEAD
`1fb9097` before edits, then recomputed from this run's v2 legacy subset; denominator drift in
availability/class-success masks turns it red. Detectability values are out-of-fold predictions
from fixed standardized LR and paired five-fold splits over the fixture inputs; returning a
literal 0.5, using an empty arm, or separating identical twins across folds turns its witnesses
red. Real REPORT values and probability/kNN shapes were read back from the published
`summary.csv`, `manifest.json`, and `arrays.npz`, not from in-memory results.
**Problems**: Ordinary stratified CV measured copied identical rows at AUC 0.387 because twins
landed unevenly across folds; paired arm folds fixed the instrument and the null/OOD witnesses
now distinguish degeneracy from signal. The first n=25 real smoke had only 17 valid CF rows and
honestly recorded `NOT_MEASURED`; n increased to 40 under a new output root without changing a
metric setting. Full suite 274 passed; changed-file Ruff, offline CLI, and 24-cell dry-run passed.
An independent audit reproduced both gates and verified the complete real artifact down to
array/CSV masks and means. It notes that detectability is a fixed linear probe over only
target-class CFs, uses deterministic leading target-matched reference rows, folds AUC to
[0.5, 1], and has only 26 rows per real arm here; these limitations travel with the metric and
prevent interpreting it as universal indistinguishability. K-th-neighbour support deliberately
uses all available returned candidates.
**Commit**: `HEAD` (this stage commit)

## 2026-09-01 23:40 -- Stage 4: Diverse baseline adapters -- DONE
**Did**: Wired the requested set size through DiCE's native genetic `total_CFs`, collected
distinct target-class returns across configured restarts, and preserved shortages as NaN-backed
unavailable slots. Post-processing now revalidates, removes factual rows, and deduplicates
candidates that collapse during pruning/contraction. Flipped the capability and bumped the
scientific implementation identity from `dice-v1` to `dice-v2`. D-6 records why NICE, Wachter,
Growing Spheres, and FACE retain their single-counterfactual guards.
**Verification**: GATE the deterministic k=3 fixture requested three candidates and returned
three distinct rows, none equal to the factual; its two-of-three witness has availability
`[true, true, false]` with the final candidate all NaN. GATE two independent real HELOC n=1,
k=3 runs at seed 42 published the same run ID
`9f5430462ba5d59e6a0df3b8741c0d6fb89a0e90cc26e7dc5d1831ea7889202a` and byte-identical
`arrays.npz` SHA-256 `d0cd8627fa094759c6ff5523af29e317da16a88537c00d528fb62e347b2f9249`;
the manifest resolves `method_implementation=dice-v2`. REPORT all three real candidates were
available and target-class valid: `set_coverage_at_k=1.0`, `set_returned_count_mean=3.0`,
`set_action_jaccard_mean=0.266667`, and `set_pairwise_gower_mean=0.0195753`.
**Provenance**: Fixture assertions read the returned candidate cube and mask, while the real
determinism gate compared every named array and the complete NPZ bytes under separate output
roots. Set metrics came from the canonical `summary.csv`; candidate count and implementation
identity came from `arrays.npz` and `manifest.json`. Repeating a candidate, inserting a factual,
using ambient RNG state, or retaining `dice-v1` turns a gate red.
**Problems**: DiCE pruning can legitimately collapse distinct raw candidates to the same
actionable endpoint; the adapter therefore deduplicates after pruning as well as before it. Full
suite: 275 passed; changed-file Ruff passed. An independent provenance audit passed every gate,
re-read the real artifacts and manifest, and corroborated D-6 against all five baseline
implementations. It notes that the shortage witness is adapter-level rather than a persisted
run, while the evaluator independently defines coverage@k from returned count; it also confirms
the three real rows only barely cross the class boundary and all fail tau=0.7, so these diversity
metrics characterize returned target-class candidates, not threshold-valid successes.
**Commit**: `HEAD` (this stage commit)

## 2026-09-02 00:10 -- Stage 5: Analysis layer -- DONE
**Did**: Added an artifact-only analysis package and `cli analyze`. Exact matrix membership is
validated before seed grouping by the complete scientific identity minus seed; numeric metrics
emit mean, sample standard deviation, and finite n. Added paired Wilcoxon tests with joint Holm
correction, explicit statistic/p/n/effect, and a below-noise-floor flag; an average-rank Demsar
diagram; auditable CSV-backed F3--F7 builders; and common-source CSV/LaTeX T1--T3 builders. D-7
records the validated read-only path needed to analyze published evaluation-v1 artifacts.
**Verification**: GATE a deliberately incomplete matrix root raises
`partial run directories are not aggregateable` before producing a survivor average. GATE
eight identical pairs return statistic 0, corrected p=1, n=8, non-significant; eight unit-gap
pairs return corrected p=0.0078125 and significant, while a significant 0.01 effect against a
0.02 noise floor is explicitly flagged. GATE all eight named figure/table builders expose only
`output_root`, `matrix_config`, and `output_dir` path inputs; analysis imports no method package
and contains no paper metric literals. A real Stage 4 smoke wrote all 18 F3--F7/T1--T3,
significance, and manifest products from canonical artifacts.
**Report**: The analysis layer read all five exact Stage 1 HELOC noise-floor cells and wrote
`results/campaign/stage5/noise_floor_aggregate.csv` (SHA-256
`d022c7df598f980f7e08c18ed65459b66db6b2d0f64bd504056a5fdba5bac94a`). Actual seed n=5;
coverage 1.0 +/- 0, returned-class validity 1.0 +/- 0, threshold validity 0 +/- 0,
grouped-Gower 0.01418244 +/- 0, and action-unit changes 1.32 +/- 0. The only nonzero reported
spread is floating roundoff in legacy sparsity (`7.76e-18`), not stochastic method variation.
**Provenance**: Loader gates derive expected cells from the matrix and rederive run and cell
hashes from each manifest identity; deleting COMPLETE, changing any scientific field, or
removing a requested seed turns aggregation red. Statistical witnesses supply independent
paired vectors rather than copying implementation output. Builders read summary/manifest/NPZ
paths and write the precise CSV used by each figure. Stage 1 values were decoded from its typed
published summary tables, not transcribed from the earlier journal.
**Problems**: The v2-only `ArtifactStore.read()` correctly rejected historical v1 reports;
analysis now owns a narrow read-only compatibility path instead of weakening current evaluator
validation. The first independent audit found that compatibility matching discarded all
evaluation settings and omitted the redundant top-level run ID check; the delegated fix now
normalizes only schema-added fields and has threshold/run-ID witnesses. A fresh audit then found
unknown schemas could enter the legacy path; the second and final light fix explicitly admits
only v1 and rejects v3 or mixed-schema roots. A third fresh audit passed every gate and reproduced
the Stage 1 hash and Stage 4 smoke. Full suite: 283 passed; analysis/CLI Ruff and offline CLI help
passed.
**Commit**: `HEAD` (this stage commit)

## 2026-09-02 00:55 -- Stage 6: Matrices and DGX launchers -- DONE
**Did**: Added every executable matrix named by the catalog, fixed its output root, and froze
the resolved counts with a contract test. D-8 corrects the plan's arithmetic: there are nine,
because E8 is explicitly a read-only rescoring pass and must not generate a tenth matrix. Added
GB10 `nohup` launchers for Stages 7--12, strict post-run aggregation, logs/PIDs/success markers,
and retrieval instructions. Stage 11 publishes `E9_DONE`, not a false stage-DONE marker before
E8. Updated the runbook to the actual matrix-owned CUDA setting and safe fast-forward pulls.
**Verification**: GATE all 1,384 resolved cells across all nine matrices are unique, carry
`countercontex.evaluation.v2`, and expose the inspected target model, backend, k, and seed. Frozen
counts: E1=540, E2=60, E3=36, E4=200, E5=210, E6=240, E7=80, E9=12, E10=6. GATE on
`gx10-bdc5`: CUDA=true/NVIDIA GB10; CEL revision
`3587f943826f6b087a0d198c8c4aa4373712c7ee`; both checkpoint content hashes verified; offline
TabICL smoke passed. GATE the exact E1 HELOC/LR/seed-42/CounterContEx/k=1 cell
`6c979577...0a8b6` published run `6d89241e...0484e2` with all six required files and COMPLETE;
strict single-cell aggregation accepted it.
**Report**: The n=250 smoke returned 249 candidates: coverage 0.996, returned-class validity
1.0, threshold validity 0.0, grouped-Gower 0.0132439. Generation took 1236.094 s, or
4.94438 s/factual; total was 1237.330 s. The scientifically equivalent Stage 1 seed-42 n=100,
k=1 cell measured 4.79635 s/factual, so the smoke is 3.09% slower; against the five-seed Stage 1
mean 5.39462 +/- 0.35936 s/factual it is 8.35% faster. The earlier 1.230 s/factual HELOC profile
used k=3 and n=25 and is not treated as an equivalent runtime baseline.
**Provenance**: Counts came from complete dry-run JSONL rows, with duplicate cell IDs checked
within and across matrices. Environment values came from torch, vendor verification, checkpoint
verification, and the offline smoke—not host assumptions. Runtime and metrics were read from the
published E1 manifest/summary; the smoke cell ID was matched back to exactly one E1 matrix row.
Changing a target family, seed, k, backend, metric version, or scientific config turns these
checks red. Launcher syntax passed `bash -n`; DONE markers occur only after matrix and aggregate.
**Problems**: E5/E6/E7 and E9 deliberately declare scientific settings whose owning method
support is implemented in Stages 10 and 11 before execution; Stage 6 proves identity expansion,
not those future capabilities. E9 and E10 counts/roots are frozen now, while Stage 10 must replace
their provisional method parameters with the disclosed selected configuration before either
executes. The first audit found E2/E7 identity collisions, an explicit-default E5 confound, and
Boolean coercion of E6's true-label name; the delegated matrix fix now has cross-matrix uniqueness
and raw-axis/type witnesses. It then found stale DONE markers could survive a relaunch; all six
launchers now synchronously clear the exact marker they pass to the child, with a contract witness.
Full suite 288 passed; matrix/launcher-test Ruff, launcher syntax, and diff checks passed. A fresh
independent provenance audit follows before the stage commit.
**Commit**: `HEAD` (this stage commit)

## 2026-09-02 14:35 -- Stage 7: E1 main comparison -- DONE
**Did**: Executed the full E1 k=1 comparison on `gx10-bdc5`: six datasets, three target-model
families, five seeds, and six methods. Strict aggregation accepted exactly 540 cells under
`results/campaign/e1_main_dice_v5`; the Stage 5 artifact-only analysis wrote 18 products under
`results/campaign/analysis/e1_main`, including T1, F3, F6, and one joint Holm family over all
five T1 metrics. D-9 records the DiCE categorical-vocabulary repair and `dice-v5` identity.
Every canonical manifest retains prepare/generate/evaluate/write/total timings per cell.

**Verification**: GATE all 540 run directories contain `COMPLETE`; each of six datasets has 90
cells, each method 90, each seed 108, and all request exactly one counterfactual. All 540 resolve
`countercontex.evaluation.v2`; implementations are CounterContEx v3, DiCE v5, FACE v1,
Growing Spheres v1, NICE v1, and Wachter v1. GATE a fresh production case load for every one of
the 18 dataset-family pairs reproduced the exact single `model_content_id` present in its 30
manifests; no wrong-family cache was observed. Strict aggregation reported 540/540, so cells
absent by the Stage 2 expected-failure policy: **0**.

**Report**: Across the 18 dataset-family blocks, T1 method means +/- sample SD are shown below;
the metrics are primary coverage, returned-class validity, threshold validity, grouped-Gower
proximity, and action-unit changes, respectively:

- CounterContEx: .997111 +/-.008547; 1.000000 +/-0; .124030 +/-.160885;
  .055860 +/-.035722; 1.288793 +/-.461191.
- DiCE: .976493 +/-.038741; 1.000000 +/-0; .111434 +/-.096086;
  .091708 +/-.046491; 1.687267 +/-.555783.
- FACE: 1.000000 +/-0; 1.000000 +/-0; .196667 +/-.214299;
  .110146 +/-.063182; 4.360259 +/-2.001345.
- Growing Spheres: .999911 +/-.000377; 1.000000 +/-0; .116507 +/-.125890;
  .060882 +/-.032454; 1.313916 +/-.667387.
- NICE: .916685 +/-.164459; 1.000000 +/-0; .142359 +/-.141368;
  .055139 +/-.032719; 1.574491 +/-.549634.
- Wachter: .985833 +/-.036554; 1.000000 +/-0; .056520 +/-.067908;
  .060144 +/-.034712; 1.200064 +/-.336631.

The 25 jointly Holm-corrected CounterContEx-minus-baseline tests use n=18 paired blocks each.
Significant results are coverage vs DiCE (difference +.020619, corrected p=.023501), proximity
vs DiCE (-.035848, p=.002563) and FACE (-.054287, p=.001907), and action-unit changes vs DiCE
(-.398474, p=.009232) and FACE (-3.071466, p=.004510). The other 20 corrected p-values are
non-significant; exact statistics, effects, per-metric noise-floor flags, and p-values are in
`significance.csv`. F3 truthfully ties every method at average rank 3.5 because its frozen
returned-class-validity metric is 1.0 for every block; F6 contains 120,709 candidate probability
rows.

Canonical phase totals are prepare 42.072 s, generate 45,856.782 s, evaluate 520.188 s, write
5.468 s, and total 46,447.199 s (12.9020 h). Mean total seconds/cell +/- SD by method are
CounterContEx 469.813819 +/-463.589587, DiCE 25.965897 +/-11.375246, FACE 12.594671
+/-10.861771, Growing Spheres 2.163561 +/-1.338030, NICE 1.689811 +/-1.467083, and Wachter
3.852232 +/-1.610667. Including published superseded DiCE v2/v3/v4 attempts gives a measured
compute-consumption lower bound of 14.3626 h; canonical artifact wall span is 14.6555 h.

**Artifacts**: aggregate SHA-256 `604eebe8...bcf991`; T1 `5cb787a9...615f1`; F3 CSV
`1f3d2cb8...abd0`; F6 CSV `0c2446ed...16ba9c`; significance `9bcc3932...f53dc`.
Historical evidence remains untouched in `e1_main` (DiCE v2 failure), `e1_main_dice_v3`
(DiCE v3 failure), and `e1_main_dice_v4` (DiCE v4 failure); logs are `launch/stage07.log`,
`stage07_dice_v4.log`, and `stage07_dice_v5.log`. Recoverably quarantined copied obsolete runs
remain in the adjacent v3/v5 quarantine roots and never entered strict aggregation.

**Provenance**: Membership came from the matrix's freshly resolved 540 cell IDs and strict
`ArtifactStore.aggregate_expected`, not directory count alone. Evaluation and implementation
versions, scientific k, model IDs, and timings were read from each canonical manifest. Model IDs
were independently recomputed through `_default_case_loader`; loading a classifier from another
family turns that equality red. T1 and statistics read only canonical artifacts; deleting a
marker, changing a scientific field, or omitting a seed makes the loader reject the root. The
compute lower bound sums canonical timings once plus only superseded DiCE implementation rows,
avoiding copied non-DiCE double counting.

**Problems**: The original run stopped at 470 cells when DiCE rejected German Credit `purpose`;
v3 declared permitted ranges but then stopped on Give Me Some Credit with an unseen encoder
label. v4 refit encoders but its German query dummy width no longer matched the prebuilt KD-tree.
The focused missing-category witness led to v5 categorical dtype metadata plus full encoders;
the exact German LR/seed-17 n=120 smoke passed in 13.27 s before the resumable full run. All
failed roots were preserved. The first E1 analysis also exposed a Stage 5 builder that tested
only the universally tied returned-class validity; TDD expanded it to all five T1 metrics in one
Holm family. Full suite: 290 passed; 24 focused tests, changed-file Ruff, diff check, offline CLI,
540-row dry-run, strict aggregation, and analysis rebuild passed. A fresh independent audit
accepted exact 540/540 membership, production-recomputed all 18 model fingerprints, validated
every available DiCE-v5 candidate's target class, atomic groups, immutable columns, and NaN
shortages, rebuilt T1/F3/F6/significance byte-identically, and independently reproduced the
single 25-test Holm family. It passed 27 focused tests and the 290-test full suite; its only
caveat is the disclosed, valid but uninformative F3 tie.
**Commit**: `HEAD` (this stage commit)
