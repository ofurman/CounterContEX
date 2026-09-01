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
