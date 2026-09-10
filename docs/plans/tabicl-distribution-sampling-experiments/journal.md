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

## 2026-09-10 02:16 -- Stage 1: Freeze sampling contracts and witnesses -- DONE
**Did**: Added a pure NumPy proposal-policy kernel with stable SHA-256-keyed uniforms, fixed equal-mass bin truncation, categorical truncation, log-space pi-beta weights, resampling, and post-projection no-op/duplicate accounting. Added independent analytical fixtures and a minimal authenticated nine-quantile witness from the preserved E3 HELOC trace.
**Verification**: GATE passed: the focused proposal-policy suite passed `17` tests, scoped Ruff passed, and `git diff --check` passed. The full repository suite passed `332` tests with `5` pre-existing warnings. An independent verifier returned PASS after mutation-oriented review. REPORT values: analytical_fixture_sha256=`75d3ac1e10c8ba1a70a7afd62987d904826f4e9c61b86623bf8c22144a3aa66f`, e3_witness_fixture_sha256=`d9cf18eb60874ed62e2171600ad651b56d948edab87d9ee9394cbd3094581b66`, historical_source_sha256=`d50d476bb5d476f96cd39cf757b59a0a68be5164b12f74a85b502ac90324fb8a`, historical_complete_sha256=`49c50de0e41be3d31817e895135fd5c2f4f8afddbfe68b32a2263edf7db1f687`, historical_trace_analysis_sha256=`6596636a03da030d13f681e6169b79b4ed9173dc92b55f855e75212131bc0eda`, baseline_raw/projected/unique_classifier/no_op/duplicate=`5/5/2/2/1`. The analytical fixture defines categorical `q=[.45,.30,.20,.05]`, classifier scores `[.10,.80,.40,.20]`, hand-normalized beta-one weights `[.12,.64,.213333,.026667]`, an eight-bin density-order witness, and projection rows containing one-hot, immutable, no-op, and duplicate cases.
**Provenance**: The focused GATE reads the committed analytical JSON and E3 witness plus the live policy kernel; wrong normalization or an extra q factor changes hand-derived weights, log normalizer, or ESS, process-dependent RNG changes the golden canonical-bytes/SHA-256/uniform triple, and dropped or misclassified projected rows change explicit reverse links and all five budget counts. Historical provenance was reauthenticated from `experiments/zeroshot_cf/results/diagnostics/tabicl-empirical-20260908/heloc-v2/00_tabicl_common.json`, its `COMPLETE.json` inventory entry, and the campaign `trace-analysis.json`; a missing member from a partially materialized bundle, hash mismatch, wrong inventory link, selector drift, or changed nine values turns the test red. If the complete ignored source bundle is absent in another checkout, the source reauthentication test reports an explicit skip while the committed hash and value witness remains mandatory. Ruff reads both changed Python files; style/import defects turn it red. The full suite reads all collected repository tests and the changed imports; collection or behavioral regression turns it red.
**Problems**: The first within-bin tail test exposed IEEE rounding of the declared final-bin expression to exactly `1.0` -> capped the final coordinate at `nextafter(1, 0)` while retaining the protocol's `2^-53` within-bin clip -> inline. The first independent review found missing golden RNG evidence, untested normalizer/ESS, and optional-only E3 authentication -> added immutable golden values, record validation, and explicit source/COMPLETE/trace hash verification, then obtained verifier PASS -> subagent.
**Commit**: `pending`
