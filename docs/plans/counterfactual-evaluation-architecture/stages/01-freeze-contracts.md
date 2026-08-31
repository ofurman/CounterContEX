# Stage 1: Freeze Contracts and Compatibility Evidence

**Goal**: Turn the current public behavior and intended availability/validity semantics into small, deterministic fixtures before moving ownership boundaries.
**Dependencies**: None
**References**: [`architecture.md`](../resources/architecture.md), [`current-state.md`](../resources/current-state.md)

---

## Steps

1. Define the v1 compatibility inventory.
   - Where: new `experiments/zeroshot_cf/tests/fixtures/architecture_v1/README.md` and machine-readable fixture files.
   - Record current method IDs, common summary columns, common point columns, per-method legacy filename stems, required NPZ keys, public CLI commands, and `generate_counterfactual_batch()` import path.
   - Derive fixtures from reasoned synthetic cases and existing documented contracts; do not fabricate rows to satisfy a count or assume ignored local results exist.

2. Add semantic fixtures that distinguish output states.
   - Where: new `test_generation_result_contract.py` and `test_evaluation_semantics.py` under `experiments/zeroshot_cf/tests/`.
   - Cover returned-valid, returned-invalid, unavailable, best-effort-only, threshold-failing target-class, `k=1`, partial `k>1`, and duplicate/invalid padding rejection.
   - State which existing behavior is compatibility-only and which new metric names intentionally correct misleading coverage/validity semantics.

3. Freeze dependency and offline entry-point baselines.
   - Where: new `test_architecture_boundaries.py` plus existing Exp9/11–14 and generator tests.
   - Record current forbidden import edges so later stages can remove them monotonically. Verify all retained `--help` paths and aggregation work with `HF_HUB_OFFLINE=1` and without checkpoint loading.

4. Treat the local full-reference output only as optional evidence.
   - Where: planning journal entry produced during execution.
   - If present, record aggregate paths, row counts, config, and file hashes as a REPORT. A missing ignored directory is `NOT MEASURED`, not a gate failure.

---

## Verification

- [ ] GATE `uv run pytest -q experiments/zeroshot_cf/tests/test_dataset_contract.py experiments/zeroshot_cf/tests/test_exp9_benchmark.py experiments/zeroshot_cf/tests/test_metrics_harness.py experiments/zeroshot_cf/tests/test_generator.py` — current dataset, protocol, metric, and generator inputs stay green; a split, target-policy, formula, or non-padding regression turns this red.
- [ ] GATE `HF_HUB_OFFLINE=1` help/aggregate contract test for every retained CLI — subprocess output and checkpoint-load spies are the inputs; importing a model or losing a command turns it red.
- [ ] REPORT inspect `experiments/zeroshot_cf/results/local/full_reference` when present — record 24-cell completeness, hashes, and runtime in `journal.md`; otherwise record `NOT MEASURED`.

---

## Commit

`test(architecture): freeze evaluation and compatibility contracts`
