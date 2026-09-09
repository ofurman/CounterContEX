# Stage 6: Freeze Identities, Matrices, and Launchers

**Goal**: Assign new scientific identities, materialize exact dry-resolved E11-E13 specifications, and add canonical campaign sealing without running checkpoints.
**Dependencies**: Stages 3, 4, and 5
**Reference**: [Frozen experiment protocol](../resources/experiment-protocol.md), Pilot and confirmation.

---

## Steps

1. Version the scientific behavior.
   - Where: the `countercontex` registry entry in `experiments/zeroshot_cf/methods/registry.py`, `TABICL_BACKEND_IMPLEMENTATION_VERSION`, and `_BACKEND_POLICIES` in `methods/countercontex/runtime.py` / `backends/tabicl.py`.
   - Details: Create new implementation identities for the expanded backend and search policy. Existing v3/v1 manifests remain immutable and are never resumed or aggregated with new variants. If historical defaults are byte-compatible, preserve that as a separate compatibility variant rather than silently relabeling history.

2. Add an explicit factual-partition protocol field.
   - Where: `FactualSelection` / `BenchmarkCase` in `experiments/zeroshot_cf/core/contracts.py`, benchmark-case selection in `datasets/benchmark.py`, protocol identity in `orchestration/spec.py`, generic protocol routing in `orchestration/runner.py`, and matrix parsing/tests.
   - Details: Support `validation` for pilots and `test` for confirmation, with `test` as the historical default. Pop the factual-only field before constructing the dataset provider, bind `BenchmarkCase` to the selected split, and identify rows by `(partition, local_source_index)`. Add a new target-model-specific `target_stratified` policy that predicts the full partition before deterministic selection; preserve the existing truth-label `stratified` policy and every historical default. The partition, policy, selected qualified IDs, predictions, and targets enter case/scientific identity. Classifier training/reference construction is unchanged.

3. Write tracked experiment definitions.
   - Where: `campaign_e11_distribution_sampling.yaml` and `campaign_e13_classifier_guidance.yaml` under `experiments/zeroshot_cf/configs/matrices/`, plus an E12 spec under `configs/diagnostics/`.
   - Details: Encode every dataset, classifier, factual count, seed, k, B/M, numerical and categorical decoder, fixed 256-bin rule, top-k/p, beta, `classifier-top-B`, temperature, threshold, context, capability, and implementation setting. Separate pilot, confirmation, and k=3 replication specs or explicit profiles. Disable legacy export.

4. Prove controlled axes and exact cell counts.
   - Where: `experiments/zeroshot_cf/tests/test_campaign_matrices.py`, `test_orchestration_spec.py`, and focused diagnostic-config tests.
   - Details: Pairwise audits must show only the intended policy/budget/beta and required identity bundle differ. Freeze resolved cell counts from the actual parser. Scientific fields change cell/run IDs; device/cache/output/resume do not.

5. Add managed launch support but do not execute it.
   - Where: `experiments/zeroshot_cf/athena/` and its README.
   - Details: Add check/preflight, new-output-root enforcement, per-cell status, aggregation, and phase-timing capture. Launchers must default to dry-run or require an explicit authorization record. Document the exact authorization boundary for Stages 7 and 8.

6. Add external campaign sealing for canonical E11/E13 runs.
   - Where: new `experiments/zeroshot_cf/orchestration/campaign_inventory.py` (final name may follow the package convention), strict readers, and artifact-store/aggregation tests.
   - Details: After strict aggregation accepts exact membership, hash every required file and resolved run ID into `campaign-inventory-v1`, then publish the inventory atomically outside immutable run directories. Verification must reject a missing/extra run, a changed CSV/NPZ/manifest/marker, a mismatched matrix, or an inventory created before all cells are complete. Do not modify a run directory after its `COMPLETE` marker.

7. Update suite documentation.
   - Where: `experiments/zeroshot_cf/README.md` and repository `README.md` only where public configuration/commands change.
   - Details: State decoder definitions, budget semantics, pi-beta equation, target-oracle reuse, method-specific E12 status, and interpretation limits.

---

## Verification

Each line is `GATE` (must pass) or `REPORT` (measure and record in `journal.md`).
A measured GATE names the input its value is read from and the defect that turns it red -- a line
that only names a command passes whenever the command exits 0.

- [ ] GATE each matrix/diagnostic dry-run resolves the exact cells recorded from its own YAML, and a temporary canonical campaign seals only after exact membership passes; focused checks reject injected missing/extra/duplicate/mixed-identity cells and separate mutations of CSV, NPZ, manifest, `COMPLETE`, matrix identity, or inventory content.
- [ ] GATE identity tests vary every new scientific field, including factual partition and `target_stratified`, independently; validation/test qualified IDs differ by partition even when local integers overlap; historical `test` plus truth-label `stratified` manifests retain their selection behavior and are rejected for resume/aggregation under new identities. Any unchanged hash under semantic drift or unqualified source ID turns the test red.
- [ ] GATE focused tests, full `uv run pytest -q`, changed-package Ruff, offline CLI help/list-methods, launcher `bash -n`, and `git diff --check` pass.
- [ ] REPORT publish the exact pilot/confirmation/k=3 cell counts and a lower-bound cost estimate based on measured E3 phase timings, explicitly labelled as an estimate.

---

## Commit

`feat(experiments): define TabICL distribution studies`
