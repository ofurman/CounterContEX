# Stage 6: Matrices and DGX launchers

**Goal**: Define every campaign matrix, verify each one resolves without executing a method, and
verify the DGX can actually run one — so no expensive run starts on an unresolved assumption.
**Dependencies**: Stages 2, 3, 4

This is the gate between Phase A and Phase B. Nothing expensive runs before it passes.

---

## Steps

1. **Write the ten campaign matrices** named in
   [experiment-catalog.md](../resources/experiment-catalog.md), under
   `experiments/zeroshot_cf/configs/matrices/`. Use the shared protocol block from that document:
   five seeds, `max_test: 250`, stratified selection, `probability_threshold: 0.7`,
   `legacy_export: false`.
   - Each matrix varies **one scientific axis** beyond the shared protocol. If a matrix needs two
     axes crossed, say so explicitly in a comment naming why.
   - E1 crosses the three classifier families through the `target_models:` axis Stage 2 added
     to matrix expansion; do not hand-write one matrix per family.
   - Fix the output roots exactly as the catalog specifies. A resumed stage must write to the
     same place, or resume silently starts a second campaign.
   - Size E1 for whichever dataset count Stage 1 established. If German Credit failed its
     feasibility check, E1 is five datasets, not six, and the expected cell count changes.

2. **Dry-run every matrix** and inspect the resolved specification of every cell, not just the
   count. Confirm for each: the resolved target model, backend, evaluation version, seed,
   `n_counterfactuals`, and that no cell resolves to an identity that already exists under a
   different configuration.

3. **Verify the DGX runbook end-to-end.** Work through
   [dgx-runbook.md](../resources/dgx-runbook.md) on `gx10-bdc5` and correct whatever has drifted —
   that runbook was written for the older TabPFN-era layout and is a starting point, not a
   guarantee. Confirm `torch.cuda.is_available()`, checkpoint content-identity verification, and
   `vendor_setup.py --check`.

4. **Run one full E1 cell on the DGX** end to end — HELOC, logistic regression, seed 42, n=250 —
   and confirm it publishes a complete artifact directory with a `COMPLETE` marker, then
   aggregates. Compare its per-factual cost against Stage 1's measurement.

5. **Write the detached launch scripts** using the `nohup` + marker-file pattern, one per
   campaign stage, plus an rsync retrieval command. Put them where the existing Slurm launchers
   live (`experiments/zeroshot_cf/athena/`) or in a sibling `dgx/` directory, and update the
   relevant README.

6. **Record the frozen expected cell count per matrix** in `journal.md`. Stages 7–12 gate their
   completeness against these numbers, so they must be recorded before any of them runs.

---

## Verification

- [ ] GATE `uv run python -m experiments.zeroshot_cf.cli matrix --config <each> --dry-run`
      resolves every campaign matrix with zero errors — read from each command's resolved
      specification output.
- [ ] GATE Every resolved cell across all matrices carries `evaluation_version:
      countercontex.evaluation.v2` — read from the dry-run resolved identities. A stale default
      anywhere turns it red and would split the campaign across two versions.
- [ ] GATE One full E1 cell completes on the DGX and publishes `manifest.json`, `summary.csv`,
      `points.csv`, `candidates.csv`, `arrays.npz` and `COMPLETE`, and `cli aggregate` accepts it
      — read from the published run directory.
- [ ] REPORT Expected cell count for each of the ten matrices, and the DGX per-factual cost of
      the smoke cell against Stage 1's measurement — record both in `journal.md`. Stages 7–12
      gate their completeness against the recorded counts.

---

## Commit

`feat(configs): add campaign matrices and DGX launch scripts`
