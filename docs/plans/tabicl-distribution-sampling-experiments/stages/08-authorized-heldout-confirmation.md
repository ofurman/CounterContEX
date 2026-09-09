# Stage 8: Run Authorized Held-Out Confirmation

**Goal**: After separate cost-aware authorization, execute the frozen E11-E13 confirmation and limited k=3 replication exactly once under new identities.
**Dependencies**: Stage 7
**Reference**: [Frozen experiment protocol](../resources/experiment-protocol.md), Pilot and confirmation.

---

## Steps

1. Require a second explicit authorization.
   - Where: a new append-only `decisions.md` entry quoting the Stage-7 measured cost range and naming approved confirmation specs, host, scope, and untouched output roots.
   - Details: Pilot authorization does not carry forward. Without this entry, add an authorization backlog item, mark Stage 8 BLOCKED, and launch nothing.

2. Re-freeze code and resolved identities before execution.
   - Where: source commit/tree hash, uv lock hash, dataset/case/model/checkpoint IDs, matrices, diagnostic spec, and dry-run inventories.
   - Details: No scientific field may change after seeing pilot quality. An implementation correction requires fresh dry-runs, new identities when behavior changes, and a new output root; record it before any confirmation cell.

3. Execute E11 and E13 at k=1.
   - Where: their confirmation matrix profiles and managed launchers.
   - Details: Four retained datasets, LR, 100 deterministically target-stratified test factuals, seeds 17/42/101/202/303, all frozen arms and exact per-expansion budgets. For E13, run both the exactly matched original-factual companion and adaptive end-to-end search; report realized cumulative oracle work for the latter. Preserve failed/missing slots and phase timings.

4. Execute E12 confirmation.
   - Where: the E12 diagnostic confirmation profile.
   - Details: Four datasets, 100 deterministically target-stratified test factuals, LR/MLP/XGBoost, exact categorical integration and deterministic 256-point numerical stratification. Do not multiply deterministic points into fake seed replications.

5. Execute the predeclared k=3 replication.
   - Where: a separate matrix/output root containing only grid-9, IID-9, beta=0, and beta=1.
   - Details: 50 deterministically target-stratified test factuals, seeds 17/42/101, explicit common categorical cap, beam/DPP settings held fixed. Report it separately; do not pool with k=1 or historical E3.

6. Verify and seal all outputs before interpretation.
   - Where: strict aggregate, Stage-6 campaign inventory, and diagnostic hash readers.
   - Details: Compare resolved membership to complete artifacts, atomically publish external inventories without modifying completed run directories, verify them, retain exact failures, and write artifact roots/hashes to the journal before calculating headline deltas.

---

## Verification

Each line is `GATE` (must pass) or `REPORT` (measure and record in `journal.md`).
A measured GATE names the input its value is read from and the defect that turns it red -- a line
that only names a command passes whenever the command exits 0.

- [ ] GATE a separate confirmation authorization entry exists and cites the measured pilot cost; absence blocks and cannot be amended by the executor.
- [ ] GATE source/spec/content identities captured before launch match every completed manifest/trace, and strict membership accepts exactly the dry-run inventory with no partial/extra/duplicate/mismatched cell.
- [ ] GATE every standard run has all canonical payloads plus `COMPLETE` and is covered by a verified external `campaign-inventory-v1`; every E12 trace has a verified `COMPLETE.json`. Deleting or mutating any payload or inventory turns verification red.
- [ ] REPORT record all outcomes and total phase/GPU hours exactly as measured, including unavailable cells and non-independent Monte Carlo repeats.

---

## Commit

`experiment(tabicl): run distribution sampling confirmation`
