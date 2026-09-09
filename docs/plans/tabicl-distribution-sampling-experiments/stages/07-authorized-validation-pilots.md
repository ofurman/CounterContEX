# Stage 7: Run Authorized Diagnostics and Validation Pilots

**Goal**: After explicit authorization, establish that E11-E13 mechanisms execute on real TabICL checkpoints and measure the confirmation cost without selecting outcomes.
**Dependencies**: Stage 6
**Reference**: [Frozen experiment protocol](../resources/experiment-protocol.md), Pilot and confirmation.

---

## Steps

1. Enforce the authorization boundary before any checkpoint or GPU call.
   - Where: an append-only authorization entry in `decisions.md` naming the approved host, pilot specs, factual counts, seeds, maximum scope, and new output roots.
   - Details: If absent, create a self-contained authorization backlog item, mark this stage BLOCKED, and do not run checkpoint lookup, smoke, pilot, or matrix commands. Planning is not authorization.

2. Validate the exact environment after authorization.
   - Where: pinned CEL/vendor check, locked uv environment, checkpoint checker, model/classifier content IDs, and clean isolated source snapshot on the approved GPU host.
   - Details: Reuse verified caches only after checksum and identity validation. Do not disable offline/checksum guards or modify historical roots.

3. Replay the historical real TabICL adapter before new pilots.
   - Where: the authenticated seed-42 E3 trace cases on the approved pinned checkpoint/host.
   - Details: Require identical availability and maximum absolute candidate difference at most `1e-8`, the tolerance used by the prior audit. Record raw/projected differences and identities. This is the real-adapter compatibility gate that cannot be satisfied by offline stored-value replay.

4. Run the real proposal-only E12 pilot.
   - Where: the tracked E12 diagnostic spec and a fresh timestamped diagnostic root.
   - Details: Four retained datasets, 20 deterministically target-stratified validation factuals, LR/MLP/XGBoost, exact categorical support, 256 stratified numerical points, and the five-by-64 Monte Carlo sensitivity bank. Verify trace-on/off candidate purity where search replay exists.

5. Run E11/E13 mechanism pilots.
   - Where: pilot profiles of the tracked matrices and new canonical output roots.
   - Details: Four datasets, LR, 20 validation factuals, seeds 17/42/101, all predeclared E11 and E13 arms. For E13, retain the exactly matched original-factual companion and the adaptive end-to-end run; common pool IDs are compared only at identical state/action keys. The pilot may fix implementation defects but must not remove or select arms based on quality. Record M/B, unique oracle rows, realized cumulative classifier calls/rows, stopping depth, ESS, projection collapse, phase timings, and failures.

6. Authenticate, aggregate, and estimate confirmation cost.
   - Where: the Stage-6 campaign inventory writer/strict reader and plan journal.
   - Details: Verify exact membership, publish an external `campaign-inventory-v1` for E11/E13, and verify every hash before analysis. Then project GPU-hours from measured prepare/generate/evaluate/write totals with a range rather than a point promise. Record any factuals repeated across seeds as Monte Carlo repeats.

---

## Verification

Each line is `GATE` (must pass) or `REPORT` (measure and record in `journal.md`).
A measured GATE names the input its value is read from and the defect that turns it red -- a line
that only names a command passes whenever the command exits 0.

- [ ] GATE a specific pilot authorization entry exists before the first checkpoint/GPU command; absence blocks this stage and is not amendable by the executor.
- [ ] GATE the real E3 adapter replay on the authorized pinned host/checkpoints has identical availability and candidate maximum absolute difference `<=1e-8`; authenticated historical arrays are the input, and adapter/checkpoint/projection drift turns it red.
- [ ] GATE E12 `COMPLETE.json` and external E11/E13 `campaign-inventory-v1` files authenticate every matrix/spec-resolved row from their new roots; a missing, partial, extra, duplicate, hash-mismatched, wrong-identity, or post-seal-mutated payload turns the owning check red.
- [ ] GATE rerunning one stochastic witness with the same keys reproduces raw pool IDs and deterministic outputs, while a different seed changes at least one non-degenerate draw; RNG reset or fabricated seed variation turns the check red.
- [ ] REPORT publish every mechanism/quality outcome, ESS diagnostic, failure, and measured confirmation cost estimate; no observed advantage gates continuation.

---

## Commit

`experiment(tabicl): run distribution sampling pilots`
