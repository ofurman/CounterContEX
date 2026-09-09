# Stage 4: Integrate Budgeted q Decoders

**Goal**: Add the E11 mode, grid, IID, top-k, and top-p proposal policies to CounterContEx under strict, auditable raw and classifier caps.
**Dependencies**: Stage 2
**Reference**: [Frozen experiment protocol](../resources/experiment-protocol.md), E11.

---

## Steps

1. Add immutable scientific configuration.
   - Where: `CounterContExSearchConfig` and validation in `experiments/zeroshot_cf/methods/countercontex/config.py`.
   - Details: Represent decoder, B, numerical grid, top-k, top-p, strict category budget, temperature requirement, deduplication/accounting policy, and RNG scheme. Reject inapplicable combinations and unsupported backend capabilities during build/prepare. Keep the historical default path unchanged.

2. Route q policies through the backend-neutral search bridge.
   - Where: `_SessionSampler` and `generate_with_backend()` in `methods/countercontex/search.py`.
   - Details: Use session ICDF/distribution primitives and Stage-1 policy kernels. Grid arms use exact levels; IID/truncation arms use keyed uniforms. Do not make the generic runner aware of any policy.

3. Enforce the budget in k=1 search.
   - Where: `greedy_mixed_counterfactual()` and categorical helpers in `experiments/zeroshot_cf/grouped_categorical.py`.
   - Details: The strict experimental mode permits at most B raw proposals per state/action unit, counts no-ops/duplicates/projection, scores each unique row once, and never activates the current unbounded categorical fallback. Retain the fallback only for historical policy compatibility.

4. Persist method-owned accounting arrays and summaries.
   - Where: `GenerationResult.artifacts` assembly in CounterContEx method/search code.
   - Details: Use `method.*` portable arrays for policy, raw/projected/unique counts, classifier rows, TabICL calls/rows, and terminal dispositions. Do not put large ragged traces in JSON point diagnostics.

---

## Verification

Each line is `GATE` (must pass) or `REPORT` (measure and record in `journal.md`).
A measured GATE names the input its value is read from and the defect that turns it red -- a line
that only names a command passes whenever the command exits 0.

- [ ] GATE focused config, grouped-search, diverse-search, method-contract, and proposal-policy tests pass; over-budget fallback, repeated mode disguised as unique samples, illegal top-k/p, lost duplicates, or extra classifier queries turns a named fixture red.
- [ ] GATE the legacy nine-quantile configuration reproduces its authenticated witness and retains historical fallback semantics, while strict E11 mode cannot enumerate beyond its declared category cap.
- [ ] GATE two fake-backend runs with reordered/chunked pairs produce identical candidates and accounting arrays from this run's fixtures; ambient RNG or batch-order dependence turns the byte comparison red.
- [ ] REPORT record grid/IID/truncation raw-to-unique ratios and oracle rows on deterministic fixtures.

---

## Commit

`feat(countercontex): add budgeted distribution decoders`
