# Stage 5: Integrate Classifier-Guided pi-beta Sampling

**Goal**: Implement E13 sampling from a finite common-pool approximation to `pi_beta` without a cost proxy or hidden classifier work.
**Dependencies**: Stages 2 and 4
**Reference**: [Frozen experiment protocol](../resources/experiment-protocol.md), E13; decision D-5 in `../decisions.md`.

---

## Steps

1. Add validated guidance configuration.
   - Where: `CounterContExSearchConfig` in `experiments/zeroshot_cf/methods/countercontex/config.py`.
   - Details: Encode beta, M, B, epsilon, with-replacement SIR policy, score-cache requirement, and common-random-number scheme. Require M >= B, finite beta >= 0, TabICL full-q capability, temperature 1, and confidence/joint scoring off for E13.

2. Construct and score the common q pool once.
   - Where: `_SessionSampler`, `generate_with_backend()`, and the owning trial builder in `methods/countercontex/search.py` / `grouped_categorical.py`.
   - Details: Draw M keyed proposals with replacement from numerical or exact categorical q, project, aggregate duplicate mass, batch-evaluate unique legal rows, and cache target probabilities. For both action types, empirical-pool weights are only `p_clf^beta`; do not multiply by q again because the sampled pool already represents q.

3. Select B proposals for each beta with common uniforms.
   - Where: the Stage-1 proposal-policy kernel.
   - Details: Normalize in log space, use weighted resampling with replacement, map selected raw indices back through duplicates, and expose beta=0 as the pure-q control. Repeated indices consume budget; handle all-zero/underflow, single-support, and no-op-only pools explicitly without padding candidates. Share pool IDs and resampling uniforms only for identical canonical state/action keys; after beta arms diverge, generate separately keyed pools for their different states.

4. Carry cached scores and diagnostics into search/results.
   - Where: classifier-output helpers in `grouped_categorical.py` and `diverse_search.py`, and CounterContEx `GenerationResult` construction.
   - Details: Reuse scores rather than query the discriminator again. Persist normalizer, ESS, selected unique fraction, raw/unique oracle rows, and calls/rows to first validity and total. Final missing slots remain missing.

5. Add the paid-pool operational comparator.
   - Where: the same method-owned selection layer.
   - Details: `classifier-top-B` passes the B highest-scoring unique projected rows from the same M pool with canonical tie-breaking. Label it as a deterministic REPORT comparator, not pi-beta sampling, and give it no extra model or classifier calls.

6. Separate exactly matched mechanism evidence from adaptive utility.
   - Where: the proposal diagnostic and end-to-end CounterContEx artifacts.
   - Details: At every original factual, evaluate all beta arms on one fixed common state/pool and identical classifier rows. In adaptive search, enforce the same M/B per-expansion caps, record realized cumulative classifier rows/calls and stopping depth, and never claim those realized totals or later pool IDs are equal after trajectories diverge.

---

## Verification

Each line is `GATE` (must pass) or `REPORT` (measure and record in `journal.md`).
A measured GATE names the input its value is read from and the defect that turns it red -- a line
that only names a command passes whenever the command exits 0.

- [ ] GATE analytical categorical and sampled-continuous fixtures prove beta=0 empirical-pool recovery, correct q-sampling of the common categorical pool, log-space stability, common pools/uniforms, and no accidental `q^2` factor; each named mathematical defect turns its own check red.
- [ ] GATE classifier-spy tests prove every unique pool row is scored exactly once and search reuses that score; a second query, hidden refill, or missing budget row turns the trace red.
- [ ] GATE fixed-state beta arms differ only in beta/selection and share raw pool IDs, M, B, target classifier, and resampling keys; end-to-end fixtures require shared pool IDs only at identical canonical state/action keys and prove that divergent states receive distinct reproducible keys. Pool aliasing across different states or unequal per-expansion caps turns the pair audit red.
- [ ] REPORT record ESS and selection-frequency convergence for M=64 versus M=256 on analytical fixtures; this is an instrument report, not an outcome gate.

---

## Commit

`feat(countercontex): add classifier-guided proposal sampling`
