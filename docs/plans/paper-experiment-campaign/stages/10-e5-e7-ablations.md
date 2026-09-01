# Stage 10: E5–E7 — ablations and cost Pareto

**Goal**: Test the remaining hypotheses the method document lists as untested, and give the
runtime section the analysis it needs.
**Dependencies**: Stage 6

Closes the rest of gap B5 and gap B7.

---

## Steps

1. **E5 — search and diversity ablations** (`campaign_e5_search.yaml`, n=100). Mode versus
   9-point quantile grid; `sparse` versus `data_plausible`; DPP versus random versus
   greedy-farthest pool selection; revisits on and off. One axis per cell group.
   - The DPP alternatives need implementing as selectable strategies inside the CounterContEx
     diversity module, not as a fork of the search. Keep method policy inside the method package.

2. **E6 — context ablation** (`campaign_e6_context.yaml`, n=100). Context size ∈ {64, 128, 256,
   512} crossed with context labels ∈ {classifier predictions, true labels}.
   - The label axis is the interesting one. Labelling the context with the classifier's
     predictions conditions the proposal model on the model's view of the world rather than
     reality's. It is a defensible choice and currently an unmeasured one.
   - The true-label arm requires the case loader to expose training labels to the backend.
     Confirm this does not leak test labels into the method — `MethodContext` deliberately does
     not carry them, and that boundary must survive this stage.

3. **E7 — cost-quality Pareto** (`campaign_e7_cost.yaml`). Four configurations from cheap to
   expensive across four datasets: mode-only proposals with context 128 at k=1, through the full
   reference configuration.
   - Fold in Stage 1's Lending Club root cause. If a fix was identified, this is where its effect
     is measured — as a new configuration point under a **new implementation version**, never as
     a silent change to an existing one.

4. Produce F5 from the measured phase timings, and record total campaign hours to date.

5. **Freeze the configuration for Stage 12** at the end of this stage. Record the frozen
   configuration and its justification in `decisions.md`, including which stages' results
   informed the choice — that record is what makes the Stage 12 disclosure honest.

---

## Verification

- [ ] GATE The E6 true-label arm reads training labels only; no test label reaches
      `MethodContext` — read from the constructed context in a contract test. A leak turns every
      E6 result invalid and would also invalidate E1 if the same path is shared.
- [ ] GATE Each ablation cell group differs from its control in exactly one resolved scientific
      field — read by diffing resolved identities from the manifests.
- [ ] REPORT E5 and E6 results per axis against the Stage 1 noise floor, F5, and the frozen
      Stage 12 configuration — record in `journal.md` and `decisions.md`.

---

## Commit

`docs(plan): record E5-E7 ablation and cost-Pareto results`
