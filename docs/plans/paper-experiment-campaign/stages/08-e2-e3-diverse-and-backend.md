# Stage 8: E2 and E3 — diverse sets and backend ablation

**Goal**: Give the diversity claim a comparator, and answer the question every reviewer will
ask first — does the foundation model do anything?
**Dependencies**: Stage 6

E3 is the most consequential experiment in the plan. If the checkpoint-free `empirical` backend
matches TabICL, the foundation model is decorative and the paper's framing must change. **That
outcome is a legitimate result and must be reported if it occurs.** It is a REPORT, never a GATE.

---

## Steps

1. **E2 — diverse sets.** Launch `campaign_e2_diverse.yaml`: six datasets × five seeds,
   CounterContEx and DiCE both at `n_counterfactuals: 3`, logistic-regression target. Report
   `set_coverage_at_k`, `set_action_jaccard_mean`, `set_pairwise_gower_mean`, alongside validity
   and proximity so a diversity gain bought with invalidity is visible.

2. **E3 — backend ablation.** Launch `campaign_e3_backend.yaml`: six datasets × three seeds ×
   {`tabicl`, `empirical`}, everything else held fixed — same search configuration, same
   quantiles, same seeds, same target model, same requested k.
   - The `empirical` backend does not support confidence conditioning or joint scoring. The
     TabICL arm must therefore run **without** those options too, or the comparison confounds
     the backend with the search configuration. Record this constraint in `decisions.md`.

3. Aggregate both, run the analysis layer, and produce T2 and T3.

4. **Write the E3 conclusion before looking at the confidence interval.** State in `journal.md`
   what the measured difference is, its seed-to-seed spread from Stage 1, and whether the
   difference exceeds that spread — in that order. A difference smaller than the noise floor is
   a null result and must be recorded as one.

---

## Verification

- [ ] GATE ~~The TabICL and empirical arms of E3 differ in exactly one resolved scientific field,
      `backend_implementation`~~ -> Paired arms differ only in the backend identity bundle:
      declared backend, resolved backend implementation, and backend-owned checkpoint content
      IDs — read by diffing the two arms' scientific specifications and resolved identities from
      their manifests. Any non-backend difference means the ablation is confounded.
- [ ] REPORT E2 set metrics for CounterContEx and DiCE, and the E3 TabICL-vs-empirical difference
      per metric compared against the Stage 1 noise floor — record in `journal.md`, whichever way
      it lands.

---

## Commit

`docs(plan): record E2 diverse-set and E3 backend ablation results`
