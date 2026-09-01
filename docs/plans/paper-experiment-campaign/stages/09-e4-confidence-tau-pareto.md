# Stage 9: E4 — confidence and threshold Pareto

**Goal**: Turn the boundary-hugging finding into the paper's most differentiated result, and
give contribution 3 the evidence it currently lacks.
**Dependencies**: Stage 6

Contribution 3 — confidence-conditioned counterfactuals — is presently a capability with no
experiment behind it. This stage is the whole of its evidence.

---

## Steps

1. **Establish that boundary-hugging is field-wide, not a CounterContEx defect.** Using the
   per-candidate `p_f(y*|x')` array added in Stage 3, plot the distribution for every method from
   the Stage 7 E1 artifacts. Measured baseline: threshold validity 0.000 at τ=0.7 across all
   2,991 HELOC candidates, from
   `results/local/architecture_full_reference/cf9d0c3a…/summary.csv`.
   - If every method concentrates just above 0.5, the finding is about sparse counterfactual
     search in general and the framing is strong. If CounterContEx is alone in it, the finding is
     about CounterContEx and must be reported that way instead.

2. **Launch `campaign_e4_confidence.yaml`.** Cross the confidence-quantile configuration against
   generation threshold τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9} on four datasets.
   - **Generation τ and evaluation τ are distinct.** Generation τ changes what the search accepts
     and is a scientific axis; evaluation τ is a reporting threshold. Vary the first; report at
     several values of the second. Confusing them would make the Pareto curve meaningless.
   - Include a confidence-conditioning-off arm at each τ, or the curve cannot separate the effect
     of the confidence anchors from the effect of raising τ.

3. Produce F4: proximity versus achieved confidence, baselines as points, CounterContEx as a
   curve. Report coverage and runtime alongside — a curve bought with collapsed coverage is not
   a Pareto frontier.

4. **Do not select a τ here and report it later as an unbiased headline.** If Stage 12's frozen
   configuration takes a τ chosen from this stage's results, that selection must be disclosed in
   the paper. This is already a repository contract; record the disclosure text in
   `decisions.md` now, while the reason is fresh.

---

## Verification

- [ ] GATE Achieved `p_f(y*|x')` rises monotonically with generation τ across the arms, or the
      departure is explained in `journal.md` — read from the per-candidate probability arrays. A
      flat response means the generation threshold is not reaching the search, which would make
      the entire curve an artifact.
- [ ] REPORT F4 and the per-method probability distributions from Stage 7, plus coverage and
      runtime at each τ — record in `journal.md`.

---

## Commit

`docs(plan): record E4 confidence and threshold Pareto results`
