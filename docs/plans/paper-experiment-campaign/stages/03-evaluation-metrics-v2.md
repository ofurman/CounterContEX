# Stage 3: Evaluation metrics v2

**Goal**: Add every metric the paper needs, in one bump to `countercontex.evaluation.v2`, before
any campaign run starts.
**Dependencies**: Stage 1

`evaluation_version` is part of scientific identity (`orchestration/spec.py`). A metric added
after Stage 7 makes Stage 7 un-aggregatable with Stages 8–12. **This is the only evaluation
version bump in this plan.** Anything discovered later goes to `backlog.md` for a successor
plan, not into a second bump.

---

## Steps

1. **Capture the per-candidate target probability.** The boundary-hugging finding — threshold
   validity 0.000 at τ=0.7 across all 2,991 HELOC candidates — is currently visible only as a
   single aggregate. Store `p_f(y*|x')` per returned candidate in `arrays.npz` so Stage 9 can
   plot its distribution and Stage 11 can rescore it after retraining.
   - Where: the evaluator in `experiments/zeroshot_cf/evaluation/evaluator.py`; the array
     contract in `orchestration/artifacts.py`.
   - This one addition unblocks E4, E8 and figure F6. It is the highest-value item in the stage.

2. **Add detectability AUC.** Train a discriminator to separate real target-class reference rows
   from returned counterfactuals; report AUC over a fixed cross-validation split. ≈0.5 means
   indistinguishable from real data.
   - Fix the classifier family, hyperparameters and split seed. Report the count of rows in each
     arm alongside the AUC — an AUC over 40 counterfactuals is not the same measurement as one
     over 3,000, and a reader cannot tell them apart from the number alone.
   - **Name the degenerate satisfier and forbid it**: an AUC near 0.5 is also what an empty or
     near-empty counterfactual arm produces. The metric must be reported together with its arm
     sizes, and a run whose CF arm is below a declared minimum records `NOT MEASURED`, never a
     number.

3. **Add distance to the k-th nearest training neighbour** in grouped-Gower space, matching the
   space the search itself uses. Report the mean and the distribution.

4. **Decide on the joint log-density metric.** The TabICL joint score currently exists as a
   method-internal ranking signal. Promoting it to a method-blind evaluation metric would let
   all methods be scored on the same density model — but it favours the model class CounterContEx
   itself proposes from. Record the decision in `decisions.md` either way. If promoted, it must
   always be reported beside a neutral measure, never alone.

5. **Bump the version.**
   - Where: `METRIC_SCHEMA_VERSION` in `experiments/zeroshot_cf/evaluation/result.py` and the
     `evaluation_version` default in `orchestration/spec.py`, to `countercontex.evaluation.v2`.
   - Preserve every existing metric name, denominator and semantic. The two-population rule
     (grouped-Gower and continuous proximity over target-class candidates; sparsity,
     actionability, bounds, LOF and Isolation Forest over all available candidates) is unchanged.
     A silent denominator change here would invalidate comparison against the historical runs.

6. **Document the new metrics** in `experiments/zeroshot_cf/README.md` under Metric semantics,
   including denominators and the detectability-AUC arm-size caveat.

---

## Verification

- [ ] GATE Every pre-existing metric produces an identical value on a fixed deterministic fixture
      before and after the bump — read from the fixture evaluation output on both revisions. A
      changed denominator or population turns it red.
- [ ] GATE Detectability AUC on a fixture where counterfactuals are copied verbatim from real
      target-class rows measures ≈0.5, **and** on a fixture where they are pushed far out of
      distribution measures near 1.0 — read from the metric output. A metric that returns 0.5 for
      both is measuring nothing and this pair of fixtures is what catches it. On a third fixture
      with an empty CF arm it must produce `NOT MEASURED`, never 0.5.
- [ ] REPORT All new metrics computed on one dataset — record values and arm sizes in `journal.md`.

---

## Commit

`feat(evaluation): add plausibility and probability metrics as evaluation v2`
