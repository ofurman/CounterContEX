# Stage 7: E1 — main comparison

**Goal**: Execute the headline experiment: six datasets × three classifier families × five seeds
× six methods at k=1, producing the `primary_*` head-to-head that replaces the k-mismatched table.
**Dependencies**: Stage 6

This is the long pole — roughly 38 GPU-hours of CounterContEx time. It closes gaps B1, B2, B3
and half of B4.

---

## Steps

1. Launch `campaign_e1_main.yaml` on the DGX with the detached pattern from
   [dgx-runbook.md](../resources/dgx-runbook.md) and `--resume`. Poll the marker file; do not
   hold an interactive session open for it.

2. **Report CounterContEx at `n_counterfactuals: 1`.** The comparison is against `k=1` baselines,
   so CounterContEx must request one candidate, not three with only rank 0 reported. Requesting
   three and reading the primary rank is a *different search* — the beam and DPP path — and
   would not be the same method under comparison.

3. On completion, retrieve artifacts by rsync including every `manifest.json` and `COMPLETE`
   marker, then aggregate and run the Stage 5 analysis layer to produce T1 and F3 and F6.

4. **Record every failure as evidence.** Stage 2 measured which method/target-model
   combinations run; the repository's Wachter is black-box (`predict_proba` only) and is
   expected to run against all three families, so do not assume any cell will be absent. If a
   combination does fail at method preparation, it must be one Stage 2 recorded as an expected
   clean failure. Such a failure is a result, not an incident: record it in `journal.md` and
   let the cell be absent by identity. Do not substitute a nearby run, and do not disable the
   arm to make aggregation pass.

5. Record measured phase timings per cell for the Stage 10 cost analysis, and the total campaign
   hours consumed so far.

---

## Verification

- [ ] GATE Every aggregated cell carries `evaluation_version: countercontex.evaluation.v2` and a
      `model_content_id` matching its declared family — read from the manifests. A cell that
      loaded a cached classifier from the wrong family turns it red.
- [ ] REPORT T1, F3 and F6 built by the Stage 5 analysis layer, plus mean ± std and corrected
      p-values per metric, and the count of cells absent by expected failure — record in
      `journal.md` with artifact paths.

---

## Commit

`docs(plan): record E1 main comparison results`
