# Stage 9: Consolidated Results Table (Proximity Surfaced) + REPORT Update

**Goal**: Produce the meeting's headline **"tabelka"** — one consolidated table across all datasets and configs with **Proximity explicitly surfaced** alongside Validity, L0, Steps, and Plausibility — and fold the new results (budget sweep, discrete dataset, routing audit, trajectory figures) into `results/REPORT.md` as the next-meeting deliverable.
**Dependencies**: Stages 5, 6, 7, 8 DONE (this is the synthesis stage). If any is BLOCKED, build the table from whatever is available and mark the gap.

---

## Motivation (from the meeting)

> "widzę, że jeszcze nie liczę Proximity, jest tylko Plausibility i L0 … więc jeszcze tu dorzucę."
> "robisz tabelkę … na następny tydzień, i zobaczymy."

**Note**: `proximity_l2_jaccard` is **already computed and written** by every runner (`metrics_harness.py:89–96`, valid CFs only; present in exp4/5/6 CSVs). The gap the meeting flagged is that the *headline table* a viewer was looking at omitted it. So this stage is mostly **surfacing + consolidation**, not new metric code.

---

## Steps

1. **Build the consolidated table** `results/summary_table.md` (and a `summary_table.csv` for the paper). One row per (dataset, config) covering:
   - MOONS + HELOC at the Stage-4 recommended configs (`prob_ascent` + `random_both@512` / `knn_both@256`).
   - The discrete dataset from Stage 7.
   - The best budget cell from the Stage-5 sweep (revisit-enabled) per dataset.
   - The Stage-8 routing-override HELOC variant (if it improved anything).
   - Columns: `dataset, config, validity, failure_rate, l0_count_mean (distinct), steps_mean, proximity_l2_jaccard, lof_scores_cf, frac_oob, true_actionability, n_test`. **Proximity is a first-class column.**
   - Include the one-pass baseline row (predecessor Stage-8) for contrast where available.

2. **Update `results/REPORT.md`** with a new section (§8 or "Follow-up: budget/revisit, discrete, routing") that:
   - States the MOONS budget→validity finding (Stage 5) and embeds the trajectory figure reference (Stage 6).
   - Reports the discrete-dataset validity (Stage 7).
   - Summarizes the binning audit + routing experiment verdict (Stage 8).
   - Renders the consolidated table inline.

3. **Verify proximity is in all headline tables.** Audit the existing exp4/5/6/7 summaries and `build_notebook.py` (`:148–150`) so every results table that shows validity/L0/LOF also shows `proximity_l2_jaccard`. Add it wherever it's missing in the *presentation* layer.

4. **Write the next-meeting digest** at the top of the new REPORT section: 3–5 bullets a reviewer can read in 30 seconds (the validity lift from revisiting, the discrete-dataset sanity check, the proximity numbers, the routing verdict).

---

## Verification

- [ ] `results/summary_table.md` + `.csv` exist with one row per (dataset, config) and **proximity as an explicit column**, including a one-pass baseline contrast row.
- [ ] `results/REPORT.md` has a follow-up section embedding the table, the MOONS budget finding, the trajectory figure, the discrete result, and the routing verdict.
- [ ] Every headline results table (exp4/5/6/7 summaries + notebook) surfaces `proximity_l2_jaccard`.
- [ ] No regression: `uv run pytest experiments/zeroshot_cf/tests -q` passes.
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` empty; no `tabpfn_client` import.

---

## Notes

- This stage writes **no new model code** — it consolidates and presents. If a number is missing because an upstream stage was deferred, leave a clearly-marked TODO cell rather than fabricating it.
- Persist the headline numbers to NeoCortex (update the `iterative-greedy-cf-results` memory) so the next planning session starts from the latest baselines.

## Commit

`docs(greedy-cf): consolidated results table with proximity + REPORT follow-up section`
