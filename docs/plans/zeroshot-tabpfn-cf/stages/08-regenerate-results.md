# Stage 8: Regenerate Results & Report with Corrected Metrics

**Goal**: Re-run the experiments and sweeps with the Stage 7 fixes in place, then rebuild every results artifact (CSVs, REPORT.md, notebook, README) so the published numbers are trustworthy.
**Dependencies**: Stage 7 DONE (all code fixes committed).

> The Stage 7 validity-reference and LOF fixes change the numbers. Until this stage runs, `results/` reflects the **old, buggy** metrics. This stage's job is to make `results/` consistent with the corrected code and to record what changed.

**Guardrail reminder**: do NOT modify `src/tabpfn/**`. Runs must work offline (checkpoints already staged). Be honest about deltas — if corrected validity is lower than the old buggy value, report it plainly.

---

## Steps

1. **Re-run Experiment 1 (calibration fix).**
   - Command: `uv run python experiments/zeroshot_cf/exp1_single_feature.py --dataset moons` and `--dataset heloc`.
   - Regenerates `results/exp1_{moons,heloc}.csv`, `results/exp1_summary.md`, `results/exp1_moons.png` (now plotting the scored class-conditioned reconstruction).
   - Confirm the gate verdict logic is unchanged in spirit; note any shift in calibration numbers from the MAP/posterior separation.

2. **Re-run Experiment 2 (corrected validity + LOF).**
   - Command: `uv run python experiments/zeroshot_cf/exp2_counterfactuals.py --dataset moons` and `--dataset heloc` (baseline config / defaults).
   - Regenerates `results/exp2_{moons,heloc}_metrics.csv`, `results/exp2_summary.md`, `results/exp2_examples.md`.
   - **Validity is now `predict(X_cf) == y_target`.** Record old → new for both datasets. Expect HELOC validity to move most (oracle 72% acc → ~28% of rows previously mis-scored); MOONS less so.
   - LOF now reported on unclipped CFs (or flagged degenerate per Stage 7 choice). HELOC LOF should no longer be a bare `2.5e9`.
   - The immutability `assert` must pass (true_actionability stays 1.0).

3. **Re-run the refinement sweeps.**
   - Command: `uv run python experiments/zeroshot_cf/refine.py --dataset moons` and `--dataset heloc`.
   - Regenerates `results/exp2_sweep_{moons,heloc}.csv` with corrected validity. Re-derive the recommended configs from the corrected numbers (the MOONS `t=0.5, all_classes` recommendation may or may not still win — re-check, don't assume).
   - **Optional (addresses P2 cherry-pick finding)**: re-run the HELOC best-validity sweep config at the larger Exp2 sample size (n=50) to confirm whether the earlier validity drop was noise, rather than asserting it. If skipped for budget, `log()` it and soften the REPORT wording accordingly.

4. **Rebuild the consolidated report.**
   - File: `experiments/zeroshot_cf/results/REPORT.md`.
   - Update every metric table with the corrected numbers. Add a short **"Post-review corrections (2026-06-15)"** subsection summarizing: the validity-reference fix and its effect on each dataset's validity, the LOF relabeling, and the recommended-config re-derivation. Reference Backlog item 1 (cel baselines) in §5 instead of the old TODO.
   - Keep the verdict honest: if corrected HELOC validity falls below the ≥0.5 success target, say so and update the Success Criteria read in the report.

5. **Rebuild the notebook & README.**
   - Command: `uv run python experiments/zeroshot_cf/build_notebook.py` (regenerates `results.ipynb` deterministically from the new CSVs).
   - Update `experiments/zeroshot_cf/README.md` headline results + the reproducible run commands (now using the Stage 7 CLI flags for the recommended configs).

6. **Update the plan index & memory.**
   - Mark the Success Criteria read in `index.md` against the corrected numbers (note any target now missed).
   - Apply the Stage 6 memory-claim resolution (Stage 7 step 9): if persisting to memory, write the corrected final results; otherwise ensure the claim points to REPORT.md.

---

## Verification

- [ ] All `results/exp1_*`, `results/exp2_*` CSVs regenerated; timestamps newer than the Stage 7 commit.
- [ ] Every metric value quoted in `results/REPORT.md` matches the regenerated CSVs (spot-check ≥3 per dataset, as the review did).
- [ ] `results/REPORT.md` has a "Post-review corrections" subsection with old→new validity for MOONS and HELOC.
- [ ] HELOC LOF in REPORT is no longer presented as a bare finite `2.5e9` (unclipped value or degenerate label).
- [ ] `results.ipynb` regenerated and its embedded tables match the new CSVs (no stale numbers).
- [ ] README recommended-config command runs and reproduces the recommended-config metrics (validity within run-to-run noise).
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` empty.

---

## Commit

`docs(zeroshot-cf): regenerate Exp1/Exp2/sweep results + report with corrected metrics (post-review)`
