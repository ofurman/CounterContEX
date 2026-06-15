# Stage 6: Inference Refinement & Results Report

**Goal**: If Experiment 2 results are weak, refine the **inference process only** (context selection, temperature, permutations, feature ordering) — never retrain — then produce a consolidated results report with conclusions.
**Dependencies**: Stage 5 (baseline Exp 2 results)

---

## Steps

1. **Sweep harness.**
   - File: `experiments/zeroshot_cf/refine.py` + `experiments/zeroshot_cf/configs/sweep.yaml`
   - Grid over inference knobs (all in-context, zero retraining):
     - **Context selection**: target-class-only vs. mixed; nearest-neighbour context (rows closest to the factual) vs. random; `max_context` size ∈ {100, 256, 512, full}.
     - **Temperature** `t` ∈ {1e-9, 0.25, 0.5, 1.0} — proximity/diversity trade-off.
     - **`n_permutations`** ∈ {1, 3, 10} — ordering Monte-Carlo.
     - **Feature ordering**: random permutations vs. a fixed `dag` (e.g. order actionable features by correlation with the target).
   - Keep the grid small per dataset (cost: each impute re-fits TabPFN per column/permutation). Use MOONS first (cheap) to find promising regions, then a reduced grid on HELOC. **Log any grid points skipped for budget** (no silent truncation).

2. **Optimize for the right objective.**
   - Track the joint trade-off: validity ↑, LOF plausibility ↓ (more plausible), proximity L2 ↓, sparsity ↓, true_actionability = 1.0.
   - Surface the Pareto front; pick a recommended config per dataset. Note proximity is a known structural weakness (no minimal-perturbation mechanism) — if it stays poor, say so and note that a proximity-enforcing wrapper (e.g. post-hoc nearest-feasible projection) is future work, not part of this zero-shot test.

3. **Consolidated report.**
   - File: `experiments/zeroshot_cf/results/REPORT.md`
   - Sections: setup (offline, checkpoints, datasets, oracle, actionability split), Exp 1 findings + gate verdict, Exp 2 baseline metrics, refinement sweep results + Pareto front + recommended configs, comparison vs. cel baselines, and an **honest verdict**: is zero-shot autoregressive TabPFN a viable CF generator out-of-the-box? What works, what doesn't, what would the next iteration change?
   - Update `experiments/zeroshot_cf/README.md` with the final run commands and headline results.

4. **Persist learnings to memory** (per task-planning skill): record the offline-checkpoint mechanism, the Y-as-column conditioning trick, the chosen HELOC immutable split, baseline metric numbers, and the verdict — so future conversations don't re-derive them.

---

## Verification

- [ ] `uv run python experiments/zeroshot_cf/refine.py --dataset moons` completes; writes sweep CSV.
- [ ] Reduced HELOC sweep completes; any skipped grid points are logged.
- [ ] `results/REPORT.md` exists with all sections, recommended configs, and the cel-baseline comparison.
- [ ] Recommended configs reproduce their reported metrics when re-run via `exp2_counterfactuals.py` with those settings.

---

## Commit

`docs(zeroshot-cf): inference refinement sweep + consolidated results report`
