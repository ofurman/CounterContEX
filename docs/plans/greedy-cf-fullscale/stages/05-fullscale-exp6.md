# Stage 5: Full-Scale Context Ablation (Exp6)

**Goal**: Re-run the 16-cell context ablation (size {256,512,1024,2048} × strategy {random_target, random_both, knn_target, knn_both}) at HELOC `--max-test 200` / MOONS full, so the validity *trends* (not just the monotone frac_oob/LOF signal the predecessor leaned on at n=15) are robust.
**Dependencies**: Host preflight passing (DGX). Stage 2 (programmatic `write_summary`) DONE so the regenerated summary isn't a placeholder. Benefits from Stage 3 (`--beam`) for runtime. This is the dominant-cost stage (~3 DGX-days at n≈200).

---

## Context

Predecessor Exp6 was bounded to HELOC `--max-test 15` (Decision #13; the 16-cell grid at n=15 took
≈5.3 h, size-2048 kNN cells dominating). At n=15 validity is ±0.12 noisy, so the predecessor's
recommendation (`knn_both@256`) rests on the monotone frac_oob/LOF trends, not validity. Full scale
removes that caveat.

Headline to confirm: bigger *random* context HURTS HELOC (frac_oob 256→2048: 0.13→0.53); kNN beats
random at every size; best cell `knn_both@256` (frac_oob 0.000 / LOF 1.98). MOONS recommended
`random_both@512` (validity 0.82).

---

## Steps

1. **Run Exp6 at full scale on the DGX** (detached; the long pole — use a `*.DONE` sentinel and poll):
   - MOONS: full test split. HELOC: `--max-test 200`.
   - Selector = the Stage-4 winner (expected `prob_ascent` → full 16 cells; if `class_divergence`,
     the 8 `*_target` cells are skipped with a logged note per predecessor Decision #6).
   - Hold temperature / `n_permutations` / `--max-test` identical across all cells within a dataset.
   - Use `--beam` if needed to fit the time budget; if used, apply the **same** beam to every cell and
     record it (a beam is part of the method — it must be constant across the grid to keep size×strategy
     the only varying factors). Cap sizes at the available pool and record `effective_size` per cell.
2. **Regenerate artefacts**: `results/exp6_context_{moons,heloc}.csv` (one row per cell, effective-n and
   `effective_size` columns) and `results/exp6_summary.md` (**programmatic** verdict from Stage 2:
   recommended (size, strategy) per dataset, whether larger context helped, whether kNN beat random at
   equal size, whether both-classes vs target-only mattered).
3. **Pull back and commit.**

---

## Verification

- [ ] `results/exp6_context_{moons,heloc}.csv` regenerated with the expected cell count (16 for
      `prob_ascent`; 8 for `class_divergence` with the skip logged), HELOC effective n≈200, MOONS full.
- [ ] `effective_size <= size` and `<= pool_size` per cell; held values identical across cells (logged).
- [ ] `true_actionability == 1.0` for every cell.
- [ ] `results/exp6_summary.md` is programmatically generated (no `_Placeholder_`) and states whether the
      predecessor's `knn_both@256` recommendation survives at n≈200, with the tighter-CI validity trend.
- [ ] `git diff --name-only <base>..HEAD -- src/tabpfn` empty; no `tabpfn_client` import.

---

## Expected outcomes (record actuals)

- The frac_oob/LOF ordering (kNN ≫ random; small context ≫ large on HELOC) holds; the validity trend now
  has a tight enough CI to either confirm or revise `knn_both@256` as the HELOC recommendation. Either
  way is a reportable, defensible result.

---

## Commit

`feat(greedy-cf): full-scale context ablation (Exp6 16-cell grid, HELOC n=200 / MOONS full)`
