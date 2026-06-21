# Stage 2: Selector Ablation — Strategy 1 vs Strategy 2

**Goal**: Compare the two candidate-selection strategies head-to-head — **Strategy 1** (steepest-ascent on target-class probability) vs **Strategy 2** (class-divergence) — across both datasets at a fixed baseline context, and pick the selector used for the downstream context ablation.
**Dependencies**: Stage 1 DONE (greedy loop + both selectors + Exp4 runner exist). Independent of Stage 3.

---

## Design: a one-factor ablation

| Factor | Levels | Notes |
|--------|--------|-------|
| **Selector** | `prob_ascent` (Strategy 1) · `class_divergence` (Strategy 2) | The only thing that varies. |

Datasets: MOONS + HELOC ⇒ **4 cells** (2 selectors × 2 datasets). Everything else is held at the **Stage-1 baseline**: `max_context=256`, `t=1e-9` (MAP commit), `n_permutations` and `--max-test` at the Stage-1 values, identical across the two selectors **within a dataset** so the comparison is fair.

**Context-strategy constraint (Decision #6).** Strategy 2 requires a both-classes context pool (non-constant Y); Strategy 1 uses target-only. So the two cells within a dataset *necessarily* differ in context scope. **Record this** in the summary: the comparison is "each selector at its natural/required context," not "both at an identical context." (An identical-context comparison is impossible without crippling Strategy 2; the apples-to-apples contrast is deferred to Stage 4, where the context strategy is itself a controlled axis and Strategy 1 is run across all four.)

---

## Steps

1. **Add the selector-ablation driver.**
   - New file: `experiments/zeroshot_cf/exp5_selector_ablation.py`.
   - For a dataset, run the Stage-1 Exp4 generation+metrics path once per selector (`prob_ascent`, `class_divergence`), wiring the required context scope per selector (target_only / all_classes).
   - Hold `max_context`, temperature, `n_permutations`, and `--max-test` identical across the two selectors within the dataset; log every held value.
   - Reuse the Stage-1 metric computation (incl. the **inline `frac_oob`** on unclipped CFs — `compute_metrics` does not return it; see Stage 1 Step 3).
   - Writes `results/exp5_selector_{moons,heloc}.csv` — one row per selector with columns: `selector, context_scope, n_test, validity, l0_count_mean, steps_mean, steps_median, steps_max, failure_rate, lof_scores_cf, true_actionability, proximity_l2_jaccard, frac_oob, runtime_s`.

2. **Write the comparison summary.**
   - `results/exp5_summary.md` — per-dataset table + an honest verdict: which selector wins on **validity**, on **L0 count (`l0_count_mean`)**, on **steps-to-flip**, and on **plausibility (frac_oob/LOF)**? State plainly if they tie (expected on MOONS — only 2 actionable features) and note the context-scope caveat above.
   - Record the **chosen downstream selector** (used by Stage 4). Default tie-break: prefer `prob_ascent` (compatible with all four context strategies; directly optimizes the flip). If `class_divergence` clearly wins on plausibility without losing validity, choose it and note that Stage-4's `*_target` context cells will be skipped (Decision #6).

3. **Run the ablation (offline, v2 checkpoints).**
   - See `resources/commands.md`. Use the same `--max-test` for both selectors within a dataset.

4. **Tests.**
   - Extend `experiments/zeroshot_cf/tests/test_greedy.py` (or add `test_selector_ablation.py`, using the conftest `models` fixture): assert the driver produces a CSV with one row per selector and that the `context_scope` column is `all_classes` for `class_divergence` and `target_only` for `prob_ascent`. No new model behaviour to test beyond Stage 1.

---

## Verification

- [ ] `results/exp5_selector_{moons,heloc}.csv` each have exactly 2 rows (one per selector) with the columns above.
- [ ] Within each dataset, `max_context`, temperature, `n_permutations`, and `n_test` are identical across the two selector rows (logged).
- [ ] `results/exp5_summary.md` names the winning selector per metric and the single **chosen downstream selector**, with the context-scope caveat stated.
- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` passes.
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` is empty; no `tabpfn_client` import.

---

## Expected outcomes (record actuals against these)

- **MOONS**: `prob_ascent` ≈ `class_divergence` (only 2 actionable features → little selection freedom); both validity ≈ 1.0, steps ≤ 2.
- **HELOC**: `prob_ascent` should reach the flip in **fewer steps** (it optimizes the flip directly) and likely higher validity; `class_divergence` may yield more class-coherent / plausible features (lower frac_oob) at a possible cost in steps. Record which trade-off materializes.

---

## Commit

`feat(greedy-cf): selector ablation — prob-ascent vs class-divergence across MOONS+HELOC (Exp5)`
