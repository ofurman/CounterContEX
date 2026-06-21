# Stage 4: Context Ablation — Size × Strategy

**Goal**: Ablate the conditioning context across **size** {256, 512, 1024, 2048} × **strategy** {random_target, random_both, knn_target, knn_both} for the greedy CF generator, on both datasets, at the Stage-2 winning selector — then write the consolidated report and recommend a production context configuration.
**Dependencies**: Stage 1 DONE (greedy + Exp4), Stage 2 DONE (chosen selector), Stage 3 DONE (kNN context support in `set_context`).

---

## Design: a two-factor grid

| Factor | Levels | Notes |
|--------|--------|-------|
| **Context size** (`max_context`) | 256 · 512 · 1024 · 2048 | Capped at the available pool and logged (Decision #7). MOONS (~800 train) saturates above ~400/class — report effective size. |
| **Context strategy** | `random_target` · `random_both` · `knn_target` · `knn_both` | Encodes class scope × selection method (Decision #4). `random_target` is the predecessor-plan baseline. |

Grid = **4 × 4 = 16 cells per dataset**, run at the **Stage-2 winning selector** (default `prob_ascent`). Everything else (temperature, `n_permutations`, `--max-test`) held at the Stage-1 baseline, **identical across all 16 cells within a dataset** so size/strategy are the only things that vary.

**Selector compatibility (Decision #6).** If the Stage-2 winner is `class_divergence`, it needs a both-classes pool → the 8 `*_target` cells are **skipped with a logged note**, leaving an 8-cell grid (4 sizes × {random_both, knn_both}). If the winner is `prob_ascent` (compatible with all four), run the full 16. Record which case applied.

**kNN context is per-query (Decision #5).** Unlike the one-pass path (one context fit per target-class batch), `knn_*` strategies select context from each factual point, so context is fit **per test point** inside the greedy driver: the exp6 `knn_*` loop is `for each test point: set_context(..., selection="knn", query=x) → greedy_counterfactual(...)`. This is consistent with Stage 1's contract that `greedy_counterfactual` takes a **pre-fitted** sampler and is agnostic to fit granularity. It is the dominant cost on HELOC — bound it with a small `--max-test` held identical across cells, and `log()` the chosen value. `random_*` cells may reuse the per-class fit for speed.

See `resources/grids.md` for the full cell list and `resources/commands.md` for run commands.

---

## Steps

1. **Add the context-ablation driver.**
   - New file: `experiments/zeroshot_cf/exp6_context_ablation.py`.
   - For a dataset and the chosen selector, iterate the (size, strategy) grid. For each cell, wire `set_context(..., max_context=size, target_class=<target or None per strategy>, selection=<random|knn>, query=<factual point for knn>)` and run the Stage-1 greedy generation + metrics.
   - Map strategy → (`target_class`, `selection`): `random_target`→(t, random), `random_both`→(None, random), `knn_target`→(t, knn), `knn_both`→(None, knn).
   - Reuse the Stage-1 metric computation (incl. the **inline `frac_oob`** on unclipped CFs — `compute_metrics` does not return it; see Stage 1 Step 3).
   - Writes `results/exp6_context_{moons,heloc}.csv` — one row per cell: `selector, size, effective_size, strategy, class_scope, selection, n_test, validity, l0_count_mean, steps_mean, failure_rate, lof_scores_cf, true_actionability, proximity_l2_jaccard, frac_oob, runtime_s`.

2. **Write the consolidated report.**
   - `results/exp6_summary.md` — per-dataset grid tables (size × strategy) for the headline metrics (validity, l0_count_mean, frac_oob, LOF), plus an honest verdict:
     - Does **larger context** lift HELOC validity / lower frac_oob, and where does it saturate?
     - Does **kNN** (relevance-based) beat **random** at equal size? Does drawing from **both classes** vs **target only** matter?
     - The single **recommended (selector, size, strategy)** for HELOC and for MOONS, with the metric trade-off stated.
   - Extend `results/REPORT.md` with an **"Iterative greedy CF + context ablation"** section consolidating Exp4 (greedy vs one-pass), Exp5 (selector), and Exp6 (context), and an updated TL;DR on whether greedy + better context salvages HELOC.

3. **Run the ablation (offline, v2 checkpoints).**
   - See `resources/commands.md`. Keep `--max-test` identical across all cells within a dataset; if reduced for runtime, log it. Sizes above the available pool are capped and the effective size recorded.

4. **Tests.**
   - Extend `experiments/zeroshot_cf/tests/test_context.py` (or add `test_context_ablation.py`, using the shared `models` fixture in `tests/conftest.py`): assert the driver emits the expected number of rows (16, or 8 if the selector is `class_divergence`), that `effective_size <= size` and `effective_size <= pool_size`, and that `class_scope`/`selection` columns match the strategy name. No new model behaviour beyond Stage 3.

---

## Verification

- [ ] `results/exp6_context_{moons,heloc}.csv` exist with the expected cell count (16 for `prob_ascent`; 8 for `class_divergence`, with the skip logged).
- [ ] Within each dataset, temperature / `n_permutations` / `n_test` are identical across all cells (logged); `effective_size <= size` recorded per cell.
- [ ] `true_actionability == 1.0` for every cell.
- [ ] `results/exp6_summary.md` states whether larger context and/or kNN improved HELOC validity & plausibility, and names the recommended (selector, size, strategy) per dataset — even if the answer is "no improvement."
- [ ] `results/REPORT.md` has the consolidated greedy + ablation section.
- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` passes.
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` is empty; no `tabpfn_client` import.

---

## Expected outcomes (record actuals against these)

- **MOONS**: size axis saturates early (small train); strategy likely matters little (2 actionable features, easy class structure). Validity ≈ 1.0 throughout.
- **HELOC**: the informative grid. Hypotheses — (i) larger context monotonically helps validity/plausibility up to a saturation point; (ii) `knn_*` beats `random_*` at equal size by supplying more relevant conditioning; (iii) `*_both` vs `*_target` trades Y-informativeness against on-target density. Record which hold. The best HELOC cell becomes the recommended config for any future work.

---

## Commit

`feat(greedy-cf): context ablation (size × strategy) + consolidated report (Exp6)`
