# Stage 7: Post-Review Correctness & Reproducibility Fixes

**Goal**: Fix the P1/P2/P3 issues found by `/plan-post-review` — at the **code + test + docs** level only, without re-running the expensive experiments (that's Stage 8).
**Dependencies**: Stages 1–6 DONE.

> **Why this stage exists**: a post-implementation review found 3×P1, 4×P2, 5×P3. The headline defect is that Exp2 **validity is scored against the wrong reference label**, so the reported Exp2 metric table is not trustworthy until the code is fixed and the results regenerated. This stage makes every fix that can be verified by unit tests / `--help` / inspection. Stage 8 then regenerates the numbers.

**Guardrail reminder**: do NOT modify `src/tabpfn/**`. Keep the commit buildable; never weaken a test to pass.

---

## Steps

### P1 — must fix before Exp2 numbers can be cited

1. **Fix the validity reference (the crux bug).**
   - Files: `experiments/zeroshot_cf/metrics_harness.py` (validity ≈ line 61, proximity `valid_mask` ≈ lines 80-86), `experiments/zeroshot_cf/exp2_counterfactuals.py` (call site ≈ lines 165-172), `experiments/zeroshot_cf/refine.py` (if it computes validity — check; if it calls `compute_metrics`, the fix propagates automatically, otherwise fix its inline copy too).
   - Current bug: generation target is `y_target = 1 - y_pred` (`exp2_counterfactuals.py:81`), but `compute_metrics` scores `validity = (y_cf_pred != y_test).mean()`. These diverge on misclassified factuals (~28% HELOC, ~13% MOONS).
   - Fix: add a required `y_target` parameter to `compute_metrics(...)`. Define `validity = float((y_cf_pred == y_target_arr).mean())` and base the proximity `valid_mask` on the **same** `y_cf_pred == y_target_arr`. Update the `exp2` call site to pass `y_target=info["y_target"]`. Keep `y_test` only if still needed for other purposes (it isn't for validity/proximity).
   - Make `write_examples` (`exp2:217`, currently `!= y_pred`) and the metric use the **identical** definition (`== y_target`, equivalently `!= y_pred` for the binary flip) so generation, examples, and metric all agree.
   - Add a unit test `tests/test_metrics_harness.py`: construct a tiny case with a deliberately misclassified factual (so `y_pred != y_test`) and assert `validity` counts a CF as valid iff `predict(X_cf) == y_target`, NOT `!= y_test`. This is the regression guard for the bug.

2. **Stop presenting HELOC LOF as a finite measurement.**
   - File: `experiments/zeroshot_cf/metrics_harness.py` (LOF ≈ lines 71-73), `exp2_counterfactuals.py` (clip ≈ line 155).
   - Root cause: LOF is computed on the **clipped** `X_cf`; with 66% OOB rows collapsed onto [0,1] corners, distances degenerate → LOF ≈ 2.5e9, which is an artefact, not a plausibility score.
   - Fix (choose the cleaner): compute `lof_scores_cf` on the **unclipped** `X_cf` so the distances reflect the true generated geometry; and/or, when `frac_oob` is high, flag the LOF value as degenerate. Recommended: pass both `X_cf` (unclipped) for LOF and the clipped version for validity/proximity, OR add a boolean `lof_degenerate = frac_oob > 0` to the metrics dict. Whichever is chosen, REPORT.md (Stage 8) must label HELOC LOF accordingly rather than printing a bare `2.5e9`.

3. **Make recommended configs reproducible via the documented command.**
   - Files: `experiments/zeroshot_cf/exp2_counterfactuals.py` (hardcoded `TEMPERATURE=1.0` ≈ line 41, target-only context ≈ line 112), `experiments/zeroshot_cf/README.md`.
   - Fix: add CLI flags `--temperature`, `--context-type {target_only,all_classes}`, `--n-permutations`, `--max-context` (argparse), threaded into the `ConditionalDensitySampler` / `set_context` calls. Defaults = the current baseline so existing behavior is unchanged. Update the README run command so the MOONS recommended config (`t=0.5, context=all_classes`) is reproducible exactly: e.g. `... exp2_counterfactuals.py --dataset moons --temperature 0.5 --context-type all_classes`.
   - Verify `exp2_counterfactuals.py --help` lists the new flags.

### P2 — should fix

4. **Seed all RNGs from `random_state`.**
   - File: `experiments/zeroshot_cf/sampler.py` (`__init__` and/or `set_context`/`impute_masked`).
   - Current gap: `random_state` only seeds subsampling; the underlying `TabPFNUnsupervisedModel.impute` draws permutations via Python's global `random` and samples via `torch` at `t>0`, so `n_permutations>1` / `t>0` runs aren't reproducible.
   - Fix: in `set_context` (before fit) and `impute_masked`/`sample_feature` (before the impute call), seed `random.seed(self.random_state)`, `np.random.seed(self.random_state)`, `torch.manual_seed(self.random_state)`. Note MPS float nondeterminism may remain — document that caveat.

5. **Assert immutability instead of just printing it.**
   - File: `experiments/zeroshot_cf/exp2_counterfactuals.py` (≈ line 160, currently prints `max_dev`).
   - Fix: `assert max_dev < 1e-9, f"Immutable columns drifted: max_dev={max_dev}"` (skip when `immutable_idx` is empty, e.g. MOONS).

### P3 — polish

6. **Align Exp1 calibration to spec.**
   - Files: `experiments/zeroshot_cf/sampler.py` (`sample_feature` ≈ lines 236-250, currently draw 1 = MAP at configured `t`, draws 2..N at `t=1.0`), `exp1_single_feature.py` (calibration interval ≈ lines 138-153).
   - Fix: separate the **point estimate** (1 draw at `t=1e-9`, MAP) from the **posterior interval** (N draws all at `t=1.0`). Don't mix a MAP draw into the quantile sample. Expose a `sample_temperature` arg on `sample_feature` rather than hardcoding `1.0`.

7. **Fix the MOONS scatter to plot the scored reconstruction.**
   - File: `experiments/zeroshot_cf/exp1_single_feature.py` (≈ lines 225-250). The plot re-runs the sampler without class conditioning (`set_context` w/o `target_class`), so it visualizes a different quantity than the CSV. Use the same per-class conditioning that was scored.

8. **Add the cel-baselines Backlog entry & clear the stale TODO.**
   - The Backlog entry has already been added to `index.md` (item 1). This step only needs to remove/replace the literal `TODO (fill during execution)` placeholder in `resources/api-reference.md` (≈ line 111) with a pointer to Backlog item 1, so there is no dangling TODO.

9. **Resolve the "persisted to memory" claim.**
   - File: `docs/plans/zeroshot-tabpfn-cf/index.md` (tracker note for Stage 6, line ~109).
   - The review flagged this claim as unverifiable. Either (a) actually write the learnings to NeoCortex/memory and keep the claim, or (b) soften it to point at the concrete sink (`results/REPORT.md` + Decisions section). Pick (b) if no memory store is wired in this environment.

---

## Verification

- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` passes, including the **new** `test_metrics_harness.py` validity-reference regression test.
- [ ] The existing `test_sampler.py` 4 tests still pass (seeding change must not break them).
- [ ] `uv run python experiments/zeroshot_cf/exp2_counterfactuals.py --help` shows `--temperature`, `--context-type`, `--n-permutations`, `--max-context`.
- [ ] `grep -rn "TODO (fill" experiments/zeroshot_cf/ ../../docs/plans/zeroshot-tabpfn-cf/resources/` returns nothing (stale TODO cleared).
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` is empty (core untouched).
- [ ] Index Backlog item 1 (cel baselines) present; Stage 6 memory claim resolved.

---

## Commit

`fix(zeroshot-cf): correct validity reference, LOF presentation, repro flags + seeding (post-review)`
