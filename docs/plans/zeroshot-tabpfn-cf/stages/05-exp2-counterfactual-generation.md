# Stage 5: Experiment 2 — Counterfactual Generation via Inference

**Goal**: Generate counterfactuals by freezing immutable features, masking actionable ones, conditioning on the target class, and imputing — then evaluate the 5-metric subset on HELOC and MOONS.
**Dependencies**: Stage 3 (sampler), Stage 2 (data, discriminator, metrics), Stage 4 (gate verdict)

---

## Steps

1. **Runner.**
   - File: `experiments/zeroshot_cf/exp2_counterfactuals.py`
   - For each dataset, for each factual test point `x` with predicted class `c` (from the cel `disc_model`), set `target = 1 - c` (binary flip; matches cel's default `y_target = abs(1 - y_test)`).
   - Build the **context** from train rows of the **target class** (Decision: class-conditional context). Configure `ConditionalDensitySampler(append_target=True, ...)`, `set_context(X_train, y_train, target_class=target, max_context=...)`.
   - Construct the query: copy `x`, keep immutable features frozen (observed), NaN-mask actionable features, fix the appended Y column to `target`. Call `impute_masked(x, mask_cols=actionable_idx, fixed_target=target)` → counterfactual `x_cf`.
   - Note: with `append_target=True`, context must be fit per target class. To avoid refitting per point, **batch points by target class** (all points needing target=1 share one fitted context, likewise target=0).

2. **Assemble CF set & evaluate.**
   - Stack `X_cf` aligned to `X_test`; immutable columns must equal the factual exactly (assert).
   - Run `metrics_harness`: `validity`, `lof_scores_cf`, `sparsity`, `actionability` (cel's), `true_actionability` (ours), `proximity_l2_jaccard`. Pass `disc_model` = cel oracle, `continuous_features`, `y_target`.
   - Report metrics in **scaled space** (consistent with cel); optionally also inverse-transform CFs to original space for human-readable examples.

3. **Edge cases & honesty.**
   - Some imputations may produce out-of-[0,1] values (TabPFN samples in raw space) — report the fraction and optionally clip; document the choice.
   - Validity may be low (the brief's main risk). Record the **actual** numbers without massaging. If validity is near zero, that itself is a finding and motivates Stage 6.
   - `true_actionability` must be 1.0 by construction (immutables frozen). If not, there's a column-indexing bug — fix before trusting other metrics.

4. **Outputs.**
   - `results/exp2_<dataset>_metrics.csv` (one row per config), a few concrete CF examples (factual vs CF, original space) in `results/exp2_examples.md`, and a comparison table vs. cel baselines (validity/proximity/sparsity/LOF) in `results/exp2_summary.md`.

---

## Verification

- [ ] `uv run python experiments/zeroshot_cf/exp2_counterfactuals.py --dataset moons` completes; writes metrics CSV.
- [ ] `--dataset heloc` completes; writes metrics CSV.
- [ ] `true_actionability == 1.0` on both datasets (immutables provably untouched).
- [ ] All 5 metrics present and finite; summary table compares against cel baselines.
- [ ] At least 3 concrete CF examples (original feature space) written for HELOC.

---

## Commit

`feat(zeroshot-cf): Experiment 2 counterfactual generation + metric evaluation`
