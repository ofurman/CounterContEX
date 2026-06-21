# Stage 1: Iterative Greedy Core + Both Selectors

**Goal**: Build the iterative greedy counterfactual loop — change actionable features one at a time, stop at the class flip — with both candidate-selection strategies (prob-ascent, class-divergence), a single-column predictive-distribution helper on the sampler, the Exp4 runner with L0/steps metrics, and tests.
**Dependencies**: Predecessor plan `zeroshot-tabpfn-cf` Stages 1–9 DONE. Reuses `ConditionalDensitySampler`, the discriminator oracle, `load_dataset`/`get_actionable_immutable`, and `metrics_harness`. Gates all later stages in this plan.

---

## The approach: relabel-conditioned, one-feature-at-a-time, stop at flip

**Setup (identical conditioning trick to predecessor Stage 5).** Context = training set with the target label appended as a categorical column. For factual point `x` with predicted class `c = disc.predict(x)`, target `y_target = 1 − c`. Build the CF up from `x`:

- `x_cf ← copy(x)` ; `changed ← {}`.
- **Candidates** `R` = actionable features not yet changed. **Immutable features are never candidates** → `true_actionability = 1.0` by construction.
- Each masked feature is drawn class-conditionally `x_j ~ p(x_j | x_cf_{−j}, Y=y_target)` via a **single-column** call (`impute_masked(mask_cols=[j], fixed_target=y_target)`): all other features observed, only `j` masked.

**The greedy loop:**

```
x_cf = copy(x);  changed = {}
while disc.predict(x_cf) != y_target  and  |changed| < budget:                # budget = |A|
    j*  = select_candidate(R = A \ changed, x_cf, y_target)                    # Strategy 1 or 2
    v   = sample_feature(x_cf, target_col=j*, fixed_target=y_target, t)        # near-MAP, single column, class-conditional
    x_cf[j*] = v ;  changed.add(j*)
return x_cf, changed                                                          # L0 sparsity = |changed|
```

**Committing the single-column value — the commit MUST be class-conditional (`Y=y_target`).** ⚠️ **The greedy draw is `p(x_j | x_cf_{−j}, Y=y_target)`, which requires `append_target=True`.** Two facts about the current sampler force a small prerequisite change:
- `impute_masked(X_query, mask_cols, fixed_target=...)` **raises `ValueError("fixed_target required when append_target=True")`** (sampler.py:207–208) when `fixed_target is None` — i.e. you *must* pass `fixed_target` under the class-conditional regime.
- `sample_feature(X_query, target_col, sample_temperature=...)` (sampler.py:249–304) is the **only** path that accepts a **per-call** temperature, but as written it calls `impute_masked(X_query, mask_cols=[target_col])` **without `fixed_target`** (sampler.py:287). So under `append_target=True` it would crash, and under `append_target=False` it is **not** class-conditional. It therefore cannot commit a `Y=y_target` value as-is.

  **Fix (prerequisite, Step 1):** add a `fixed_target: Optional[int] = None` parameter to `sample_feature` and forward it to `impute_masked` (`self.impute_masked(X_query, mask_cols=[target_col], fixed_target=fixed_target)`). Default `None` preserves the existing `append_target=False` test (`test_sampler.py:73`). The greedy loop then commits via `sampler.sample_feature(x_cf, target_col=j*, fixed_target=y_target, sample_temperature=temperature)`. (`sampler.py → sample_feature` is added to "Files That May Be Changed" in `index.md`.) Note: with a **single** masked column, `n_permutations` is irrelevant to determinism — there is only one column to fill, so near-MAP single-column commits are deterministic regardless of `n_permutations`. The near-MAP commit (`t≈1e-9`) is the distribution's **mode bucket** (`sample(logits, t)` ≈ `icdf(logits/t, U)` saturates to the argmax bucket as `t→0`; `bar_distribution.py:583`), which is *not* the same as the bar-distribution **mean** used by Strategy-2 divergence / `mean_of_prediction` — keep the two distinct.

**Stopping condition (primary): the class flip** — `disc.predict(x_cf) == y_target`. Auxiliary stop knobs: `--tau` (stop when `disc.predict_proba(x_cf)[y_target] ≥ τ`, default 0.5 ≡ hard flip) and `--budget` (≤ |A|; if exhausted without a flip, return best-effort `x_cf`, mark **invalid**, count in a failure rate). Committed value drawn at **near-MAP** (`t ≈ 1e-9`) for determinism.

**Why this optimizes sparsity.** L0 = `|changed|`, optimized by forward selection: features added only until the flip. Contrast with one-pass, where L0 is fixed at `|A|`.

---

## Candidate-selection strategies (two alternatives — compared in Stage 2, not combined)

Both share the same loop, single-column MAP value generation, and flip stop condition. They differ **only** in `select_candidate`.

### Strategy 1 — Steepest-ascent on target-class probability (wrapper / score-driven)

For each remaining candidate `j ∈ R`: draw its near-MAP value `v_j = sampler.sample_feature(x_cf, target_col=j, fixed_target=y_target, sample_temperature=1e-9)` (class-conditional — see the commit note above); form `x_cf[j := v_j]`; score `s_j = disc.predict_proba(x_cf[j := v_j])[y_target]`. Select `j* = argmax_j s_j` and commit `v_{j*}`.

- **Rationale**: ranks by the *actual* effect on the classifier whose flip is the stop condition (SEDC best-first / NICE sparsity reward); minimal-L0 for a linear discriminator.
- **Context**: target-class context (`context_type="target_only"`) is sufficient.
- **Cost**: `O(|R|)` imputes + `O(|R|)` classifier evals per step → `O(|A|²)` imputes per run. Imputes reuse `sampler.sample_feature`/`impute_masked`; classifier evals are cheap sklearn calls.

### Strategy 2 — Class-divergence (TabPFN-intrinsic, classifier-free selection)

For each remaining candidate `j ∈ R`, compute the two class-conditional predictive distributions given the current row — `P_tgt = p(x_j | x_cf_{−j}, Y=y_target)` and `P_cur = p(x_j | x_cf_{−j}, Y=c)` — and a divergence `div_j = D(P_tgt, P_cur)`. For these all-continuous datasets the feature routes to the **regressor**, so `density_`'s `predict(..., output_type="full")` yields a `FullSupportBarDistribution` for each. Default divergence = **absolute mean shift** `|E[x_j|Y=y_target] − E[x_j|Y=c]|` (normalized by feature range); **symmetric KL** between the bar distributions (shared borders) is the alternative. Select `j* = argmax_j div_j`, then draw its near-MAP value under `Y=y_target` via `sample_feature(x_cf, target_col=j*, fixed_target=y_target, sample_temperature=1e-9)` (class-conditional — see the commit note above) and commit.

- **Rationale**: selects the most class-determined feature using only TabPFN's density — no dependence on the discriminator being explained.
- **Context**: **requires `context_type="all_classes"`** (Y must be non-constant for the contrast; same as predecessor Decision #10).
- **Cost**: `2·O(|R|)` predictive-distribution reads per step + one MAP impute for the chosen feature → `O(|A|²)`. Note `density_` calls `model.fit(...)` on **every** invocation (`unsupervised.py:643`), so each predictive-distribution read still pays a per-column TabPFN fit — the same fit cost both strategies incur. The saving over Strategy 1 is the skipped per-candidate **sample + discriminator eval**, not the fit.

The stop condition is the same in both — the origin classifier's flip; only the per-step selection signal differs.

---

## Steps

1. **Make `sample_feature` class-conditional, then expose a single-feature predictive distribution on the sampler.**
   - File: `experiments/zeroshot_cf/sampler.py`.
   - **1a. Class-conditional `sample_feature` (prerequisite for the greedy commit).** In `sample_feature` (≈ lines 249–304) add a `fixed_target: Optional[int] = None` parameter and forward it to **both** `impute_masked` calls inside it (the `n_samples==1` path at ≈ line 287 and the `n_samples>1` loop): `self.impute_masked(X_query, mask_cols=[target_col], fixed_target=fixed_target)`.
     - **Why required**: the greedy loop draws `p(x_j | x_cf_{−j}, Y=y_target)` ⇒ `append_target=True`, and `impute_masked` raises `ValueError` when `append_target=True` and `fixed_target is None` (sampler.py:207–208). Without forwarding, every greedy commit crashes.
     - **Backward compatibility**: default `None` preserves the existing `append_target=False` behaviour — the current `sample_feature` test (`test_sampler.py:73`) does not pass `fixed_target` and must still pass unchanged. Leave `impute_masked` itself untouched.
   - **1b. `predictive_distribution` (for Strategy 2).** Add `predictive_distribution(self, X_query, target_col, fixed_target)` returning, per query row, the masked feature's conditional distribution under `Y=fixed_target` **without sampling**. ⚠️ This is **not** "`impute_masked` minus the sample" — `impute_masked` samples internally via `model.impute → sample_from_model_prediction_`, and there is **no public path** that returns the raw distribution. Use the concrete recipe below (verified reachable: `smoke_test.py:43–49` shows `reg.predict(X, output_type="full") → {"logits", "criterion"}`; the underlying `TabPFNUnsupervisedModel` exposes `density_`).
     - Build the same augmented matrix as `impute_masked`: NaN-mask `target_col`, append the `Y=fixed_target` categorical column, apply the same RNG re-seeding (mirror `impute_masked` lines 199–201).
     - Call the underlying model's conditional-density primitive: `model_j, X_predict, _ = self.model.density_(X_masked_rows, self.model.X_, conditional_idx, column_idx)` where `column_idx = target_col` and `conditional_idx = [every augmented column except target_col]` (i.e. all observed features + the appended Y column).
     - For a **regressor** column (all HELOC/MOONS features are continuous → regressor; verified: the sampler marks **only** the appended Y column categorical via `set_categorical_features([last_idx])`, so `use_classifier_` routes every feature column to the regressor): `out = model_j.predict(X_predict, output_type="full")` → return `out["logits"]` and `out["criterion"]` (a `FullSupportBarDistribution`). Pass `X_predict` as the **tensor** returned by `density_` (do **not** `.numpy()` it) — this matches the verified internal caller `outliers_single_permutation_` (`tabpfn_extensions/unsupervised/unsupervised.py:753`). For a **classifier** column (not expected here): return `model_j.predict_proba(X_predict.numpy())`.
   - Add helpers `mean_of_prediction(logits, criterion)` (the bar-distribution expected value, e.g. `criterion.mean(logits)` — confirm the exact accessor against the installed `FullSupportBarDistribution`) and `symmetric_kl(logitsA, logitsB, criterion)` (KL between two bar distributions that **share the same borders**) so Strategy 2 reads divergences without duplicating bar-border logic.
   - Leave `impute_masked` untouched (Strategy 1 reuses it; only `sample_feature` gains the `fixed_target` pass-through from Step 1a).
   - **Correctness test** (add in Step 5): pin that `predictive_distribution(X, j, fixed_target=t)` describes the **same class-conditional** distribution that `sample_feature` draws from. ⚠️ Do **not** compare the distribution **mean** against a single near-MAP sample: the `t≈1e-9` draw is the **mode** bucket, not the mean (they differ for skewed bar distributions, so a correct implementation could fail a mean-vs-MAP check). Compare like-for-like, both conditioned on `Y=t`:
     - **Mean-vs-empirical-mean (preferred):** `mean_of_prediction(predictive_distribution(X, j, fixed_target=t))` ≈ empirical mean of many full-temperature draws `sample_feature(X, target_col=j, fixed_target=t, sample_temperature=1.0, n_samples=200)` (mean over the sample axis), within a stated tolerance.
     - **Mode-vs-MAP (alternative):** the `predictive_distribution` mode (`criterion.mode(logits)` / argmax-bucket value) ≈ `sample_feature(X, target_col=j, fixed_target=t, sample_temperature=1e-9)`.

2. **Add the greedy loop module.**
   - New file: `experiments/zeroshot_cf/greedy.py`.
   - `def greedy_counterfactual(sampler, disc, x, y_target, actionable_idx, selector, *, tau=0.5, budget=None, temperature=1e-9) -> (x_cf, changed, history)`.
   - Implements the loop above. `selector ∈ {"prob_ascent", "class_divergence"}` dispatches to Strategy 1 / Strategy 2 via two private functions `_select_prob_ascent(...)` / `_select_class_divergence(...)`.
   - `history`: per-step list of `(feature_idx, value, p_target_after, selection_score)` for diagnostics/report.
   - Exclude immutable columns from `R`; assert at the end that **all non-actionable columns are byte-identical** to the factual (Stage-7 immutability assert, extended).
   - `budget` defaults to `len(actionable_idx)`. On exhaustion without flip, return `x_cf` with a flag `flipped=False`.
   - **Context is pre-fitted by the caller** — `greedy_counterfactual` requires a sampler with `set_context` already called and is **agnostic to fit granularity** (per-class batch in exp4, per-point in exp6's kNN cells). It does not fit context itself. The caller uses target_only context for Strategy 1, all_classes for Strategy 2.
   - **Runtime**: Strategy 1 issues `O(|R|)` single-column imputes per step → `O(|A|²)` per point (HELOC ≈ up to 289). Use a **low `n_permutations` (1–3)** for the greedy inner loop and keep `--max-test` bounded; `log()` a per-point timing estimate so an unattended run is observably progressing, not hung.

3. **Add the experiment runner.**
   - New file: `experiments/zeroshot_cf/exp4_greedy_cf.py` (mirror `exp2_counterfactuals.py` structure: load data, train discriminator, batch, generate, metrics, write artefacts). **Artefact path convention**: reuse exp2's `RESULTS_DIR = Path(__file__).parent / "results"` — i.e. all `results/...` paths in this plan resolve to `experiments/zeroshot_cf/results/`, **not** a repo-root `results/`. The existing `REPORT.md` to extend is `experiments/zeroshot_cf/results/REPORT.md`.
   - argparse flags: `--dataset {moons,heloc,all}`, `--selector {prob_ascent,class_divergence}` (default `prob_ascent`), `--tau` (0.5), `--budget` (default `|A|`), `--max-test`, `--n-permutations`, `--max-context` (256 baseline).
   - For Strategy 1 set context `target_only`; for Strategy 2 set context `all_classes` (and reject/skip target-only with a clear log line).
   - Per test point: `y_target = 1 − disc.predict(x)`; run `greedy_counterfactual`; assemble `X_cf`.
   - Metrics: reuse `compute_metrics` for `validity, lof_scores_cf, sparsity, true_actionability, proximity_l2_jaccard`. **`compute_metrics` does NOT return `frac_oob`** — compute it inline on the **unclipped** `X_cf` exactly as `exp2_counterfactuals.py:264–267` does: `frac_oob = (((X_cf < 0.0) | (X_cf > 1.0)).any(axis=1)).mean()`. **Also add** new greedy-specific keys: `l0_count_mean`/`l0_count_median`/`l0_count_max` (integer count of features changed per CF; keep this distinct from the existing fractional `sparsity`), `steps_mean`/`steps_median`/`steps_max` (steps to flip), and `failure_rate` (fraction that hit budget without flipping).
   - Writes `results/exp4_greedy_{moons,heloc}_metrics.csv` (one row, all metrics incl. the new ones) and `results/exp4_examples.md` (factual vs CF in **original** feature space, with the **ordered list of changed features** per example — the recourse path).

4. **Run the baseline greedy generation (offline, v2 checkpoints).**
   - See `resources/commands.md`. Smoke-test MOONS first (≤2 steps), then HELOC with a bounded `--max-test`.

5. **Tests.**
   - File: `experiments/zeroshot_cf/tests/test_greedy.py`. ⚠️ The `models` fixture is **not** in `tests/conftest.py` today — `conftest.py` only does `sys.path` setup; the fixture is duplicated in `test_sampler.py:30` and `test_ordering.py:26` (`@pytest.fixture(scope="module")` → `get_models(n_estimators=2)`, real v2 checkpoints). **As the first step of this Stage's tests, lift that `models` fixture into `tests/conftest.py`** (remove the two duplicates) so `test_greedy.py` (and the Stage-3 `test_context.py`) can share it; verify the prior 13 tests still pass after the move. Reuse the deterministic `_make_synthetic()` helper pattern from `test_sampler.py:36`/`test_ordering.py:32` (there is **no** `FAST_TEST_MODE` env var — do not invent one).
   - (a) loop terminates and returns `x_cf` with `disc.predict == y_target` on a small synthetic case where a flip is reachable within budget; (b) `changed ⊆ actionable_idx` and all non-actionable columns byte-identical (immutability assert holds); (c) per-CF L0 count `= |changed|` and `≤ |A|`; (d) `prob_ascent` picks the candidate maximizing `disc.predict_proba[y_target]` on a constructed 2-feature case; (e) `class_divergence` picks the higher mean-shift feature on a constructed case; (f) budget exhaustion returns `flipped=False` and the point is counted invalid; (g) **`predictive_distribution` correctness** (see Step 1b, compare like-for-like, both conditioned on `Y=t`): `mean_of_prediction(predictive_distribution(X, j, fixed_target=t))` ≈ empirical mean of `sample_feature(X, target_col=j, fixed_target=t, sample_temperature=1.0, n_samples=200)` within a stated tolerance — **not** a single near-MAP sample (mode ≠ mean for skewed distributions). (h) **`sample_feature` class-conditional pass-through**: `sample_feature(X, target_col=j, fixed_target=t, ...)` with `append_target=True` runs without raising and returns class-conditioned values; the existing `append_target=False` call (no `fixed_target`) still works.
   - All prior predecessor-plan tests (13/13 across `test_sampler.py`, `test_ordering.py`, `test_metrics_harness.py`) must still pass.

---

## Verification

- [ ] `uv run python experiments/zeroshot_cf/exp4_greedy_cf.py --help` lists `--selector {prob_ascent,class_divergence}`, `--tau`, `--budget`.
- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` passes, including `test_greedy.py` (incl. the `predictive_distribution` correctness test (g) and the `sample_feature` class-conditional pass-through test (h)); prior 13 tests still pass.
- [ ] The greedy commit runs class-conditionally with no `ValueError`: `sample_feature(..., fixed_target=y_target, ...)` under `append_target=True` succeeds end-to-end on a smoke run.
- [ ] `results/exp4_greedy_{moons,heloc}_metrics.csv` exist with `l0_count_mean`, `steps_mean`, `failure_rate`, and `frac_oob` columns.
- [ ] `true_actionability == 1.0` for every run.
- [ ] MOONS mean steps-to-flip ≤ 2 and validity ≈ 1.0.
- [ ] Greedy `l0_count_mean` is **strictly lower** than the one-pass `|A|` for valid CFs; record the delta.
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` is empty; `grep -rn "tabpfn_client" experiments/zeroshot_cf` finds nothing.

---

## Commit

`feat(greedy-cf): iterative greedy CF core + prob-ascent & class-divergence selectors (Exp4)`
