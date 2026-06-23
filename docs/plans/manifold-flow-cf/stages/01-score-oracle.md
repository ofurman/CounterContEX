# Stage 1: Class-conditional score oracle

**Goal**: Build and validate a usable class-conditional score field `s_t(x) = ∇_x log p(x | Y=t)` from TabPFN's per-column bar distribution (continuous columns) plus a Gibbs proposal (discrete columns), gated on accuracy vs a numerical ground truth.
**Dependencies**: None (first stage; gates all others).

---

## Background (read `resources/math.md` §1–2 first)

The j-th component of the joint score equals the conditional log-density derivative:
`[∇_x log p(x|t)]_j = ∂/∂x_j log p(x_j | x_{−j}, Y=t)`. TabPFN returns that conditional per column. **But the bar density is piecewise-constant per bucket**, so the naive analytic derivative is unusable. This stage implements and *validates* a robust estimator before any flow exists.

---

## Steps

1. Create the score module.
   - File: `experiments/zeroshot_cf/score.py`
   - Implement `conditional_score(sampler, x, y_target, actionable_idx, *, method="mean_shift", eps=1e-2) -> dict`:
     - `x`: shape `(d,)`. For each **continuous** actionable column `j`, call `sampler.predictive_distribution(x[None, :], target_col=j, fixed_target=y_target)`.
     - **`method="mean_shift"` (primary, Decision #3):** score component `s_j = (μ_j − x_j)` where `μ_j = mean_of_prediction(dist["logits"], dist["criterion"])` (the bar-distribution conditional mean). This is the Gaussian-conditional score up to scale and avoids the piecewise problem.
     - **`method="findiff"` (fallback):** `s_j = (logp(x_j+eps) − logp(x_j−eps)) / (2·eps)` using the bar distribution's log-prob at shifted query values. Evaluate log-prob via `dist["criterion"]` (use `.cdf` differences or `compute_scaled_log_probs` mapped to the bucket of the shifted value); clamp shifted values to `[0,1]`.
     - Return `{"score": np.ndarray(d,) with non-actionable & discrete entries = 0, "mu": μ vector, "method": method, "discrete_cols": [...]}`.
   - Implement `gibbs_proposal(sampler, x, y_target, col) -> float`: for a **classifier-routed** column, draw from `p(x_j|x_{−j},t)` via `sampler.predictive_distribution(...)["proba"]/["classes"]` (sample a class, map back to its value) — reuses the routing logic from `class_conditional_shift`. Used by the flow's discrete jumps in Stage 2.
   - Add `is_classifier_column(sampler, x, col, y_target) -> bool` (probe the dist dict shape: `"proba"` present ⟹ classifier-routed). Cache per column.

2. Build a numerical ground-truth score for validation (MOONS only, 2-D ⟹ tractable).
   - File: `experiments/zeroshot_cf/tests/test_score.py`
   - Fit a class-conditional KDE on `X_train[y_train==t]` (use a Gaussian KDE; scipy is available in the env). Define `kde_score(x) = ∇ log kde(x)` by central finite differences on the KDE log-density.
   - On a grid of ~50 in-distribution MOONS test points, compare `conditional_score(..., method="mean_shift")` and `method="findiff"` against `kde_score` by **mean cosine similarity** of the (2-D) score vectors.

3. Select the estimator.
   - Record which method wins the cosine test in the stage notes + index Decisions. `mean_shift` is the expected primary; if `findiff` is clearly better (cosine higher by >0.05) make it the default and note it.

---

## Verification

- [ ] `pytest experiments/zeroshot_cf/tests/test_score.py -q` passes (run via the provisioned env — see `resources/commands.md`).
- [ ] **Gate**: chosen estimator's mean cosine similarity vs KDE-score on MOONS ≥ **0.9** (Success Criteria). If neither method clears 0.9, this is a heavy problem → Backlog with the measured cosines and a note to try `method="smoothed"` (estimator (a)); do **not** proceed to Stage 2 with an unvalidated score.
- [ ] `conditional_score` returns 0 for immutable and discrete columns (asserted in a unit test).
- [ ] `gibbs_proposal` returns an in-`classes` value for a HELOC classifier-routed column (unit test with a 1-row HELOC fixture; reuse the `models` fixture from `conftest.py`).
- [ ] Score computation is deterministic for fixed input + seed (mean_shift is deterministic; findiff with fixed eps is too).

---

## Commit

`feat(manifold-flow): class-conditional score oracle from bar distribution (Stage 1)`
