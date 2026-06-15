# Stage 3: Conditional Density Sampler Wrapper

**Goal**: Build `ConditionalDensitySampler`, a reusable wrapper around `TabPFNUnsupervisedModel` that handles context selection, the Y-as-appended-column trick, and masked imputation — the shared engine for both experiments.
**Dependencies**: Stage 1 (models), Stage 2 (data loading)

---

## Steps

1. **Core class.**
   - File: `experiments/zeroshot_cf/sampler.py`
   - `ConditionalDensitySampler(clf, reg, append_target=False, n_permutations=10, temperature=1e-9, random_state=0)`.
   - Internally builds `TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)` from the staged models (Stage 1).

2. **Context selection.**
   - Method `set_context(X_context, y_context=None, target_class=None, max_context=None)`.
   - If `target_class` is given, filter `X_context` to rows of that class (class-conditional context — the brief's "target-class samples as context").
   - If `append_target=True`, append `y` (or a constant `target_class` column) as the **last column** and register it categorical via `model.set_categorical_features([last_idx])` **before** `fit`. This is how class conditioning is injected (no native Y-conditioning — see `resources/api-reference.md`).
   - Subsample to `max_context` rows if set (TabPFN context-size knob; HELOC train is large). Subsample deterministically with `random_state`. This is a key refinement lever in Stage 6.
   - Call `model.fit(X_context_aug)` (stores the whole matrix as the conditioning set).

3. **Masked imputation / sampling.**
   - Method `impute_masked(X_query, mask_cols, fixed_target=None) -> X_filled`.
     - Copy `X_query`; set `X_query[:, mask_cols] = np.nan`.
     - If `append_target=True`, append the target column set to `fixed_target` (observed, **not** NaN) so generation is conditioned on `Y=target`.
     - Call `model.impute(X_query_aug, t=self.temperature, n_permutations=self.n_permutations)`.
     - Drop the appended target column from the result; return only the original feature columns with masked cells now filled.
   - Method `sample_feature(X_query, target_col, n_samples=1)` for Experiment 1 single-feature use: mask exactly `target_col`, return the filled value(s). Support drawing multiple samples (loop or higher temperature) to inspect the conditional distribution, not just the MAP.

4. **Determinism & device hygiene.**
   - Seed numpy/torch from `random_state`. Convert numpy↔torch carefully (the unsupervised model works in `torch.float32`; outputs are tensors — return numpy for downstream cel metrics).
   - Respect `device` from the staged models. Log wall-clock per impute call (the inner loop re-fits TabPFN per column/permutation — expensive; informs Stage 6 budget).

5. **Unit test.**
   - File: `experiments/zeroshot_cf/tests/test_sampler.py`
   - On a small synthetic 3-feature dataset with a known conditional relationship (e.g. `x2 ≈ x0 + x1`), confirm `sample_feature` reconstructs `x2` with error well below the marginal-mean baseline. Confirm `impute_masked` preserves non-masked (incl. immutable) columns exactly.

---

## Verification

- [ ] `test_sampler.py` passes: reconstructed feature beats marginal-mean baseline; non-masked columns are byte-identical to input.
- [ ] `impute_masked` with `append_target=True` produces an output with the **original** feature count (target column stripped) and NaN-free masked cells.
- [ ] Switching `target_class` visibly shifts the sampled values (conditioning has an effect) on a 2-class toy set.

---

## Commit

`feat(zeroshot-cf): ConditionalDensitySampler over TabPFNUnsupervisedModel`
