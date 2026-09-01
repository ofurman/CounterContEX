# Stage 2: Target-model registry

**Goal**: Let the benchmark explain three classifier families — logistic regression, MLP, and
XGBoost — through one registry, so the model-agnosticism contribution has evidence.
**Dependencies**: Stage 1

---

## Steps

1. **Replace the hard-coded target-model check.**
   - Where: `_default_case_loader()` in `experiments/zeroshot_cf/orchestration/runner.py`, which
     currently rejects anything but `_SUPPORTED_TARGET_MODEL_NAME` with exactly
     `{"C": 1.0, "max_iter": 1000, "seed": 42}`.
   - Introduce a target-model registry in the layer that owns model construction — `datasets/`,
     not `orchestration/`. The generic runner must look a name up, never branch on a family.
     This mirrors the existing `methods/registry.py` pattern; follow it rather than inventing a
     second registry shape.
   - Registered names: `retained_logistic_regression` (unchanged defaults, so existing specs keep
     their identity), `retained_mlp`, `retained_xgboost`.

2. **Add the XGBoost arm.**
   - Where: `train_discriminator()` in `experiments/zeroshot_cf/discriminator.py`, which already
     has `disc_type` of `"lr"` and `"mlp"` with `DEFAULT_LR_PARAMS` and `DEFAULT_MLP_PARAMS`.
     Add `"xgb"` in the same shape.
   - Add the dependency to `experiments/zeroshot_cf/pyproject.toml` and re-lock with
     `uv sync --locked`. Keep the import lazy, as `dice-ml` is, so a missing optional dependency
     fails at method preparation rather than at import.
   - Fix all hyperparameters and the seed. A tuned-per-dataset classifier would make the three
     arms incomparable.

3. **Preserve identity semantics.**
   - Where: `_model_identity()` and `_implementation_digest()` in
     `experiments/zeroshot_cf/datasets/benchmark.py`
   - `model_content_id` must differ across the three families and be stable within one. Confirm
     the digest actually reaches into the fitted estimator — a digest computed only from the
     spec's declared params would give two differently-fitted models the same identity.
   - The classifier cache key in `_default_case_loader` (`cache_tag`) currently encodes dataset
     and preprocessing only. It **must** also encode the model family, or the MLP run will load
     the logistic-regression pickle. This is the single most likely silent defect in this stage.

4. **Measure baseline compatibility — do not presume any method fails.** The repository's
   Wachter (`wachter_coordinate_counterfactual` in `methods/optimization.py`) is a black-box
   coordinate minimization over `predict_proba`; it needs no gradients and is expected to run
   against all three families. Run each baseline once against each family and record in
   `decisions.md` which combinations run and which fail. A method that cannot support a family
   must fail cleanly during method preparation with a clear message — never emit candidates
   whose validity was not checked against the actual target model. Only combinations recorded
   here may later be treated as expected clean failures by the campaign completeness gate.

5. **Extend matrix expansion with a target-model axis.**
   - Where: `load_matrix_config()` in `experiments/zeroshot_cf/orchestration/matrix.py`, which
     currently parses `target_model` as a **single mapping** and expands only
     `datasets × methods × seeds`. E1 needs one matrix crossing three classifier families, so
     this axis must exist before Stage 6 writes the campaign matrices.
   - Accept a `target_models:` list (keep the single `target_model` mapping valid so existing
     matrices resolve unchanged) and include it in the Cartesian product. This is a matrix
     schema change: decide whether `countercontex.matrix.v1` is bumped or extended
     compatibly, and record that decision in `decisions.md`.
   - Add a contract test: a matrix crossing two target-model names resolves to distinct cell
     identities that differ only in the target-model fields.

---

## Verification

- [ ] GATE Three benchmark cases built on the same dataset with the three families produce three
      distinct `model_content_id` values and three distinct cached classifier files — read from
      the constructed cases and the model cache directory. A `cache_tag` that omits the family
      turns it red by returning the same fitted model twice.
- [ ] GATE Wachter against an XGBoost target either produces valid counterfactuals or raises
      during method preparation — never returns candidates whose validity was not checked against
      the actual target model. Read from the generation result, not from the exit code.
- [ ] GATE A matrix crossing two target-model names dry-runs to cells whose resolved
      identities differ only in the target-model fields — read from the resolved dry-run
      output. An expansion that still parses `target_model` as a single mapping turns it red.
- [ ] REPORT Test accuracy of each family on each of the six datasets — record in `journal.md`.
- [ ] REPORT The measured baseline-compatibility table (method × family: runs / fails at
      preparation) — record in `decisions.md`.

---

## Commit

`feat(experiments): add target-model registry, XGBoost arm, and matrix target-model axis`
