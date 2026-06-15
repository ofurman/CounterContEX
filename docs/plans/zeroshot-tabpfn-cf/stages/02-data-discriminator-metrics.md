# Stage 2: Data, Discriminator & Metrics Harness

**Goal**: Load HELOC & MOONS via `cel`, define the HELOC actionable/immutable split, train the cel `disc_model` (LR/MLP) as the validity oracle, and wire the 5-metric evaluation subset through `cel.evaluate_cf` — all reusable by both experiments.
**Dependencies**: Stage 1 (env + deps)

---

## Steps

1. **Dataset loading.**
   - File: `experiments/zeroshot_cf/data.py`
   - Use `cel.datasets.FileDataset(config_path=...)` + `MethodDataset(file_dataset, preprocessing)` with the cel default preprocessing pipeline (`MinMaxScalingStep()` → `TorchDataTypeStep()`), so features are MinMax-scaled to [0,1] (fit on train) and splits match cel (80/20 stratified, `random_state=42`).
   - Config paths: HELOC `config/datasets/heloc.yaml`, MOONS `config/datasets/moons.yaml` (relative to the cel repo root — resolve via the installed/vendored cel package).
   - Expose a `load_dataset(name) -> obj` returning `X_train, X_test, y_train, y_test`, `numerical_features_indices`, `categorical_features_indices`, `feature_names`, and the fitted preprocessing (for `inverse_transform` back to original space when reporting).

2. **Define the HELOC actionable/immutable split.**
   - File: `experiments/zeroshot_cf/configs/heloc_actionability.yaml`
   - Immutable (frozen) features per Decision #2: `MSinceOldestTradeOpen`, `MSinceMostRecentTradeOpen`, `AverageMInFile`, `NumTotalTrades`, `MSinceMostRecentDelq`, `MSinceMostRecentInqexcl7days`. All other HELOC features are actionable.
   - MOONS: both features actionable, no immutables.
   - Provide `get_actionable_immutable(name) -> (actionable_idx, immutable_idx)` mapping names→column indices (in the **scaled feature matrix** column order — verify the order against `dataset.features`).
   - Document the split rationale in the file header; it is a judgment call.

3. **Train the validity-oracle discriminator (Decision #1).**
   - File: `experiments/zeroshot_cf/discriminator.py`
   - Use cel's standard classifier (`cel.models` logistic regression first; MLP as a config option). Train on `X_train, y_train` (scaled space). Wrap so it satisfies the cel metrics contract: `.predict(X_np) -> array` and `.eval()` (no-op if not torch).
   - Persist the trained model to `experiments/zeroshot_cf/models/disc_<dataset>_<type>.pkl` for reuse across runs (gitignored).
   - Sanity: report test accuracy; LR on MOONS should be ~0.85+, HELOC ~0.7+. If far off, check scaling/label mapping.

4. **Metrics harness.**
   - File: `experiments/zeroshot_cf/metrics_harness.py`
   - Wrap `cel.metrics.metrics.evaluate_cf(...)` configured to run our subset: `validity`, `lof_scores_cf`, `sparsity`, `actionability`, `proximity_l2_jaccard`. Either point it at a trimmed metrics yaml or call the registry metrics directly (whichever is cleaner — see `resources/api-reference.md`). Metrics whose `required_inputs` aren't satisfiable (e.g. density needing a `gen_model`) are skipped — that's fine; we pass `gen_model=None`.
   - Pass `continuous_features = numerical_features_indices`, `categorical_features = []`, `y_target` = desired class array.
   - **Add a `true_actionability` metric ourselves**: fraction of CFs whose immutable columns are exactly unchanged vs. factual. (cel's `actionability` metric is mislabeled — it measures unchanged-CF fraction, not constraint compliance. Report both, clearly named.)
   - Return a dict and a tidy one-row-per-run record for aggregation.

5. **Harness self-test with a trivial CF.**
   - Construct a degenerate CF set (e.g. `X_cf = X_test` copy with one actionable feature nudged) and confirm all 5 metrics + `true_actionability` compute and return sensible numbers.

---

## Verification

- [ ] `load_dataset("heloc")` returns 23 continuous features, target in {0,1}; `load_dataset("moons")` returns 2 features, 1000 rows.
- [ ] `get_actionable_immutable("heloc")` returns immutable indices matching the 6 named features (assert names map correctly to indices).
- [ ] Discriminator test accuracy printed and above sanity thresholds (MOONS ≥0.85, HELOC ≥0.7).
- [ ] `metrics_harness` on the degenerate CF returns all of `{validity, lof_scores_cf, sparsity, actionability, true_actionability, proximity_l2_jaccard}` as finite numbers, with `true_actionability == 1.0` when immutables are untouched.

---

## Commit

`feat(zeroshot-cf): cel dataset loading, actionability split, validity oracle, metrics harness`
