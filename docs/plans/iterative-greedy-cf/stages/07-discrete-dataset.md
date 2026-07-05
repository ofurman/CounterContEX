# Stage 7: Add a Discrete/Categorical Dataset + Validity Sanity Check

**Goal**: Wire a genuinely **discrete (categorical)** dataset into the pipeline and confirm the meeting's prediction that greedy + TabPFN should achieve **near-100% validity** there. All current datasets (MOONS, HELOC) are continuous; this stage closes that gap and stress-tests the classifier-head path that HELOC's int-collapsed columns only partially exercise.
**Dependencies**: Stage 1 DONE (Exp4 runner + metrics). Independent of Stages 5/6/8. Benefits from Stage 5's revisit loop but does not require it.

---

## Motivation (from the meeting)

> "warto pewnie dorzucić jakieś dyskretne dataset … na takim [dyskretnym] to powinno to podejście działać, także przynajmniej validity będzie 100%."

TabPFN models a categorical feature with a softmax over its categories (the classifier head), so a class-conditional commit lands on an actual in-support category — there is no extrapolation off a continuous bar distribution, so validity *should* be easy. Demonstrating this isolates the continuous-binning effects (Stage 8) from the method's core soundness.

---

## Design

Use **TabPFN's native categorical-feature handling**, not one-hot encoding.

The previous sketch suggested reusing CEL configs with `one_hot_encode`. That is the wrong
representation for this stage: the greedy loop changes one column at a time, so a one-hot
category group can become invalid if only one dummy column is changed. Stage 7 must keep each
categorical variable as **one semantic feature column** and pass its column index to TabPFN as
categorical.

Representation:
- Categorical columns are encoded as stable integer/category codes (`0..K-1`) for sklearn
  discriminator compatibility and for numpy arrays. They are **not** min-max scaled and **not**
  one-hot encoded.
- Continuous columns, if any, keep the existing MinMax-[0,1] scaling.
- `DatasetBundle.categorical_features_indices` and `numerical_features_indices` are authoritative.
  The sampler receives these explicit indices; it must not rely on TabPFN's auto low-cardinality
  inference to decide semantic feature types.
- When `append_target=True`, the appended `Y` column is also categorical, so the sampler passes
  `categorical_features_indices + [last_idx]` to `set_categorical_features(...)`.

**Dataset choice** (resolve during execution, record under Decisions): prefer a dataset that is
categorical-dominant and can be represented without one-hot encoding. Existing CEL candidates
(`adult_census`, `german_credit`, `bank_marketing`) are acceptable only if their YAML is copied or
adapted into a **native-categorical** variant whose `initial_transforms` omit `one_hot_encode`.
If that leaves a mixed continuous/categorical dataset, state that explicitly in the report. If the
meeting's "genuinely discrete" premise is required, construct a small synthetic all-categorical
dataset under `experiments/zeroshot_cf/vendor/counterfactuals/`.

---

## Steps

1. **Select & wire the dataset in native-categorical form.**
   - If adapting an existing CEL config, create a separate dataset name such as
     `<name>_native_cat` under
     `experiments/zeroshot_cf/vendor/counterfactuals/config/datasets/`. The YAML must declare
     `categorical_features` but must **not** include `one_hot_encode` in `initial_transforms`.
   - Add or reuse a deterministic categorical-code transform for categorical columns before
     `MethodDataset` exposes `X_train`/`X_test`. If no existing CEL transform fits, add a small
     Stage-7-local helper in `experiments/zeroshot_cf/data.py` (or a CEL initial transform) that
     maps each categorical column to stable `0..K-1` codes fit on train/raw data. Preserve the
     inverse mapping for examples/reporting if practical.
   - Keep `MinMaxScalingStep` applied only to continuous columns. Do not scale categorical codes.
   - Add an actionability config
     `experiments/zeroshot_cf/configs/<name>_actionability.yaml` (HELOC pattern:
     `experiments/zeroshot_cf/configs/heloc_actionability.yaml`) so
     `get_actionable_immutable` routes correctly: immutables = protected/non-actionable
     attributes.
   - Extend `get_actionable_immutable` to load `<dataset>_actionability.yaml` generically instead
     of hard-coding only `moons` and `heloc`.
   - Extend Exp4's argparse dataset handling (`--dataset`) and `_DATASET_PARAMS` so
     `--dataset <name>` works; today the CLI only accepts `moons`, `heloc`, `all`.

2. **Thread explicit categorical/numerical indices into TabPFN.**
   - Extend `ConditionalDensitySampler` with explicit modality inputs, e.g.
     `categorical_features_indices: Optional[list[int]] = None` and
     `force_numeric_cols: Optional[list[int]] = None`.
   - In `set_context`, compute the categorical set as:
     `dataset_categorical_indices - force_numeric_cols`, plus the appended `Y` index when
     `append_target=True`; pass that full list to `self.model.set_categorical_features(...)`.
   - Do not depend on TabPFN's automatic low-cardinality inference for this stage. If the current
     TabPFN/unsupervised wrapper still auto-routes low-cardinality numeric columns despite explicit
     categorical indices, configure the TabPFN models with an inference setting such as
     `MIN_UNIQUE_FOR_NUMERICAL_FEATURES=0` for the Stage-7/8 runs, then pass the semantic
     categorical indices explicitly. Record the exact choice under Decisions.
   - Ensure all sampler construction sites used by Exp4/Exp6/Exp7 can pass
     `bundle.categorical_features_indices` while preserving existing MOONS/HELOC defaults.

3. **Confirm classifier-head routing for semantic categoricals.**
   - Pick one known categorical column from the new dataset and assert
     `predictive_distribution(...)` returns `{"proba", "classes"}` for that column.
   - Pick one known continuous column, if the dataset has one, and assert it returns
     `{"logits", "criterion"}`.
   - The committed categorical value must be one of the observed category codes. There should be no
     one-hot-group validity check because there are no one-hot groups in this stage.

4. **Run Exp4** with `prob_ascent` on the new dataset (and, if Stage 5 is done, with a budget allowing revisits). Bound `--max-test` as needed.

5. **Record results** in `results/exp4_greedy_<name>_metrics.csv` (the runner writes per-dataset CSV after Step 1 extends it) and a short note in `results/REPORT.md`: did validity reach ≈1.0? What are `l0_count`, `proximity`, `frac_oob` (the latter should be ~0 for continuous range violations; categorical support validity is checked separately)?

6. **Tests.** Add `tests/test_discrete_dataset.py` (shared `models` fixture):
   - `load_dataset("<name>")` returns non-empty `categorical_features_indices`, no one-hot-expanded
     feature names for the chosen categorical columns, and categorical columns have integer-coded
     values with observed support `0..K-1`.
   - `get_actionable_immutable("<name>")` works through the generic actionability config.
   - A sampler routing smoke asserts a categorical column returns `{"proba","classes"}` and a
     continuous column (if present) returns `{"logits","criterion"}`.
   - Exp4 produces a CSV row with `validity` defined on a tiny `--max-test`. Do not gate on an
     exact validity number in the unit test (that lives in the report).

---

## Verification

- [ ] `load_dataset("<name>")` loads the native-categorical dataset with non-empty
      `categorical_features_indices`, no one-hot categorical expansion, and integer-coded
      categorical values.
- [ ] `get_actionable_immutable("<name>")` works through a generic actionability config;
      immutables correctly split.
- [ ] `ConditionalDensitySampler` receives explicit categorical indices and marks
      `categorical_features_indices + [Y]` categorical; a known categorical column routes to
      `{"proba","classes"}` and a known continuous column routes to `{"logits","criterion"}`.
- [ ] Exp4 on the dataset writes `results/exp4_greedy_<name>_metrics.csv` with validity, `l0_count`, proximity, `frac_oob`, `true_actionability`.
- [ ] `results/REPORT.md` reports the validity result and states whether the ≈100% prediction held;
      categorical commits are in observed support; `frac_oob ≈ 0` for continuous range violations.
- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` passes (incl. the new smoke test).
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` empty; no `tabpfn_client` import.

---

## Expected outcomes

- Validity **≈1.0** on a genuinely categorical dataset (the meeting's prediction); `frac_oob ≈ 0`. If validity is materially below 1.0, that is itself an interesting finding about the classifier-head commit and feeds Stage 8.
- A third dataset in the headline table (Stage 9) covering the discrete regime, complementing MOONS (continuous, simple) and HELOC (continuous, hard).

## Commit

`feat(greedy-cf): wire discrete/categorical dataset + validity sanity check`
