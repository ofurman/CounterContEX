# Stage 1: Freeze the Retained Suite and Benchmark Contract

**Goal**: Establish reproducible evidence for the exact generator, diversity, benchmark, baseline, and four-dataset behavior that later cleanup must preserve.
**Dependencies**: None

---

## Steps

1. Pin the current CEL source before measuring dataset behavior.
   - Update `vendor_setup.py` and the CEL line in `requirements.txt` to revision
     `3587f943826f6b087a0d198c8c4aa4373712c7ee`.
   - Make setup verify the checked-out revision and required four YAML/CSV pairs. It must fail
     clearly on a mismatched checkout rather than silently using current HEAD.
2. Add a reviewed benchmark contract fixture generated from that pinned source.
   - Record per dataset: raw/clean row counts, split sizes and label counts, feature names/order,
     numerical/categorical indices, one-hot groups, actionable/immutable indices, scaler bounds,
     and hashes of the final train/validation/test arrays.
   - Record the generator/benchmark defaults that affect comparability: seeds, discriminator
     parameters, factual cap, candidate/confidence quantiles, context policy, revisit/refinement
     budgets, number of CFs, and diversity configuration.
   - The fixture must state its input CEL commit and file paths. Do not copy values from a new
     replacement loader or use placeholders.
3. Add contract tests that read the pinned dataset inputs and compare fresh loader outputs with the
   fixture. Include HELOC all-`-9` filtering before split/scaling and the six immutable fields.
4. Define a retained test manifest covering conditional sampling, greedy mixed actions, quantiles,
   confidence, revisits, refinement, diversity, Exp9, Exp11–14, data cleaning, metrics, distance,
   and checkpoint validation. Remove or relocate the old global real-TabPFN fixture only if it
   prevents this retained suite from collecting; do not delete legacy runtime yet.
5. Add missing focused tests for checkpoint absence/checksum mismatch and public defaults. Treat
   real TabICL inference as a REPORT because the two weights are not present at planning time.

---

## Verification

- [ ] GATE `uv run python -m experiments.zeroshot_cf.vendor_setup --revision 3587f943826f6b087a0d198c8c4aa4373712c7ee --check` reports that exact revision and all four config/data pairs — the checked-out Git metadata and files are the inputs; an unpinned, mismatched, or incomplete vendor tree turns it red.
- [ ] GATE `uv run --python 3.12 --with-requirements experiments/zeroshot_cf/requirements.txt pytest -q experiments/zeroshot_cf/tests/test_data_cleaning.py experiments/zeroshot_cf/tests/test_diverse_search.py experiments/zeroshot_cf/tests/test_exp9_benchmark.py experiments/zeroshot_cf/tests/test_exp11_nice_nun_baseline.py experiments/zeroshot_cf/tests/test_exp12_optimization_baselines.py experiments/zeroshot_cf/tests/test_exp13_dice_baseline.py experiments/zeroshot_cf/tests/test_exp14_face_baseline.py experiments/zeroshot_cf/tests/test_grouped_categorical.py experiments/zeroshot_cf/tests/test_metrics_harness.py experiments/zeroshot_cf/tests/test_mixed_distance.py experiments/zeroshot_cf/tests/test_tabicl_backend.py experiments/zeroshot_cf/tests/test_tabicl_plausibility.py experiments/zeroshot_cf/tests/test_tabicl_checkpoints.py` passes with non-zero collection — the retained production modules and tests are the inputs; a missing desired capability or eager legacy import turns it red.
- [ ] GATE `uv run --python 3.12 --with-requirements experiments/zeroshot_cf/requirements.txt pytest -q experiments/zeroshot_cf/tests/test_dataset_contract.py` recomputes every recorded field and hash from the pinned CEL files and passes — the four source datasets/configs are the inputs; split, cleaning, scaling, feature-order, label, or actionability drift turns it red.
- [ ] REPORT Record focused test count/duration and checkpoint paths in `journal.md`; absent weights are `NOT MEASURED`, not PASS.

---

## Commit

`test(tabicl-cf): freeze retained benchmark contract`
