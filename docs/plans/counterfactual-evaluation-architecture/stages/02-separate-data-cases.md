# Stage 2: Separate Portable Data and Benchmark Cases

**Goal**: Replace the mixed `BenchmarkDatasetContext` boundary with immutable provider-neutral dataset, schema, provenance, method-context, and benchmark-case contracts.
**Dependencies**: Stage 1
**References**: [`architecture.md`](../resources/architecture.md#core-data-contracts), [`current-state.md`](../resources/current-state.md#behavior-that-must-remain-stable)

---

## Steps

1. Introduce portable core contracts and validators.
   - Where: new `core/contracts.py` and `core/validation.py`.
   - Add `FeatureSchema`, `FeatureDomains`, `DatasetProvenance`, `PreparedDataset`, `FactualSelection`, `Predictor`, `BenchmarkCase`, `MethodContext`, and `GenerationRequest`.
   - Keep numerical/categorical type, scalar/group actionability, and immutability as separate concepts. Validate disjointness, indices, shapes, one-hot groups, read-only shared arrays, and classifier `classes_` mapping.

2. Isolate CEL behind a dataset-provider adapter.
   - Where: new `datasets/base.py` and `datasets/cel.py`; current `DatasetBundle` and `load_dataset()` in `data.py` become compatibility delegates.
   - Preserve current CEL revision, HELOC cleaning, source hashes, split, train-only transform, feature ordering, grouped categories, actionability, and inverse-transform behavior.
   - Do not expose CEL `MethodDataset` through `PreparedDataset`; give DiCE or export code focused adapters when native dataframes are required.

3. Build reusable benchmark cases from prepared data.
   - Where: new `datasets/benchmark.py`; adapt `prepare_benchmark_context()` in `benchmark_protocol.py` to delegate.
   - Centralize factual selection once, store stable source indices, train/load the oracle, derive targets from factual predictions, and compute `case_id` from real dataset/protocol/model inputs.
   - Make probability lookup resolve target labels through `Predictor.classes_`.

4. Keep old imports working while migrating consumers.
   - Where: compatibility aliases in `data.py` and `benchmark_protocol.py` plus deprecation comments in source documentation.
   - Do not move algorithms or change result artifacts in this stage.

---

## Verification

- [ ] GATE `uv run pytest -q experiments/zeroshot_cf/tests/test_data_cleaning.py experiments/zeroshot_cf/tests/test_dataset_contract.py experiments/zeroshot_cf/tests/test_exp9_benchmark.py` — tests read pinned source/config fixtures; any data, split, schema, or target drift turns them red.
- [ ] GATE new core/dataset contract tests — synthetic arrays and frozen dataset fingerprints are the inputs; mutable shared arrays, CEL leakage, invalid feature partitions, duplicate factual selection, or wrong class-column lookup turns them red.
- [ ] GATE `uv run pytest -q` — the full retained suite passes with compatibility imports active.

---

## Commit

`refactor(data): introduce portable benchmark case contracts`
