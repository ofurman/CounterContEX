# Stage 3: Extract Evaluation and Artifact Ownership

**Goal**: Make a validated canonical generation result the only input to a method-independent evaluator and versioned artifact store.
**Dependencies**: Stage 2
**References**: [`architecture.md`](../resources/architecture.md#canonical-generation-result), [`architecture.md`](../resources/architecture.md#evaluation-contract), [`architecture.md`](../resources/architecture.md#artifact-contract)

---

## Steps

1. Add the canonical result and evaluation types.
   - Where: `GenerationResult` in `core/contracts.py`; new `evaluation/result.py` and `evaluation/evaluator.py`.
   - Normalize all candidates to `(n, k, d)` plus an explicit availability mask. Reject fabricated padding and keep best-effort arrays namespaced outside common candidates.
   - Define typed summary, point, candidate, and array outputs with a metric schema version.

2. Move common metric composition behind the evaluator.
   - Where: new `evaluation/metrics.py`; `compute_dicoflex_common_metrics()` in `metrics_harness.py` and set helpers in `reporting.py` become compatibility delegates used by existing production runners.
   - Derive common fields only from `BenchmarkCase`, canonical candidates, availability, oracle predictions, and `EvaluationSpec`; never read method diagnostics.
   - Report coverage, class validity, threshold validity, and valid-success rates separately. Preserve valid-only proximity, grouped Gower, action units, primary metrics, and set diversity.
   - Prepare LOF/Isolation Forest once per case for reuse across method cells.

3. Introduce manifest-backed artifact persistence.
   - Where: new `orchestration/artifacts.py` and `orchestration/legacy.py`; current `write_dataset_outputs()` and aggregation helpers delegate to compatibility export.
   - Write manifest, summary, points/candidates, NPZ, and `COMPLETE` atomically under a run ID. Validate schema versions on read and ignore incomplete directories during aggregation.
   - Keep current filenames, columns, and NPZ keys through the v1 exporter.

4. Exercise the new evaluator and writer on a real current runner path.
   - Where: route one retained runner's common metric and output assembly through compatibility adapters backed by the new implementation.
   - This prevents the new layer from existing only in tests before method migration.

---

## Verification

- [ ] GATE evaluator semantic tests use named synthetic candidates and oracle probabilities — confusing unavailable, invalid, threshold-failing, primary, or set members turns expected fields red.
- [ ] GATE artifact round-trip tests read files written to a temporary directory — schema drift, non-atomic completion, config loss, type loss, or aggregation of a partial run turns them red.
- [ ] GATE `uv run pytest -q` — existing runners still pass through compatibility delegates and the full retained suite remains green.

---

## Commit

`refactor(evaluation): centralize metrics and artifact contracts`
