# Current State and Preserved Invariants

## Existing flow

```text
Exp9/11/12/13/14 CLI
  -> prepare_benchmark_context()
     -> CEL load, HELOC cleaning, deterministic split, train-only scaling
     -> factual selection, feature/action space, target classifier, target labels
  -> method-specific setup and generation
  -> compute_dicoflex_common_metrics()
  -> runner-specific metric/point/array assembly
  -> runner-specific CSV/NPZ write and aggregation
```

`run_dataset()` in every numbered runner repeats lifecycle, timing, validity/sparsity calculations,
row shaping, and artifact construction. There is no shared method result, manifest, completion
marker, generic resume behavior, or one runner for the six-method matrix.

## Reusable seams

- `OneHotActionGroup` in `action_space.py` and `ActionUnit` in `baseline_common.py` are lightweight
  structural types.
- `DatasetBundle` and `BenchmarkDatasetContext` centralize useful information but mix provider,
  protocol, oracle, factual selection, and evaluation concerns.
- `compute_dicoflex_common_metrics()` in `metrics_harness.py` and `mixed_distance.py` contain the
  reusable common metric kernels despite the historical method name.
- `TabICLGeneratorInputs`, `TabICLGeneratorConfig`, and `TabICLGeneratorResult` in `generator.py`
  are a strong method-level starting point.
- NICE, Wachter, Growing Spheres, DiCE, and FACE algorithm functions are mostly standalone and can
  be wrapped without algorithm rewrites.
- `_retained_contract.py` and dataset tests freeze source hashes, arrays, feature groups, split
  sizes, preprocessing, and actionability.

## Coupling to remove

- `BenchmarkDatasetContext` combines prepared data, protocol selection, target model, targets, and
  action-space metadata; `tabicl_runtime.py` imports it directly.
- Test-row selection is repeated in `data.py`, `generator.py`, and `benchmark_protocol.py`.
- Baselines expose heterogeneous point-function arguments and `(row, info: dict)` results.
- Exp8 and `tabicl_runtime.py` duplicate TabICL backend construction.
- Evaluation reaches into dataset loading for grouped action metadata and into method diagnostics
  for some success fields.
- Failure commonly returns the factual/best-effort row, so common coverage can remain 1.0.
- Class validity and probability-threshold search validity can disagree for thresholds above 0.5.
- Per-cell names, aggregate paths, CSV schemas, NPZ keys, and timing definitions vary by runner.
- Configuration is spread across constants, defaults, CLI arguments, and output rows, making
  ablation identity incomplete.
- Predictor probability indexing sometimes assumes labels are exactly `0/1` and align with
  probability columns instead of resolving through `classes_`.

## Behavior that must remain stable

- Dataset order: HELOC, Bank Marketing, Give Me Some Credit, Lending Club.
- Deterministic seed-42 64/16/20 split and seed-42 stratified factual selection.
- Training-only scaling and maximum 1,000 factuals.
- HELOC all-`-9` filtering before splitting.
- Logistic-regression target model defaults: `C=1`, `max_iter=1000`, seed 42.
- Target label is `1 - classifier.predict(factual)`, not ground-truth complement.
- Atomic one-hot interventions and exact immutable-column preservation.
- Common class validity is oracle prediction equal to target.
- Search success may additionally require target probability at or above `tau`; the new evaluator
  reports both rather than replacing one with the other.
- Valid-only proximity, grouped Gower, and action-unit sparsity semantics.
- Primary-CF metrics remain distinct from set coverage and diversity.
- Diverse output never pads missing candidates with invalid or duplicate rows.
- Aggregation and every CLI `--help` path work offline without loading checkpoints.
- Existing numbered commands and v1 artifacts remain available through compatibility adapters.
- `generate_counterfactual_batch()` remains a supported public API.

## Planning evidence

- `uv run pytest -q`: 87 passed, 5 warnings at the completed cleanup baseline.
- One-factual preflight: all 24 method/dataset cells executed.
- Full reference matrix: 24 rows with `n_test=1000`, no missing required metrics, 77 expected
  artifacts, 9.42 summed runtime hours.
- DiCoFlex/Lending Club alone took 27,519 generation seconds, so full-matrix execution is unsuitable
  as a repeated stage gate.
- The current Exp9 Athena launcher and submit helper use a six-hour walltime, which is shorter than
  the measured 7.64-hour DiCoFlex/Lending Club cell; the cutover must raise or parameterize it.
- Full-suite Ruff currently has pre-existing failures; lint checks in this plan target new package
  directories until a separate mechanical cleanup is performed.

The ignored full-reference directory is useful local comparison evidence but is not assumed to
exist in a fresh execution. Stage 1 creates small tracked semantic/compatibility fixtures from
reasoned cases and records any optional comparison to local full outputs as a REPORT.
