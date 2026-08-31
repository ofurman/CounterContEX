# Stage 8: Cut Over Entry Points and Run the Final Audit

**Goal**: Make the generic architecture the production path, retain only thin compatibility shims, and publish structural, parity, and full-matrix evidence.
**Dependencies**: Stages 6 and 7
**References**: [`architecture.md`](../resources/architecture.md), [`current-state.md`](../resources/current-state.md)

---

## Steps

1. Reduce legacy entry points to translation shims.
   - Where: Exp8/9/11–14 modules and Athena launchers.
   - Parsers translate legacy flags to concrete `RunSpec`s, invoke the generic runner/aggregator, and request the v1 compatibility exporter. Remove duplicated data preparation, common evaluation, row assembly, and direct persistence.
   - Update the Exp9 Athena submit path to pass a configurable Slurm walltime with a default of `10:00:00`; the current six-hour limit is below the measured 7.64-hour DiCoFlex/Lending Club cell. Record the selected limit in the manifest environment metadata.

2. Audit and remove obsolete helpers after import proof.
   - Where: `data.py`, `benchmark_protocol.py`, `metrics_harness.py`, `reporting.py`, `tabicl_runtime.py`, and numbered runners.
   - Delete only implementations with no remaining caller; retain documented public names as focused re-exports. Confirm one factual-selection implementation and one common evaluator remain.

3. Update operational documentation.
   - Where: root/suite READMEs, example matrix configs, and Athena README/scripts.
   - Document layer ownership, method/backend extension recipes, semantic metric definitions, run identity, resume/completion behavior, artifact layout, and legacy-shim lifetime.

4. Run the cheap final compatibility matrix.
   - Run `uv run python -m experiments.zeroshot_cf.cli matrix --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml --resume` and then `uv run python -m experiments.zeroshot_cf.cli aggregate --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml`.
   - Where: add `experiments/zeroshot_cf/tests/test_legacy_cli_compatibility.py`; reuse `test_architecture_boundaries.py` from Stage 1.
   - Run `uv run pytest -q experiments/zeroshot_cf/tests/test_legacy_cli_compatibility.py experiments/zeroshot_cf/tests/test_architecture_boundaries.py`. Together these commands validate all 24 real manifests, v1 outputs and shim translations, the common schema, aggregation, offline help, and import boundaries.

5. Run the full reference matrix once as a REPORT.
   - Run `uv run python -m experiments.zeroshot_cf.cli matrix --config experiments/zeroshot_cf/configs/matrices/full_reference.yaml --resume` and then `uv run python -m experiments.zeroshot_cf.cli aggregate --config experiments/zeroshot_cf/configs/matrices/full_reference.yaml`.
   - The tracked config fixes four datasets, six methods, 1,000 factuals, seed 42, and the recorded DiCoFlex `k=3` configuration. Publish 24-cell completeness, metrics, availability/validity semantic differences, per-phase timing, and artifact hashes.
   - Do not block the architecture on noisy quality point comparisons. A missing/incomplete cell opens a focused defect with the exact run ID and evidence.

6. Sweep backlog and verify the repository boundary.
   - Confirm no new optional dependency is eagerly imported, no ignored results/models/vendor data are committed, and the locally excluded architecture note remains untouched.

---

## Verification

- [ ] GATE `uv sync --locked && uv run pytest -q` — dependency lock or retained-suite regressions turn it red.
- [ ] GATE `uv run ruff check experiments/zeroshot_cf/core experiments/zeroshot_cf/datasets experiments/zeroshot_cf/methods experiments/zeroshot_cf/evaluation experiments/zeroshot_cf/orchestration experiments/zeroshot_cf/cli.py` — lint/import defects in every new production module, including the generic CLI, turn it red.
- [ ] GATE run the exact Step 4 compatibility-matrix and focused-test commands — missing cells, schema violations, import-boundary violations, non-offline help, shim-translation errors, or legacy-export failures turn it red.
- [ ] REPORT run the exact Step 5 full-reference commands — record completeness, common metrics, intentional semantic differences, phase timings, and hashes in `journal.md`; publish `NOT MEASURED` if the expensive run cannot be executed.

---

## Commit

`refactor(benchmark): cut over to isolated evaluation architecture`
