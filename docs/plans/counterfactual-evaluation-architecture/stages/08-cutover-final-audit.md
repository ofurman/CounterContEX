# Stage 8: Cut Over Entry Points and Run the Final Audit

**Goal**: Make the generic architecture the production path, retain only thin compatibility shims, and publish structural, parity, and full-matrix evidence.
**Dependencies**: Stages 6 and 7
**References**: [`architecture.md`](../resources/architecture.md), [`current-state.md`](../resources/current-state.md)

---

## Steps

1. Reduce legacy entry points to translation shims.
   - Where: Exp8/9/11–14 modules and Athena launchers.
   - Parsers translate legacy flags to concrete `RunSpec`s, invoke the generic runner/aggregator, and request the v1 compatibility exporter. Remove duplicated data preparation, common evaluation, row assembly, and direct persistence.

2. Audit and remove obsolete helpers after import proof.
   - Where: `data.py`, `benchmark_protocol.py`, `metrics_harness.py`, `reporting.py`, `tabicl_runtime.py`, and numbered runners.
   - Delete only implementations with no remaining caller; retain documented public names as focused re-exports. Confirm one factual-selection implementation and one common evaluator remain.

3. Update operational documentation.
   - Where: root/suite READMEs, example matrix configs, and Athena README/scripts.
   - Document layer ownership, method/backend extension recipes, semantic metric definitions, run identity, resume/completion behavior, artifact layout, and legacy-shim lifetime.

4. Run the cheap final compatibility matrix.
   - Execute all 24 method/dataset cells with one factual through the generic runner and legacy shims. Validate manifests, v1 outputs, common schema, aggregation, offline help, and import boundaries.

5. Run the full reference matrix once as a REPORT.
   - Use four datasets, six methods, 1,000 factuals, seed 42, and the recorded DiCoFlex `k=3` configuration. Publish 24-cell completeness, metrics, availability/validity semantic differences, per-phase timing, and artifact hashes.
   - Do not block the architecture on noisy quality point comparisons. A missing/incomplete cell opens a focused defect with the exact run ID and evidence.

6. Sweep backlog and verify the repository boundary.
   - Confirm no new optional dependency is eagerly imported, no ignored results/models/vendor data are committed, and the locally excluded architecture note remains untouched.

---

## Verification

- [ ] GATE `uv sync --locked && uv run pytest -q` plus Ruff on new `core`, `datasets`, `methods`, `evaluation`, and `orchestration` packages — source/test inputs expose suite or new-module lint regressions.
- [ ] GATE the 24-cell one-factual compatibility audit reads real manifests and artifacts — missing cells, schema violations, import-boundary violations, non-offline help, or legacy-export failures turn it red.
- [ ] REPORT the 1,000-factual full matrix — record completeness, common metrics, intentional semantic differences, phase timings, and hashes in `journal.md`; publish `NOT MEASURED` if the expensive run cannot be executed.

---

## Commit

`refactor(benchmark): cut over to isolated evaluation architecture`
