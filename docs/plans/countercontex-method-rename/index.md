# Plan: Rename the Method to CounterContEx

**Date**: 2026-09-01
**Branch**: `cleanup/tabicl-suite`
**Planning baseline**: `3b50745`
**Predecessor**: [counterfactual-evaluation-architecture](../counterfactual-evaluation-architecture/index.md)
**Goal**: Make `CounterContEx` the only active name for the method across Python APIs, runtime identity, commands, configurations, tests, and documentation.

Executed per [PROTOCOL.md](PROTOCOL.md). Status of record: [state.json](state.json).
Runtime record: [journal.md](journal.md) · [decisions.md](decisions.md) · [backlog.md](backlog.md)

---

## Context

The method is currently named `DiCoFlex` in public Python types and prose and `dicoflex` in its
package, registry key, implementation version, matrix specifications, command names, launchers,
tests, and result paths. The requested proper name is exactly `CounterContEx`; machine identifiers
use `countercontex`.

This is more than a text replacement. `MethodSpec.name` contributes to `cell_id`, and the registry
implementation version contributes to `run_id`. The rename must therefore create a new scientific
identity and must not resume or rewrite existing content-addressed runs. The v1 compatibility
export is separable: its file stems, CSV fields, NPZ keys, and legacy method values already use
TabICL-era identifiers and can retain their schema and deterministic values while the internal
dispatch key changes.

The repository title and schema namespace already use `CounterContEX` and `countercontex.*`.
Those project-level values stay as they are; method-facing prose and Python types use the exact
`CounterContEx` capitalization. See [resources/rename-contract.md](resources/rename-contract.md)
for the mapping, exclusions, and audited surface.

Completed plan directories and ignored result manifests are historical evidence. Do not edit,
rename, or delete them to force a global zero-match result. The active-source audit covers tracked
`README.md`, `experiments/`, and the living `docs/plans/LESSONS.md`; completed plan directories are
excluded. Git history is naturally out of scope.

---

## Strategy

**Move the implementation** (Stage 1): rename the implementation package, public types, direct
imports, and method-owned tests while temporarily retaining the old registry key, method ID, and
implementation version. This keeps existing identity producers and v1 dispatch coherent.

**Cut over identity atomically** (Stage 2): change the registry key/version, method ID, every
`MethodSpec` producer, matrix method value, and v1 dispatch key together; prove the legacy schema
and deterministic values stay stable and isolate new run hashes from old manifests.

**Cut over operations and documentation** (Stage 3): rename command, launcher, config, test, and
documentation paths and update all user-facing instructions.

**Audit the cutover** (Stage 4): remove ignored old-package cache remnants, enforce the exact-name
and import-absence audits, and run the complete offline verification suite.

---

## Success Criteria

Every row declares a **Kind**. These gates are deterministic checks over tracked source, fixtures,
and offline test inputs. `NOT MEASURED` never passes a GATE. No benchmark method or full matrix is
executed by this plan.

| Metric | Baseline | Target | Kind | If missed | If unmeasurable |
|--------|----------|--------|------|-----------|-----------------|
| Active name surface | 302 old-name references outside historical plan records; 17 tracked old-name paths under `README.md` and `experiments/` | no case-insensitive old-name content under tracked `README.md`, `experiments/`, or `docs/plans/LESSONS.md`; no old-name tracked path or importable old package; method labels use exact `CounterContEx` | GATE | block final stage and finish the cutover | REPORT `NOT MEASURED` and block |
| Canonical method API | package `methods.dicoflex`, `DiCoFlex*` types, registry key `dicoflex`, version `dicoflex-v3` | importable `methods.countercontex`, `CounterContEx*` types, sole registry key `countercontex`, version `countercontex-v3` | GATE | block owning stage and repair imports/registration | REPORT `NOT MEASURED` and block |
| Scientific identity | old method and implementation strings determine existing cell/run hashes | canonical specs use the new strings, produce new hashes, and reject old manifests for resume | GATE | block identity stage | REPORT `NOT MEASURED` and block |
| v1 artifact compatibility | frozen `exp9_tabicl_*` stems, ordered CSV fields, NPZ keys, and `tabicl_v2_*` method values | exact stems/schema/IDs, every deterministic CSV field, and every NPZ array value/dtype/shape remain unchanged while dispatch uses `countercontex`; variable timing fields are excluded from value equality | GATE | block compatibility stage | REPORT `NOT MEASURED` and block |
| Operational surface | old Exp9 module, Athena scripts, ablation config, commands, and result roots | renamed CLI help works offline, shell launchers parse, and matrices dry-resolve with canonical specs | GATE | block operational stage | REPORT `NOT MEASURED` and block |
| Repository suite | 263 tests passed after the architecture fixes; full-tree Ruff has 30 unrelated baseline violations | `uv run pytest -q`, Ruff on Python files changed from `3b50745`, and `git diff --check` pass | GATE | block stage and fix the regression | REPORT `NOT MEASURED` and block |
| Historical local artifacts | ignored manifests and result directories contain the old scientific identity | inventory them without mutation; do not run or resume them | REPORT | publish and continue | publish `NOT MEASURED` and continue |

---

## Files That May Be Changed

- `experiments/zeroshot_cf/methods/` — package path, public types, backend types, method ID,
  registry factory, implementation version, and removal of ignored old-package cache remnants.
- `experiments/zeroshot_cf/orchestration/` — canonical method specs, legacy dispatch, v1 contract
  lookup, identity/resume checks, and runtime compatibility imports.
- `experiments/zeroshot_cf/exp8_tabicl_cf.py` and the renamed Exp9 module — canonical internal
  method name and operational command surface.
- `experiments/zeroshot_cf/configs/matrices/` and `experiments/zeroshot_cf/athena/` — matrix names,
  launchers, case tables, commands, and output roots.
- `experiments/zeroshot_cf/evaluation/`, `metrics_harness.py`, and related helpers — method-specific
  function names, diagnostics, docstrings, and errors.
- `experiments/zeroshot_cf/tests/` — renamed tests, canonical API/identity assertions, v1 artifact
  parity, launchers, dry runs, and the final absence check.
- Root `README.md`, `experiments/zeroshot_cf/README.md`, and the Athena README — exact method name
  and renamed commands and paths.
- `docs/plans/countercontex-method-rename/` and `docs/plans/LESSONS.md` — execution plan and durable
  identity lesson.

Excluded: rewriting completed plans, Git history, ignored result artifacts, schema namespace
`countercontex.*`, repository title `CounterContEX`, algorithm changes, metric changes, dataset or
split changes, and real benchmark execution.

---

## Stages

Routing table only. Status, notes, and commits live only in `state.json`.

| # | Stage |
|---|-------|
| 1 | [Establish the canonical method surface](stages/01-canonical-method-surface.md) |
| 2 | [Migrate identity and compatibility contracts](stages/02-migrate-identity-contracts.md) |
| 3 | [Rename operational paths and documentation](stages/03-rename-operations-docs.md) |
| 4 | [Remove the old active name and audit the cutover](stages/04-remove-old-name.md) |
