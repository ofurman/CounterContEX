# Backlog (Deferred Issues)

Each entry must be self-contained enough for a future run to pick it up cold.

| # | Title | Origin | Severity | Why deferred | Next step | Status |
|---|-------|--------|----------|--------------|-----------|--------|
| 1 | Legacy v1 repair fails on null optional diagnostics | Pre-existing architecture full-reference resume | medium | `orchestration/legacy.py::export_generic_v1` can call `float(None)` when repairing a canonical manifest whose optional `initial_valid_action_sparsity` diagnostic is JSON `null`; this is a semantic resume/export bug unrelated to naming. | Reproduce with the saved HELOC historical artifact or a focused fixture, define the optional-diagnostic policy, and fix under a separate bug task without running the full matrix. | OPEN |

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`.

When an item flips to RESOLVED, **revisit its origin stage in the same commit** -- a stage may
not stay BLOCKED on a resolved item. Summarize the fix in `journal.md`. Heavy items may warrant
their own follow-up plan; link it here.

Real one-factual and 24-cell benchmark execution is outside this rename. The full reference matrix
previously took 9.42 measured hours and the user explicitly stopped further runs.
