# Backlog (Deferred Issues)

Each entry must be self-contained enough for a future run to pick it up cold.

| # | Title | Origin | Severity | Why deferred | Next step | Status |
|---|-------|--------|----------|--------------|-----------|--------|

No issues are deferred at planning time. Actual TabPFN and TabFM adapters are deliberately outside
this refactor: Stage 7 proves the extension seam with TabICL and a fake second backend, after which
each real integration can be planned against its concrete native capabilities and dependencies.

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`.

When an item flips to RESOLVED, **revisit its origin stage in the same commit** -- a stage may
not stay BLOCKED on a resolved item. Summarize the fix in `journal.md`. Heavy items may warrant
their own follow-up plan; link it here.
