# Backlog (Deferred Issues)

Each entry must be self-contained enough for a future run to pick it up cold.

| # | Title | Origin | Severity | Why deferred | Next step | Status |
|---|-------|--------|----------|--------------|-----------|--------|
| B-1 | Bound the long tail when diverse search cannot fill the configured candidate pool | Stage 1 | High runtime | The n=25 Lending Club profile spent 498.785 s on one factual, searching to depth 65 for a 16-row pool and ending with 13 although k=3 was already returnable. Changing the stopping rule changes search behavior and scientific identity, which Stage 1 may not do. | In a dedicated behavior-change stage, freeze the outlier as a deterministic witness; compare patience, maximum depth, and stop-after-enough-valid policies on returned-set quality and runtime; bump the CounterContEx implementation version for any adopted policy. | OPEN |
| B-2 | Validate discriminator cache entries against training identity | Stage 2 | Medium reproducibility | Cache filenames separate dataset/preprocessing/family, and fitted-content hashing keeps run identity honest, but loading does not verify the cached estimator was trained under the registry's current fixed params and implementation. The Stage 2 active files were freshly trained/current, so this does not invalidate its evidence. | Store and validate dataset fingerprint, registry params, family, and training implementation beside each cached classifier; retrain rather than reuse on mismatch. | OPEN |

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`.

When an item flips to RESOLVED, **revisit its origin stage in the same commit** -- a stage may
not stay BLOCKED on a resolved item. Summarize the fix in `journal.md`. Heavy items may warrant
their own follow-up plan; link it here.
