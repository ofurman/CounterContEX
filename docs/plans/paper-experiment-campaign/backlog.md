# Backlog (Deferred Issues)

Each entry must be self-contained enough for a future run to pick it up cold.

| # | Title | Origin | Severity | Why deferred | Next step | Status |
|---|-------|--------|----------|--------------|-----------|--------|
| B-1 | Bound the long tail when diverse search cannot fill the configured candidate pool | Stage 1 | High runtime | The n=25 Lending Club profile spent 498.785 s on one factual, searching to depth 65 for a 16-row pool and ending with 13 although k=3 was already returnable. Changing the stopping rule changes search behavior and scientific identity, which Stage 1 may not do. | In a dedicated behavior-change stage, freeze the outlier as a deterministic witness; compare patience, maximum depth, and stop-after-enough-valid policies on returned-set quality and runtime; bump the CounterContEx implementation version for any adopted policy. | OPEN |
| B-2 | Validate discriminator cache entries against training identity | Stage 2 | Medium reproducibility | Cache filenames separate dataset/preprocessing/family, and fitted-content hashing keeps run identity honest, but loading does not verify the cached estimator was trained under the registry's current fixed params and implementation. The Stage 2 active files were freshly trained/current, so this does not invalidate its evidence. | Store and validate dataset fingerprint, registry params, family, and training implementation beside each cached classifier; retrain rather than reuse on mismatch. | OPEN |
| B-12 | Add directional and monotonic action constraints | Stage 12 | Medium research scope | The campaign evaluates immutable and atomic-group actionability only. Adding directionality changes the legal action space, candidate set, metrics, and method identity and therefore needs its own controlled experiment. | Define dataset-owned directional constraints, enforce them in every comparable method, add evaluation measures, and run a separately versioned matrix. | OPEN |
| B-13 | Extend the benchmark target policy to multiclass | Stage 12 | Medium research scope | The current target contract is binary and the campaign contains only binary datasets. A multiclass target policy changes case construction, validity semantics, and method compatibility. | Specify an explicit multiclass target policy, add at least one dataset, and test reversed/nonconsecutive labels and every retained method. | OPEN |
| B-14 | Formalize the search objective and optimality gap | Stage 12 | Medium paper scope | The campaign measures heuristic search behavior but does not prove optimality. A formal constrained-first-crossing statement requires a separate mathematical review and must not overclaim the proposal-support prior. | Draft and independently review a proposition defining the objective, assumptions, and honest optimality gap before adding it to a paper. | OPEN |
| B-15 | Add human or utility evaluation | Stage 12 | Medium study scope | Artifact metrics cannot establish whether users find the explanations useful or actionable. A user study requires protocol, ethics, recruitment, and analysis work beyond this computational campaign. | Design a preregistered user/utility study and obtain the required review before collecting data. | OPEN |

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`.

When an item flips to RESOLVED, **revisit its origin stage in the same commit** -- a stage may
not stay BLOCKED on a resolved item. Summarize the fix in `journal.md`. Heavy items may warrant
their own follow-up plan; link it here.
