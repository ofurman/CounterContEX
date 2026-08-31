# Backlog (Deferred Issues)

| # | Title | Origin | Severity | Why deferred | Next step | Status |
|---|-------|--------|----------|--------------|-----------|--------|
| 1 | Root `uv run pytest` requires licensed TabPFN v2.5/v2.6 downloads | Stage 1 | medium | The project-wide suite fails outside `experiments/zeroshot_cf` because root architecture/interface tests for `v2.5` and `v2.6` checkpoints attempt interactive license acceptance or weight downloads in this non-interactive environment. | Resolved in Stage 7 by deleting the upstream root suite and repointing root `pytest` at the retained TabICL suite. | RESOLVED |
| 2 | Root `uv run pytest` still has upstream non-TabICL failures outside the retained suite | Stage 3 | medium | After Stage 3 the retained generator slice passed, but the root suite still reproduced upstream failures in `tests/test_architectures/test_tabpfn_v2_5.py`, `tests/test_classifier_interface.py`, and `tests/test_consistency.py` before the run was interrupted; those checks live outside the retained `experiments/zeroshot_cf` boundary and Stage 7 later removes or repurposes much of that root surface. | Resolved in Stage 7 by deleting the upstream root suite and making the retained suite the only root `pytest` target. | RESOLVED |

No issues are deferred at planning time. CEL remains part of the retained suite, and repository
creation/history rewriting are explicitly outside scope.

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`.

When an item becomes resolved, revisit its origin stage in the same commit if that stage is blocked
on it. Heavy items may receive their own linked follow-up plan.
