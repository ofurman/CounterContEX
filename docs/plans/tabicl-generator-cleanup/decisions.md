# Decisions

Append-only after execution starts. **<=15 lines per entry**; detail belongs in `resources/`.

### D-1: Retain the complete comparison suite
**Date**: 2026-08-30 - **Stage**: planning
**Options**: A) retain only single-CF generation; B) retain generator, Exp9, baselines, diversity, and operations.
**Chosen**: B.
**Rationale**: The requested deliverable explicitly includes Exp9, NICE, Wachter, Growing Spheres, DiCE, FACE, bounded-beam/DPP diversity, checkpoints, tests, metrics, and Athena.

### D-2: Remove the upstream TabPFN project after suite isolation
**Date**: 2026-08-30 - **Stage**: planning
**Options**: A) isolate `experiments/zeroshot_cf` first; B) simultaneously delete the upstream TabPFN repository and rewrite root packaging.
**Chosen**: B, sequenced after Stages 1–6 establish an independently green suite.
**Rationale**: The user wants the final repository to contain only the TabICL suite. Delaying root deletion until the suite has a locked environment protects behavior while keeping removal in this plan.

### D-3: Retain and pin CEL
**Date**: 2026-08-30 - **Stage**: planning
**Options**: A) replace CEL with a local loader; B) keep CEL and pin the current revision.
**Chosen**: B, commit `3587f943826f6b087a0d198c8c4aa4373712c7ee`.
**Rationale**: The user explicitly requires CEL to remain. Exp9 continues to use its preprocessing and vendor YAML/CSV assets; the plan only makes that dependency reproducible.

### D-4: Preserve public experiment entry points during cleanup
**Date**: 2026-08-30 - **Stage**: planning
**Options**: A) rename the package and runners immediately; B) add stable internal modules while keeping Exp8/9/11–14 CLI shims.
**Chosen**: B.
**Rationale**: Athena and user commands already target `experiments.zeroshot_cf.*`; compatibility shims allow internal isolation without combining it with a namespace migration.

### D-5: Keep both supported counterfactual modes explicit
**Date**: 2026-08-30 - **Stage**: planning
**Options**: A) force refinement and diversity into one algorithm; B) preserve single-CF plausibility refinement and multi-CF sparse bounded-beam/DPP as separate modes.
**Chosen**: B.
**Rationale**: Current Exp9 does not combine `data_plausible` refinement with `n_counterfactuals > 1`. Cleanup records and tests that contract rather than changing experimental behavior.

### D-6: Preserve local user state
**Date**: 2026-08-30 - **Stage**: planning
**Options**: A) delete caches, downloaded data, checkpoints, and untracked docs; B) change tracked scope only and report local state.
**Chosen**: B.
**Rationale**: `ARCHITECTURE.md`, vendor/data trees, checkpoints, model caches, and generated outputs may be user-owned and are not cleanup targets. The top-level `tabpfn/` cache is explicitly authorized for deletion by D-2.

### D-7: Work on a branch and squash after removal
**Date**: 2026-08-30 - **Stage**: planning
**Options**: A) create a fresh repository or filter history; B) create a cleanup branch with atomic stage commits and squash later.
**Chosen**: B; branch `cleanup/tabicl-suite` from `e80925d`.
**Rationale**: The user requested ordinary branch-based cleanup and will squash commits after the removal. No repository initialization or history rewriting belongs in this plan.

### D-8: Focused Stage 1 tests import vendored CEL from `PYTHONPATH`
**Date**: 2026-08-30 - **Stage**: 1
**Options**: A) install the vendored CEL checkout as a normal editable requirement; B) keep vendor setup as the runtime installer and let focused tests import the pinned checkout directly.
**Chosen**: B.
**Rationale**: Installing the vendor checkout through `requirements.txt` pulled CEL's unsupported optional dependency graph (`onnx` build failure). The retained Stage 1 suite only needs the pinned source tree plus minimal transitive deps, so `tests/conftest.py` adds the vendor path directly while `vendor_setup.py` remains the authoritative runtime bootstrap.

### D-9: Stage 1 pins `tabpfn-extensions` to upstream `v0.4.2`
**Date**: 2026-08-30 - **Stage**: 1
**Options**: A) drop `tabpfn-extensions` before the legacy imports are removed; B) keep a compatible upstream tag until later isolation stages cut the dependency.
**Chosen**: B, tag `v0.4.2`.
**Rationale**: `exp8_tabicl_cf` still reaches `sampler.py` through legacy imports, and upstream metadata shows `v0.4.2` is the last tag compatible with this repo's `tabpfn==8.0.8` (`v0.5.0` requires `tabpfn>=8.1.0`, `v0.6.0` requires `>=8.4.0`). This preserves the real import path without widening Stage 1 into the later isolation refactor.

### D-10: Keep Stage 2 compatibility shims only on retained legacy entry points
**Date**: 2026-08-30 - **Stage**: 2
**Options**: A) move shared helpers and immediately break `exp4`/`greedy` imports; B) move the helpers into neutral modules and leave temporary re-export names only on the still-live legacy runners.
**Chosen**: B.
**Rationale**: Stage 2 must cut retained cross-runner dependencies without widening into Stage 5 deletion work. `retained_config.py`, `candidate_domains.py`, `action_space.py`, and `baseline_common.py` become the sources of truth, while `exp4_greedy_cf` and `greedy.py` keep compatibility names marked for Stage 5 removal.

### D-11: Stage 3 gates measure the isolated public generator boundary
**Date**: 2026-08-30 - **Stage**: 3
**Options**: A) keep counting legacy `test_tabicl_backend.py` in the Stage 3 gate; B) measure the new public generator API plus grouped/diverse/plausibility/distance coverage and leave the legacy backend test outside this stage gate.
**Chosen**: B.
**Rationale**: `test_tabicl_backend.py` still imports `greedy.py -> sampler.py -> tabpfn_extensions`, so it re-measures the legacy coupling Stage 3 is explicitly isolating away from the retained public boundary. The new `test_generator.py` covers the same confidence/quantile/revisit/refinement/diversity behaviors through `generator.py` with fake backends.

### D-12: Stage 4 centralizes benchmark protocol and TabICL runtime adapters
**Date**: 2026-08-30 - **Stage**: 4
**Options**: A) keep Exp9 and each baseline reconstructing splits, targets, classifier tags, output paths, and CSV writers independently; B) move the fixed four-dataset benchmark contract into neutral helpers and let Exp9 call the stable generator through a non-runner runtime adapter.
**Chosen**: B.
**Rationale**: Stage 4 needs Exp9 and Exp11-14 to share one explicit protocol without importing each other's private helpers. `benchmark_protocol.py` now owns dataset order, split/selection defaults, classifier-derived targets, common result fields, and output naming, while `tabicl_runtime.py` keeps Exp9 off `exp8_tabicl_cf.py` and preserves lazy checkpoint loading behind the stable `generator.py` API.

### D-13: Keep only the retained fake-TabICL assertions from `test_tabicl_backend.py`
**Date**: 2026-08-30 - **Stage**: 5
**Options**: A) delete `test_tabicl_backend.py` entirely with the old `greedy.py` path; B) keep the fake-backend `tabicl_sampler.py` coverage and remove only the legacy `greedy_counterfactual` cases plus the real-TabPFN fixture.
**Chosen**: B.
**Rationale**: The stage brief explicitly preserves migrated quantile/data assertions while deleting only legacy-only test coverage. `test_tabicl_backend.py` still measures retained TabICL context selection, candidate batching, confidence conditioning, and quantile behavior without loading real checkpoints, so only the `greedy.py`-dependent tail was removed.

### D-14: Make the root a thin `uv` workspace wrapper around the retained suite
**Date**: 2026-08-30 - **Stage**: 7
**Options**: A) duplicate the suite dependency list and lock in the root project; B) make the root depend on `experiments/zeroshot_cf` as a `tool.uv.workspace` member and generate a root wrapper lock.
**Chosen**: B.
**Rationale**: Stage 7 requires `uv sync --locked` and `uv run pytest -q` to work from the repository root without creating a second hand-maintained dependency manifest. The root now wraps the suite's existing project metadata, while the suite-local lock remains available for isolated Athena and local runs.
