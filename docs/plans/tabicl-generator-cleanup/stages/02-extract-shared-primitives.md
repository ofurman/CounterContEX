# Stage 2: Extract Dependency-Neutral Shared Primitives

**Goal**: Remove the small legacy import edges that currently pull TabPFN/CEL or one baseline runner into otherwise independent retained modules.
**Dependencies**: Stage 1

---

## Steps

1. Extract `OneHotActionGroup` from `data.py` into a dependency-light action-space module.
   - Update `data.py`, grouped search, distances, metrics, baselines, and tests to use the same type.
   - Importing the type module must not import CEL, sklearn, TabICL, or dataset files.
2. Extract `infer_feature_domains()` and `project_candidate_values()` from `greedy.py` into a
   neutral candidate-domain module.
   - Update Exp8, `grouped_categorical.py`, `diverse_search.py`, and relevant tests.
   - Migrate projection and retained quantile-selection assertions from `test_greedy.py`; preserve
     small-support projection, range clipping, integer/categorical support, and batch semantics.
3. Extract shared action operations from numbered baseline runners.
   - Move `ActionUnit`, action-unit construction, action pruning, and scalar contraction into a
     baseline-common module.
   - Update Exp11–14 so Exp12/13/14 do not import NICE or each other for helpers.
4. Extract shared immutable configuration such as the target threshold from Exp4 into a retained
   configuration module. Leave standalone Exp4 reporting behind until Stage 3 replaces its use.
5. Keep compatibility imports only where a still-live legacy test/routine needs them. Mark them for
   Stage 5 deletion; do not create a new bidirectional dependency.

---

## Verification

- [ ] GATE focused tests for action-space types, candidate projection, grouped search, diversity, and Exp11–14 pass — production helper modules and migrated tests are the inputs; semantic drift turns it red.
- [ ] GATE `rg -n 'experiments\.zeroshot_cf\.(greedy|sampler|exp4_greedy_cf|exp11_nice_nun_baseline|exp12_optimization_baselines)' experiments/zeroshot_cf/grouped_categorical.py experiments/zeroshot_cf/diverse_search.py experiments/zeroshot_cf/mixed_distance.py experiments/zeroshot_cf/exp1[1-4]_*.py` returns no retained cross-runner/helper matches — a legacy or numbered-runner utility dependency turns it red.
- [ ] GATE importing the new action-space, candidate-domain, and baseline-common modules in a fresh Python process succeeds without CEL, TabICL, TabPFN, DiCE, or checkpoints — eager dependency side effects turn it red.
- [ ] GATE the Stage 1 dataset-contract test remains green — the pinned CEL outputs are the input; type extraction that changes metadata turns it red.

---

## Commit

`refactor(tabicl-cf): extract shared action primitives`
