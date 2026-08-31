# Stage 5: Encapsulate DiCoFlex

**Goal**: Present the existing TabICL-backed greedy/diverse generator as one benchmark-facing method while preserving its public generator API and search behavior.
**Dependencies**: Stage 3
**References**: [`architecture.md`](../resources/architecture.md#method-contract), [`current-state.md`](../resources/current-state.md#behavior-that-must-remain-stable)

---

## Steps

1. Introduce typed DiCoFlex configuration and method lifecycle.
   - Where: new `methods/dicoflex/config.py` and `methods/dicoflex/method.py`.
   - Separate search, diversity, and foundation settings. `prepare()` owns current runtime/checkpoint/backend setup; `generate()` adapts `generate_counterfactual_batch()` into canonical candidates, availability, and namespaced diagnostics.

2. Keep search algorithms stable during encapsulation.
   - Where: current `generator.py`, `diverse_search.py`, `candidate_domains.py`, and grouped search modules, optionally re-exported from `methods/dicoflex/search.py`.
   - Do not genericize TabICL calls yet. Preserve single-CF refinement, bounded-beam candidate generation, exact fixed-size DPP selection, no-padding behavior, and current deterministic seeds.

3. Preserve public compatibility APIs.
   - Where: `generate_counterfactual_batch()` in `generator.py`, `run_tabicl_benchmark()` in `tabicl_runtime.py`, and Exp8/Exp9 entry points.
   - Make compatibility functions delegate to `DiCoFlexMethod` or focused adapters without changing their documented arguments/results. Exp9 common evaluation and persistence use Stage 3 layers.

4. Normalize common versus method diagnostics.
   - Where: DiCoFlex result adapter and compatibility exporter.
   - Rename generic metadata concepts internally (`proposal_backend`, `joint_scoring`, `cache`) while mapping legacy TabICL field names only at the compatibility boundary.

---

## Verification

- [ ] GATE `uv run pytest -q experiments/zeroshot_cf/tests/test_generator.py experiments/zeroshot_cf/tests/test_diverse_search.py experiments/zeroshot_cf/tests/test_tabicl_backend.py experiments/zeroshot_cf/tests/test_exp9_benchmark.py` — existing fixture inputs detect search, DPP, backend, public API, or Exp9 regressions.
- [ ] GATE canonical DiCoFlex contract tests cover `k=1`, partial `k=3`, no-padding, config serialization, and legacy adapter equivalence — shape, availability, or metadata drift turns them red.
- [ ] GATE offline import/help tests prove aggregation and CLI parsing load no checkpoint — a backend construction side effect turns them red.

---

## Commit

`refactor(dicoflex): encapsulate generator behind method contract`
