# Stage 1: Establish the Canonical Method Surface

**Goal**: Rename the implementation package, types, imports, and method-owned tests while retaining the old runtime identity until Stage 2 can cut over every identity producer atomically.
**Dependencies**: None

Read [the rename contract](../resources/rename-contract.md) before editing.

---

## Steps

1. Rename the implementation package and every method-owned public type.
   - Where: `experiments/zeroshot_cf/methods/dicoflex/` and its `__init__.py` exports.
   - Details: use `git mv` to create `methods/countercontex/`; rename `DiCoFlexConfig`, search,
     diversity, foundation, method, prepared-method, backend-input, backend-runtime, and
     prepared-backend types to the exact `CounterContEx*` spelling. Rename private helpers and
     method-specific diagnostics consistently without changing algorithms or defaults.

2. Update every direct import and type reference to the renamed implementation.
   - Where: the registry factory metadata, `orchestration/tabicl_runtime_compat.py`, method and
     orchestration modules, and all tests importing `methods.dicoflex` submodules.
   - Details: import only `methods.countercontex` and `CounterContEx*` types. Do not create an old
     package re-export, forwarding submodules, class aliases, or `sys.modules` entries.

3. Point the registry at the renamed implementation without changing scientific identity yet.
   - Where: `_dicoflex_factory`, `_dicoflex_variant`, `_dicoflex_runtime`, and the registration
     record in `experiments/zeroshot_cf/methods/registry.py`; `method_id` on the method class.
   - Details: update lazy module/class/config references and private helper names, but deliberately
     retain registry key `dicoflex`, `method_id = "dicoflex"`, and implementation version
     `dicoflex-v3`. Existing `MethodSpec` producers and v1 dispatch therefore remain coherent until
     Stage 2 changes all identity-bearing strings together.

4. Move method-owned tests to the canonical package and type names.
   - Where: the method contract and proposal-backend test modules and direct-import assertions.
   - Details: rename test files where they contain the old method name. Assert the new package and
     type spelling while explicitly asserting the temporary old registry/method/version identity;
     Stage 2 owns those expected-value changes.

---

## Verification

- [ ] GATE `uv run pytest -q experiments/zeroshot_cf/tests/test_countercontex_method_contract.py experiments/zeroshot_cf/tests/test_countercontex_proposal_backends.py experiments/zeroshot_cf/tests/test_method_registry.py` — the new package/types and the deliberately retained Stage 1 registry/method/version identity pass against tracked tests; an early identity change or stale implementation import turns this red.
- [ ] GATE `! git grep -n 'methods\.dicoflex' -- 'experiments/**/*.py'` — no tracked Python module imports the old package path; a stale direct or monkeypatch import turns this red.
- [ ] GATE `uv run pytest -q` — the repository suite passes without a forwarding package or registry alias; an unmigrated import or behavior change turns this red.
- [ ] GATE `uv run ruff check experiments/zeroshot_cf/methods experiments/zeroshot_cf/tests/test_countercontex_method_contract.py experiments/zeroshot_cf/tests/test_countercontex_proposal_backends.py` — renamed modules and tests pass static checks.
- [ ] GATE `git diff --check` — the staged source diff has no whitespace errors.

---

## Commit

`refactor(countercontex): rename method implementation API`
