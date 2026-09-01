# Stage 2: Migrate Identity and Compatibility Contracts

**Goal**: Change the registry, method, specification, matrix, and v1 dispatch identities to `countercontex` atomically while preserving the legacy artifact schema and deterministic values.
**Dependencies**: Stage 1

Read [the rename contract](../resources/rename-contract.md), especially the identity and immutable
artifact rules, before editing.

---

## Steps

1. Cut the registry and every scientific-identity producer over in one change.
   - Where: the method registration record and factory/runtime helpers in `methods/registry.py`,
     `method_id` on `CounterContExMethod`, `legacy_run_spec()` callers in the Exp8/Exp9 modules,
     `exp8_compat.py`, all matrix `method.name` values, and orchestration tests.
   - Details: change the sole registry key to `countercontex`, implementation version to
     `countercontex-v3`, method ID to `countercontex`, and every newly constructed `MethodSpec` to
     the same name. Preserve the algorithm generation `v3`. Keep historical Exp8/TabICL names only
     where they describe the v1 artifact interface rather than current method identity.

2. Rekey the v1 compatibility dispatcher without changing its output contract.
   - Where: the top-level method record in `orchestration/v1_contract.py`, `_legacy_method_id()` and
     method-specific branches in `orchestration/legacy.py`, and the architecture-v1 fixture.
   - Details: change the internal lookup to `countercontex`. Preserve exact `exp9_tabicl_*` file
     stems, ordered CSV fields, NPZ keys, and `tabicl_v2_sparse`, `tabicl_v2_data_plausible`, and
     `tabicl_v2_diverse_dpp` values. Extend the deterministic compatibility fixture/test to compare
     every non-timing CSV value and every NPZ array's values, dtype, and shape; check timing fields
     for presence and type only. Update lookup metadata only where canonical dispatch changed.

3. Make the scientific identity break explicit and safe.
   - Where: `RunSpec.scientific_payload()`, registry implementation metadata, manifest/resume tests,
     and orchestration matrix/runner/spec tests.
   - Details: prove `countercontex` and `countercontex-v3` enter the identity payload, yield new
     cell/run hashes, and prevent an old-name manifest from being selected or resumed. Do not edit
     old manifests or rename content-addressed directories.

4. Update method-specific metric and diagnostic helpers used outside the package.
   - Where: `compute_dicoflex_common_metrics()` and related strings/imports in `metrics_harness.py`,
     `evaluation/metrics.py`, and their tests.
   - Details: rename symbols and user-visible diagnostics only; keep formulas and result semantics
     unchanged.

---

## Verification

- [ ] GATE `uv run pytest -q experiments/zeroshot_cf/tests/test_legacy_cli_compatibility.py experiments/zeroshot_cf/tests/test_orchestration_matrix.py experiments/zeroshot_cf/tests/test_orchestration_runner.py experiments/zeroshot_cf/tests/test_orchestration_spec.py experiments/zeroshot_cf/tests/test_metrics_harness.py` — tracked deterministic fixtures prove canonical identity, normal old-run isolation plus rejection of a misplaced old manifest, exact v1 stems/schema/IDs and non-timing CSV values, and every NPZ array value/dtype/shape; any changed deterministic artifact value or accepted old manifest turns this red.
- [ ] GATE `set -euo pipefail; out=$(mktemp); trap 'rm -f "$out"' EXIT; uv run python -m experiments.zeroshot_cf.cli matrix --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml --dry-run > "$out"; test "$(wc -l < "$out" | tr -d ' ')" = 24; test "$(rg -c '\"name\": \"countercontex\"' "$out")" = 4; ! rg -i dicoflex "$out"` — the tracked fixture produces 24 specs including four canonical method cells, with no old identity, and loads no checkpoints; fail-fast shell semantics preserve a cardinality or name failure.
- [ ] GATE `uv run pytest -q` — the full repository suite passes after all runtime producers migrate.
- [ ] GATE `uv run ruff check experiments/zeroshot_cf/orchestration experiments/zeroshot_cf/metrics_harness.py experiments/zeroshot_cf/evaluation` — migrated identity and compatibility code passes static checks.
- [ ] GATE `git diff --check` — the staged source diff has no whitespace errors.

---

## Commit

`refactor(countercontex): migrate run and artifact identity`
