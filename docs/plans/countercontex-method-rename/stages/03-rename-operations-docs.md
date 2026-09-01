# Stage 3: Rename Operational Paths and Documentation

**Goal**: Make commands, launchers, configurations, tests, result roots, and active documentation use only the new method name.
**Dependencies**: Stage 2

---

## Steps

1. Rename the Exp9 command module and its tests.
   - Where: `experiments/zeroshot_cf/exp9_dicoflex_benchmark.py`, imports in tests, and the
     architecture-v1 fixture's CLI/source-module metadata.
   - Details: use `git mv` to create `exp9_countercontex_benchmark.py`; update module invocation,
     help text, default result directory, and all callers. The old module is not retained after the
     final stage.

2. Rename Athena launchers and case-table paths as one operational unit.
   - Where: `athena/exp9_dicoflex_array.sbatch`, `athena/exp9_dicoflex_cases.tsv`,
     `athena/submit_exp9_dicoflex.sh`, launcher tests, and Athena README commands.
   - Details: use `exp9_countercontex_*`, update internal paths and default output roots, and keep
     scheduler behavior unchanged.

3. Rename the ablation configuration path and operational metadata.
   - Where: `configs/matrices/dicoflex_ablation_example.yaml`, `full_reference.yaml`,
     `one_factual_compat.yaml`, and config references in tests/docs.
   - Details: rename the example to `countercontex_ablation_example.yaml` and update its suite and
     output-root values plus all path references. Stage 2 already owns every matrix method-name
     value, including the unchanged-filename reference and compatibility configs. Do not change
     datasets, seeds, hyperparameters, or cell cardinality.

4. Update all active documentation and user-visible prose.
   - Where: root README, suite README, Athena README, `docs/plans/LESSONS.md`, docstrings,
     diagnostics, and test names under tracked `README.md` and `experiments/`.
   - Details: use exact `CounterContEx` for the method and `countercontex` for machine names. Keep
     project title `CounterContEX`, schema namespace `countercontex.*`, legacy TabICL artifact names,
     and completed plan directories unchanged. Update the living durable lesson to the current
     method name without rewriting its linked historical plan.

---

## Verification

- [ ] GATE `HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp9_countercontex_benchmark --help` — the renamed tracked CLI imports and renders help without a checkpoint or network access; an old module reference turns this red.
- [ ] GATE `bash -n experiments/zeroshot_cf/athena/submit_exp9_countercontex.sh experiments/zeroshot_cf/athena/exp9_countercontex_array.sbatch` — renamed launchers parse as shell scripts.
- [ ] GATE `uv run pytest -q experiments/zeroshot_cf/tests/test_athena_launchers.py experiments/zeroshot_cf/tests/test_legacy_cli_compatibility.py experiments/zeroshot_cf/tests/test_architecture_boundaries.py experiments/zeroshot_cf/tests/test_exp9_benchmark.py` — commands, fixture modules, and repository boundaries use renamed tracked paths.
- [ ] GATE `set -euo pipefail; tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT; for name in full_reference countercontex_ablation_example; do uv run python -m experiments.zeroshot_cf.cli matrix --config "experiments/zeroshot_cf/configs/matrices/$name.yaml" --dry-run > "$tmp/$name"; test "$(wc -l < "$tmp/$name" | tr -d ' ')" = 24; ! rg -i dicoflex "$tmp/$name"; done; test "$(rg -c '\"name\": \"countercontex\"' "$tmp/full_reference")" = 4` — both tracked matrices retain their cardinality and canonical identity without executing a method; fail-fast shell semantics preserve a changed matrix or stale-name failure.
- [ ] GATE `uv run pytest -q` — the full repository suite passes after the operational rename.
- [ ] GATE `git diff --check` — the staged source and documentation diff has no whitespace errors.

---

## Commit

`refactor(countercontex): rename commands and operational files`
