# Journal

Append-only. Newest entries at the bottom. Never rewrite an earlier entry.

One entry per invocation, in this shape:

```
## YYYY-MM-DD HH:MM -- Stage N: [Name] -- DONE
**Did**: [1-3 lines]
**Verification**: GATE lines passed. REPORT values: [metric]=[value]
**Provenance**: [per measured GATE: the input the value was read from, and the defect that would
turn it red] - [or `NOT MEASURED` for any that could not be produced from this run's own inputs]
**Problems**: [symptom -> root cause -> resolution -> inline/subagent] or "none"
**Commit**: `abc1234`
```

---

## 2026-09-01 11:04 CEST -- Stage 1: Establish the Canonical Method Surface -- DONE
**Did**: Moved the ten-module implementation to `methods/countercontex`, renamed public types and method-owned tests, migrated direct imports, and pointed the still-`dicoflex` registry entry at the canonical implementation.
**Verification**: GATE lines passed: targeted tests 27 passed; full suite 263 passed with five existing warnings; scoped Ruff, old-import grep, and diff checks passed. REPORT values: none.
**Provenance**: Tests read the tracked renamed modules and explicit temporary identity assertions; stale imports, an early method/version change, algorithm regressions, lint defects, or whitespace errors turn the corresponding GATE red. Independent audit found all ten implementation ASTs equivalent to baseline after name normalization.
**Problems**: Initial tracked-file replacement walked an ignored binary `.py` file -> reran over `git ls-files`; targeted test retained two old AST-inspection paths and renamed identifiers exceeded Ruff line limits -> focused subagent fixed them; identity assertions were initially implicit -> focused follow-up added exact method/version checks.
**Commit**: `pending`

## 2026-09-01 11:15 CEST -- Stage 2: Migrate Identity and Compatibility Contracts -- DONE
**Did**: Migrated the registry, method ID, all `MethodSpec` producers, matrices, metric helper, and v1 dispatcher to `countercontex` / `countercontex-v3`. Added explicit old-run isolation and misplaced-manifest rejection tests plus a complete v1 semantic golden.
**Verification**: GATE lines passed: targeted tests 62 passed with one existing warning; one-factual dry-run emitted 24 specs with exactly four canonical method cells and no old name; full suite 265 passed with five existing warnings; scoped Ruff and diff checks passed. REPORT values: none.
**Provenance**: The targeted suite reads tracked identity producers, manifests, compatibility tables, and the semantic golden; old-name selection, accepted mismatched manifests, schema/value drift, or missing arrays turn it red. The golden matched byte-for-byte against the old `dicoflex` v1 fake export from commit `1ab5a9b` in a detached worktree. The dry-run reads the tracked one-factual matrix; changed cardinality or method names turn it red. The full suite reads all tracked tests; exercised regressions turn it red. Ruff and `git diff --check` read the migrated source diff; configured lint or whitespace defects turn them red. An independent audit passed the identity, resume, and compatibility evidence.
**Problems**: Temporary baseline instrumentation initially missed an `os` import and the golden lacked a terminal newline -> fixed only the detached verifier and normalized the tracked JSON newline; full suite found a stale registry-order expectation after alphabetic renaming -> focused subagent reordered the expected tuple and its focused test passed.
**Commit**: `pending`

## 2026-09-01 11:20 CEST -- Stage 3: Rename Operational Paths and Documentation -- DONE
**Did**: Renamed the Exp9 command module, three Athena files, ablation matrix, result roots, fixture metadata, imports, tests, and active documentation to `CounterContEx` / `countercontex`. Preserved the project title and historical v1 artifact interface.
**Verification**: GATE lines passed: offline renamed CLI help rendered; both launcher scripts passed `bash -n`; targeted tests 33 passed; both tracked matrices dry-ran with 24 specs, full reference contained exactly four canonical cells, and neither output contained the old name; full suite 265 passed with five existing warnings; diff check passed. REPORT values: Athena case rows=4.
**Provenance**: CLI help imports the renamed tracked module offline; a stale module or checkpoint/network import turns it red. Shell parsing reads both renamed launchers; syntax or broken quoting turns it red. Targeted tests read launcher paths, fixture modules, CLI boundaries, and Exp9 imports; stale references turn them red. Matrix dry-runs read both tracked YAML files without executing methods; changed cardinality, identity, or path turns them red. The full suite and diff check read the tracked repository and source diff; exercised regressions or whitespace defects turn them red. The case count comes from the renamed tracked TSV. An independent audit normalized moved files back to their Stage 2 names and found them byte-equivalent.
**Problems**: none.
**Commit**: `pending`

## 2026-09-01 11:27 CEST -- Stage 4: Remove the Old Active Name and Audit the Cutover -- DONE
**Did**: Removed the final searchable pre-rename literals while preserving the runtime identity-isolation witnesses, deleted five ignored old-name bytecode files, and completed the active content/path/import/alias audit. Swept the backlog and left the unrelated null-diagnostic repair bug open.
**Verification**: GATE lines passed: no tracked active old-name content or paths; old package did not resolve and no old implementation/cache path remained; full suite 265 passed with five existing warnings; Ruff passed for all 31 changed Python files since `3b50745`; offline renamed CLI and both 24-cell matrix dry-runs passed with four canonical full-reference cells; diff check passed. REPORT values: manifests=25, old-identity manifests=5, old-named result files=26 across two directories (13 each).
**Provenance**: Content/path gates read tracked `README.md`, `experiments/`, and living lessons plus tracked filenames; a stale token or path turns them red. Import and filesystem gates inspect the execution workspace; a forwarding package, namespace remnant, or old cache turns them red. The split-string tests still construct the exact pre-rename identity at runtime and fail if hashes collide, the old run is selected, or a misplaced manifest resumes. Pytest reads the full tracked suite; Ruff reads the exact 31 changed Python files from baseline; CLI and dry-runs read the renamed module and tracked matrices offline without method execution. The REPORT reads ignored results only; modification times predate this rename. Independent audit passed all surfaces.
**Problems**: Final changed-file Ruff exposed nine line-length violations -> focused subagent applied formatting-only fixes and 50 focused tests passed; independent audit found five ignored old-name `.pyc` files outside the moved package -> deleted only those caches and rechecked that no non-result remnant remains.
**Commit**: `pending`
