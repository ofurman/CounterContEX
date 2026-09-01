# Stage 4: Remove the Old Active Name and Audit the Cutover

**Goal**: Prove no tracked active source, path, command, compatibility alias, ignored cache remnant, or importable package retains the previous method name.
**Dependencies**: Stage 3

---

## Steps

1. Remove old-package workspace remnants and audit for accidental aliases.
   - Where: ignored `__pycache__` directories left below the moved old package path, plus registry,
     import, class-alias, matrix, and CLI surfaces changed in Stages 1–3.
   - Details: if the old directory exists, run `find experiments/zeroshot_cf/methods/dicoflex
     -type d -name __pycache__ -prune -exec rm -rf {} +` followed by `find
     experiments/zeroshot_cf/methods/dicoflex -depth -type d -empty -delete`; do not touch ignored
     result artifacts. Confirm no forwarding package, old class alias, registry-name normalizer,
     old matrix value, or old Exp9 module remains. The old package must not resolve even as a
     namespace package in the execution workspace.

2. Audit exact spelling and tracked paths.
   - Where: all tracked `README.md`, `experiments/`, and `docs/plans/LESSONS.md` content plus tracked
     `README.md` and `experiments/` filenames.
   - Details: remove any remaining case-insensitive old-name token. Confirm method prose/types use
     `CounterContEx`, machine identifiers use `countercontex`, and repository-level
     `CounterContEX` / `countercontex.*` values were not changed merely to satisfy the audit.

3. Record historical artifacts without mutating them.
   - Where: ignored manifests below `experiments/zeroshot_cf/results/` and completed plans below
     `docs/plans/`.
   - Details: append counts/locations to `journal.md` as a REPORT. Never edit, delete, rename,
     resume, or rerun these artifacts. Do not run the one-factual or full-reference matrices.

4. Run the final offline quality pass and sweep the backlog.
   - Where: the full repository and this plan's `backlog.md`.
   - Details: run Ruff only on Python files changed from planning baseline `3b50745`; the full tree
     has 30 unrelated baseline violations. Fix rename regressions within scope. Leave the known
     legacy null-diagnostic repair issue deferred unless a separate task takes ownership.

---

## Verification

- [ ] GATE `! git grep -I -n -i dicoflex -- README.md experiments docs/plans/LESSONS.md` — no tracked active file content contains the old token; any stale alias, prose, durable lesson, fixture metadata, or command turns this red.
- [ ] GATE `! git ls-files README.md experiments | rg -i dicoflex` — no tracked active path contains the old token; any stale module, test, launcher, or config filename turns this red.
- [ ] GATE `uv run python -c "import importlib.util; assert importlib.util.find_spec('experiments.zeroshot_cf.methods.dicoflex') is None"` — the old package is absent after tracked moves and ignored-cache cleanup; a forwarding or namespace package turns this red.
- [ ] GATE `uv run pytest -q` — the complete repository suite passes with no old-name compatibility alias.
- [ ] GATE `set -euo pipefail; changed=$(git diff --name-only --diff-filter=ACMR 3b50745 -- 'experiments/**/*.py'); test -n "$changed"; git diff --name-only -z --diff-filter=ACMR 3b50745 -- 'experiments/**/*.py' | xargs -0 uv run ruff check` — every Python file changed by the rename passes static checks without expanding into unrelated baseline lint cleanup.
- [ ] GATE `HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp9_countercontex_benchmark --help` and the cardinality/name assertions from the two Stage 3 matrix dry runs — final commands resolve offline, emit only canonical method identities, and do not execute methods.
- [ ] GATE `git diff --check` — the final staged diff has no whitespace errors.
- [ ] REPORT `find experiments/zeroshot_cf/results -type f \( -name manifest.json -o -path '*dicoflex*' \) -print 2>/dev/null` — record historical ignored locations in `journal.md`; publish `NOT MEASURED` if unavailable and continue.

---

## Commit

`refactor(countercontex): remove legacy method name`
