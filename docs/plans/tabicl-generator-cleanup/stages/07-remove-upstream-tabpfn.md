# Stage 7: Remove the Upstream TabPFN Project and Repurpose the Root

**Goal**: Make the repository itself a focused home for the retained TabICL counterfactual suite by deleting upstream TabPFN code and replacing its root project surface.
**Dependencies**: Stage 6

---

## Steps

1. Convert the root project into the authoritative entry point for the isolated suite.
   - Replace the TabPFN package metadata in root `pyproject.toml` with a thin uv workspace/project
     configuration for `experiments/zeroshot_cf`; regenerate and track the root lock.
   - Root `uv sync --locked`, `uv run pytest`, lint, and supported module commands must delegate to
     or install the same locked suite proven in Stage 6 without duplicating divergent dependencies.
2. Delete upstream TabPFN implementation and validation material.
   - Remove tracked `src/tabpfn/`, root `tests/`, `examples/`, `changelog/`, `CHANGELOG.md`, and
     TabPFN-only `scripts/`.
   - Remove the ignored top-level `tabpfn/` directory; inspection at planning time found only
     bytecode/cache artifacts. This path is explicitly authorized for deletion.
3. Remove historical material that documents only the deleted suite.
   - Delete `docs/plans/zeroshot-tabpfn-cf/` and `docs/plans/iterative-greedy-cf/` after confirming
     their durable dependency lessons exist in `docs/plans/LESSONS.md` and this plan's boundary.
4. Rewrite the root-facing repository surface.
   - Replace root `README.md` and update `SECURITY.md`, `.gitignore`, pre-commit/tool settings,
     CODEOWNERS, issue/PR templates, Dependabot, and workflows for the TabICL suite.
   - Keep only CI/review automation that applies to the focused project. Remove TabPFN release,
     notebook, changelog, model-download, and package-publication workflows unless rewritten for a
     concrete retained requirement.
5. Update third-party notices for the retained code and dependencies.
   - Preserve root `LICENSE`. Do not infer a relicensing decision from cleanup.
   - If a notice obligation cannot be determined, record the exact dependency/file in the backlog;
     do not keep an upstream notice that falsely describes deleted code.
6. Do not initialize a new repository, filter history, or squash during execution. Keep the stage
   commit atomic and leave the final squash to the user after the removal is reviewed.

---

## Verification

- [ ] GATE `test ! -e src/tabpfn && test ! -e tabpfn && test ! -e tests && test ! -e examples && test ! -e changelog && test ! -e scripts && ! git ls-files | rg '^(src/tabpfn/|tests/|examples/|changelog/|scripts/|docs/plans/(zeroshot-tabpfn-cf|iterative-greedy-cf)/)'` succeeds — the filesystem and Git index are the inputs; any upstream source, cached package, test, example, script, or predecessor-plan survivor turns it red.
- [ ] GATE `uv sync --locked && uv run pytest -q` succeeds with non-zero focused-suite discovery, followed by the Stage 6 CLI and Athena checks — root metadata/lock and retained code are the inputs; a broken workspace, missing dependency, or suite regression turns it red.
- [ ] GATE `! rg -n 'src/tabpfn|TabPFNClassifier|TabPFNRegressor|tabpfn-extensions|benchmarking_tabpfn|release-(create-pr|publish|tag)' pyproject.toml README.md SECURITY.md .github .pre-commit-config.yaml THIRD-PARTY-NOTICES.md` and `rg -n 'uv (sync|run).*pytest' .github/workflows` both succeed — root project files are the inputs; stale upstream identity or CI that does not test the retained suite turns it red.

---

## Commit

`refactor(repo): remove upstream TabPFN project`
