# Stage 8: Run the Final Boundary and Behavior Audit

**Goal**: Prove from the cleaned repository that the complete retained suite is behaviorally covered, independently installable, offline-safe, and free of legacy or upstream TabPFN code.
**Dependencies**: Stage 7

---

## Steps

1. Audit the keep/extract/remove manifest in `resources/boundary.md` against the filesystem, Git
   index, source imports, CLI entry points, root metadata, CI, and test discovery. Inspect every
   match rather than suppressing broad patterns.
2. Run both supported installation paths from a clean environment.
   - Verify root `uv sync --locked`/`uv run pytest` and the suite project commands resolve the same
     retained dependency set and test corpus.
   - Confirm explicit coverage for all requested generator features, bounded-beam/DPP diversity,
     Exp9, NICE, Wachter, Growing Spheres, DiCE, FACE, datasets, metrics, and checkpoints.
3. Exercise all offline-safe entry points.
   - Import core generator and benchmark-protocol modules.
   - Run fake single/diverse generator smoke tests, Exp9 aggregation from fixtures, all `--help`
     commands, compile checks, Athena syntax/case validation, and root CI-equivalent checks.
4. Attempt real-model smoke only when both staged checkpoints pass checksum validation.
   - Run the conditional smoke and a one-row generator smoke covering quantiles, confidence,
     revisits, and single-CF refinement.
   - If absent, record `NOT MEASURED` with exact paths. Never download during the offline audit.
5. Confirm local-state and licensing guardrails, then sweep `backlog.md`.
   - Preserve `ARCHITECTURE.md`, datasets, vendor tree, checkpoints, models, outputs, and caches
     outside the explicitly deleted top-level `tabpfn/` path.
   - Confirm CEL remains pinned and operational. Confirm the branch contains ordinary atomic stage
     commits and no repository initialization, history filtering, or executor-side squash occurred.

---

## Verification

- [ ] GATE root and suite-project locked sync/test commands both succeed with non-zero identical retained test discovery — clean manifests, locks, tests, fixtures, and production modules are the inputs; dependency or behavior defects turn it red.
- [ ] GATE the final keep/remove/import audit finds every required entry point and no removed experiment, upstream TabPFN path/import, obsolete plan, or root metadata reference — the filesystem, `git ls-files`, and retained sources/configs are the inputs; a missing retained concern or survivor turns it red.
- [ ] GATE compile checks, all offline `--help` commands, fake single/diverse smoke, Exp9 fixture aggregation, Athena checks, and CI-equivalent checks pass — cleaned source/config assets are the inputs; syntax, import, schema, eager-network, shell, or automation defects turn it red.
- [ ] REPORT Record real-checkpoint smoke result/runtime/validated paths, retained CEL revision, notice review, preserved untracked/ignored state, and the cleanup commit range ready for the user's squash; absent weights are `NOT MEASURED` and local user files remain untouched.

---

## Commit

`chore(tabicl-cf): complete repository isolation audit`
