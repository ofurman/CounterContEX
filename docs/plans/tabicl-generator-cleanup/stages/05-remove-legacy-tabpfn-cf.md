# Stage 5: Remove Legacy TabPFN Counterfactual Experiments

**Goal**: Delete the original TabPFN Exp1–6 counterfactual suite and its artifacts after proving that no retained generator, benchmark, baseline, or test depends on it.
**Dependencies**: Stage 4

---

## Steps

1. Delete legacy runners and runtime listed under **Remove** in `resources/boundary.md`:
   Exp1–6, `refine.py`, `checkpoints.py`, `sampler.py`, `smoke_test.py`, and `greedy.py`.
2. Delete legacy assembly and committed outputs: notebook builder, notebook, sweep config,
   `results/REPORT.md`, and tracked Exp1–6 CSV/Markdown/PNG files.
3. Delete legacy-only tests and the real-TabPFN fixture.
   - Retain the projection, quantile, metric, and data assertions migrated in Stages 1–4.
   - Remove only legacy parts of mixed test files; never weaken a retained assertion to make the
     deletion pass.
4. Prune Exp2/4-only functions from `metrics_harness.py`, unused imports from `data.py` and
   `discriminator.py`, and old documentation references in retained source docstrings.
5. Remove `-e .`, `tabpfn-extensions`, matplotlib, and other now-unused legacy entries from the
   suite requirement source. Stage 6 owns the final project/lock migration.
6. Update `.gitignore` only for obsolete tracked-suite patterns. Do not delete ignored vendor,
   checkpoint, dataset, model, result, or cache files from the working tree.

---

## Verification

- [ ] GATE every tracked path classified **Remove** in `resources/boundary.md` is absent from `git ls-files` — the Git index is the input; any legacy survivor turns it red.
- [ ] GATE `! rg -n 'TabPFNUnsupervisedModel|tabpfn_extensions|experiments\.zeroshot_cf\.(exp[1-6]_|greedy|sampler|checkpoints)' experiments/zeroshot_cf --glob '*.py'` — retained Python sources are the input; stale imports turn it red.
- [ ] GATE the complete retained generator/diversity/benchmark/baseline/data/metrics/checkpoint test manifest passes — production files and frozen contract fixture are the inputs; deletion of a required behavior turns it red.
- [ ] GATE `git diff --exit-code e80925d -- src/tabpfn tabpfn tests examples pyproject.toml` succeeds — upstream paths are the input; out-of-scope root edits turn it red.
- [ ] REPORT `git status --short --ignored` records local user state without deleting it; unexpected files are published and preserved.

---

## Commit

`refactor(tabicl-cf): remove legacy TabPFN experiments`
