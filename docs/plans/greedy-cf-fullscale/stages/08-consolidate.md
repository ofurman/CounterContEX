# Stage 8: Full Test Suite + Consolidated Table + REPORT + Memory

**Goal**: Run the **entire** pytest suite (not the bounded per-stage subsets), rebuild `summary_table.{md,csv}` and the REPORT follow-up section from the new full-scale CSVs, flip predecessor Backlog #4 → RESOLVED, and refresh the two NeoCortex memories so the next session starts from the tight-CI baselines.
**Dependencies**: Stages 4–7 DONE (their full-scale CSVs are the inputs). If any Phase-B stage is BLOCKED, build the table from whatever is available and mark the gap explicitly (no fabricated cells).

---

## Steps

1. **Run the full test suite (the "full test sets" requirement).**
   - `uv run pytest experiments/zeroshot_cf/tests -q` — the **entire** suite in one run, not the
     per-file bounded subsets used during development. All tests must pass (Fixed Issue #3 of the
     predecessor: `poetry` is unavailable on the host — `uv` is the sanctioned runner; record if so).
   - Log the pass count.

2. **Rebuild the consolidated table from the full-scale CSVs.**
   - File: `experiments/zeroshot_cf/results/summary_table.{md,csv}`.
   - One row per (dataset, config) sourced from the **new** Exp5/Exp6/Exp7/Exp9 CSVs: MOONS + HELOC at
     the (possibly revised) recommended configs, the discrete `binary_cat` row, the best measured
     budget cell, and the stable Exp9 override row. Keep the one-pass baseline contrast row.
   - Columns unchanged incl. **proximity as a first-class column**; add/keep an **effective-n** column so
     every cell's sample size is visible. Every value must trace to a source CSV — no fabrication;
     if a Phase-B stage was bounded/deferred, mark the cell and log the effective n.

3. **Update `REPORT.md`.**
   - Refresh §7c/§7d/§7e (and §8 follow-up) with the full-scale numbers: the tighter-CI selector and
     context verdicts, the **measured** budget-vs-validity curve (replacing the copied-rows narrative),
     the stable routing verdict, and the `binary_cat` circular-label caveat (from Stage 2).
   - Update the 30-second next-meeting digest bullets to the new n and numbers.

4. **Resolve predecessor Backlog #4** in `docs/plans/iterative-greedy-cf/index.md`: flip #4 →
   RESOLVED, summarize the beam/cache fix + stable Exp9 result in its Fixed Issues (documentation-only
   edit to the predecessor plan).

5. **Update memory.**
   - `iterative-greedy-cf-results`: replace the n=15/50 headline numbers with the full-scale (n≈200)
     values and note the P1/P2 corrections landed.
   - `dgx-remote-experiments`: add the full-scale runtimes actually observed (Exp5/6/7/9 at n≈200) so
     the next planner sizes jobs correctly. Add the `--beam`/cache speedup as a recorded lever.
   - Update `MEMORY.md` pointers if any hook/line changed.

6. **Guardrail sweep + final commit.**

---

## Verification

- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` — the **entire** suite passes in one run; pass
      count logged.
- [ ] `results/summary_table.{md,csv}` regenerated from the full-scale CSVs with proximity **and** an
      effective-n column; every value traces to a source CSV (spot-check 2–3); no fabricated cells.
- [ ] `results/REPORT.md` reflects the full-scale, measured numbers and the `binary_cat` caveat; the
      budget-vs-validity narrative no longer relies on copied rows.
- [ ] Predecessor `docs/plans/iterative-greedy-cf/index.md` Backlog #4 = RESOLVED with a Fixed-Issues entry.
- [ ] Both memories updated with full-scale baselines + observed runtimes.
- [ ] `git diff --name-only <base>..HEAD -- src/tabpfn` empty; `grep -rn "tabpfn_client" experiments/zeroshot_cf` finds nothing.

---

## Commit

`docs(greedy-cf): full-scale consolidated table + REPORT + Backlog #4 resolved + memory refresh`
