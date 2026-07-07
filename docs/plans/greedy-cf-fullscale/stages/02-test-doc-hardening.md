# Stage 2: Test & Doc Hardening (P2 + P3s)

**Goal**: Turn the vacuous kNN regression guard into a real one, make the Exp6 summary writer non-clobbering, and fix the predecessor plan's documentation defects (wrong Stage-1 SHA, missing SHAs, `binary_cat` circular-label caveat).
**Dependencies**: None (local, code + docs; no GPU). Independent of Stages 1 and 3.

---

## Motivation (from the post-review)

- **P2** — `tests/test_context.py:36` case (a) is a **vacuous** regression guard: it builds
  `rng.choice(...).sort()` twice inline and asserts the two are equal, never calling
  `set_context`. It always passes regardless of sampler behaviour, so the plan's most important
  "default context selection is byte-identical" guarantee is unguarded.
- **P3** — `exp6_context_ablation.py:506–519` `write_summary()` emits a placeholder verdict
  (`"_Placeholder — to be filled by the orchestrator…_"`); the committed `exp6_summary.md` was
  hand-finalized, so re-running the driver would clobber the real verdict. (exp5's writer derives
  its verdict programmatically and is safe to re-run — mirror that.)
- **P3** — the predecessor index (`docs/plans/iterative-greedy-cf/index.md`) tracker row 1 cites
  Stage-1 commit `5dfcaba`, but that hash lives on the **successor `manifold-flow-cf` branch** and
  is **not** an ancestor of this branch. The actual Stage-1 commit is `d1a5352`. Rows 2/3/4/7/8 say
  "(see git log)" instead of pinning a SHA.
- **P3** — `binary_cat` (`data.py`) sets label `Y = decision_code = actionable feature 0`, so a
  single-column flip trivially achieves `validity=1.0`; the "≈100% validity confirmed" claim is
  partly circular and only weakly stresses the classifier head. Add an honest caveat (a full
  non-circular variant is optional, see Step 4).

---

## Steps

1. **Make `test_context.py` case (a) a real regression guard.**
   - File: `experiments/zeroshot_cf/tests/test_context.py` (≈`:36–58`).
   - Rewrite `test_random_selection_regression_indices` so it **actually calls the sampler's
     random-selection path** with a fixed seed and pins the selected row indices against a
     frozen expected array. Two acceptable shapes:
     - (preferred) call `set_context(X, y, target_class=None, selection="random", max_context=k)`
       on a `ConditionalDensitySampler` (shared `models` fixture) and assert the rows entering the
       fit (or a captured `_last_context_indices`, adding a tiny debug attribute if none exists)
       match a hard-coded index array; **or**
     - if wiring the full fit is too heavy, factor the exact random-subsample logic out of
       `set_context` into a module-level helper (mirroring `_knn_indices`) and pin **that** helper's
       output — provided `set_context` then calls the same helper, so the test genuinely covers the
       production path (not a re-implementation).
   - The test must **fail** if the default random path's index selection drifts (e.g. an RNG
     reseed or reorder). Do not assert inline-computed values against themselves.

2. **Make `exp6_context_ablation.write_summary()` programmatic (no clobber).**
   - File: `experiments/zeroshot_cf/exp6_context_ablation.py:506–519`.
   - Replace the placeholder verdict with a derived one, mirroring `exp5_selector_ablation.py`'s
     `write_summary`: compute the recommended `(size, strategy)` per dataset from the grid
     (best validity, tie-break lower `frac_oob` then lower `size`), state whether larger context
     helped and whether kNN beat random at equal size, and render it. Re-running the driver must
     regenerate a correct, non-placeholder summary — never overwrite a hand-written verdict with a
     placeholder.

3. **Fix the predecessor plan's tracker SHAs.**
   - File: `docs/plans/iterative-greedy-cf/index.md` (Progress Tracker).
   - Row 1: change the Stage-1 commit reference from `5dfcaba` to **`d1a5352`** (verify with
     `git log --oneline | grep "iterative greedy CF core"`).
   - Rows with "(see git log)": backfill the actual SHAs from `git log --oneline` — Stage 2 =
     `8c6bb4d` (+ fix `7524e33`), Stage 3 = `94f8602`, Stage 4 = `d9d81a8`, Stage 7 = `910a209`,
     Stage 8 = `c6c6615`. Confirm each with `git show --stat <sha>` before writing.
   - This is a **documentation-only** edit to the predecessor plan (Decision #6); do not touch its code/results.

4. **Add the `binary_cat` circular-label caveat.**
   - File: `experiments/zeroshot_cf/results/REPORT.md` (§7d native-categorical section) **and**
     `docs/plans/greedy-cf-fullscale` Decisions if a variant is built.
   - Add one honest sentence: `binary_cat`'s label equals actionable feature 0
     (`Y = decision_code`), so `validity=1.0` is partly guaranteed by construction and only weakly
     stresses the classifier-head imputation of *other* features; the number is a sanity floor, not
     a strong claim.
   - **Optional (only if quick and low-risk)**: add a non-circular variant `binary_cat_v2` whose
     label is a deterministic function of ≥2 features (e.g. `Y = channel_code XOR context_code`)
     so no single actionable column trivially flips it, wire it exactly like `binary_cat`, and note
     it. If it risks scope creep, record it as a Backlog item instead and keep the caveat.

---

## Verification

- [ ] `uv run pytest experiments/zeroshot_cf/tests/test_context.py -q` passes, and the rewritten
      case (a) **fails** if the random-selection index logic is perturbed (sanity-check by a
      temporary local edit, then revert).
- [ ] Re-running `exp6_context_ablation.py` regenerates `exp6_summary.md` with a real
      (non-placeholder) verdict — confirm no `_Placeholder_` string remains in the writer.
- [ ] `docs/plans/iterative-greedy-cf/index.md` tracker cites `d1a5352` for Stage 1 and real SHAs
      for the "(see git log)" rows; each SHA resolves via `git show --stat`.
- [ ] `REPORT.md` §7d carries the `binary_cat` circular-label caveat.
- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` — full suite green.
- [ ] `git diff --name-only <base>..HEAD -- src/tabpfn` empty; no `tabpfn_client` import.

---

## Commit

`fix(greedy-cf): real kNN regression guard + programmatic exp6 summary + predecessor SHA/caveat fixes`
