# Stage 7: Full-Scale Routing Override (Exp9) — Resolve Backlog #4

**Goal**: Using the Stage-3 caching + beam speedup, run the HELOC classifier-routing override (force low-cardinality integer columns to the regressor/bar head) at a statistically useful sample size — baseline vs override — and report Δ validity / Δ proximity / Δ frac_oob / Δ L0, resolving predecessor Backlog #4 (only an n=1 smoke exists today).
**Dependencies**: Stage 3 DONE (`--beam` + cache — the tractability lever). Host preflight passing (DGX).

---

## Context

Predecessor Stage 8 added the `--force-numeric-cols` override (routes the 5 HELOC low-cardinality int
columns `5,6,9,10,12` through the regressor instead of the classifier head, preserving their ordered
support). The n=1 smoke showed the override kept validity 1.0 and improved proximity 0.346→0.133 /
LOF 1.86→1.08 — promising but not statistical. The full run never finished: forced-numeric routing makes
points consume their whole budget, hitting `prob_ascent`'s O(|A|²) worst case with per-query kNN fits
(predecessor Backlog #4). Stage 3's beam cap + impute cache is the fix.

---

## Steps

1. **Confirm tractability** with the Stage-3 verification smoke (`exp9 --force-numeric-cols all --beam K
   --max-test 3`) completes quickly; pick a beam `K` (e.g. 4–8) that keeps per-point cost bounded while
   preserving the override's direction observed in the smoke. Record the chosen beam.
2. **Run Exp9 at scale on the DGX** (detached, sentinel-polled):
   - HELOC, Stage-4 config (`prob_ascent` + `knn_both@256`), `--budget 17` (natural |A|).
   - Two cells: `--force-numeric-cols none` (baseline) vs `--force-numeric-cols 5,6,9,10,12` (or `all`).
   - Target `--max-test 200`; if the beamed cost still can't reach 200 in the time budget, fall back to
     the largest n that finishes (≥30) and **log the effective n** — do not fabricate rows. Use the
     **same** beam and n for both cells so the Δ is apples-to-apples.
   - MOONS is all-continuous (no misrouted columns) — it needs no override run; mention it as the null control.
3. **Regenerate artefacts**: `results/exp9_routing_heloc.csv` (baseline + override rows, effective-n and
   beam columns) and `results/exp9_routing_summary.md` — replace the n=1 smoke with the stable n result;
   state the verdict: does forcing ordered/numeric treatment improve proximity and/or validity at scale,
   or does the coarse support hurt? Remove/relabel the "bounded smoke only" caveat now that it is stable.
4. **Flip predecessor Backlog #4** → RESOLVED in `docs/plans/iterative-greedy-cf/index.md` and summarize
   the fix in its Fixed Issues (done fully in Stage 8 if cleaner to batch).
5. **Pull back and commit.**

---

## Verification

- [ ] `results/exp9_routing_heloc.csv` has baseline + override rows at the same effective n (≥30, target
      200) and beam, with Δ validity / proximity / frac_oob / L0 derivable.
- [ ] `results/exp9_routing_summary.md` reports the routing verdict at the stable n (not the n=1 smoke),
      with the beam and effective n stated.
- [ ] The override cell used the Stage-3 speedup (beam recorded); the run finished within the DGX budget.
- [ ] `--force-numeric-cols none` cell matches the current (auto-routing) behaviour at the same n.
- [ ] `true_actionability == 1.0` for both cells.
- [ ] `git diff --name-only <base>..HEAD -- src/tabpfn` empty; no `tabpfn_client` import.

---

## Expected outcomes (record actuals)

- Either forcing numeric treatment improves HELOC proximity/validity (→ a recommended preprocessing step)
  or the too-coarse support hurts (→ justifies the current auto-routing). Both are useful, reportable
  results. Record which materializes at the stable n.

---

## Commit

`feat(greedy-cf): full-scale routing override (Exp9) via beam/cache — resolves Backlog #4`
