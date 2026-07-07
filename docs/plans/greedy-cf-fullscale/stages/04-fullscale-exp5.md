# Stage 4: Full-Scale Selector Ablation (Exp5)

**Goal**: Re-run the `prob_ascent` vs `class_divergence` selector ablation at statistically-stable sample sizes (HELOC `--max-test 200`, MOONS full test split) so the winner is decided at a publishable CI rather than the noisy n=50.
**Dependencies**: Host preflight (see `resources/commands.md`) passing on the DGX. Benefits from Stage 3 (`--beam`) but the exhaustive default is fine here. Independent of the other Phase-B stages.

---

## Context

Predecessor Exp5 ran HELOC at `--max-test 50` (validity granularity coarse). Result: `prob_ascent`
won decisively (HELOC validity 0.90 / L0 1.67 / frac_oob 0.04 / fail 0.10 vs `class_divergence`
0.52 / 14.27 / 0.08 / 0.48). MOONS: 0.70 vs 0.64. This stage confirms the winner at n≈200 so the
selector decision (used by Stages 5–7) rests on a tight number.

---

## Steps

1. **Run Exp5 at full scale on the DGX** (detached, per `resources/commands.md`):
   - MOONS: full test split (no `--max-test`, or set it to the full split size and log it).
   - HELOC: `--max-test 200`.
   - Keep `--max-test`, temperature, and `--n-permutations` identical across the two selectors
     within a dataset (the ablation's fairness constraint). Log every held value and the effective n.
   - Optional: pass `--beam` only if runtime demands it; if used, apply the **same** beam to both
     selectors within a dataset and record it (a beam changes the method, so it must be symmetric).
2. **Regenerate artefacts**: `results/exp5_selector_{moons,heloc}.csv` (2 rows each, one per selector,
   with the effective-n column) and `results/exp5_summary.md` (programmatic verdict; names the winner
   per metric and the single chosen downstream selector, with the context-scope caveat).
3. **Pull results back and commit** from the local machine (the DGX pushes or the local machine pulls
   the CSVs/summary), per the predecessor DGX workflow (Decision #4 / memory `dgx-remote-experiments`).

---

## Verification

- [ ] `results/exp5_selector_{moons,heloc}.csv` regenerated with HELOC effective n≈200, MOONS full;
      effective n logged in the CSV and summary.
- [ ] Within each dataset, held values (`max_context`, temperature, `n_permutations`, beam if any) are
      identical across the two selector rows.
- [ ] `results/exp5_summary.md` names the winning selector at the new n and states the CI is tighter
      than the predecessor n=50; the chosen downstream selector is recorded (expected `prob_ascent`).
- [ ] `true_actionability == 1.0` for both rows.
- [ ] `git diff --name-only <base>..HEAD -- src/tabpfn` empty; no `tabpfn_client` import.

---

## Expected outcomes (record actuals)

- `prob_ascent` remains the winner; the HELOC validity number lands near 0.90 with a tighter CI. If
  the full-scale value differs materially from the n=50 estimate, that itself is a reportable finding
  (record it, do not chase it).

---

## Commit

`feat(greedy-cf): full-scale selector ablation (Exp5, HELOC n=200 / MOONS full)`
