# Stage 5: Analysis layer

**Goal**: Turn published artifacts into the paper's tables and figures programmatically, so no
reported number is ever transcribed by hand.
**Dependencies**: Stage 3

The provenance rule for the whole plan lands here: **every figure and table reads from a
published `summary.csv` or `manifest.json`.** A plotting script that accepts inline numbers is
how a literal ends up in a paper.

---

## Steps

1. **Create `experiments/zeroshot_cf/analysis/`.** It depends on `orchestration/` for artifact
   reading and on nothing else; it must not import `methods/` or branch on a method name. This
   respects the existing dependency direction — the analysis layer sits beside the CLI, not
   inside the runner.

2. **Multi-seed aggregation.** Extend or wrap `ArtifactStore.aggregate_expected()` to group cells
   by scientific identity minus seed, and emit mean, standard deviation, and n per metric.
   - Refuse to aggregate a group whose members differ in anything other than seed. The existing
     aggregation already rejects missing, extra, partial, duplicate and identity-mismatched
     cells; keep that strictness rather than relaxing it for convenience.
   - A group with fewer seeds than requested reports its actual n. Never silently average over
     three seeds and present it as five.

3. **Significance testing.** Paired Wilcoxon signed-rank across the dataset × classifier grid,
   with Holm correction across the method comparisons. Report the test statistic, the corrected
   p-value, and n — not a bare asterisk.
   - Add a Demšar critical-difference diagram over methods ranked per dataset.
   - **The noise floor from Stage 1 is the sanity check**: a "significant" difference smaller
     than the measured seed-to-seed spread must be flagged as such in the output, not reported
     as a finding.

4. **Figure builders**, one function per figure named in the positioning draft §6: F3 critical
   difference, F4 proximity-vs-confidence Pareto, F5 cost-quality Pareto, F6 the distribution of
   `p_f(y*|x')` per method, F7 the qualitative case study. F1 and F2 are hand-made and are not
   this stage's business.
   - Each builder takes an output root and a matrix config, reads artifacts, and writes both the
     figure and the CSV it was drawn from. The CSV is what makes the figure auditable.

5. **Table builders** for T1 (main k=1 with variance), T2 (k=3 diversity) and T3 (backend
   ablation), emitting LaTeX and CSV from the same aggregation.

6. **A CLI entry point**, `cli analyze --config <matrix> [--output <dir>]`, following the
   existing argument-translation-only pattern in `cli.py`.

---

## Verification

- [ ] GATE Every table and figure builder reads its values from a published artifact file — no
      builder accepts a numeric literal as input. Verify by `git grep` for hardcoded metric values
      in `analysis/`, and by confirming each builder's only data argument is a path. A builder
      that can be called with inline numbers turns it red.
- [ ] GATE Aggregating a group where one cell is missing its `COMPLETE` marker raises rather than
      averaging over the survivors — read from the aggregation call on a deliberately incomplete
      fixture directory.
- [ ] GATE The significance test on two identical inputs returns a non-significant result, and on
      a fixture with a known large separation returns a significant one — a test that reports
      significance for identical inputs is measuring nothing.
- [ ] REPORT Analysis layer run against the Stage 1 noise-floor cells — record the output in
      `journal.md`.

---

## Commit

`feat(analysis): add multi-seed aggregation, significance tests, and figure builders`
