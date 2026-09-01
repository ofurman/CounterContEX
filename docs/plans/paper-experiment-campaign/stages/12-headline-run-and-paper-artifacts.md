# Stage 12: Headline run and paper artifacts

**Goal**: Execute the frozen 1,000-factual run and build every paper table and figure from
published artifacts.
**Dependencies**: Stages 5, 7, 8, 9, 10, 11

---

## Steps

1. **Execute E10 once** (`campaign_e10_headline.yaml`, n=1000, six datasets, the configuration
   frozen at the end of Stage 10, logistic-regression target for continuity with the historical
   24-cell run).
   - This must not be a run that configurations were also selected on. Stage 10 froze them; do
     not revise the configuration after seeing this stage's output. If a revision proves
     necessary, it is disclosed selection and the disclosure text drafted in Stage 9 applies.
   - E1's five seeds at n=250 are the statistically stronger evidence. E10 exists for continuity
     with the published run and for the qualitative examples. If cost forces a cut, **cut E10,
     not E1** — record that decision rather than silently reducing E1's seeds.

2. **Build every table and figure** through the Stage 5 analysis layer: T1, T2, T3 and F3–F7. F1
   (the NICE-versus-conditional-proposal mechanism diagram) and F2 (the differentiation table)
   are hand-made from the positioning draft and are not built here.

3. **Produce the qualitative case study (F7).** One HELOC and one Adult factual: the row, the
   returned counterfactuals, which features moved, and the same two baselines for contrast. Read
   from `candidates.csv` and `points.csv`; inverse-transform to original feature units so the
   reader sees real values, not MinMax-scaled ones.

4. **Update documentation** to match what was actually run: `README.md` (supported benchmark,
   classifier families, datasets), `experiments/zeroshot_cf/README.md` (metric semantics for the
   v2 additions, extension procedure for a second foundation backend),
   `docs/countercontex-method.md` (§9 reference protocol, §11 hypotheses now answered — record
   the answers, including any that came out negative).

5. **Write the results summary** to `docs/papers/campaign-results.md`: every experiment, its
   configuration, its artifact paths, its measured outcome, and its denominators. This is the
   document the paper is written from, and the one that makes each number traceable.

6. **Sweep the backlog.** Close what this plan resolved; carry forward B12 (directional
   constraints), B13 (multiclass), B14 (formal statement) and B15 (human evaluation), which are
   out of scope here. Add the durable cross-plan facts to `docs/plans/LESSONS.md` — one line each.

---

## Verification

- [ ] GATE Every value in every generated table and figure traces to a published artifact file.
      Verify by regenerating all outputs from artifacts alone in a clean directory and diffing
      against the committed versions. A value that survives only in a committed file and cannot be
      regenerated turns it red — that is the literal-in-the-paper failure mode this gate exists
      to catch.
- [ ] REPORT Total measured campaign GPU-hours from manifest phase timings, against the ~120 h
      estimate; and the answer recorded for each of the four hypotheses in
      `docs/countercontex-method.md` §11 — record in `journal.md`.

---

## Commit

`docs: publish campaign results, paper artifacts, and updated protocol documentation`
