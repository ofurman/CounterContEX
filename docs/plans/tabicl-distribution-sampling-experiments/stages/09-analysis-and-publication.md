# Stage 9: Analyze and Publish the Experiments

**Goal**: Produce reproducible E11-E13 tables, figures, and an evidence-bounded research report from authenticated artifacts.
**Dependencies**: Stage 8
**Reference**: [Frozen experiment protocol](../resources/experiment-protocol.md), Interpretation constraints.

---

## Steps

1. Build all products from strict artifact readers.
   - Where: new E11-E13 builders under `experiments/zeroshot_cf/analysis/`.
   - Details: Verify the external `campaign-inventory-v1` before reading any canonical E11/E13 value and verify E12 `COMPLETE.json` inventories. Never transcribe metrics into source. Regenerate CSV/LaTeX/figures plus a manifest of input/output hashes under `docs/papers/campaign-artifacts/` or a clearly separate follow-up artifact directory.

2. Analyze E11 with its declared comparisons and budgets.
   - Where: grid/IID/truncation summaries from canonical final candidates and method accounting arrays.
   - Details: Report grid-9 versus IID-9 and grid-49 versus IID-49 first, then budget and truncation curves. Coverage/validity denominators, raw/effective proposals, oracle rows, runtime, Gower, action units, actionability, bounds, and neighbour support travel with every result.

3. Analyze E12 at factual-cluster level.
   - Where: authenticated proposal-diagnostic records.
   - Details: Compute the predeclared D_i and secondary effects by dataset, classifier, target direction, and action type. Average within factual before inference; use hierarchical bootstrap as a REPORT. Disclose HELOC exclusion from paired action-type effects and LR/one-hot/cardinality/projection limitations.

4. Analyze E13 and k=3 separately.
   - Where: canonical results and pi-beta accounting artifacts.
   - Details: Lead with beta=1 versus beta=0; show dose response, ESS, unique fraction, validity/coverage, proximity/support, oracle work and runtime. State that the same classifier guides and judges success. Do not interpret k=3 as an independent replication or pool it with k=1/E3.

5. Publish an accurate research narrative.
   - Where: new dated report under `docs/research/`, `docs/papers/campaign-results.md` if this follow-up belongs in the campaign trace, and relevant README links.
   - Details: State whether each hypothesis is supported, unsupported, heterogeneous, or not measured. The maximum justified claim concerns the frozen decoder/guidance interfaces, not full TabICL distribution quality, causal actionability, or independent model robustness. Record exact commands, specs, versions, roots, hashes, denominators, failures, and compute.

6. Run final reproducibility gates and backlog sweep.
   - Where: full repository tests, analysis regeneration into a fresh temporary directory, plan journal/state/backlog.
   - Details: Byte-compare regenerated products; review B-1/B-2 without silently resolving them; add only newly learned cross-plan facts to `docs/plans/LESSONS.md`.

---

## Verification

Each line is `GATE` (must pass) or `REPORT` (measure and record in `journal.md`).
A measured GATE names the input its value is read from and the defect that turns it red -- a line
that only names a command passes whenever the command exits 0.

- [ ] GATE strict readers accept exactly the frozen E11-E13 inventories and all published numbers are regenerated from named authenticated inputs; a missing marker, identity mismatch, or hard-coded metric turns loading/regeneration red.
- [ ] GATE a fresh `SOURCE_DATE_EPOCH=0` rebuild is byte-identical to committed products, and focused analysis tests plus full pytest, Ruff, dry-runs, offline CLI, and `git diff --check` pass.
- [ ] GATE narrative audit confirms every metric carries its denominator/orientation and every E11/E12/E13/k=3 claim carries the protocol limitation that changes its interpretation; an unlabeled pooled or causal/full-distribution claim turns the review red.
- [ ] REPORT publish all scientific outcomes, uncertainty, failures, and phase timings whether positive, null, heterogeneous, or negative.

---

## Commit

`docs(research): report TabICL distribution experiments`
