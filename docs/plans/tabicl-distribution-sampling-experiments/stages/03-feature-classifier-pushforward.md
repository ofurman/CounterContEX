# Stage 3: Trace Feature Distributions Through Classifiers

**Goal**: Build a trace-pure, hash-checked E12 diagnostic that measures each action unit's proposal distribution and its pushforward through a fixed classifier before search.
**Dependencies**: Stage 2
**Reference**: [Frozen experiment protocol](../resources/experiment-protocol.md), E12 and Trace and result provenance.

---

## Steps

1. Extend the existing diagnostic rather than the common evaluator.
   - Where: `SearchTrace`, `SessionTrace`, `common_probe()`, and `trace_dataset()` in `experiments/zeroshot_cf/diagnostics/proposal_backends.py`, or a focused sibling module under `diagnostics/`.
   - Details: On fixed factual/common states, record every field required by the protocol from raw q representation through projection and batch classifier score. The diagnostic must not call the proposal backend or classifier twice merely to observe it.

2. Add strict, resumable diagnostic artifacts.
   - Where: the diagnostic CLI/serializer and `diagnostics/summarize_proposals.py`.
   - Details: Write metadata and columnar proposal records to a new output directory, then publish `COMPLETE.json` only after a SHA-256 inventory is complete. Reject non-finite JSON, source/spec mismatch, partial output, populated output roots, and changed classifier/case/checkpoint/context identities.

3. Implement the E12 analysis from artifact reads.
   - Where: new analysis code under `experiments/zeroshot_cf/analysis/`.
   - Details: Compute per-action-unit q-weighted delta-probability and log-odds summaries, then the factual-level categorical-minus-numerical estimand. Preserve one-hot atomicity, equal feature/group weighting, no-op conditioning, target direction, dataset and classifier strata. Use hierarchical bootstrap only as a REPORT; never treat proposals or seeds as independent observations.

4. Add synthetic witnesses and trace-purity tests first.
   - Where: `experiments/zeroshot_cf/tests/test_proposal_diagnostic.py` and a focused E12 analysis test.
   - Details: Include known categorical-dominant, numerical-dominant, tied, all-no-op, unequal-cardinality, HELOC-like no-category, reversed-target, projection-collapse, missing-row, and tracing-on/off cases.

---

## Verification

Each line is `GATE` (must pass) or `REPORT` (measure and record in `journal.md`).
A measured GATE names the input its value is read from and the defect that turns it red -- a line
that only names a command passes whenever the command exits 0.

- [ ] GATE `uv run pytest -q experiments/zeroshot_cf/tests/test_proposal_diagnostic.py` plus the focused E12 analysis tests -- real trace fixtures detect extra calls, state mismatch, cardinality weighting, denominator drift, projection/no-op loss, or corrupt inventories.
- [ ] GATE a deterministic fake-backend/fake-classifier E12 CLI run is byte-identical across repeated output roots under `SOURCE_DATE_EPOCH=0`, and source/spec mutation or a missing row makes verification fail.
- [ ] GATE architecture tests show E12 is method-specific and does not change `METRIC_SCHEMA_VERSION` or import CounterContEx into `evaluation/`.
- [ ] REPORT publish synthetic categorical-minus-numerical estimates and bootstrap behavior only as instrument checks, not research evidence.

---

## Commit

`feat(diagnostics): trace proposal classifier pushforwards`
