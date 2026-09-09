# Stage 2: Expose a Portable TabICL Distribution API

**Goal**: Make TabICL continuous ICDF and log-probability evaluation available through the proposal-session boundary without leaking Torch or TabICL objects.
**Dependencies**: Stage 1
**Reference**: [Frozen experiment protocol](../resources/experiment-protocol.md), Decoder semantics.

---

## Steps

1. Extend the portable backend contract explicitly.
   - Where: `ProposalSession`, `ProposalCapabilities`, and `CategoryProposals` in `experiments/zeroshot_cf/methods/countercontex/backends/base.py`.
   - Details: Add validated NumPy-only batched primitives that accept caller-provided uniforms/quantile levels, return numerical values, and evaluate log probability at those values for the fixed D-1 grid. Declare capability support; do not use dynamic `hasattr()` probes. Leave empirical capabilities unchanged unless a conformance adapter is necessary.

2. Adapt TabICL at its owning layer.
   - Where: `TabICLProposalSession` and `PreparedTabICLBackend` in `methods/countercontex/backends/tabicl.py`; `TabICLConditionalDensitySampler.sample_candidates_batch()`, `sample_candidate_grid_batch()`, and `_LocalCheckpointTabICLUnsupervised._sample_numerical()` in `tabicl_sampler.py`.
   - Details: Evaluate ICDF and log probability at caller inputs without exporting `QuantileDistribution`. Preserve batching across state/feature pairs. Remove dependence on the class-level quantile hook if the new explicit path can replace it safely; otherwise scope and restore it with a concurrency guard.

3. Make stochastic behavior independent of upstream RNG resets.
   - Where: the new session method and the Stage-1 keyed-uniform helper.
   - Details: One logical draw key must return the same value regardless of call order or chunking. Factual sessions remain serial because the prepared TabICL sampler is mutable; encode this operational requirement in tests/docs.

4. Preserve the historical path.
   - Where: existing `propose_numerical*()` methods and tests in `test_tabicl_backend.py` and `test_countercontex_proposal_backends.py`.
   - Details: Existing nine-quantile specs must retain their call shape and offline projection/search replay. Exact real-adapter compatibility is deferred to the authorized Stage-7 checkpoint replay. Unsupported full-q modes must fail during configuration/preparation, before search.

---

## Verification

Each line is `GATE` (must pass) or `REPORT` (measure and record in `journal.md`).
A measured GATE names the input its value is read from and the defect that turns it red -- a line
that only names a command passes whenever the command exits 0.

- [ ] GATE `HF_HUB_OFFLINE=1 uv run --locked pytest -q experiments/zeroshot_cf/tests/test_tabicl_backend.py experiments/zeroshot_cf/tests/test_countercontex_proposal_backends.py experiments/zeroshot_cf/tests/test_proposal_sampling_policy.py` -- fake estimators and analytical fixtures detect changed ICDF, probability support, batching, capability, seed, or serialization behavior; the suite also reads stored authenticated raw proposals and makes projection/search drift in the nine-quantile offline replay red without claiming to execute the real adapter.
- The focused GATE must replay the authenticated nine-quantile trace exactly, including NaN/availability and projected values; exact real-adapter compatibility belongs to Stage 7.
- The focused GATE must exercise the full-q API through a mixed-representation codec as well as a numerical-only case; original action-unit columns must map to the same compact columns and values as the existing grid path.
- Architecture-boundary coverage is part of the focused GATE above: portable modules contain no Torch/TabICL type and the generic runner does not import the backend; a concrete dependency edge must turn a named test red.
- [ ] REPORT record batch sizes, imputation-call counts, and any unavoidable mutable-session constraint in `journal.md`.

---

## Commit

`feat(countercontex): expose portable TabICL distributions`
