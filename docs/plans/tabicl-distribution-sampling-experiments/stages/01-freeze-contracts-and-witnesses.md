# Stage 1: Freeze Sampling Contracts and Witnesses

**Goal**: Implement the backend-neutral mathematical, RNG, and accounting primitives against independent analytical and authenticated historical witnesses.
**Dependencies**: None
**Reference**: [Frozen experiment protocol](../resources/experiment-protocol.md), especially Shared definitions, Decoder semantics, and Budget and RNG contract.

---

## Steps

1. Read predecessor evidence before editing.
   - Where: E3/E5 in `docs/papers/campaign-results.md`; D-5/D-10/D-11 in `docs/plans/paper-experiment-campaign/decisions.md`; the final lesson in `docs/plans/LESSONS.md`; and `docs/research/2026-09-08-tabicl-empirical-diagnostic.md`.
   - Details: Preserve the distinction between proposal mechanics, search behavior, and final evaluation. Do not modify historical artifacts.

2. Add a pure NumPy proposal-policy kernel owned by CounterContEx.
   - Where: new `experiments/zeroshot_cf/methods/countercontex/proposal_policy.py` (final name may follow the package convention).
   - Details: Define immutable validated records for raw draws, projected/duplicate linkage, budget counts, the fixed 256-bin quantile approximation, stable keyed uniforms, truncation masks, and log-space normalized weights. Hash canonical bytes with a stable digest; never use Python `hash()` or ambient RNG.

3. Create independent fixtures before implementing each behavior.
   - Where: new `experiments/zeroshot_cf/tests/test_proposal_sampling_policy.py` and `experiments/zeroshot_cf/tests/fixtures/proposal_sampling/`.
   - Details: Hand-derive small categorical and piecewise-continuous distributions with known ICDF, interval mass, greedy/top-k/top-p support, and pi-beta weights. Add an authenticated minimal nine-quantile witness derived from the preserved E3 trace, carrying source hashes and provenance rather than copying a new implementation output.

4. Freeze accounting and RNG invariants.
   - Where: the same policy test and existing projection helpers in `candidate_domains.py`.
   - Details: Exercise clipping, support snapping, no-op, duplicates, zero/tiny classifier probabilities, beta zero, ordering/chunking changes, reversed targets, one-hot units, and immutable units. Every raw draw must map to exactly one terminal disposition and at most one cached unique classifier row.

---

## Verification

Each line is `GATE` (must pass) or `REPORT` (measure and record in `journal.md`).
A measured GATE names the input its value is read from and the defect that turns it red -- a line
that only names a command passes whenever the command exits 0.

- [ ] GATE `uv run pytest -q experiments/zeroshot_cf/tests/test_proposal_sampling_policy.py`, scoped Ruff, and `git diff --check` pass -- the tests read independent analytical fixtures and the authenticated E3 witness; wrong normalization, `q^2` weighting, unstable RNG, lost draws, or incorrect duplicate/no-op accounting turns a named case red.
- [ ] REPORT record the historical witness paths/hashes, analytical fixture definitions, and baseline raw/projected/unique counts in `journal.md`.

---

## Commit

`feat(countercontex): freeze proposal sampling contracts`
