# Plan: TabICL Distribution Sampling Experiments

**Date**: 2026-09-10
**Branch**: `paper-experiment-campaign`
**Predecessors**: [paper-experiment-campaign](../paper-experiment-campaign/index.md), [CounterContEx evaluation architecture](../counterfactual-evaluation-architecture/index.md), [E3 result diagnosis](../tabicl-empirical-result-diagnosis.md)
**Goal**: Determine whether fuller use of TabICL's conditional proposal distributions and classifier-guided sampling improves low-budget counterfactual discovery, while separately measuring how numerical and categorical proposal distributions affect the target classifier.

Executed per [PROTOCOL.md](PROTOCOL.md). Status of record: [state.json](state.json).
Runtime record: [journal.md](journal.md) · [decisions.md](decisions.md) · [backlog.md](backlog.md)

---

## Context

E3 compared TabICL and the empirical backend through nine numerical quantiles, shared sparse
search, and `k=3`. It found no broad TabICL advantage. The executed diagnostic nevertheless
showed that 2,519/2,520 raw numerical quantiles differed and that TabICL proposals responded to
other feature values. Different numerical proposals often failed to enter the valid pool, while
the `k=3` path enumerated the same categorical alternatives in both arms.

The current portable `ProposalSession` does not expose a continuous distribution. The TabICL
adapter briefly constructs a `QuantileDistribution`, then returns either configured ICDF points
or a sample. At the historical `temperature=1e-9`, the nominal sampling path is practically a
mode. Full categorical probabilities are already portable. This plan adds a backend-owned,
NumPy-portable distribution boundary; policy and classifier guidance remain method/search-owned.
The generic runner and method-blind evaluator remain unaware of TabICL.

The three experiments are frozen in [the experiment protocol](resources/experiment-protocol.md):
E11 compares grid and sampling decoders under explicit budgets; E12 measures the feature-to-
classifier pushforward and tests the categorical-versus-numerical hypothesis; E13 samples from
the classifier-guided tilt
`pi_beta(z) proportional to q_TabICL(z) * p_clf(y_target | x_-j, z)^beta`.

All scientific outcomes are exploratory REPORTs. A negative result is a valid completion. The
plan does not authorize checkpoint-backed or costly execution: Stages 7 and 8 require separate,
recorded user authorization and must block without it.

---

## Strategy

**Phase A -- contracts and instrumentation (Stages 1-3).** Freeze analytical witnesses,
reproducible RNG and budget semantics, expose portable ICDF/sampling primitives, and build a
hash-checked proposal/pushforward diagnostic without changing common evaluation metrics.

**Phase B -- experimental policies (Stages 4-6).** Add budgeted grid/IID/truncated decoders and
classifier-guided sampling inside CounterContEx search, then assign new scientific identities,
write tracked matrices and dry-run-only launchers, and add external hash sealing for canonical
campaign artifacts.

**Phase C -- evidence (Stages 7-9).** After explicit authorization, run a small mechanism pilot,
estimate cost, separately authorize frozen confirmation, and publish complete positive or negative
results with denominators, query budgets, and limitations.

---

## Success Criteria

Every row declares a **Kind**. GATE blocks the stage; REPORT is measured and published and
never blocks. **Default to REPORT** -- a row earns GATE only if you can name the deciding
command *and* its tolerance today. `If unmeasurable` must differ from `If missed`:
a measurement that returns no signal is information about the instrument, not the change.

A GATE value must be **derived from a measurement of this run's own inputs** -- never a literal, a
default, a band midpoint, or a row generated to satisfy a count. Keep the provenance row below if
any gate here is a measurement; delete it only if none are.

| Metric | Baseline | Target | Kind | If missed | If unmeasurable |
|--------|----------|--------|------|-----------|-----------------|
| Focused and full test suites | 315 tests passed after the E3 diagnostic | all existing and new contract tests pass | GATE | block owning stage | n/a |
| Every GATE value is derived from a measurement of this run's own inputs | n/a | no status is a literal, a default, a band midpoint, or a row generated to satisfy a count | GATE | block stage | REPORT `NOT MEASURED` and block |
| Historical nine-quantile compatibility | authenticated E3 trace and historical candidates exist locally | offline projection/search replay matches the committed witness exactly; after Stage-7 authorization, real adapter replay has identical availability and candidate values within absolute `1e-8` on the pinned checkpoint/host | GATE | block owning stage | REPORT `NOT MEASURED` and block |
| Distribution and guidance kernels | no portable continuous-q contract | analytical fixtures prove finite normalized categorical distributions, monotone ICDF mapping, `beta=0` recovery of the empirical q measure, and stable log-space normalization | GATE | block owning stage | n/a |
| Budget and RNG integrity | current calls can repeat RNG streams across chunks and `k=3` categories are unbounded | keyed uniforms are invariant to batch/order; every raw draw, no-op, projection, duplicate, unique oracle row, and cached classifier score is accounted once; exact common pools are required only for identical state/action keys, while adaptive runs report realized cumulative oracle work | GATE | block owning stage | REPORT `NOT MEASURED` and block |
| Architecture and scientific identity | E3 uses `countercontex-v3` / `tabicl-proposal-v1` | import-boundary tests keep policy out of the generic runner/evaluator; every policy, budget, truncation, temperature, and beta change alters the new identity while old manifests remain read-only | GATE | block Stage 6 | n/a |
| Matrix and artifact completeness | no E11-E13 specifications | each tracked specification dry-resolves to its frozen cells; every authorized run has exact membership, required payloads, hash inventory, and no partial/extra/mismatched cell | GATE | block owning run stage | REPORT `NOT MEASURED` and block |
| E11 decoder outcomes | nine-quantile E3 was mixed/negative and not budget-matched to full-q sampling | report grid-versus-IID and truncation effects at each declared budget | REPORT | publish and continue | publish `NOT MEASURED` |
| E12 categorical-versus-numerical effect | traced categorical dominance is mechanism-specific, not a general estimate | report factual-clustered effects by dataset, classifier, target direction, and action-unit type | REPORT | publish and continue | publish `NOT MEASURED` |
| E13 classifier-guidance outcomes | unmeasured | report `beta=1` versus `beta=0`, beta dose response, ESS, validity, coverage, proximity, support, and query/runtime costs | REPORT | publish and continue | publish `NOT MEASURED` |

---

## Files That May Be Changed

### CounterContEx method and backend
- `experiments/zeroshot_cf/methods/countercontex/backends/base.py` -- portable distribution and capability contracts.
- `experiments/zeroshot_cf/methods/countercontex/backends/tabicl.py` -- TabICL implementation of the new contract.
- `experiments/zeroshot_cf/methods/countercontex/config.py` -- scientific policy and budget settings.
- `experiments/zeroshot_cf/methods/countercontex/search.py` -- backend-to-search bridge and classifier-guided selection.
- `experiments/zeroshot_cf/methods/countercontex/runtime.py` and `experiments/zeroshot_cf/methods/registry.py` -- implementation identities.
- `experiments/zeroshot_cf/tabicl_sampler.py` -- controlled ICDF evaluation and batched sampling.
- `experiments/zeroshot_cf/grouped_categorical.py` and `experiments/zeroshot_cf/diverse_search.py` -- strict proposal budgets in `k=1` and the limited `k=3` replication.

### Diagnostics, analysis, and experiment definitions
- `experiments/zeroshot_cf/diagnostics/` -- proposal/pushforward trace, strict serializer, and hash inventory.
- `experiments/zeroshot_cf/analysis/` -- factual-clustered E11-E13 summaries and figures.
- `experiments/zeroshot_cf/core/contracts.py`, `experiments/zeroshot_cf/datasets/benchmark.py`, `experiments/zeroshot_cf/orchestration/spec.py`, `experiments/zeroshot_cf/orchestration/runner.py`, and matrix parsing -- partition-qualified factual selection with the historical test default preserved.
- `experiments/zeroshot_cf/orchestration/campaign_inventory.py` (or a package-conventional sibling) -- external hash inventory and strict reader for complete canonical campaign runs.
- `experiments/zeroshot_cf/configs/matrices/` -- new E11 and E13 matrices.
- `experiments/zeroshot_cf/configs/diagnostics/` -- tracked E12 diagnostic specification.
- `experiments/zeroshot_cf/athena/` -- dry-run and, only after authorization, managed launch support.
- `experiments/zeroshot_cf/tests/` -- analytical, contract, identity, trace, and matrix witnesses.
- `README.md`, `experiments/zeroshot_cf/README.md`, and `docs/research/` -- protocol and results.

### Forbidden scope
- No TabICL branch in `orchestration/runner.py` or common evaluation metric based on TabICL's own density.
- No rewrite, resume, relabel, or deletion of E3/E5 or diagnostic historical artifacts.
- No change to the empirical backend unless a separately paired comparator requires it.

---

## Stages

Routing table only. **Status, notes and commits live in `state.json` and nowhere else** --
never mirror them here. To read the current status:

```bash
jq -r '.stages[] | "\(.id)  \(.status)  \(.title)"' state.json
```

| # | Stage |
|---|-------|
| 1 | [Freeze sampling contracts and witnesses](stages/01-freeze-contracts-and-witnesses.md) |
| 2 | [Expose a portable TabICL distribution API](stages/02-portable-tabicl-distribution-api.md) |
| 3 | [Trace feature distributions through classifiers](stages/03-feature-classifier-pushforward.md) |
| 4 | [Integrate budgeted q decoders](stages/04-budgeted-q-decoders.md) |
| 5 | [Integrate classifier-guided pi-beta sampling](stages/05-classifier-guided-pi-beta.md) |
| 6 | [Freeze identities, matrices, and launchers](stages/06-identities-matrices-and-launchers.md) |
| 7 | [Run authorized diagnostics and validation pilots](stages/07-authorized-validation-pilots.md) |
| 8 | [Run authorized held-out confirmation](stages/08-authorized-heldout-confirmation.md) |
| 9 | [Analyze and publish the experiments](stages/09-analysis-and-publication.md) |
