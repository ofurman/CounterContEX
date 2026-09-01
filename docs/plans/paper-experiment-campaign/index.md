# Plan: paper-experiment-campaign

**Date**: 2026-09-01
**Branch**: `paper-experiment-campaign` (off `main` @ `e1195ce`)
**Predecessors**: [counterfactual-evaluation-architecture](../counterfactual-evaluation-architecture/index.md), [countercontex-method-rename](../countercontex-method-rename/index.md)
**Goal**: Produce the complete experimental evidence base for a CounterContEx NeurIPS/ICLR
submission — three classifier families, six datasets, five seeds, all ten planned
experiments — under one frozen evaluation version, with every reported number traceable to a
published artifact.

Executed per [PROTOCOL.md](PROTOCOL.md). Status of record: [state.json](state.json).
Runtime record: [journal.md](journal.md) · [decisions.md](decisions.md) · [backlog.md](backlog.md)

---

## Context

The positioning analysis in [`docs/papers/positioning-draft.md`](../../papers/positioning-draft.md)
identified six gaps that individually justify rejection, and four more that reviewers expect.
This plan executes the experimental programme that closes them. That document is the source
for every baseline number quoted here; do not re-derive them.

What the current evidence is: one target classifier (logistic regression), one seed, four
datasets, zero executed ablations, and a headline table where CounterContEx ran at `k=3`
against `k=1` baselines. What it must become: `LR + MLP + XGBoost` × six datasets × five seeds,
a `k=1` head-to-head on `primary_*` metrics, a `k=3` comparison against a DiCE that can
actually return sets, and every ablation the method document lists as an untested hypothesis.

**Three findings from the planning session shape the stage order:**

1. **`evaluation_version` is part of scientific identity** (`orchestration/spec.py`). Any
   evaluator change after a run makes that run un-aggregatable with later ones. Therefore
   **every metric addition lands in Stage 3, before the first expensive run**, in a single
   bump to `countercontex.evaluation.v2`. There is no second bump in this plan.
2. **Lending Club is 75% of the campaign's GPU cost** — 27.52 s/factual against HELOC's 2.61 s,
   on fewer columns than Give Me Some Credit. Stage 1 profiles it before the campaign starts,
   because a fix there roughly halves the total budget. See
   [resources/compute-budget.md](resources/compute-budget.md).
3. **Much less code is needed than expected.** The matrix already expands over seeds (but
   **not** over target models — `orchestration/matrix.py` parses `target_model` as a single
   mapping, so Stage 2 adds that axis); `primary_*`
   metrics for the `k=1` comparison already exist; `discriminator.py` already has an MLP arm;
   and `datasets/cel.py` is dataset-generic, deriving action schemas from CEL's own
   `config/datasets/*.yaml`. The real code work is the target-model registry, DiCE at `k>1`,
   the new metrics, and an analysis layer.

**Scope decisions taken with the user**: full E1–E10 programme; runs execute on the DGX
`gx10-bdc5` node; classifier families `LR + MLP + XGBoost`; datasets are the current four plus
Adult and German Credit.

---

## Strategy

**Phase A — enabling infrastructure (Stages 1–6).** Everything that changes scientific identity
lands here, verified by contract tests, one-factual runs, and matrix dry-runs. No expensive
run starts until Stage 6 resolves every campaign matrix without executing a method. Phase A is
local; only Stage 1's profiling touches the DGX.

**Phase B — the campaign (Stages 7–11).** Each stage owns one or more of E1–E10, runs it on the
DGX under a fixed output root, and publishes complete aggregates. Stages 7–10 are independent of
each other and all depend only on Stage 6, so a blocked campaign stage never blocks its siblings.

**Phase C — the deliverable (Stage 12).** The frozen headline run, then every paper table and
figure built programmatically from published artifacts.

**Two protocol commitments that hold across all of Phase B**, both already repository contracts:

- Configurations are frozen at the end of Stage 10. Stage 12's headline run must not be a run
  that configurations were also selected on.
- No scientific outcome is a GATE anywhere in this plan. Whether CounterContEx beats a
  baseline, or whether TabICL beats the empirical backend, is the research question — gating on
  it would make the plan unable to record a negative result. The gates are about whether the
  experiments ran correctly and completely.

---

## Success Criteria

Every row declares a **Kind**. GATE blocks the stage; REPORT is measured and published and
never blocks. A GATE value must be **derived from a measurement of this run's own inputs** —
never a literal, a default, a band midpoint, or a row generated to satisfy a count.

| Metric | Baseline | Target | Kind | If missed | If unmeasurable |
|--------|----------|--------|------|-----------|-----------------|
| `uv run pytest -q` | 188 tests green | green, and new contract tests added by Stages 2–5 included | GATE | block stage | n/a |
| `uv run ruff check` on changed packages | clean | clean | GATE | block stage | n/a |
| Every GATE value is derived from a measurement of this run's own inputs | n/a | no status is a literal, a default, a band midpoint, or a row generated to satisfy a count | GATE | block stage | REPORT `NOT MEASURED` and block |
| Campaign aggregation completeness | n/a | for every campaign matrix, `cli aggregate` returns exactly the expected cell count recorded in Stage 6, minus only cells whose method/target-model combination Stage 2 recorded as an expected clean failure; no missing, extra, partial, duplicate, or identity-mismatched cell | GATE | block the owning stage | REPORT `NOT MEASURED` and block |
| Evaluation version count across the campaign | `v1` | exactly one version, `countercontex.evaluation.v2`, across every Phase B manifest | GATE | block stage | REPORT `NOT MEASURED` and block |
| Run determinism | unmeasured | re-running one campaign cell reproduces its `run_id` and every deterministic `summary.csv` column exactly | GATE | block Stage 1 | REPORT `NOT MEASURED` and block |
| Seed-to-seed noise floor for `proximity_grouped_gower` | unmeasured | measured in Stage 1 over 5 seeds; the value sets tolerance bands for every later numeric comparison | REPORT | publish + continue | publish `NOT MEASURED`; later stages then report point values with no band and say so |
| Lending Club per-factual cost | measured 27.52 s at n=1000, LR, k=3 — 10× HELOC on fewer columns; owned by Stage 1 | root cause identified and recorded; a fix is optional | REPORT | publish + open backlog | publish `NOT MEASURED` + backlog |
| Total campaign GPU-hours | estimated ~120 h | recorded from manifest phase timings | REPORT | publish + continue | publish `NOT MEASURED` |
| E1–E10 scientific outcomes | see positioning draft | recorded as measured, whichever way they land | REPORT | publish + continue | publish `NOT MEASURED` |

The first four rows are **global**: they apply to every stage and are deliberately not repeated
in the stage briefs. A stage's own verification list carries only what is specific to it.

---

## Files That May Be Changed

- `experiments/zeroshot_cf/orchestration/runner.py` — replace the hard-coded target-model check with a registry lookup
- `experiments/zeroshot_cf/orchestration/spec.py` — target-model identity, evaluation version constant
- `experiments/zeroshot_cf/orchestration/matrix.py` — target-model axis in matrix expansion; the schema-version decision is recorded in `decisions.md`
- `experiments/zeroshot_cf/datasets/benchmark.py`, `datasets/cel.py` — target-model construction; Adult / German Credit support if the feasibility check finds a gap
- `experiments/zeroshot_cf/discriminator.py` — XGBoost arm alongside the existing LR and MLP arms
- `experiments/zeroshot_cf/evaluation/` — new metrics and the single `v2` schema bump
- `experiments/zeroshot_cf/methods/dice.py`, `methods/base.py` — DiCE at `k>1`
- `experiments/zeroshot_cf/methods/countercontex/backends/` — a second foundation backend (Stage 11)
- `experiments/zeroshot_cf/analysis/` — new package: multi-seed aggregation, significance tests, figure builders
- `experiments/zeroshot_cf/configs/matrices/` — campaign matrices
- `experiments/zeroshot_cf/athena/` or a new `dgx/` directory — launch scripts
- `experiments/zeroshot_cf/tests/` — contract tests for everything above
- `README.md`, `experiments/zeroshot_cf/README.md`, `docs/countercontex-method.md` — protocol and metric documentation
- **Never**: `experiments/zeroshot_cf/results/local/full_reference/` and `architecture_full_reference/` are historical evidence and are read-only

---

## Stages

Routing table only. **Status, notes and commits live in `state.json` and nowhere else.**

```bash
jq -r '.stages[] | "\(.id)  \(.status)  \(.title)"' state.json
```

| # | Stage | Phase |
|---|-------|-------|
| 1 | [Feasibility and noise floor](stages/01-feasibility-and-noise-floor.md) | A |
| 2 | [Target-model registry](stages/02-target-model-registry.md) | A |
| 3 | [Evaluation metrics v2](stages/03-evaluation-metrics-v2.md) | A |
| 4 | [Diverse baseline adapters](stages/04-diverse-baseline-adapters.md) | A |
| 5 | [Analysis layer](stages/05-analysis-layer.md) | A |
| 6 | [Matrices and DGX launchers](stages/06-matrices-and-dgx-launchers.md) | A |
| 7 | [E1 main comparison](stages/07-e1-main-comparison.md) | B |
| 8 | [E2/E3 diverse sets and backend ablation](stages/08-e2-e3-diverse-and-backend.md) | B |
| 9 | [E4 confidence and threshold Pareto](stages/09-e4-confidence-tau-pareto.md) | B |
| 10 | [E5–E7 ablations and cost Pareto](stages/10-e5-e7-ablations.md) | B |
| 11 | [Second foundation backend and robustness](stages/11-tabpfn-backend-and-robustness.md) | B |
| 12 | [Headline run and paper artifacts](stages/12-headline-run-and-paper-artifacts.md) | C |

## Resources

- [compute-budget.md](resources/compute-budget.md) — measured per-factual costs and the campaign estimate
- [experiment-catalog.md](resources/experiment-catalog.md) — E1–E10 mapped to stages, matrices, and output roots
- [dgx-runbook.md](resources/dgx-runbook.md) — provisioning, environment, and the detached launch pattern
