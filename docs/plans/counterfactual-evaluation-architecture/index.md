# Plan: Isolate Counterfactual Methods from Evaluation

**Date**: 2026-08-31
**Branch**: `cleanup/tabicl-suite`
**Planning baseline**: `4f87c2a`
**Predecessor**: [`tabicl-generator-cleanup`](../tabicl-generator-cleanup/index.md)
**Goal**: Make datasets, counterfactual methods, evaluation, and experiment orchestration independent layers so new baselines and DiCoFlex foundation-model ablations require only a method or backend adapter.

Executed per [PROTOCOL.md](PROTOCOL.md). Status of record: [state.json](state.json).
Runtime record: [journal.md](journal.md) · [decisions.md](decisions.md) · [backlog.md](backlog.md)

---

## Context

The cleanup plan isolated useful low-level modules, but the five numbered benchmark runners still
own the whole lifecycle: prepare data and target classifier, build method state, generate
counterfactuals, calculate common and method-specific metrics, shape rows, and write CSV/NPZ files.
Adding a baseline therefore means copying another `run_dataset()` implementation.

There is no common method output. Baselines return different arrays and ad hoc `point_info`
dictionaries; Exp9 dismantles its typed generator result into runner-specific rows. Evaluation also
mixes three different concepts: a method returning a candidate, the candidate reaching the target
class, and search satisfying a probability threshold. Factual fallbacks can make coverage appear
to be 1.0 even when a method failed.

The refactor should preserve the strong existing seams: the pinned CEL dataset contract,
deterministic benchmark protocol, dependency-light action types, array-based metric kernels,
stable `generate_counterfactual_batch()` API, and standalone baseline algorithms. It should not
introduce dynamic plugin discovery, a dependency-injection framework, or a false universal API
for every tabular foundation model.

The target design and dependency rules are in
[resources/architecture.md](resources/architecture.md). Current coupling, preserved invariants,
and verification evidence are in [resources/current-state.md](resources/current-state.md).

---

## Strategy

**Contracts first** (Stages 1–3): freeze compatibility evidence, introduce immutable data/method
contracts, and extract one method-independent evaluator and artifact store.

**Encapsulate methods** (Stages 4–5): wrap all baselines and DiCoFlex behind the same
`prepare()`/`generate()` lifecycle without rewriting their algorithms.

**Unify execution** (Stages 6–7): add a static registry, typed run specifications, matrix
expansion, and a DiCoFlex-internal proposal-backend boundary for TabICL and future TabPFN/TabFM
adapters.

**Cut over safely** (Stage 8): reduce numbered modules to compatibility shims, audit dependency
direction, run the cheap compatibility matrix, and publish one final full-reference comparison.

---

## Success Criteria

Every row declares a **Kind**. GATE blocks; REPORT is published and never blocks. Deterministic
GATE evidence is produced by named tests and source-import checks in this run. `NOT MEASURED`
never passes a GATE.

| Metric | Baseline | Target | Kind | If missed | If unmeasurable |
|--------|----------|--------|------|-----------|-----------------|
| Root suite | 87 tests pass at planning baseline | `uv run pytest -q` passes after every stage | GATE | block stage and fix regression | REPORT `NOT MEASURED` and block |
| Dependency direction | Numbered runners compose every layer and evaluation imports data helpers | import-boundary tests prove datasets and evaluation import no concrete methods, numbered runners, or TabICL modules | GATE | block owning stage and remove edge | REPORT `NOT MEASURED` and block |
| Method extension | A baseline requires a new runner | a fake method and every retained method execute through the same method contract without evaluator or dataset-provider changes | GATE | block method/runner stage | REPORT `NOT MEASURED` and block |
| Canonical semantics | return, validity, and search success are conflated | tests distinguish availability, class validity, threshold validity, and valid-success rate for `k=1` and `k>1` | GATE | block evaluator stage | REPORT `NOT MEASURED` and block |
| Data/protocol parity | frozen four-dataset hashes and deterministic split/target contracts | existing dataset and benchmark protocol contract tests remain green through the refactor | GATE | block data stage | REPORT `NOT MEASURED` and block |
| Ablation identity | parameters are split across constants, CLIs, and result rows | canonical `RunSpec` expansion gives distinct stable run IDs/manifests for method, backend, hyperparameter, dataset, seed, and evaluation-version changes | GATE | block orchestration stage | REPORT `NOT MEASURED` and block |
| Compatibility surface | consumers call Exp8/9/11–14 and read legacy CSV/NPZ names | legacy CLIs remain thin offline shims and a compatibility exporter preserves documented v1 artifacts during migration | GATE | block cutover stage | REPORT `NOT MEASURED` and block |
| Full reference matrix | 24 cells, 1,000 factuals, 9.42 measured hours at planning baseline | run once after cutover; publish completeness, common metrics, semantic differences, and phase timings | REPORT | publish and continue; open a focused defect if incomplete | publish `NOT MEASURED` and continue |

---

## Files That May Be Changed

- `experiments/zeroshot_cf/core/` — portable schemas, predictor/method protocols, requests, and canonical results.
- `experiments/zeroshot_cf/datasets/` — dataset-provider contract, CEL adapter, preprocessing provenance, and benchmark-case construction.
- `experiments/zeroshot_cf/methods/` — baseline adapters, DiCoFlex method, explicit registry, typed configs, and proposal backends.
- `experiments/zeroshot_cf/evaluation/` — method-independent evaluator, metric composition, and typed reports.
- `experiments/zeroshot_cf/orchestration/` — run specs, matrix expansion, runner, manifest store, aggregation, and compatibility export.
- `experiments/zeroshot_cf/exp8_tabicl_cf.py`, `exp9_dicoflex_benchmark.py`, and `exp11–14` runners — reduce to thin compatibility CLIs.
- Existing shared modules such as `data.py`, `benchmark_protocol.py`, `generator.py`, `tabicl_runtime.py`, `metrics_harness.py`, and `reporting.py` — delegate to new ownership boundaries while public compatibility APIs remain.
- `experiments/zeroshot_cf/tests/` — contract, import-boundary, parity, runner, artifact, method, backend, and ablation tests.
- `experiments/zeroshot_cf/README.md`, root `README.md`, and Athena launchers — document and use the generic runner.
- `docs/plans/counterfactual-evaluation-architecture/` and `docs/plans/LESSONS.md` — plan, architecture, decisions, and durable findings.

Excluded: actual TabPFN/TabFM algorithm integrations, changing dataset splits or target-model
defaults, adopting dynamic plugin discovery/Hydra, moving the package to `src/`, and modifying the
locally excluded `experiments/zeroshot_cf/ARCHITECTURE.md`.

---

## Stages

Routing table only. Status, notes, and commits live only in `state.json`.

| # | Stage |
|---|-------|
| 1 | [Freeze contracts and compatibility evidence](stages/01-freeze-contracts.md) |
| 2 | [Separate portable data and benchmark cases](stages/02-separate-data-cases.md) |
| 3 | [Extract evaluation and artifact ownership](stages/03-extract-evaluation-artifacts.md) |
| 4 | [Encapsulate all reference methods](stages/04-encapsulate-baselines.md) |
| 5 | [Encapsulate DiCoFlex](stages/05-encapsulate-dicoflex.md) |
| 6 | [Add the generic runner and static registry](stages/06-generic-runner-registry.md) |
| 7 | [Add proposal backends and ablation specifications](stages/07-proposal-backends-ablations.md) |
| 8 | [Cut over entry points and run the final audit](stages/08-cutover-final-audit.md) |
