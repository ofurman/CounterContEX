# Plan: Clean and Isolate the TabICL Counterfactual Benchmark Suite

**Date**: 2026-08-30
**Branch**: `cleanup/tabicl-suite` (created from `manik24901/tabicl-comparison` at `e80925d`)
**Planning baseline**: `e80925d`
**Predecessors**: [`zeroshot-tabpfn-cf`](../zeroshot-tabpfn-cf/index.md), [`iterative-greedy-cf`](../iterative-greedy-cf/index.md)
**Goal**: Leave the repository focused on an independently runnable TabICL counterfactual suite containing the generator, Exp9 DiCoFlex benchmark, comparison baselines, bounded-beam/DPP diversity, and their operational support.

Executed per [PROTOCOL.md](PROTOCOL.md). Status of record: [state.json](state.json).
Runtime record: [journal.md](journal.md) · [decisions.md](decisions.md) · [backlog.md](backlog.md)

---

## Context

The desired suite is implemented, but its boundaries follow experiment history rather than runtime
responsibility. Exp8 and Exp9 import constants and reporting from legacy Exp4. The generator,
grouped categorical search, and diversity search import two projection helpers through `greedy.py`,
which eagerly imports the old TabPFN sampler and `tabpfn-extensions`. Baselines also import private
helpers from Exp8, Exp9, NICE, and each other.

The four-dataset benchmark has a second isolation problem: `data.py` imports CEL and loads YAML/CSV
assets from a gitignored vendor checkout. The current checkout is commit
`3587f943826f6b087a0d198c8c4aa4373712c7ee`, while setup scripts fetch an unpinned HEAD. This plan
keeps CEL, pins that dependency, and freezes the current 64/16/20 split and feature/action-space
contract before cleanup. It does not replace CEL or create a new dataset loader.

The retained and removable boundaries, import hazards, protocol details, and historical anchors are
in [resources/boundary.md](resources/boundary.md). This planning turn changes only plan files; source
cleanup begins during execution.

---

## Strategy

- **Protect behavior (Stage 1)**: freeze the retained file/API and four-dataset benchmark contract,
  pin the current CEL source, and make the focused tests reproducible before deleting anything.
- **Cut legacy edges (Stages 2–4)**: extract neutral projection, action-space, benchmark-protocol,
  baseline-common, generator, and reporting boundaries. Preserve single-CF refinement and multi-CF
  bounded-beam/DPP behavior as distinct supported modes.
- **Delete and reproduce (Stages 5–6)**: remove legacy TabPFN counterfactual experiments and
  assets, then create a suite-owned uv environment, documentation, and operational checks.
- **Repurpose and audit (Stages 7–8)**: rewrite root packaging/CI/docs around the retained suite,
  delete the upstream TabPFN source/tests/examples and cached top-level `tabpfn/` directory, then
  audit imports, CLIs, Athena, tests, offline behavior, and the full keep/drop manifest.

Execution stays on `cleanup/tabicl-suite` and produces one reviewable commit per stage. It does not
initialize a new repository or rewrite history; the user will squash the cleanup commits afterward.

---

## Success Criteria

Every row declares a **Kind**. GATE blocks; REPORT is recorded and never blocks. Deterministic GATE
evidence comes from the named files and commands in this run. `NOT MEASURED` never passes a GATE.

| Metric | Baseline | Target | Kind | If missed | If unmeasurable |
|--------|----------|--------|------|-----------|-----------------|
| Retained capabilities | Desired behavior is spread across Exp8/9 and shared modules | Focused tests cover conditional sampling, greedy mixed interventions, quantiles, confidence, revisits, plausibility refinement, bounded-beam/DPP diversity, Exp9, and all four baselines | GATE | block final stage and restore behavior | REPORT `NOT MEASURED` and block |
| Benchmark protocol | Protocol is implicit and cross-imported | Tests enforce the four datasets, deterministic 64/16/20 split, seed 42, classifier-prediction targets, atomic categories, immutable preservation, common output schema, and primary-vs-diverse metric distinction | GATE | block owning stage and fix protocol | REPORT `NOT MEASURED` and block |
| Legacy import boundary | Retained imports reach Exp4, `greedy.py`, old TabPFN sampler, CEL side effects, and numbered-runner private helpers | Retained generator, benchmark, baseline, aggregate, and `--help` imports exclude removed Exp1–6 runtime, local TabPFN, and `tabpfn-extensions`; aggregation remains independent of TabICL | GATE | block owning stage and remove coupling | REPORT `NOT MEASURED` and block |
| Cleanup boundary | Legacy experiment and upstream TabPFN files are tracked | Every path classified **Remove** in `resources/boundary.md` is absent, including `src/tabpfn`, root tests/examples/changelog/scripts, predecessor plans, and cached top-level `tabpfn/`; every **Keep** entry point remains runnable | GATE | block owning deletion stage | REPORT `NOT MEASURED` and block |
| Reproducible suite environment | Root packaging describes TabPFN and broad requirements are required | Root and suite-owned locked uv entry points install direct dependencies, use `tabicl==2.1.1` and pinned CEL bootstrap, discover focused tests, and run without local TabPFN or `tabpfn-extensions` | GATE | block environment stage | REPORT `NOT MEASURED` and block |
| Focused repository surface | 116 `src/tabpfn` files, 121 root tests, 13 examples, 7 changelog fragments, upstream scripts/docs/CI, and ignored top-level bytecode exist | Root metadata, README, CI, notices, and tooling describe only the retained TabICL suite; upstream source/tests/examples and TabPFN-only automation are absent | GATE | block repository stage | REPORT `NOT MEASURED` and block |
| Real TabICL smoke | Checkpoints are absent in this workspace | Run conditional and generator smoke tests when both checksum-verified weights exist; otherwise record exact missing paths | REPORT | publish failure and open backlog item | publish `NOT MEASURED` and continue |

---

## Files That May Be Changed

- `experiments/zeroshot_cf/*.py` — extract stable generator, protocol, action, baseline, and
  reporting boundaries; delete legacy TabPFN counterfactual runtime.
- `experiments/zeroshot_cf/tests/` — retain and strengthen generator, diversity, benchmark,
  baseline, data-contract, checkpoint, distance, and metric tests; remove legacy-only tests.
- `experiments/zeroshot_cf/athena/` — retain Exp9 launchers and align them with stable entry points.
- `experiments/zeroshot_cf/configs/` and tracked `results/` — retain HELOC actionability; remove
  legacy sweep config and committed Exp1–6 outputs.
- `experiments/zeroshot_cf/README.md`, suite dependency metadata, and lockfile — document and
  reproduce only the retained suite.
- `experiments/zeroshot_cf/vendor_setup.py`, `patches/`, `.gitignore` — pin CEL bootstrap and align
  cache/output rules without deleting local user data.
- `src/tabpfn/`, root `tests/`, `examples/`, `changelog/`, `scripts/`, and the cached top-level
  `tabpfn/` directory — remove after the retained suite is independently green.
- Root `pyproject.toml`, `uv.lock`, `README.md`, `SECURITY.md`, `.github/`, `.gitignore`,
  `.pre-commit-config.yaml`, and `THIRD-PARTY-NOTICES.md` — repurpose for the focused suite.
- Predecessor plan directories — remove after durable findings are retained in `LESSONS.md`.
- `docs/plans/tabicl-generator-cleanup/` and `docs/plans/LESSONS.md` — plan and durable findings.

Excluded: root `LICENSE`, the pre-existing untracked `experiments/zeroshot_cf/ARCHITECTURE.md`, and
ignored checkpoints, datasets, vendor checkout, model caches, logs, and experiment test caches.

---

## Stages

Routing table only. Status, notes, and commits live only in `state.json`.

| # | Stage |
|---|-------|
| 1 | [Freeze the retained suite and benchmark contract](stages/01-freeze-retained-contract.md) |
| 2 | [Extract dependency-neutral shared primitives](stages/02-extract-shared-primitives.md) |
| 3 | [Isolate the TabICL generator and diversity APIs](stages/03-isolate-generator-diversity.md) |
| 4 | [Normalize Exp9 and baseline protocol boundaries](stages/04-normalize-benchmark-baselines.md) |
| 5 | [Remove legacy TabPFN counterfactual experiments](stages/05-remove-legacy-tabpfn-cf.md) |
| 6 | [Create the focused uv environment and documentation](stages/06-environment-docs-operations.md) |
| 7 | [Remove the upstream TabPFN project and repurpose the root](stages/07-remove-upstream-tabpfn.md) |
| 8 | [Run the final boundary and behavior audit](stages/08-final-boundary-audit.md) |
