# Agent Guide: CounterContEX Research Benchmark

## Project Mission

This is a research repository for developing and evaluating counterfactual
explanations for tabular classifiers. Act as a researcher and software engineer:
make changes that improve counterfactual quality, experimental validity,
reproducibility, and runtime efficiency, then support conclusions with measured
evidence.

The main research method is **CounterContEx**. Its machine name is
`countercontex`, and its default foundation backend is TabICL. The benchmark also
contains NICE, Wachter, Growing Spheres, DiCE, and FACE reference methods.

The maintained implementation is under `experiments/zeroshot_cf/`. The core
objective is to compare all methods through one dataset, generation, evaluation,
artifact, and run-identity pipeline across the four retained datasets:

- HELOC
- Bank Marketing
- Give Me Some Credit
- Lending Club

## Researcher Mode

- **Protect metric validity.** Coverage, validity, proximity, sparsity,
  actionability, plausibility, diversity, and runtime are the judges. A faster or
  higher-scoring method is not an improvement if its denominator, candidate set,
  target definition, or protocol changed unnoticed.
- **State the hypothesis.** Before a substantive experiment, name the expected
  effect, primary metric, guardrail metrics, datasets, seeds, and comparison
  baseline.
- **Experiment with controlled axes.** Change one scientific axis at a time when
  possible. Use tracked matrix configurations so resolved specifications can be
  inspected and repeated.
- **Prefer cheap evidence first.** Use contract tests, deterministic fixtures,
  the empirical backend, and matrix `--dry-run` before checkpoint-backed runs.
- **Use TDD for behavior changes.** Add a focused failing test or deterministic
  witness before implementation when it can distinguish the defect from the
  intended behavior.
- **Do not fabricate evidence.** Never pad missing counterfactuals, replace missing
  metrics with constants, copy implementation output into an expected fixture
  without independent provenance, or report a dry-run as an executed benchmark.
- **Preserve failures.** Missing or failed candidate slots are evidence. Keep them
  out of valid-candidate denominators and retain method diagnostics when useful.
- **Reuse existing code.** Search before adding helpers, metric kernels, adapters,
  or configuration types. Extend the established layer instead of creating a
  parallel execution path.
- **Document durable findings.** Record expensive or surprising findings in the
  relevant plan journal. Add only cross-plan lessons to
  `docs/plans/LESSONS.md`.

## Coding Discipline

These rules apply to all code written or modified in this repository. Use
judgment for trivial changes, but preserve the underlying contracts.

### 1. Think Before Coding

Do not make silent scientific or architectural assumptions.

- Read `README.md`, the relevant package code, tests, and active plan before
  editing.
- State assumptions that affect metrics, protocols, identities, artifacts,
  checkpoints, or compatibility.
- If multiple interpretations materially change scientific results, present the
  viable options and recommend one before implementation.
- Prefer the smallest design consistent with the existing dependency direction.
- Push back on requests that would mix method policy into the generic runner,
  compare non-equivalent protocols, or destroy historical evidence.

When a choice affects scientific validity, compare two or three approaches in
terms of expected quality, runtime, reproducibility, and artifact compatibility.
Ask for direction only when repository evidence cannot resolve the choice safely.

### 2. Simplicity First

- Write the minimum code needed for the requested hypothesis or behavior.
- Do not add speculative configuration, extension points, or generalized
  frameworks.
- Do not create an abstraction for a single use unless it enforces an existing
  architectural boundary.
- Validate at system boundaries and through the immutable core contracts.
- Keep method-specific logic inside the method package; do not make the generic
  lifecycle configurable around one method.
- If a change can be expressed as a small extension to an existing config,
  adapter, evaluator, or matrix, use that seam.

### 3. Surgical Changes

- Touch only files required by the task.
- Do not reformat, rename, or refactor unrelated code.
- Match the existing style and frozen dataclass patterns.
- Remove imports, variables, functions, and files made obsolete by your change.
- Report unrelated defects instead of fixing them unless they block the task.
- Every changed line should trace to the request, an explicit contract, or a
  verification failure caused by the change.

### 4. Goal-Driven Execution

Turn work into verifiable outcomes. For a multi-step task, use a short plan such
as:

```text
1. Freeze the failing or baseline behavior -> verify with a focused test/fixture
2. Implement through the owning layer -> verify focused tests and identity
3. Run repository gates -> verify full tests, Ruff, offline CLI, and dry-run
```

For bugs, first create a focused reproduction when it adds meaningful protection.
For refactors, compare behavior before and after. For experimental changes,
declare success and guardrail metrics before looking at the result.

### 5. Compatibility Is an Explicit Contract

This repository retains numbered Experiment 8, 9, and 11-14 commands and the v1
flat CSV/NPZ interface. Do not apply an MVP-style “delete and replace” policy.

- Preserve v1 filenames, ordered columns, method IDs, and array semantics unless
  the task explicitly owns a compatibility break.
- Keep compatibility modules as thin translations into the generic runner.
- New automation should use `experiments.zeroshot_cf.cli` and tracked matrices.
- Remove a compatibility interface only in a separately reviewed change with
  frozen before/after evidence.

## Naming Rules

- **Repository/product:** `CounterContEX`
- **Research method and Python types:** `CounterContEx`
- **Machine identifiers, registry key, package, schema namespace:**
  `countercontex`

Do not introduce spelling aliases or normalizers for former method names. A
registry name or implementation-version change is a scientific identity change,
not a cosmetic migration.

## Tech Stack and Architecture

**Language:** Python 3.12, locked and run with `uv`

**Core libraries:** NumPy, pandas, scikit-learn, SciPy, PyTorch, TabICL, PyYAML,
and optional `dice-ml`

**Experiment definition:** typed Python specifications plus tracked YAML matrices

**Storage:** content-addressed local artifacts in JSON, CSV, and NPZ formats

The production dependency direction is:

```text
core <- datasets
core <- methods
core <- evaluation
datasets + methods + evaluation <- orchestration <- CLI and compatibility shims
```

Layer ownership:

- `core/`: immutable contracts and validation shared across portable layers.
- `datasets/`: CEL loading, preprocessing, feature/action schemas, fingerprints,
  factual selection, targets, and benchmark-case construction.
- `methods/`: method configuration, preparation, candidate generation, legal
  actions, and namespaced diagnostics.
- `evaluation/`: method-blind common metrics derived from a benchmark case and
  canonical candidates.
- `orchestration/`: matrix expansion, scientific identity, lifecycle timing,
  manifests, persistence, resume, aggregation, and v1 export.
- CLI and numbered modules: argument translation only.

The generic runner must not import or branch on CounterContEx, TabICL, empirical,
or another concrete method/backend policy.

## Repository Map

```text
experiments/zeroshot_cf/
  core/                     # Portable immutable contracts
  datasets/                 # Providers, preprocessing, and benchmark cases
  methods/                  # CounterContEx, baselines, and lazy registry
    countercontex/          # Search, config, runtime, and proposal backends
  evaluation/               # Common evaluator, metric kernels, reports
  orchestration/            # Specs, matrices, runner, artifacts, v1 contracts
  configs/matrices/         # Tracked experiment definitions
  athena/                   # Slurm launchers and operational instructions
  tests/                    # Contract, unit, integration, and boundary tests
  models/                   # Ignored checkpoint and model caches
  results/                  # Ignored local/Athena artifacts
  vendor/                   # Ignored pinned CEL checkout
docs/
  papers/                   # Research references and notes
  plans/                    # Plans, state, decisions, journals, and lessons
data/                       # Local benchmark data materialized by setup
```

## Scientific Contracts

### Benchmark Protocol

The reference protocol uses a deterministic 64/16/20 train, validation, and test
split, seed 42, and targets derived from classifier predictions. Numerical
features and atomic one-hot groups define the action space. Immutable features
must remain unchanged.

Do not compare result rows as if they shared a protocol when any of these differ:

- dataset source or provenance fingerprint
- split, seed, factual-selection policy, or number of factuals
- target classifier or model content
- target construction or probability threshold
- feature preprocessing or action schema
- method/backend implementation identity or checkpoint content
- requested number or rank of counterfactuals

### Metric Semantics

Availability and validity have different denominators:

- `coverage`: factuals with at least one returned candidate divided by factuals.
- `validity_returned_class`: target-class candidates divided by returned
  candidates.
- `validity_returned_threshold`: returned candidates that also meet the target
  probability threshold divided by returned candidates.
- `valid_success_rate_*_per_requested_slot`: successful candidates divided by
  every requested slot, including unavailable slots.
- `valid_success_rate_*_per_factual`: factuals with at least one success divided
  by factuals.
- `primary_*`: only the configured primary rank.
- `set_*` and diversity metrics: the complete returned set.

The current evaluator uses two candidate populations deliberately:

- grouped-Gower and continuous proximity use returned candidates that reach the
  target class;
- sparsity, action-unit changes, immutable-feature actionability, out-of-bounds,
  LOF, and Isolation Forest use all available returned candidates.

Unavailable slots and diagnostic-only best-effort rows enter neither population.
Do not change these denominators silently; a new interpretation requires explicit
metric names, tests, and documentation.

When changing metrics, add tests for reversed class labels, missing candidates,
multiple candidate ranks, one-hot groups, immutable features, and denominator
selection as applicable.

### Actionability and Plausibility

Current actionability is deliberately narrow:

- immutable features cannot change;
- one-hot categorical groups can change only atomically.

The suite does not currently model directional, monotonic, causal, feasibility,
or user-cost constraints. Do not claim those forms of actionability without a
new contract and evaluation measure.

LOF and Isolation Forest scores are distribution-support diagnostics fitted on
the benchmark reference features. Report their orientation: the stored LOF value
is `-score_samples`, so larger values are more outlying; the Isolation Forest
value is `decision_function`, so larger values are more inlying. Neither is a
direct probability that a counterfactual is realistic. Out-of-bounds fraction is
the fraction of available candidates with at least one normalized feature below
0 or above 1. It is separate from density-based plausibility.

## CounterContEx Boundaries

CounterContEx follows this internal dependency:

```text
CounterContEx method -> search -> ProposalSession -> TabICL or empirical backend
```

- The search layer consumes only the proposal-session contract.
- The TabICL backend owns categorical encoding, neighbor context, confidence
  anchors, proposal sampling, and optional joint-density scoring.
- The empirical backend provides deterministic target-class quantiles and
  categorical frequencies without checkpoints. It intentionally does not claim
  confidence-conditioning or joint-scoring capabilities.
- Runtime policy owns devices, cache paths, checkpoint lookup, model content IDs,
  and backend implementation identity.
- Unsupported backend/search combinations must fail during configuration or
  preparation, before the search loop.

When adding a method or backend, follow the extension procedures in
`experiments/zeroshot_cf/README.md`. Register a stable implementation version and
test capability declarations, seeding, action constraints, failures, and optional
imports.

## Reproducibility and Run Identity

Scientific settings and resolved content identities determine `cell_id` and
`run_id`. Execution settings do not.

Scientific identity includes the resolved dataset/case, method and backend
implementations, target model, checkpoint/model content, protocol, evaluation
settings, and seed. Devices, hosts, cache paths, output roots, scheduler limits,
and `--resume` are execution metadata.

- Any scientific behavior change requires an implementation-version change when
  existing specs would otherwise keep the same identity.
- Never rewrite, rename, or resume a historical manifest under a new identity.
- Resume must validate the complete manifest against the freshly resolved
  identity.
- Aggregation must reject missing, extra, partial, duplicate, and mismatched
  cells.
- Preserve seeds at every stochastic boundary; do not rely on ambient global RNG
  state.

## Artifact Safety

A canonical run is complete only when its `COMPLETE` marker is published after
all payloads are ready:

```text
<run-id>/
  manifest.json
  summary.csv
  points.csv
  candidates.csv
  arrays.npz
  COMPLETE
```

`experiments/zeroshot_cf/results/`, `models/`, and `vendor/` are ignored working
trees, not scratch space for source refactors.

- Do not edit, rename, delete, relabel, or fabricate historical results to make a
  gate pass.
- Do not commit checkpoints, caches, vendor data, or generated benchmark output.
- Read ignored artifacts only when the task needs historical evidence.
- Record unavailable or incomplete cells by exact identity; do not substitute a
  nearby run.
- Preserve portable dtypes, shapes, finite/null semantics, and strict JSON rules.

## Performance and Timing

The generic lifecycle records prepare, generate, evaluate, write, and total
timings in the manifest. Preserve this instrumentation.

- Use `time.perf_counter()` for new elapsed-time measurements.
- Keep scientific identity independent of timing and machine metadata.
- Report dataset, method/backend, factual count, candidate count, device, and
  checkpoint/model identity with benchmark timings.
- Compare performance only under equivalent scientific specifications.
- Do not turn noisy runtime values into exact golden assertions. Verify their
  presence, type, and sensible bounds; compare deterministic artifact values
  separately.
- Treat accelerator warm-up, model loading, caches, and preparation separately
  from per-factual generation when interpreting performance.

## Development Workflow

### Running Python

Always use `uv` and the locked environment. Run commands from the repository root
unless a documented command says otherwise.

```bash
uv sync --locked
uv run pytest -q
uv run pytest -q experiments/zeroshot_cf/tests/test_evaluation_semantics.py
uv run ruff check experiments/zeroshot_cf/evaluation \
  experiments/zeroshot_cf/tests/test_evaluation_semantics.py
uv run python -m experiments.zeroshot_cf.cli list-methods
```

Do not use `pip`, `pipx`, or Poetry in this repository.

### Validation Ladder

Use the cheapest gate that can detect the defect, then expand verification:

1. Run focused tests for the owning module and contract.
2. Run Ruff on changed Python files or the affected package.
3. Run `uv run pytest -q` for load-bearing or cross-layer changes.
4. Check retained CLIs offline when imports, runtime resolution, or compatibility
   changes:

   ```bash
   HF_HUB_OFFLINE=1 uv run python -m \
     experiments.zeroshot_cf.exp9_countercontex_benchmark --help
   ```

5. Resolve affected matrices without executing methods:

   ```bash
   uv run python -m experiments.zeroshot_cf.cli matrix \
     --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml \
     --dry-run
   ```

6. Run checkpoint-backed smoke or benchmark work only when required, staged, and
   authorized.

Do not run the real one-factual compatibility matrix, ablation matrix, or
`full_reference.yaml` merely as a routine code gate. The 24-cell full reference
run has taken about 9.42 hours, including about 7.64 hours for one Lending Club
CounterContEx cell. An explicit user request is required before launching costly
or checkpoint-backed matrices.

### Dataset and Checkpoint Setup

The CEL dependency is pinned to revision
`3587f943826f6b087a0d198c8c4aa4373712c7ee`:

```bash
uv run python experiments/zeroshot_cf/vendor_setup.py
uv run python experiments/zeroshot_cf/vendor_setup.py --check
```

Do not use `--repin` unless the task explicitly changes the pinned dependency.
Review dataset and upstream licenses before redistributing materialized data.

Stage TabICL checkpoints only when network access and the task allow it:

```bash
uv run python -m experiments.zeroshot_cf.tabicl_checkpoints
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.tabicl_smoke_test
```

Never disable checksum/content-identity verification to make checkpoint loading
pass.

### Experiment Workflow

For a new experiment:

1. Define the hypothesis, baseline, primary metric, guardrails, datasets, seeds,
   factual count, and candidate count.
2. Add or modify a tracked matrix with only the intended axes.
3. Dry-run it and inspect every resolved specification and expected cell count.
4. Run focused deterministic or empirical-backend checks first.
5. If a real run is authorized, preserve its exact output root and run IDs.
6. Check completeness before interpreting metrics.
7. Compare coverage and validity before conditional quality metrics.
8. Report proximity, plausibility, sparsity, actionability, diversity, and phase
   timings with their denominators and availability.
9. Record the configuration, artifact paths/hashes, failures, and findings.

Never select a threshold or configuration on the same held-out results later
reported as an unbiased final comparison without disclosing that selection.

### Version Control

- Inspect `git status` before and after edits.
- Keep commits atomic and use descriptive Conventional Commit messages.
- Do not use `git commit --no-verify`.
- Do not commit ignored result, model, vendor, or cache trees.
- Commit only when the user asks or an active plan protocol requires it.
- Do not rewrite shared history or force-push without explicit authorization.

## Planning and Documentation

For multi-stage architecture work, use the plan structure under `docs/plans/`.
Read the plan's `PROTOCOL.md`, current `state.json`, relevant stage brief, and
journal tail before executing it. Plan state is authoritative; journals are
append-only.

Read `docs/plans/LESSONS.md` before planning a change that touches dataset
ownership, imports, compatibility, metrics, run identity, or expensive benchmark
execution.

When architecture, protocols, metrics, commands, or experiment matrices change,
update the relevant documentation:

1. `README.md` for repository-level setup, architecture, supported benchmark, and
   primary workflows.
2. `experiments/zeroshot_cf/README.md` for suite contracts, metric semantics,
   extension procedures, environment variables, and compatibility details.
3. The owning `docs/plans/<plan>/` state, journal, decisions, or backlog when work
   is plan-driven.
4. `experiments/zeroshot_cf/athena/README.md` when Slurm files, environment
   variables, wall times, output roots, or aggregation change.
5. Matrix YAML files when the scientific protocol changes; do not describe a
   protocol in prose that the tracked matrix does not resolve.

## Key References

- `README.md` — repository mission, setup, architecture, benchmark commands, and
  layout.
- `experiments/zeroshot_cf/README.md` — detailed suite contracts, metric
  semantics, extension procedures, and Athena link.
- `experiments/zeroshot_cf/core/contracts.py` — immutable portable interfaces.
- `experiments/zeroshot_cf/methods/registry.py` — method variants and
  implementation identities.
- `experiments/zeroshot_cf/orchestration/spec.py` — scientific versus execution
  identity.
- `experiments/zeroshot_cf/orchestration/runner.py` — shared lifecycle and resume
  behavior.
- `experiments/zeroshot_cf/orchestration/artifacts.py` — publication and
  aggregation contracts.
- `experiments/zeroshot_cf/evaluation/` — common metric implementation.
- `experiments/zeroshot_cf/configs/matrices/` — tracked scientific
  specifications.
- `docs/plans/LESSONS.md` — durable facts learned across plans.

When modifying a subsystem, read its tests and the plan that introduced its
current contracts before changing behavior.
