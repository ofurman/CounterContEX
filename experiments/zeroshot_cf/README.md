# TabICL counterfactual benchmark suite

This directory is the retained CounterContEX surface: the TabICL generator,
the Exp9 CounterContEx benchmark, four comparison baselines, pinned CEL dataset
bootstrap, checkpoint staging, offline smoke checks, and Athena launchers.
Legacy TabPFN counterfactual experiments are intentionally out of scope.

## Setup

Create the suite-owned environment from a fresh checkout:

```bash
uv sync --project experiments/zeroshot_cf --python 3.12 --locked
```

Bootstrap the pinned CEL checkout and validate the four benchmark assets:

```bash
uv run --project experiments/zeroshot_cf \
  python experiments/zeroshot_cf/vendor_setup.py

uv run --project experiments/zeroshot_cf \
  python experiments/zeroshot_cf/vendor_setup.py --check
```

`vendor_setup.py` uses the pinned CEL revision
`3587f943826f6b087a0d198c8c4aa4373712c7ee` under
`experiments/zeroshot_cf/vendor/counterfactuals/`. It verifies four required
config/data pairs:

- `config/datasets/heloc.yaml` -> `data/heloc.csv`
- `config/datasets/bank_marketing.yaml` -> `data/bank_marketing.csv`
- `config/datasets/give_me_some_credit.yaml` -> `data/give_me_some_credit.csv`
- `config/datasets/lending_club.yaml` -> `data/lending_club.csv`

The generated vendor checkout is local-only and ignored by git. If an existing
checkout is at the wrong revision, setup stops with a recovery instruction;
rerun with `--repin` only when you intentionally want to rewrite that local
tree.

The CEL checkout can materialize dataset files under the suite vendor tree and
the repo-root `data/` area, depending on the upstream config. Review the
upstream dataset licenses and provenance before redistributing any copied raw
data.

## Checkpoints and offline commands

Stage the TabICLv2 checkpoint pair once on a machine with network access:

```bash
uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.tabicl_checkpoints
```

The classifier and regressor checkpoints are stored under
`experiments/zeroshot_cf/models/tabicl/`. Runtime checksum verification stays
enabled and the offline smoke test refuses missing or mismatched files.

The following `--help` commands are the documentation sanity checks for the
locked environment. With `HF_HUB_OFFLINE=1`, they must stay free of network
access and checkpoint loading:

```bash
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp8_tabicl_cf --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp9_countercontex_benchmark --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp11_nice_nun_baseline --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp12_optimization_baselines --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp13_dice_baseline --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp14_face_baseline --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.tabicl_checkpoints --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.tabicl_smoke_test --help
```

Run the real offline smoke after the checkpoints are staged:

```bash
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.tabicl_smoke_test
```

## Public generator API

The stable programmatic entry point is
`experiments.zeroshot_cf.generator.generate_counterfactual_batch()`. It accepts
`TabICLGeneratorInputs`, a `TabICLGeneratorConfig`, and an injected point
backend. `exp8_tabicl_cf.py` is the CLI compatibility adapter over that API.

Single-counterfactual mode keeps the original greedy validity search plus
optional plausibility refinement. Multi-counterfactual mode uses bounded-beam
search and DPP-based selection without padding fabricated rows when fewer valid
diverse outputs exist.

## Benchmark and baselines

Exp9 is the fixed four-dataset benchmark:

- `heloc`
- `bank_marketing`
- `give_me_some_credit`
- `lending_club`

The protocol uses one deterministic 64/16/20 train/validation/test split,
seed 42, classifier-prediction targets, atomic categorical edits, immutable
feature preservation, and a shared result schema across TabICL and the
baselines.

Current actionability is intentionally narrow: immutable columns may not change,
and one-hot categorical groups may change only atomically. The suite does not
encode directional, monotonic, or causal constraints.

Available retained CLI entry points:

- `python -m experiments.zeroshot_cf.exp8_tabicl_cf`
- `python -m experiments.zeroshot_cf.exp9_countercontex_benchmark`
- `python -m experiments.zeroshot_cf.exp11_nice_nun_baseline`
- `python -m experiments.zeroshot_cf.exp12_optimization_baselines`
- `python -m experiments.zeroshot_cf.exp13_dice_baseline`
- `python -m experiments.zeroshot_cf.exp14_face_baseline`

The generic runner is the typed entry point for new experiment suites. Matrix
files resolve every dataset, method variant, method parameter, evaluation
setting, target-model family, and seed before execution. A matrix may retain the
singular `target_model` mapping or use a mutually exclusive `target_models` list
for Cartesian expansion. The resulting run ID contains only those
scientific inputs and the measured dataset, case, implementation, model, and
checkpoint identities. Output paths, cache paths, devices, hosts, and resume
settings stay in manifest execution metadata and do not change the run ID.

The production dependency direction is:

```text
core <- datasets
core <- methods
core <- evaluation
datasets + methods + evaluation <- orchestration <- CLI and compatibility shims
```

Datasets own loading, preprocessing, schemas, factual selection, and target
case construction. The dataset-owned registry resolves the fixed
`retained_logistic_regression`, `retained_mlp`, and `retained_xgboost` families;
their fitted content and implementation determine scientific identity. Methods
own preparation, candidate generation, legal-action
enforcement, and namespaced diagnostics. Evaluation derives every common
metric only from the benchmark case and canonical candidates. Orchestration
owns matrix expansion, lifecycle timing, manifests, persistence, resume,
aggregation, and temporary v1 export.

```bash
uv run python -m experiments.zeroshot_cf.cli list-methods
uv run python -m experiments.zeroshot_cf.cli matrix \
  --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml \
  --dry-run
uv run python -m experiments.zeroshot_cf.cli matrix \
  --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml \
  --resume
uv run python -m experiments.zeroshot_cf.cli aggregate \
  --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml
```

Published matrices can be converted into auditable paper products without loading a method or
checkpoint. Every F3--F7 figure is accompanied by its source CSV, and T1--T3 are emitted as CSV
and LaTeX from the same seed aggregation:

```bash
uv run python -m experiments.zeroshot_cf.cli analyze \
  --config experiments/zeroshot_cf/configs/matrices/full_reference.yaml \
  --output experiments/zeroshot_cf/results/local/analysis
```

Confidence-Pareto analyses can add comparable baseline points from a second published matrix;
both inputs remain strict, read-only artifact roots resolved from their tracked configs:

```bash
uv run python -m experiments.zeroshot_cf.cli analyze \
  --config experiments/zeroshot_cf/configs/matrices/campaign_e4_confidence.yaml \
  --baseline-config experiments/zeroshot_cf/configs/matrices/campaign_e1_main.yaml \
  --output experiments/zeroshot_cf/results/campaign/analysis/e4_confidence
```

Analysis refuses partial, missing, extra, duplicate, or identity-mismatched cells. Seed groups
are keyed by the complete scientific specification except seed and report mean, sample standard
deviation, and the actual finite `n` for each metric. Historical evaluation-v1 artifacts remain
read-only inputs: their COMPLETE markers, typed tables, identities, and exact matrix membership
are validated without rewriting or re-evaluating them.

Each run directory is complete only after its `COMPLETE` marker is published.
Resume validates the complete manifest against the resolved identity. Matrix
aggregation reads the expected manifest set and rejects missing, extra,
partial, duplicate, or identity-mismatched cells.

The directory contains `manifest.json`, `summary.csv`, `points.csv`,
`candidates.csv`, `arrays.npz`, and the final `COMPLETE` marker. The manifest
records the scientific spec, resolved dataset/case/method/backend/model and
checkpoint identities, execution metadata, method diagnostics, and measured
prepare/generate/evaluate/write/total phases. Execution-only settings such as
output roots, checkpoint locations, devices, hosts, scheduler limits, and
resume do not change run identity.

## Metric semantics

Availability and validity answer different questions:

- `coverage` is the fraction of factuals with at least one returned candidate.
- `validity_returned_class` is target-class candidates divided by returned
  candidates.
- `validity_returned_threshold` additionally requires the configured target
  probability threshold.
- `valid_success_rate_*_per_requested_slot` uses all requested candidate slots
  as the denominator.
- `valid_success_rate_*_per_factual` counts factuals with at least one success.
- `primary_*` fields evaluate only the configured primary rank; `set_*` and
  diversity fields evaluate the complete returned set.

Grouped-Gower and continuous proximity use returned candidates that reach the
target class. Sparsity, action-unit changes, immutable-feature actionability,
out-of-bounds, LOF, Isolation Forest, and k-th-nearest grouped-Gower distance use
all available returned candidates. A failed best-effort row can be retained
under `method.*` for diagnosis, but it is not a returned counterfactual and
cannot enter common metric denominators.

Evaluation schema `countercontex.evaluation.v2` additionally stores
`common.target_probabilities` with one slot per requested candidate (unavailable
slots are NaN), and `candidate.gower_kth_neighbor` with one value per available
candidate. The summary reports the mean k-th-neighbour distance; larger values
mean weaker support in the benchmark reference distribution.

`detectability_auc` is an orientation-independent, five-fold out-of-fold AUC for
separating target-label-matched real training rows from returned target-class
counterfactuals using a fixed standardized logistic regression. Values near 0.5
mean the arms are indistinguishable to this probe; values near 1 mean easily
separable. Always read it with `detectability_n_reference`,
`detectability_n_counterfactual`, and `detectability_status`. When the matched CF
arm has fewer than `detectability_min_cf_rows` (20 by default), AUC is null and
status is `NOT_MEASURED`; empty or near-empty arms never receive a synthetic 0.5.

## Adding a counterfactual method

1. Implement a frozen config plus `CounterfactualMethod.prepare()` and the
   prepared method's `generate()` under `methods/`.
2. Return a validated `GenerationResult`; unavailable slots contain NaN and
   method-specific arrays use the `method.*` namespace.
3. Register the lazy factory, supported variants, and implementation version
   in `methods/registry.py`.
4. Add deterministic contract tests for reversed class labels, request seeds,
   action constraints, genuine failures, and optional-import safety.
5. Add a matrix entry. No dataset loader, evaluator, artifact schema, or
   numbered runner change is required.

## CounterContEx proposal backends

CounterContEx search depends on the protocols in
`methods/countercontex/backends/base.py`. A backend declares whether it supplies
numerical proposals, confidence conditioning, categorical distributions, and
joint scoring. Preparation creates dataset-level state, and `for_factual()`
creates a seeded proposal session. Unsupported search and backend combinations
fail during validation instead of being discovered inside the search loop.

The TabICL adapter owns compact categorical encoding, neighbor context,
confidence anchors, proposal sampling, and joint-density scoring. Its
method-owned runtime policy resolves checkpoint identity, cache paths, and
device scope. The search layer consumes only the proposal-session contract,
including one paired batch operation used for beam expansion. This boundary
keeps the generic runner, dataset preparation, and common evaluation free of
foundation-model imports.

The deterministic `empirical` adapter supplies target-class numerical
quantiles and categorical frequencies without checkpoints. It provides a
runnable backend ablation and intentionally declares no confidence or joint
scoring capability, so incompatible search settings fail before generation.

To add a TabPFN or TabFM adapter:

1. Implement `ProposalBackend.prepare()` and the prepared backend's
   `for_factual()` method in a new module under `methods/countercontex/backends/`.
2. Translate the model's numerical, categorical, confidence, and joint-score
   outputs into `ProposalSession`; keep model loading, caches, checkpoints, and
   native representations inside the adapter.
3. Declare only capabilities the adapter implements and add conformance tests
   for every declared operation and every rejected unsupported combination.
4. Register a stable backend identifier and implementation version so a backend
   or model-content change creates a new run identity, and register its
   execution policy in `methods/countercontex/runtime.py`.
5. Add a matrix variant that changes only the foundation/backend fields. Do not
   modify datasets, the evaluator, or common artifact schemas for an ablation.

See `configs/matrices/countercontex_ablation_example.yaml` for independent search,
diversity, backend-parameter, dataset, and seed variants. The tracked
`full_reference.yaml` fixes the four retained datasets, all six methods, seed
42, 1,000 factuals, and the recorded three-counterfactual CounterContEx setup.

Run the cheap compatibility matrix before the full reference workload. The
full matrix requires staged TabICL checkpoints and optional baseline
dependencies. Its recorded runtime was about 9.42 hours, including 7.64 hours
for CounterContEx on Lending Club. Use `--resume` with a scheduler limit above that
longest cell, and publish completeness, named availability/validity metrics,
phase timings, and stable artifact hashes. If a cell is unavailable, record
its exact run ID rather than substituting a value.

## Compatibility-shim lifetime

`exp8_tabicl_cf.py`, `exp9_countercontex_benchmark.py`, and Exp11-14 retain their
CLI flags and flat v1 CSV/NPZ filenames for existing local and Athena
workflows. They are translation shims into the generic runner; new automation
should use `experiments.zeroshot_cf.cli` and matrix configs. Keep the shims
until downstream jobs have migrated, then remove them in a separately reviewed
compatibility change.

Example offline runs:

```bash
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.exp8_tabicl_cf \
  --dataset heloc \
  --max-test 10 \
  --cache-dir experiments/zeroshot_cf/models/tabicl

HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.exp9_countercontex_benchmark \
  --dataset heloc \
  --tabicl-cache-dir experiments/zeroshot_cf/models/tabicl

uv run --project experiments/zeroshot_cf \
  pytest -q
```

## Outputs and environment variables

Local benchmark and baseline runs write under `experiments/zeroshot_cf/results/local/`.
Athena runs write under `experiments/zeroshot_cf/results/athena/`. Both trees
are ignored by git. Discriminator caches live under `experiments/zeroshot_cf/models/`.

Relevant environment variables:

- `TABICL_LOCAL_CACHE`: default checkpoint directory.
- `TABICL_DEVICE`: `auto`, `cpu`, `mps`, or `cuda`.
- `HF_HUB_OFFLINE=1`: enforce offline operation after staging.
- `ZEROSHOT_CF_MODELS_DIR`: override the discriminator cache/output model directory.

## Athena

See [athena/README.md](athena/README.md) for the four-case Exp9 Slurm launcher,
environment variables, and the aggregation command.
