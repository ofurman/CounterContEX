# CounterContEX

CounterContEX is a counterfactual-explanation benchmark for tabular classifiers.
It has one execution path for dataset preparation, counterfactual generation,
evaluation, persistence, and comparison across methods.

The main research method is CounterContEx with a TabICL proposal backend. The repository
also includes NICE, Wachter, Growing Spheres, DiCE, and FACE baselines. All methods
use the same benchmark cases, evaluation metrics, artifact schema, and run identity
rules.

See the [CounterContEx method overview](docs/countercontex-method.md) for the
research motivation, algorithm, assumptions, and evaluation protocol.

The maintained implementation is under
[`experiments/zeroshot_cf/`](experiments/zeroshot_cf/). See the
[suite README](experiments/zeroshot_cf/README.md) for detailed metric definitions,
extension procedures, and Athena instructions.

## Supported benchmark

The full reference benchmark uses four binary-classification datasets:

- HELOC
- Bank Marketing
- Give Me Some Credit
- Lending Club

The default protocol uses a deterministic 64/16/20 train, validation, and test
split. It uses seed 42 and selects counterfactual targets from classifier
predictions. The target-model registry provides fixed logistic-regression, MLP,
and XGBoost families. Numerical features and atomic one-hot groups define the
action space. Immutable features must remain unchanged.

The method registry contains:

| Method | Registry name | Multiple CFs | Optional runtime |
|---|---|---:|---|
| CounterContEx | `countercontex` | Yes | TabICL checkpoints |
| NICE | `nice` | No | None |
| Wachter | `wachter` | No | None |
| Growing Spheres | `growing_spheres` | No | None |
| DiCE | `dice` | No | `dice-ml` |
| FACE | `face` | No | None |

The target-model registry contains `retained_logistic_regression`, `retained_mlp`,
and `retained_xgboost`. Matrix files may use the backward-compatible singular
`target_model` mapping or a `target_models` list to expand classifier family as a
scientific axis.

CounterContEx also has a deterministic `empirical` proposal backend. This backend is
useful for checkpoint-free ablations. It does not support confidence conditioning
or joint-density scoring.

## Setup

CounterContEX requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).
Run all commands from the repository root.

Install the locked workspace:

```bash
uv sync --locked
```

Create and verify the pinned local CEL checkout:

```bash
uv run python experiments/zeroshot_cf/vendor_setup.py
uv run python experiments/zeroshot_cf/vendor_setup.py --check
```

The setup script pins CEL revision
`3587f943826f6b087a0d198c8c4aa4373712c7ee`. It stores the ignored checkout at
`experiments/zeroshot_cf/vendor/counterfactuals/`.

TabICL runs require a classifier and regressor checkpoint. Stage and verify them
once on a machine with network access:

```bash
uv run python -m experiments.zeroshot_cf.tabicl_checkpoints
```

The checkpoint command stores the files under
`experiments/zeroshot_cf/models/tabicl/`. Normal benchmark runs can then operate
with `HF_HUB_OFFLINE=1`.

## Run experiments

The typed matrix CLI is the primary command interface.

List registered methods:

```bash
uv run python -m experiments.zeroshot_cf.cli list-methods
```

Resolve a matrix without running it:

```bash
uv run python -m experiments.zeroshot_cf.cli matrix \
  --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml \
  --dry-run
```

Run or resume a matrix:

```bash
uv run python -m experiments.zeroshot_cf.cli matrix \
  --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml \
  --resume
```

Aggregate the expected cells:

```bash
uv run python -m experiments.zeroshot_cf.cli aggregate \
  --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml
```

Available matrix files include:

- `one_factual_compat.yaml`: a fast compatibility matrix across all methods.
- `countercontex_ablation_example.yaml`: CounterContEx search and backend ablations.
- `full_reference.yaml`: four datasets, all six methods, and 1,000 factuals.

The full reference matrix is expensive. A recorded run took about 9.42 hours,
including 7.64 hours for the Lending Club CounterContEx cell.

## Architecture

The package uses portable contracts between each layer:

```text
CLI and compatibility commands
              |
              v
       orchestration/
    matrix, runner, artifacts
       /       |       \
      v        v        v
 datasets/  methods/  evaluation/
      \        |        /
       \       v       /
          core/contracts.py
```

### Core contracts

[`core/contracts.py`](experiments/zeroshot_cf/core/contracts.py) defines immutable
and validated data types. These types include `PreparedDataset`, `FeatureSchema`,
`BenchmarkCase`, `MethodContext`, `GenerationRequest`, and `GenerationResult`.

A method receives reference features, an action schema, and the target classifier.
It does not receive output paths, evaluation state, or test labels.

### Datasets and benchmark cases

[`datasets/`](experiments/zeroshot_cf/datasets/) owns CEL loading, preprocessing,
feature schemas, provenance fingerprints, factual selection, and benchmark-case
construction. The default case loader trains or loads the retained logistic
regression classifier.

Dataset provenance includes source hashes, the CEL revision, preprocessing,
split identity, and a fingerprint of the prepared arrays and schema.

### Methods

[`methods/`](experiments/zeroshot_cf/methods/) owns method configuration,
preparation, candidate generation, action constraints, and method-specific
diagnostics. Each method follows two phases:

1. `prepare(MethodContext)` creates reusable dataset-level state.
2. `generate(GenerationRequest)` returns canonical candidate arrays and masks.

[`methods/registry.py`](experiments/zeroshot_cf/methods/registry.py) loads methods
lazily. A new baseline does not require changes to dataset preparation, common
evaluation, or the generic runner.

CounterContEx has a second internal boundary under
[`methods/countercontex/`](experiments/zeroshot_cf/methods/countercontex/):

```text
CounterContEx method -> search -> ProposalSession -> TabICL or empirical backend
                                      ^
                                      |
                    backend identity and runtime policy
```

The proposal contract covers numerical batches, confidence conditioning,
categorical distributions, and optional joint scoring. The generic runner has no
TabICL, empirical, or CounterContEx-specific policy.

### Evaluation

[`evaluation/`](experiments/zeroshot_cf/evaluation/) calculates common metrics from
a benchmark case and canonical candidates. It does not import concrete methods.

The common report includes:

- coverage and class validity
- probability-threshold validity
- grouped Gower, Manhattan, and Euclidean proximity
- changed feature and action-unit sparsity
- immutable-feature actionability
- LOF and Isolation Forest plausibility
- out-of-bounds frequency
- returned-set coverage and diversity

Availability and validity have separate denominators. Failed or missing candidate
slots cannot enter proximity, plausibility, sparsity, or actionability metrics.

### Orchestration and artifacts

[`orchestration/runner.py`](experiments/zeroshot_cf/orchestration/runner.py) owns the
shared lifecycle:

```text
prepare case -> prepare method -> generate -> evaluate -> persist
```

[`orchestration/spec.py`](experiments/zeroshot_cf/orchestration/spec.py) separates
scientific settings from execution settings. Scientific settings and resolved
content identities determine a run ID. Output paths, devices, cache locations,
hosts, and resume flags do not change that identity.

Each completed run is a content-addressed directory with:

```text
<run-id>/
├── manifest.json
├── summary.csv
├── points.csv
├── candidates.csv
├── arrays.npz
└── COMPLETE
```

The artifact store writes strict JSON and portable NumPy arrays. It serializes
writers with a per-run file lock. It publishes `COMPLETE` only after all payload
files are ready. Aggregation rejects missing, extra, partial, duplicate, or
identity-mismatched cells.

## Repository layout

```text
CounterContEX/
├── data/                         # local benchmark data
├── docs/
│   ├── papers/                   # research notes and papers
│   └── plans/                    # implementation plans and journals
├── experiments/zeroshot_cf/
│   ├── core/                     # portable contracts and validation
│   ├── datasets/                 # providers and benchmark cases
│   ├── methods/                  # baselines, CounterContEx, and registry
│   ├── evaluation/               # common metrics and report types
│   ├── orchestration/            # specs, runner, artifacts, compatibility
│   ├── configs/matrices/         # tracked experiment definitions
│   ├── athena/                   # Slurm launch and aggregation scripts
│   ├── models/                   # ignored checkpoints and model caches
│   ├── results/                  # ignored local and Athena artifacts
│   └── tests/                    # contract and integration tests
├── pyproject.toml                # root uv workspace and test configuration
└── uv.lock                       # locked workspace dependencies
```

## Compatibility interfaces

The numbered Experiment 8, 9, and 11 through 14 modules remain available for old
commands and flat CSV or NPZ consumers. They translate arguments into `RunSpec`
and delegate to the generic runner. New automation should use the matrix CLI.

The stable low-level generator entry point remains:

```python
from experiments.zeroshot_cf.generator import generate_counterfactual_batch
```

The public `experiments.zeroshot_cf.tabicl_runtime` module remains a compatibility
re-export for existing Python callers.

A current limitation affects legacy flat-output repair. A resumed CounterContEx run can
fail if a stored diagnostic is JSON `null` and legacy export converts it directly
to `float`. The canonical run directory remains valid. This limitation does not
affect fresh canonical artifact publication.

## Verification

Run the full test suite:

```bash
uv run pytest -q
```

Run lint checks for production modules:

```bash
uv run ruff check \
  experiments/zeroshot_cf/core \
  experiments/zeroshot_cf/datasets \
  experiments/zeroshot_cf/methods \
  experiments/zeroshot_cf/evaluation \
  experiments/zeroshot_cf/orchestration \
  experiments/zeroshot_cf/cli.py
```

Run the checkpoint-backed offline smoke test:

```bash
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.tabicl_smoke_test
```

Git ignores generated vendor files, checkpoints, model caches, and results.
Review dataset and upstream dependency licenses before you
redistribute those files.

## License

See [`LICENSE`](LICENSE) and [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).
