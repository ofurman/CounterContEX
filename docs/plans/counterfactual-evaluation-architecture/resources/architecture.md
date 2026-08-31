# Target Architecture

## Design objective

A new counterfactual method should require one typed method adapter and registration entry. It
must not require a new dataset loader, evaluator, result schema, aggregation script, or numbered
runner. A DiCoFlex foundation-model ablation should require one proposal-backend adapter and a
configuration variant; it must not appear in the evaluation layer.

The architecture stays deliberately small:

```text
DatasetProvider -> PreparedDataset -> EvaluationProtocol -> BenchmarkCase
                                                        |            |
                                                        |            v
                                                        |   CounterfactualMethod
                                                        |     prepare/generate
                                                        |            |
                                                        +------------v
                                                                  GenerationResult
                                                                        |
                                                                        v
                                                                    Evaluator
                                                                        |
                                                                        v
                                                                 EvaluationReport
                                                                        |
                                                                        v
                                                                  ArtifactStore

RunSpec -> MatrixExpander -> Runner -> explicit MethodRegistry
```

Only orchestration knows all layers. Dependencies point toward portable contracts:

```text
core <- datasets
core <- methods
core <- evaluation
core <- orchestration -> datasets/methods/evaluation
cli  -> orchestration
```

`datasets` and `evaluation` must not import concrete methods, numbered runners, CEL internals, or
TabICL modules. Concrete methods may import `core`; only their adapters import optional third-party
libraries. Numbered scripts become translation shims into orchestration.

## Proposed package structure

```text
experiments/zeroshot_cf/
  core/
    contracts.py          # arrays, schemas, predictor/method protocols, results
    validation.py         # shape, mask, JSON metadata, and immutability checks
  datasets/
    base.py               # DatasetSpec, DatasetProvider, provenance
    cel.py                # CEL-only loading and inverse-transform adapter
    benchmark.py          # split, scaling, factual selection, target policy/model
  methods/
    base.py               # method protocols and typed config helpers
    registry.py           # explicit name -> factory map
    nice.py
    optimization.py       # Wachter and Growing Spheres
    dice.py
    face.py
    dicoflex/
      method.py           # complete benchmark-facing method
      search.py           # current greedy/diverse search
      config.py           # search/diversity/foundation config composition
      backends/
        base.py           # proposal capabilities and protocols
        tabicl.py         # current TabICL preparation/runtime adapter
  evaluation/
    evaluator.py          # common primary/set evaluation, no method imports
    metrics.py            # pure kernels, migrated from metrics_harness/reporting
    result.py             # typed summary, point, and candidate results
  orchestration/
    spec.py               # RunSpec, MethodSpec, EvaluationSpec
    matrix.py             # expand concrete dataset x method x backend x seed cases
    runner.py             # prepare/generate/evaluate lifecycle and phase timing
    artifacts.py          # manifest store, COMPLETE marker, aggregation
    legacy.py             # temporary v1 CSV/NPZ compatibility export
  cli.py                  # one generic offline-safe CLI
  exp8/9/11-14*.py        # temporary argument-translation shims only
```

`datasets/` avoids colliding with the current `data.py`. Moving the whole project to
`src/countercontex` is not required to establish stable internal boundaries and is outside this
plan.

## Core data contracts

```python
@dataclass(frozen=True)
class FeatureSchema:
    names: tuple[str, ...]
    numerical: tuple[int, ...]                 # feature type, not actionability
    categorical_groups: tuple[OneHotActionGroup, ...]
    actionable_scalars: tuple[int, ...]
    actionable_groups: tuple[OneHotActionGroup, ...]
    immutable: tuple[int, ...]
    domains: FeatureDomains

@dataclass(frozen=True)
class DatasetProvenance:
    provider: str
    source_revision: str
    source_hashes: Mapping[str, str]
    preprocessing_id: str
    split_id: str
    fingerprint: str

@dataclass(frozen=True)
class PreparedDataset:
    name: str
    X_train: NDArray
    y_train: NDArray
    X_validation: NDArray
    y_validation: NDArray
    X_test: NDArray
    y_test: NDArray
    schema: FeatureSchema
    provenance: DatasetProvenance

class DatasetProvider(Protocol):
    def prepare(self, spec: DatasetSpec) -> PreparedDataset: ...
```

Arrays shared across matrix cells should be read-only. CEL's `MethodDataset`, dataframe codecs,
and inverse transforms stay inside `CelDatasetProvider` or method-specific adapters.

The evaluation protocol turns a prepared dataset into one reusable case:

```python
@dataclass(frozen=True)
class FactualSelection:
    indices: NDArray[np.int_]
    values: NDArray[np.float64]
    true_labels: NDArray[np.int_]

@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    dataset: PreparedDataset
    factuals: FactualSelection
    oracle: Predictor
    factual_predictions: NDArray[np.int_]
    targets: NDArray[np.int_]
    protocol: BenchmarkProtocol
```

`case_id` hashes dataset provenance, split/preprocessing, target-model config, selection config,
and target policy. Methods receive a narrower view without test truth or output paths:

```python
@dataclass(frozen=True)
class MethodContext:
    X_reference: NDArray
    feature_schema: FeatureSchema
    oracle: Predictor

@dataclass(frozen=True)
class GenerationRequest:
    factuals: NDArray
    targets: NDArray
    n_counterfactuals: int
    seed: int
```

`Predictor.predict_proba()` must expose `classes_`; probability lookup maps target labels through
that array instead of assuming label values are column indices.

## Method contract

Dataset-level setup and point/batch generation are distinct because FACE builds a graph, NICE
fits neighbor/LOF state, DiCE builds a codec/explainer, and DiCoFlex loads a foundation model.

```python
class CounterfactualMethod(Protocol):
    method_id: str
    capabilities: MethodCapabilities

    def config_dict(self) -> Mapping[str, JsonValue]: ...
    def prepare(self, context: MethodContext) -> PreparedMethod: ...

class PreparedMethod(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
```

The method owns algorithm setup, search, legal-action enforcement intrinsic to the algorithm,
postprocessing such as DiCE atomic pruning, randomness, and method-specific diagnostics. It never
owns dataset selection, ground-truth test labels, common metrics, result paths, aggregation, or CSV
formatting.

## Canonical generation result

```python
@dataclass(frozen=True)
class GenerationResult:
    candidates: NDArray[np.float64]          # (n_factuals, k, n_features)
    available: NDArray[np.bool_]             # (n_factuals, k)
    point_diagnostics: tuple[Mapping[str, JsonValue], ...] = ()
    run_diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)
    artifacts: Mapping[str, NDArray] = field(default_factory=dict)
```

Rules:

- `k=1` baselines still return three-dimensional `candidates`.
- Unavailable slots have `available=False` and NaN payloads; they are never filled with duplicate
  or factual rows.
- Best-effort/factual fallbacks may be retained in namespaced `artifacts`, but are not available
  counterfactuals.
- The evaluator derives all common metrics from `BenchmarkCase`, `candidates`, and `available`.
  It never interprets `point_diagnostics` or `run_diagnostics`.
- Diagnostics are JSON-serializable and namespaced. Raw candidates and large arrays go in
  `artifacts`, not the common summary schema.
- A validator rejects shape mismatches, non-NaN unavailable slots, non-finite available slots,
  and diagnostics with incompatible row counts.

## Evaluation contract

```python
@dataclass(frozen=True)
class EvaluationSpec:
    metric_version: str
    sparsity_epsilon: float
    probability_threshold: float
    primary_rank: int = 0

class Evaluator:
    def prepare(self, case: BenchmarkCase, spec: EvaluationSpec) -> PreparedEvaluator: ...

class PreparedEvaluator:
    def evaluate(self, result: GenerationResult) -> EvaluationReport: ...
```

Preparation fits reusable dataset-level metric state such as LOF and Isolation Forest once per
case. Evaluation owns:

- `coverage`: factuals with at least one available candidate / factual count.
- `validity_returned_class`: target-class candidates / available candidates.
- `validity_returned_threshold`: candidates satisfying target class and probability threshold /
  available candidates.
- `valid_success_rate_class` and `valid_success_rate_threshold`: corresponding successes /
  requested slots or factuals, with the denominator named explicitly.
- actionability, immutable preservation, exact and epsilon sparsity, action-unit sparsity, grouped
  Gower, continuous distances, OOB rate, LOF, and Isolation Forest scores.
- primary-rank metrics and separate set coverage/diversity metrics.
- common factual/candidate rows and oracle predictions/probabilities.

Proximity remains valid-only. Class validity and threshold validity are both reported; neither is
silently substituted for the other. Method-specific success signals can be published under a
`method.*` namespace but do not define common validity.

## Foundation-model boundary inside DiCoFlex

Evaluation sees only `CounterfactualMethod`. DiCoFlex composes search and a proposal backend:

```python
@dataclass(frozen=True)
class ProposalCapabilities:
    confidence_conditioning: bool
    categorical_distribution: bool
    joint_scoring: bool

class ProposalBackend(Protocol):
    backend_id: str
    capabilities: ProposalCapabilities
    def prepare(self, context: MethodContext) -> PreparedProposalBackend: ...

class PreparedProposalBackend(Protocol):
    def for_factual(self, factual, target, *, seed: int) -> ProposalSession: ...

class ProposalSession(Protocol):
    def propose_numerical(self, rows, columns, *, quantiles, confidence) -> NDArray: ...
    def categorical_distribution(self, row, group, *, confidence) -> CategoryProposals: ...
    def score_joint(self, rows, target) -> NDArray: ...
```

TabICL owns compact one-hot representation, context selection, checkpoints, device choice,
confidence anchors, and cache behavior. Future TabPFN/TabFM adapters translate their native
capabilities into this contract. Configuration validation rejects, for example, joint-density
refinement with a backend whose `joint_scoring` capability is false. The search never uses
`Any`/`hasattr()` capability probing.

DiCoFlex config separates independent ablation axes:

```python
@dataclass(frozen=True)
class DiCoFlexConfig:
    search: SearchConfig
    diversity: DiversityConfig
    foundation: FoundationSpec
```

## Run specifications and ablations

```python
@dataclass(frozen=True)
class RunSpec:
    dataset: DatasetSpec
    protocol: BenchmarkProtocol
    method: MethodSpec
    evaluation: EvaluationSpec
    seed: int

@dataclass(frozen=True)
class MethodSpec:
    name: str
    variant: str
    params: Mapping[str, JsonValue]
```

The explicit registry maps `MethodSpec.name` to a factory. Each method owns a frozen typed config
and validates its own parameters. The runner never grows method-specific flags.

A small YAML/TOML suite manifest may list concrete variants or sweep selected fields. Matrix
expansion materializes each fully resolved `RunSpec` before execution. The `run_id` is a canonical
hash of the resolved run spec, dataset/case fingerprint, method implementation/version, and
evaluation version. Changing any method, backend, hyperparameter, seed, dataset, protocol, or
evaluator creates a distinct run; field ordering does not.

The runner lifecycle is fixed:

```text
prepare dataset/case once
  -> create and prepare method
  -> generate canonical result
  -> validate result
  -> evaluate with prepared evaluator
  -> assemble manifest and artifacts
  -> write atomically and mark COMPLETE
```

Timing fields are uniformly owned by the runner: `prepare_s`, `generate_s`, `evaluate_s`,
`write_s`, and `total_s`. Method diagnostics may subdivide phases but cannot redefine them.

## Artifact contract

```text
results/<suite>/<run_id>/
  manifest.json       # complete resolved config, versions, provenance, environment
  summary.csv         # one stable common-summary row
  points.csv          # common factual/primary-candidate rows + method.* diagnostics
  candidates.csv      # optional long-form candidate rows for k > 1
  arrays.npz          # canonical arrays + namespaced method arrays
  COMPLETE            # written last
```

Aggregation scans validated manifests and COMPLETE markers; it never infers identity from
experiment-number filenames or first-seen CSV column order. Partial directories are resumable but
not aggregateable. A temporary compatibility exporter writes current v1 filenames, columns, and
NPZ keys for numbered CLI consumers.

## Adding a new baseline

1. Define a frozen typed config.
2. Implement `CounterfactualMethod.prepare()` and `PreparedMethod.generate()`.
3. Return validated `GenerationResult`; keep optional details namespaced.
4. Add one explicit registry entry.
5. Run shared method contract tests plus focused algorithm tests.

No data, evaluator, artifact, aggregation, or runner changes are expected. An architecture test
uses a fake method to enforce this claim.

## Adding a foundation-model ablation

1. Implement `ProposalBackend` and declare capabilities.
2. Add backend-specific config resolution inside DiCoFlex.
3. Run backend contract tests and capability-rejection tests.
4. Add a `MethodSpec` variant such as `dicoflex-tabpfn` or `dicoflex-tabfm`.

The method registry still contains one `dicoflex` method. The evaluator and data layers do not
import or name the backend.

## Migration and compatibility principles

- Wrap existing algorithms before moving or simplifying them.
- Keep old CLIs as offline-safe translators until the generic runner reproduces the v1 surface.
- Preserve `generate_counterfactual_batch()` as a public compatibility API.
- Use synthetic semantic tests and one-factual parity gates on every stage.
- Reserve the 9.42-hour full matrix for the final REPORT, not repeated stage gates.
- Document intentional metric changes, especially truthful coverage and split validity fields;
  do not force new semantics to equal misleading legacy values.
- Keep optional libraries lazily imported inside method/backend factories.
- Fit evaluation-only state once per benchmark case and reuse it across matrix cells.
