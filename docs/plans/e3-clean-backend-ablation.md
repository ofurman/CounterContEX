# E3: Clean Proposal-Backend Attribution

## Question

Does CounterContEx benefit from TabICL's learned conditional proposals beyond
the locality available to a simple empirical estimator?

The historical `campaign_e3_backend.yaml` run remains unchanged. It compared
TabICL with global empirical proposals at `k=3`; this experiment uses a new
identity and output root.

## Hypothesis and estimand

The preregistered hypothesis is that TabICL improves
`valid_success_rate_threshold_per_requested_slot` at evaluation threshold 0.7
relative to `empirical_local`.

The primary contrast is `tabicl - empirical_local`, computed over every matched
factual in each dataset and target-model block. Positive values favour TabICL.
Two secondary contrasts decompose the result:

- `empirical_local - empirical` measures the effect of factual-local context.
- `tabicl - empirical` measures the complete backend-bundle effect.

Scientific outcomes are reports, never completion gates.

## Frozen protocol

- Datasets: HELOC, Bank Marketing, Give Me Some Credit, Lending Club.
- Target models: retained logistic regression, MLP, and XGBoost.
- Factuals: 250 deterministic stratified factuals per dataset/model block.
- Generation seed: 42. Repeated empirical seed cells are deliberately omitted.
- Requested counterfactuals: `k=1`.
- Search: sparse, numerical quantiles 0.1 through 0.9, at most 100 validity
  steps, revisits enabled, generation threshold 0.5.
- TabICL ensemble size: one estimator; proposal temperature `1e-9` (the fixed
  quantile-grid path itself is deterministic).
- Confidence conditioning and joint scoring: disabled.
- Evaluation: `countercontex.evaluation.v2`, threshold 0.7.
- Matrix: `experiments/zeroshot_cf/configs/matrices/campaign_e3_clean_backend.yaml`.
- Output root: `experiments/zeroshot_cf/results/campaign/e3_clean_backend`.

The three arms share the benchmark case, factuals, targets, action schema,
search, proposal cardinality, evaluator, and execution limits:

- `empirical`: global classifier-target-class marginal quantiles and smoothed
  categorical frequencies.
- `empirical_local`: the same statistics after selecting the factual's up-to-512
  mixed-Gower-nearest reference rows and retaining target-class rows. It falls
  back to global target-class rows if the local context contains none.
- `tabicl`: learned row- and target-conditional proposals from the corresponding
  factual-local context.

## Reporting

Report the primary metric first, followed by coverage and class success. Then
report all applicable single-counterfactual proximity, sparsity, plausibility,
actionability, bounds, and phase-timing metrics. Set/diversity metrics do not
apply at `k=1`.

Paired differences use identical factual positions. Binary success metrics use
all requested factuals, including unavailable candidates. Conditional metrics
use only factuals on which both arms define the metric and report that count.
Within-block 95% percentile intervals use 2,000 paired factual bootstrap
resamples with seed 42. These intervals describe the selected factual sample;
they do not represent alternative dataset splits or independent datasets.

## Execution gates

Before submission:

1. The matrix dry-run resolves exactly 36 unique cells.
2. Within every dataset/model block, scientific specifications differ only in
   the declared backend; resolved identities additionally carry backend-owned
   implementation and checkpoint identities.
3. Focused backend, matrix, paired-analysis, and PLGrid launcher tests pass.
4. The full test suite and Ruff checks pass, apart from explicitly demonstrated
   environment-only failures.
5. Slurm `--test-only` accepts the live allocation and job shape.

After execution, strict aggregation must accept exactly 36 COMPLETE runs before
the artifact-only paired analysis is built or interpreted.
