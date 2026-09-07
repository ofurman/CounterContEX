---
date: 2026-09-07T04:53:11+02:00
researcher: Oleksii Furman
topic: "Current TabPFN v2 APIs for conditional feature proposals in counterfactual search"
tags: [TabPFN, counterfactuals, conditional-proposals, classification, regression]
sources: [official-docs, official-github, peer-reviewed]
status: complete
last_updated: 2026-09-07
---

# Research: TabPFN v2 conditional feature proposals

**Date**: 2026-09-07T04:53:11+02:00
**Researcher**: Oleksii Furman

## Research Question

How can official/current TabPFN v2 classifier and regressor APIs support conditional feature proposals for tabular counterfactual search, and what do they not support?

## Summary

TabPFN can honestly be used as a collection of supervised conditional models: for feature `j`, fit the other selected columns as `X` and feature `j` as `y`. The classifier yields probabilities for a categorical feature; the regressor yields point estimates, arbitrary quantiles, and a full binned predictive distribution that can be sampled. Prior Labs' experimental unsupervised extension implements exactly this pattern. The core APIs do not expose conditioning on a desired downstream-classifier confidence, nor a single normalized joint density over a whole row; chained conditionals are an additional, order-sensitive construction.

## Detailed Findings

### Categorical feature proposals

- `TabPFNClassifier.fit(X, y)` plus `predict_proba(X_query)` supplies a conditional categorical distribution when the feature to change is encoded as the target and the remaining/context features are inputs ([official classifier source](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py)). Classes are label-encoded during fit and probabilities correspond to `classes_`.
- For an atomic one-hot group, the scientifically clean target is one category label, not one independent binary model per dummy. Sample or rank a category from `predict_proba`, then decode exactly one active dummy. This atomic-group advice is an application inference, not a dedicated TabPFN API.
- The exact v2 checkpoint can be selected with `create_default_for_version(ModelVersion.V2)` ([official repository README](https://github.com/PriorLabs/TabPFN#using-the-tabpfn-v2-model)). Current defaults may target newer checkpoints, so model version and package/checkpoint identity must be pinned.
- v2 classification has a small supported class vocabulary (official inference configuration exposes a 10-class pretraining limit); high-cardinality categorical targets need rejection, grouping, or the separate experimental many-class extension ([official inference config](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/inference_config.py), [extensions repository](https://github.com/PriorLabs/tabpfn-extensions)).

### Numerical feature proposals

- `TabPFNRegressor.predict` supports `mean`, `median`, `mode`, and arbitrary requested `quantiles`; `output_type="full"` returns those summaries plus a distribution criterion and logits ([official regression documentation](https://docs.priorlabs.ai/capabilities/regression), [official regressor source](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/regressor.py)).
- Quantiles can define deterministic proposal grids or intervals. Full output supports genuine predictive-distribution sampling through the returned criterion/logits, but this is a lower-level PyTorch-facing interface and should be version-tested rather than treated as a stable NumPy sampling method.
- Prior Labs' extension demonstrates `criterion.sample(logits, t=...)` for numeric targets and `torch.distributions.Categorical(probs=...)` for categorical targets ([official extension source](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/unsupervised/unsupervised.py)). Temperature here is proposal-sampling policy; it is not conditioning on downstream target confidence.

### Conditioning and joint scoring

- The experimental `TabPFNUnsupervisedModel.density_` explicitly models one feature conditional on selected others, routing declared categorical columns to the classifier and numerical columns to the regressor ([official extension source](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/unsupervised/unsupervised.py)). This is strong evidence that feature-as-target use is intended and feasible.
- Neither core estimator accepts a requested downstream class, probability threshold, or confidence value as a conditioning argument. Such conditioning would require making those quantities observed predictor columns in a carefully constructed training table, rejection/reranking with the actual task classifier, or a new model. The classifier's `softmax_temperature` calibrates/sharpens its own output; it does not impose counterfactual target confidence.
- A single feature model provides `p(x_j | x_-j)` (relative to its supervised predictive model), not `p(x_1,...,x_d)`. Sequential products of conditionals can form an autoregressive score only after choosing an order and refitting/using the relevant prefix conditionals. Different orders need not agree, so this should be named an order-dependent chain score, not an exact coherent joint likelihood.
- The extensions repository labels its code experimental and less rigorously tested, with APIs subject to change ([official extensions README](https://github.com/PriorLabs/tabpfn-extensions)). Its recent changelog also documents a repaired chain-rule density bug, further supporting caution ([official changelog](https://github.com/PriorLabs/tabpfn-extensions/blob/main/CHANGELOG.md)).

## Sources Consulted

- [TabPFN classifier source](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py) — fit, class encoding, probabilities, limits, temperature.
- [TabPFN regressor source](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/regressor.py) — output types and full distribution.
- [Prior Labs regression guide](https://docs.priorlabs.ai/capabilities/regression) — supported public prediction outputs and quantile/full examples.
- [TabPFN unsupervised extension](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/unsupervised/unsupervised.py) — official experimental feature-as-target, sampling, and conditional-density implementation.
- [TabPFN extensions README](https://github.com/PriorLabs/tabpfn-extensions) — experimental stability warning.
- [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) — TabPFN v2 model context and supervised foundation-model scope.

## Key Insights

The best-supported integration is a proposal oracle per actionable feature, with task-classifier validity checked separately. Cache one fitted conditional model per feature/context schema, use class probabilities for atomic categoricals, and use quantiles or distribution samples for numerics. Treat proposal probability as a local plausibility signal, not as proof of causal feasibility or whole-row likelihood.

## Confidence Notes

The exact stability guarantee of `criterion.sample` is unclear: it is used by Prior Labs' own experimental extension, but the public guide foregrounds quantiles and distribution internals rather than promising a stable sampling facade. Claims about incompatibility of separately fit full-conditionals with one coherent joint are a statistical inference from the API design, not an explicit vendor disclaimer.

## Open Questions

- Whether CounterContEx should depend only on stable core outputs (probabilities and quantiles) or optionally adopt the experimental full-distribution sampling interface.

## Clarifications Log

N/A
