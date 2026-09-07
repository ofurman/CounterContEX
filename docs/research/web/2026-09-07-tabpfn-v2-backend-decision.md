---
date: 2026-09-07T04:55:31+02:00
researcher: Oleksii Furman
topic: "TabPFN v2 as CounterContEX's second conditional-proposal backend on NVIDIA GB10"
tags: [tabpfn-v2, counterfactuals, conditional-proposals, checkpoint-identity, nvidia-gb10]
sources: [official-docs, official-source, package-metadata, vendor-docs, peer-reviewed]
status: complete
last_updated: 2026-09-07
---

# Research: TabPFN v2 backend decision

**Date**: 2026-09-07T04:55:31+02:00
**Researcher**: Oleksii Furman

## Research Question

Which second foundation backend should Stage 11 implement, how can it provide CounterContEx
feature proposals, and what installation, capability, offline, and identity constraints apply on
the NVIDIA GB10 host?

## Summary

Use the released `tabpfn==8.5.0` package with the v2 classifier and regressor checkpoints selected
explicitly; bare constructors now select v3. Implement per-feature supervised conditionals:
native classifier probabilities for atomic categorical groups up to v2's 10-class limit,
normalized one-vs-rest probabilities for wider groups, and public regression quantiles for
numerical features. Declare confidence conditioning and joint scoring unsupported. Stage exact
local checkpoint bytes, hash them into run identity, and require a GB10 classifier/regressor smoke
test because Python 3.12/aarch64/CUDA 13 are supported in principle but individual SM 12.1 kernel
paths have had gaps.

## Detailed Findings

### Stable proposal surface

- TabPFN exposes sklearn-style classifier `fit`/`predict_proba` and regressor `fit`/`predict`;
  regression supports arbitrary quantiles as a public output mode
  ([classifier source](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/classifier.py#L815-L829),
  [regressor source](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/regressor.py#L1211-L1305)).
- Prior Labs' experimental unsupervised extension explicitly fits one feature as the target from
  selected other columns, using classifier probabilities for categorical targets and regressor
  distributions for numerical targets
  ([official extension](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/unsupervised/unsupervised.py)).
  This validates the feature-as-target design, while the benchmark can remain on stable core
  probabilities and quantiles.
- Atomic one-hot groups are decoded from one categorical target. TabPFN v2 supports at most ten
  classes natively; wider groups therefore use independently fitted one-vs-rest conditionals
  whose positive probabilities are normalized over the group. This approximation still emits
  one atomic categorical distribution, but is not a coherent native multiclass conditional.
- Core estimators do not condition on the downstream classifier's desired class or confidence and
  do not expose one coherent whole-row joint density. The backend must therefore declare
  `confidence_conditioning=False` and `joint_scoring=False`; validity remains the search/oracle's
  responsibility.

### Version, checkpoints, and offline identity

- The current released package is `tabpfn==8.5.0`, with PyPI provenance tied to commit
  `9ed44abd...`; bare constructors select v3. V2 must be selected explicitly via
  `create_default_for_version(ModelVersion.V2)` or exact local model paths
  ([PyPI release](https://pypi.org/project/tabpfn/8.5.0/),
  [classifier factory](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/classifier.py#L543-L576)).
- Official v2 files are `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` and
  `tabpfn-v2-regressor.ckpt`
  ([model table](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/model_loading.py#L80-L112)).
  Missing files can trigger network fallbacks, so strict offline execution must require staged
  files and pass absolute paths
  ([official offline instructions](https://github.com/PriorLabs/TabPFN/tree/9ed44abd5882140b88c9f2816c5791987ce059b9#q-how-do-i-use-tabpfn-without-an-internet-connection)).
- Model version and filename are not content identities. CounterContEX must independently SHA-256
  both checkpoint files and include those digests plus a distinct backend implementation version
  in scientific identity. This is inferred from the official loader, whose cache bookkeeping is
  not a cryptographic benchmark identity
  ([model loader](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/model_loading.py)).
- V2 weights use the Prior Labs license, described as Apache 2.0 plus attribution
  ([official license summary](https://github.com/PriorLabs/TabPFN/tree/9ed44abd5882140b88c9f2816c5791987ce059b9#tabpfn)).

### GB10 feasibility

- TabPFN metadata supports Python 3.10+ and PyTorch 2.5+; PyTorch publishes Python 3.12 Linux
  aarch64 CUDA 13 wheels and lists Blackwell support in that build family
  ([TabPFN metadata](https://github.com/PriorLabs/TabPFN/blob/main/pyproject.toml),
  [PyTorch support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix)).
- NVIDIA's current DGX Spark stack uses CUDA 13.0.2
  ([release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)). GB10 is SM 12.1,
  and an official PyTorch issue records a missing SM 12.x attention-kernel path, so local
  classifier and regressor fit/predict smoke tests are mandatory
  ([PyTorch issue](https://github.com/pytorch/pytorch/issues/176093)).

## Sources Consulted

- [TabPFN release and source](https://github.com/PriorLabs/TabPFN/tree/9ed44abd5882140b88c9f2816c5791987ce059b9) — released API, v2 selection, model loading, and license.
- [TabPFN unsupervised extension](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/unsupervised/unsupervised.py) — official feature-as-target precedent.
- [Prior Labs regression guide](https://docs.priorlabs.ai/capabilities/regression) — public regression outputs.
- [PyTorch release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix) — aarch64/CUDA/Blackwell support.
- [NVIDIA DGX Spark documentation](https://docs.nvidia.com/dgx/dgx-spark/) — GB10 platform stack.
- [TabPFN v2 Nature paper](https://www.nature.com/articles/s41586-024-08328-6) — model scope and scientific context.

## Key Insights

The portable `ProposalSession` boundary is sufficient: TabPFN supplies conditional feature
distributions while CounterContEx retains target-class search, action legality, and evaluation.
The honest comparison is therefore two foundation proposal mechanisms under the shared C3 search,
not an attempt to imitate TabICL-only confidence or joint-density features.
Adult Census exercises the documented normalized one-vs-rest fallback in factual contexts where
education or occupation has more than ten observed categories.

## Conflicting Information

Older examples treat `TabPFNClassifier()` as v2, whereas the current release defaults to v3.
Stage 11 follows the release-pinned factory/path behavior rather than mutable examples.

## Confidence Notes

The package's declared platform range does not constitute TabPFN-specific GB10 certification.
Actual checkpoint access, hashes, and kernel compatibility are local measurements. On the
`gx10-bdc5` GB10 host, both explicitly selected v2 estimators completed fit/predict smoke tests
under the locked environment. The staged classifier SHA-256 is
`cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`; the staged regressor
SHA-256 is `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`.

## Open Questions

None for Stage 11. Exact checkpoint bytes were hashed and both v2 estimator paths passed local
GB10 fit/predict smoke tests before the E9 campaign run.

## Clarifications Log

- 2026-09-07: Resolved the two local-measurement questions with staged checkpoint hashes and
  successful classifier/regressor GB10 smoke tests.
- 2026-09-07: Clarified the v2 ten-class limit and normalized one-vs-rest fallback used by wider
  atomic groups; the original summary implied native multiclass conditionals at every width.
