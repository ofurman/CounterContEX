---
date: 2026-09-07T04:53:53+02:00
researcher: Oleksii Furman
topic: "Current official TabPFN v2 local APIs, checkpoint caching, offline operation, identity, and NVIDIA GB10 platform support"
tags: [tabpfn-v2, pytorch, checkpoint, offline, nvidia-gb10, aarch64]
sources: [official-docs, official-source, package-metadata, vendor-docs]
status: complete
last_updated: 2026-09-07
---

# Research: TabPFN v2 Local API and Platform Support

**Date**: 2026-09-07T04:53:53+02:00
**Researcher**: Oleksii Furman

## Research Question

Research authoritative current TabPFN v2 official APIs for classifier/regressor fitting, local model/checkpoint caching, offline operation, model/version/content identity, and supported Python/PyTorch platforms, for a CounterContEX backend on NVIDIA GB10.

## Summary

The current OSS package is `tabpfn` 8.5.0, whose bare constructors select v3; v2 must be selected explicitly with `create_default_for_version(ModelVersion.V2)` or an exact local v2 checkpoint. The estimators use sklearn-style `fit`/`predict`, with `predict_proba` for classification and multiple regression output modes. Reliable offline operation requires pre-staging the exact checkpoint because no public estimator-level offline flag exists. Package/model enum/filename identity is insufficient for benchmark identity: record the package version, resolved checkpoint path, and an independently computed cryptographic digest. Python 3.12 is supported; on GB10/aarch64, use a CUDA 13 Linux-aarch64 PyTorch wheel and validate actual kernels because Blackwell SM 12.1 gaps have existed in individual PyTorch paths.

## Detailed Findings

### Local API

- Pin v2 with `TabPFNClassifier.create_default_for_version(ModelVersion.V2)` and the corresponding regressor classmethod; bare constructors select the latest model generation ([classifier source](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/classifier.py#L543-L576), [regressor source](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/regressor.py#L567-L600)).
- Classification exposes sklearn-style `fit`, `predict`, and `predict_proba`; probabilities have shape `(n_samples, n_classes)` ([official source](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/classifier.py#L815-L829), [prediction API](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/classifier.py#L1414-L1481)).
- Regression exposes `fit` and `predict`; `output_type` supports `mean`, `median`, `mode`, `quantiles`, `main`, and `full` ([official source](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/regressor.py#L1124-L1135), [prediction API](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/regressor.py#L1211-L1305)).

### Checkpoints, cache, and offline operation

- Official v2 default filenames are `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` and `tabpfn-v2-regressor.ckpt` ([model-source table](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/model_loading.py#L80-L112)).
- `TABPFN_MODEL_CACHE_DIR` overrides the platform cache; defaults are `%APPDATA%/tabpfn`, `~/Library/Caches/tabpfn`, or `$XDG_CACHE_HOME/tabpfn`/`~/.cache/tabpfn` ([settings](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/settings.py#L24-L37), [cache resolution](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/model_loading.py#L419-L470)).
- The supported offline workflow is manual/scripted pre-download plus direct `model_path` or cache placement ([official offline instructions](https://github.com/PriorLabs/TabPFN/tree/9ed44abd5882140b88c9f2816c5791987ce059b9#q-how-do-i-use-tabpfn-without-an-internet-connection)). There is no public estimator `offline=True`; missing files enter download logic ([loader source](https://github.com/PriorLabs/TabPFN/blob/9ed44abd5882140b88c9f2816c5791987ce059b9/src/tabpfn/model_loading.py#L625-L696)).
- TabPFN v2 weights use the Prior Labs license, described by the project as Apache 2.0 plus an attribution requirement ([official README](https://github.com/PriorLabs/TabPFN/tree/9ed44abd5882140b88c9f2816c5791987ce059b9#tabpfn)).

### Identity and reproducibility

- The reproducible package anchor is PyPI `tabpfn==8.5.0`, published with provenance tied to source commit `9ed44abd…` ([PyPI release](https://pypi.org/project/tabpfn/8.5.0/)).
- `ModelVersion.V2` and the default filename select a logical model, but are not content identifiers. TabPFN's internal checkpoint cache invalidation uses file metadata rather than a cryptographic scientific identity ([model loader](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/model_loading.py)). Therefore CounterContEX should independently store SHA-256 of every resolved checkpoint along with package version and source/implementation version. This recommendation is an inference from the official implementation.

### Python, PyTorch, and NVIDIA GB10

- Current package metadata requires Python 3.10+ and PyTorch 2.5+; it enumerates Python 3.10 through 3.14 and POSIX/Unix/macOS ([official package metadata](https://github.com/PriorLabs/TabPFN/blob/main/pyproject.toml)). Python 3.12 therefore satisfies the declared package range.
- NVIDIA documents DGX Spark/GB10 as supporting PyTorch and using an ARM CPU architecture; its current stack uses CUDA 13.0.2 ([hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)).
- PyTorch publishes Linux-aarch64 CUDA 13 wheels, including Python 3.12 variants ([official wheel index](https://download.pytorch.org/whl/variant/torch/)); its support matrix lists Blackwell in CUDA 13 Linux-aarch64 builds ([PyTorch release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix)).
- GB10 is SM 12.1, and an official PyTorch issue documents missing SM 12.x dispatch in one memory-efficient-attention path even on a 2026 main-branch build ([PyTorch issue](https://github.com/pytorch/pytorch/issues/176093)). Thus package compatibility does not prove every kernel path works; run a focused TabPFN fit/predict smoke test and record `torch`, CUDA, device capability, and architecture list.

## Sources Consulted

- [TabPFN 8.5.0 on PyPI](https://pypi.org/project/tabpfn/8.5.0/) — released package and provenance.
- [TabPFN source at release provenance commit](https://github.com/PriorLabs/TabPFN/tree/9ed44abd5882140b88c9f2816c5791987ce059b9) — API, model selection, cache, offline, and license behavior.
- [PyTorch release support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix) — Python/CUDA/architecture support.
- [NVIDIA DGX Spark documentation](https://docs.nvidia.com/dgx/dgx-spark/) — GB10 platform and current CUDA stack.

## Key Insights

For CounterContEX, treat TabPFN v2 generation, package implementation, checkpoint content, estimator settings, and device/runtime as separate identities. Use the explicit local checkpoint path as the offline boundary and SHA-256 as the immutable model-content identity.

## Conflicting Information

Older examples using `TabPFNClassifier()` for v2 conflict with current 8.5.0 behavior because the default is now v3. Mutable `main` metadata currently reports a newer package version than the 8.5.0 release-provenance commit; integration should pin a released tag/commit rather than `main`.

## Confidence Notes

The package declares broad POSIX support but does not publish a TabPFN-specific GB10 certification matrix. GB10 suitability is inferred from TabPFN's PyTorch dependency plus official NVIDIA/PyTorch platform support and therefore requires a local smoke test.

## Open Questions

- Which exact licensed v2 checkpoint bytes will be staged on the target machine, and what is their SHA-256 digest?
- Does the chosen locked PyTorch/CUDA wheel pass TabPFN v2 classifier and regressor smoke tests on the specific GB10 host?

## Clarifications Log

N/A
