# Journal

Append-only. Newest entries at the bottom. Never rewrite an earlier entry.

One entry per invocation, in this shape:

```
## YYYY-MM-DD HH:MM -- Stage N: [Name] -- DONE
**Did**: [1-3 lines]
**Verification**: GATE lines passed. REPORT values: [metric]=[value]
**Provenance**: [per measured GATE: the input the value was read from, and the defect that would
turn it red] - [or `NOT MEASURED` for any that could not be produced from this run's own inputs]
**Problems**: [symptom -> root cause -> resolution -> inline/subagent] or "none"
**Commit**: `abc1234`
```

---

## 2026-08-31 15:25 -- Stage 1: Freeze Contracts and Compatibility Evidence -- DONE
**Did**: Added a machine-readable v1 inventory for six methods and eight documented CLIs, reasoned availability/validity fixtures, source-scoped dependency evidence, and executable offline/checkpoint-free compatibility tests. The generation/evaluation helpers in these tests are frozen specifications until Stage 3 connects them to production contracts.
**Verification**: GATE 1 passed: the required dataset/protocol/metric/generator command passed `24` tests. GATE 2 passed: all eight retained `--help` subprocesses and Exp9 aggregation ran with `HF_HUB_OFFLINE=1`, checkpoint-loader spies, and optional-model import spies. Full suite passed `117` tests with `5` warnings in `14.14s`. REPORT full-reference evidence: `24/24` cells, `runtime_total_s=33905.925`, seed `42`, `n_test=1000`; aggregate SHA-256 values were Exp11 `2f18a36d3a2bd2afee2be9f9ebb3d07cc96fd9b064d3d8b163934b929815885e`, Exp12 `142cfb9ab232667ea583eb67f2625b6d6c1147bffee5d2c66c9fd71d43e8e8b6`, Exp13 `4e23866b6e24288a34b0e95a3d852a055b42c4aabf1910c05c5efc24a243d872`, Exp14 `75bcb45b721428296da75c7525a20e121a8a431fa1e0d2bc22df7b2dbc73d8cb`, and Exp9 `873bf7cd128f71fb2828f1782bc374b951baa81e94a56ec4cfeee749d10e0413`.
**Provenance**: GATE 1 reads the pinned CEL files and tracked dataset contract, production benchmark selection/target logic, production metric kernels, and `generate_counterfactual_batch()` with reasoned arrays; split, target-policy, formula, immutability, or duplicate/factual-padding defects turn named tests red. GATE 2 reads `compatibility.json`, real CLI parsers, generated per-dataset aggregate inputs, actual aggregate output, and marker files written by spies for `torch.load`, `joblib.load`, Hugging Face download, TabICL, TabPFN, DiCE, or CEL imports; eager model/checkpoint work, a lost command/token, or aggregation failure turns it red. Method IDs, stems, common point columns, and exact NPZ keys are checked against current runner ASTs and production constants, rather than echoed fixture values. A fresh verifier independently probed the spies and audited both gates; no literal, default, or count-generated value supplies a gate verdict.
**Problems**: The first offline gate omitted the documented checkpoint and smoke-test CLIs, and the first compatibility/boundary assertions were partly self-fulfilling -> the provenance audit caught both -> added the two commands, loader/model-import spies, runner-source AST checks, and accurately scoped import monitoring. A decorator was initially attached to the spy helper instead of the parametrized test -> a focused fix subagent moved it and reran the tests. The optional full-reference directory was present, so its aggregate rows, configuration fields, runtimes, and hashes were measured rather than inferred.
**Commit**: `pending`

## 2026-08-31 16:15 -- Stage 2: Separate Portable Data and Benchmark Cases -- DONE
**Did**: Added frozen provider-neutral core contracts, validation, a CEL dataset provider with a separate native compatibility adapter, and reusable benchmark-case construction. `data.py` and `benchmark_protocol.py` now delegate through the new layers while retaining their public compatibility views.
**Verification**: GATE 1 plus the new contract gate passed `22` focused tests with `4` warnings in `2.46s`. GATE 3 passed the full `125`-test suite with `5` warnings in `14.76s`. Scoped Ruff and `git diff --check` passed. The pinned contract now records exact prepared provenance for HELOC `46e8421bbc16165e26616282575e6c94e99bb4519ee2f24d96579635da818349`, Bank Marketing `a3ac75c48c8485d24a74f01ddd0ac43db920d0c77347d45d6308aae99c0779fd`, Give Me Some Credit `8ed89b4fb6e7ebcd48bcddcd156f39fc5654b0fbe35aa5e2adb3857fec57cdb5`, and Lending Club `78af68c5ff118055e6d5b43b8aec7f9fd28de1dfe5e0039ba35ffae45d173c04`.
**Provenance**: The pinned gate reads the live CEL YAML/CSV files, local HELOC actionability config, all six prepared arrays, train-only scaler outputs, schema/type/action partitions, source hashes, split/preprocessing IDs, and four frozen fingerprints; source, cleaning, split, scaling, feature, actionability, or provenance drift turns it red. Synthetic contract checks read owned arrays/mappings and inject invalid partitions, mutable inputs, wrong factual rows/labels, duplicate/out-of-range indices, reversed classifier labels, and model/selection mutations; each named defect turns a focused assertion red. `case_id` is derived from the measured dataset fingerprint, factual selection, target policy, required resolved estimator parameters, implementation source, classes, and fitted predictor state. A fresh verifier traced the compatibility path through `CelDatasetProvider` and `build_benchmark_case()` and approved all three gates.
**Problems**: The first audit found that factual indices were only bounds-checked, dataset identity omitted schema/actionability, model identity could be an empty caller literal, and real prepared fingerprints were not frozen -> added exact NaN-aware row/label binding, complete schema/source fingerprinting, fitted-estimator identity, sensitivity tests, and pinned provenance for all four datasets -> focused fix subagent. Whole-directory Ruff still reports pre-existing errors in unrelated retained modules; every Stage 2 owned file passes scoped Ruff.
**Commit**: `pending`
