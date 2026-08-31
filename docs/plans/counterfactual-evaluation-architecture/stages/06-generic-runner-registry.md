# Stage 6: Add the Generic Runner and Static Registry

**Goal**: Execute every dataset/method/config cell through one typed runner, explicit registry, and manifest-driven aggregation path.
**Dependencies**: Stages 4 and 5
**References**: [`architecture.md`](../resources/architecture.md#run-specifications-and-ablations), [`architecture.md`](../resources/architecture.md#artifact-contract)

---

## Steps

1. Define serializable run specifications and stable identities.
   - Where: new `orchestration/spec.py`.
   - Add typed dataset, protocol, target model, method variant, evaluation, and seed specs. Keep output location, resume flags, cache paths, device/host details, and the captured software environment in a separate execution/manifest spec.
   - Define `run_id = sha256(canonical_json(identity_payload))`. The identity payload includes the fully resolved scientific spec, dataset/case fingerprints, method/backend implementation versions, checkpoint/model content identifiers, and evaluation/schema versions; it excludes suite labels, output roots, local checkpoint/cache paths, resume flags, device, host, timestamps, and other execution-only metadata.

2. Add an explicit method registry.
   - Where: new `methods/registry.py`.
   - Map stable names to config parsers/factories for DiCoFlex, NICE, Wachter, Growing Spheres, DiCE, and FACE. Detect duplicate names and unknown/invalid parameters. Keep optional imports inside factories or `prepare()`.
   - Add the registry/help lazy-import gate deferred from Stage 4: importing/listing the registry must not import or initialize DiCE, FACE, TabICL, checkpoints, or models.

3. Implement one runner lifecycle.
   - Where: new `orchestration/runner.py`.
   - Prepare each dataset case and evaluator once, then time method preparation, generation, evaluation, and writing separately. Validate canonical output before evaluation and write the COMPLETE marker last.
   - Support resume/skip only when the existing manifest exactly matches the resolved run ID and validates as complete.

4. Add simple matrix expansion and aggregation.
   - Where: new `orchestration/matrix.py` and aggregation in `orchestration/artifacts.py`.
   - Expand dataset x method variant x seed from a small YAML/TOML manifest into concrete `RunSpec`s. Aggregate validated common summaries from manifests, never numbered filenames.
   - Add tracked `experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml` with all four datasets, all six methods, seed 42, `max_test: 1`, legacy export enabled, and output root `experiments/zeroshot_cf/results/local/architecture_one_factual`.

5. Add one generic offline-safe CLI.
   - Where: new `cli.py`.
   - Define the command surface as `single --config PATH`, `matrix --config PATH [--resume] [--dry-run]`, `aggregate --config PATH`, and `list-methods`. Parsing, dry-run, listing, and aggregation must not initialize models; matrix configs own their output root and expected cell set so aggregation can reject missing, extra, partial, or identity-mismatched runs.

---

## Verification

- [ ] GATE `uv run pytest -q experiments/zeroshot_cf/tests/test_orchestration_matrix.py experiments/zeroshot_cf/tests/test_orchestration_runner.py` — a 2-dataset x 2-fake-method matrix reads fully expanded specs/manifests; repeated case/evaluator preparation, registry coupling, wrong cell count, or incomplete artifact aggregation turns it red.
- [ ] GATE `uv run pytest -q experiments/zeroshot_cf/tests/test_orchestration_spec.py` — field ordering, output root, cache path, device, and host changes must not change identity, while dataset/case fingerprint, method/backend parameter, implementation/checkpoint version, seed, protocol, or evaluator-version changes must.
- [ ] GATE `uv run pytest -q experiments/zeroshot_cf/tests/test_method_registry.py` — duplicate/unknown entries fail and registry/list/help imports initialize no optional runtime.
- [ ] GATE `uv run python -m experiments.zeroshot_cf.cli matrix --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml --resume` followed by `uv run python -m experiments.zeroshot_cf.cli aggregate --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml` — all six real methods execute one factual for all four datasets and aggregation rejects any missing, partial, or identity-mismatched cell.
- [ ] GATE `uv run pytest -q` — the retained suite passes after the generic path is introduced.

---

## Commit

`feat(orchestration): add generic counterfactual matrix runner`
