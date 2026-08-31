# Stage 6: Add the Generic Runner and Static Registry

**Goal**: Execute every dataset/method/config cell through one typed runner, explicit registry, and manifest-driven aggregation path.
**Dependencies**: Stages 4 and 5
**References**: [`architecture.md`](../resources/architecture.md#run-specifications-and-ablations), [`architecture.md`](../resources/architecture.md#artifact-contract)

---

## Steps

1. Define serializable run specifications and stable identities.
   - Where: new `orchestration/spec.py`.
   - Add typed dataset, protocol, target model, method variant, evaluation, seed, output, and environment specs. Canonical JSON serialization and hashes include resolved defaults, implementation/schema versions, and dataset/case fingerprints.

2. Add an explicit method registry.
   - Where: new `methods/registry.py`.
   - Map stable names to config parsers/factories for DiCoFlex, NICE, Wachter, Growing Spheres, DiCE, and FACE. Detect duplicate names and unknown/invalid parameters. Keep optional imports inside factories or `prepare()`.

3. Implement one runner lifecycle.
   - Where: new `orchestration/runner.py`.
   - Prepare each dataset case and evaluator once, then time method preparation, generation, evaluation, and writing separately. Validate canonical output before evaluation and write the COMPLETE marker last.
   - Support resume/skip only when the existing manifest exactly matches the resolved run ID and validates as complete.

4. Add simple matrix expansion and aggregation.
   - Where: new `orchestration/matrix.py` and aggregation in `orchestration/artifacts.py`.
   - Expand dataset x method variant x seed from a small YAML/TOML manifest into concrete `RunSpec`s. Aggregate validated common summaries from manifests, never numbered filenames.

5. Add one generic offline-safe CLI.
   - Where: new `cli.py`.
   - Support single-run, matrix, aggregate, dry-run/list, and resume modes without method-specific flags or model initialization during parsing/listing.

---

## Verification

- [ ] GATE a 2-dataset x 2-fake-method matrix integration test reads fully expanded specs/manifests — repeated case/evaluator preparation, registry coupling, wrong cell count, or incomplete artifact aggregation turns it red.
- [ ] GATE run-ID tests vary one real input at a time — config field ordering must not change identity, while dataset fingerprint, method/backend parameter, seed, protocol, or evaluator-version changes must.
- [ ] GATE all six real methods execute one factual through the generic runner and `uv run pytest -q` passes — a method-specific runner requirement or suite regression turns the gate red.

---

## Commit

`feat(orchestration): add generic counterfactual matrix runner`
