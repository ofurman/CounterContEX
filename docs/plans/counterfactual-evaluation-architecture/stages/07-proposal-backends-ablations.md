# Stage 7: Add Proposal Backends and Ablation Specifications

**Goal**: Isolate DiCoFlex search from TabICL through capability-checked proposal backends and prove foundation/hyperparameter ablations need no evaluator or data changes.
**Dependencies**: Stages 5 and 6
**Reference**: [`architecture.md`](../resources/architecture.md#foundation-model-boundary-inside-dicoflex)

---

## Steps

1. Define proposal-backend contracts and explicit capabilities.
   - Where: new `methods/dicoflex/backends/base.py`.
   - Add `ProposalBackend`, prepared backend/session protocols, numerical proposals, categorical distributions, optional joint scoring, and capability/config validation.
   - Replace `Any`/`hasattr()` capability probing with protocol methods and declared flags.

2. Move TabICL-specific preparation behind the adapter.
   - Where: new `methods/dicoflex/backends/tabicl.py`; current `tabicl_runtime.py`, `tabicl_sampler.py`, `tabicl_joint_plausibility.py`, and checkpoint helpers become its implementation or compatibility delegates.
   - Keep compact representation, neighbor context, confidence anchors, device/checkpoint/cache details, and TabICL diagnostics inside this adapter.

3. Make DiCoFlex search backend-neutral.
   - Where: `DiCoFlexMethod` and current generator/search internals.
   - Search consumes only proposal sessions/capabilities. Invalid combinations, such as requested joint refinement without joint scoring, fail during config validation or `prepare()` with a targeted message.

4. Prove the future extension seam with a fake backend.
   - Where: backend contract fixtures and generic matrix manifests in tests.
   - Run the same DiCoFlex search with TabICL-shaped and fake deterministic backend variants; evaluation/data code and common output schema remain byte-identical for identical canonical candidates.

5. Add ablation manifests and documentation.
   - Where: tracked configs under `experiments/zeroshot_cf/configs/matrices/` plus `README.md`.
   - Add `dicoflex_ablation_example.yaml` demonstrating search, diversity, backend, backend hyperparameter, dataset, and seed variants.
   - Add `full_reference.yaml` with the exact four datasets, six retained methods, seed 42, `max_test: 1000`, the recorded DiCoFlex `k=3` configuration, legacy export enabled, and output root `experiments/zeroshot_cf/results/local/architecture_full_reference`. Document the exact adapter steps for future TabPFN/TabFM and why unsupported capabilities must be explicit.

---

## Verification

- [ ] GATE backend conformance and capability-rejection tests use fake prepared sessions — missing proposal semantics, hidden `hasattr()` paths, or unsupported joint/confidence requests turn them red.
- [ ] GATE offline fake-backend parity compares canonical candidate arrays from the legacy and new search adapters — a backend-bound search assumption or result drift turns it red.
- [ ] GATE architecture import tests prove `datasets/` and `evaluation/` contain no TabICL, TabPFN, TabFM, concrete-method, or numbered-runner imports.
- [ ] REPORT run the real TabICL adapter parity smoke when checksum-verified checkpoints exist; otherwise record exact missing paths as `NOT MEASURED`.

---

## Commit

`refactor(dicoflex): isolate foundation proposal backends`
