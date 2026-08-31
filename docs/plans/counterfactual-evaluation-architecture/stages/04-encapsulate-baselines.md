# Stage 4: Encapsulate All Reference Methods

**Goal**: Wrap NICE, Wachter, Growing Spheres, DiCE, and FACE behind one two-phase method lifecycle without changing their algorithms.
**Dependencies**: Stage 3
**Reference**: [`architecture.md`](../resources/architecture.md#method-contract)

---

## Steps

1. Add shared method protocols, capabilities, and typed configs.
   - Where: new `methods/base.py` and per-method config dataclasses.
   - `CounterfactualMethod.prepare(MethodContext)` owns dataset-level state; `PreparedMethod.generate(GenerationRequest)` returns only validated `GenerationResult`.
   - Capabilities explicitly declare categorical/actionability, multiple-CF, probability, and optional-dependency requirements.

2. Wrap lightweight baselines first.
   - Where: new `methods/nice.py` and `methods/optimization.py`, reusing `greedy_nice_counterfactual()`, `wachter_counterfactual()`, and `growing_spheres_counterfactual()`.
   - Move neighbor/LOF state and value grids into `prepare()`. Seed per run/request; do not access dataset name, test truth, evaluator, or output paths.

3. Wrap FACE and DiCE with explicit adapters.
   - Where: new `methods/face.py` and `methods/dice.py`.
   - FACE graph construction belongs to `prepare()`. DiCE owns dataframe codec, lazy optional import, explainer, RNG isolation, raw output, atomic-category repair, and pruning.
   - Convert genuine failures to unavailable candidates; retain raw/best-effort rows only as namespaced artifacts.

4. Make Exp11–14 use the wrappers and common evaluator/writer.
   - Where: each numbered runner's `run_dataset()`.
   - Keep existing CLI flags and v1 outputs, but remove method-specific common metric calculations and I/O assembly. Record uniform preparation/generation/evaluation timing while retaining legacy timing fields in compatibility output.

---

## Verification

- [ ] GATE shared method contract tests run each baseline on deterministic synthetic/current fixtures — invalid shapes, masks, action constraints, seed behavior, or metadata namespaces turn them red.
- [ ] GATE legacy Exp11–14 algorithm and one-factual runner tests compare compatibility summary/point/array inputs — unintended candidate or v1 artifact drift turns them red.
- [ ] GATE lazy-import tests execute registry/help paths without DiCE or FACE optional runtime initialization — eager dependency loading turns them red.

---

## Commit

`refactor(methods): encapsulate reference counterfactual baselines`
