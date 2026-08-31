# Stage 4: Normalize Exp9 and Baseline Protocol Boundaries

**Goal**: Make Exp9 and all four baseline entry points consume one explicit benchmark protocol instead of importing private helpers and constants from numbered runners.
**Dependencies**: Stage 3

---

## Steps

1. Create a benchmark-protocol module owning:
   - the exact four dataset names and ordering;
   - validation fraction, factual cap, seeds, target threshold, and deterministic stratified
     factual selection;
   - discriminator configuration and classifier-prediction target construction;
   - shared result schemas, output naming, and aggregation order.
2. Update Exp9 and Exp11–14 to use the protocol and Stage 2 baseline-common helpers.
   - Baselines must not import Exp8/Exp9 for private selection/constants or each other for helpers.
   - Exp9 imports the stable generator API; aggregate mode must not initialize TabICL or checkpoints.
3. Consolidate retained metric/reporting behavior.
   - Keep `compute_dicoflex_common_metrics()` and grouped mixed-data costs.
   - Remove duplicated writer/schema logic while preserving each CLI's metrics, points, arrays, and
     multi-dataset aggregate outputs.
   - Explicitly distinguish Exp9's primary-CF common metrics from complete-set diversity metrics.
4. Add runner-level tests with synthetic data/fake generators for the common output schema,
   classifier-derived targets, threshold semantics, valid-only metrics, complete coverage policy,
   cache path handling, and deterministic aggregation.
5. Retain all CLI flags that affect published experiments. Correct stale four/five-dataset docs,
   but leave full README/Athena rewriting to Stage 6.

---

## Verification

- [ ] GATE Exp9 and Exp11–14 unit/runner tests pass against the shared protocol and synthetic inputs — inconsistent splits, targets, actionability, schemas, or aggregation turn it red.
- [ ] GATE `rg -n 'from experiments\.zeroshot_cf\.exp(8|9|11|12|13|14)' experiments/zeroshot_cf/exp{9,11,12,13,14}_*.py` returns no cross-runner private imports — retained runner sources are the input; a numbered-runner dependency turns it red.
- [ ] GATE Exp9 aggregate and baseline `--help` run with TabICL/checkpoint loading disabled or unavailable — CLI modules and fixture metrics files are the inputs; eager generator loading turns it red.
- [ ] GATE the Stage 1 dataset-contract test and the complete retained benchmark/baseline test manifest pass — protocol drift or lost baseline behavior turns it red.

---

## Commit

`refactor(tabicl-cf): unify benchmark and baseline protocol`
