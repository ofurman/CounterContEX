# Stage 3: Isolate the TabICL Generator and Diversity APIs

**Goal**: Expose reusable single- and multiple-counterfactual APIs that contain all requested TabICL behavior without importing legacy experiments, dataset adapters, or reporting code.
**Dependencies**: Stage 2

---

## Steps

1. Move reusable orchestration from `generate_tabicl_counterfactuals()` in `exp8_tabicl_cf.py` into
   a stable generator module with typed configuration and result/diagnostic objects.
   - Accept arrays, discriminator protocol, targets, feature domains, immutable indices, and
     one-hot groups. Do not load named datasets, write results, or train/cache classifiers.
   - Preserve conditional sampling, batched numerical proposals, atomic categorical swaps, global
     greedy choice, quantile candidates, confidence conditioning, revisits, and failure fallback.
2. Preserve plausibility refinement as the single-CF `data_plausible` mode.
   - Keep `TabICLJointScorer`, one-shot valid-candidate reranking, shortlist/step budgets, and
     `min_joint_log_gain` rejection explicit in configuration and diagnostics.
3. Preserve `diverse_search.py` as the multiple-CF sparse mode.
   - Keep bounded beam expansion, archive quality rules, invalid-candidate rejection, grouped
     signatures, and exact fixed-size DPP selection.
   - Expose it through the stable generator boundary without forcing it into refinement mode.
4. Reduce Exp8 to a compatibility CLI/dataset adapter that calls the stable APIs. Replace its use
   of Exp4 `_DATASET_PARAMS`, `TAU`, and `evaluate_and_report()` with retained config/reporting.
5. Add end-to-end fake-backend tests through the public API for single-CF and diverse modes,
   including confidence-to-quantile-to-greedy flow, revisit controls, validity-step cap,
   refinement acceptance/rejection, immutable preservation, and no invalid diversity padding.

---

## Verification

- [ ] GATE focused generator, grouped-search, plausibility, distance, and diversity tests pass — public production APIs and fake backend inputs are measured; orchestration or mode regressions turn it red.
- [ ] GATE `rg -n 'experiments\.zeroshot_cf\.(exp[1-7]|greedy|sampler|checkpoints|data|metrics_harness)' experiments/zeroshot_cf/generator.py experiments/zeroshot_cf/tabicl_sampler.py experiments/zeroshot_cf/grouped_categorical.py experiments/zeroshot_cf/diverse_search.py experiments/zeroshot_cf/mixed_distance.py experiments/zeroshot_cf/tabicl_joint_plausibility.py experiments/zeroshot_cf/action_space.py experiments/zeroshot_cf/candidate_domain.py` returns no matches — the listed source modules are the input; legacy, dataset, or reporting coupling turns it red.
- [ ] GATE a fresh offline process imports the generator API and runs one fake single-CF and one fake diverse-CF case without loading datasets, checkpoints, CEL, DiCE, or local TabPFN — import or runtime side effects turn it red.
- [ ] GATE Exp8 `--help` and a fake adapter smoke remain runnable through the compatibility entry point — stale CLI wiring turns it red.

---

## Commit

`refactor(tabicl-cf): isolate generator and diverse search`
