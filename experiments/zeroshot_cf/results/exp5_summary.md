# Experiment 5: Selector Ablation — prob_ascent (Strategy 1) vs class_divergence (Strategy 2)

One factor (the selector), two levels, across MOONS + HELOC at the Stage-1 baseline context.
Held identical across the two selectors **within a dataset**: `max_context=256`, `temperature=1e-9` (MAP commit), `n_permutations`, `n_test` (= `--max-test`).

> **Context-scope caveat (Decision #6).** Strategy 2 (`class_divergence`) *requires* a both-classes context pool (`all_classes`) so the Y column is non-constant; Strategy 1 (`prob_ascent`) uses a target-only context (`target_only`). The two cells within a dataset therefore *necessarily* differ in context scope — this is **each selector at its natural/required context**, not both at an identical context. The apples-to-apples contrast is deferred to Stage 4, where the context strategy is a controlled axis.

## MOONS

| selector | context_scope | n_test | validity | l0_count_mean | steps_mean | steps_median | steps_max | failure_rate | lof_scores_cf | true_actionability | proximity_l2_jaccard | frac_oob | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prob_ascent | target_only | 100 | 0.7 | 1.271 | 1.271 | 1 | 2 | 0.3 | 1.009 | 1 | 0.5975 | 0 | 64.5 |
| class_divergence | all_classes | 100 | 0.64 | 1.312 | 1.312 | 1 | 2 | 0.36 | 1.011 | 1 | 0.619 | 0 | 107 |

**Per-metric winner:**

- validity (higher better): **prob_ascent**
- L0 count (lower better): **prob_ascent**
- steps-to-flip (lower better): **prob_ascent**
- plausibility frac_oob (lower better): **tie**
- plausibility LOF (lower better): **prob_ascent**

## HELOC

| selector | context_scope | n_test | validity | l0_count_mean | steps_mean | steps_median | steps_max | failure_rate | lof_scores_cf | true_actionability | proximity_l2_jaccard | frac_oob | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prob_ascent | target_only | 50 | 0.9 | 1.667 | 1.667 | 2 | 4 | 0.1 | 9.491e+06 | 1 | 0.6349 | 0.04 | 1.056e+03 |
| class_divergence | all_classes | 50 | 0.52 | 14.27 | 14.27 | 17 | 17 | 0.48 | 3.073e+06 | 1 | 1.075 | 0.08 | 2.963e+03 |

**Per-metric winner:**

- validity (higher better): **prob_ascent**
- L0 count (lower better): **prob_ascent**
- steps-to-flip (lower better): **prob_ascent**
- plausibility frac_oob (lower better): **prob_ascent**
- plausibility LOF (lower better): **class_divergence**

## Chosen downstream selector (used by Stage 4)

**`prob_ascent`**

Default tie-break (Decision #6): prob_ascent is compatible with all four Stage-4 context strategies and directly optimizes the flip; class_divergence did not clearly win plausibility on HELOC without a validity cost.

