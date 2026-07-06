# Consolidated Greedy-CF Results Table

This is the Stage-9 headline table with `proximity_l2_jaccard` surfaced as a first-class
metric. Lower is better for L0, steps, proximity, LOF, and `frac_oob`; higher is better for
validity and true actionability.

| dataset | config | validity | failure_rate | l0_count_mean | steps_mean | proximity_l2_jaccard | LOF | frac_oob | true_actionability | n_test | source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| moons | one_pass_mask_all_full | 0.995 | — | — | — | 0.6742 | 1.060 | 0.010 | 1.000 | 200 | `exp2_moons_metrics.csv` |
| heloc | one_pass_mask_all_full | 0.538 | — | — | — | 1.6905 | 5.68e9 | 0.653 | 0.999 | 2092 | `exp2_heloc_metrics.csv` |
| moons | greedy_prob_ascent_random_both_512_budget2 | 0.820 | 0.180 | 1.329 | 1.329 | 0.6589 | 1.027 | 0.000 | 1.000 | 100 | `exp7_budget_moons.csv` |
| heloc | greedy_prob_ascent_knn_both_256_budget17 | 0.800 | 0.200 | 1.833 | 1.833 | 0.6646 | 1.848 | 0.000 | 1.000 | 30 | `exp7_budget_heloc.csv` |
| binary_cat | greedy_prob_ascent_native_categorical | 1.000 | 0.000 | 1.000 | 1.000 | 1.0000 | 1.000 | 0.000 | 1.000 | 50 | `exp4_greedy_binary_cat_metrics.csv` |
| heloc | routing_override_force_numeric_smoke | 1.000 | 0.000 | 1.000 | 1.000 | 0.1333 | 1.082 | 0.000 | 1.000 | 1 | `exp9_routing_heloc.csv` |

Notes:

- The one-pass baseline runners did not emit integer `l0_count_mean`, `steps_mean`, or
  greedy `failure_rate`. The HELOC predecessor masks all 17 actionables in one pass, but
  the table leaves integer L0 blank because the committed CSV only records fractional
  `sparsity`.
- The MOONS and HELOC greedy rows use the Stage-4 recommended configs and the Stage-5
  best/saturated budget cells. Raising budget above `|A|` did not improve validity for
  either dataset.
- The routing-override row is a bounded smoke diagnostic (`n_test=1`, `budget=1`), not a
  stable HELOC estimate. The full-size Exp9 rerun remains Backlog #4.
