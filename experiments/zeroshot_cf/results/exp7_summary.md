# Experiment 7: Budget Sweep with Feature Revisiting

Budget is swept above `|A|` at the Stage-4 recommended context configs: `random_both@512` for MOONS and `knn_both@256` for HELOC. `l0_count_mean` counts distinct changed features; `steps_*` counts commits, so repeated features show up as extra steps without inflating L0.

## MOONS

Config: `random_both@512` · n_test: 100

| budget | validity | failure_rate | l0_count_mean | steps_mean | steps_max | proximity_l2_jaccard | frac_oob | true_actionability | runtime_s |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.82 | 0.18 | 1.329 | 1.329 | 2 | 0.6589 | 0 | 1 | 74.82 |
| 4 | 0.82 | 0.18 | 1.329 | 1.329 | 2 | 0.6589 | 0 | 1 | 86.86 |
| 8 | 0.82 | 0.18 | 1.329 | 1.329 | 2 | 0.6589 | 0 | 1 | 88.07 |
| 16 | 0.82 | 0.18 | 1.329 | 1.329 | 2 | 0.6589 | 0 | 1 | 89.3 |
| 32 | 0.82 | 0.18 | 1.329 | 1.329 | 2 | 0.6589 | 0 | 1 | 88.02 |
| 64 | 0.82 | 0.18 | 1.329 | 1.329 | 2 | 0.6589 | 0 | 1 | 87.39 |

Verdict: Validity does not improve over the baseline budget; best observed value is 0.82 at budget 2. The curve saturates by budget 2 (steps_max=2 < next budget 4).

## HELOC

Config: `knn_both@256` · n_test: 30

| budget | validity | failure_rate | l0_count_mean | steps_mean | steps_max | proximity_l2_jaccard | frac_oob | true_actionability | runtime_s |
|---|---|---|---|---|---|---|---|---|---|
| 17 | 0.8 | 0.2 | 1.833 | 1.833 | 6 | 0.6646 | 0 | 1 | 724.6 |
| 34 | 0.8 | 0.2 | 1.833 | 1.833 | 6 | 0.6646 | 0 | 1 | 0 |
| 51 | 0.8 | 0.2 | 1.833 | 1.833 | 6 | 0.6646 | 0 | 1 | 0 |
| 100 | 0.8 | 0.2 | 1.833 | 1.833 | 6 | 0.6646 | 0 | 1 | 0 |
| 250 | 0.8 | 0.2 | 1.833 | 1.833 | 6 | 0.6646 | 0 | 1 | 0 |
| 1000 | 0.8 | 0.2 | 1.833 | 1.833 | 6 | 0.6646 | 0 | 1 | 0 |

Verdict: Validity does not improve over the baseline budget; best observed value is 0.8 at budget 17. The curve saturates by budget 17 (steps_max=6 < next budget 34).

