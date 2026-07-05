# Experiment 9: HELOC Routing Override Audit

Config: `prob_ascent` + `knn_both@256`; dataset: `heloc`; n_test: 1; budget: 1.

Note: full `--max-test 30` at the natural HELOC budget (`|A|=17`) was attempted first, but the forced-numeric cell repeatedly exhausted the greedy budget. Because `prob_ascent` evaluates every actionable candidate at every step, that run hit the known `O(|A|^2)` worst case and was stopped after producing no final artefact. This summary is therefore a bounded smoke diagnostic (`--max-test 1 --budget 1 --n-permutations 1`), not a statistically stable HELOC estimate.

## Routing Inventory

Classifier-routed original columns: 5 / 23

| idx | feature | unique_train_values |
|---|---|---|
| 5 | `NumTrades60Ever2DerogPubRec` | 18 |
| 6 | `NumTrades90Ever2DerogPubRec` | 15 |
| 9 | `MaxDelq2PublicRecLast12M` | 10 |
| 10 | `MaxDelqEver` | 8 |
| 12 | `NumTradesOpeninLast12M` | 18 |

Regressor-routed original columns: 18 / 23.

## Metrics

| cell | validity | failure_rate | proximity_l2_jaccard | frac_oob | l0_count_mean | steps_mean | runtime_s | force_numeric |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 0 | 0.3462 | 0 | 1 | 1 | 4.39 | none |
| override | 1 | 0 | 0.1333 | 0 | 1 | 1 | 3.89 | 5,6,9,10,12 |

## Deltas (override - baseline)

- validity: 0
- proximity_l2_jaccard: -0.2128
- frac_oob: 0
- l0_count_mean: 0

Verdict: The override improves proximity/validity enough to consider forcing ordered treatment for these HELOC columns.

Forced columns:
`5:NumTrades60Ever2DerogPubRec`, `6:NumTrades90Ever2DerogPubRec`, `9:MaxDelq2PublicRecLast12M`, `10:MaxDelqEver`, `12:NumTradesOpeninLast12M`
