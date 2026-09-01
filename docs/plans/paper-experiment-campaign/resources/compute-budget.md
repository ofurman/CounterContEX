# Compute budget

## Measured per-factual CounterContEx cost

From the 24-cell reference run, `results/local/full_reference/exp9_dicoflex/*_metrics.csv`,
at n=1000 factuals, k=3, logistic-regression target, TabICL backend, 9-point quantile grid.

| Dataset | n_train | Columns | Generation s | Per factual |
|---|---:|---:|---:|---:|
| give_me_some_credit | 10,696 | 44 | 1,161.3 | 1.16 s |
| bank_marketing | 25,602 | 50 | 1,981.0 | 1.98 s |
| heloc | 6,316 | 23 | 2,605.6 | 2.61 s |
| lending_club | 19,200 | 32 | 27,519.4 | **27.52 s** |

**The anomaly Stage 1 must explain.** Lending Club costs 10.5× HELOC per factual on fewer
columns than Give Me Some Credit, which is the cheapest cell. Dimensionality does not explain
it. Candidate causes, in the order Stage 1 should test them: search depth (more steps to first
crossing), categorical breadth (more one-hot groups expanded per depth), beam-level candidate
counts, and Gower-neighbour selection over a 19,200-row training set. The manifest already
records `attempt_steps` and `diverse_histories` per point, so the first two are answerable
from stored artifacts without a new run.

**Lending Club is 75% of the campaign's cost.** Any reduction there is worth more than every
other optimization combined.

## Campaign estimate

Adult and German Credit have no measured cost. Estimates below assume Adult behaves like Bank
Marketing (~2.5 s/factual; comparable row count and one-hot breadth) and German Credit is
cheap (~1.0 s/factual; 1,000 rows). **These are estimates and are labelled as such** — Stage 1
replaces them with measurements at n=25 before Stage 6 sizes the matrices.

Per-seed, per-classifier sweep over all six datasets at n=250:

```
250 × (1.16 + 1.98 + 2.61 + 27.52 + 2.5 + 1.0) s = 250 × 36.77 s = 9,193 s = 2.55 h
```

| Stage | Experiment | Cells | Est. GPU-h |
|---|---|---|---:|
| 7 | E1 main comparison | 6 datasets × 3 classifiers × 5 seeds | ~38 |
| 8 | E2 diverse sets (k=3) | 6 × 5 seeds, CounterContEx + DiCE | ~13 |
| 8 | E3 backend ablation | 6 × 3 seeds × 2 backends | ~8 |
| 9 | E4 confidence / τ Pareto | 4 datasets × grid | ~15 |
| 10 | E5 search + diversity ablations | n=100 | ~6 |
| 10 | E6 context ablation | 8 configurations, n=100 | ~8 |
| 10 | E7 cost-quality Pareto | 4 configurations × 4 datasets | ~10 |
| 11 | E8 robustness | scoring pass over Stage 7 arrays | ~0 |
| 11 | E9 foundation-model swap | 2 datasets × 2 backends | ~4 |
| 12 | E10 frozen headline, n=1000 | 6 datasets, best configuration | ~10 |

**Total ≈ 120 GPU-hours**, or roughly **70** if Stage 1 finds and fixes the Lending Club
anomaly. Baseline methods add under 3 hours in aggregate; only DiCE (124 s/1000) and FACE
(103 s/1000) are non-trivial.

Record actual hours from manifest `phase_timings` as the campaign proceeds. The index carries
a REPORT row for the total.
