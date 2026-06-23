# Ablation Grids

## Stage 2 — Selector ablation (Exp5)

One factor, two levels, both datasets ⇒ 4 cells.

| Selector | Strategy | Required context scope | Datasets |
|----------|----------|------------------------|----------|
| `prob_ascent` | Strategy 1 — steepest-ascent on `disc.predict_proba[y_target]` | `target_only` | MOONS, HELOC |
| `class_divergence` | Strategy 2 — argmax divergence between `p(x_j\|·,Y=target)` and `p(x_j\|·,Y=current)` | `all_classes` (required) | MOONS, HELOC |

Held constant within a dataset: `max_context=256`, `t=1e-9`, `n_permutations`, `--max-test`.
Caveat: the two selectors necessarily differ in context scope (Strategy 2 needs both-classes).
The apples-to-apples contrast lives in Stage 4 (context scope is a controlled axis there).

## Stage 4 — Context ablation (Exp6)

Two factors: **size** (4) × **strategy** (4) ⇒ 16 cells per dataset, at the Stage-2 winning selector.

### Size (`max_context`)
`256`, `512`, `1024`, `2048` — capped at the available pool, `effective_size` logged.
MOONS train ≈ 800 (~400/class) → saturates above ~400 (target) / ~800 (both). HELOC train ≈ 8k → all sizes feasible.

### Strategy (class scope × selection method)

| Strategy | `target_class` | `selection` | Pool | Selection within pool |
|----------|----------------|-------------|------|------------------------|
| `random_target` | target | `random` | target class only | uniform subsample (**predecessor baseline**) |
| `random_both`   | `None` | `random` | both classes | uniform subsample |
| `knn_target`    | target | `knn` | target class only | `max_context` nearest neighbours to the factual point |
| `knn_both`      | `None` | `knn` | both classes | `max_context` nearest neighbours to the factual point |

### Full cell list (prob_ascent → 16 cells)

```
sizes      = [256, 512, 1024, 2048]
strategies = [random_target, random_both, knn_target, knn_both]
cells      = sizes × strategies          # 16 per dataset
```

### Selector compatibility (Decision #6)

If the Stage-2 winner is `class_divergence` (needs both-classes pool), the 8 `*_target`
cells are **skipped (logged)**, leaving 8 cells: `[256,512,1024,2048] × [random_both, knn_both]`.
If the winner is `prob_ascent`, run the full 16.

Held constant within a dataset: selector, `t=1e-9`, `n_permutations`, `--max-test`.
kNN context is selected **per query point** (Decision #5) → fit context per test point; the
dominant cost on HELOC, bounded by a small `--max-test` held identical across all cells.

## Stage 5 — Budget sweep (Exp7)

One factor (`budget`), swept per dataset at the **Stage-4 recommended config** with the
revisit-enabled loop (Decision #15). `prob_ascent` selector throughout.

| Dataset | Config (fixed) | `\|A\|` | Budget grid |
|---------|----------------|------|-------------|
| MOONS | `random_both@512` | 2 | `2, 4, 8, 16, 32, 64` |
| HELOC | `knn_both@256` | 17 | `17, 34, 51, 100, 250, 1000` |

Held constant within a dataset: selector, context strategy/size, `t=1e-9`, `n_permutations`,
`--stall-eps`, `--max-test`. Report `validity, failure_rate, l0_count_mean (distinct), steps_mean,
steps_max, proximity_l2_jaccard, lof_scores_cf, frac_oob, true_actionability, runtime_s` per budget.
Question answered: does validity climb toward 1.0 as budget exceeds `\|A\|`, and at what budget
does it saturate (or does it plateau below 1.0 = TabPFN-vs-classifier ceiling)?

## Stage 8 — Routing override (Exp9)

HELOC only (MOONS is all-continuous → null control). Two cells at `prob_ascent` + `knn_both@256`:

| Cell | `--force-numeric-cols` | Effect |
|------|------------------------|--------|
| baseline | `none` | low-cardinality int cols auto-route to classifier head (current) |
| override | `<int-col idx list>` (or `all`) | those cols forced to regressor (ordered bar-dist) head |

Report Δ `validity`, Δ `proximity_l2_jaccard`, Δ `frac_oob`, Δ `l0_count` between cells.

## Metric columns (all ablation CSVs)

`validity, l0_count_mean/median/max (integer features changed per CF), steps_mean/median/max, failure_rate,`
`lof_scores_cf, sparsity (existing mean-fraction, kept for comparability), true_actionability (==1.0 by construction),`
`proximity_l2_jaccard, frac_oob, runtime_s`
plus run identifiers (`selector, size, effective_size, strategy, class_scope, selection, n_test`).

> **`frac_oob` is NOT returned by `compute_metrics`** — compute it inline on the **unclipped** `X_cf`,
> exactly as `exp2_counterfactuals.py:264–267`: `frac_oob = (((X_cf < 0.0) | (X_cf > 1.0)).any(axis=1)).mean()`.
> Keep the existing fractional `sparsity` *and* add the integer `l0_count_*` keys — they are different metrics.
