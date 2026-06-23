# Ablation grids

Keep `--max-test` and `--seed` **identical** across all cells compared within a table.
HELOC counts may be bounded (`iterative-greedy-cf` Decision #13) — when bounded, frac_oob /
LOF / joint-NLL trends are the robust signal over noisy validity.

## Stage 1 — score estimator selection

| method | continuous cols | classifier-routed cols | gate |
|--------|-----------------|------------------------|------|
| `mean_shift` (primary) | `(μ_j − x_j)` from bar mean | n/a (Gibbs) | cosine ≥ 0.9 vs KDE-score (MOONS) |
| `findiff` (fallback) | central diff on bar log-prob | n/a (Gibbs) | cosine ≥ 0.9 |
| `smoothed` (last resort) | derivative of smoothed density | n/a (Gibbs) | only if above fail |

Pick the highest-cosine method as the flow default; record in index Decisions.

## Stage 3 — headline (flow vs greedy)

| dataset | mode | budget B | context | n | gate |
|---------|------|----------|---------|---|------|
| MOONS | flow | 2 | random_both@512 | 100 | **validity > 0.82** (plateau gate) |
| MOONS | greedy | — | random_both@512 | 100 | paired baseline (≈0.82) |
| HELOC | flow | 2 | knn_both@256 | bounded | **validity ≥ 0.85, frac_oob ≤ 0.05** (hold gate) |
| HELOC | greedy | — | knn_both@256 | bounded | paired baseline (≈0.90) |

Report per cell: validity, L0_count, failure_rate, frac_oob, LOF, **joint_nll**, true_actionability.

## Stage 4 — path-adaptive context (HELOC, flow, knn_both@256)

| context_refit | meaning |
|---------------|---------|
| 0 | static (anchor at x0) — regression baseline = Stage 3 |
| 5 | periodic re-anchor every 5 steps |
| 1 | re-anchor every step (max cost) |

Report: frac_oob, LOF, joint_nll, validity, context_refits, wall-clock.

## Stage 5 — drift-mix ablation (both datasets, bounded n)

| (α, β) | corner | predecessor analogue |
|--------|--------|----------------------|
| (1, 0) | generative-only | ≈ `class_divergence` |
| (0, 1) | discriminative-only | ≈ `prob_ascent` |
| (1, 1) | dual drift | (the unification claim) |
| (0.5, 2) | validity-weighted dual | — |

Gate: dual drift ≥ either-alone on validity at equal L0. Plus sampling run (`n_samples>1`,
`noise>0`, `beta_schedule∈{linear,geometric}`): report diversity (mean pairwise L2) + coverage.
