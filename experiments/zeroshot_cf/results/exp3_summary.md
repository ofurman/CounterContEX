# Experiment 3: Feature-Ordering (DAG) Ablation

## Setup

- temperature=1.0, max_context=256, context_type=all_classes (see note below)
- MOONS: n_permutations=5, max_test=100
- HELOC: n_permutations=1, max_test=20 (reduced from baseline 5/50 for runtime feasibility; all 4 cells identical)
- Reduced actionable set: top-6 features by |LR coef| (HELOC only)
- **Note on context_type**: Exp3 uses `all_classes` for all cells. Reason: the DAG places Y as an explicit conditioning parent; with `target_only`, Y is constant in context (single class) and TabPFN raises a constant-feature validation error. `all_classes` keeps Y informative and makes the random/dag comparison within Exp3 fair. Stage-8 results (target_only) remain the reference for the recommended production configuration.

## Mechanism framing

In the **random-permutation path** (`dag=None`), every masked cell already conditions on all observed columns (Y + immutables). Putting Y/immutables "first" is therefore a **no-op** — only the relative ordering of the masked actionable columns matters, and that effect is averaged away over multiple random permutations.

The **DAG path** imposes `p(A₁|Y,immut) · p(A₂|Y,immut,A₁) · …` — a strict left-to-right chain where each actionable also conditions on the already-filled siblings. This differs from the random path in two ways: (1) the parent set for each actionable is a **subset** (not the full conditioning set), and (2) the ordering is **deterministic** rather than averaged.

**Expected behaviour:**
- MOONS: DAG ≈ random (only 2 actionable features; little ordering freedom).
- HELOC full: DAG neutral-to-worse (smaller parent set than random path).
- HELOC reduced: the cell most likely to improve OOB (denser conditioning).

## HELOC

| ordering | actionable_set | n_masked | validity | lof_scores_cf | sparsity | true_actionability | proximity_l2 | frac_oob | runtime_s |
|----------|---------------|---------|---------|--------------|---------|-------------------|-------------|---------|-----------|
| random | full | 17 | 0.400 | 1031816513.506 | 0.704 | 1.000 | 1.6228 | 0.500 | 145.8 |
| dag | full | 17 | 0.550 | 2467634051.735 | 0.687 | 1.000 | 1.4549 | 0.650 | 16.2 |
| random | reduced | 6 | 0.200 | 99452005.475 | 0.248 | 1.000 | 0.3701 | 0.100 | 20.2 |
| dag | reduced | 6 | 0.500 | 11863431.370 | 0.248 | 1.000 | 0.5354 | 0.100 | 4.2 |

### Verdict

- random/full   : validity=0.400, frac_oob=0.500
- dag/full      : validity=0.550, frac_oob=0.650
- random/reduced: validity=0.200, frac_oob=0.100
- dag/reduced   : validity=0.500, frac_oob=0.100

DAG vs random at full actionable set: validity improved (0.400 → 0.550). DAG gives each actionable a *subset* of the full conditioning set, so neutral-to-worse is expected here.
Reduced actionable set (random): validity 0.200 vs 0.400 (full); frac_oob 0.100 vs 0.500 (full). OOB fraction reduced — denser conditioning helps as expected.
Best HELOC cell by validity: ordering=dag, actionable_set=full (validity=0.550, frac_oob=0.650).

## Summary

- `true_actionability` must be 1.0 for every cell — immutable and frozen columns are preserved by construction.
- `frac_oob` measures extrapolation artefacts; only the `reduced` HELOC cells are expected to show meaningful improvement (denser conditioning).
- The DAG result is an honest test of structured vs. random-permutation ordering; no further ablation dimensions are explored here.
