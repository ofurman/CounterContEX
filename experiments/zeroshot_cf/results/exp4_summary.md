# Experiment 4: Counterfactuals via Task-Guided Beam Search

Two regimes, identical beam settings — they differ **only** in whether the immutable features are masked:

- **Set 1 (frozen immutables)** — immutables are *observed* (held at the factual value); the beam generates only the actionable features. Directly comparable to the Exp 2/3 imputation baseline; `true_actionability = 1.0`.
- **Set 2 (from scratch)** — *no* feature is masked; every feature is generated, conditioned only on `Y=target`. The factual enters only via the proximity penalty.

Settings: beam_width=8, n_candidates=6, lambda_actionable=1.0, max_context=256, context_type=all_classes. (For MOONS, which has no immutables, Set 1 ≡ Set 2.)

## Metrics

| Dataset | Set | Validity | LOF | Proximity L2 | OOB frac | Immut drift | True-action |
|---------|-----|---------|-----|-------------|---------|------------|------------|
| moons | frozen | 1.000 | 0.977 | 0.4697 | 0.000 | 0.0000 | 1.000 |
| heloc | frozen | 0.133 | 7918099.357 | 0.4546 | 0.000 | 0.0000 | 1.000 |
| moons | fromscratch | 1.000 | 0.977 | 0.4697 | 0.000 | 0.0000 | 1.000 |
| heloc | fromscratch | 1.000 | 1.006 | 0.8303 | 0.000 | 0.1153 | 0.000 |

## Notes

- `validity`: fraction whose discriminator class == target (higher = better).
- `lof_scores_cf`: mean negative-LOF plausibility on unclipped CFs (lower = better).
- `proximity_l2_jaccard`: mean L2 to factual on *valid* CFs (lower = closer).
- `frac_oob`: fraction of CF rows with a feature outside [0,1] before clipping; the hard [0,1] candidate rejection keeps this at 0.
- **Set 2** generates immutables too, so `true_actionability` < 1.0 and `immutable_drift` reports how far they wandered.

Full comparison vs. Exp 2 (imputation baseline) is in `results/REPORT.md §8`.
