# Experiment 4: From-Scratch Counterfactuals via Task-Guided Beam Search

Settings: beam_width=8, n_candidates=6, lambda_actionable=1.0, lambda_immutable=100.0, max_context=256, context_type=all_classes

## Metrics

| Dataset | Validity | LOF | Proximity L2 | OOB frac | Immut drift (mean) | True-action |
|---------|---------|-----|-------------|---------|-------------------|------------|
| moons | 1.000 | 0.977 | 0.4697 | 0.000 | 0.0000 | 1.000 |
| heloc | 0.333 | 1.028 | 0.8443 | 0.000 | 0.0395 | 0.000 |

## Notes

- **From scratch**: every feature is generated autoregressively conditioned only on Y=target; the factual enters solely via the per-feature proximity penalty `λ·|f − factual|`.
- Immutables are soft-frozen (large `lambda_immutable`); they are still generated, so `true_actionability` < 1.0 is expected and `immutable_drift` (mean |Δ| over immutable columns) quantifies how far they wandered.
- `validity`: fraction whose discriminator class == target (higher = better).
- `lof_scores_cf`: mean negative-LOF plausibility on unclipped CFs (lower = better).
- `proximity_l2_jaccard`: mean L2 to factual on *valid* CFs (lower = closer).
- `frac_oob`: fraction of CF rows with a feature outside [0,1] before clipping. Hard [0,1] candidate rejection during search should keep this low.

Comparison vs. Exp 2 (imputation baseline) is recorded in `results/REPORT.md`.
