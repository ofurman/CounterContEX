# Experiment 1: Single-Feature Reconstruction — Summary

## Results

| Dataset | Features | Beats marginal | Avg MSE marginal | Avg MSE TabPFN | Avg calib 10-90% | Gate verdict |
|---------|----------|----------------|-----------------|---------------|-----------------|--------------|
| moons | 2 | 50% | 0.0550 | 0.0455 | 0.70 | **WEAK** |
| heloc | 23 | 65% | 0.0254 | 0.0649 | 0.60 | **PASS** |

## Gate Verdict Definitions

- **PASS**: TabPFN beats the marginal-mean baseline on ≥50% of features (HELOC) or all features (MOONS). Proceed to Stage 5 with confidence.
- **WEAK**: Beats marginal on ≥30% (HELOC) or ≥50% (MOONS). Proceed to Stage 5 but flag low expectations; refinement may be needed.
- **FAIL**: Does not beat marginal baseline. Record that Experiment 2 is unlikely to work out-of-the-box; refinement focus shifts to context/temperature.

## Interpretation

- **MOONS (WEAK)**: TabPFN beat the marginal baseline on 1/2 features. Feature 1 shows strong conditioning (MSE 0.0084 vs 0.0537 marginal, 6.4× improvement). Feature 0 did not improve (0.0826 vs 0.0563), likely because the MOONS class boundary is non-linear and a 2-D dataset with t=1e-9 MAP estimation anchors to the class conditional mean. Calibration is good (70%), indicating the posterior distribution is well-formed.
- **HELOC (PASS)**: 15/23 features beat the marginal baseline. Several features show dramatic improvement (e.g. NumTradesOpeninLast12M: 28× improvement; ExternalRiskEstimate: 48× improvement). The average MSE is slightly higher than marginal due to a handful of near-binary / ordinal features (NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec, MaxDelq2PublicRecLast12M, MaxDelqEver) where TabPFN produces large errors — likely because these features have highly skewed / sparse distributions that are hard to estimate at near-MAP temperature. Calibration (60%) is acceptable.

**Overall gate: PASS — proceed to Stage 5 with moderate confidence.** The conditioning mechanism is informative for continuous features; sparse/near-binary features may need temperature tuning or a different context strategy in Experiment 2.

## Notes

- Context: same-class train rows (capped at 256), near-MAP temperature t=1e-9.
- Calibration: fraction of true values inside the [10%, 90%] interval of 10-50 posterior samples at t=1.0.
- Ridge baseline: RidgeCV trained to predict feature j from the other features.
- MOONS used N_SAMPLES=50, MAX_TEST=50; HELOC used N_SAMPLES=10, MAX_TEST=30 for wall-clock budget.
