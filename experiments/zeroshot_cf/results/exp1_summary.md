# Experiment 1: Single-Feature Reconstruction — Summary

## Results

| Dataset | Features | Beats marginal | Avg MSE marginal | Avg MSE TabPFN | Avg calib 10-90% | Gate verdict |
|---------|----------|----------------|-----------------|---------------|-----------------|--------------|
| moons | 2 | 50% | 0.0550 | 0.0455 | 0.69 | **WEAK** |
| heloc | 23 | 65% | 0.0254 | 0.0634 | 0.62 | **PASS** |

## Gate Verdict Definitions

- **PASS**: TabPFN beats the marginal-mean baseline on ≥50% of features (HELOC) or all features (MOONS). Proceed to Stage 5 with confidence.
- **WEAK**: Beats marginal on ≥30% (HELOC) or ≥50% (MOONS). Proceed to Stage 5 but flag low expectations; refinement may be needed.
- **FAIL**: Does not beat marginal baseline. Record that Experiment 2 is unlikely to work out-of-the-box; refinement focus shifts to context/temperature.

## Notes

- Context: same-class train rows (capped at 256), near-MAP temperature t=1e-9.
- Calibration: fraction of true values inside the [10%, 90%] interval of N_SAMPLES posterior samples at t=1.0 (per-sample seeds varied to ensure independent draws).
- Ridge baseline: RidgeCV trained to predict feature j from the other features.
