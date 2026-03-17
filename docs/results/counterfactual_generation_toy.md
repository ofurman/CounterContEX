# Counterfactual Generation Toy Experiment — Results

**Date**: 2026-03-17
**Config**: 5 features, 128 seq_len, 16 batch_size, emsize=64, 4 layers, 2 heads, 20 epochs, 50 steps/epoch

## Training

| Metric | Value |
|--------|-------|
| Initial MSE loss | 34.69 |
| Final MSE loss | 33.54 |
| Loss reduction | ~3.3% |
| Model parameters | 0.14M |
| Training time | ~55s (20 epochs on CPU) |

**Observation**: Loss barely decreases. The model learns to predict near-zero deltas (a safe default since ~70% of features are unperturbed), but fails to learn the actual counterfactual structure.

## Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Delta MSE | 36.46 | High — predicted deltas are far from ground truth |
| Validity rate | 99.8% | Misleading — heuristic only checks if delta norm > 0.01 |
| Proximity (L2) | 0.050 +/- 0.017 | Very small — model predicts near-zero deltas |
| Sparsity | 96.7% | Very high — model outputs near-zero for almost all features |

## Example Predictions

The model predicts deltas of magnitude ~0.01-0.04 across all features, while true deltas are on the order of 2-26 per feature. The model has essentially learned to output near-zero everywhere.

```
Example: True delta = [0, 0, -2.768, 0, 0]  →  Pred delta = [0.014, 0.008, -0.018, 0.002, -0.004]
Example: True delta = [0, 0, 0, 11.154, 0]  →  Pred delta = [0.030, 0.014, -0.022, 0.011, 0.007]
```

## Diagnosis

The model is **not learning** meaningful counterfactual generation. Root causes:

1. **Insufficient training**: 20 epochs x 50 steps = 1000 gradient updates with a 0.14M parameter model. Standard TabPFN trains for 80-200 epochs x 100 steps.

2. **High variance targets**: The SCM generates different causal structures per batch element, so the deltas have very high variance (MSE ~34). The model defaults to the mean prediction (near-zero) which is the MSE-optimal constant predictor.

3. **Label flip rate is low (~6%)**: Most training targets are near-zero deltas (non-flipped pairs). The signal-to-noise ratio for learning actual counterfactual structure is poor.

4. **No normalization**: Feature scales vary widely across SCMs (values range from -20 to +30), making it hard for the model to learn generalizable delta magnitudes.

## Conclusion

The pipeline works end-to-end (data generation, training, evaluation), but the model does not yet learn meaningful counterfactual generation. This is expected for a minimal toy experiment. Key improvements needed:

1. **Longer training** (100+ epochs, 100+ steps)
2. **Only use label-flipped pairs** for query targets, or increase perturbation magnitude
3. **Normalize features** per-dataset before feeding to the model
4. **Larger model** or **curriculum learning** (start with simple SCMs)
5. **Consider predicting which features to change** (binary mask) separately from **how much to change them** (continuous delta)
