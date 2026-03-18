# Plan: Scale Up Model & Implement Alternative Loss Functions

## Overview

The current model (emsize=128, ~1.2M params, batch_size=4) is ~15x smaller than original TabPFN and uses MSE loss on specific random deltas — which creates multi-modal targets for stochastic perturbation. This plan:

1. Scales the model to medium size (~5M params, emsize=256, batch_size=16)
2. Implements two alternative loss functions:
   - **Distributional (NLL)**: output mean + variance per feature, train with Gaussian NLL
   - **Validity-focused**: optimize for label-flip outcome via differentiable SCM forward pass + proximity/sparsity penalties
3. Runs comparative experiments (Exp 3 and 4) with all three losses to find what works

---

## Stage 1: Scale Up Model Configuration

**Goal**: Add medium-scale config and verify training works at this size.

**Files**: `tabpfn/experiments/configs.py`, `tabpfn/experiments/run_experiment.py`

### Changes

1. **In `configs.py`**: Add `MEDIUM_MODEL` base config and scaled versions of Exp 3 and 4:
   ```python
   MEDIUM_MODEL = dict(
       emsize=256,
       nlayers=6,
       nhead=4,
       nhid=512,
       dropout=0.1,
       weight_decay=1e-4,
   )

   # Exp 3 scaled: SCM family with medium model
   EXP3S_CONFIG = dict(
       **MEDIUM_MODEL,
       num_features=5,
       seq_len=256,
       batch_size=16,
       epochs=200,
       steps_per_epoch=200,
       lr=0.0001,
       warmup_epochs=20,
   )
   EXP3S_SCM = dict(EXP3_SCM)  # same SCM config
   EXP3S_CRITERIA = dict(scm_validity=0.70, delta_mse=0.5)

   # Exp 4 scaled: diverse SCMs with medium model
   EXP4S_CONFIG = dict(
       **MEDIUM_MODEL,
       num_features=5,
       seq_len=256,
       batch_size=16,
       epochs=200,
       steps_per_epoch=200,
       lr=0.0001,
       warmup_epochs=20,
   )
   EXP4S_SCM = dict(EXP4_SCM)  # same SCM config
   EXP4S_CRITERIA = dict(scm_validity=0.40, context_ablation_gap=0.10)
   ```

2. **Register** `exp3s` and `exp4s` in `EXPERIMENT_REGISTRY`.

3. **Run a quick sanity check**: `python -m tabpfn.experiments.run_experiment --experiment exp3s --output-dir docs/results/exp3s_sanity` with `epochs=5` to verify the model trains without OOM or NaN.

### Verification
- [ ] Model has ~5M parameters
- [ ] Training runs for 5 epochs without errors
- [ ] Loss decreases in the first few epochs
- [ ] All existing tests pass

### Commit
`feat(experiments): add medium-scale model configs (5M params, batch_size=16)`

---

## Stage 2: Distributional Loss (Gaussian NLL)

**Goal**: Instead of predicting a point delta, predict a Gaussian distribution per feature (mean + log-variance). Train with NLL. At inference, use the mean as the predicted delta.

**Files**: `tabpfn/train_counterfactual.py`, `tabpfn/experiments/configs.py`, `tabpfn/experiments/run_experiment.py`, `tabpfn/eval_counterfactual.py`

### Changes

1. **In `train_counterfactual.py`**: Add `_distributional_loss()`:
   ```python
   def _distributional_loss(output, query_targets, num_features):
       """Gaussian NLL loss: model outputs mean + log_var per feature.

       output shape: (num_query, batch, num_features * 2)
         first num_features = predicted mean (delta)
         last num_features = predicted log_variance

       query_targets shape: (num_query, batch, num_features) — true deltas
       """
       mu = output[..., :num_features]
       log_var = output[..., num_features:]

       # Clamp log_var for numerical stability
       log_var = log_var.clamp(-10, 10)

       # Gaussian NLL: 0.5 * (log_var + (target - mu)^2 / exp(log_var))
       nll = 0.5 * (log_var + (query_targets - mu) ** 2 / log_var.exp())
       return nll.mean()
   ```

2. **In `train_counterfactual.py`**: Add `loss_type` parameter to `train_counterfactual()` and `_train_one_epoch()`. Options: `"mse"` (current), `"distributional"`, `"validity"` (Stage 3). When `loss_type="distributional"`, set `n_out = num_features * 2` and use `_distributional_loss`.

3. **In `run_experiment.py`**: Pass `loss_type` from config to training. Add to `exp_config`:
   ```python
   loss_type = exp_config.get("loss_type", "mse")
   ```

4. **In `eval_counterfactual.py` `run_inference()`**: When `loss_type="distributional"`, extract only the mean (first `num_features` channels) as `pred_deltas`. Optionally also extract variance for analysis.

5. **In `configs.py`**: Add distributional variants:
   ```python
   EXP3S_DIST_CONFIG = dict(**EXP3S_CONFIG, loss_type="distributional")
   EXP4S_DIST_CONFIG = dict(**EXP4S_CONFIG, loss_type="distributional")
   ```

### Verification
- [ ] `_distributional_loss` produces finite gradients
- [ ] Training with `loss_type="distributional"` runs without errors for 5 epochs
- [ ] Predicted variance is positive (exp(log_var) > 0)
- [ ] Mean predictions are reasonable (not all zeros, not exploding)
- [ ] All existing tests pass (MSE path unchanged)

### Commit
`feat(train): add distributional loss (Gaussian NLL) for multi-modal delta prediction`

---

## Stage 3: Validity-Focused Loss

**Goal**: Instead of matching a specific delta, optimize for the *outcome* — does the predicted counterfactual flip the label? Use the SCM's differentiable forward pass as a learned validity signal, plus proximity and sparsity penalties.

**Files**: `tabpfn/train_counterfactual.py`, `tabpfn/priors/counterfactual.py`, `tabpfn/priors/counterfactual_prior.py`, `tabpfn/experiments/configs.py`

### Design

The key insight: the SCM (`_MLP`) is a differentiable neural network. We can backpropagate through it. During training:
1. Model predicts `delta` from context + query
2. Compute `x_cf = x_query + delta`
3. Feed `x_cf` through the SCM's forward path → get `y_cf`
4. Loss = "did y_cf flip?" + "was delta small?" + "was delta sparse?"

This requires the training loop to have access to the SCM and its internals for each batch — the same plumbing we built for SCM validity evaluation.

### Changes

1. **In `counterfactual_prior.py`**: Add a `return_scm_for_training` flag to `get_batch()` and `get_batch_fixed_scm()`. When enabled, return the SCM objects and internals alongside `(x, y, target_y)` so the training loop can do differentiable forward passes.

2. **In `counterfactual_prior.py` data loaders**: When `loss_type="validity"`, yield `(data, target_y, sep, scm_data)` instead of `(data, target_y, sep)`. The `scm_data` contains the SCM, internals, class_assigner, and query_source_indices for each batch element.

3. **In `counterfactual.py`**: Add `differentiable_forward(x_cf, internals, feature_indices)` method to `_MLP`:
   ```python
   def differentiable_forward(self, x_cf, internals, feature_indices):
       """Feed counterfactual feature values through SCM and return y_cf.

       Unlike forward_with_intervention which replaces nodes in outputs_flat,
       this method takes the predicted CF features, places them at the correct
       positions in the causal graph, and re-propagates downstream layers.

       Args:
           x_cf: (seq_len, 1, num_features) predicted CF features
           internals: dict from forward_with_internals
           feature_indices: tensor mapping feature index to outputs_flat index

       Returns:
           y_cf: (seq_len, 1) predicted target value (pre-classification)
       """
   ```
   This is similar to `forward_with_intervention` but:
   - Takes `x_cf` directly (not as a dict of interventions)
   - Keeps gradients flowing (no `.clone()` detach)
   - Returns raw y (before classification) so we can use BCE loss

4. **In `train_counterfactual.py`**: Add `_validity_loss()`:
   ```python
   def _validity_loss(output, query_x, target_labels, scm_data,
                      num_features, single_eval_pos,
                      proximity_weight=0.1, sparsity_weight=0.01):
       """Validity-focused loss using differentiable SCM forward pass.

       Args:
           output: (num_query, batch, num_features) — predicted deltas
           query_x: (num_query, batch, num_features) — factual query features
           target_labels: (num_query, batch) — desired class labels
           scm_data: list of dicts with SCM, internals, class_assigner
           proximity_weight: weight for L2 proximity penalty
           sparsity_weight: weight for L1 sparsity penalty
       """
       pred_delta = output[..., :num_features]
       x_cf = query_x + pred_delta

       # Validity: feed through SCM, compare class to target
       # (differentiable through the SCM's neural network layers)
       y_cf_raw = scm.differentiable_forward(x_cf, internals, feature_indices)
       validity_loss = F.binary_cross_entropy_with_logits(
           y_cf_raw - threshold, target_labels
       )

       # Proximity: keep CF close to factual
       proximity_loss = pred_delta.norm(dim=-1).mean()

       # Sparsity: change few features (L1)
       sparsity_loss = pred_delta.abs().mean()

       return validity_loss + proximity_weight * proximity_loss + sparsity_weight * sparsity_loss
   ```

5. **In `_train_one_epoch()`**: When `loss_type="validity"`, unpack `scm_data` from the data loader and call `_validity_loss`. The query features `query_x` come from `data[1][single_eval_pos:]`.

6. **In `configs.py`**: Add validity variants:
   ```python
   EXP3S_VAL_CONFIG = dict(**EXP3S_CONFIG, loss_type="validity", mask_supervision=False)
   EXP4S_VAL_CONFIG = dict(**EXP4S_CONFIG, loss_type="validity", mask_supervision=False)
   ```
   Note: validity loss does NOT need mask supervision (it optimizes for outcome, not specific mask).

### Verification
- [ ] `differentiable_forward()` produces output with gradients (`.requires_grad == True`)
- [ ] `_validity_loss` produces finite gradients
- [ ] Training with `loss_type="validity"` runs without errors for 5 epochs
- [ ] Loss decreases (validity component and proximity component both decrease)
- [ ] All existing tests pass (MSE path unchanged)

### Commit
`feat(train): add validity-focused loss with differentiable SCM forward pass`

---

## Stage 4: Run Comparative Experiments

**Goal**: Run Exp 3 and 4 with all three loss types at medium scale. Compare results.

**Files**: `tabpfn/experiments/configs.py`, experiment output directories

### Experiments to run

| ID | Experiment | Loss | Config |
|----|-----------|------|--------|
| exp3s_mse | SCM family (10 SCMs) | MSE | EXP3S_CONFIG |
| exp3s_dist | SCM family (10 SCMs) | Distributional NLL | EXP3S_DIST_CONFIG |
| exp3s_val | SCM family (10 SCMs) | Validity-focused | EXP3S_VAL_CONFIG |
| exp4s_mse | Diverse SCMs | MSE | EXP4S_CONFIG |
| exp4s_dist | Diverse SCMs | Distributional NLL | EXP4S_DIST_CONFIG |
| exp4s_val | Diverse SCMs | Validity-focused | EXP4S_VAL_CONFIG |

### Run order
1. Run `exp3s_mse`, `exp3s_dist`, `exp3s_val` (can be sequential or parallel if resources allow)
2. Run `exp4s_mse`, `exp4s_dist`, `exp4s_val`

### Expected outcomes
- **MSE**: Similar to current Exp 3/4 but better due to larger model + batch size
- **Distributional**: Better calibrated uncertainty, possibly similar or better validity
- **Validity**: Highest SCM validity (directly optimized), possibly at cost of delta MSE (deltas may differ from ground truth but still be valid)

### Verification
- [ ] All 6 experiments complete without errors
- [ ] Results saved to `docs/results/exp{3,4}s_{mse,dist,val}/`
- [ ] Each experiment produces results.json with all metrics

### Commit
`feat(experiments): run comparative experiments with MSE, distributional, and validity losses`

---

## Stage 5: Comparison Notebook

**Goal**: Create a notebook comparing all three loss strategies side-by-side.

**File**: `tabpfn/LossComparison.ipynb`

### Contents

1. **Summary table**: All 6 experiments with key metrics (delta MSE, SCM validity, proximity, sparsity)
2. **SCM validity comparison**: Bar chart — MSE vs distributional vs validity for Exp 3 and 4
3. **Training curves**: Loss over epochs for each loss type
4. **Context ablation**: Does larger model + better loss improve ablation gap for Exp 4?
5. **Delta distribution analysis**: For distributional loss, show predicted variance vs actual error
6. **Validity vs proximity trade-off**: Scatter plot — do validity-optimized CFs sacrifice proximity?
7. **Conclusions**: Which loss works best for which scenario?

### Verification
- [ ] Notebook executes without errors
- [ ] All 6 experiment results loaded and compared

### Commit
`feat(notebook): add loss function comparison notebook`

---

## Execution Protocol

### For each stage:
1. Read the stage steps
2. Implement each step
3. Run verification checks
4. Create atomic commit

### Dependencies
- Stage 1: No dependencies
- Stage 2: Requires Stage 1 (uses scaled configs)
- Stage 3: Requires Stage 1 (uses scaled configs)
- Stage 4: Requires Stages 1, 2, 3
- Stage 5: Requires Stage 4

### Automation Progress Tracker

| # | Stage | Status | Notes | Updated |
|---|-------|--------|-------|---------|
| 1 | Scale Up Model | DONE | Added MEDIUM_MODEL base, EXP3S/EXP4S configs (~3.3M params, 4x scale-up), verified training | 2026-03-18 |
| 2 | Distributional Loss (NLL) | DONE | Added _distributional_loss(), loss_type param through training/eval pipeline, exp3s_dist/exp4s_dist configs | 2026-03-18 |
| 3 | Validity-Focused Loss | DONE | Added differentiable_forward to _MLP, _validity_loss with SCM forward+proximity+sparsity, exp3s_val/exp4s_val configs | 2026-03-18 |
| 4 | Run Comparative Experiments | PENDING | | |
| 5 | Comparison Notebook | PENDING | | |

Last stage completed: Stage 3 — Validity-Focused Loss
Last updated by: plan-runner-agent
