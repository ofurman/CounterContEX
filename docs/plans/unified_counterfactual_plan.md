# Plan: Counterfactual Generation — Unified Improvements & Experiments

## Overview

Combined plan that first builds the improved training infrastructure (fixing the root causes from v1), then validates the architecture through a progressive sequence of experiments with increasing complexity.

**Root causes from v1** (see `docs/results/counterfactual_generation_toy.md`):
1. Low label-flip rate (~6%) drowns signal in zero-delta noise
2. No per-dataset feature normalization — scales vary wildly across SCMs
3. Insufficient training duration / model too small
4. Monolithic delta prediction — hard to jointly learn "what to change" and "by how much"
5. No ground-truth validity evaluation (only heuristic)

**Experiment progression** — each level must pass before proceeding:
```
Exp 0: Linear SCM, 2 features, 1 perturbed        "Can the model learn a linear shift?"
  │ pass
  ▼
Exp 1: Nonlinear SCM, 3 features, fixed SCM        "Can it learn one nonlinear SCM?"
  │ pass
  ▼
Exp 2: Fixed SCM, more features + multi-perturb    "Does it scale within one SCM?"
  │ pass
  ▼
Exp 3: Small family of similar SCMs (5-10)          "Can it do in-context learning?"
  │ pass
  ▼
Exp 4: Diverse SCMs, varying structure              "Can it generalize across SCMs?"
```

If the model fails at level N, the problem is at level N — fix it before moving on.

---

## Stage 1: Increase Label-Flip Rate and Filter Training Signal

**Goal**: Ensure query positions contain meaningful (non-zero) deltas by increasing perturbation aggressiveness and filtering.

**Files**: `tabpfn/priors/counterfactual_prior.py`, `tabpfn/priors/counterfactual.py`

### Changes

1. **In `counterfactual_prior.py` `get_batch()`**: Add a retry loop — if a generated batch has flip rate < 20%, regenerate with `uniform_random` strategy (highest flip rate at ~7.4%) and increased `perturbation_magnitude`.

2. **In `counterfactual_prior.py` `_reorder_and_encode()`**: When not enough flipped samples exist for query positions, fill remaining queries with **duplicated flipped samples** (with noise) rather than non-flipped samples. This ensures every query has a meaningful delta target.

3. **In `counterfactual_prior.py`**: Add a `flip_only_queries` option (default `True`). When enabled, **all query positions** have label-flipped samples. Non-flipped pairs go exclusively to context. If fewer flipped samples than query slots, duplicate flipped samples.

4. **In `get_default_counterfactual_config()`**: Change defaults:
   - `perturbation_strategy`: `"uniform_random"` (highest flip rate)
   - `perturbation_prob`: `0.5` (perturb more features -> larger effect)
   - `perturbation_magnitude`: `2.0`

### Verification
- [ ] Average label flip rate per batch > 20%
- [ ] All query position deltas are non-zero (when `flip_only_queries=True`)
- [ ] Context positions still have correct factual labels
- [ ] Existing tests updated and passing

### Commit
`feat(prior): increase flip rate and filter queries to flipped-only samples`

---

## Stage 2: Per-Dataset Feature Normalization

**Goal**: Normalize features to zero mean / unit variance per dataset (per batch element) so the model sees consistent scales regardless of the SCM.

**Files**: `tabpfn/priors/counterfactual_prior.py`, `tabpfn/train_counterfactual.py`, `tabpfn/eval_counterfactual.py`

### Changes

1. **In `counterfactual_prior.py` `get_batch()`**: After generating and reordering, normalize features per batch element:
   ```python
   # Per-batch-element normalization
   for b in range(batch_size):
       mean = x[:, b, :].mean(dim=0, keepdim=True)  # (1, num_features)
       std = x[:, b, :].std(dim=0, keepdim=True).clamp(min=1e-6)
       x[:, b, :] = (x[:, b, :] - mean) / std
       # Scale deltas by the same std so they're in normalized space
       target_y[:, b, :] = target_y[:, b, :] / std
   ```
   This ensures:
   - All features have comparable scale (mean~0, std~1)
   - Deltas are expressed in units of feature standard deviations
   - The model can learn generalizable delta magnitudes

2. **In `eval_counterfactual.py` `generate_test_data()`**: Apply the same normalization so evaluation is consistent.

3. **In `eval_counterfactual.py` `run_inference()`**: When computing `x_cf_predicted`, un-normalize: `x_cf = x_query + pred_delta` is already in normalized space — report metrics in normalized space for comparability.

### Verification
- [ ] After normalization, per-feature mean ~ 0 and std ~ 1 across each batch element
- [ ] Deltas are in normalized scale (typical magnitude 0.1-3.0 rather than 2-26)
- [ ] MSE loss at initialization is much lower (should be ~1-5 instead of ~35)
- [ ] All tests pass

### Commit
`feat(prior): add per-dataset feature normalization for consistent scales`

---

## Stage 3: Improved Loss Function — Weighted MSE + Mask Supervision

**Goal**: Address the monolithic delta prediction problem by adding auxiliary supervision for **which features should change** (intervention mask).

**Files**: `tabpfn/train_counterfactual.py`, `tabpfn/priors/counterfactual_prior.py`

### Changes

1. **Expose intervention mask in training data**: Modify `counterfactual_prior.py` `get_batch()` to return the intervention mask alongside deltas. Pack it into `target_y` as extra channels:
   ```python
   # target_y shape: (seq_len, batch, num_features * 2)
   # First num_features channels: delta values
   # Last num_features channels: intervention mask (0/1)
   target_y = torch.cat([delta, mask.float()], dim=-1)
   ```

2. **In `train_counterfactual.py`**: Change `n_out = num_features * 2` — first half predicts deltas, second half predicts mask logits.

3. **Composite loss**:
   ```python
   pred_delta = output[..., :num_features]
   pred_mask_logits = output[..., num_features:]
   true_delta = target[..., :num_features]
   true_mask = target[..., num_features:]

   # Delta MSE — only on features that were actually changed
   delta_loss = ((pred_delta - true_delta) ** 2 * true_mask).sum(-1) / true_mask.sum(-1).clamp(min=1)

   # Mask BCE — learn which features to perturb
   mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, true_mask, reduction='none').mean(-1)

   # Combined: delta loss + mask loss (weighted)
   loss = delta_loss + 0.5 * mask_loss
   ```

   This decomposition means:
   - The model is **not penalized** for predicting non-zero delta on unperturbed features (mask handles this)
   - The model learns feature selection (mask) and perturbation magnitude (delta) jointly but with separate supervision
   - Zero-delta prediction is no longer the optimal lazy strategy

4. **At inference time**: Apply mask by thresholding `sigmoid(pred_mask_logits) > 0.5`, then zero out delta for non-selected features.

### Verification
- [ ] Output shape is `(num_query, batch, num_features * 2)`
- [ ] Composite loss computes without errors
- [ ] Mask BCE loss decreases during training (model learns which features are perturbed)
- [ ] Delta MSE loss decreases (conditioned on correctly predicted mask)
- [ ] All tests updated and passing

### Commit
`feat(train): add mask-supervised composite loss for decomposed prediction`

---

## Stage 4: Fixed-SCM Data Generator

**Goal**: Create a data generator that reuses one frozen SCM instance across all batches, generating fresh samples each time but with the same causal structure, node mapping, and class boundary. This is the foundation for all progressive experiments.

**Files**: `tabpfn/priors/counterfactual.py`, `tabpfn/priors/counterfactual_prior.py`

### Problem

Currently, `forward_with_internals()` re-randomizes the node-to-feature mapping (`random_perm`) on every call. For a fixed SCM, we need:
- Same MLP weights (already fixed per `_MLP` instance)
- Same `causes_mean` / `causes_std` (already fixed with `pre_sample_causes=True`)
- Same `random_perm` -> same `feature_indices` and `target_indices` (currently re-randomized)
- Same class boundary (median threshold from `BalancedBinarize` — currently recomputed per batch)

### Changes

1. **In `counterfactual.py`**: Add `forward_with_internals_fixed_mapping()` method to `_MLP`:
   ```python
   def forward_with_internals_fixed_mapping(self, fixed_perm=None):
       """Like forward_with_internals but with a fixed node permutation.

       If fixed_perm is None, sample a new permutation and return it.
       If fixed_perm is provided, reuse it for node selection.

       Returns:
           x, y, internals, fixed_perm
       """
   ```
   This method is identical to `forward_with_internals()` except it accepts/returns the `random_perm` tensor instead of always generating a new one.

2. **In `counterfactual.py`**: Add `generate_batch_fixed_scm()` method to `CounterfactualSCMGenerator`:
   ```python
   def generate_batch_fixed_scm(self, scm, fixed_perm, class_assigner,
                                 seq_len, num_features, num_outputs=1,
                                 perturbation_strategy=None):
       """Generate a batch using a pre-built SCM with fixed node mapping."""
   ```
   Key difference from `generate_batch()`: instead of creating `batch_size` independent SCMs, it calls `scm.forward_with_internals_fixed_mapping(fixed_perm)` once per batch element, getting fresh samples from the same SCM.

3. **In `counterfactual_prior.py`**: Add `get_batch_fixed_scm()` function and `FixedSCMDataLoader`:
   ```python
   def get_batch_fixed_scm(batch_size, seq_len, num_features, hyperparameters,
                            device, single_eval_pos, scm, fixed_perm,
                            class_assigner, **kwargs):
       """Generate training data from a single fixed SCM."""
   ```
   The `FixedSCMDataLoader` pre-creates the SCM once at init and reuses it for every `get_batch` call.

4. **Fix `BalancedBinarize`**: For a fixed SCM, compute the median from a large calibration set once and freeze it:
   ```python
   class FixedThresholdBinarize(nn.Module):
       def __init__(self, threshold):
           super().__init__()
           self.threshold = threshold

       def forward(self, x):
           return (x > self.threshold).float()
   ```
   At init: generate a large calibration batch (e.g., 10000 samples), compute median, freeze.

### Verification
- [ ] Fixed SCM produces data with consistent feature scales across batches
- [ ] `feature_indices` and `target_indices` are identical across calls
- [ ] Class assigner uses a frozen threshold (not recomputed per batch)
- [ ] Label flip rate is consistent across batches
- [ ] Calling `forward_with_internals_fixed_mapping` twice with same `fixed_perm` on same SCM yields different data points but same node structure

### Commit
`feat(prior): add fixed-SCM data generator for single-SCM overfitting experiments`

---

## Stage 5: SCM-Based Ground-Truth Validity Evaluation

**Goal**: Replace the heuristic validity check (delta norm > threshold) with ground-truth validation using the original SCM. This is the gold-standard validity check: feed the predicted counterfactual back through the same causal model and classify with the same decision boundary. Used across all experiments.

**Files**: `tabpfn/priors/counterfactual.py`, `tabpfn/priors/counterfactual_prior.py`, `tabpfn/eval_counterfactual.py`

### How it works

The SCM defines a causal DAG where features are intermediate nodes and the target y is a downstream node. To check if a predicted counterfactual `x_cf_pred` would actually achieve the target label:

1. **Intervene**: Set `do(X = x_cf_pred)` — overwrite all feature node values in `outputs_flat`
2. **Propagate**: Call `scm.forward_with_intervention(internals, interventions)`
3. **Classify**: Apply the same class assigner to the resulting `y_cf_pred`
4. **Compare**: Check if `class(y_cf_pred) == target_label`

### Changes

1. **In `counterfactual.py`**: Add `generate_batch_with_scm()` method to `CounterfactualSCMGenerator` that returns the batch **plus** a list of `(scm, internals, class_assigner)` tuples — one per batch element.

2. **In `counterfactual_prior.py`**: Add `get_batch_with_scm()` that wraps `generate_batch_with_scm()` and returns the formatted `(x, y, target_y)` plus the `scm_data` list.

3. **In `eval_counterfactual.py`**: New `compute_scm_validity()` function:
   ```python
   def compute_scm_validity(query_x, pred_deltas, target_labels,
                             scm_data_list, single_eval_pos, num_features):
       """Compute true validity by feeding predicted CFs through original SCMs.

       For each query point:
       1. x_cf_pred = query_x + pred_delta
       2. Build interventions: set all feature nodes to x_cf_pred values
       3. Re-propagate through SCM
       4. Classify and compare to target label

       Returns:
           validity_rate: float
           per_dataset_validity: list of floats
       """
   ```

4. **Sanity checks** (run automatically):
   - Feed true counterfactuals through SCM -> validity should be ~100%
   - Feed factual points (zero delta) through SCM -> validity should be ~0%
   - Feed random deltas through SCM -> validity should be ~50%

5. **Update `compute_metrics()` and `print_report()`**: Add `scm_validity` field, show both heuristic and SCM validity.

### Verification
- [ ] `generate_batch_with_scm()` returns SCM objects that can be re-used for intervention
- [ ] Re-feeding the **true** counterfactual through the SCM reproduces the original y_cf (sanity check)
- [ ] SCM validity for a random (untrained) model is near chance (~50% for binary)
- [ ] SCM validity for the true counterfactuals is ~100% (by construction)
- [ ] Report shows both heuristic and SCM validity side-by-side
- [ ] All tests pass

### Commit
`feat(eval): add SCM-based ground-truth validity evaluation using original causal model`

---

## Stage 6: Experiment Runner Framework

**Goal**: Create a unified experiment runner that takes a config and executes: data generator setup, model creation, training with logging, evaluation with all metrics, and results output. Used by all progressive experiments.

**Files**: `tabpfn/experiments/run_experiment.py`, `tabpfn/experiments/__init__.py`, `tabpfn/experiments/configs.py`

### Changes

1. **Create `tabpfn/experiments/run_experiment.py`** with:
   ```python
   def run_experiment(exp_config, scm_config, output_dir, device='cpu'):
       """Run a complete experiment: setup -> train -> evaluate -> report.

       Args:
           exp_config: dict with model/training params (emsize, nlayers, epochs, ...)
           scm_config: dict with SCM/data params (num_features, num_layers, ...)
           output_dir: path for results and checkpoints
           device: 'cpu' or 'cuda'

       Returns:
           results: dict with all metrics
       """
   ```

   Flow:
   - Set up data generator (fixed-SCM or random-SCM based on config)
   - Create model
   - Training loop with per-epoch logging (loss, delta MSE, sign accuracy, magnitude ratio)
   - Evaluation with SCM validity
   - Save results to `{output_dir}/results.json` and `{output_dir}/training_log.json`
   - Print summary report

2. **Create `tabpfn/experiments/configs.py`** with all experiment configs:
   ```python
   EXP0_CONFIG = dict(...)  # Linear sanity check
   EXP1_CONFIG = dict(...)  # Single nonlinear SCM
   EXP2A_CONFIG = dict(...) # Single SCM, 5 features
   EXP2B_CONFIG = dict(...) # Single SCM, 10 features
   EXP3_CONFIG = dict(...)  # Small SCM family
   EXP4_CONFIG = dict(...)  # Diverse SCMs
   ```
   Plus corresponding SCM configs for each.

3. **Training loop improvements** (in the runner):
   - Gradient accumulation (`aggregate_k=4` steps before optimizer.step())
   - Best-model checkpointing (track best loss, save state dict)
   - Per-epoch metrics: MSE loss, mean absolute delta, sign accuracy, max delta magnitude
   - Optional mask loss logging (when composite loss is used)

4. **CLI entry point**:
   ```python
   if __name__ == '__main__':
       parser.add_argument('--experiment', choices=['exp0','exp1','exp2a','exp2b','exp3','exp4'])
       parser.add_argument('--output-dir', default='docs/results')
       parser.add_argument('--device', default='cpu')
   ```

### Verification
- [ ] `run_experiment(EXP0_CONFIG, EXP0_SCM, ...)` runs end-to-end without errors
- [ ] Training log contains per-epoch metrics
- [ ] Results JSON contains all evaluation metrics including SCM validity
- [ ] Best model checkpoint is saved
- [ ] CLI entry point works

### Commit
`feat(experiments): add unified experiment runner framework with configs`

---

## Stage 7: Experiment 0 — Linear Sanity Check

**Goal**: Verify the pipeline works with the simplest possible case: a 1-layer linear SCM, 2 features, 1 perturbed feature, deterministic.

**Files**: `tabpfn/experiments/configs.py` (update), `tabpfn/experiments/run_experiment.py` (if needed)

### Setup

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| SCM depth | 1 layer (Linear only, `nn.Identity` activation) | Simplest possible causal model |
| Features | 2 | Minimal, easy to visualize |
| Perturbed features | 1 (always feature 0) | Removes "which feature" ambiguity |
| Perturbation | Fixed +1 std | Deterministic, no randomness in delta |
| SCM count | 1 fixed | Pure memorization |
| Noise std | 0.0 | Fully deterministic SCM |

### What the model must learn

With a 1-layer linear SCM: `y = W @ x + b`. The counterfactual delta is constant: `[1.0, 0.0]` in normalized space. This is a constant-output regression problem. Any model should learn this.

### Model config

```python
EXP0_CONFIG = dict(
    num_features=2, seq_len=32, batch_size=1,
    emsize=32, nlayers=2, nhead=2, nhid=64,
    dropout=0.0, epochs=50, steps_per_epoch=50,
    lr=0.001, warmup_epochs=3,
)
EXP0_SCM = dict(
    num_layers=2, num_causes=2, noise_std=0.0,
    prior_mlp_hidden_dim=4, prior_mlp_dropout_prob=0.0,
    prior_mlp_activations='Identity',
    perturbation_strategy="fixed_magnitude",
    perturbation_prob=1.0, fixed_magnitude_k=1.0,
)
```

### Success criteria

| Metric | Threshold |
|--------|-----------|
| Delta MSE (normalized) | < 0.01 |
| Delta sign accuracy | 100% |
| SCM validity | > 95% |
| Epochs to converge | < 30 |

### What failure means

Problem is in the **data pipeline** (encoding, normalization, target formatting), not the model. Debug the data flow.

### Run & report
```bash
python -m tabpfn.experiments.run_experiment --experiment exp0 --output-dir docs/results/exp0
```

### Commit
`feat(experiments): run Experiment 0 — linear sanity check`

---

## Stage 8: Experiment 1 — Single Nonlinear SCM, Few Features

**Goal**: Verify the model can learn a nonlinear counterfactual mapping for one fixed SCM.

### Setup

| Parameter | Value |
|-----------|-------|
| SCM depth | 2 layers + Tanh |
| Features | 3 |
| Perturbed features | 1-2 random |
| Perturbation | uniform_random |
| SCM count | 1 fixed |
| Noise std | 0.05 |

### Model config

```python
EXP1_CONFIG = dict(
    num_features=3, seq_len=64, batch_size=1,
    emsize=64, nlayers=4, nhead=2, nhid=128,
    dropout=0.0, epochs=200, steps_per_epoch=50,
    lr=0.001, warmup_epochs=5,
)
EXP1_SCM = dict(
    num_layers=3, num_causes=3, noise_std=0.05,
    prior_mlp_hidden_dim=8, prior_mlp_dropout_prob=0.2,
    prior_mlp_activations='Tanh',
    perturbation_strategy="uniform_random", perturbation_prob=0.5,
)
```

### Success criteria

| Metric | Threshold |
|--------|-----------|
| Delta MSE (normalized) | < 0.1 |
| Delta sign accuracy | > 85% |
| SCM validity | > 70% |
| Loss reduction | > 80% from initial |

### What failure means
Can't learn a fixed nonlinear mapping -> try (a) mask+delta loss, (b) larger model, (c) direct x_cf prediction instead of deltas, (d) MLP baseline (no transformer).

### Commit
`feat(experiments): run Experiment 1 — single nonlinear SCM overfitting`

---

## Stage 9: Experiment 2 — Single SCM, Scaling Features

**Goal**: Test scaling to more features within one fixed SCM (5, then 10 features).

### Setup

| Parameter | 2A | 2B |
|-----------|----|----|
| Features | 5 | 10 |
| SCM depth | 3 layers + Tanh | 3 layers + Tanh |
| seq_len | 128 | 256 |
| epochs | 300 | 500 |

### Model configs

```python
EXP2A_CONFIG = dict(
    num_features=5, seq_len=128, batch_size=1,
    emsize=128, nlayers=6, nhead=4, nhid=256,
    dropout=0.0, epochs=300, steps_per_epoch=100,
    lr=0.0005, warmup_epochs=10,
)
EXP2B_CONFIG = dict(
    num_features=10, seq_len=256, batch_size=1,
    emsize=128, nlayers=6, nhead=4, nhid=256,
    dropout=0.0, epochs=500, steps_per_epoch=100,
    lr=0.0003, warmup_epochs=20,
)
```

### Success criteria

| Metric | 5 features | 10 features |
|--------|-----------|-------------|
| Delta MSE (norm.) | < 0.15 | < 0.25 |
| Delta sign accuracy | > 80% | > 70% |
| SCM validity | > 60% | > 50% |
| Zero-feature accuracy | > 90% | > 85% |

"Zero-feature accuracy" = fraction of unperturbed features where `|pred_delta| < threshold`.

### What failure means
Scaling problem -> try (a) decomposed mask+delta, (b) per-feature output heads, (c) curriculum training.

### Commit
`feat(experiments): run Experiment 2 — single SCM feature scaling (5 & 10 features)`

---

## Stage 10: Experiment 3 — Small Family of Similar SCMs (In-Context Learning)

**Goal**: First test of in-context learning — can the model use context data from a specific SCM to infer which SCM it is and predict correct counterfactuals?

### Setup

| Parameter | Value |
|-----------|-------|
| SCM count | 10 fixed (pre-generated) |
| SCM structure | All same depth/width, different random weights |
| Features | 5 |
| Per batch | 4 randomly selected SCMs |
| seq_len | 128 (64 context + 64 query) |

### What the model must learn

All 10 SCMs have the same graph topology but different edge weights. The model sees context data `(x, y)` from one specific SCM and must infer which one to predict correct deltas. This is a 10-way implicit identification problem solved via in-context learning.

### Data loader changes

- Pre-generate 10 SCMs with fixed weights and node mappings at init
- Each batch element randomly selects one SCM and generates fresh data
- The model must use the context to figure out the causal parameters

### Model config

```python
EXP3_CONFIG = dict(
    num_features=5, seq_len=128, batch_size=4,
    emsize=128, nlayers=6, nhead=4, nhid=256,
    dropout=0.0, epochs=500, steps_per_epoch=100,
    lr=0.0003, warmup_epochs=20,
)
```

### Success criteria

| Metric | Threshold |
|--------|-----------|
| Delta MSE (norm.) | < 0.3 |
| SCM validity | > 50% |
| Per-SCM validity std | Low (works for all, not just some) |

### Context ablation diagnostic

To verify the model uses the context:
1. **Correct context**: feed data from same SCM -> measure validity
2. **Wrong context**: feed data from a different SCM -> validity should drop
3. **No context**: feed zeros for context y values -> validity should drop to random

If correct ~ wrong context -> model ignores context -> architecture problem.

### Commit
`feat(experiments): run Experiment 3 — SCM family in-context learning`

---

## Stage 11: Experiment 4 — Diverse SCMs, New at Test Time

**Goal**: True meta-learning — generalize to **unseen** SCMs with different structures.

### Setup

| Parameter | Value |
|-----------|-------|
| SCM depth | 2-4 layers (random) |
| SCM width | 8-32 hidden (random) |
| Features | 5 |
| Activation | Tanh or ReLU (random) |
| SCM count | New random SCM per batch element |

### Model config

```python
EXP4_CONFIG = dict(
    num_features=5, seq_len=256, batch_size=8,
    emsize=256, nlayers=8, nhead=4, nhid=512,
    dropout=0.1, epochs=200, steps_per_epoch=200,
    lr=0.0001, warmup_epochs=20, weight_decay=1e-4,
)
```

### Success criteria

| Metric | Threshold |
|--------|-----------|
| Delta MSE (norm.) | < 1.0 |
| SCM validity | > 30% |
| Context ablation gap | > 15% |

### What failure means
Exp 3 passes but Exp 4 fails -> model can do in-context identification but not in-context causal inference. Suggests: need more context data, architectural change, or problem is too hard for this model size.

### Commit
`feat(experiments): run Experiment 4 — diverse SCMs generalization`

---

## Stage 12: Results Notebook — Progressive Experiment Comparison

**Goal**: Notebook comparing all experiments: loss curves, validity progression, scaling analysis, and findings.

**File**: `tabpfn/ExperimentProgression.ipynb`

### Contents

1. **Summary table**: All experiments side-by-side with key metrics
2. **Loss curves**: Overlaid training curves for all experiments
3. **Validity progression**: SCM validity across experiments (bar chart)
4. **Context ablation**: Results from Exp 3 showing in-context learning
5. **Feature scaling analysis**: How metrics degrade with more features (Exp 2A vs 2B)
6. **Predicted vs true deltas**: Scatter plots per experiment
7. **Conclusions**: Which experiments passed, what was learned, next steps

### Verification
- [ ] Notebook runs end-to-end without errors
- [ ] All experiment results are loaded and compared
- [ ] Visualizations are clear and informative

### Commit
`feat(notebook): add progressive experiment comparison notebook`

---

## Execution Protocol

### For each stage:
1. Read the stage steps
2. Implement each step
3. Run verification checks
4. Create atomic commit

### Dependencies
- Stages 1-3: Independent infrastructure improvements (can be done in any order, but numbered for logical sequence)
- Stage 4: Requires Stages 1-2 (normalization, flip-only queries)
- Stage 5: Requires Stage 4 (needs SCM data for intervention)
- Stage 6: Requires Stages 1-5 (uses all infrastructure)
- Stage 7: Requires Stage 6 (uses experiment runner)
- Stages 8-11: Each requires the previous experiment to pass its success criteria
- Stage 12: Requires Stages 7-11 (compares all results)

### Automation Progress Tracker

| # | Stage | Status | Notes | Updated |
|---|-------|--------|-------|---------|
| 1 | Increase Flip Rate & Filter | DONE | Fixed _reorder_and_encode to accept flip_only_queries; duplicate flipped samples with noise when insufficient; added 3 new tests | 2026-03-17 |
| 2 | Per-Dataset Normalization | DONE | Added _normalize_per_batch in counterfactual_prior.py; normalize_features config flag (default True); 4 new tests | 2026-03-17 |
| 3 | Composite Loss (Delta + Mask) | DONE | Added mask supervision to prior (target_y=2*nf channels); composite loss with masked delta MSE + mask BCE; updated eval for mask thresholding; 4 new tests | 2026-03-17 |
| 4 | Fixed-SCM Data Generator | PENDING | | |
| 5 | SCM-Based Validity Evaluation | PENDING | | |
| 6 | Experiment Runner Framework | PENDING | | |
| 7 | Experiment 0: Linear Sanity Check | PENDING | | |
| 8 | Experiment 1: Single Nonlinear SCM | PENDING | | |
| 9 | Experiment 2: Feature Scaling | PENDING | | |
| 10 | Experiment 3: SCM Family (ICL) | PENDING | | |
| 11 | Experiment 4: Diverse SCMs | PENDING | | |
| 12 | Results Notebook | PENDING | | |

Last stage completed: Stage 3 — Composite Loss (Delta + Mask)
Last updated by: plan-runner-agent
