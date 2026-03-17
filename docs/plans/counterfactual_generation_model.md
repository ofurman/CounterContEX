# Plan: Adapt TabPFN for Counterfactual Point Generation (Toy Experiment)

## Overview

Adapt the TabPFN transformer architecture to generate counterfactual points instead of class predictions. The model receives a tabular dataset with labels, a query point, and a target label, then outputs the counterfactual version of the query point that would achieve the target label. Training data comes from the existing SCM counterfactual generator.

**Core idea**: Instead of `f(dataset, query_x) → class`, we want `f(dataset, query_x, target_label) → delta_x` where `query_x + delta_x` is the counterfactual.

## Stage 1: Counterfactual Data Loader

**Goal**: Create a prior data loader that packages SCM counterfactual pairs into the format expected by `train.py`.

**File**: `tabpfn/priors/counterfactual_prior.py`

### Data format

The data loader must produce `(data, targets, single_eval_pos)` tuples where:
- `data = (style, x_src, y_src)` — the input tuple
- `targets` — the regression target (counterfactual deltas)
- `single_eval_pos` — train/test split position

### Encoding strategy

For each batch element (one SCM):
1. Generate factual-counterfactual pairs using `CounterfactualSCMGenerator`
2. Filter to only label-flipped pairs (where `label_flipped == True`)
3. If too few flips, relax to include all pairs
4. Split into context (training) and query (test) portions

**Context tokens** (positions 0..single_eval_pos-1):
- `x_src[i] = concat(x_factual[i], num_features)` — the factual features (padded to fixed size)
- `y_src[i] = y_factual_class[i]` — the factual class label

**Query tokens** (positions single_eval_pos..seq_len-1):
- `x_src[i] = concat(x_factual[i], num_features)` — the query point's features
- `y_src[i] = y_target_class[i]` — the **target** (desired) class label (= y_counterfactual_class)

**Targets** (regression ground truth):
- `targets[i] = x_counterfactual[i] - x_factual[i]` — the delta to predict
- Shape: `(num_query_points, batch_size, num_features)`

### Steps

1. Create `CounterfactualPriorDataLoader` class that wraps `CounterfactualSCMGenerator`
2. Implement `get_batch()` that:
   a. Calls `generator.generate_batch()` for each batch element
   b. Splits samples into context/query sets
   c. For context: uses factual (x, y_class) pairs
   d. For queries: uses factual x with **target** y_class (the desired counterfactual label)
   e. Target = delta (x_cf - x_factual) for query samples only
3. Handle the case where no labels flip by using all samples anyway (delta will be near-zero for non-flipped)
4. Register as a DataLoader via `get_batch_to_dataloader`

### Verification
- [ ] Data loader produces tensors with correct shapes
- [ ] Context y values are factual labels, query y values are target labels
- [ ] Targets are correct deltas matching x_cf - x_factual
- [ ] Label-flipped samples are prioritized for queries

### Commit
`feat(prior): add counterfactual generation data loader for TabPFN training`

---

## Stage 2: Model Architecture Adaptation

**Goal**: Modify the transformer to output continuous feature vectors (deltas) instead of class logits.

**Files**: `tabpfn/train_counterfactual.py` (new training script), reuse existing `transformer.py`

### Architecture changes (minimal)

The existing `TransformerModel` is already flexible enough. Key settings:
- **n_out = num_features** (output delta per feature dimension)
- **Loss = MSE** instead of CrossEntropy
- **y_encoder**: Must handle the fact that y is now a class label (integer) for both context and query. Use the same `Linear` y_encoder as standard TabPFN — it encodes a scalar y value.
- **Decoder**: Default `nn.Sequential(Linear(ninp, nhid), GELU, Linear(nhid, n_out))` works — it outputs `num_features` continuous values.

### Training script

Create `train_counterfactual.py` that:
1. Instantiates the `CounterfactualPriorDataLoader` from Stage 1
2. Sets up `TransformerModel` with:
   - `n_out = num_features` (predict delta for each feature)
   - `criterion = nn.MSELoss(reduction='none')`
   - Small architecture: `emsize=64, nlayers=4, nhead=2, nhid=128`
3. Uses the standard training loop from `train.py` (import and call `train()`)
4. Handles the MSE loss path (already supported in `train.py` line 142)

### Toy experiment parameters
```python
# SCM settings
num_features = 5
num_classes = 2
seq_len = 128  # samples per SCM dataset
batch_size = 16  # number of SCM datasets per batch

# Model settings
emsize = 64
nlayers = 4
nhead = 2
nhid = 128
dropout = 0.0

# Training settings
epochs = 50
steps_per_epoch = 100
lr = 0.001
bptt = 128  # max sequence length
```

### Steps

1. Create `train_counterfactual.py` with hardcoded toy experiment config
2. Wire up the `CounterfactualPriorDataLoader` as the prior
3. Set `n_out = num_features`, `criterion = MSELoss`
4. Use existing `train()` function from `train.py`
5. Add simple logging of MSE loss per epoch

### Verification
- [ ] Model instantiates without errors
- [ ] Forward pass produces output of shape `(num_query, batch, num_features)`
- [ ] Loss computes and backpropagation works
- [ ] Training runs for at least 5 epochs without NaN

### Commit
`feat(train): add counterfactual generation training script for toy experiment`

---

## Stage 3: Evaluation and Inference

**Goal**: Evaluate whether the trained model can generate valid counterfactuals.

**File**: `tabpfn/eval_counterfactual.py` (new)

### Evaluation metrics

1. **Delta MSE**: Mean squared error between predicted and true deltas
2. **Validity rate**: What fraction of generated counterfactuals actually flip the label when evaluated by the ground-truth SCM classifier
3. **Proximity**: L2 distance between query and generated counterfactual (lower = better, but must be non-zero for flipped labels)
4. **Sparsity**: Fraction of features with near-zero predicted delta (should be ~0.7 matching perturbation_prob=0.3)

### Inference pipeline

1. Generate a test batch from `CounterfactualSCMGenerator`
2. Feed context (x_factual, y_factual_class) + query (x_factual, y_target_class) to trained model
3. Model predicts `delta_predicted`
4. Compute `x_cf_predicted = x_factual + delta_predicted`
5. Classify `x_cf_predicted` using the same SCM class assigner
6. Compare with ground truth counterfactuals

### Steps

1. Create evaluation script that loads trained model
2. Generate 100 test SCM datasets
3. Compute all four metrics
4. Print summary report
5. Save a few example predictions for visual inspection

### Verification
- [ ] Evaluation runs end-to-end
- [ ] Metrics are computed correctly
- [ ] Results are printed in a readable format

### Commit
`feat(eval): add counterfactual generation evaluation script`

---

## Stage 4: End-to-End Notebook

**Goal**: Create a notebook that runs the full pipeline: data generation → training → evaluation.

**File**: `tabpfn/CounterfactualGenerationExperiment.ipynb`

### Contents

1. Introduction and problem statement
2. Data generation demo (show factual/counterfactual pairs)
3. Training with progress plots
4. Evaluation results with metrics
5. Example predictions visualization
6. Analysis: what works, what doesn't, next steps

### Verification
- [ ] Notebook runs end-to-end
- [ ] Training converges (loss decreases)
- [ ] At least some generated counterfactuals are valid

### Commit
`feat(notebook): add counterfactual generation experiment notebook`

---

## Execution Protocol

### For each stage:
1. Read the stage steps
2. Implement each step
3. Run verification checks
4. Create atomic commit

### Automation Progress Tracker

| # | Stage | Status | Notes | Updated |
|---|-------|--------|-------|---------|
| 1 | Counterfactual Data Loader | IN_PROGRESS | | 2026-03-17 |
| 2 | Model Architecture + Training | PENDING | | |
| 3 | Evaluation | PENDING | | |
| 4 | Notebook | PENDING | | |

Last stage completed: None
Last updated by: plan-runner-agent
