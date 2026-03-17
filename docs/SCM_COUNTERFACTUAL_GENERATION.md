# Counterfactual Pair Generation from SCM Prior — Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Background: Structural Causal Models](#background-structural-causal-models)
3. [Architecture Overview](#architecture-overview)
4. [The SCM Prior (MLP-based)](#the-scm-prior-mlp-based)
5. [Data Generation Flow: Step-by-Step](#data-generation-flow-step-by-step)
6. [Counterfactual Generation: Pearl's Three-Step Framework](#counterfactual-generation-pearls-three-step-framework)
7. [Perturbation Strategies](#perturbation-strategies)
8. [Classification and Label Assignment](#classification-and-label-assignment)
9. [Tensor Shapes and Conventions](#tensor-shapes-and-conventions)
10. [Configuration Reference](#configuration-reference)
11. [File Map](#file-map)

---

## Overview

This module generates **factual-counterfactual pairs** from synthetic tabular data. The data comes from randomly sampled Structural Causal Models (SCMs), which are implemented as Multi-Layer Perceptrons (MLPs) with random weights and structure. Each SCM defines a causal graph over latent and observed variables, and the counterfactual generation follows **Pearl's Abduction-Action-Prediction** framework:

1. **Abduction** — Run the SCM forward, recording all exogenous noise terms.
2. **Action** — Apply interventions (do-operator) to selected feature nodes.
3. **Prediction** — Re-propagate through the SCM with the *same* noise but intervened values.

This produces paired `(x_factual, y_factual)` and `(x_counterfactual, y_counterfactual)` samples that share identical latent noise, differing only due to the causal effect of interventions.

---

## Background: Structural Causal Models

A Structural Causal Model `M = (U, V, F)` consists of:

- **U** — Exogenous (noise) variables, drawn from known distributions. These represent unobserved randomness.
- **V** — Endogenous (observed) variables, determined by structural equations.
- **F** — Structural equations `v_i = f_i(pa(v_i), u_i)`, where `pa(v_i)` are the causal parents of `v_i`.

In this implementation:

| SCM concept | Implementation |
|---|---|
| Exogenous variables U | Root cause samples (`causes`) + per-layer `GaussianNoise` |
| Endogenous variables V | Hidden-layer neurons (internal nodes) + observed feature/target nodes |
| Structural equations F | Linear transformations + nonlinear activations (Tanh, etc.) |
| Causal graph structure | MLP layer connectivity with random weight initialization + dropout |

The key insight: an MLP with random weights naturally defines a Directed Acyclic Graph (DAG). Each neuron at layer `L` depends on all neurons at layer `L-1` (weighted by the random weight matrix), plus an independent noise term. Sparsity in the causal graph is induced by **dropout** on the weight matrices (setting random connections to zero during initialization).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                  CounterfactualSCMGenerator                         │
│                                                                     │
│  generate_batch(batch_size, seq_len, num_features)                 │
│       │                                                             │
│       ├── For each batch element:                                   │
│       │     ├── _sample_scm()  ──→  _build_mlp()  ──→  _MLP()     │
│       │     │         Constructs a random MLP with:                 │
│       │     │         • Random weight matrices (Gaussian init)      │
│       │     │         • Dropout-based sparsity                      │
│       │     │         • Per-layer GaussianNoise modules             │
│       │     │                                                       │
│       │     ├── forward_with_internals()  [ABDUCTION]              │
│       │     │         Runs factual forward pass, captures:          │
│       │     │         • Root cause samples                          │
│       │     │         • Per-layer noise tensors                     │
│       │     │         • All layer outputs                           │
│       │     │         • Node-to-feature/target mapping              │
│       │     │                                                       │
│       │     ├── _select_perturbation_targets()  [ACTION setup]     │
│       │     │         Selects random subset of features to perturb  │
│       │     │                                                       │
│       │     ├── _compute_interventions()  [ACTION]                 │
│       │     │         Computes delta values using chosen strategy   │
│       │     │                                                       │
│       │     └── forward_with_intervention()  [PREDICTION]          │
│       │               Re-propagates SCM with same noise +           │
│       │               intervened node values                        │
│       │                                                             │
│       └── _build_class_assigner()                                   │
│             Shared classifier applied to both factual and CF        │
│             targets to produce discrete labels                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The SCM Prior (MLP-based)

### How the MLP Defines a Causal Graph

The core SCM is an MLP with the following structure:

```
Layer 0:  Linear(num_causes → hidden_dim)
Layer 1:  Sequential(Activation → Linear(hidden_dim → hidden_dim) → GaussianNoise)
Layer 2:  Sequential(Activation → Linear(hidden_dim → hidden_dim) → GaussianNoise)
  ...
Layer N:  Sequential(Activation → Linear(hidden_dim → hidden_dim) → GaussianNoise)
```

Each `Sequential` block represents one "layer" of the causal graph:

```
              Activation
                  │
          Linear transform           ← structural equation (weighted sum of parents)
                  │
          GaussianNoise              ← exogenous noise (U_i)
                  │
          [layer output]             ← endogenous variable (V_i)
```

**Key settings:**

- **`num_causes`** (default: 5) — Number of root exogenous variables. These are the "external inputs" to the SCM.
- **`num_layers`** (default: 4) — Depth of the causal graph. More layers = deeper causal chains.
- **`prior_mlp_hidden_dim`** (default: 64, but at least `num_outputs + 2*num_features`) — Width of each layer, i.e., number of endogenous nodes per depth level.
- **`prior_mlp_dropout_prob`** (default: 0.3) — Controls sparsity: 30% of weight connections are zeroed out at initialization, meaning ~30% of causal edges are absent.
- **`noise_std`** (default: 0.1) — Standard deviation of the per-layer Gaussian noise.
- **`init_std`** (default: 1.0) — Scale of the random weight initialization, controlling the magnitude of causal effects.

### Root Cause Sampling

Root causes (the exogenous inputs to the graph) are sampled via `causes_sampler_f`:

```python
def causes_sampler_f(num_causes):
    means = np.random.normal(0, 1, (num_causes))
    std = np.abs(np.random.normal(0, 1, (num_causes)) * means)
    return means, std
```

Each root cause dimension gets its own randomly-sampled mean and standard deviation. When `pre_sample_causes=True`, these parameters are fixed per SCM, so all `seq_len` samples share the same distributional parameters but get independent draws.

Three sampling modes are supported:
- **`normal`** (default) — Gaussian with per-cause mean/std.
- **`mixed`** — Randomly mixes Gaussian, multinomial (categorical), and Zipf (heavy-tailed) distributions across cause dimensions.
- **`uniform`** — Uniform [0,1] for all causes.

### Weight Initialization and Sparsity

Weight matrices are initialized with Gaussian weights, then sparsified:

```python
# Standard initialization with dropout-based sparsity
nn.init.normal_(p, std=init_std / (1.0 - dropout_prob ** 0.5))
p *= torch.bernoulli(torch.zeros_like(p) + 1.0 - dropout_prob)
```

This means:
1. Weights are drawn from `N(0, init_std / (1 - dropout^0.5))` — scaled up to compensate for the dropout.
2. Each weight is independently zeroed with probability `dropout_prob`.
3. The result is a sparse weight matrix where ~30% of causal connections are absent.

An alternative **block-wise dropout** mode creates structured sparsity by only initializing weights within random diagonal blocks.

### Node Selection: From MLP to Observed Variables

The MLP produces many more internal nodes than the user requests as `num_features`. The mapping from internal nodes to observed variables works as follows:

1. All layer outputs (except the root causes and the first hidden layer) are concatenated into a flat vector: `outputs_flat`.
2. A random permutation selects which flat indices become:
   - **Target (y)**: By default (`y_is_effect=True`), the target is always the **last** node(s) in `outputs_flat` — i.e., the deepest layer. This ensures y is causally downstream of features.
   - **Features (x)**: `num_features` nodes are randomly selected from the remaining positions.

This means features can come from **different layers** of the MLP, representing variables at different depths in the causal graph. Some features may be ancestors of others.

---

## Data Generation Flow: Step-by-Step

### Step 1: Sample an SCM

`_sample_scm()` → `_build_mlp()` instantiates the `_MLP` class, which:

1. Randomly samples root cause distribution parameters via `causes_sampler_f`.
2. Constructs `num_layers` sequential blocks, each with `Activation → Linear → GaussianNoise`.
3. Randomly initializes all weight matrices with Gaussian weights + dropout sparsity.
4. Per-layer `GaussianNoise` modules get their own noise standard deviations (sampled once, reused for all data points).

### Step 2: Factual Forward Pass (Abduction)

`forward_with_internals()` runs the factual data generation:

```
Root causes U₀ ~ N(μ_cause, σ_cause)       shape: (seq_len, 1, num_causes)
       │
       ▼
  [Linear₀]  → H₀                          shape: (seq_len, 1, hidden_dim)
       │
       ▼
  [Tanh → Linear₁ → +Noise₁] → H₁         shape: (seq_len, 1, hidden_dim)
       │                  │
       │            capture noise₁
       ▼
  [Tanh → Linear₂ → +Noise₂] → H₂         shape: (seq_len, 1, hidden_dim)
       │                  │
       │            capture noise₂
       ▼
      ...                ...
       │
       ▼
  [Tanh → LinearN → +NoiseN] → HN          shape: (seq_len, 1, hidden_dim)
                          │
                    capture noiseN
```

After propagation, the method:

1. Concatenates `H₁, H₂, ..., HN` into `outputs_flat` (shape: `(seq_len, 1, total_nodes)`).
2. Randomly selects `num_features` node indices → these become **x** (observed features).
3. Selects target node indices (typically the last nodes) → these become **y** (target).
4. Returns `x`, `y`, and an `internals` dict containing:
   - `cause_noise`: the root cause samples
   - `layer_noises`: list of all noise tensors `[noise₁, noise₂, ..., noiseN]`
   - `layer_outputs`: full list of per-layer outputs
   - `outputs_flat`: concatenated node values
   - `node_mapping`: which flat indices correspond to features and targets
   - `layer_boundaries`: `[(start, end, layer_idx), ...]` for mapping flat indices back to layers

### Step 3: Select Perturbation Targets

`_select_perturbation_targets()` creates a boolean mask `(seq_len, num_features)`:

- Each feature is independently selected with probability `perturbation_prob` (default: 0.3).
- At least one feature per sample is guaranteed to be perturbed.

### Step 4: Compute Interventions (Action)

`_compute_interventions()` computes the actual delta values using the chosen perturbation strategy (see [Perturbation Strategies](#perturbation-strategies)).

The result is a dict mapping `outputs_flat` indices to new node values:
```python
interventions = {
    flat_idx_feat_3: new_value_tensor,   # (seq_len, 1)
    flat_idx_feat_7: new_value_tensor,   # (seq_len, 1)
    ...
}
```

### Step 5: Counterfactual Forward Pass (Prediction)

`forward_with_intervention()` re-propagates through the **same SCM** with:

- **Same root causes** (identical `causes` tensor)
- **Same noise terms** at every layer (replayed from `layer_noises`)
- **Interventions applied** at the correct layer depth

The re-propagation proceeds layer by layer:

```
Same root causes U₀                        (unchanged)
       │
       ▼
  [Linear₀]  → H₀'                         (may differ if a downstream
       │                                     intervention propagated back —
       │                                     but in practice, H₀ = H₀'
       ▼                                     since interventions only affect
  [Tanh → Linear₁ → +Noise₁'] → H₁'        downstream nodes)
       │
       │  ← Apply intervention at layer 1 if any feature maps here
       │     H₁'[:, :, within_idx] = intervened_value
       ▼
  [Tanh → Linear₂ → +Noise₂'] → H₂'
       │
       │  ← Apply intervention at layer 2 if any feature maps here
       ▼                                     The intervention at layer 1
      ...                                    has already causally propagated
       │                                     through layers 2, 3, ...
       ▼
  [Tanh → LinearN → +NoiseN'] → HN'
```

**Critical property**: When a feature at layer L is intervened on, all nodes at layers L+1, L+2, ... are affected (they recompute their values using the intervened input). Nodes at layers before L remain unchanged because they have the same inputs and noise. This is exactly the causal semantics of Pearl's `do(X=x)` operator.

### Step 6: Classification

Both factual and counterfactual continuous targets are passed through a **shared** class assigner:

- **`BalancedBinarize`** (default): `class = (y > median(y)).float()` — produces perfectly balanced binary labels. The median is computed over the *full* batch.
- **`MulticlassRank`**: Picks random threshold samples, counts how many thresholds each y exceeds.
- **`MulticlassValue`**: Uses fixed learned thresholds.
- **`MulticlassMultiNode`**: Uses multinomial sampling from sigmoid outputs.

Using the **same** class assigner for both factual and counterfactual ensures that the decision boundary is identical, so `label_flipped` is meaningful.

### Step 7: Batch Assembly

The per-SCM results are concatenated along the batch dimension:

```python
x_factual:         (seq_len, batch_size, num_features)
y_factual_class:   (seq_len, batch_size)
x_counterfactual:  (seq_len, batch_size, num_features)
y_cf_class:        (seq_len, batch_size)
label_flipped:     (seq_len, batch_size)  — boolean
intervention_mask: (seq_len, batch_size, num_features)  — boolean
perturbation_delta:(seq_len, batch_size, num_features)  — float
```

---

## Counterfactual Generation: Pearl's Three-Step Framework

### 1. Abduction

**Goal**: Infer the values of all exogenous variables U given the observed evidence.

In this implementation, abduction is trivial because we *generate* the data — we already know all noise terms. The `forward_with_internals()` method explicitly captures:
- Root cause samples: `causes` ~ `N(μ_cause, σ_cause)`
- Per-layer noise: `noise_i` ~ `N(0, σ_noise_i)` for each GaussianNoise module

These are stored in the `internals` dict for replay.

### 2. Action (Intervention / do-operator)

**Goal**: Modify the structural equations to set `do(X_j = x_j')` for selected features.

The implementation:
1. Selects features to perturb (random subset, `perturbation_prob=0.3`).
2. Computes new values: `x_j' = x_j + delta_j` where `delta_j` depends on the perturbation strategy.
3. Maps each selected feature to its corresponding node in `outputs_flat`.
4. Stores `{flat_index: new_value}` in the `interventions` dict.

The intervention **replaces** the node's value at the appropriate layer, overriding the output of the structural equation. Subsequent layers then take this intervened value as input.

### 3. Prediction

**Goal**: Compute the counterfactual outcome under the intervention with the *same* exogenous noise.

`forward_with_intervention()` replays the entire SCM:
- Uses identical root causes
- Injects the **same** per-layer noise tensors via `pre_sampled_noise`
- After computing each layer's output, checks if any intervention applies and overwrites the relevant neuron values
- Downstream layers see the intervened values and propagate causally

The result is a counterfactual world where "everything else held equal" (same U), but the intervened features take new values and their causal descendants update accordingly.

---

## Perturbation Strategies

All strategies return a delta tensor of shape `(seq_len, num_features)`. Deltas are zeroed for non-selected features.

### `additive_noise`

```
delta = N(0,1) * feature_std * magnitude
```

Adds Gaussian noise scaled by each feature's empirical standard deviation. Produces moderate, naturally-scaled perturbations.

### `marginal_replacement`

```
delta = x[random_other_sample] - x[current_sample]
```

Replaces the feature value with a value drawn from another sample in the same dataset. Effectively samples from the feature's marginal distribution.

### `gradient_guided`

```
grad_i = (y(x + eps*e_i) - y(x)) / eps       # finite-difference gradient
delta = sign(grad) * step_size * feature_std * magnitude * random_flip
```

Estimates the numerical gradient of y with respect to each feature via forward finite differences. Steps in the direction of steepest change. A random sign flip (50%) adds diversity. This is the most computationally expensive strategy (~2.7x slower) because it requires one extra forward pass per feature.

### `fixed_magnitude`

```
delta = random_direction(+1/-1) * k * feature_std
```

Shifts each feature by exactly `k` standard deviations (default `k=1.0`) in a random direction. Produces uniform-magnitude perturbations.

### `uniform_random`

```
delta = Uniform(feature_min, feature_max) - current_value
```

Replaces the feature with a uniformly random value from the observed feature range. Tends to produce the largest perturbations and highest validity rates.

### Strategy Comparison Summary

| Strategy | Proximity | Validity | Speed | Character |
|---|---|---|---|---|
| `additive_noise` | Moderate | Low (~3.9%) | Fast | Natural, Gaussian perturbations |
| `marginal_replacement` | Low | Moderate (~5.7%) | Fastest | Realistic replacement values |
| `gradient_guided` | Highest | Lowest (~0.7%) | Slowest | Minimal perturbations, too small to flip |
| `fixed_magnitude` | Moderate | Low (~4.2%) | Fast | Controlled, uniform magnitude |
| `uniform_random` | Lowest | Highest (~7.4%) | Fast | Aggressive, covers full range |

---

## Classification and Label Assignment

The continuous target `y` (from the SCM) is converted to discrete class labels using a **shared** class assigner. This is critical: both factual and counterfactual targets pass through the *same* assigner so that label comparison is meaningful.

### `BalancedBinarize` (default)

```python
class BalancedBinarize(nn.Module):
    def forward(self, x):
        return (x > torch.median(x)).float()
```

Splits at the median → perfectly balanced 50/50 classes. The median is computed over all samples in the batch, creating a global decision boundary.

**Why validity is low**: Because the median boundary is fixed, a counterfactual label flip requires the perturbation to push the continuous target `y_cf` from one side of the median to the other. Most perturbations change `y` slightly but not enough to cross this boundary.

---

## Tensor Shapes and Conventions

All tensors follow TabPFN's `(seq_len, batch_size, ...)` convention:

| Tensor | Shape | Description |
|---|---|---|
| `x_factual` | `(S, B, F)` | Factual feature values |
| `x_counterfactual` | `(S, B, F)` | Counterfactual feature values |
| `y_factual` | `(S, B)` | Continuous factual target |
| `y_factual_class` | `(S, B)` | Classified factual target |
| `y_counterfactual` | `(S, B)` | Continuous counterfactual target |
| `y_counterfactual_class` | `(S, B)` | Classified counterfactual target |
| `label_flipped` | `(S, B)` | Boolean: did class label change? |
| `intervention_mask` | `(S, B, F)` | Boolean: which features were intervened on |
| `perturbation_delta` | `(S, B, F)` | Actual delta applied to each feature |

Where `S = seq_len`, `B = batch_size`, `F = num_features`.

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| **Perturbation** | | |
| `perturbation_strategy` | `"additive_noise"` | Which strategy to use |
| `perturbation_prob` | `0.3` | Probability each feature is perturbed per sample |
| `perturbation_magnitude` | `1.0` | Scale factor for additive/gradient strategies |
| `gradient_step_size` | `0.1` | Step size for gradient-guided strategy |
| `fixed_magnitude_k` | `1.0` | Number of std devs for fixed_magnitude strategy |
| **Classification** | | |
| `num_classes` | `2` | Number of target classes (0 = regression) |
| `balanced` | `True` | Use BalancedBinarize for 2-class |
| `multiclass_type` | `"rank"` | Multiclass method: rank/value/multi_node |
| **SCM Structure** | | |
| `num_layers` | `4` | Depth of the causal graph |
| `prior_mlp_hidden_dim` | `64` | Width per layer (auto-expanded if needed) |
| `prior_mlp_dropout_prob` | `0.3` | Weight sparsity (fraction of zero connections) |
| `noise_std` | `0.1` | Exogenous noise magnitude per layer |
| `init_std` | `1.0` | Weight initialization scale |
| `num_causes` | `5` | Number of root exogenous variables |
| `is_causal` | `True` | Enable causal node selection from graph |
| `y_is_effect` | `True` | Force target to be at the deepest layer |
| `pre_sample_weights` | `True` | Pre-sample per-dimension noise stds |
| `pre_sample_causes` | `True` | Pre-sample per-cause distribution params |
| `sampling` | `"normal"` | Root cause distribution: normal/mixed/uniform |
| `prior_mlp_activations` | `Tanh` | Nonlinearity in structural equations |
| `block_wise_dropout` | `False` | Use structured block sparsity |
| `sort_features` | `False` | Sort selected feature indices |
| `in_clique` | `False` | Select features from a contiguous block |
| `random_feature_rotation` | `False` | Cyclically rotate feature ordering |
| `prior_mlp_scale_weights_sqrt` | `True` | Use sqrt scaling for dropout compensation |

---

## File Map

| File | Role |
|---|---|
| `tabpfn/priors/counterfactual.py` | Main module: `CounterfactualSCMGenerator`, perturbation strategies, `CounterfactualBatch` dataclass |
| `tabpfn/priors/mlp.py` | Original MLP prior with `get_batch()`, `GaussianNoise` module, `causes_sampler_f`, and the `MLP` class with `forward_with_internals()` / `forward_with_intervention()` methods |
| `tabpfn/priors/flexible_categorical.py` | Classification modules: `BalancedBinarize`, `MulticlassRank`, `MulticlassValue`, `MulticlassMultiNode`, and the `FlexibleCategorical` wrapper |
| `tabpfn/priors/prior.py` | Base `PriorDataLoader` class |
| `tabpfn/priors/utils.py` | Utilities: `get_batch_to_dataloader`, sampling helpers, `CategoricalActivation`, `randomize_classes` |
