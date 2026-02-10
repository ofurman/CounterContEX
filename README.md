## Counterfactual Generation from SCM Prior

This fork extends TabPFN's SCM (Structural Causal Model) prior with a **counterfactual pair generation** module. It implements Pearl's Abduction-Action-Prediction framework to produce causally consistent factual-counterfactual pairs from the same generative process TabPFN uses for training data.

### Usage

```python
from tabpfn.priors.counterfactual import (
    CounterfactualSCMGenerator,
    get_default_counterfactual_config,
)

config = get_default_counterfactual_config()
config['perturbation_strategy'] = 'fixed_magnitude'
config['perturbation_prob'] = 0.3
config['fixed_magnitude_k'] = 2.0

gen = CounterfactualSCMGenerator(config, device='cpu')
batch = gen.generate_batch(batch_size=16, seq_len=200, num_features=10)

# batch.x_factual:              (200, 16, 10)
# batch.x_counterfactual:       (200, 16, 10)
# batch.y_factual_class:        (200, 16)
# batch.y_counterfactual_class: (200, 16)
# batch.label_flipped:          (200, 16) boolean
# batch.intervention_mask:      (200, 16, 10) boolean
# batch.perturbation_delta:     (200, 16, 10)
```

### Available Perturbation Strategies

| Strategy | Description |
|----------|-------------|
| `additive_noise` | Add Gaussian noise scaled by feature std |
| `marginal_replacement` | Replace with value from another sample |
| `gradient_guided` | Finite-difference gradient to find steepest direction |
| `fixed_magnitude` | Shift by k standard deviations in random direction |
| `uniform_random` | Replace with uniform sample from feature range |

---

## Counterfactual Generation Report

**TabPFN SCM Prior - Factual/Counterfactual Pair Statistics**

Configuration: `batch_size=16, seq_len=200, num_features=10, repeats=3`
Total samples per experiment: **9,600**

---

### Section 1: Strategy Comparison (default config)

#### 1.1 Validity (label flip rate)

| Strategy | Validity Rate | Samples |
|----------|---------------|---------|
| additive_noise | 0.0344 | 9600 |
| marginal_replacement | 0.0571 | 9600 |
| gradient_guided | 0.0052 | 9600 |
| fixed_magnitude | 0.0481 | 9600 |
| uniform_random | **0.0744** | 9600 |

#### 1.2 Proximity Metrics (mean +/- std)

| Strategy | L1 | L2 | Linf | Cosine Sim | Sparsity |
|----------|----|----|------|------------|----------|
| additive_noise | 21.564 +/- 15.631 | 14.193 +/- 8.984 | 11.978 +/- 7.638 | 0.9305 +/- 0.0872 | 0.301 +/- 0.140 |
| marginal_replacement | 31.118 +/- 25.131 | 20.252 +/- 14.249 | 16.901 +/- 11.704 | 0.8318 +/- 0.2182 | 0.299 +/- 0.142 |
| gradient_guided | **2.736 +/- 1.512** | **1.599 +/- 0.616** | **1.172 +/- 0.380** | **0.9992 +/- 0.0008** | 0.300 +/- 0.140 |
| fixed_magnitude | 29.655 +/- 15.611 | 17.095 +/- 6.019 | 12.303 +/- 3.518 | 0.9204 +/- 0.0693 | 0.304 +/- 0.141 |
| uniform_random | 42.419 +/- 29.358 | 27.529 +/- 16.042 | 22.920 +/- 13.075 | 0.7486 +/- 0.2541 | 0.302 +/- 0.138 |

#### 1.3 Per-Feature Mean Absolute Deviation

| Strategy | MAD (mean) | MAD (std) | # Features Perturbed |
|----------|------------|-----------|----------------------|
| additive_noise | 2.1564 | 0.2381 | 3.01 +/- 1.40 |
| marginal_replacement | 3.1118 | 0.2476 | 3.00 +/- 1.41 |
| gradient_guided | **0.2736** | **0.0250** | 3.00 +/- 1.40 |
| fixed_magnitude | 2.9655 | 0.2347 | 3.04 +/- 1.41 |
| uniform_random | 4.2419 | 0.4167 | 3.02 +/- 1.38 |

#### 1.4 Timing

| Strategy | Total Time (s) | Time/Sample (ms) |
|----------|----------------|------------------|
| additive_noise | 0.122 | 0.0127 |
| marginal_replacement | 0.080 | **0.0083** |
| gradient_guided | 0.230 | 0.0239 |
| fixed_magnitude | 0.082 | 0.0086 |
| uniform_random | 0.086 | 0.0090 |

---

### Section 2: Configuration Variant Comparison (strategy=fixed_magnitude)

| Variant | Validity | L2 dist | Cosine Sim | Sparsity | Time/Sample (ms) |
|---------|----------|---------|------------|----------|------------------|
| default | 0.0465 | 16.487 +/- 6.312 | 0.9254 | 0.302 | 0.0086 |
| high_perturbation | **0.1600** | 75.232 +/- 16.134 | 0.4474 | 0.699 | 0.0085 |
| low_perturbation | 0.0177 | 5.972 +/- 2.551 | **0.9879** | 0.171 | 0.0086 |
| deep_scm_6layers | 0.0821 | 26.996 +/- 8.741 | 0.9063 | 0.305 | 0.0200 |
| shallow_scm_2layers | 0.0000 | 8.589 +/- 3.893 | 0.9385 | 0.306 | **0.0046** |
| many_features_20 | 0.0667 | 23.700 +/- 7.039 | 0.9269 | 0.303 | 0.0112 |

---

### Section 3: Aggregate Summary

| Metric | Best Strategy | Value |
|--------|---------------|-------|
| Highest validity rate | uniform_random | 0.0744 |
| Lowest L2 distance | gradient_guided | 1.5987 |
| Highest cosine similarity | gradient_guided | 0.9992 |
| Fastest generation | marginal_replacement | 0.0083 ms/sample |
| Most sparse changes | marginal_replacement | sparsity=0.2993 |

#### Validity vs Proximity Tradeoff

```
additive_noise            validity: 0.034 |#
                          L2 dist:  14.193 |########################################

marginal_replacement      validity: 0.057 |##
                          L2 dist:  20.252 |########################################

gradient_guided           validity: 0.005 |
                          L2 dist:  1.599  |######

fixed_magnitude           validity: 0.048 |#
                          L2 dist:  17.095 |########################################

uniform_random            validity: 0.074 |##
                          L2 dist:  27.529 |########################################
```

### Key Observations

- **Gradient-guided** produces the most minimal (proximal) counterfactuals (L2=1.60, cosine=0.999) but has the lowest validity (0.5%) -- perturbations are too small to cross decision boundaries.
- **Uniform_random** achieves the highest validity (7.4%) but at the cost of large distances (L2=27.5), producing less realistic counterfactuals.
- **Deep SCMs (6 layers)** amplify perturbations through causal propagation, yielding 8.2% validity vs 0% for shallow 2-layer SCMs.
- **High perturbation config** (prob=0.7, k=3.0) reaches **16% validity** -- demonstrating that the generation framework scales with perturbation intensity.
- All strategies run at **0.008--0.024 ms/sample** on CPU. Generation is not a bottleneck.
- Average sparsity is ~0.30 across strategies (matching the default `perturbation_prob=0.3`), confirming that ~3 out of 10 features are perturbed per sample.
