# Experiment 4 (greedy): CF Examples — MOONS

Selector: prob_ascent, context: target_only, tau: 0.5, temperature: 1e-09, n_permutations: 3, max_context: 256

## Example 1 (idx=0, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (features changed): 1
Recourse path (ordered): ['1']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| 0 | 0.7692 | 0.7692 | +0  |
| 1 | -0.4074 | 0.5831 | +0.9905 * |

---

## Example 2 (idx=1, VALID (flipped))
Factual class: 0, CF target: 1, CF predicted: 1
L0 (features changed): 1
Recourse path (ordered): ['0']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| 0 | -0.807 | 1.939 | +2.746 * |
| 1 | 0.2121 | 0.2121 | +0  |

---

## Example 3 (idx=3, VALID (flipped))
Factual class: 0, CF target: 1, CF predicted: 1
L0 (features changed): 1
Recourse path (ordered): ['0']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| 0 | -0.8687 | 1.801 | +2.67 * |
| 1 | 0.02874 | 0.02874 | +0  |

---

## Example 4 (idx=5, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (features changed): 1
Recourse path (ordered): ['1']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| 0 | 0.2535 | 0.2535 | +0  |
| 1 | -0.3404 | 0.9999 | +1.34 * |

---

## Example 5 (idx=7, VALID (flipped))
Factual class: 0, CF target: 1, CF predicted: 1
L0 (features changed): 2
Recourse path (ordered): ['0', '1']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| 0 | -0.5887 | 2.096 | +2.684 * |
| 1 | 0.8983 | 0.3765 | -0.5218 * |

---

