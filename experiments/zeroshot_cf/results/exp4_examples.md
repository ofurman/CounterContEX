# Experiment 4 (greedy): CF Examples — BINARY_CAT

Selector: prob_ascent, context: all_classes, tau: 0.5, temperature: 1e-09, stall_eps: 1e-06, n_permutations: 1, max_context: 128

## Example 1 (idx=0, VALID (flipped))
Factual class: 0, CF target: 1, CF predicted: 1
L0 (distinct features changed): 1; steps: 1
Recourse path (ordered): ['decision_code']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| decision_code | 0 | 1 | +1 * |
| segment_code | 1 | 1 | +0  |
| channel_code | 1 | 1 | +0  |

---

## Example 2 (idx=1, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (distinct features changed): 1; steps: 1
Recourse path (ordered): ['decision_code']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| decision_code | 1 | 0 | -1 * |
| segment_code | 0 | 0 | +0  |
| channel_code | 1 | 1 | +0  |

---

## Example 3 (idx=2, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (distinct features changed): 1; steps: 1
Recourse path (ordered): ['decision_code']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| decision_code | 1 | 0 | -1 * |
| segment_code | 0 | 0 | +0  |
| channel_code | 1 | 1 | +0  |

---

## Example 4 (idx=3, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (distinct features changed): 1; steps: 1
Recourse path (ordered): ['decision_code']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| decision_code | 1 | 0 | -1 * |
| segment_code | 0 | 0 | +0  |
| channel_code | 1 | 1 | +0  |

---

## Example 5 (idx=4, VALID (flipped))
Factual class: 0, CF target: 1, CF predicted: 1
L0 (distinct features changed): 1; steps: 1
Recourse path (ordered): ['decision_code']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| decision_code | 0 | 1 | +1 * |
| segment_code | 0 | 0 | +0  |
| channel_code | 0 | 0 | +0  |

---

# Experiment 4 (greedy): CF Examples — BINARY_CAT

Selector: prob_ascent, context: all_classes, tau: 0.5, temperature: 1e-09, stall_eps: 1e-06, n_permutations: 1, max_context: 64

## Example 1 (idx=0, VALID (flipped))
Factual class: 0, CF target: 1, CF predicted: 1
L0 (distinct features changed): 1; steps: 1
Recourse path (ordered): ['decision_code']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| decision_code | 0 | 1 | +1 * |
| segment_code | 1 | 1 | +0  |
| channel_code | 1 | 1 | +0  |

---

## Example 2 (idx=1, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (distinct features changed): 1; steps: 1
Recourse path (ordered): ['decision_code']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| decision_code | 1 | 0 | -1 * |
| segment_code | 0 | 0 | +0  |
| channel_code | 1 | 1 | +0  |

---

## Example 3 (idx=2, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (distinct features changed): 1; steps: 1
Recourse path (ordered): ['decision_code']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| decision_code | 1 | 0 | -1 * |
| segment_code | 0 | 0 | +0  |
| channel_code | 1 | 1 | +0  |

---

## Example 4 (idx=3, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (distinct features changed): 1; steps: 1
Recourse path (ordered): ['decision_code']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| decision_code | 1 | 0 | -1 * |
| segment_code | 0 | 0 | +0  |
| channel_code | 1 | 1 | +0  |

---

