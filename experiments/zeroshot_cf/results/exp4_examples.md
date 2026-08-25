# Experiment 4 (greedy): CF Examples — ADMISSION

Selector: prob_ascent, context: target_only, tau: 0.5, temperature: 1e-09, n_permutations: 3, max_context: 256

## Example 1 (idx=1, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (features changed): 1
Recourse path (ordered): ['CGPA']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| GRE Score | 317 | 317 | +0  |
| TOEFL Score | 107 | 107 | +0  |
| University Rating | 1 | 1 | +0  |
| SOP | 4 | 4 | +0  |
| LOR | 4 | 4 | +0  |
| CGPA | 8.563 | 7.53 | -1.034 * |
| Research | 0 | 0 | +0  |

---

## Example 2 (idx=2, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (features changed): 2
Recourse path (ordered): ['CGPA', 'GRE Score']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| GRE Score | 331 | 301 | -30 * |
| TOEFL Score | 107 | 107 | +0  |
| University Rating | 2 | 2 | +0  |
| SOP | 5 | 5 | +0  |
| LOR | 4 | 4 | +0  |
| CGPA | 9.428 | 7.53 | -1.899 * |
| Research | 1 | 1 | +0  |

---

## Example 3 (idx=3, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (features changed): 4
Recourse path (ordered): ['Research', 'LOR', 'CGPA', 'TOEFL Score']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| GRE Score | 331 | 331 | +0  |
| TOEFL Score | 115 | 98 | -17 * |
| University Rating | 3 | 3 | +0  |
| SOP | 6 | 6 | +0  |
| LOR | 5 | 0 | -5 * |
| CGPA | 9.428 | 7.53 | -1.899 * |
| Research | 1 | 0 | -1 * |

---

## Example 4 (idx=14, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (features changed): 3
Recourse path (ordered): ['CGPA', 'LOR', 'Research']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| GRE Score | 317 | 317 | +0  |
| TOEFL Score | 107 | 107 | +0  |
| University Rating | 3 | 3 | +0  |
| SOP | 5 | 5 | +0  |
| LOR | 6 | 0 | -6 * |
| CGPA | 8.563 | 7.53 | -1.034 * |
| Research | 1 | 0 | -1 * |

---

## Example 5 (idx=15, VALID (flipped))
Factual class: 1, CF target: 0, CF predicted: 0
L0 (features changed): 3
Recourse path (ordered): ['CGPA', 'GRE Score', 'LOR']

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| GRE Score | 331 | 301 | -30 * |
| TOEFL Score | 107 | 107 | +0  |
| University Rating | 3 | 3 | +0  |
| SOP | 6 | 6 | +0  |
| LOR | 7 | 0 | -7 * |
| CGPA | 8.563 | 7.53 | -1.034 * |
| Research | 1 | 1 | +0  |

---

