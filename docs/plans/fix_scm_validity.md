# Plan: Fix SCM Validity — Reuse Exact Internals from Data Generation

## Overview

SCM validity of TRUE counterfactuals is ~85.6% instead of ~100% because `compute_scm_validity()` uses `internals` from a **separate** forward pass, not the one that generated the factual-counterfactual pairs. The exogenous noise (root causes + layer noises) in those internals doesn't match the query points being validated. Fix: thread the per-sample internals from data generation all the way through to the validity checker.

## Root Cause

In `_evaluate_fixed_scm()` (run_experiment.py:67-68):
```python
_, _, internals, _ = fixed_dl.scm.forward_with_internals_fixed_mapping(fixed_dl.fixed_perm)
```
This creates **new** internals with **new** noise for each test batch. These internals are passed to `compute_scm_validity()`, which uses them to re-propagate predicted CFs through the SCM. But the noise in these internals is unrelated to the noise used when generating the test data.

The correct flow: when `get_batch_fixed_scm()` calls `generate_batch_fixed_scm()` → `forward_with_internals_fixed_mapping()`, it already generates internals with the right noise. These internals must be preserved and returned alongside `(x, y, target_y)`.

---

## Stage 1: Return Per-Sample Internals from Data Generation

**Goal**: Make `generate_batch_fixed_scm()` and `get_batch_fixed_scm()` return the internals used to generate each sample, so the validity check can reuse the exact same noise.

**Files**: `tabpfn/priors/counterfactual.py`, `tabpfn/priors/counterfactual_prior.py`

### Changes

1. **In `counterfactual.py` `generate_batch_fixed_scm()`**: Already generates internals per batch element via `scm.forward_with_internals_fixed_mapping(fixed_perm)`. Collect and return them:
   ```python
   def generate_batch_fixed_scm(self, scm, fixed_perm, class_assigner, ...):
       # ... existing code ...
       all_internals = []
       for _ in range(batch_size):
           x_f, y_f, internals = scm.forward_with_internals_fixed_mapping(fixed_perm)
           all_internals.append(internals)
           # ... existing perturbation + counterfactual generation ...
       return batch, all_internals  # NEW: return internals list
   ```

2. **In `counterfactual_prior.py` `get_batch_fixed_scm()`**: Accept and return internals:
   ```python
   def get_batch_fixed_scm(..., return_internals=False, ...):
       batch, batch_internals = gen.generate_batch_fixed_scm(...)
       # ... existing reorder + normalize ...
       if return_internals:
           return x, y, target_y, batch_internals
       return x, y, target_y
   ```

3. **In `counterfactual_prior.py` `FixedSCMDataLoader.__iter__()`**: When return_internals is requested, yield internals alongside data:
   ```python
   def __iter__(self):
       for _ in range(self.num_steps):
           x, y, target_y, batch_internals = get_batch_fixed_scm(
               ..., return_internals=True, ...
           )
           yield (None, x, y), target_y, self.single_eval_pos, batch_internals
   ```
   When return_internals is False (default for training), keep the current 3-tuple yield for backward compatibility.

### Verification
- [ ] `generate_batch_fixed_scm()` returns `(batch, internals_list)` where each internals has the correct noise
- [ ] `get_batch_fixed_scm(..., return_internals=True)` returns 4-tuple
- [ ] `get_batch_fixed_scm(..., return_internals=False)` returns 3-tuple (backward compat)
- [ ] FixedSCMDataLoader can yield internals when requested
- [ ] Existing tests pass unchanged

### Commit
`feat(prior): thread per-sample internals from data generation for validity checking`

---

## Stage 2: Fix compute_scm_validity to Use Matched Internals

**Goal**: Update `compute_scm_validity()` and `_evaluate_fixed_scm()` to use the exact internals from data generation instead of creating new ones.

**Files**: `tabpfn/eval_counterfactual.py`, `tabpfn/experiments/run_experiment.py`

### Changes

1. **In `run_experiment.py` `_evaluate_fixed_scm()`**: Collect internals from the test data loader instead of generating new ones:
   ```python
   # BEFORE (wrong — creates new internals with new noise):
   for data, target_y, sep in test_dl:
       ...
       _, _, internals, _ = fixed_dl.scm.forward_with_internals_fixed_mapping(fixed_dl.fixed_perm)
       scm_data_list.append({'scm': ..., 'internals': internals, ...})

   # AFTER (correct — uses internals from data generation):
   for data, target_y, sep, batch_internals in test_dl:
       ...
       scm_data_list.append({'scm': ..., 'internals': batch_internals[0], ...})
   ```

2. **In `eval_counterfactual.py` `compute_scm_validity()`**: Fix the per-query-point intervention to use the correct sample position from internals:
   ```python
   # BEFORE (wrong — broadcasts one CF value across all seq_len positions):
   for q in range(batch_x_cf.shape[0]):
       new_val = outputs_flat[:, :, flat_idx].clone()
       new_val[:, :] = batch_x_cf[q, feat_idx]  # ALL positions set to same value

   # AFTER (correct — only set the specific sample position):
   for q in range(batch_x_cf.shape[0]):
       sample_pos = q  # the query position in the sequence
       new_val = outputs_flat[:, :, flat_idx].clone()
       new_val[sample_pos, :] = batch_x_cf[q, feat_idx]  # only this position
   ```

   And read the result from the correct position:
   ```python
   # BEFORE:
   pred_class = y_cf_class[0, 0].item()  # always position 0

   # AFTER:
   pred_class = y_cf_class[sample_pos, 0].item()  # correct position
   ```

3. **Handle sample position mapping**: The query points come from positions `single_eval_pos` through `seq_len-1` in the original batch. The `_reorder_and_encode()` function reorders samples, so we need to track which original sample index maps to each query position. Add a `query_sample_indices` return from `_reorder_and_encode()` to enable correct position lookup.

### Verification
- [ ] SCM validity of TRUE counterfactuals is ~100% (the key test)
- [ ] SCM validity of ZERO deltas is ~0%
- [ ] SCM validity of random deltas is ~50%
- [ ] SCM validity of trained model predictions is higher than before
- [ ] All existing tests pass

### Commit
`fix(eval): use matched internals in SCM validity for correct noise correspondence`

---

## Stage 3: Sanity Test and Experiment Re-run

**Goal**: Add an automated sanity test that verifies true-CF validity is ~100%, then re-run Exp 1 with the fix.

**Files**: `tabpfn/tests/test_eval_counterfactual.py`, `tabpfn/experiments/run_experiment.py`

### Changes

1. **Add test `test_true_cf_scm_validity_is_near_100()`**: Generate data from a fixed SCM, compute SCM validity of the true deltas using matched internals, assert > 0.95.

2. **Fix the existing failing test `test_true_cf_through_scm_matches_original_label`** to use matched internals.

3. **Re-run Experiment 1**: With corrected SCM validity, verify that the model's validity is now accurately measured.

### Verification
- [ ] New sanity test passes (true CF validity > 95%)
- [ ] Previously failing test now passes
- [ ] Exp 1 results show corrected SCM validity
- [ ] All tests pass

### Commit
`test(eval): add sanity test for true-CF validity and re-run Exp 1 with fix`

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
| 1 | Return Per-Sample Internals | DONE | generate_batch_fixed_scm returns (batch, internals), get_batch_fixed_scm supports return_internals flag, FixedSCMDataLoader yields internals when requested | 2026-03-18 |
| 2 | Fix compute_scm_validity | DONE | _evaluate_fixed_scm uses matched internals from FixedSCMDataLoader; _reorder_and_encode returns query_source_indices; compute_scm_validity uses per-sample positions | 2026-03-18 |
| 3 | Sanity Test & Re-run | PENDING | | |

Last stage completed: Stage 2 — Fix compute_scm_validity
Last updated by: plan-runner-agent
