# Stage 1: Fix the Exp7 Saturation Shortcut (P1)

**Goal**: Correct the unsound saturation shortcut in `exp7_budget_sweep.py` so it copies larger-budget rows **only** when no point could possibly bind a larger budget — measured over *all* points (incl. failures) against the *current* budget — otherwise every swept budget is genuinely measured.
**Dependencies**: None (local, code + unit test only; no GPU).

---

## The bug (from the post-review, P1)

`experiments/zeroshot_cf/exp7_budget_sweep.py:306–319`:

```python
steps_max = float(row["steps_max"])
next_idx = budget_idx + 1
if next_idx < len(budgets) and np.isfinite(steps_max) and steps_max < budgets[next_idx]:
    # ... copy identical rows for all remaining larger budgets, runtime_s=0.0 ...
    break
```

Two defects:
1. **`row["steps_max"]` is valid-only.** It comes from `exp4_greedy_cf.py:296` as `steps_all[flipped].max()` — computed over **flipped CFs only**. Points that hit the budget without flipping (`steps == budget`, `flipped=False`) are excluded. But those budget-bound failures are *exactly* the points a larger budget could rescue, so the test that's supposed to prove "a larger budget can't change anything" ignores the only points that could change.
2. **It compares against `budgets[next_idx]` (the next budget), not `budget` (the current one).** The sound predicate for "no point was cut off by the current cap" is `max_steps_over_all_points < current_budget`.

Consequence: on HELOC the shortcut fired at budget 17 and rows for budgets 34/51/100/250/1000 were **copied, not measured**, so the "validity plateaus at 0.8 across all budgets" conclusion is unsupported. Proof the copy is lossy: `exp7_budget_moons.csv` shows `lof_scores_cf` differing between budget 2 (1.0267) and budget ≥4 (1.0075) — a point *was* cut off at budget 2.

---

## Steps

1. **Expose an all-points max-steps signal from the runner.**
   - File: `experiments/zeroshot_cf/exp4_greedy_cf.py` (metrics assembly, ≈`:281–298`).
   - The metrics currently compute `steps_all = np.array([...])` over all points and then
     restrict `steps_*` aggregates to `steps_all[flipped]`. Add a new key
     `steps_max_all = float(steps_all.max())` (or `0.0` when `n==0`) computed over **all**
     points — flipped and failed alike. Keep the existing valid-only `steps_max` unchanged
     (it is still the honest "how many steps did successful flips take" number).
   - Thread `steps_max_all` through `_run_budget`'s returned `row` in
     `exp7_budget_sweep.py` (add it to the row dict). Adding it to `CSV_COLUMNS` is acceptable
     and more transparent — if added, document it in the header/comment.

2. **Fix the shortcut predicate.**
   - File: `experiments/zeroshot_cf/exp7_budget_sweep.py:306–319`.
   - Fire the shortcut **only** when `np.isfinite(steps_max_all) and steps_max_all < budget`
     (current budget) — i.e. **no point** (valid or failed) reached the current cap, so no
     larger cap can bind. When this holds, copying the remaining larger-budget rows is exact.
     Otherwise, continue the loop and measure the next budget.
   - Update the printed message to reference `steps_max_all` and the **current** budget.
   - Keep `runtime_s=0.0` on copied rows and keep the `break` after copying.

3. **Add a unit test pinning the shortcut logic.**
   - File: `experiments/zeroshot_cf/tests/test_greedy.py` (no model needed — test the predicate
     directly).
   - Refactor the predicate into a small pure helper `_can_saturate(steps_max_all, budget) -> bool`
     so the test is a pure-Python assertion with no TabPFN dependency.
   - **(a)** budget-bound failure present (`steps_max_all == budget`) ⇒ `_can_saturate` is
     **False** (sweep must proceed to the next budget).
   - **(b)** every point flipped/stalled strictly below the cap (`steps_max_all < budget`) ⇒
     `_can_saturate` is **True** (shortcut fires; remaining budgets copied with `runtime_s=0.0`).

---

## Verification

- [ ] `exp7_budget_sweep.py` fires the shortcut only on `steps_max_all < current_budget`; the
      valid-only `steps_max` is no longer used for the gate.
- [ ] `exp4_greedy_cf.py` emits `steps_max_all` (over all points) alongside the valid-only `steps_max`.
- [ ] `uv run pytest experiments/zeroshot_cf/tests/test_greedy.py -q` passes, including the two
      new shortcut tests (budget-bound-failure ⇒ no fire; all-below-cap ⇒ fire).
- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` — full suite green.
- [ ] `git diff --name-only <base>..HEAD -- src/tabpfn` empty; `grep -rn "tabpfn_client" experiments/zeroshot_cf` finds nothing.

> Do **not** regenerate the Exp7 CSVs in this stage — that is Stage 6, which re-runs both
> datasets from this corrected code state (closing the P2 reproducibility gap in one motion).

---

## Commit

`fix(greedy-cf): correct Exp7 saturation shortcut to gate on all-points max-steps (P1)`
