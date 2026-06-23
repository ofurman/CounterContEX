# Stage 5: Budget > |A| + Feature Revisiting (the MOONS-validity fix)

**Goal**: Let the greedy loop change **more features than once and exceed the actionable-set size**, so a feature can be re-imputed under the *new conditioning* created by earlier changes — the mechanism the meeting hypothesized is the cause of MOONS validity stalling at ≈0.70. Add a **no-progress guard** for clean termination, then sweep `budget` to see whether validity climbs toward 1.0.
**Dependencies**: Stage 1 DONE (greedy loop + `prob_ascent` selector + Exp4 runner). Independent of Stages 6–8.

---

## Motivation (from the meeting)

> "zmieniamy pierwszą cechę, potem drugą, a potem nie zmieniamy trzeciej, tylko znowu pierwszą … no ten warunek się po prostu zmienia … na pewno powinno być tak, że nawet jeździmy wszystkie cechy, to i validity nam się nie będzie tak jak chcemy, to możemy znowu zmienić tą samą cechę."

Today the loop **forbids** this:
- `experiments/zeroshot_cf/greedy.py:156–157` — `budget` defaults to `len(actionable)`.
- `experiments/zeroshot_cf/greedy.py:172–174` — `candidates = [j for j in actionable if j not in changed]` **hard-excludes** any already-changed feature. So once every actionable feature has been touched once, the candidate set empties and the loop stops (`if not candidates: break`) — raising `budget` alone does nothing.

On MOONS (2 actionable features) this caps the loop at **2 commits**, and Decision #9 already attributes the ≈0.31 MOONS failure rate to points that plateau within those 2 deterministic MAP commits. Re-conditioning a feature on the *updated* partner value can move a point that a single pass cannot.

---

## Design: unlimited revisits + no-progress guard

Chosen policy (user decision, planning 2026-06-23): **allow re-changing any actionable feature every step; let `budget` exceed `|A|`; terminate on flip, budget exhaustion, OR a full-pass no-progress stall.**

```
while not flipped and steps < budget:
    candidates = list(actionable)          # NO exclusion — every actionable feature eligible
    j*, val, score = best_by_selector(candidates)   # prob_ascent: score = resulting p_target
    gain = score - p_target_current
    if gain <= eps:                        # no candidate improves the objective → stall
        break
    x_cf[j*] = val
    steps += 1
    flipped, p_target_current = flip_state(x_cf)
```

Key points:
- **No feature exclusion.** Replace `greedy.py:172–174` candidate construction with the full actionable list.
- **No-progress guard** (`eps`, default `1e-6`, exposed as `--stall-eps`): break when the best achievable `p_target` does not exceed the current `p_target` by more than `eps`. For `prob_ascent` this is exact (the selector already evaluates the resulting `p_target` of every candidate; the guard reuses that, no extra cost). This prevents the loop from re-committing the same MAP value forever once it reaches a fixed point.
- **`budget` decoupled from `|A|`**: default stays `len(actionable)` for backward-compat, but values `> |A|` are now meaningful. `steps` (commit count) may exceed the number of **distinct** features changed.
- **`class_divergence`** (`greedy.py:70–101`) has no `p_target` to gate on. Its stall guard: break if the selected `j*` equals the previous step's `j*` **and** its committed value is within `eps` of the current value (no movement) — i.e. a fixed point. Document this asymmetry; `prob_ascent` is the primary selector (Stage 2 winner) and gets the principled guard.

### Metrics consequence (must fix in the runner)

With revisits, `info["changed"]` (the ordered commit list, `greedy.py:145–151`) can contain **duplicate** indices. Two distinct quantities now diverge and **both must be reported**:
- **`steps`** = number of commits (`len(info["history"])`).
- **`l0_count`** = number of **distinct** features changed = `len(set(changed))` — this is the sparsity metric that matters for the paper.

Update Exp4's L0 aggregation (currently `exp4_greedy_cf.py:236–239`, computed over valid CFs) to count **distinct** indices, not commit-list length. Add `steps_*` alongside (already present) so the revisit overhead is visible.

---

## Steps

1. **Relax the loop in `greedy.py`.**
   - Candidate set = full `actionable` list every step (remove the `j not in changed` filter at `:172–174`; keep the immutables-never-eligible guarantee — `actionable` already excludes them).
   - Add `--stall-eps`-driven no-progress break for `prob_ascent` (gain ≤ eps); add the fixed-point break for `class_divergence`.
   - Keep `budget=None → len(actionable)` default; the loop bound `steps < budget` (`:171`) is unchanged but now `budget` may be passed `> |A|`.
   - Preserve return shape `(x_cf, changed, info)` (Decision #8). `changed` keeps insertion order **with duplicates**; add `info["distinct_changed"] = sorted(set(changed))` for convenience.

2. **Fix L0 accounting in `exp4_greedy_cf.py`.**
   - `l0_count_*` over valid CFs must use **distinct** feature count (`len(set(changed))`), not `len(changed)`. Keep `steps_*` as the commit count. Document the distinction in the CSV header / a comment.

3. **Add the budget-sweep driver `exp7_budget_sweep.py`.**
   - Sweep `budget` over a per-dataset grid (see `resources/grids.md` Stage 5) at the **Stage-4 recommended config** (HELOC `prob_ascent` + `knn_both@256`; MOONS `prob_ascent` + `random_both@512`).
   - For each budget, run Exp4 generation+metrics; write `results/exp7_budget_{moons,heloc}.csv` with one row per budget: `budget, validity, failure_rate, l0_count_mean (distinct), steps_mean, steps_max, proximity_l2_jaccard, lof_scores_cf, frac_oob, true_actionability, runtime_s`.
   - Reuse the inline `frac_oob` computation (`exp2_counterfactuals.py:264–267`).

4. **Write `results/exp7_summary.md`.**
   - A validity-vs-budget table per dataset and an honest verdict: **does MOONS validity climb toward 1.0 as budget grows past |A|?** Report the budget at which validity saturates (if it does), and the cost paid in `steps_mean`, `proximity_l2_jaccard`, and `true_actionability`. If validity plateaus *below* 1.0 even at budget=1000, that is the predicted "TabPFN ≠ the external classifier" gap (Decision: meeting discussion) — record it, do not chase it with gradients.

5. **Tests.** Extend `tests/test_greedy.py` (shared `models` fixture in `tests/conftest.py`):
   - A test asserting that with `budget > |A|` on a crafted point, `info["history"]` can contain a **repeated** feature index (revisiting actually happens).
   - A test asserting the no-progress guard terminates: a point already at `p_target ≥ tau` (or one where no candidate improves `p_target`) returns without exhausting a large budget (loop stops on `gain ≤ eps`, not on the budget bound).
   - A regression test that the **default** call (`budget=None`) still flips/behaves as before on a simple MOONS point (backward-compat).

---

## Verification

- [ ] `experiments/zeroshot_cf/greedy.py` no longer filters out already-changed features; full suite `uv run pytest experiments/zeroshot_cf/tests -q` passes (existing + 3 new).
- [ ] On a MOONS point, `greedy_counterfactual(..., budget=20)` can produce `len(info["history"]) > len(set(changed))` (a revisit occurred).
- [ ] `results/exp7_budget_{moons,heloc}.csv` exist with one row per swept budget and the columns above.
- [ ] `results/exp7_summary.md` states the validity-vs-budget trend and the saturation budget (or that validity plateaus below 1.0), with the proximity/steps cost.
- [ ] `l0_count_mean` in Exp4/Exp7 reflects **distinct** features changed (manually verify on one CF whose history has a duplicate).
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` is empty; `grep -rn "tabpfn_client" experiments/zeroshot_cf` finds nothing.

---

## Expected outcomes (record actuals against these)

- **MOONS**: validity rises above the ≈0.70 Stage-1/2 plateau as `budget` exceeds 2; the meeting's hypothesis is that re-conditioning unblocks the boundary points (see Stage 6 for the geometric "blocked region" picture). Quantify the lift and the budget at which it saturates.
- **HELOC**: already ≈0.90 at `knn_both@256`; revisiting may close the remaining gap or expose the TabPFN-vs-classifier ceiling. Watch `proximity_l2_jaccard` — unlimited revisits can drift the point, trading proximity for validity.
- A validity that plateaus below 1.0 even at large budget is a **legitimate finding** (TabPFN's conditional is an approximation of the external classifier), not a stage failure.

---

## Commit

`feat(greedy-cf): budget>|A| + feature revisiting with no-progress guard + budget sweep (Exp7)`
