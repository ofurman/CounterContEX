# Stage 3: Greedy Speedup — Candidate Caching + Beam Cap

**Goal**: Make the greedy candidate scan cheap enough for full-scale HELOC and the forced-numeric Exp9 (predecessor Backlog #4) via (a) transparent within-run memoization of deterministic near-MAP imputes keyed by conditioning state, and (b) an optional `--beam K` candidate cap — both **off/transparent by default** (no-beam ⇒ exhaustive scan ⇒ byte-identical results).
**Dependencies**: None to implement (local, code + tests). Required by Stage 7 (Exp9 at scale); speeds Stages 4–6. Independent of Stages 1 and 2.

---

## Why (from the post-review / predecessor Backlog #4)

`prob_ascent` (`greedy.py:_select_prob_ascent`) scans **every** remaining candidate each step,
drawing a near-MAP value and a classifier score per candidate → O(|A|) imputes/step → O(|A|·steps)
per point. With revisits enabled (predecessor Stage 5) and forced-numeric routing (Exp9), HELOC
points consume their whole budget, hitting the O(|A|²) worst case with per-query kNN context fits —
this is exactly why the full Exp9 never finished (only an n=1 smoke is committed).

Two correctness-preserving levers:
- **Caching**: near-MAP (`temperature=1e-9`) single-column imputes are **deterministic** given
  identical conditioning `(x_cf, target_col, fixed_target)`. In revisit/oscillation loops the row
  state recurs, so a memo keyed by the conditioning avoids recomputing an identical value. This
  **never changes results** — it only skips recomputation of a value that is provably identical.
- **Beam cap** (`--beam K`, default `None` = no cap): evaluate only the K most-promising candidates
  per step instead of all |A|. This **is** an approximation, so it is **opt-in**; the default
  exhaustive scan is preserved and remains byte-identical to today.

---

## Steps

1. **Add a transparent impute/score cache.**
   - File: `experiments/zeroshot_cf/greedy.py`.
   - Add an optional per-call memo (a plain `dict`) scoped to a single `greedy_counterfactual`
     invocation — **not** a global (avoids cross-point contamination and keeps determinism).
   - Key on `(x_cf.tobytes(), target_col, int(fixed_target), float(temperature))`. Cache the drawn
     near-MAP `val` (and, for `prob_ascent`, the resulting `p_target` score) so a recurring
     conditioning state re-uses the prior result instead of re-imputing.
   - **Correctness constraint**: only cache the **near-MAP / deterministic** path
     (`temperature` at the near-MAP default). If `temperature` is raised (stochastic sampling),
     bypass the cache (or key includes an explicit "no-cache" marker) so randomness is preserved.
   - Wire it into `_select_prob_ascent` (per-candidate draw+score) and the fallback draw at
     `greedy.py:198–206`.

2. **Add an optional beam cap.**
   - Add `beam: Optional[int] = None` to `greedy_counterfactual`. When set, `_select_prob_ascent`
     evaluates only the top-`beam` candidates. Ranking heuristic (document the choice under
     Decisions): reuse the **previous step's** per-candidate scores to prioritise; on the first step
     (no prior scores) evaluate all candidates once to seed the ranking (or a cheap proxy — e.g.
     `|disc.coef_[y_target]|`-weighted feature magnitude for a linear discriminator). `beam=None`
     ⇒ evaluate all candidates (exhaustive, unchanged).
   - `class_divergence` may keep the exhaustive scan (its cost profile differs); if beam is applied
     there too, rank by the prior step's divergence scores. Document whichever you choose.

3. **Thread `--beam` through the runners.**
   - Files: `exp4_greedy_cf.py`, `exp6_context_ablation.py`, `exp7_budget_sweep.py`,
     `exp9_routing_audit.py`.
   - Add an argparse `--beam` (default `None`/unset) passed to `greedy_counterfactual`. Default runs
     must be identical to today (no beam). Log the effective beam value in each run.

4. **Strengthen / document the `class_divergence` stall guard (P3).**
   - File: `experiments/zeroshot_cf/greedy.py:212`.
   - The current guard only catches a consecutive same-feature no-move. Either (a) extend it to
     detect a short **cycle** (the last committed *state* `x_cf.tobytes()` recurs within a small
     window → break), reusing the Step-1 cache's state keys; or (b) if extending is risky, add a
     clear code comment documenting the two-feature-oscillation gap and that `budget` bounds it, and
     note the asymmetry vs `prob_ascent` in the docstring. Prefer (a) if cheap.

5. **Tests (no assertions weakened).**
   - File: `experiments/zeroshot_cf/tests/test_greedy.py` (shared `models` fixture).
   - **(a) Caching equivalence**: on a fixed synthetic/MOONS point, `greedy_counterfactual(...)`
     with caching enabled produces a **byte-identical** `x_cf` and identical `changed`/`info` to a
     run with caching disabled (add an internal toggle or compare against a from-scratch recompute).
   - **(b) Beam default no-op**: `beam=None` yields identical `x_cf`/`changed` to the pre-Stage-3
     code path on the same point (backward-compat).
   - **(c) Beam bounds work**: `beam=1` limits per-step candidate evaluations to ≤1 beyond the seed
     (assert via a counter/spy on `sampler.sample_feature`), and still returns a valid CF when one
     is reachable.
   - **(d)** (if Step 4a taken) a crafted `class_divergence` oscillation terminates via the cycle
     guard without exhausting a large budget.

---

## Verification

- [ ] `uv run pytest experiments/zeroshot_cf/tests/test_greedy.py -q` passes, incl. caching-equivalence
      (byte-identical CFs) and beam tests.
- [ ] `greedy_counterfactual` with default args (no beam, cache transparent) is byte-identical to the
      pre-Stage-3 behaviour on a MOONS point (regression).
- [ ] `--beam` flag present on exp4/exp6/exp7/exp9 `--help`; default runs unchanged.
- [ ] A bounded HELOC `exp9 --force-numeric-cols all --beam 4 --max-test 3` completes in a fraction of
      the un-beamed time (record the speedup) — a smoke that the tractability lever works before the
      Stage-7 full run.
- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` — full suite green.
- [ ] `git diff --name-only <base>..HEAD -- src/tabpfn` empty; no `tabpfn_client` import.

---

## Commit

`feat(greedy-cf): candidate impute cache + optional beam cap + class_divergence cycle guard`
