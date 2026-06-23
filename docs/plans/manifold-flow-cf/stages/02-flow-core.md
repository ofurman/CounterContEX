# Stage 2: Sparse manifold-guided flow core

**Goal**: Implement `flow_counterfactual(...)` — a sparse, manifold-guided proximal/IHT update with dual drift (generative score + discriminative classifier gradient), hard-thresholding to a budget B, and hybrid continuous-Langevin / discrete-Gibbs dynamics — mirroring the `greedy_counterfactual` interface.
**Dependencies**: Stage 1 (DONE) — needs the validated `conditional_score` and `gibbs_proposal`.

---

## Background (read `resources/math.md` §3 first)

The update in displacement coords `δ = x_A − (x0)_A` (immutables drop out by construction):

```
δ ← H_B( δ + η·[ α·s_t(x)_A + β·σ'(κ(h_t−τ))·∇h_t(x)_A ] + sqrt(2η/β)·ξ )
```

- `s_t(x)_A` = generative drift from Stage-1 `conditional_score` (continuous cols).
- `∇h_t(x)_A` = discriminative drift: for `LogisticRegression`, the logit gradient is the constant `disc._clf.coef_[0]` (Decision #6); scale by `σ'` of the margin.
- `H_B` = hard-threshold keeping the B largest-|δ| **continuous** actionable coords (the L0 prox).
- Discrete (classifier-routed) actionable cols evolve by **Gibbs jumps** (`gibbs_proposal`), proposed each step with prob ∝ their class-divergence, accepted if they raise `h_t` (or always under the annealed chain in Stage 5).
- Stage 2 runs the **deterministic** flow (noise ξ=0); annealing + noise is Stage 5.

---

## Steps

1. Create the flow module.
   - File: `experiments/zeroshot_cf/flow.py`
   - Implement `flow_counterfactual(sampler, disc, x, y_target, actionable_idx, immutable_idx, *, budget=2, alpha=1.0, beta=10.0, eta=0.1, n_steps=50, noise=0.0, kappa=4.0, tau=0.5, score_method="mean_shift", seed=0) -> (x_cf, changed, info)`:
     - Split `actionable_idx` into continuous vs discrete via `score.is_classifier_column`.
     - Init `x_cf = x.copy()`. Loop up to `n_steps` or until `disc.predict(x_cf)==y_target`:
       - Compute generative drift `s = conditional_score(sampler, x_cf, y_target, cont_cols, method=score_method)["score"]`.
       - Compute discriminative drift `g`: `coef = disc._clf.coef_[0]`; margin `m = decision_function-equivalent`; `g = sigmoid'(κ·(h_t−τ))·coef`, restricted to continuous actionable cols and sign-oriented toward `y_target`.
       - `δ += eta·(alpha·s + beta·g)` on continuous actionable cols; add `sqrt(2·eta/beta)·N(0,1)` if `noise>0` (seeded RNG).
       - Apply `H_B`: zero all but the B largest-|δ| continuous coords. `x_cf[cont] = clip(x0[cont] + δ, 0, 1)`.
       - Discrete jump: pick the discrete actionable col with max class-divergence (`class_conditional_shift`); propose `gibbs_proposal`; accept if it does not lower `h_t`.
     - `changed` = actionable cols where `x_cf != x` (continuous: |δ|>0 after H_B; discrete: value changed). `info = {"flipped": bool, "steps": int, "history": [...], "joint_nll": None}` (joint_nll filled by the runner).
   - **Immutability is structural**: never touch `immutable_idx`. Assert `x_cf[immutable_idx] == x[immutable_idx]` byte-identical before return.

2. Mirror the greedy return contract.
   - Match `greedy_counterfactual`'s `(x_cf, changed, info)` shape so the Exp7 runner can call either behind one interface (Decision: `info` keys are a superset of greedy's `flipped/steps/history`).

3. Unit tests.
   - File: `experiments/zeroshot_cf/tests/test_flow.py`
   - Cases: (a) immutables byte-identical after a run on HELOC; (b) `len(changed) ≤ budget` for continuous cols; (c) loop terminates (≤ n_steps); (d) **determinism** — two runs with `noise=0`, same seed → identical `x_cf`; (e) on a trivially-separable 2-D toy, the flow flips a point that single-coordinate greedy cannot (the interaction case) — or, if that fixture is hard to construct deterministically, assert the dual drift moves ≥2 coords jointly in one step; (f) `gibbs_proposal` integration: a HELOC discrete col can change and stays in-`classes`.

---

## Verification

- [ ] `pytest experiments/zeroshot_cf/tests/test_flow.py -q` passes (provisioned env).
- [ ] Full suite still green: `pytest experiments/zeroshot_cf/tests/ -q` (Stage-1 + predecessor tests unaffected).
- [ ] HELOC smoke (n=5, local CPU OK): flow runs end-to-end, `true_actionability=1.0`, produces valid CFs, no crash on classifier-routed columns.
- [ ] MOONS smoke (n=5): flow runs, at least one point flips with 2 coords moved jointly (sanity that the interaction path is exercised).
- [ ] Offline: no `tabpfn_client` import; models via `get_models()` only.

---

## Commit

`feat(manifold-flow): sparse manifold-guided flow generator with dual drift (Stage 2)`
