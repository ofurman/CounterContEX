# Stage 3: Exp7 headline test (MOONS plateau gate)

**Goal**: Run the deterministic flow vs the greedy baseline on MOONS (the **plateau gate**, B=2) and HELOC (the **hold gate**), with a joint-NLL plausibility monitor, and decide whether continuous gradient-coupled selection beats myopic greedy.
**Dependencies**: Stage 2 (DONE) — needs `flow_counterfactual`.

---

## Background

This is the decisive experiment. The headline gate (Success Criteria) is **MOONS validity > 0.82 at B=2** while **HELOC validity ≥ 0.85, frac_oob ≤ 0.05**. A negative result is a valid recorded finding (index Decision #9) — report it with failure_rate, joint-NLL, and the Stage-1 score accuracy so the cause is attributable.

---

## Steps

1. Create the Exp7 runner (mirror `exp4_greedy_cf.py`'s structure).
   - File: `experiments/zeroshot_cf/exp7_flow_cf.py`
   - `generate_counterfactuals(dataset_name, mode="flow", budget=2, alpha=1.0, beta=10.0, eta=0.1, n_steps=50, score_method="mean_shift", max_context=256, context_strategy="knn_both", max_test=None, seed=0) -> (X_test, y_test, X_cf, info)`:
     - `mode="flow"` → `flow_counterfactual`; `mode="greedy"` → `greedy_counterfactual` (so the same driver produces the paired baseline).
     - Use the predecessor's per-dataset best context: HELOC `knn_both@256`, MOONS `random_both@512` (set_context per point for kNN, as in Exp6).
     - Collect `changed_per_point`, `flipped_per_point`, `steps_per_point`, and per-point `x_cf`.
   - CLI: `--dataset {moons,heloc,all}`, `--mode {flow,greedy}`, `--budget`, `--alpha`, `--beta`, `--eta`, `--n-steps`, `--score-method`, `--max-context`, `--context-strategy`, `--max-test`, `--seed`.

2. Add the joint-NLL plausibility monitor.
   - In the runner (or a helper in `metrics_harness.py` — prefer the runner to avoid touching shared code): compute the **joint** negative log-likelihood of each CF under the train distribution. Simplest defensible estimator: a class-conditional Gaussian KDE fit on `X_train[y==t]`, `joint_nll = −mean(log kde(X_cf))`. Report `joint_nll_mean` for flow and greedy. This instruments the ICM trap (per-step LOF can pass while joint density fails).
   - Recompute `frac_oob` inline the same way the predecessor runners do (it is **not** in `compute_metrics`).

3. Run the paired comparison (heavy → remote DGX, Decision #8).
   - MOONS: flow (B=2) vs greedy, n=100. The plateau gate.
   - HELOC: flow vs greedy, `knn_both@256`, n bounded as in `iterative-greedy-cf` Decision #13 (`--max-test`, logged). The hold gate.
   - Write `results/exp7_*.csv` (per-point metrics) + `results/exp7_summary.md` (flow-vs-greedy table: validity, L0, failure_rate, frac_oob, LOF, joint_nll, true_actionability).

---

## Verification

- [ ] `python exp7_flow_cf.py --dataset moons --mode flow --budget 2 --max-test 20` runs end-to-end and writes CSV + summary (smoke; full n=100 on DGX).
- [ ] Paired greedy baseline runs via the same driver (`--mode greedy`).
- [ ] `results/exp7_summary.md` contains the flow-vs-greedy table for both datasets including `joint_nll_mean` and `failure_rate`.
- [ ] **Headline gate evaluated and recorded** (pass or fail): MOONS flow validity vs 0.82 at B=2; HELOC flow validity ≥0.85 & frac_oob ≤0.05. Record the verdict in the index Progress Tracker notes regardless of outcome.
- [ ] `true_actionability == 1.0` for flow on both datasets.
- [ ] Offline guarantee holds.

---

## Commit

`feat(manifold-flow): Exp7 flow-vs-greedy headline test + joint-NLL monitor (Stage 3)`
