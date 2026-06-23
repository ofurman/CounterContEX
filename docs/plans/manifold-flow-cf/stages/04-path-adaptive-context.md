# Stage 4: Path-adaptive kNN context

**Goal**: Re-select the kNN conditioning context around the *moving* iterate (not just the static factual `x0`) so the score stays low-bias along the whole flow path, and ablate static vs adaptive context on HELOC.
**Dependencies**: Stage 3 (DONE). Mutually independent of Stage 5.

---

## Background (read `resources/math.md` §4)

kNN context is a variable-bandwidth local score estimator (`iterative-greedy-cf` Decision #14: relevance beats volume on HELOC). The predecessor selects context **once** around `x0`; once the iterate moves toward the target class, that context is stale and the score bias grows. Re-selecting context around `x_k` keeps the local-likelihood bias low along the entire path. This should further reduce HELOC `frac_oob` and joint-NLL.

---

## Steps

1. Add a context-refit cadence to the flow.
   - File: `experiments/zeroshot_cf/flow.py`
   - Add param `context_refit: int = 0` to `flow_counterfactual` (0 = static, the Stage-2 default; k>0 = re-call `sampler.set_context(..., selection="knn", query=x_cf)` every k steps around the current iterate).
   - Refit must respect the same `max_context`, `context_strategy`, and pool (`both`/`target`) the run was launched with. Log the number of refits in `info["context_refits"]`.
   - Default `context_refit=0` keeps Stage 2/3 behaviour byte-identical (regression-safe).

2. Wire the knob through Exp7.
   - File: `experiments/zeroshot_cf/exp7_flow_cf.py`
   - Add `--context-refit` CLI arg + pass-through.

3. Unit test.
   - File: `experiments/zeroshot_cf/tests/test_flow.py` (extend)
   - Assert `context_refit=0` reproduces the Stage-2 result exactly (determinism); `context_refit=5` triggers `info["context_refits"] > 0` and still preserves immutability + budget.

4. Ablation run (heavy → DGX).
   - HELOC `knn_both@256`, flow, `context_refit ∈ {0, 5, 1}` (static, periodic, every-step), n bounded + logged.
   - Compare frac_oob, LOF, joint_nll, validity, and added cost (refits, wall-clock). Write `results/exp7_context_refit.csv` + a short verdict appended to `results/exp7_summary.md`.

---

## Verification

- [ ] `pytest experiments/zeroshot_cf/tests/test_flow.py -q` passes incl. the new refit cases.
- [ ] `context_refit=0` is byte-identical to the Stage-2/3 flow output (regression guard).
- [ ] Smoke: `python exp7_flow_cf.py --dataset heloc --mode flow --context-refit 5 --max-test 5` runs and logs `context_refits > 0`.
- [ ] Ablation table written; verdict recorded (does per-step kNN lower HELOC frac_oob/joint-NLL, and at what cost?).
- [ ] Offline guarantee holds.

---

## Commit

`feat(manifold-flow): path-adaptive kNN context refit + ablation (Stage 4)`
